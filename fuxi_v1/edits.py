"""Bounded single-node edits; JSON patches never execute source code."""
from copy import deepcopy
from .contracts import keys, require, ContractError, digest
from .dsl import CATALOG
from .search import factor_spec, node
from .learning import ast_diff_motifs

KINDS = ('window', 'field', 'operator', 'wrap', 'unwrap', 'binary')
WINDOWS = (2, 3, 5, 7, 10, 15, 20, 30, 40, 60)


def paths(ast, path=()):
    yield path, ast
    for i, arg in enumerate(ast.get('args', [])):
        yield from paths(arg, path+(i,))


def apply_patch(parent, edit, fields, *, source_ids=()):
    keys(edit, ('parent_id', 'kind', 'path', 'value'))
    require(edit['parent_id'] == parent['candidate_id'] and edit['kind'] in KINDS, 'patch parent/kind invalid')
    require(isinstance(edit['path'], list) and len(edit['path']) <= CATALOG['max_depth'] and
            all(type(i) is int and 0 <= i < 2 for i in edit['path']), 'bounded AST path required')
    ast = deepcopy(parent['canonical_ast'])
    target, owner, slot = ast, None, None
    for index in edit['path']:
        require(index < len(target.get('args', [])), 'patch path outside AST')
        owner, slot, target = target, index, target['args'][index]
    value, kind = edit['value'], edit['kind']
    replacement = deepcopy(target)
    if kind == 'window':
        require('window' in target and type(value) is int and value in WINDOWS, 'window edit invalid')
        replacement['window'] = value
    elif kind == 'field':
        require('field' in target and value in fields, 'field edit invalid')
        replacement['field'] = value
    elif kind == 'operator':
        require('op' in target and value in CATALOG['operators'] and
                CATALOG['operators'][value] == len(target['args']), 'operator arity invalid')
        replacement['op'] = value
    elif kind == 'wrap':
        require(value in ('rank', 'zscore', 'neg', 'abs'), 'wrapper invalid')
        replacement = node(value, target)
    elif kind == 'unwrap':
        require(value is None and target.get('op') in ('rank', 'zscore', 'neg', 'abs') and
                len(target['args']) == 1, 'unwrap invalid')
        replacement = target['args'][0]
    else:
        keys(value, ('op', 'field'))
        require(value['op'] in ('add', 'sub', 'mul', 'div') and value['field'] in fields, 'binary edit invalid')
        replacement = node(value['op'], node('rank', target), node('rank', {'field': value['field']}))
    require(replacement != target, 'no-op patch')
    if owner is None:
        ast = replacement
    else:
        owner['args'][slot] = replacement
    child = factor_spec(ast, fields, parent['hypothesis']['mechanism'], parent_ids=[parent['candidate_id']],
                        action='llm_patch', source_ids=source_ids, mutation={'axis': 'one_node_edit', 'edit': edit})
    child['mutation_patch']['motifs'] = ast_diff_motifs(parent['canonical_ast'], ast)
    # factor_spec hashes all lineage, so rebind after adding motif metadata.
    child['candidate_id'] = 'factor_'+digest({k:v for k,v in child.items() if k != 'candidate_id'})
    return child


def refine_local(fields, parents, count, seen=()):
    require(type(count) is int and 1 <= count <= 10, 'local budget invalid')
    proposals, known, attempts, rejected = [], set(seen), 0, []
    for parent in sorted(parents, key=lambda p: p['candidate_id']):
        for path, target in paths(parent['canonical_ast']):
            if 'window' not in target:
                continue
            old = target['window']
            neighbors = sorted((w for w in WINDOWS if w != old), key=lambda w: (abs(w-old), w))[:2]
            for w in neighbors:
                attempts += 1
                edit = {'parent_id':parent['candidate_id'], 'kind':'window', 'path':list(path), 'value':w}
                try:
                    child = apply_patch(parent, edit, fields)
                    child['action'] = 'refine_local'
                    child['candidate_id'] = 'factor_'+digest({k:v for k,v in child.items() if k != 'candidate_id'})
                except ContractError as exc:
                    rejected.append(str(exc))
                    continue
                if child['formula_hash'] in known:
                    continue
                proposals.append(child)
                known.add(child['formula_hash'])
                if len(proposals) == count:
                    return proposals, {'attempts':attempts, 'compiler_rejections':rejected}
    return proposals, {'attempts':attempts, 'compiler_rejections':rejected}


def parse_llm_patches(parsed, parents, fields, evidence_cards, count):
    keys(parsed, ('patches',))
    require(isinstance(parsed['patches'], list) and len(parsed['patches']) <= count, 'patch list/count invalid')
    lookup = {p['candidate_id']:p for p in parents}
    proposals, rejected = [], []
    for row in parsed['patches']:
        try:
            require(row.get('parent_id') in lookup, 'unknown patch parent')
            proposals.append(apply_patch(lookup[row['parent_id']], row, fields,
                                        source_ids=[x['evidence_id'] for x in evidence_cards]))
        except (ValueError, KeyError, TypeError) as exc:
            rejected.append(str(exc))
    return proposals, rejected, len(parsed['patches'])


def patch_prompt(fields, parents, evidence_cards, count):
    """Expose exact node addresses, without repairing or relaxing returned edits."""
    require(parents and len(parents) <= 8, 'bounded parent set required')
    require(type(count) is int and 1 <= count <= 10, 'bounded patch count required')
    parent_rows = []
    for parent in parents:
        nodes, windows = [], []
        for path, target in paths(parent['canonical_ast']):
            nodes.append({'path':list(path), 'node':deepcopy(target)})
            if 'window' in target:
                windows.append({'path':list(path), 'operator':target['op'],
                                'current_window':target['window'],
                                'allowed_values':[w for w in WINDOWS if w != target['window']]})
        parent_rows.append({'parent_id':parent['candidate_id'], 'ast':parent['canonical_ast'],
                            'nodes':nodes, 'window_edit_targets':windows})
    first = parent_rows[0]
    example = {'parent_id':first['parent_id'], 'kind':'wrap', 'path':[], 'value':'neg'}
    if first['window_edit_targets']:
        target = first['window_edit_targets'][0]
        value = min(target['allowed_values'], key=lambda w: (abs(w-target['current_window']), w))
        example = {'parent_id':first['parent_id'], 'kind':'window',
                   'path':target['path'], 'value':value}
    prompt = {'task':'Return ONLY one JSON object with patches: a list of at most count single-node edits; no code or metrics.',
              'count':count, 'parents':parent_rows,
              'fields':fields, 'evidence':evidence_cards, 'kinds':list(KINDS), 'window_values':list(WINDOWS),
              'schema':{'patches':[example]},
              'path_contract':'Copy the exact args-index path of the node being edited from parents.nodes. A window edit MUST use a listed window_edit_targets path and one of its allowed_values. The field child of a time operator has a different path and has NO window; do not append a leaf index to the window path. Never guess or rewrite a supplied parent_id.',
              'rules':'Each edit has exactly parent_id,kind,path,value. Path is args indices, [] means root. window replaces a window; field replaces a leaf with field name; operator preserves arity; wrap uses rank/zscore/neg/abs; unwrap value=null; binary value={op:add/sub/mul/div,field:catalog field}. Catalog compilation is final. Missing data propagate.',
              'catalog':CATALOG}
    return prompt


def llm_patches(fields, parents, evidence_cards, *, count=3, max_tokens=1800, timeout=40):
    from .search import llm_candidates
    prompt = patch_prompt(fields, parents, evidence_cards, count)
    return llm_candidates(fields, evidence_cards, count=count, max_tokens=max_tokens, timeout=timeout,
                          _prompt=prompt, _parser=lambda parsed: parse_llm_patches(parsed, parents, fields, evidence_cards, count))
