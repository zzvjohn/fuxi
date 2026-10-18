"""Read-only legacy/native asset federation and explicit signal equivalence proofs.

Names, historical passes and cached receipt ids never establish equivalence or
current qualification. Outputs are new immutable projections; sources are not
modified and no market data or statistical evaluation is requested.
"""
import ast as pyast
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import io
import json
import math
from pathlib import Path

from .contracts import ContractError, digest, require, load_json
from .datasets import file_hash
from .dsl import compile_ast, evaluate_ast, CATALOG
from .artifacts import safe_name


SCHEMA = 'fuxi.shared_catalog.1'
FIELD_ALIASES = {f + '_p': f for f in ('open', 'high', 'low', 'close', 'volume', 'amount')}
FIELD_ALIASES.update({f: f for f in ('open', 'high', 'low', 'close', 'volume', 'amount',
                                  'turnover_rate', 'total_shares', 'float_shares', 'free_shares')})
NATIVE_FIELDS = {f: {'unit': 'price' if f in ('open', 'high', 'low', 'close') else
                    'currency' if f == 'amount' else 'ratio' if f == 'turnover_rate' else 'shares',
                    'frequency': 'daily', 'adjustment': 'raw' if f != 'turnover_rate' else 'not_applicable'}
                 for f in FIELD_ALIASES.values()}
REPRESENTATIVES = (
    ('volume_stability', '-(volume_p.rolling(20).std() / volume_p.rolling(20).mean())'),
    ('anchoring_20d_high', 'close_p / close_p.rolling(20).max()'),
    ('relative_value_zscore', '-(close_p - close_p.rolling(120).mean()) / close_p.rolling(120).std()'),
)
DEFAULT_INPUT = 'reports/fuxi_v1_full_landing_20261009/cohort64/accepted_research_v4/built/input.json'


def data_contracts():
    """Each source field keeps units, clocks, price basis and missingness explicit."""
    rows = []
    for field, spec in sorted(NATIVE_FIELDS.items()):
        source = ('daily_basic.' + field.replace('_shares', '_share') if field.endswith('_shares')
                  else 'daily_basic.turnover_rate' if field == 'turnover_rate'
                  else 'daily.vol' if field == 'volume' else 'daily.' + field)
        transform = ('*100 shares/lot' if field == 'volume' else '*1000 CNY/thousand-CNY'
                     if field == 'amount' else '*10000 shares/ten-thousand-shares'
                     if field.endswith('_shares') else '*0.01 ratio/percentage'
                     if field == 'turnover_rate' else 'identity CNY/share')
        rows.append({'field': field, 'legacy_aliases': [k for k, v in FIELD_ALIASES.items() if v == field],
            'source_column': source, 'native_unit': spec['unit'], 'source_to_native': transform,
            'frequency': 'explicit_exchange_daily_sessions', 'adjustment': spec['adjustment'],
            'available_at': 'declared_18:00_Asia_Shanghai_conservative_daily_proxy',
            'decision_time': 'source_snapshot_declared_18:30_Asia_Shanghai',
            'first_execution': 'following_exchange_session_09:30',
            'missing': 'propagate_null; complete_rolling_windows; no_fill_or_zero_substitution',
            'action_history': 'snapshot_action_barrier; previous_price_or_share_basis_not_retroactively_rewritten',
            'historical_PIT_certified': False, 'source_revision_archive_certified': False,
            'contract_status': 'declared_native_research_contract_not_legacy_production_equivalence'})
    rows.append({'field': 'legacy_S5_turnover', 'legacy_aliases': ['turnover', 'turnover_p'],
        'definition': 'volume/close', 'native_unit': 'shares/price', 'source_to_native': 'no turnover_rate mapping',
        'frequency': 'daily', 'adjustment': 'must follow its actual input price basis',
        'available_at': 'derived from source field availability', 'decision_time': 'consumer declared',
        'first_execution': 'consumer declared next session', 'missing': 'unresolved until actual consumer verified',
        'historical_PIT_certified': False, 'source_revision_archive_certified': False,
        'contract_status': 'unsupported_as_true_turnover_rate'})
    for channel in ('financial_statements', 'moneyflow', 'margin', 'top_list', 'analyst_forecasts',
                    'industry_membership', 'hk_hold', 'corporate_events'):
        rows.append({'field': channel, 'legacy_aliases': [], 'native_unit': 'requires_per_column_review',
            'source_to_native': 'not_automatically_mapped', 'frequency': 'source_specific_event_or_daily',
            'adjustment': 'requires_actual_consumer_review', 'available_at': 'requires_published_or_conservative_next_session_rule',
            'decision_time': 'requires_frozen_contract', 'first_execution': 'after_available_at_next_admitted_session',
            'missing': 'unknown; never missing_equals_zero', 'historical_PIT_certified': False,
            'source_revision_archive_certified': False, 'contract_status': 'unsupported_pending_source_consumer_contract'})
    body = {'schema_version': 'fuxi.shared_data_contracts.1', 'fields': rows,
        'turnover_is_not_turnover_rate': True, 'legacy_production_equivalence_certified': False,
        'qualification_granted': False}
    body['contract_hash'] = digest(body)
    return body


def signal_contract(fields=None):
    specs = fields if isinstance(fields, dict) else {f: NATIVE_FIELDS[f] for f in (fields or NATIVE_FIELDS)}
    require(all(spec.get('frequency') == 'daily' and spec.get('unit') == NATIVE_FIELDS[f]['unit'] and
                spec.get('adjustment') in ('raw', 'not_applicable') for f, spec in specs.items()),
            'explicit raw daily field contract required; production adjusted inputs unsupported')
    body = {'schema_version': 'fuxi.shared_signal_semantics.1',
        'fields': {f: deepcopy(specs[f]) for f in sorted(specs)}, 'session_axis': 'explicit_exchange_calendar',
        'cross_section': 'exact_current_snapshot_universe_no_refill', 'missing': 'propagate',
        'rolling': 'complete_window; std_ddof_1', 'division': 'zero_or_nonfinite_to_null',
        'price_basis': 'source_native_raw_not_legacy_production_adjusted',
        'history_basis': 'source_action_barrier_at_current_decision',
        'availability': 'snapshot_declared_conservative_proxy; no_historical_PIT_certification',
        'frequency': 'daily', 'direction': 'expression_as_written_no_external_sign', 'dsl_catalog_hash': digest(CATALOG),
        'legacy_production_equivalence_certified': False}
    body['semantic_contract_hash'] = digest(body)
    return body


def _op(name, *args, window=None):
    row = {'op': name, 'args': list(args)}
    if window is not None: row['window'] = window
    return row


def translate_expression(expression, fields=None):
    """Strict supported subset, with no eval, constants, pct_change fill or axis inference."""
    require(isinstance(expression, str) and expression.strip(), 'formula missing')
    fields = fields or NATIVE_FIELDS
    try: tree = pyast.parse(expression, mode='eval')
    except SyntaxError as exc: raise ContractError('unsupported legacy syntax') from exc
    def walk(node):
        if isinstance(node, pyast.Name):
            require(node.id in FIELD_ALIASES, 'unknown legacy field: ' + node.id)
            return {'field': FIELD_ALIASES[node.id]}
        if isinstance(node, pyast.UnaryOp) and isinstance(node.op, pyast.USub):
            return _op('neg', walk(node.operand))
        if isinstance(node, pyast.BinOp) and type(node.op) in (pyast.Add, pyast.Sub, pyast.Mult, pyast.Div):
            name = {pyast.Add: 'add', pyast.Sub: 'sub', pyast.Mult: 'mul', pyast.Div: 'div'}[type(node.op)]
            return _op(name, walk(node.left), walk(node.right))
        if isinstance(node, pyast.Call) and isinstance(node.func, pyast.Attribute):
            call, receiver = node.func.attr, node.func.value
            require(not node.args and not node.keywords, 'call arguments unsupported')
            if call == 'abs': return _op('abs', walk(receiver))
            require(call in ('mean', 'std', 'min', 'max') and isinstance(receiver, pyast.Call)
                    and isinstance(receiver.func, pyast.Attribute) and receiver.func.attr == 'rolling',
                    'unsupported method; pct_change/rank/ffill require separate semantic review')
            require(len(receiver.args) == 1 and not receiver.keywords and
                    isinstance(receiver.args[0], pyast.Constant) and type(receiver.args[0].value) is int,
                    'explicit integral rolling window required')
            leaf = walk(receiver.func.value)
            require(set(leaf) == {'field'}, 'rolling composite unsupported')
            return _op(call, leaf, window=receiver.args[0].value)
        raise ContractError('unsupported legacy expression node: ' + type(node).__name__)
    native = walk(tree.body)
    compiled = compile_ast(native, fields)
    contract = signal_contract({f: fields[f] for f in compiled['required_fields']})
    identity = digest({'canonical_ast': native, 'semantic_contract_hash': contract['semantic_contract_hash']})
    return {'canonical_ast': native, **compiled, 'semantic_contract': contract,
            'semantic_factor_id': identity, 'source_expression': expression,
            'mapping_scope': 'explicit_raw_signal_contract_only', 'legacy_production_equivalence_certified': False,
            'qualification_transferred': False}


def reverse_projection(native):
    """Export only the same supported subset; output is a signal component, never an admission."""
    def render(node):
        if 'field' in node:
            require(node['field'] in NATIVE_FIELDS, 'unknown native projection field')
            return node['field'] + '_p' if node['field'] in ('open', 'high', 'low', 'close', 'volume', 'amount') else node['field']
        op, args = node['op'], node['args']
        if op in ('mean', 'std', 'min', 'max'):
            require(set(args[0]) == {'field'}, 'time field leaf required')
            return render(args[0]) + '.rolling(' + str(node['window']) + ').' + op + '()'
        if op == 'neg': return '-(' + render(args[0]) + ')'
        if op == 'abs': return '(' + render(args[0]) + ').abs()'
        require(op in ('add', 'sub', 'mul', 'div'), 'unsupported native reverse projection')
        token = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/'}[op]
        return '(' + render(args[0]) + ' ' + token + ' ' + render(args[1]) + ')'
    compile_ast(native, NATIVE_FIELDS)
    expression = render(native)
    mapped = translate_expression(expression)
    require(mapped['canonical_ast'] == native, 'reverse projection changed native AST')
    return {'formula': expression, 'canonical_ast': deepcopy(native),
            'semantic_factor_id': mapped['semantic_factor_id'], 'usage': 'research_signal_component',
            'status': 'mapped_component_requires_independent_evaluation', 'eligible_for_jq': False,
            'qualification_receipt_id': None, 'qualification_transferred': False}


def _sources(workspace):
    root = Path(workspace).resolve()
    selected = []
    data = root / 'data'
    if data.exists():
        for path in sorted(data.iterdir()):
            if path.is_file() and path.suffix in ('.json', '.csv') and (
                    'factor' in path.name or path.name in ('experience_memory.json', 'jq_codegen_queue.json')):
                selected.append(path)
    # Include all real saved cards and their historical copies; duplicates remain visible.
    for base in (root / 'reports', data / 'fuxi_v1', data / 'fuxi_unified'):
        if base.exists():
            for folder in sorted(base.rglob('cards')):
                if folder.is_dir(): selected.extend(sorted(folder.glob('*.json')))
            for folder in sorted(base.rglob('bundles')):
                if not folder.is_dir(): continue
                for candidate in sorted(folder.rglob('candidate.json')):
                    manifest = candidate.with_name('manifest.json')
                    if candidate.parent.name.startswith('run_') and manifest.is_file():
                        selected.extend((candidate, manifest))
    alchemy = root / 'research/factor_alchemy'
    factors = alchemy / 'factors'
    if factors.exists(): selected.extend(sorted(factors.rglob('*.py')))
    selected.extend(p for p in (alchemy / 'config.py', alchemy / '__init__.py') if p.is_file())
    return sorted(set(selected))


def _published_pair(path):
    return (path.name in ('candidate.json', 'manifest.json') and path.parent.name.startswith('run_')
            and any(parent.name == 'bundles' for parent in path.parents))


def _publication_binding(path):
    """Validate only the candidate/manifest pair; never load values or evaluation."""
    directory = path.parent
    candidate, manifest_path = directory / 'candidate.json', directory / 'manifest.json'
    require(not any(p.is_symlink() for p in (directory, candidate, manifest_path)), 'published pair symlinks forbidden')
    require(candidate.is_file() and manifest_path.is_file(), 'published candidate/manifest pair missing')
    manifest_sha = file_hash(manifest_path)
    try: manifest = json.loads(manifest_path.read_bytes().decode('utf-8-sig'))
    except (ValueError, UnicodeError) as exc: raise ContractError('published manifest invalid JSON') from exc
    require(isinstance(manifest, dict) and set(manifest) == {
        'schema_version', 'run_id', 'created_at', 'python_version', 'files', 'provenance', 'complete', 'manifest_hash'},
        'published manifest fields differ')
    require(manifest['schema_version'] == 'fuxi.run.1' and manifest['complete'] is True,
            'published manifest incomplete or unsupported')
    require(manifest['run_id'] == directory.name, 'published manifest run identity mismatch')
    require(manifest['manifest_hash'] == digest({k: v for k, v in manifest.items() if k != 'manifest_hash'}),
            'published manifest hash changed')
    files = manifest['files']
    require(isinstance(files, dict) and 'candidate.json' in files, 'published manifest candidate binding missing')
    for name, expected in files.items():
        safe_name(name)
        require(isinstance(expected, str) and len(expected) == 64 and all(c in '0123456789abcdef' for c in expected),
                'published manifest file SHA256 invalid')
    candidate_sha = file_hash(candidate)
    require(candidate_sha == files['candidate.json'], 'published candidate SHA256 changed')
    require(file_hash(manifest_path) == manifest_sha, 'published manifest changed while bound')
    return {'manifest_path': str(manifest_path), 'manifest_sha256': manifest_sha,
        'manifest_hash': manifest['manifest_hash'], 'candidate_sha256': candidate_sha,
        'run_id': manifest['run_id'], 'complete': True, 'other_declared_files_verified': False,
        'verification_scope': 'candidate_and_manifest_only_no_scores_or_evaluation_loaded'}


def _source_module_records(path, blob):
    """Static top-level symbols are source metadata, never imported factor code."""
    symbols, parse_error = [], None
    try:
        tree = pyast.parse(blob, filename=str(path))
        for node in tree.body:
            if isinstance(node, (pyast.ClassDef, pyast.FunctionDef, pyast.AsyncFunctionDef)):
                symbols.append({'name': node.name, 'symbol_kind': 'class' if isinstance(node, pyast.ClassDef)
                    else 'async_function' if isinstance(node, pyast.AsyncFunctionDef) else 'function',
                    'lineno': node.lineno, 'end_lineno': node.end_lineno,
                    'docstring': pyast.get_docstring(node, clean=False), 'source_executed': False,
                    'status': 'source_definition_metadata'})
    except (SyntaxError, ValueError, UnicodeError) as exc:
        parse_error = {'type': type(exc).__name__, 'reason': str(exc)}
    module = {'schema_version': 'fuxi.source_module_asset.1', 'name': path.name,
        'source_executed': False, 'static_parse_status': 'unsupported_syntax' if parse_error else 'valid',
        'static_parse_error': parse_error, 'top_level_symbols': symbols, 'status': 'source_module_metadata'}
    yield '/__source_module__', module, None, 'source_module_asset'
    for i, symbol in enumerate(symbols): yield '/__symbols__/' + str(i), symbol, None, 'source_code_definition'


def _records(path, blob):
    if path.suffix == '.py':
        yield from _source_module_records(path, blob)
        return
    if path.suffix == '.csv':
        reader = csv.DictReader(io.StringIO(blob.decode('utf-8-sig'), newline=''))
        require(reader.fieldnames and len(set(reader.fieldnames)) == len(reader.fieldnames), 'CSV header invalid')
        for i, row in enumerate(reader):
            yield '/' + str(i), row, None, 'csv_row'
        return
    # A valid native card is a single entity, including every original stage state.
    value = json.loads(blob.decode('utf-8-sig'))
    if _published_pair(path):
        yield '/', value, None, 'native_candidate' if path.name == 'candidate.json' else 'publication_manifest_metadata'
    elif isinstance(value, dict) and str(value.get('schema_version', '')).startswith('fuxi.factor_card.'):
        require(value.get('card_hash') == digest({k: v for k, v in value.items() if k != 'card_hash'}), 'native card hash changed')
        yield '/', value, None, 'native_card'
    elif isinstance(value, list):
        for i, row in enumerate(value): yield '/' + str(i), row, None, 'json_list_row'
    elif isinstance(value, dict):
        # No state filter. Every root section/item survives, including metadata and
        # unknown structures. Original bytes separately bind deeper nested records.
        if path.name == 'jq_codegen_queue.json':
            for key, row in value.items(): yield '/' + key, row, key, 'queue_entry'
        else:
            metadata = {}
            for key, section in value.items():
                if isinstance(section, list):
                    for i, row in enumerate(section): yield '/' + key + '/' + str(i), row, None, key
                elif isinstance(section, dict):
                    for name, row in section.items(): yield '/' + key + '/' + name, row, name, key
                else: metadata[key] = section
            if metadata: yield '/__metadata__', metadata, None, 'source_metadata'
    else:
        yield '/', value, None, 'unsupported_root'


def _name(row, fallback, section):
    if isinstance(row, dict):
        for key in ('name', 'factor_name', 'factor_id', 'candidate_id', 'pattern_id', 'slug', 'name_zh'):
            if isinstance(row.get(key), str) and row[key].strip(): return row[key]
    if section in ('selected_factors', 'names') and isinstance(row, str): return row
    return fallback


def _explicit_nonfinite(value):
    """Legacy JSON NaN is source-invalid data, retained as a typed marker, never zero.

    Original bytes and pointer bind the raw representation; marker paths avoid
    confusing an existing dictionary with a number that was encoded as NaN.
    """
    marked = []
    def visit(row, pointer):
        if isinstance(row, float) and not math.isfinite(row):
            token = 'NaN' if math.isnan(row) else 'Infinity' if row > 0 else '-Infinity'
            marked.append({'pointer': pointer, 'token': token})
            return {'__fuxi_legacy_nonfinite_number__': token}
        if isinstance(row, dict): return {k: visit(v, pointer+'/'+k) for k, v in row.items()}
        if isinstance(row, list): return [visit(v, pointer+'/'+str(i)) for i, v in enumerate(row)]
        return row
    return visit(value, ''), marked


def _write_new(path, body):
    path = Path(path)
    encoded = (json.dumps(body, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2) + '\n').encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): require(path.read_bytes() == encoded, 'immutable catalog output differs: ' + str(path))
    else:
        with path.open('xb') as stream: stream.write(encoded)


def build_catalog(workspace, output):
    """Index complete selected assets, preserving states and duplicate appearances.

    Source policy is explicit: current data/*factor*.{csv,json}, experience/JQ
    memory, saved cards and published candidate/manifest pairs in reports and
    data/fuxi_v1/fuxi_unified, plus static FA factors/config/package sources.
    CSV backup-named files remain separately marked historical assets. No .bak
    files, raw market files, test scratch or unrelated report JSON are consumed.
    """
    root, out = Path(workspace).resolve(), Path(output).resolve()
    paths = _sources(root)
    require(paths, 'no shared factor sources')
    require(all(out != p and out not in p.parents for p in paths), 'output cannot replace a source')
    rows, sources, by_name = [], [], defaultdict(list)
    for path in paths:
        before = file_hash(path); blob = path.read_bytes()
        require(file_hash(path) == before, 'source changed while read')
        source = {'path': str(path), 'relative_path': str(path.relative_to(root)), 'sha256': before,
                  'bytes': len(blob), 'historical_copy': any(x in path.name for x in ('backup', 'before', 'archived'))
                      or 'sealed_delivery' in path.parts, 'row_count': 0}
        source['asset_type'] = ('source_module_asset' if path.suffix == '.py' else
            'published_native_candidate' if _published_pair(path) and path.name == 'candidate.json' else
            'published_bundle_manifest' if _published_pair(path) else
            'saved_native_card_source' if path.parent.name == 'cards' else 'legacy_structured_asset')
        binding = _publication_binding(path) if _published_pair(path) else None
        if binding: source['publication_binding'] = binding
        counts, kinds = Counter(), Counter()
        for pointer, original, fallback, section in _records(path, blob):
            original, nonfinite_markers = _explicit_nonfinite(original)
            original_hash = digest(original)
            obj = original if isinstance(original, dict) else {}
            name = _name(original, fallback, section)
            formula = next((obj[k] for k in ('formula', 'formula_pandas', 'expression', 'local_formula')
                            if isinstance(obj.get(k), str) and obj[k].strip()), None)
            status = obj.get('status', '__missing__')
            status_key = str(status) if isinstance(status, (str, int, float, bool)) else '__structured_status__'
            counts[status_key] += 1; kinds[section] += 1
            native = obj.get('canonical_ast')
            mapping = None
            reason = 'formula_missing'
            try:
                if path.suffix == '.py':
                    reason = 'static_source_metadata_not_evaluable'
                elif section == 'publication_manifest_metadata':
                    reason = 'publication_metadata_not_evaluation_or_qualification'
                elif section == 'native_candidate' and obj.get('schema_version') != 'fuxi.factor.2':
                    reason = 'unsupported_native_candidate_schema'
                elif native is not None:
                    reverse = reverse_projection(native)
                    mapping = translate_expression(reverse['formula'])
                elif formula:
                    mapping = translate_expression(formula)
                if mapping: reason = 'mapped_under_explicit_raw_signal_contract'
            except (ContractError, KeyError, TypeError) as exc:
                reason = str(exc)
            row = {'source_sha256': before, 'source_path': str(path), 'source_pointer': pointer,
                'source_asset_type': source['asset_type'],
                'section': section, 'original_record': original, 'original_record_hash': original_hash,
                'source_nonfinite_markers': nonfinite_markers,
                'name': name, 'formula': formula, 'historical_declared_status': status,
                'historical_direction': obj.get('direction'), 'source_semantic_contract_verified': False,
                'evaluation_ready': False,
                'formula_text_hash': digest(formula) if formula else None,
                'mapping_status': 'supported_signal_contract' if mapping else 'unsupported_or_missing',
                'mapping_reason': reason, 'semantic_factor_id': mapping['semantic_factor_id'] if mapping else None,
                'mapping': mapping, 'current_qualification': 'unknown_requires_live_authority',
                'historical_receipt_reference': obj.get('qualification_receipt_id'),
                'qualification_receipt_id': None, 'eligible_for_jq': False,
                'qualification_transferred': False, 'live_execution_allowed': False}
            if binding: row['publication_binding'] = binding
            row['entry_id'] = digest([str(path.relative_to(root)), before, pointer, original_hash])
            rows.append(row)
            if name: by_name[name].append(row)
            source['row_count'] += 1
        source['status_counts'] = dict(sorted(counts.items())); source['section_counts'] = dict(sorted(kinds.items()))
        sources.append(source)
    conflicts = []
    for name, appearances in sorted(by_name.items()):
        formulas = sorted({r['formula_text_hash'] for r in appearances if r['formula_text_hash']})
        semantic = sorted({r['semantic_factor_id'] for r in appearances if r['semantic_factor_id']})
        states = sorted({str(r['historical_declared_status']) for r in appearances})
        if len(formulas) > 1 or len(semantic) > 1 or len(states) > 1:
            conflicts.append({'name': name, 'appearances': len(appearances), 'formula_text_hashes': formulas,
                'supported_semantic_ids': semantic, 'historical_states': states,
                'entry_ids': [r['entry_id'] for r in appearances], 'resolution': 'retain_all_no_name_equivalence'})
    require(sum(s['row_count'] for s in sources) == len(rows) and len(rows) > 0, 'catalog denominator mismatch')
    require(len({r['entry_id'] for r in rows}) == len(rows), 'duplicate source appearance identity')
    summary = {'schema_version': SCHEMA, 'source_count': len(sources), 'input_sources': sources,
        'row_count': len(rows), 'named_rows': sum(bool(r['name']) for r in rows), 'unique_names': len(by_name),
        'missing_name_rows': sum(not r['name'] for r in rows),
        'missing_formula_rows': sum(r['formula'] is None and r['mapping'] is None for r in rows),
        'supported_signal_rows': sum(r['mapping'] is not None for r in rows),
        'unsupported_or_missing_rows': sum(r['mapping'] is None for r in rows),
        'native_card_rows': sum(r['section'] == 'native_card' for r in rows),
        'published_native_candidate_rows': sum(r['section'] == 'native_candidate' for r in rows),
        'publication_manifest_metadata_rows': sum(r['section'] == 'publication_manifest_metadata' for r in rows),
        'source_module_assets': sum(r['section'] == 'source_module_asset' for r in rows),
        'source_code_definition_rows': sum(r['section'] == 'source_code_definition' for r in rows),
        'asset_type_counts': dict(sorted(Counter(s['asset_type'] for s in sources).items())),
        'legacy_nonfinite_cells': sum(len(r['source_nonfinite_markers']) for r in rows),
        'conflict_name_count': len(conflicts), 'status_counts': dict(sorted(Counter(str(r['historical_declared_status']) for r in rows).items())),
        'qualification_granted_count': 0, 'source_rows_filtered_by_status': 0,
        'source_policy': 'data primary factor CSV/JSON + experience/JQ memory + all saved native cards + published candidate/manifest pairs in reports/data-fuxi-v1/data-fuxi-unified + static FA factors/config/package sources; complete original bytes and metadata retained',
        'count_unit': 'source section entries and metadata; repeated snapshots/references retained, not independent factors',
        'semantic_equivalence_scope': 'explicit reviewed raw signal contract; no automatic production parity',
        'catalog_rows_hash': digest(rows), 'data_contract_hash': data_contracts()['contract_hash']}
    summary['catalog_hash'] = digest(summary)
    for source in sources:
        path = Path(source['path'])
        require(file_hash(path) == source['sha256'], 'catalog input changed before publication')
        binding = source.get('publication_binding')
        if binding:
            require(file_hash(binding['manifest_path']) == binding['manifest_sha256'] and
                    file_hash(Path(binding['manifest_path']).with_name('candidate.json')) == binding['candidate_sha256'],
                    'published pair changed before catalog publication')
        original_path = out / 'original_sources' / (source['sha256'] + path.suffix)
        original_path.parent.mkdir(parents=True, exist_ok=True)
        if original_path.exists(): require(file_hash(original_path) == source['sha256'], 'original source archive changed')
        else:
            with original_path.open('xb') as stream: stream.write(path.read_bytes())
    _write_new(out / 'DATA_CONTRACTS.json', data_contracts())
    _write_new(out / 'CATALOG.json', {'schema_version': SCHEMA, 'rows': rows})
    _write_new(out / 'CONFLICTS.json', {'rows': conflicts, 'conflict_name_count': len(conflicts)})
    _write_new(out / 'SUMMARY.json', summary)
    return summary


def _legacy_values(expression, snapshot, session_index):
    """Independent pandas execution of a prevalidated unchanged expression.

    Each decision builds its own action-barrier/as-of masked field frame. This
    proves the explicit bridge contract, not old adjusted/ffill production use.
    """
    import numpy as np
    import pandas as pd
    mapping = translate_expression(expression, snapshot.payload['fields'])
    lo = max(0, session_index - mapping['lookback'])
    members = snapshot.universe[snapshot.sessions[session_index]]
    clock = snapshot.clocks[snapshot.sessions[session_index]]
    namespace = {}
    for alias, field in FIELD_ALIASES.items():
        if field in mapping['required_fields']:
            namespace[alias] = pd.DataFrame([[snapshot.value(i, s, field, clock) for s in members]
                for i in range(lo, session_index + 1)], columns=members, dtype=float)
    # translate_expression accepts only field arithmetic and the fixed safe
    # pandas method subset above; arbitrary Python, kwargs or attributes reject.
    value = eval(compile(expression, '<frozen_legacy_expression>', 'eval'), {'__builtins__': {}}, namespace)
    row = value.iloc[-1]
    return {s: float(row[s]) if np.isfinite(row[s]) else None for s in members}


def prove_representatives(workspace, output, input_path=None, *, decision_count=20):
    """Fixed three representative signals, raw field/clock comparison only; no economics."""
    from .data import snapshot_from_payload
    root, out = Path(workspace).resolve(), Path(output).resolve()
    source = Path(input_path).resolve() if input_path else root / DEFAULT_INPUT
    pool = root / 'data/unified_factor_pool.csv'
    require(type(decision_count) is int and decision_count > 0, 'positive comparison scope required')
    source_sha, pool_sha = file_hash(source), file_hash(pool)
    payload = load_json(source)
    original = payload['snapshot']
    require(original['schema_version'] == 'fuxi.historical_snapshot.1' and
            original['exposure'] == 'research_exposed', 'frozen compact historical input required')
    require(len(original['instruments']) == 64, 'fixed original 64-member support required')
    sessions = original['sessions']; start = sessions.index(original['evaluation_start'])
    # Tail is a fixed physical source selection; signal comparison uses no labels.
    selected = list(range(max(start, len(sessions) - decision_count), len(sessions)))
    require(len(selected) == decision_count, 'insufficient declared signal comparison dates')
    with pool.open(encoding='utf-8-sig', newline='') as stream: pool_rows = list(csv.DictReader(stream))
    reps = []
    for name, expression in REPRESENTATIVES:
        matches = [r for r in pool_rows if r['name'] == name]
        require(len(matches) == 1 and matches[0]['formula'] == expression, 'fixed representative source formula changed: ' + name)
        mapping = translate_expression(expression, original['fields'])
        reps.append({'name': name, 'source_record_hash': digest(matches[0]), 'historical_status': matches[0].get('status'),
                     'mapping': mapping, 'reverse_projection': reverse_projection(mapping['canonical_ast'])})
    protocol = {'schema_version': 'fuxi.shared_signal_proof.1', 'input_path': str(source), 'input_sha256': source_sha,
        'pool_path': str(pool), 'pool_sha256': pool_sha, 'source_snapshot_hash': digest(original),
        'selected_sessions': [sessions[i] for i in selected], 'selection': 'fixed_last_20_source_signal_sessions; no performance/label selection',
        'instruments': original['instruments'], 'representatives': reps,
        'support': 'source current universe; all original lifecycle members retained; no refill; source action barriers',
        'clock_scope': 'field/signal availability under source declared clock, separately from execution',
        'cost_scope': {'evaluated': False, 'source_declared_execution': payload.get('market', {}).get('execution'),
                       'claim': 'signal parity only; no returns/account/qualification/cost equivalence'},
        'historical_PIT_certified': False, 'legacy_production_equivalence_certified': False,
        'absolute_tolerance': 1e-12, 'relative_tolerance': 1e-12,
        'new_market_requests': 0, 'new_future_observations': 0, 'qualification_transferred': False}
    protocol['protocol_hash'] = digest(protocol)
    _write_new(out / 'REPRESENTATIVE_PROTOCOL.json', protocol)
    slim = deepcopy(original)
    slim['evaluation_start'] = sessions[selected[0]]
    snapshot = snapshot_from_payload(slim)
    value_rows, summary_rows = [], []
    for representative in reps:
        mapping = representative['mapping']; expression = mapping['source_expression']
        native = evaluate_ast(mapping['canonical_ast'], snapshot)
        projected = translate_expression(representative['reverse_projection']['formula'], original['fields'])
        reverse = evaluate_ast(projected['canonical_ast'], snapshot)
        counts = Counter(); maximum = 0.0
        for index in selected:
            day = sessions[index]; legacy = _legacy_values(expression, snapshot, index)
            for code in snapshot.universe[day]:
                a, b, c = legacy[code], native[day][code], reverse[day][code]
                counts['n_pairs'] += 1
                same_null = (a is None) == (b is None) == (c is None)
                close = same_null and (a is None or math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
                                      and math.isclose(b, c, rel_tol=1e-12, abs_tol=1e-12))
                counts['null_pairs' if a is None and b is None and c is None else 'numeric_pairs'] += 1
                if not close: counts['mismatches'] += 1
                delta = abs(a-b) if a is not None and b is not None else None
                if delta is not None: maximum = max(maximum, delta)
                value_rows.append({'name': representative['name'], 'session': day, 'instrument': code,
                    'legacy_value': a, 'native_value': b, 'reverse_value': c,
                    'same_null_support': same_null, 'equal_within_tolerance': close, 'absolute_delta': delta,
                    'decision_time': original['decision_times'][day]})
        require(counts['n_pairs'] > 0 and counts['numeric_pairs'] > 0, 'empty representative comparison')
        summary_rows.append({'name': representative['name'], **dict(counts), 'mismatches': counts['mismatches'],
                             'declared_member_grid': len(original['instruments'])*len(selected),
                             'outside_current_source_universe': len(original['instruments'])*len(selected)-counts['n_pairs'],
                             'maximum_absolute_delta': maximum})
    require(file_hash(source) == source_sha and file_hash(pool) == pool_sha, 'source changed during signal proof')
    result = {'protocol_hash': protocol['protocol_hash'], 'representative_count': len(reps), 'rows': summary_rows,
        'total_pairs': len(value_rows), 'mismatches': sum(r['mismatches'] for r in summary_rows),
        'source_inputs_unchanged': True, 'economic_evaluation_performed': False, 'qualification_transferred': False,
        'legacy_production_equivalence_certified': False, 'verdict': 'signal_contract_parity' if all(not r['mismatches'] for r in summary_rows) else 'mismatch'}
    _write_new(out / 'REPRESENTATIVE_VALUES.json', {'rows': value_rows, 'row_count': len(value_rows)})
    _write_new(out / 'REPRESENTATIVE_RESULT.json', result)
    require(result['mismatches'] == 0, 'representative signal mismatch; retain published evidence')
    return result
