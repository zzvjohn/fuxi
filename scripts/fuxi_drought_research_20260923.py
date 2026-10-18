"""Read-only inventory for drought attribution; never treats old PASS as eligibility."""

import ast
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/holdings_20260923'
PRICE_FIELDS = {'open', 'high', 'low', 'close', 'volume', 'vol', 'amount', 'returns',
                'vwap', 'pre_close'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(rows, key):
    counts = Counter(str(row.get(key) or '(missing)') for row in rows)
    total = sum(counts.values())
    hhi = sum((n / total) ** 2 for n in counts.values()) if total else None
    return {'n': total, 'counts': dict(counts.most_common()),
            'top_three_share': sum(n for _, n in counts.most_common(3)) / total if total else None,
            'inverse_hhi': 1 / hhi if hhi else None}


def formula_summary(rows):
    syntax, fields, parse_errors, only_price, present = Counter(), Counter(), 0, 0, 0
    for row in rows:
        formula = str(row.get('formula') or '').strip()
        if not formula:
            continue
        present += 1
        try:
            tree = ast.parse(formula, mode='eval')
        except SyntaxError:
            parse_errors += 1
            continue
        syntax[hashlib.sha256(ast.dump(tree).encode()).hexdigest()] += 1
        used = {node.id.removesuffix('_p') for node in ast.walk(tree)
                if isinstance(node, ast.Name) and
                (node.id.endswith('_p') or node.id in PRICE_FIELDS)}
        fields.update(used)
        only_price += bool(used) and used <= PRICE_FIELDS
    return {'formula_present': present, 'parse_errors': parse_errors,
            'unique_syntactic_ast': len(syntax), 'recognized_price_only': only_price,
            'recognized_fields': dict(fields.most_common()),
            'caveat': 'syntax/recognized names, not empirical signal correlation or proof of economic independence'}


def main():
    names = ['data/experience_memory.json', 'data/trajectory_log.json', 'data/mab_scheduler_state.json',
             'research/factor_alchemy/forge/paradigm_profiles.py',
             'research/factor_alchemy/forge/__init__.py', 'research/factor_alchemy/ralph_loop.py',
             'research/factor_alchemy/model_proxy_eval.py']
    inputs = {name: digest(ROOT / name) for name in names}
    memory = json.loads((ROOT / names[0]).read_text(encoding='utf-8'))
    trajectories = json.loads((ROOT / names[1]).read_text(encoding='utf-8'))['trajectories']
    mab = json.loads((ROOT / names[2]).read_text(encoding='utf-8'))
    arms = mab['directions']
    profiles_tree = ast.parse((ROOT / names[3]).read_text(encoding='utf-8'))
    profiles = next(ast.literal_eval(node.value) for node in profiles_tree.body
                    if isinstance(node, ast.Assign) and
                    any(isinstance(t, ast.Name) and t.id == 'MATURE_PARADIGM_PROFILES' for t in node.targets))
    summary = {'created_at': datetime.now().astimezone().isoformat(), 'source_sha256': inputs,
               'scope': 'stored candidate observations, not independent trials; no writes to live research state',
               'memory_total': len(memory['attempts']), 'trajectory_total': len(trajectories),
               'windows': {}, 'mab': {'updated_at': mab['updated_at'], 'arms': len(arms),
                                     'reported_total_pulls': mab['total_pulls'],
                                     'sum_arm_pulls': sum(d.get('pulls', 0) for d in arms),
                                     'distinct_paradigms': len({d['paradigm'] for d in arms}),
                                     'model_types': distribution(arms, 'model_type'),
                                     'arms_without_rewards': sum(not d.get('n_rewards', 0) for d in arms),
                                     'arms_without_reward_source': sum(not d.get('reward_source') for d in arms),
                                     'arms_with_generator_rewards': sum(bool(d.get('generator_rewards')) for d in arms),
                                     'reward_sources': distribution(arms, 'reward_source'),
                                     'status': distribution(arms, 'status')},
               'forge_profiles': {'names': list(profiles),
                                  'weighted_terminal_union': sorted(set().union(*(set(p['terminal_weights']) for p in profiles.values()))),
                                  'caveat': 'profiles weight the same terminal family; vwap may be disabled by data contract'}}
    for start in ['2026-09-02', '2026-09-17']:
        end = '2026-09-23'
        attempts = [r for r in memory['attempts'] if start <= str(r.get('timestamp', ''))[:10] < end]
        traces = [r for r in trajectories if start <= str(r.get('created_at', ''))[:10] < end]
        actual_logs = [r for r in traces if r.get('_eval_log')]
        logs = [entry for row in actual_logs for entry in row['_eval_log']]
        summary['windows'][start + '_to_2026-09-22'] = {
            'memory_observations': len(attempts), 'paradigms': distribution(attempts, 'paradigm'),
            'engines': distribution(attempts, 'engine'), 'formula_syntax': formula_summary(attempts),
            'trajectory_observations': len(traces), 'trajectories_with_structured_eval_log': len(actual_logs),
            'structured_eval_events': len(logs), 'terminal_event_stages': distribution(logs, 'stage'),
            'caveat': 'historical records may use older evaluator; stage events are not an S1-S5 funnel or valid economic failure count'}
    consumers, unparsed = [], []
    for path in (ROOT / 'research/factor_alchemy').rglob('*.py'):
        if path.name == 'model_proxy_eval.py':
            continue
        try:
            tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        except SyntaxError as exc:
            unparsed.append({'file': path.relative_to(ROOT).as_posix(), 'line': exc.lineno})
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or '').endswith('model_proxy_eval'):
                consumers.append({'file': path.relative_to(ROOT).as_posix(), 'line': node.lineno})
    summary['model_proxy_static_import_consumers'] = consumers
    summary['static_scan_unparsed_files'] = unparsed
    summary['live_cooptim_results_exists'] = (ROOT / 'data/cooptim_results.json').exists()
    summary['inputs_unchanged'] = all(digest(ROOT / name) == value for name, value in inputs.items())
    assert summary['inputs_unchanged']
    OUT.mkdir(exist_ok=True)
    (OUT / 'research_inventory.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
