"""Bounded, source-bound fusion experiments; research feedback never grants authority.

The shared control permits two one-axis comparisons: remembered versus unremembered
parents under one emitter schedule, and fixed versus adaptive emitter scheduling
with memory.  Callers own the workspace writer lock.  No data or LLM IO occurs here.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
import argparse
import json
from pathlib import Path
import sqlite3
from statistics import mean

from .artifacts import code_manifest, verify
from .campaign import run_campaign, write_new
from .contracts import digest, load_json, require, timestamp
from .data import Snapshot, Labels
from .datasets import file_hash, validate_label_calendar
from .dsl import ranks
from .evaluation import correlation


SCHEMA = 'fuxi.fusion_study.1'
ARMS = ('fixed_memory', 'fixed_no_memory', 'adaptive_memory')
SCHEDULE = ('fixed', 'random', 'refine', 'combine', 'refine')


def derive_prefix(payload, session_count):
    """Take a calendar prefix, retaining exact observations, nulls and barrier lineage.

    Labels whose source-defined endpoints leave the prefix remain absent at the
    tail.  No values, dates, original availability, source rules or costs are edited.
    """
    require(payload.get('schema_version') == 'fuxi.campaign_input.1', 'native campaign input required')
    original = payload['snapshot']
    require(original.get('schema_version') == 'fuxi.snapshot.2', 'fusion prefix needs revision-aware native64 input')
    require(original.get('exposure') == 'research_exposed', 'only already exposed historical research input allowed')
    require(type(session_count) is int and 120 <= session_count <= len(original['sessions']), 'prefix session bound')
    result = deepcopy(payload)
    sessions = original['sessions'][:session_count]
    days = set(sessions)
    original_supported = set().union(*(set(v) for v in original['universe'].values()))
    declared_members = original.get('provenance', {}).get('original_cohort_declaration', {}).get('members')
    if declared_members:
        from .history import jq_code
        declared = sorted(jq_code(member['ts_code']) for member in declared_members)
        require(len(declared) == len(set(declared)) and original_supported <= set(declared),
                'original declared cohort/member support mismatch')
    else:
        declared = sorted(original_supported)
    snapshot = result['snapshot']
    snapshot['sessions'] = sessions
    snapshot['decision_times'] = {d: original['decision_times'][d] for d in sessions}
    snapshot['universe'] = {d: original['universe'][d] for d in sessions}
    snapshot['observations'] = [r for r in original['observations'] if r['session'] in days]
    require(snapshot['evaluation_start'] in days, 'prefix removes evaluation start/lookback')
    snapshot['provenance']['fusion_prefix'] = {
        'original_snapshot_hash': digest(original), 'session_count': session_count,
        'original_declared_instruments': declared,
        'selection': 'first declared calendar sessions; no outcome-based window or member selection',
        'observation_values_availability_nulls_and_barrier_lineage_unchanged': True}
    labels = result['labels']
    labels['rows'] = [r for r in payload['labels']['rows'] if r['session'] in days
                     and r['label_start'][:10] in days and r['label_end'][:10] in days]
    labels['feature_snapshot_hash'] = digest(snapshot)
    market = result['market']
    market['sessions'] = sessions
    market['bars'] = [r for r in payload['market']['bars'] if r['session'] in days]
    # Date-aware event consumers ignore terms outside their replay range.  Keeping
    # the complete original terms preserves independently bound event evidence.
    market['market_hash'] = digest({k: v for k, v in market.items() if k != 'market_hash'})
    checked = Snapshot(snapshot)
    Labels(labels, checked)
    validate_label_calendar(labels, sessions)
    # Members first eligible after the prefix remain declared in provenance,
    # while their historical universe membership is not manufactured.
    require(labels['rows'] and snapshot['observations'] and market['bars'], 'empty derived research support')
    return result


def _request(request):
    require(isinstance(request, dict), 'study request required')
    require(set(request) <= {'session_count', 'seed', 'rounds', 'candidates_per_round', 'source_sha256'},
            'unknown study request fields')
    value = {'session_count': 180, 'seed': 17, 'rounds': 5, 'candidates_per_round': 2, **request}
    require(value['rounds'] == 5 and value['candidates_per_round'] == 2,
            'fusion paired pilot is frozen at five rounds/two candidates')
    require(type(value['seed']) is int and value['seed'] >= 0, 'study seed invalid')
    return value


def freeze(input_path, output, request):
    root = Path(output).resolve()
    require(not (root / 'PROTOCOL.json').exists(), 'study already frozen; use run/status')
    source = Path(input_path).resolve()
    require(source.is_file(), 'study source input missing')
    req = _request(request)
    source_hash = file_hash(source)
    require(not req.get('source_sha256') or req['source_sha256'] == source_hash, 'study source hash mismatch')
    original = load_json(source)
    payload = derive_prefix(original, req['session_count'])
    sessions = payload['snapshot']['sessions']
    horizon = payload['labels']['label_spec']['horizon_sessions']
    checked = Snapshot(payload['snapshot'])
    cal_index = max(checked.start + 25, min(80, len(sessions) // 2))
    native_request_window = {'calibration_end': sessions[cal_index],
                             'research_start': sessions[cal_index + horizon + 2], 'research_end': sessions[-1]}
    protocol = {'schema_version': SCHEMA, 'frozen_at': datetime.now().astimezone().isoformat(),
        'source': {'path': str(source), 'sha256': source_hash, 'payload_hash': digest(original)},
        'request': req, 'input_hash': digest(payload), 'snapshot_hash': digest(payload['snapshot']),
        'label_hash': digest(payload['labels']), 'cost_profile_hash': digest(payload['market']['execution']),
        'code_hashes': code_manifest(), 'arms': list(ARMS), 'fixed_schedule': list(SCHEDULE),
        'native_window': native_request_window, 'input_window': [sessions[0], sessions[-1]],
        'members': len(payload['snapshot']['provenance']['fusion_prefix']['original_declared_instruments']),
        'observed_members_in_prefix': len(set().union(*(set(v) for v in payload['snapshot']['universe'].values()))),
        'paired_axes': [{'left': 'fixed_memory', 'right': 'fixed_no_memory', 'axis': 'memory'},
                        {'left': 'fixed_memory', 'right': 'adaptive_memory', 'axis': 'scheduler'}],
        'budget': {'candidate_limit_per_arm': 10, 'candidate_limit_total': 30,
            'native_reserved_worker_wall_seconds_per_arm': 1620,
            'new_market_requests': 0, 'LLM_calls': 0, 'model_algorithms': []},
        'scope': 'already exposed latest-vintage native64 prefix; engineering/feedback diagnostic',
        'native_gate_changes': False, 'legacy_S5_cost_equivalence_asserted': False,
        'qualification_transfer_allowed': False, 'live_execution_allowed': False}
    protocol['protocol_hash'] = digest(protocol)
    root.mkdir(parents=True, exist_ok=True)
    write_new(root / 'INPUT.json', payload)
    for name in protocol['code_hashes']:
        write_new(root / 'code' / name, (Path(__file__).parent / name).read_bytes())
    write_new(root / 'PROTOCOL.json', protocol)
    return protocol


def _checked(output, *, current_code=False):
    root = Path(output).resolve()
    protocol = load_json(root / 'PROTOCOL.json')
    require(protocol.get('schema_version') == SCHEMA and
            protocol['protocol_hash'] == digest({k: v for k, v in protocol.items() if k != 'protocol_hash'}),
            'fusion study protocol changed')
    require(file_hash(protocol['source']['path']) == protocol['source']['sha256'], 'study source drift')
    require(digest(load_json(root / 'INPUT.json')) == protocol['input_hash'], 'study input drift')
    for name, expected in protocol['code_hashes'].items():
        require(file_hash(root / 'code' / name) == expected, 'frozen study code changed: ' + name)
    if current_code:
        require(code_manifest() == protocol['code_hashes'], 'study code drift; new freeze required')
    return root, protocol


def _facts(directory, kind):
    path = (Path(directory) / 'evidence.sqlite').resolve()
    connection = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
    try:
        result = []
        for body, expected in connection.execute('SELECT payload,hash FROM facts WHERE kind=? ORDER BY rowid', (kind,)):
            row = json.loads(body)
            require(digest(row) == expected, 'study consumed corrupt native fact')
            result.append(row)
        return result
    finally:
        connection.close()


def _summarize_arm(root, arm, directory, report_path, report):
    decisions = _facts(directory, 'research_decision')
    feedback = _facts(directory, 'feedback_v2')
    candidates = _facts(directory, 'candidate')
    evaluated_ids = {r['candidate_id'] for r in report['candidate_results'] if r.get('evaluation_id')}
    expected_snapshot = load_json(root / 'PROTOCOL.json')['snapshot_hash']
    for card in feedback:
        require(card['snapshot_hash'] == expected_snapshot, 'feedback snapshot mismatch')
        require(card['policy_hash'] == report['policy']['policy_hash']
                and card['qualification_transfer_allowed'] is False, 'feedback policy/authority mismatch')
    results = report['candidate_results']
    parent_pairs = [p for r in results for p in r.get('parent_comparisons', [])]
    valid_parent_pairs = [p for p in parent_pairs if p.get('status') == 'valid' and p.get('n_pairs', 0) > 0]
    worker_seconds = sum(r.get('process', {}).get('elapsed_seconds', 0) for r in results)
    actual_seconds = worker_seconds + sum(r.get('portfolio_action', {}).get('elapsed_seconds', 0) for r in report['rounds'])
    native_config = load_json(Path(directory) / 'config.json')
    require(native_config['input_hash'] == digest(load_json(root / 'INPUT.json')),
            'native arm input differs from frozen common input')
    stages = Counter(r.get('e3_status', 'error' if r['eval_status'] == 'error' else 'not_evaluated') for r in results)
    return {'arm': arm, 'campaign_id': report['campaign_id'], 'native_status': report['status'],
        'report_path': str(Path(report_path).resolve()), 'report_sha256': file_hash(report_path),
        'report_hash': digest(report), 'evidence_path': str((Path(directory) / 'evidence.sqlite').resolve()),
        'config_hash': digest(native_config), 'input_hash': native_config['input_hash'],
        'policy_hash': report['policy']['policy_hash'], 'native_request_hash': digest(report['policy']['request']),
        'completed_nonempty_rounds': sum(r['evaluated_count'] > 0 for r in report['rounds']),
        'attempted_candidates': report['attempted_candidates'], 'unique_formulas': report['unique_formula_count'],
        'valid_descriptive_factors': report['valid_descriptive_factors'],
        'valid_distinct_ordinal_behaviors': len({r['behavior_hash'] for r in results if r.get('valid_e1') and r.get('behavior_hash')}),
        'engineering_errors': report['engineering_errors'], 'E1_pass_count': sum(r.get('e1_decision') == 'pass' for r in results),
        'account_status_counts': dict(stages), 'formal_account_valid': report['formal_account_valid'],
        'native_JQ_candidate_count': report['jq_candidate_count'], 'LLM_calls': report['llm_calls'],
        'capability_rewards': report['capability_reward_count'], 'economic_rewards': report['economic_reward_count'],
        'feedback_cards': len(feedback), 'memory_enabled_decisions': sum(d['memory_enabled'] for d in decisions),
        'evaluated_parent_selection_count': sum(len(set(d['parent_ids']) & evaluated_ids) for d in decisions),
        'candidate_parent_ids': sum(len(c['parent_ids']) for c in candidates),
        'actual_parent_comparisons': len(parent_pairs), 'valid_parent_comparisons': len(valid_parent_pairs),
        'positive_descriptive_parent_comparisons': sum(p['mean_ic_increment'] > 0 for p in valid_parent_pairs),
        'parent_pair_security_cells': sum(p['n_pairs'] for p in valid_parent_pairs),
        'worker_wall_seconds': worker_seconds, 'observed_native_action_wall_seconds': actual_seconds,
        'reserved_worker_wall_seconds': native_config['budget_units'], 'CPU_seconds': None,
        'CPU_measurement_status': 'not_measured; native worker budget is wall-clock seconds',
        'valid_evaluations_per_reserved_worker_second': report['valid_descriptive_factors'] / native_config['budget_units'],
        'valid_evaluations_per_observed_worker_wall_second': report['valid_descriptive_factors'] / worker_seconds if worker_seconds else None,
        'actual_arm_sequence': [r['arm'] for r in report['rounds']],
        'acquisition_rules': [d['acquisition']['rule'] for d in decisions],
        'statistical_and_economic_winner_declared': False}


def _method_pairs(root, protocol, reports, left, right, axis):
    """Match method summaries on exact day/security/label support before comparing IC.

    Candidate-specific parent increments remain a separate native statistic.  A
    method mean is equal weight over this round's emitted candidates, not an
    invested portfolio or a post-hoc best-candidate selection.
    """
    payload = load_json(root / 'INPUT.json')
    snapshot = Snapshot(payload['snapshot'])
    labels = Labels(payload['labels'], snapshot)
    clock = timestamp(payload['research_clock'])
    window = protocol['native_window']
    rows = []
    tables = {}
    for arm in (left, right):
        by_round = defaultdict(list)
        for candidate in reports[arm]['candidate_results']:
            if not candidate.get('bundle_path'):
                continue
            bundle = root / 'campaigns' / candidate['bundle_path']
            verify(bundle)
            scores = load_json(bundle / 'scores.json')
            by_round[candidate['round']].append((candidate['candidate_id'], scores))
        tables[arm] = by_round
    for round_index in sorted(set(tables[left]) & set(tables[right])):
        groups = {arm: tables[arm][round_index] for arm in (left, right)}
        for day in snapshot.sessions:
            if not window['research_start'] <= day <= window['research_end']:
                continue
            support = []
            for security in snapshot.universe[day]:
                label = labels.value(day, security, clock)
                values = [scores[day].get(security) for group in groups.values() for _, scores in group]
                if label is not None and all(v is not None for v in values):
                    support.append((security, label))
            if len(support) < 5:
                continue
            ranked_labels = list(ranks(dict(support)).values())
            method_readings = {}
            for arm in (left, right):
                values = [correlation(list(ranks({security: scores[day][security] for security, _ in support}).values()),
                                      ranked_labels) for _, scores in groups[arm]]
                method_readings[arm] = mean(values) if all(v is not None for v in values) else None
            if any(v is None for v in method_readings.values()):
                continue
            rows.append({'round': round_index, 'session': day, 'n_pairs': len(support),
                'support_hash': digest(support), 'left_mean_candidate_ic': method_readings[left],
                'right_mean_candidate_ic': method_readings[right],
                'right_minus_left': method_readings[right] - method_readings[left],
                'left_candidate_count': len(groups[left]), 'right_candidate_count': len(groups[right])})
    require(rows and sum(r['n_pairs'] for r in rows) > 0, 'method comparison has zero common support')
    per_day = defaultdict(list)
    for row in rows:
        per_day[row['session']].append(row['right_minus_left'])
    return {'axis': axis, 'left': left, 'right': right, 'eval_status': 'valid',
        'comparison': 'same-round equal-weight candidate IC; exact common security/day/label support',
        'n_round_day_pairs': len(rows), 'n_security_pairs_repeated_across_rounds': sum(r['n_pairs'] for r in rows),
        'distinct_calendar_sections': len(per_day), 'rows': rows,
        'mean_daily_method_ic_difference': mean(mean(v) for v in per_day.values()),
        'independent_sample_count_asserted': False, 'economic_reward_eligible': False,
        'qualification_granted': False, 'winner_declared': False,
        'interpretation': 'adaptive already-exposed history; repeated rounds do not add independent dates'}


def run(output, *, authority_path=None):
    root, protocol = _checked(output, current_code=True)
    if (root / 'SUMMARY.json').exists():
        return status(root)['summary']
    request = protocol['request']
    reports, summaries = {}, {}
    for arm in protocol['arms']:
        _checked(root, current_code=True)
        record_path = root / 'arms' / (arm + '.json')
        if record_path.exists():
            record = load_json(record_path)
            require(file_hash(record['report_path']) == record['report_sha256'], 'cached arm report changed')
            reports[arm] = load_json(record['report_path'])
            summaries[arm] = record
            continue
        scheduled = arm != 'adaptive_memory'
        controlled = {'schema_version': 'fuxi.controlled_study.1', 'arm_schedule': protocol['fixed_schedule'],
                      'max_trials': 10, 'checkpoint_every': 5, 'memory_sources': []} if scheduled else None
        identity = 'campaign_fusion' + protocol['protocol_hash'][:12] + '_' + arm
        path, report = run_campaign(root / 'INPUT.json', root / 'campaigns',
            rounds=request['rounds'], candidates_per_round=request['candidates_per_round'],
            mode='no_memory' if arm == 'fixed_no_memory' else 'adaptive', seed=request['seed'],
            llm_enabled=False, model_algorithms=(), campaign_id=identity, authority_path=authority_path,
            study_protocol=controlled, scheduler_policy='selection_v2')
        require(report['status'] == 'completed' and report['llm_calls'] == 0,
                'fusion arm did not complete declared bounded research')
        record = _summarize_arm(root, arm, root / 'campaigns' / identity, path, report)
        write_new(record_path, record)
        reports[arm], summaries[arm] = report, record
    require(len({r['input_hash'] for r in summaries.values()}) == 1
            and len({r['native_request_hash'] for r in summaries.values()}) == 1,
            'paired study common input/window/native policy mismatch')
    require(all(r['attempted_candidates'] > 0 for r in summaries.values()), 'empty arm comparison')
    comparisons = [_method_pairs(root, protocol, reports, **p) for p in protocol['paired_axes']]
    comparison_files = {}
    for comparison in comparisons:
        path = root / 'comparisons' / (comparison['axis'] + '.json')
        write_new(path, comparison)
        comparison_files[path.relative_to(root).as_posix()] = file_hash(path)
    summary = {'schema_version': SCHEMA, 'protocol_hash': protocol['protocol_hash'],
        'status': 'completed_research_diagnostic', 'arms': summaries,
        'paired_comparisons': [{k: v for k, v in c.items() if k != 'rows'} for c in comparisons],
        'comparison_files': comparison_files,
        'source_sha256': protocol['source']['sha256'], 'input_hash': protocol['input_hash'],
        'code_hashes': protocol['code_hashes'], 'input_window': protocol['input_window'],
        'native_window': protocol['native_window'], 'members': protocol['members'],
        'actual_candidates': sum(r['attempted_candidates'] for r in summaries.values()),
        'new_market_requests': 0, 'LLM_calls': 0, 'native_gate_changes': False,
        'economic_status': 'data_insufficient' if not any(r['formal_account_valid'] for r in summaries.values()) else 'inspect_native_per_candidate_evidence',
        'economic_rewards': sum(r['economic_rewards'] for r in summaries.values()),
        'native_JQ_candidate_count': sum(r['native_JQ_candidate_count'] for r in summaries.values()),
        'statistical_and_economic_winner_declared': False, 'automatic_default_changes': False,
        'live_execution_allowed': False}
    summary['summary_hash'] = digest(summary)
    _checked(root, current_code=True)
    write_new(root / 'SUMMARY.json', summary)
    return summary


def status(output):
    root, protocol = _checked(output)
    summary = load_json(root / 'SUMMARY.json') if (root / 'SUMMARY.json').exists() else None
    if summary:
        require(summary['summary_hash'] == digest({k: v for k, v in summary.items() if k != 'summary_hash'})
                and summary['protocol_hash'] == protocol['protocol_hash'], 'fusion summary changed')
        for arm, record in summary['arms'].items():
            require(load_json(root / 'arms' / (arm + '.json')) == record
                    and file_hash(record['report_path']) == record['report_sha256'], 'fusion arm evidence changed')
        for name, expected in summary['comparison_files'].items():
            require(file_hash(root / name) == expected, 'fusion common-support comparison changed')
    return {'schema_version': SCHEMA, 'state': summary['status'] if summary else 'frozen_or_partial',
            'protocol_hash': protocol['protocol_hash'], 'summary': summary, 'live_execution_allowed': False}


def run_study(input_path, output, request, *, authority_path=None):
    if not (Path(output) / 'PROTOCOL.json').exists():
        freeze(input_path, output, request)
    else:
        _, protocol = _checked(output)
        require(protocol['source']['path'] == str(Path(input_path).resolve())
                and protocol['request'] == _request(request), 'run_study request/source changed')
    return run(output, authority_path=authority_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'run', 'status'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--input')
    parser.add_argument('--request')
    parser.add_argument('--authority')
    args = parser.parse_args()
    if args.action == 'status':
        answer = status(args.output)
    else:
        from .runtime import controlled_writer, authority_location
        with controlled_writer(args.output):
            if args.action == 'freeze':
                require(args.input and args.request, 'freeze requires input and request')
                answer = freeze(args.input, args.output, load_json(args.request))
            else:
                answer = run(args.output, authority_path=args.authority or authority_location(args.output))
    print(json.dumps(answer, ensure_ascii=False))
