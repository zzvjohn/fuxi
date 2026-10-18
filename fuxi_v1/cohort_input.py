"""Immutable identifier-sampled research inputs; no redraw or evidence upgrade."""
from copy import deepcopy
from collections import Counter
from pathlib import Path
from .contracts import load_json, digest, require, finite
from .datasets import file_hash, read_calendar
from .pit_universe import read_lifecycle_file, reproduce_cohort, jq_code
from .campaign import write_new


def _reference(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': file_hash(path)}


def verify_declared_cohort(plan, catalog_root, sessions):
    """Rebuild membership from the original lifecycle/rule and bind its receipt."""
    cohort = plan['cohort']
    require(cohort.get('rule_kind') == 'full_population_hash_sample', 'unsupported sampled input rule')
    require(plan.get('account_evidence_mode') == 'source_bound_research', 'sampled input evidence mode')
    binding = plan.get('cohort_input_binding')
    require(isinstance(binding, dict), 'sampled input original binding missing')
    for name in ('protocol', 'draw', 'lifecycle'):
        item = binding[name]
        require(file_hash(item['path']) == item['sha256'], 'sampled input source changed: ' + name)
    protocol, draw = load_json(binding['protocol']['path']), load_json(binding['draw']['path'])
    require(protocol['declaration']['rule_kind'] == 'full_population_hash_sample', 'original sampled rule changed')
    require(protocol['draw_attempt_budget'] == 1, 'original draw budget changed')
    require(sessions and [sessions[0], sessions[-1]] == protocol['window'], 'sampled input window changed')
    require([plan['window']['start'], plan['window']['end']] == protocol['window'], 'sampled plan window changed')
    require(binding['lifecycle']['sha256'] == protocol['lifecycle_sha256'] and
            str(Path(binding['lifecycle']['path']).resolve()) == str(Path(protocol['lifecycle_path']).resolve()),
            'original lifecycle identity changed')
    require(file_hash(Path(catalog_root) / 'delist_dates.csv') == protocol['lifecycle_sha256'],
            'catalog lifecycle differs from draw')
    require(all(cohort.get(k) == v for k, v in protocol['declaration'].items()), 'sampled declaration changed')
    require(draw['declaration_hash'] == digest(protocol['declaration']), 'draw declaration hash changed')
    life = read_lifecycle_file(binding['lifecycle']['path'])
    codes, basis = reproduce_cohort(life, protocol['declaration'], sessions)
    require(codes and basis['permutation_invariant'] and len(codes) > 0, 'sampled rule cannot be reproduced')
    require(codes == draw['codes'], 'saved draw differs from reproduced rule')
    actual = [member['ts_code'] for member in cohort['members']]
    require(actual == codes, 'sampled members removed, replaced, duplicated or reordered')
    require(not cohort.get('posthoc_exclusion') and not plan.get('universe_restriction'), 'sampled posthoc exclusion')
    for row in cohort['members']:
        require(all(row.get(k) == life[row['ts_code']][k] for k in ('list_date', 'delist_date', 'list_status')),
                'sampled member lifecycle changed')
    return {'codes': codes, 'basis': basis, 'declaration_hash': draw['declaration_hash'],
            'original_draw_sha256': binding['draw']['sha256'], 'formal_data_assured': False}


def freeze(protocol_path, draw_path, catalog_root, parent_plan_path, output, *, halted_distribution_policy=None,
           issuer_action_bundle=None, independent_event_directory=None):
    """Freeze one real cohort without reading returns or replacing missing members."""
    root, catalog = Path(output), Path(catalog_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    require(not (root / 'HISTORY_PLAN.json').exists(), 'sampled input already frozen; resume build')
    protocol, draw, parent = map(load_json, (protocol_path, draw_path, parent_plan_path))
    require(protocol['declaration']['rule_kind'] == 'full_population_hash_sample', 'unsupported original draw')
    require(draw['declaration_hash'] == digest(protocol['declaration']), 'draw protocol mismatch')
    lifecycle = Path(protocol['lifecycle_path'])
    require(file_hash(lifecycle) == protocol['lifecycle_sha256'], 'original lifecycle changed')
    manifest, metadata = load_json(catalog / 'CATALOG.json'), load_json(catalog / 'CATALOG_METADATA.json')
    require(manifest['manifest_hash'] == digest({k: v for k, v in manifest.items() if k != 'manifest_hash'}),
            'sampled catalog changed')
    require(metadata['metadata_hash'] == digest({k: v for k, v in metadata.items() if k != 'metadata_hash'}),
            'sampled catalog metadata changed')
    life = read_lifecycle_file(lifecycle)
    members = [dict(ts_code=code, **life[code]) for code in draw['codes']]
    plan = deepcopy(parent)
    plan['batch'] = 'FULL-LANDING-20261009-R13-COHORT'
    plan['cohort'] = dict(protocol['declaration'], members=members, posthoc_exclusion=False,
                          selection='original identifier-only full-population hash sample; all declared members retained')
    plan.pop('universe_restriction', None)
    plan['window'].update(start=protocol['window'][0], end=protocol['window'][1])
    plan.update(account_evidence_mode='source_bound_research', catalog_source_mode='verified_immutable_partitions',
                catalog_manifest_hash=manifest['manifest_hash'], catalog_file_sha256=file_hash(catalog / 'CATALOG.json'),
                catalog_metadata_hash=metadata['metadata_hash'], raw_sources={
                    name: manifest['sources'][api] for name, api in
                    [('daily_prices.csv', 'daily'), ('adj_factor_daily.csv', 'adj_factor'),
                     ('stk_limit.csv', 'stk_limit'), ('suspend_d.csv', 'suspend_d'), ('daily_basic.csv', 'daily_basic')]},
                cohort_input_binding={'protocol': _reference(protocol_path), 'draw': _reference(draw_path),
                                      'lifecycle': _reference(lifecycle)},
                parent_cohort_plan=_reference(parent_plan_path),
                scope='Original 64-member research-exposed input; retained missing cells, no historical PIT or investor assurance grant',
                formal_data_assured=False)
    if halted_distribution_policy is not None:
        require(halted_distribution_policy.get('mode') == 'declared_supported_distribution_reference' and
                halted_distribution_policy.get('rounding') in ('half_up', 'floor', 'ceil'), 'halt distribution policy invalid')
        plan['halted_distribution_policy'] = deepcopy(halted_distribution_policy)
    if issuer_action_bundle is not None:
        plan['issuer_action_evidence'] = _reference(issuer_action_bundle)
    if independent_event_directory is not None:
        from .action_evidence import load_independent_events
        independent_root = Path(independent_event_directory).resolve()
        quotes, events, sources = load_independent_events(independent_root)
        require(quotes and sources and set(sources) <= set(draw['codes']), 'independent source outside original sample')
        plan['independent_event_directory'] = str(independent_root)
        plan['independent_event_binding'] = {'protocol': _reference(independent_root / 'AMENDMENT.json'),
                                             'completion': _reference(independent_root / 'COMPLETE.json')}
    sessions, _ = read_calendar(catalog / 'trade_calendar.csv', *protocol['window'])
    verification = verify_declared_cohort(plan, catalog, sessions)
    code = {name: file_hash(Path(__file__).parent / name) for name in
            ('cohort_input.py', 'historical_adapter.py', 'pit_universe.py', 'market_catalog.py')}
    contract = {'schema_version': 'fuxi.sampled_input_contract.1', 'catalog_root': str(catalog),
                'plan_hash': digest(plan), 'code_hashes': code, 'original_members': verification['codes'],
                'external_requests': 0, 'redraws': 0, 'formal_data_assured': False}
    contract['contract_hash'] = digest(contract)
    write_new(root / 'HISTORY_PLAN.json', plan)
    write_new(root / 'COHORT_INPUT_CONTRACT.json', contract)
    write_new(root / 'RULE_REPRODUCTION.json', verification)
    return plan


def halted_distribution_reference(term, prior_mark, policy, last_real_quote, *, prior_factor=None, current_factor=None):
    """Conditional research ex-reference from prior information and issuer rights.

    This does not supply an observed price, ownership, cash, or a permission to
    trade. A new account still separately posts evidenced hypothetical rights.
    """
    from decimal import Decimal, ROUND_HALF_UP, ROUND_FLOOR, ROUND_CEILING
    from .a_share import rule_for
    require(policy.get('mode') == 'declared_supported_distribution_reference' and
            policy.get('rounding') in ('half_up', 'floor', 'ceil'), 'halt distribution policy invalid')
    ordinary = term.get('status') == 'ordinary_distribution_supported' and bool(term.get('issuer_evidence'))
    cash_only = (term.get('status') == 'cash_terms_supported' and term.get('share_distribution') == 0 and
                 policy.get('cash_tick_reference') == 'source_bound_cash_terms_tick_reference')
    if not ordinary and not cash_only:
        return None
    share, cash = term.get('share_distribution'), term.get('gross_cash_per_share')
    if not all(finite(v) for v in (share, cash, prior_mark)) or share < 0 or cash < 0 or prior_mark <= cash:
        return None
    issuer = term.get('issuer_evidence')
    if ordinary:
        require(share > 0 and issuer['holder_shares'] == share and issuer['holder_cash'] == cash and
                all(issuer[k] == term[k] for k in ('instrument', 'ex_date', 'record_date', 'ann_date', 'pay_date', 'share_credit_date')),
                'halted issuer rights differ from account terms')
        require(issuer.get('independent_assurance') is False and issuer.get('sources'), 'halted issuer source absent')
        require(term['ann_date'] <= term['record_date'] < term['ex_date'] <= term['share_credit_date'] and
                (not cash or term['ex_date'] <= term['pay_date']), 'halted distribution dates conflict')
    else:
        require(term.get('evidence_id') and all(term.get(k) for k in ('ann_date', 'record_date', 'ex_date', 'pay_date')) and
                term['ann_date'] <= term['record_date'] < term['ex_date'] <= term['pay_date'],
                'halted cash source terms absent or dates conflict')
    require(last_real_quote and last_real_quote['session'] < term['ex_date'] and
            finite(last_real_quote['close']) and last_real_quote['close'] > 0, 'halted reference prior observed quote absent')
    rule = rule_for(term['instrument'])
    tick = Decimal(rule['tick'])
    theoretical = (Decimal(str(prior_mark)) - Decimal(str(cash))) / (1 + Decimal(str(share)))
    units = theoretical / tick
    mode = {'half_up': ROUND_HALF_UP, 'floor': ROUND_FLOOR, 'ceil': ROUND_CEILING}[policy['rounding']]
    rounded = units.to_integral_value(rounding=mode) * tick
    floor, ceil = (units.to_integral_value(rounding=r) * tick for r in (ROUND_FLOOR, ROUND_CEILING))
    require(rounded > 0, 'halted distribution reference nonpositive')
    result = {'status': 'declared_supported_distribution_reference', 'reference_mark': float(rounded),
            'theoretical_price': str(theoretical), 'rounding': policy['rounding'],
            'rounding_is_declared_research_not_venue_certified': True,
            'rounding_sensitivity_prices': [str(floor), str(ceil)], 'quantity_multiplier': str(1 + Decimal(str(share))),
            'gross_cash_per_old_share': str(cash), 'previous_declared_mark': prior_mark,
            'last_real_quote': deepcopy(last_real_quote), 'quantity_rule': rule,
            'action_evidence_id': term['evidence_id'], 'issuer_evidence': deepcopy(issuer),
            'source_bound_action_terms': deepcopy(term),
            'rights_basis': 'issuer_primary_ordinary_distribution' if ordinary else 'frozen_vendor_supported_cash_terms_research',
            'observed_quote': False, 'fair_value_certified': False, 'trade_permission_granted': False,
            'scope': 'predeclared halted ex-distribution research price reference; rounding endpoints are arithmetic sensitivity only'}
    deferred = policy.get('deferred_factor_policy')
    if deferred is not None:
        require(deferred.get('mode') == 'resume_factor_from_declared_reference' and
                deferred.get('quantum') == '0.0001' and deferred.get('rounding') == 'half_up',
                'deferred factor research precision contract invalid')
        # A deferred adjustment is eligible only when it was not applied on the
        # action day. The later resume row corroborates this earlier model; it is
        # never used to backfill a price or publication clock on that action day.
        if all(finite(v) and v > 0 for v in (prior_factor, current_factor)) and prior_factor == current_factor:
            result['deferred_factor_anchor'] = {'prior_factor': prior_factor, 'ex_date_factor': current_factor,
                'pre_action_declared_mark': prior_mark, 'policy': deepcopy(deferred),
                'precision_is_declared_research_not_vendor_certified': True}
    return result


def resumed_distribution_reference(reference, prior_factor, current_factor, pre_close):
    """Corroborate a current resume row against the prior declared tick interval."""
    from decimal import Decimal, ROUND_HALF_UP
    if not (all(finite(v) and v > 0 for v in (prior_factor, current_factor, pre_close)) and
            Decimal(str(pre_close)) in {Decimal(v) for v in reference['rounding_sensitivity_prices']}):
        return False
    if prior_factor == current_factor:
        return True
    anchor = reference.get('deferred_factor_anchor')
    if not anchor or not (prior_factor == anchor['prior_factor'] == anchor['ex_date_factor']):
        return False
    policy = anchor['policy']
    require(policy['mode'] == 'resume_factor_from_declared_reference' and policy['quantum'] == '0.0001' and
            policy['rounding'] == 'half_up', 'deferred factor reference contract changed')
    expected = (Decimal(str(anchor['prior_factor'])) * Decimal(str(anchor['pre_action_declared_mark'])) /
                Decimal(str(pre_close))).quantize(Decimal(policy['quantum']), rounding=ROUND_HALF_UP)
    return expected == Decimal(str(current_factor))


def channel_audit(payload):
    """Report full-member denominators separately from selection and model support."""
    snap = payload['snapshot']
    n, m = len(snap['sessions']), len(snap['instruments'])
    require(n > 0 and m > 0, 'empty sampled input')
    fields = {field: {'expected_cells': n * m,
                       'observed_cells': sum(value is not None for row in matrix for value in row),
                       'missing_cells': sum(value is None for row in matrix for value in row)}
              for field, matrix in snap['columns'].items()}
    terms = payload['market']['action_terms']
    return {'schema_version': 'fuxi.sampled_channel_audit.1', 'sessions': n, 'members': m,
            'market_bar_cells': len(payload['market']['bars']), 'full_member_cells': n * m,
            'field_coverage': fields, 'label_rows': len(payload['labels']['rows']),
            'label_status': dict(Counter(row['status'] for row in payload['labels']['rows'])),
            'action_term_status': dict(Counter(row['status'] for row in terms)),
            'members_without_implemented_terms_in_window': sorted(set(snap['instruments']) - {t['instrument'] for t in terms}),
            'no_term_meaning': 'No implemented term in frozen source/window; not an independently certified absence of corporate actions',
            'market_event_coverage': dict(Counter(bar['event_coverage'] for bar in payload['market']['bars'])),
            'retained_unknowns': True, 'formal_data_assured': False}


def build(output, *, baseline_account=True):
    """Build and load actual market/label bytes, optionally replay a nonempty account."""
    root = Path(output)
    plan, contract = load_json(root / 'HISTORY_PLAN.json'), load_json(root / 'COHORT_INPUT_CONTRACT.json')
    require(contract['contract_hash'] == digest({k: v for k, v in contract.items() if k != 'contract_hash'}) and
            contract['plan_hash'] == digest(plan), 'sampled frozen contract changed')
    for name, expected in contract['code_hashes'].items():
        require(file_hash(Path(__file__).parent / name) == expected, 'sampled input code changed: ' + name)
    target = root / 'built' / 'input.json'
    if target.exists():
        done = load_json(root / 'BUILD_RECEIPT.json')
        require(file_hash(target) == done['input_sha256'], 'sampled input changed after build')
        return load_json(target)
    from .historical_adapter import build_history_input, HistoricalSnapshot
    from .data import Labels
    payload = build_history_input(contract['catalog_root'], plan, contract['catalog_root'], root / 'built')
    snapshot = HistoricalSnapshot(payload['snapshot'])
    labels = Labels(payload['labels'], snapshot)
    require(payload['snapshot']['instruments'] == [jq_code(code) for code in contract['original_members']],
            'built membership changed')
    require(labels.rows and payload['market']['bars'], 'sampled market/labels empty')
    write_new(target, payload)
    write_new(root / 'CHANNEL_AUDIT.json', channel_audit(payload))
    if baseline_account:
        from .search import node
        from .dsl import evaluate_ast
        from .portfolio import strategy_spec, compile_strategy
        from .simulation import run_account
        from .account_reference import reconcile
        scores = evaluate_ast(node('rank', node('return', {'field': 'close'}, window=20)), snapshot)
        spec = strategy_spec('cohort_input_return20', top_n=6, max_weight=1 / 6, rebalance_sessions=5)
        ir = compile_strategy(spec, scores, snapshot, digest(payload['market']['execution']))
        account = run_account(ir, payload['market'], start=plan['window']['research_start'], end=plan['window']['end'])
        reference = reconcile(account, payload['market'])
        require(len(account['daily']) > 1 and reference['n_pairs'] > 0, 'sampled baseline account empty')
        require(not reference['mismatches'], 'sampled reference account mismatched')
        write_new(root / 'BASELINE_IR.json', ir)
        write_new(root / 'BASELINE_ACCOUNT.json', account)
        write_new(root / 'BASELINE_REFERENCE.json', reference)
    receipt = {'schema_version': 'fuxi.sampled_input_build.1', 'input_path': str(target.resolve()),
               'input_sha256': file_hash(target), 'snapshot_hash': snapshot.hash,
               'market_hash': payload['market']['market_hash'], 'contract_hash': contract['contract_hash'],
               'members': len(snapshot.payload['instruments']), 'sessions': len(snapshot.sessions),
               'redraws': 0, 'external_requests': 0, 'labels_loaded': True, 'formal_data_assured': False}
    write_new(root / 'BUILD_RECEIPT.json', receipt)
    return payload
