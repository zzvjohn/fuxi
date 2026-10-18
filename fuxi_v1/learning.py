"""Descriptive learning with frozen scope, explicit missingness and live invalidation.

Memory does not transfer statistical acceptance, economic rewards or JQ authority.
Cross-campaign imports recheck their source consumer and reward revisions on use.
"""
from collections import Counter
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
from statistics import mean
from .contracts import digest, require, load_json, load_fact
from .calibration import DEFAULT_BREAKEVEN_KEY, breakeven_for, economic_distance, percentile_of
from .evaluation import correlation
from .dsl import ranks


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value


def stage_feedback(result):
    rows = []
    for item in result['evaluations']:
        d = item.get('diagnostics', {})
        causes = []
        if item['eval_status'] != 'valid':
            causes.append(item['eval_status'])
        if item['stage'] == 'E1':
            n, need = d.get('valid_section_count'), d.get('power_required_sections')
            if n is not None and need is not None and n < need:
                causes.append('power_plan_sample_shortfall')
            ci = d.get('time_uncertainty') or {}
            if ci.get('lower') is not None and ci['lower'] <= 0:
                causes.append('direction_not_supported_by_interval')
            if d.get('tail_resolution_adequate') is False:
                causes.append('bootstrap_tail_resolution_insufficient')
        if item['decision'] == 'reject' and not causes:
            causes.append((item.get('reject_class') or 'unspecified')+'_reject')
        rows.append({'stage': item['stage'], 'status': item['eval_status'], 'decision': item['decision'],
                     'reason': item['reason_code'], 'causes': causes})
    return rows


def feedback_card(candidate, result, policy, scope, evaluation_id, state_id, *,
                  vehicle_ic_values=None, breakeven_key=DEFAULT_BREAKEVEN_KEY):
    stages = {r['stage']: r for r in result['evaluations']}
    e1, e2, e3 = (stages[k] for k in ('E1','E2','E3'))
    d = e1.get('diagnostics', {})
    # Missing or uncertified account data never becomes zero/negative alpha reward.
    economic = e3.get('economic_reward_eligible') is True and e3['eval_status'] == 'valid'
    gaps = Counter()
    examples = []
    for side in ('candidate_account','baseline_account'):
        for row in result.get(side, {}).get('gaps', []):
            gaps[(side,row['reason'])] += 1
            if len(examples) < 6:
                examples.append({'side':side, **row})
    mean_ic = d.get('spearman',{}).get('mean')
    anchor = breakeven_for(breakeven_key)
    vehicle = [v for v in (vehicle_ic_values or []) if _finite(v)]
    card = {'schema_version':'fuxi.feedback.2', 'candidate_id':candidate['candidate_id'],
            'candidate':candidate, 'evaluation_id':evaluation_id, 'state_id':state_id,
            'scope':scope, 'snapshot_hash':result['snapshot_hash'], 'policy_hash':policy['policy_hash'],
            'window':[policy['request']['research_start'],policy['request']['research_end']],
            'observations':stage_feedback(result), 'mean_rank_ic':mean_ic,
            'valid_sections':d.get('valid_section_count',0), 'n_pairs':e1['n_pairs'],
            'power_required_sections':d.get('power_required_sections'),
            'interval':d.get('time_uncertainty'),
            'baseline_similarity':{k:(e2.get('metrics') or {}).get(k) for k in ('mean_abs_correlation','n_pairs')},
            'net_increment':(e3.get('metrics') or {}).get('paired_mean_increment') if economic else None,
            'descriptive_net_increment':e3.get('diagnostics',{}).get('paired_mean_increment'),
            'account_evidence_status':e3['reason_code'],
            'descriptive_net_scope':'same-window declared-cost research scenario; no economic reward or qualification transfer',
            'economic_reward_eligible':economic,
            'gap_counts':[{'side':s,'reason':r,'count':n} for (s,r),n in sorted(gaps.items())],
            'gap_examples':examples, 'parent_comparisons':result.get('learning_parent_comparisons',[]),
            'interpretation':'research_exposed_descriptive_not_independent_confirmation',
            'qualification_transfer_allowed':False}
    # Additive M4 positioning fields: descriptive triage display only, granting nothing.
    card['economic_distance'] = economic_distance(mean_ic, anchor)
    card['vehicle_percentile'] = percentile_of(mean_ic, vehicle) if vehicle else None
    card['economic_distance_context'] = {'breakeven_key':breakeven_key, 'breakeven_ic':anchor,
                                         'vehicle_n':len(vehicle),
                                         'usage':'descriptive_triage_positioning_not_reward'}
    return card


def compatibility(payload, request):
    return digest({'snapshot':digest(payload['snapshot']), 'labels':digest(payload['labels']),
                   'execution':payload['market']['execution'],
                   'window':[request['research_start'],request['research_end']],
                   'baseline':'rank(return(close,20))', 'semantics':'typed_AST_raw_daily_v1'})


@contextmanager
def read_source(path):
    path = Path(path).resolve()
    require(path.is_file(), 'memory source missing')
    db = sqlite3.connect(path.as_uri()+'?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    try:
        yield db
    finally:
        db.close()


def source_usable(db, token):
    row = db.execute('SELECT state FROM consumers WHERE id=?',(token['state_id'],)).fetchone()
    if row is None or row['state'] != 'valid':
        return False
    for slot, revision in token['reward_revisions']:
        current = db.execute('SELECT MAX(revision) FROM rewards WHERE slot=?',(slot,)).fetchone()[0]
        if current != revision:
            return False
    return True


def export_memory(path, target_compatibility):
    """Read-only snapshot of rich feedback; old generic cards remain historical only."""
    with read_source(path) as db:
        config = db.execute('SELECT config FROM campaigns').fetchall()
        require(len(config) == 1, 'one campaign per source required')
        source_config = json.loads(config[0][0])
        source_key = source_config.get('learning_compatibility')
        legacy_policy = None
        if source_key is None:
            root=Path(path).resolve().parent
            legacy_input=load_json(root/'input.json');legacy_policy=load_json(root/'policy.json')
            require(digest(legacy_input)==source_config['input_hash'], 'legacy memory input changed')
            source_key=compatibility(legacy_input,legacy_policy['request'])
        require(source_key == target_compatibility,
                'memory data/window/label/cost semantics incompatible')
        records = []
        kind='experience' if legacy_policy else 'feedback_v2'
        candidates={}
        if legacy_policy:
            for r in db.execute("SELECT payload,hash FROM facts WHERE kind='candidate'"):
                c=load_fact(r['payload']);require(digest(c)==r['hash'],'legacy candidate corrupted')
                candidates[c['candidate_id']]=c
        for row in db.execute('SELECT id,payload,hash FROM facts WHERE kind=? ORDER BY rowid',(kind,)):
            card = load_fact(row['payload'])
            require(digest(card) == row['hash'], 'corrupted source memory')
            if legacy_policy:
                ev=db.execute('SELECT payload,hash FROM facts WHERE id=?',(card['evaluation_id'],)).fetchone()
                require(ev is not None and digest(load_fact(ev['payload']))==ev['hash'],'legacy evaluation missing/corrupted')
                card=feedback_card(candidates[card['candidate_id']],load_fact(ev['payload']),legacy_policy,
                                   card['scope'],card['evaluation_id'],card['state_id'])
            consumer = db.execute('SELECT dependencies FROM consumers WHERE id=?',(card['state_id'],)).fetchone()
            if consumer is None:
                continue
            dependencies = json.loads(consumer[0])
            rewards = []
            for dependency in dependencies:
                reward = db.execute('SELECT slot,revision FROM rewards WHERE event_id=?',(dependency,)).fetchone()
                if reward:
                    rewards.append([reward['slot'],reward['revision']])
            token = {'source_path':str(Path(path).resolve()),'source_event':row['id'],
                     'source_hash':row['hash'],'state_id':card['state_id'],'reward_revisions':rewards}
            if source_usable(db,token):
                records.append({'card':card,'token':token})
        return {'compatibility':target_compatibility,'records':records,
                'historical_decisions_and_rewards_imported':False}


def valid_imports(manifests):
    usable = []
    for manifest in manifests:
        for record in manifest['records']:
            token = record['token']
            with read_source(token['source_path']) as db:
                row = db.execute('SELECT payload,hash FROM facts WHERE id=?',(token['source_event'],)).fetchone()
                if row is None or row['hash'] != token['source_hash']:
                    continue
                require(digest(load_fact(row['payload'])) == row['hash'], 'source memory content changed')
                if source_usable(db,token):
                    usable.append(record['card'])
    return usable


def refresh_import_dependencies(store, manifests):
    valid = valid_imports(manifests)
    for manifest in manifests:
        for record in manifest['records']:
            state = 'import_state_'+digest(record['token'])
            if not store.db.execute('SELECT 1 FROM consumers WHERE id=?',(state,)).fetchone():
                store.register_consumer(state,[],record['token']['source_path'])
            # The same formula/candidate in another source cannot mask this source's revocation.
            active = bool(valid_imports([{'records':[record]}]))
            if not active and store.state_usable(state):
                store.invalidate(state)
                store.db.execute("UPDATE consumers SET state='invalidated' WHERE id=?",(state,))
    return valid


def feedback_dependencies(candidate, local_cards, manifests):
    parents=set(candidate['parent_ids'])
    cited=set(candidate.get('source_evidence_ids',[]))
    result=[]
    for card in local_cards:
        if card['candidate_id'] in parents or card['evaluation_id'] in cited:
            result.append(card['state_id'])
    for manifest in manifests:
        for record in manifest['records']:
            card=record['card']
            if card['candidate_id'] in parents or card['evaluation_id'] in cited:
                result.append('import_state_'+digest(record['token']))
    return sorted(set(result))


def prompt_cards(cards, limit=4):
    # Include failures and comparators, not just the highest observed IC.
    ordered = sorted(cards,key=lambda c:(-len(c['parent_comparisons']),-len(c['gap_counts']),c['candidate_id']))
    result = []
    for card in ordered[:limit]:
        compact = {k:card[k] for k in ('candidate_id','window','mean_rank_ic','valid_sections',
                   'power_required_sections','net_increment','economic_reward_eligible','observations',
                   'baseline_similarity','parent_comparisons','interpretation')}
        compact['ast'] = card['candidate']['canonical_ast']
        compact['gap_counts'] = card['gap_counts'][:4]
        result.append({'evidence_id':card['evaluation_id'],'text':json.dumps(compact,ensure_ascii=False)})
    return result


def paired_parent(child_scores, parent_scores, snapshot, labels, start, end, clock):
    """Both ICs use exactly the same security/day/label support; no unmatched averages."""
    if snapshot.payload.get("schema_version") == "fuxi.columnar_snapshot.1":
        from .columnar import paired_parent_columnar
        return paired_parent_columnar(child_scores, parent_scores, snapshot, labels, start, end, clock)
    from .contracts import timestamp
    if isinstance(clock, str):
        clock = timestamp(clock)
    rows=[]
    for day in snapshot.sessions:
        if not start <= day <= end:
            continue
        common = {s:(child_scores[day][s],parent_scores[day][s],labels.value(day,s,clock))
                  for s in snapshot.universe[day]}
        common = {s:v for s,v in common.items() if None not in v}
        if len(common) < 5:
            continue
        ranked = [list(ranks({s:v[i] for s,v in common.items()}).values()) for i in range(3)]
        a,b = correlation(ranked[0],ranked[2]),correlation(ranked[1],ranked[2])
        if a is not None and b is not None:
            rows.append({'session':day,'n_pairs':len(common),'child_ic':a,'parent_ic':b,'delta':a-b,
                         'signal_rank_correlation':correlation(ranked[0],ranked[1])})
    return {'status':'valid' if rows else 'data_insufficient','n_sections':len(rows),
            'n_pairs':sum(r['n_pairs'] for r in rows),
            'mean_ic_increment':mean(r['delta'] for r in rows) if rows else None,
            'mean_signal_rank_correlation':mean(r['signal_rank_correlation'] for r in rows) if rows else None,
            'support_hash':digest(rows),'scope':'descriptive_common_support_not_economic_reward'}


def behavioral_signature(scores, start, end):
    # Exact daily ordinal signal behavior; renamed/rank-wrapped formula duplicates collapse.
    from .columnar import ScoreGrid
    if isinstance(scores, ScoreGrid):
        from .full_market_support import ordinal_identity
        return ordinal_identity(scores, start, end)
    return digest({d:ranks(v) for d,v in scores.items() if start <= d <= end})


def structural_family(ast):
    # Window variants count as one structure; this is not proof of a new economic mechanism.
    return digest({k:([structural_family(a) for a in v] if k=='args' else v)
                   for k,v in ast.items() if k != 'window'})


def checkpoint(history, results, cards, every, previous=None):
    completed = len(history)
    recent = [r for r in results if completed-every <= r['round'] < completed]
    hashes = [r['behavior_hash'] for r in results if r.get('behavior_hash')]
    old = {r.get('behavior_hash') for r in results if r['round'] < completed-every}
    new_behaviors = len({r.get('behavior_hash') for r in recent if r.get('behavior_hash')} - old)
    ids = {r['candidate_id'] for r in recent}
    comparisons = [p for c in cards if c['candidate_id'] in ids for p in c['parent_comparisons']]
    valid = [p for p in comparisons if p['status']=='valid']
    blocked = Counter(b for r in recent for b in r.get('blockers',[]))
    # Resource decisions only. No threshold or research/confirmation window changes.
    errors = sum(r['eval_status']=='error' for r in recent)
    sterile = new_behaviors == 0 and not any(p['mean_ic_increment'] > 0 for p in valid)
    stop = errors > 0 or (sterile and previous is not None and previous['sterile'])
    return {'rounds_completed':completed,'candidate_count':len(results),'recent_candidates':len(recent),
            'new_behaviors':new_behaviors,'unique_behaviors':len(set(hashes)),
            'unique_structural_families':len({r['structural_family'] for r in results if r.get('structural_family')}),
            'behavior_duplicate_fraction':1-len(set(hashes))/len(hashes) if hashes else None,
            'parent_comparisons':len(valid),'positive_parent_comparisons':sum(p['mean_ic_increment']>0 for p in valid),
            'formal_account_valid':sum(r['formal_account_valid'] for r in results),
            'blockers':dict(blocked),'sterile':sterile,'stop':stop,
            'decision':'stop_engineering_error' if errors else 'stop_two_sterile_blocks' if stop else 'continue_within_frozen_budget',
            'economic_claim_allowed':False}


def _motif_path(path):
    return "/".join(str(i) for i in path)


def ast_diff_motifs(parent_ast, child_ast, path=()):
    """Edit-level motifs of one AST mutation (process memory, description only).

    A rewritten node is one edit; its new descendants are new code, not another
    edit, so the walk stops there instead of reporting every introduced leaf.
    """
    if parent_ast == child_ast:
        return []
    if not (isinstance(parent_ast, dict) and isinstance(child_ast, dict)):
        return ["replace_node@" + _motif_path(path)]
    motifs = []
    prefix = _motif_path(path)
    parent_op, child_op = parent_ast.get("op"), child_ast.get("op")
    if parent_op != child_op:
        args = child_ast.get("args") or []
        if len(args) == 1 and args[0] == parent_ast:
            motifs.append("wrap:%s@%s" % (child_op, prefix))
        elif child_op in ("add", "sub", "div", "mul"):
            motifs.append("binop:%s@%s" % (child_op, prefix))
        else:
            motifs.append("replace_op@%s:%s->%s" % (prefix, parent_op, child_op))
        return motifs
    if "window" in parent_ast or "window" in child_ast:
        before, after = parent_ast.get("window"), child_ast.get("window")
        if before != after:
            motifs.append("swap_window@%s:%s->%s" % (prefix, before, after))
    if "field" in parent_ast or "field" in child_ast:
        before, after = parent_ast.get("field"), child_ast.get("field")
        if before != after:
            motifs.append("swap_field@%s:%s->%s" % (prefix, before, after))
    parent_args, child_args = parent_ast.get("args") or [], child_ast.get("args") or []
    for index in range(min(len(parent_args), len(child_args))):
        motifs.extend(ast_diff_motifs(parent_args[index], child_args[index], path + (index,)))
    if len(parent_args) != len(child_args):
        motifs.append("arity@%s" % prefix)
    return motifs


def _wilson_upper(success, n, z=1.959963984540054):
    p = success / n
    center = p + z * z / (2 * n)
    margin = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5
    return (center + margin) / (1 + z * z / n)


def motif_ledger(observations):
    """Aggregate edit outcomes: {motif: {n, success, success_rate, upper95}}."""
    totals = {}
    for motif, success in observations:
        row = totals.setdefault(motif, [0, 0])
        row[0] += 1
        row[1] += 1 if success else 0
    return {m: {"n": totals[m][0], "success": totals[m][1], "success_rate": totals[m][1] / totals[m][0],
                "upper95": _wilson_upper(totals[m][1], totals[m][0])} for m in sorted(totals)}


def motif_vetoes(ledger, *, min_n=5, max_success_upper=0.45):
    """Asymmetric gate: veto only high-confidence failing edits; never promotes a candidate."""
    require(type(min_n) is int and min_n >= 1, "positive minimum observation count required")
    return sorted(m for m, row in ledger.items()
                  if row["n"] >= min_n and row["upper95"] < max_success_upper)


def motif_observations(cards, ast_of):
    """(motif, success) pairs from feedback cards; success = positive paired IC increment."""
    observations = []
    for card in cards:
        child_ast = (card.get("candidate") or {}).get("canonical_ast")
        if not child_ast:
            continue
        for comparison in card.get("parent_comparisons") or []:
            if comparison.get("status") != "valid" or comparison.get("mean_ic_increment") is None:
                continue
            parent_ast = ast_of(comparison.get("parent_id"))
            if not parent_ast:
                continue
            for motif in ast_diff_motifs(parent_ast, child_ast):
                observations.append((motif, comparison["mean_ic_increment"] > 0))
    return observations


def motif_gate(proposals, ast_of, vetoes, *, mode):
    """Check parent-derived proposals against the veto set; shadow counts, enforce filters."""
    require(mode in {"shadow", "enforce"}, "motif gate mode required")
    veto_set, checked, hits, kept = set(vetoes), 0, [], []
    for proposal in proposals:
        parent_asts = [a for a in (ast_of(pid) for pid in proposal.get("parent_ids") or []) if a]
        if not parent_asts:
            kept.append(proposal)
            continue
        checked += 1
        matched = sorted({m for a in parent_asts for m in ast_diff_motifs(a, proposal["canonical_ast"])
                          if m in veto_set})
        if matched:
            hits.append({"candidate_id": proposal["candidate_id"], "motifs": matched})
            if mode == "enforce":
                continue
        kept.append(proposal)
    return {"checked": checked, "hit_count": len(hits), "hits": hits, "kept": kept}


def feedback_prompt_key(card):
    """Existing prompt ordering (failures and comparators first); triage baseline for display."""
    return (-len(card.get('parent_comparisons') or ()), -len(card.get('gap_counts') or ()),
            str(card.get('candidate_id')))


def fitness_key(card):
    """Deterministic triage ordering; ties break on candidate_id.

    Economic distance first; legacy cards without the field fall back to the raw mean rank
    IC; cards without any reading rank last. Display priority only: grants nothing.
    """
    distance, ic = card.get("economic_distance"), card.get("mean_rank_ic")
    if _finite(distance):
        return (0, -float(distance), str(card.get("candidate_id")))
    if _finite(ic):
        return (1, -float(ic), str(card.get("candidate_id")))
    return (2, 0.0, str(card.get("candidate_id")))


def fitness_triage(cards, *, limit=4):
    """Baseline-vs-triage display comparison with a reproducible digest; grants nothing."""
    require(type(limit) is int and limit >= 1, "positive triage display limit required")
    baseline = sorted(cards, key=feedback_prompt_key)
    triage = sorted(cards, key=fitness_key)
    rows = [{"fit_rank": index, "candidate_id": card.get("candidate_id"),
             "mean_rank_ic": card.get("mean_rank_ic"),
             "economic_distance": card.get("economic_distance"),
             "vehicle_percentile": card.get("vehicle_percentile")}
            for index, card in enumerate(triage)]
    top_baseline = [card.get("candidate_id") for card in baseline[:limit]]
    top_triage = [card.get("candidate_id") for card in triage[:limit]]
    return {"limit": limit, "n_cards": len(cards),
            "top_k_baseline": top_baseline, "top_k_triage": top_triage,
            "entered_top_k": [c for c in top_triage if c not in set(top_baseline)],
            "left_top_k": [c for c in top_baseline if c not in set(top_triage)],
            "rows": rows, "triage_digest": digest(rows)}
