"""Selection-based resource scheduling. Evidence rewards remain in the authority ledger."""
import math
from .contracts import require


def counters(history, arm):
    runs = [r for r in history if r['arm'] == arm]
    return {name: sum(r.get(name, fallback(r)) for r in runs) for name, fallback in {
        'selection_count': lambda r: 1,
        'generation_attempts': lambda r: r.get('generated_count', 0),
        'unique_generated_count': lambda r: r.get('generated_count', 0),
        'evaluated_count': lambda r: 0,
        'valid_e1_count': lambda r: 0,
    }.items()}


def choose(round_index, mode, history, llm_enabled, *, context_hash, empty_limit=2):
    require(mode in {'adaptive', 'fixed', 'no_memory'} and empty_limit >= 1, 'scheduler policy invalid')
    arms = ['fixed', 'random', 'refine', 'combine', 'gflownet'] + (['llm'] if llm_enabled else [])
    if mode == 'fixed':
        arms = ['fixed']
    elif mode == 'no_memory':
        arms = ['random']
    active, inactive, scores, economic = [], {}, {}, {}
    totals = {arm: counters(history, arm) for arm in arms}
    for arm in arms:
        runs = [r for r in history if r['arm'] == arm]
        same = [r for r in runs if r.get('generation_context_hash') == context_hash]
        empty_streak = 0
        for row in reversed(same):
            if row.get('unique_generated_count', row.get('generated_count', 0)):
                break
            empty_streak += 1
        if empty_streak >= empty_limit:
            inactive[arm] = 'same_context_novelty_exhausted'
            continue
        # API failure is a resource failure, never evidence that a financial hypothesis failed.
        if arm == 'llm' and runs and runs[-1].get('generator_status') in {'error', 'temporarily_unavailable'}:
            inactive[arm] = 'provider_failure_no_automatic_retry'
            continue
        active.append(arm)
        visits = totals[arm]['selection_count']
        cost = sum(r['elapsed_seconds'] for r in runs)
        scores[arm] = totals[arm]['valid_e1_count']/max(1., cost) + math.sqrt(2*math.log(round_index+2)/max(1, visits))
        count = sum(r.get('economic_reward_count', 0) for r in runs)
        economic[arm] = sum(r.get('economic_reward_sum', 0) for r in runs)/count if count else None
    detail = {'rule': 'selection_v2', 'scores': scores, 'economic_means': economic,
              'counters': totals, 'inactive': inactive, 'context_hash': context_hash}
    if not active:
        return None, dict(detail, stop_reason='all_arms_exhausted')
    unvisited = [a for a in active if totals[a]['selection_count'] == 0]
    if unvisited:
        return unvisited[0], dict(detail, phase='initial_selection_exploration')
    admitted = [a for a in active if economic[a] is not None]
    eligible = admitted if admitted and round_index % 3 else active
    selected = max(eligible, key=lambda a: (economic[a] if eligible is admitted else scores[a], scores[a], -arms.index(a)))
    return selected, detail
