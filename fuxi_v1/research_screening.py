"""Train-only direction and diagnostic ranking; no FS or percentile acceptance gate."""
from statistics import mean
from .contracts import timestamp, require
from .dsl import ranks
from .evaluation import correlation, factor_statistics, DEFAULT_POLICY
from .research_protocol import block_inference_fast


def training_reading(scores, snapshot, labels, cutoff, training_days):
    clock = timestamp(cutoff) if isinstance(cutoff, str) else cutoff
    sections, latest, pairs = [], [], 0
    for day in training_days:
        support = {s:(v,labels.value(day,s,clock)) for s,v in scores.get(day, {}).items()}
        support = {s:v for s,v in support.items() if None not in v and timestamp(labels.rows[(day,s)]['label_end']) < clock}
        if len(support) < 3:
            continue
        x = ranks({s:v[0] for s,v in support.items()})
        y = ranks({s:v[1] for s,v in support.items()})
        value = correlation(list(x.values()),list(y.values()))
        if value is not None:
            sections.append(value)
            pairs += len(support)
            latest += [labels.rows[(day,s)]['mature_at'] for s in support]
    avg = mean(sections) if sections else None
    return {'eval_status':'valid' if sections else 'data_insufficient', 'mean_rank_ic':avg,
            'direction':1 if avg is not None and avg >= 0 else -1 if avg is not None else None,
            'n_sections':len(sections), 'n_pairs':pairs,
            'latest_mature_at':max(latest,key=timestamp) if latest else None,
            'training_cutoff':clock.isoformat(), 'decision_use':'ranking_and_orientation_only_not_alpha_reward'}


def orient(scores, direction):
    require(direction in (-1,1), 'training direction unavailable')
    return {d:{s:None if v is None else direction*v for s,v in cross.items()} for d,cross in scores.items()}


def diagnose(scores, snapshot, labels, start, end, clock, *, direction, seed=1708):
    metric = factor_statistics({d:v for d,v in scores.items() if start <= d <= end}, snapshot, labels, DEFAULT_POLICY, clock)
    sections = metric['diagnostics']['sections']
    values = [None if r['rank_ic'] is None else direction*r['rank_ic'] for r in sections] if direction is not None else []
    interval = block_inference_fast(values,block_length=12,replicates=399,seed=seed) if values else {'status':'data_insufficient'}
    return {'metric':metric, 'training_direction':direction,
            'oriented_rank_ic':None if not any(v is not None for v in values) else mean(v for v in values if v is not None),
            'time_uncertainty':interval, 'ranking_only':True, 'hard_gate':None,
            'calibration_population_interpretation':'typed_search_population_not_statistical_null'}
