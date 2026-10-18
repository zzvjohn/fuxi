"""Hash-bound full-cross-section arrays for explicitly declared large research inputs.

This adapter does not fetch data, fill missing observations, grant assurance, or
split cross-sectional operators by instrument.  Legacy JSON adapters remain
separate.  NaN is a storage null, never an observed zero.
"""

from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from datetime import date
import hashlib
import math
from pathlib import Path
from statistics import mean, stdev
import numpy as np

from .contracts import digest, require, timestamp

SNAPSHOT_SCHEMA = 'fuxi.columnar_snapshot.1'
LABEL_SCHEMA = 'fuxi.columnar_labels.1'
MARKET_SCHEMA = 'fuxi.columnar_market.1'
SCORE_SCHEMA = 'fuxi.score_grid_ref.1'


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read_array(reference, shape=None, kinds=None):
    require(isinstance(reference, dict) and set(reference) == {'path', 'sha256', 'dtype', 'shape'},
            'columnar array descriptor keys differ')
    path = Path(reference['path'])
    require(path.is_absolute() and path.is_file() and not path.is_symlink(), 'array file missing or unsafe')
    require(_sha(path) == reference['sha256'], 'columnar array source changed: ' + str(path))
    array = np.load(path, mmap_mode='r', allow_pickle=False)
    require(isinstance(array, np.ndarray) and not array.dtype.hasobject, 'object arrays forbidden')
    require(list(array.shape) == reference['shape'] and array.dtype.str == reference['dtype'],
            'array descriptor shape/dtype mismatch')
    require(shape is None or tuple(array.shape) == tuple(shape), 'columnar array shape mismatch')
    require(kinds is None or array.dtype.kind in kinds, 'columnar array dtype invalid')
    if array.dtype.kind == 'f':
        for lo in range(0, len(array), 64):
            require(not bool(np.isinf(array[lo:lo + 64]).any()), 'infinite columnar observation')
    return array


def array_reference(path, array=None):
    path = Path(path).resolve()
    value = array if array is not None else np.load(path, mmap_mode='r', allow_pickle=False)
    return {'path': str(path), 'sha256': _sha(path), 'dtype': value.dtype.str,
            'shape': list(value.shape)}


def close_arrays(*arrays):
    """Explicitly release Windows memory-map file handles after the consumer ends."""
    for array in arrays:
        mapping=getattr(array,'_mmap',None)
        if mapping is not None and not mapping.closed:mapping.close()


def _calendar(payload):
    sessions = payload['sessions']
    instruments = payload['instruments']
    require(isinstance(sessions, list) and sessions and sessions == sorted(set(sessions)), 'invalid columnar calendar')
    require(all(date.fromisoformat(d).isoformat() == d for d in sessions), 'invalid session date')
    require(isinstance(instruments, list) and instruments and len(set(instruments)) == len(instruments)
            and all(isinstance(s, str) and s.strip() for s in instruments), 'invalid instrument identities')
    return sessions, instruments, (len(sessions), len(instruments))


class Universe(Mapping):
    def __init__(self, sessions, instruments, membership):
        self.sessions, self.instruments, self.membership = sessions, instruments, membership
        self.indices = {d: i for i, d in enumerate(sessions)}
        self.cache = OrderedDict()

    def __len__(self): return len(self.sessions)
    def __iter__(self): return iter(self.sessions)

    def __getitem__(self, day):
        index = self.indices[day]
        if day not in self.cache:
            self.cache[day] = [self.instruments[j] for j in np.flatnonzero(self.membership[index])]
            if len(self.cache) > 96: self.cache.popitem(last=False)
        return self.cache[day]


class ColumnarSnapshot:
    def __init__(self, payload):
        require(payload.get('schema_version') == SNAPSHOT_SCHEMA, 'columnar snapshot schema mismatch')
        require(payload['exposure'] in {'synthetic', 'research_exposed'}, 'columnar research cannot read protected observations')
        self.payload = payload
        self.sessions, self.instruments, shape = _calendar(payload)
        require(payload['evaluation_start'] in self.sessions, 'evaluation start outside calendar')
        self.start = self.sessions.index(payload['evaluation_start'])
        require(set(payload['decision_times']) == set(self.sessions), 'decision clocks/calendar mismatch')
        self.clocks = {d: timestamp(payload['decision_times'][d]) for d in self.sessions}
        require(all(self.clocks[d].date().isoformat() == d for d in self.sessions), 'decision day mismatch')
        self._clock_list = list(self.clocks.values())
        require(self._clock_list == sorted(self._clock_list), 'decision clock ordering')
        self._clock_indices = {v: i for i, v in enumerate(self._clock_list)}
        self._codes = {s: j for j, s in enumerate(self.instruments)}
        from .dsl import compile_ast
        require(set(payload['arrays']['fields']) == set(payload['fields']), 'columnar field catalog mismatch')
        for field, spec in payload['fields'].items():
            require(set(spec)=={'unit','adjustment','frequency'} and spec['frequency']=='daily'
                    and spec['adjustment'] in {'raw','not_applicable'}, 'invalid columnar field contract')
            compile_ast({'field': field}, payload['fields'])
        self.columns = {f: read_array(r, shape, 'f') for f, r in payload['arrays']['fields'].items()}
        self.barriers = read_array(payload['arrays']['barriers'], shape, 'iu')
        for i, row in enumerate(self.barriers):
            require(bool(((row >= -1) & (row <= i)).all()), 'invalid action lookback barriers')
        require(not bool((self.barriers[1:] < self.barriers[:-1]).any()), 'action barrier regressed')
        self.membership = read_array(payload['arrays']['membership'], shape, 'b')
        require(bool(self.membership.any(axis=1).all()), 'empty research cross-section')
        self.universe = Universe(self.sessions, self.instruments, self.membership)
        declared = payload.get('availability_times')
        require(isinstance(declared, dict) and set(declared) == set(self.columns), 'field availability declaration required')
        self.availability = {}
        for field, rows in declared.items():
            require(len(rows) == len(self.sessions), 'availability/calendar shape mismatch')
            clocks = [timestamp(v) for v in rows]
            require(all(v.date().isoformat() >= d for d, v in zip(self.sessions, clocks)), 'availability precedes observation')
            self.availability[field] = clocks
        self.hash = digest(payload)
        self._score_cache = OrderedDict()

    def value(self, index, instrument, field, asof):
        if index < 0 or index >= len(self.sessions) or instrument not in self._codes: return None
        current = self._clock_indices.get(asof)
        if current is None: current = bisect_right(self._clock_list, asof) - 1
        j = self._codes[instrument]
        if current < index or index < self.barriers[current, j] or self.availability[field][index] > asof: return None
        value = float(self.columns[field][index, j])
        return value if math.isfinite(value) else None

    def values_at(self, current, indices, field):
        """Whole instrument axis, with each historic observation viewed at this decision."""
        indices = np.asarray(indices, dtype=np.int64)
        result = np.full((len(indices), len(self.instruments)), np.nan)
        clock = self._clock_list[current]
        for k, index in enumerate(indices):
            if 0 <= index <= current and self.availability[field][index] <= clock:
                result[k] = self.columns[field][index]
                result[k, index < self.barriers[current]] = np.nan
        return result

    def close(self):
        self._score_cache.clear();close_arrays(*self.columns.values(),self.membership,self.barriers)


class ScoreRow(Mapping):
    def __init__(self, grid, index): self.grid, self.index = grid, index
    def __len__(self): return int(self.grid.membership[self.index].sum())
    def __iter__(self):
        return (self.grid.instruments[j] for j in np.flatnonzero(self.grid.membership[self.index]))
    def __getitem__(self, code):
        j = self.grid.code_indices[code]
        if not self.grid.membership[self.index, j]: raise KeyError(code)
        value = float(self.grid.matrix[self.index, j])
        return value if math.isfinite(value) else None


class ScoreGrid(Mapping):
    def __init__(self, matrix, sessions, instruments, membership, start=0, end=None):
        require(tuple(matrix.shape) == (len(sessions), len(instruments)) == tuple(membership.shape), 'score grid shape mismatch')
        self.matrix, self.sessions, self.instruments, self.membership = matrix, sessions, instruments, membership
        self.start, self.end = start, len(sessions) if end is None else end
        require(0 <= self.start <= self.end <= len(sessions), 'score grid scope mismatch')
        self.day_indices = {d: i for i, d in enumerate(sessions)}
        self.code_indices = {s: j for j, s in enumerate(instruments)}
    def __len__(self): return self.end - self.start
    def __iter__(self): return iter(self.sessions[self.start:self.end])
    def __getitem__(self, day):
        i = self.day_indices[day]
        if not self.start <= i < self.end: raise KeyError(day)
        return ScoreRow(self, i)
    def subset(self, start, end):
        lo = max(self.start, bisect_right(self.sessions, start) - (start in self.day_indices))
        hi = min(self.end, bisect_right(self.sessions, end))
        return ScoreGrid(self.matrix, self.sessions, self.instruments, self.membership, lo, max(lo, hi))
    def close(self):close_arrays(self.matrix,self.membership)


def selected_by_dates(scores, start, end):
    return scores.subset(start, end) if isinstance(scores, ScoreGrid) else {d: v for d, v in scores.items() if start <= d <= end}


def rank_vector(values):
    result = np.full(len(values), np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    n = len(valid)
    if n < 2: return result
    order = valid[np.argsort(values[valid], kind='stable')]
    ordered = values[order]
    boundaries = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1, n]
    averaged=((boundaries[:-1]+1+boundaries[1:])/2-.5)/n
    result[order]=np.repeat(averaged,np.diff(boundaries))
    return result


def window_reduction(window, op):
    """Complete-window float64 reductions, with scale protection for sample sd."""
    valid=np.isfinite(window).all(axis=0);result=np.full(window.shape[1],np.nan)
    if not bool(valid.any()):return result
    values=window[:,valid]
    if op=='mean':
        # Terms are divided before addition, as in the scalar fsum interpreter.
        result[valid]=np.sum(values/window.shape[0],axis=0,dtype=np.float64)
    elif op=='std':
        # Center on an observed value first, preserving tiny variations around a
        # large level.  Normalize those differences rather than the original
        # level.  Opposite-sign extremes can overflow subtraction; handle only
        # those columns by scaling the original finite values first.
        with np.errstate(all='ignore'):
            differences=values-values[0]
            overflow=~np.isfinite(differences).all(axis=0)
            scale=np.max(np.abs(differences),axis=0)
            if bool(overflow.any()):
                scale[overflow]=np.max(np.abs(values[:,overflow]),axis=0)
                differences[:,overflow]=values[:,overflow]/scale[overflow]
            safe=np.where(scale>0,scale,1.)
            normalized=differences/safe
            if bool(overflow.any()):normalized[:,overflow]=differences[:,overflow]
            center=normalized.mean(axis=0)
            result[valid]=np.sqrt(np.sum((normalized-center)**2,axis=0)/(window.shape[0]-1))*scale
    elif op=='min':result[valid]=np.min(values,axis=0)
    elif op=='max':result[valid]=np.max(values,axis=0)
    else:raise ValueError('unknown window reduction')
    result[~np.isfinite(result)]=np.nan
    return result


def evaluate_columnar_ast(ast, snapshot):
    from .dsl import compile_ast, _score_cache_key, _score_stats
    compile_ast(ast, snapshot.payload['fields'])
    stats = _score_stats(snapshot, snapshot._score_cache)
    # Columnar identity for the nine-component key: instruments as the universe.
    key = _score_cache_key(ast, snapshot, universe=snapshot.instruments)
    if key in snapshot._score_cache:
        stats["hits"] += 1
        return snapshot._score_cache[key]
    shape = snapshot.membership.shape
    def visit(node):
        if 'field' in node:
            field = node['field']; result = np.array(snapshot.columns[field], copy=True)
            for i in range(len(snapshot.sessions)):
                if snapshot.availability[field][i] > snapshot._clock_list[i]: result[i] = np.nan
            result[~snapshot.membership] = np.nan
            return result
        op, args = node['op'], node['args']
        if op in {'lag', 'return', 'delta', 'mean', 'std', 'min', 'max'}:
            result = np.full(shape, np.nan); field = args[0]['field']; w = node['window']
            for i in range(snapshot.start, len(snapshot.sessions)):
                if op in {'lag', 'return', 'delta'}:
                    previous, current = snapshot.values_at(i, (i-w, i), field)
                    with np.errstate(all='ignore'):
                        value = previous if op == 'lag' else current - previous if op == 'delta' else current / previous - 1
                    if op == 'return': value[previous == 0] = np.nan
                else:
                    window = snapshot.values_at(i, range(i-w+1, i+1), field)
                    value = window_reduction(window,op)
                value[~np.isfinite(value) | ~snapshot.membership[i]] = np.nan
                result[i] = value
            return result
        left = visit(args[0])
        if op in {'rank', 'zscore'}:
            result = np.full(shape, np.nan)
            for i in range(snapshot.start, len(snapshot.sessions)):
                values = left[i]
                if op == 'rank': result[i] = rank_vector(values)
                else:
                    valid = np.isfinite(values); items = values[valid].tolist()
                    center = mean(items) if items else 0
                    scale = stdev(items) if len(items) >= 2 else 0
                    if scale > 0: result[i, valid] = (values[valid] - center) / scale
            return result
        with np.errstate(all='ignore'):
            if op == 'neg': result = -left
            elif op == 'abs': result = np.abs(left)
            elif op == 'square': result = left * left
            elif op == 'log':
                result = np.full(shape, np.nan)
                # Python math.log preserves the scalar interpreter's value ordering.
                for i in range(snapshot.start, len(snapshot.sessions)):
                    for j in np.flatnonzero(np.isfinite(left[i]) & (left[i] > 0)):
                        result[i, j] = math.log(float(left[i, j]))
            else:
                right = visit(args[1])
                if op == 'add': result = left + right
                elif op == 'sub': result = left - right
                elif op == 'mul': result = left * right
                else:
                    result = left / right; result[right == 0] = np.nan
        result[~np.isfinite(result)] = np.nan
        return result
    matrix = visit(ast)
    scores = ScoreGrid(matrix, snapshot.sessions, snapshot.instruments, snapshot.membership, snapshot.start)
    stats["computes"] += 1
    snapshot._score_cache[key] = scores
    if len(snapshot._score_cache) > 4: snapshot._score_cache.popitem(last=False)
    return scores


def _same_grids(left, right):
    require(isinstance(left, ScoreGrid) and isinstance(right, ScoreGrid), 'columnar score grids required')
    require(left.sessions == right.sessions and left.instruments == right.instruments
            and left.start == right.start and left.end == right.end
            and bool(np.array_equal(left.membership, right.membership)), 'paired score scope mismatch')


def mix_scores_columnar(primary, added, added_weight=.5):
    _same_grids(primary, added)
    require(0 <= added_weight <= 1, 'invalid score mix weight')
    matrix = np.full(primary.matrix.shape, np.nan)
    for i in range(primary.start, primary.end):
        a, b = rank_vector(primary.matrix[i]), rank_vector(added.matrix[i])
        matrix[i] = (1-added_weight)*a + added_weight*b
    return ScoreGrid(matrix, primary.sessions, primary.instruments, primary.membership, primary.start, primary.end)


def common_support_grid(primary, added):
    _same_grids(primary, added)
    matrix = np.array(primary.matrix, copy=True)
    matrix[~np.isfinite(added.matrix)] = np.nan
    return ScoreGrid(matrix, primary.sessions, primary.instruments, primary.membership, primary.start, primary.end)


mix_columnar_scores = mix_scores_columnar


def score_grid_view(scores):
    """A date-selected dict of ScoreRows can retain its backing matrix."""
    if isinstance(scores,ScoreGrid):return scores
    require(isinstance(scores,Mapping) and scores,'nonempty columnar score mapping required')
    rows=list(scores.values());require(all(isinstance(r,ScoreRow) for r in rows),'columnar score rows required')
    grid=rows[0].grid;require(all(r.grid is grid for r in rows),'mixed score matrices')
    days=sorted(scores);indices=[grid.day_indices[d] for d in days]
    require(indices==list(range(indices[0],indices[-1]+1)),'noncontiguous columnar score selection')
    return ScoreGrid(grid.matrix,grid.sessions,grid.instruments,grid.membership,indices[0],indices[-1]+1)


class LabelRow(Mapping):
    _keys = ('session', 'instrument', 'decision_time', 'first_eligible_execution_time', 'label_start',
             'label_end', 'mature_at', 'value', 'status', 'evidence_id')
    def __init__(self, labels, i, j): self.labels, self.i, self.j = labels, i, j
    def __iter__(self): return iter(self._keys)
    def __len__(self): return len(self._keys)
    def __getitem__(self, key):
        p, i, j = self.labels.payload, self.i, self.j
        days = self.labels.snapshot.sessions; day = days[i]; code = self.labels.snapshot.instruments[j]
        if key == 'session': return day
        if key == 'instrument': return code
        if key == 'decision_time': return self.labels.snapshot.payload['decision_times'][day]
        if key in {'first_eligible_execution_time', 'label_start'}: return days[i+1] + 'T09:30:00+08:00'
        if key == 'label_end': return days[i+1+self.labels.horizon] + 'T15:00:00+08:00'
        if key == 'mature_at': return p['mature_at'][i]
        if key == 'value':
            value = float(self.labels.values[i, j]); return value if math.isfinite(value) else None
        if key == 'status': return self.labels.status_names[int(self.labels.status[i, j])]
        if key == 'evidence_id': return digest([p['evidence_namespace'], day, code])
        raise KeyError(key)


class LabelRows(Mapping):
    def __init__(self, labels): self.labels = labels
    def __len__(self): return int(np.count_nonzero(self.labels.status))
    def __iter__(self):
        for i in range(len(self.labels.snapshot.sessions)):
            for j in np.flatnonzero(self.labels.status[i]):
                yield self.labels.snapshot.sessions[i], self.labels.snapshot.instruments[j]
    def __getitem__(self, key):
        day, code = key
        i, j = self.labels.day_indices[day], self.labels.snapshot._codes[code]
        if not self.labels.status[i, j]: raise KeyError(key)
        return LabelRow(self.labels, i, j)


class ColumnarLabels:
    def __init__(self, payload, snapshot):
        require(payload.get('schema_version') == LABEL_SCHEMA, 'columnar labels schema mismatch')
        require(payload['feature_snapshot_hash'] == snapshot.hash, 'columnar label lineage mismatch')
        self.payload, self.snapshot = payload, snapshot
        sessions, instruments, shape = _calendar(payload)
        require(sessions == snapshot.sessions and instruments == snapshot.instruments, 'label axes mismatch')
        self.day_indices = {d: i for i, d in enumerate(sessions)}
        spec = payload['label_spec']; self.horizon = spec['horizon_sessions']
        require(set(spec) == {'definition','horizon_sessions','execution_contract_hash','kind','price_basis'}
                and type(self.horizon) is int and self.horizon > 0 and spec['definition']
                and spec['execution_contract_hash'] and spec['kind'] == 'prediction_proxy'
                and spec['price_basis'] == 'declared_in_definition', 'invalid columnar label specification')
        require(payload['status_codes'] == {'absent':0, 'valid':1, 'data_insufficient':2, 'not_mature':3, 'invalid':4},
                'columnar label status codes differ')
        self.status_names = {v:k for k,v in payload['status_codes'].items()}
        self.values = read_array(payload['arrays']['values'], shape, 'f')
        self.status = read_array(payload['arrays']['status'], shape, 'iu')
        require(bool((self.status <= 4).all()), 'unknown label status')
        require(bool(np.array_equal(np.isfinite(self.values), self.status == 1)), 'label status/value inconsistency')
        require(not bool(((self.status > 0) & ~snapshot.membership).any()), 'label outside declared universe')
        require(len(payload['mature_at']) == len(sessions) and payload['evidence_namespace'], 'label maturity provenance missing')
        self.maturity = [timestamp(v) if v else None for v in payload['mature_at']]
        for i, day in enumerate(sessions):
            if i+1+self.horizon >= len(sessions):
                require(not bool(self.status[i].any()) and self.maturity[i] is None, 'tail labels must remain absent')
            else:
                first = timestamp(sessions[i+1] + 'T09:30:00+08:00')
                last = timestamp(sessions[i+1+self.horizon] + 'T15:00:00+08:00')
                require(self.maturity[i] is not None and snapshot.clocks[day] < first < last <= self.maturity[i], 'label chronology violation')
        self.rows = LabelRows(self)
        self.hash = digest(payload)

    def value(self, session, instrument, research_clock):
        i, j = self.day_indices.get(session), self.snapshot._codes.get(instrument)
        if i is None or j is None or self.status[i, j] != 1 or self.maturity[i] > research_clock: return None
        return float(self.values[i, j])

    def values_at(self, i, research_clock):
        if self.maturity[i] is None or self.maturity[i] > research_clock:
            return np.full(len(self.snapshot.instruments), np.nan)
        return np.asarray(self.values[i])
    def close(self):close_arrays(self.values,self.status)


def validate_columnar_label_calendar(payload, sessions):
    require(payload['schema_version'] == LABEL_SCHEMA and payload['sessions'] == sessions, 'columnar label calendar mismatch')
    horizon = payload['label_spec']['horizon_sessions']
    require(type(horizon) is int and horizon > 0 and len(payload['mature_at']) == len(sessions), 'columnar label horizon invalid')
    for i, value in enumerate(payload['mature_at']):
        if i+1+horizon >= len(sessions): require(value is None, 'future tail maturity fabricated')
        else: require(value is not None and timestamp(value) >= timestamp(sessions[i+1+horizon]+'T15:00:00+08:00'), 'label maturity too early')


def correlation_array(a, b):
    require(len(a) == len(b) and len(a) > 0, 'nonzero matched array pairs required')
    scale_a, scale_b = float(np.max(np.abs(a))), float(np.max(np.abs(b)))
    if scale_a == 0 or scale_b == 0: return None
    aa, bb = a/scale_a, b/scale_b
    da, db = aa-float(np.mean(aa)), bb-float(np.mean(bb))
    va, vb = float(np.dot(da, da)), float(np.dot(db, db))
    if va == 0 or vb == 0: return None
    return max(-1., min(1., float(np.dot(da, db))/math.sqrt(va*vb)))


def factor_statistics_columnar(scores, snapshot, labels, policy, research_clock):
    from .contracts import evaluation
    from .evaluation import validate_policy, describe
    scores=score_grid_view(scores);validate_policy(policy); clock = timestamp(research_clock)
    require(scores and clock >= max(snapshot.clocks[d] for d in scores), 'evaluated decisions beyond research clock')
    sections, ic, rank_ic, n_pairs = [], [], [], 0
    for day in scores:
        i = scores.day_indices[day]; member = scores.membership[i]
        require(bool(np.array_equal(member, snapshot.membership[i])), 'score/universe mismatch')
        x, y = np.asarray(scores.matrix[i]), labels.values_at(i, clock)
        selected = member & np.isfinite(x) & np.isfinite(y)
        n = int(selected.sum()); n_pairs += n; pearson = spearman = None
        if n >= policy['min_cross_section_pairs']:
            a, b = x[selected], y[selected]
            pearson = correlation_array(a, b)
            spearman = correlation_array(rank_vector(a), rank_vector(b))
        if pearson is not None and spearman is not None: ic.append(pearson); rank_ic.append(spearman)
        count = int(member.sum())
        sections.append(dict(session=day, universe_count=count, pair_count=n,
            signal_count=int((member & np.isfinite(x)).sum()), mature_label_count=int((member & np.isfinite(y)).sum()),
            pair_coverage=n/count, pearson_ic=pearson, rank_ic=spearman))
    diagnostics = dict(sections=sections, valid_section_count=len(ic), requested_section_count=len(sections),
        pearson=describe(ic), spearman=describe(rank_ic), label_kind='prediction_proxy', uncertainty_tested=False, multiple_testing_corrected=False)
    if len(ic) < policy['min_valid_sections'] or n_pairs == 0:
        return evaluation('E1','data_insufficient','too_few_nonconstant_mature_sections',diagnostics=diagnostics,n_pairs=n_pairs)
    return evaluation('E1','valid','descriptive_only_no_calibrated_decision',metrics={'pearson':describe(ic),'spearman':describe(rank_ic)},diagnostics=diagnostics,n_pairs=n_pairs)


def paired_scores_columnar(left, right, start, end):
    _same_grids(left, right); sections = []
    for day in left:
        if not start <= day <= end: continue
        i = left.day_indices[day]; a, b = left.matrix[i], right.matrix[i]
        valid = left.membership[i] & np.isfinite(a) & np.isfinite(b)
        n = int(valid.sum())
        sections.append({'session':day,'n_pairs':n,'pearson':correlation_array(a[valid],b[valid]) if n>=3 else None,
                         'missing_pattern_equal':bool(np.array_equal(np.isfinite(a[left.membership[i]]),np.isfinite(b[left.membership[i]])))})
    n = sum(x['n_pairs'] for x in sections); require(n>0,'zero feature pairs')
    correlations = [abs(x['pearson']) for x in sections if x['pearson'] is not None]
    return {'sections':sections,'mean_abs_correlation':mean(correlations) if correlations else None,'n_pairs':n}


def paired_parent_columnar(child, parent, snapshot, labels, start, end, clock):
    _same_grids(child,parent); rows=[]
    for day in snapshot.sessions:
        if not start <= day <= end: continue
        i=child.day_indices[day]; y=labels.values_at(i,clock); a,b=child.matrix[i],parent.matrix[i]
        valid=snapshot.membership[i]&np.isfinite(a)&np.isfinite(b)&np.isfinite(y)
        n=int(valid.sum())
        if n<5:continue
        ra,rb,ry=rank_vector(a[valid]),rank_vector(b[valid]),rank_vector(y[valid])
        ca,cb=correlation_array(ra,ry),correlation_array(rb,ry)
        if ca is not None and cb is not None:
            rows.append({'session':day,'n_pairs':n,'child_ic':ca,'parent_ic':cb,'delta':ca-cb,
                         'signal_rank_correlation':correlation_array(ra,rb)})
    return {'status':'valid' if rows else 'data_insufficient','n_sections':len(rows),'n_pairs':sum(r['n_pairs'] for r in rows),
            'mean_ic_increment':mean(r['delta'] for r in rows) if rows else None,
            'mean_signal_rank_correlation':mean(r['signal_rank_correlation'] for r in rows) if rows else None,
            'support_hash':digest(rows),'scope':'descriptive_common_support_not_economic_reward'}


class BarRow(Mapping):
    def __init__(self, market, i, j): self.market, self.i, self.j = market, i, j
    def _sparse(self): return self.market.sparse.get(self.market.sessions[self.i]+'|'+self.market.instruments[self.j], {})
    def __iter__(self): return iter(self.market.bar_keys | set(self._sparse()))
    def __len__(self): return len(self.market.bar_keys | set(self._sparse()))
    def __getitem__(self,key):
        m,i,j=self.market,self.i,self.j
        if key=='session':return m.sessions[i]
        if key=='instrument':return m.instruments[j]
        if key=='evidence_id':return digest([m.evidence_namespace,m.sessions[i],m.instruments[j]])
        sparse=self._sparse()
        if key in sparse:return sparse[key]
        if key=='action_ids':return []
        if key in m.numeric:
            value=float(m.numeric[key][i,j]);return value if math.isfinite(value) else None
        if key in m.status:
            value=int(m.status[key][i,j]);return None if value==-1 else bool(value)
        if key in m.categorical:
            return m.category_values[key][str(int(m.categorical[key][i,j]))]
        raise KeyError(key)


class BarIndex(Mapping):
    def __init__(self,market):self.market=market
    def __len__(self):return len(self.market.sessions)*len(self.market.instruments)
    def __iter__(self):
        return ((d,s) for d in self.market.sessions for s in self.market.instruments)
    def __getitem__(self,key):
        d,s=key;return BarRow(self.market,self.market.day_indices[d],self.market.code_indices[s])


class DayBars(Mapping):
    def __init__(self,market,i):self.market,self.i=market,i
    def __iter__(self):return iter(self.market.instruments)
    def __len__(self):return len(self.market.instruments)
    def __getitem__(self,code):return BarRow(self.market,self.i,self.market.code_indices[code])


class BarsByDay(Mapping):
    def __init__(self,market):self.market=market
    def __iter__(self):return iter(self.market.sessions)
    def __len__(self):return len(self.market.sessions)
    def __getitem__(self,day):return DayBars(self.market,self.market.day_indices[day])


class BarSequence(Sequence):
    def __init__(self,market):self.market=market
    def __len__(self):return len(self.market.sessions)*len(self.market.instruments)
    def __getitem__(self,index):
        if isinstance(index,slice):return [self[k] for k in range(*index.indices(len(self)))]
        if index<0:index+=len(self)
        if not 0<=index<len(self):raise IndexError(index)
        i,j=divmod(index,len(self.market.instruments));return BarRow(self.market,i,j)


class ColumnarMarket(Mapping):
    """Virtual bars are deliberately absent from manifest iteration and hashing."""
    _columnar_market=True
    def __init__(self,payload):
        require(payload.get('schema_version')==MARKET_SCHEMA,'columnar market schema mismatch')
        require(digest({k:v for k,v in payload.items() if k!='market_hash'})==payload['market_hash'],'columnar market manifest changed')
        # P-20261015-012: a certified market (assurance-bound annotation fields
        # present) is loadable here; the loader itself never validates the
        # assurance - run_account/bind_assurance verify hash bindings.
        self.manifest_payload=payload
        self.sessions,self.instruments,shape=_calendar(payload)
        self.day_indices={d:i for i,d in enumerate(self.sessions)}
        self.code_indices={s:j for j,s in enumerate(self.instruments)}
        arrays=payload['arrays']
        self.numeric={k:read_array(r,shape,'f') for k,r in arrays['numeric'].items()}
        self.status={k:read_array(r,shape,'biu') for k,r in arrays['status'].items()}
        for key,array in self.status.items():require(bool(np.isin(array,[-1,0,1]).all()),'market status not tri-state: '+key)
        self.categorical={k:read_array(r,shape,'iu') for k,r in arrays['categorical'].items()}
        self.category_values=payload['category_values']
        for key,array in self.categorical.items():
            require(set(map(str,map(int,np.unique(array))))<=set(self.category_values[key]),'market category unknown')
        self.sparse=payload.get('sparse_bars',{})
        for key,row in self.sparse.items():
            d,s=key.split('|');require(d in self.day_indices and s in self.code_indices,'sparse market cell outside axes')
            require(not {'session','instrument','evidence_id'}&set(row),'sparse bar identity override forbidden')
        self.evidence_namespace=payload.get('evidence_namespace') or digest(payload.get('provenance',{}))
        self.bar_keys={'session','instrument','evidence_id','action_ids'}|set(self.numeric)|set(self.status)|set(self.categorical)
        self.bars,self.bar_index,self.by_day=BarSequence(self),BarIndex(self),BarsByDay(self)
        self._context=None
    def __len__(self):return len(self.manifest_payload)
    def __iter__(self):return iter(self.manifest_payload)
    def __getitem__(self,key):
        if key=='bars':return self.bars
        return self.manifest_payload[key]
    def research_context(self):
        if self._context is None:
            from .simulation import ResearchAccountContext
            self._context=ResearchAccountContext(self)
        return self._context
    def close(self):close_arrays(*self.numeric.values(),*self.status.values(),*self.categorical.values())


def hydrate_market(payload):
    return ColumnarMarket(payload) if payload.get('schema_version')==MARKET_SCHEMA else payload


def populate_research_context(context,market):
    from .a_share import instrument_rule
    require(isinstance(market,ColumnarMarket),'columnar market required')
    context.market=market;context.execution_hash=digest(market['execution'])
    context.rules={s:instrument_rule(market['execution'],s) for s in market.instruments}
    context.bars,context.by_day=market.bar_index,market.by_day
    context.events=sorted(market['actions'],key=lambda row:timestamp(row['timestamp']))


def account_quotes(market,account,day):
    """Exactly the lookup domain of Account.mark, including share receivables."""
    needed=set(account.positions)
    needed.update(claim['instrument'] for claim in account.share_receivables.values() if claim is not None)
    when=day+'T18:00:00+08:00';quotes={}
    for code in needed:
        bar=market.bar_index.get((day,code),{});close=bar.get('close')
        if close is not None and close>0:
            quotes[code]={'price':str(close),'asof':when,'evidence_id':bar['evidence_id']}
    return quotes


def style_attribution_columnar(scores,style_panels,labels,snapshot,research_clock,*,provenance=None):
    """Full-section OLS metrics with compact support counts/hash; no cell dictionaries.

    All source score grids remain separately reproducible.  This explicitly new
    output form does not pretend to contain the legacy expanded fitted panels.
    """
    scores=score_grid_view(scores);names=sorted(style_panels);require(names,'explicit style exposures required')
    clock=timestamp(research_clock);rows=[]
    for day in scores:
        i=scores.day_indices[day];values=scores.matrix[i];y=labels.values_at(i,clock)
        columns=[style_panels[n].matrix[i] for n in names]
        valid=scores.membership[i]&np.isfinite(values)&np.isfinite(y)
        for column in columns:valid&=np.isfinite(column)
        n=int(valid.sum())
        if n<=len(names)+2:rows.append({'session':day,'n_pairs':n,'status':'data_insufficient'});continue
        values=values[valid];target=y[valid];x=np.column_stack([c[valid] for c in columns])
        x=x-x.mean(axis=0);scale=x.std(axis=0);x=x/np.where(scale>0,scale,1)
        design=np.column_stack([np.ones(n),x]);coefficients,_,rank,_=np.linalg.lstsq(design,values,rcond=None)
        explained=design@coefficients;residual=values-explained
        identified=np.linalg.norm(residual)>1e-12*max(1.,np.linalg.norm(values))
        ry=rank_vector(target);support=[snapshot.instruments[j] for j in np.flatnonzero(valid)]
        rows.append({'session':day,'n_pairs':n,'status':'valid','design_rank':int(rank),'design_columns':int(design.shape[1]),
            'rank_deficient':int(rank)<design.shape[1],'raw_spearman_ic':correlation_array(rank_vector(values),ry),
            'residual_spearman_ic':correlation_array(rank_vector(residual),ry) if identified else None,
            'raw_pearson_ic':correlation_array(values,target),'explained_spearman_ic':correlation_array(rank_vector(explained),ry),
            'explained_pearson_ic':correlation_array(explained,target),'residual_pearson_ic':correlation_array(residual,target) if identified else None,
            'decomposition_max_error':float(np.max(np.abs(values-explained-residual))),'support_hash':digest(support),
            'support_instruments_count':n,'coefficient_identification':'minimum_norm_nonunique' if int(rank)<design.shape[1] else 'full_column_rank',
            'coefficients':{name:float(v) for name,v in zip(['intercept']+names,coefficients)}})
    return {'purpose':'attribution_only_not_investment_neutralization','style_names':names,'sections':rows,
        'n_pairs':sum(r['n_pairs'] for r in rows),'missing_styles':[n for n in ('PIT_industry','PIT_market_cap') if not (provenance or {}).get(n+'_certified',False)],
        'style_provenance':provenance or {'status':'caller supplied diagnostic panels; PIT unverified'},'causal_identification':False,
        'storage_form':'columnar_full_cross_section_metrics; expanded fitted panels not embedded',
        'source_scores_hash':digest(scores),'source_style_hashes':{n:digest(style_panels[n]) for n in names}}


def save_score_grid(scores,directory,stem):
    """A hash-bound transport reference; never used as logical StrategyIR score bytes."""
    require(isinstance(scores,ScoreGrid),'score grid required')
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
    require(stem and Path(stem).name==stem,'invalid score artifact stem')
    paths=[directory/(stem+'.values.npy'),directory/(stem+'.membership.npy')]
    for path,value in zip(paths,(scores.matrix,scores.membership)):
        require(not path.exists(),'score artifact already exists')
        with path.open('xb') as stream:np.save(stream,value,allow_pickle=False)
    descriptor={'schema_version':SCORE_SCHEMA,'sessions':scores.sessions,'instruments':scores.instruments,
                'start':scores.start,'end':scores.end,'values':array_reference(paths[0]),'membership':array_reference(paths[1])}
    descriptor['reference_hash']=digest(descriptor)
    return descriptor


def load_score_grid(reference):
    require(reference.get('schema_version')==SCORE_SCHEMA,'score reference schema mismatch')
    require(digest({k:v for k,v in reference.items() if k!='reference_hash'})==reference['reference_hash'],'score reference changed')
    shape=(len(reference['sessions']),len(reference['instruments']))
    values=read_array(reference['values'],shape,'f');membership=read_array(reference['membership'],shape,'b')
    return ScoreGrid(values,reference['sessions'],reference['instruments'],membership,reference['start'],reference['end'])


def walk_forward_columnar(features,snapshot,labels,*,algorithm='ridge',train_start=None,predict_start=None,
                          retrain_every=20,min_train_sessions=40,lookback_sessions=160,gap_sessions=1,seed=0,
                          predict_end=None):
    """The native chronological walk-forward protocol, with array training rows.

    Row order remains day then declared instrument order.  No global normalization,
    immature label, or future label participates in a fit.  The actual model fitter
    is shared with the legacy implementation.
    """
    from .models import fit_model
    require(features and retrain_every>0 and gap_sessions>=1,'invalid training protocol')
    names=sorted(features);grids={n:score_grid_view(features[n]) for n in names}
    base=grids[names[0]]
    for grid in grids.values():_same_grids(base,grid)
    days=snapshot.sessions[snapshot.start:];predict_start=predict_start or days[min_train_sessions+gap_sessions]
    require(predict_end is None or predict_end in days and predict_end>=predict_start,'prediction end outside declared scope')
    matrix=np.full(snapshot.membership.shape,np.nan)
    scores=ScoreGrid(matrix,snapshot.sessions,snapshot.instruments,snapshot.membership,snapshot.start)
    states,provenance,fit=[],[],None;last_fit=-retrain_every
    for index,day in enumerate(days):
        if predict_end is not None and day>predict_end:break
        if day<predict_start:continue
        cutoff=snapshot.clocks[day];i=snapshot.sessions.index(day)
        if fit is None or index-last_fit>=retrain_every:
            training=[d for d in days[max(0,index-lookback_sessions):max(0,index-gap_sessions)]
                      if train_start is None or d>=train_start]
            matrices,targets,maturity,used=[],[],[],[]
            for training_day in training:
                ti=snapshot.sessions.index(training_day);target=labels.values_at(ti,cutoff)
                valid=snapshot.membership[ti]&np.isfinite(target)
                columns=[grids[n].matrix[ti] for n in names]
                for column in columns:valid&=np.isfinite(column)
                if not bool(valid.any()):continue
                row=labels.rows[(training_day,snapshot.instruments[int(np.flatnonzero(valid)[0])])]
                require(timestamp(row['label_end'])<cutoff,'training label overlaps prediction time')
                matrices.append(np.column_stack([column[valid] for column in columns]))
                targets.append(target[valid]);maturity.append(row['mature_at']);used.append(training_day)
            total=sum(len(y) for y in targets)
            if len(used)<min_train_sessions or total<=len(names)+1:
                provenance.append({'session':day,'status':'data_insufficient','training_sessions':len(used)})
                continue
            x,y=np.concatenate(matrices,axis=0),np.concatenate(targets)
            state,fit=fit_model(algorithm,x,y,seed)
            state.update({'features':names,'training_cutoff':cutoff.isoformat(),
                'latest_training_label_mature_at':max(maturity,key=timestamp),'training_days':sorted(used),
                'training_rows':len(y),'feature_snapshot_hash':snapshot.hash,'labels_hash':labels.hash,
                'gap_sessions':gap_sessions,'score_origin':'walk_forward'})
            state['bundle_hash']=digest(state);states.append(state);last_fit=index
        valid=snapshot.membership[i].copy();columns=[grids[n].matrix[i] for n in names]
        for column in columns:valid&=np.isfinite(column)
        prediction=fit(np.column_stack([c[valid] for c in columns])) if bool(valid.any()) else []
        require(timestamp(states[-1]['latest_training_label_mature_at'])<=timestamp(states[-1]['training_cutoff'])<=cutoff,
                'walk-forward provenance violation')
        values=np.asarray(prediction,dtype=float);values[~np.isfinite(values)]=np.nan;matrix[i,valid]=values
        provenance.append({'session':day,'status':'valid','model_hash':states[-1]['bundle_hash'],
            'training_cutoff':states[-1]['training_cutoff'],'latest_training_label_mature_at':states[-1]['latest_training_label_mature_at'],
            'score_origin':'walk_forward','prediction_count':int(valid.sum())})
    return {'scores':scores,'models':states,'prediction_provenance':provenance,'algorithm':algorithm,
            'trained_model_count':len(states),'eval_status':'valid' if states else 'data_insufficient'}
