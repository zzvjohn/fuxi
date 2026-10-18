"""E2 pool-redundancy gate: empty pool, independent, redundant, residual-alpha pass."""
import unittest
import numpy as np
from fuxi_v1.research_evaluation import pool_redundancy, signal_rank_correlation, residual_rank_ic


class _Labels:
    """Legacy-dict labels with a deterministic cross-section per day."""
    def __init__(self, value_by_day):
        self.value_by_day = value_by_day

    def value(self, day, instrument, clock):
        return self.value_by_day.get(day, {}).get(instrument)


def _request():
    return {'research_gate_version': 'history_research_v2', 'block_length': 3, 'alpha': .05,
            'bootstrap_replicates': 399, 'max_trials': 1}


def _grid(days, stocks, fn):
    """dict-of-dict scores: {day: {stock: value}}."""
    return {d: {s: fn(i) for i, s in enumerate(stocks)} for d in days}


class PoolRedundancyTests(unittest.TestCase):
    def test_empty_pool_passes(self):
        r = pool_redundancy({}, [], None, None, None, '2024-01-01', '2024-01-03', _request())
        self.assertEqual(r['decision'], 'pass')
        self.assertEqual(r['reason_code'], 'empty_pool_no_reference')
        self.assertEqual(r['pool_size'], 0)

    def test_signal_rank_correlation_identical_and_reversed(self):
        days = ['2024-01-01', '2024-01-02']
        stocks = ['A', 'B', 'C', 'D', 'E']
        left = _grid(days, stocks, lambda i: float(i))
        same = _grid(days, stocks, lambda i: float(i))
        reversed_ = _grid(days, stocks, lambda i: float(4 - i))
        self.assertAlmostEqual(signal_rank_correlation(left, same, None, '2024-01-01', '2024-01-02'), 1.0)
        self.assertAlmostEqual(signal_rank_correlation(left, reversed_, None, '2024-01-01', '2024-01-02'), 1.0)

    def test_low_correlation_passes(self):
        # Candidate rank is the reverse of the pool rank -> |corr| = 1 actually; use a
        # scrambled pool that decorrelates from the candidate ordering.
        days = ['2024-01-%02d' % d for d in range(1, 11)]
        stocks = ['S%02d' % i for i in range(20)]
        rng = np.random.default_rng(7)
        pool_order = rng.permutation(20)  # random rank of the pool factor
        candidate = _grid(days, stocks, lambda i: float(i))
        pool = {d: {s: float(pool_order[i]) for i, s in enumerate(stocks)} for d in days}
        r = pool_redundancy(candidate, [{'candidate_id': 'p0', 'scores': pool}],
                            None, None, None, '2024-01-01', '2024-01-10', _request())
        self.assertEqual(r['decision'], 'pass')
        self.assertEqual(r['reason_code'], 'independent_of_pool')
        self.assertLess(r['max_corr'], 0.7)

    def test_identical_high_corr_no_residual_rejects(self):
        days = ['2024-01-%02d' % d for d in range(1, 11)]
        stocks = ['S%02d' % i for i in range(20)]
        candidate = _grid(days, stocks, lambda i: float(i))
        pool = [{'candidate_id': 'p0', 'scores': _grid(days, stocks, lambda i: float(i))}]
        # Labels are irrelevant here because identical ranks yield a zero-variance
        # residual, so residual_rank_ic returns no rows -> redundant.
        labels = _Labels({d: {s: float(i) for i, s in enumerate(stocks)} for d in days})
        r = pool_redundancy(candidate, pool, None, labels, None, '2024-01-01', '2024-01-10', _request())
        self.assertEqual(r['decision'], 'reject')
        self.assertEqual(r['reject_class'], 'redundancy')
        self.assertAlmostEqual(r['max_corr'], 1.0)

    def test_high_corr_but_independent_residual_alpha_passes(self):
        # Two orthogonal latent components A and B; label loads on both, the pool
        # factor captures A only, and the candidate = 0.7*A + 0.3*B. The candidate is
        # highly correlated with the pool (both carry A) yet retains independent alpha
        # through B, so E2 must pass.
        days = ['2024-01-%02d' % d for d in range(1, 41)]
        stocks = ['S%02d' % i for i in range(40)]
        rng = np.random.default_rng(20261018)
        A = rng.normal(size=40)
        B = rng.normal(size=40)
        label_rank = 0.6 * A + 0.4 * B  # forward-return proxy
        pool_raw = A.copy()
        cand_raw = 0.7 * A + 0.3 * B
        # scores: value = rank position, but we must preserve each factor's ordering.
        def order(values):
            return np.argsort(np.argsort(values)).astype(float)
        pool_rank = order(pool_raw)
        cand_rank = order(cand_raw)
        label_rank = order(label_rank)
        pool = [{'candidate_id': 'p0', 'scores': {d: {s: float(pool_rank[i]) for i, s in enumerate(stocks)} for d in days}}]
        candidate = {d: {s: float(cand_rank[i]) for i, s in enumerate(stocks)} for d in days}
        labels = _Labels({d: {s: float(label_rank[i]) for i, s in enumerate(stocks)} for d in days})
        r = pool_redundancy(candidate, pool, None, labels, None, '2024-01-01', days[-1],
                            _request(), corr_threshold=0.5)
        self.assertGreater(r['max_corr'], 0.5)
        self.assertEqual(r['decision'], 'pass')
        self.assertEqual(r['reason_code'], 'high_corr_but_independent_residual_alpha')
        self.assertGreater(r['residual_alpha']['lower'], 0)

    def test_correlated_cleaner_estimate_passes_as_new_peak(self):
        # A candidate that is a cleaner estimate of the SAME signal as an admitted
        # pool factor is highly correlated with it yet materially higher quality
        # (higher rank IC). It is the "new peak" of its family and must be kept,
        # not rejected as a duplicate: higher IC with high correlation implies a
        # positive residual alpha, which the residual branch admits.
        days = ['2024-01-%02d' % d for d in range(1, 41)]
        stocks = ['S%02d' % i for i in range(40)]
        rng = np.random.default_rng(6)
        A = rng.normal(size=40)                          # latent signal
        pool_raw = A + 1.0 * rng.normal(size=40)         # noisy estimate
        cand_raw = A + 0.2 * rng.normal(size=40)         # cleaner estimate

        def order(v):
            return np.argsort(np.argsort(v)).astype(float)
        pool_rank, cand_rank, label_rank = order(pool_raw), order(cand_raw), order(A)
        pool = [{'candidate_id': 'p0', 'scores': {d: {s: float(pool_rank[i]) for i, s in enumerate(stocks)} for d in days}}]
        candidate = {d: {s: float(cand_rank[i]) for i, s in enumerate(stocks)} for d in days}
        label_scores = {d: {s: float(label_rank[i]) for i, s in enumerate(stocks)} for d in days}
        labels = _Labels(label_scores)

        corr = signal_rank_correlation(candidate, pool[0]['scores'], None, days[0], days[-1])
        self.assertGreater(corr, 0.7)                     # still highly correlated
        cand_ic = signal_rank_correlation(candidate, label_scores, None, days[0], days[-1])
        pool_ic = signal_rank_correlation(pool[0]['scores'], label_scores, None, days[0], days[-1])
        self.assertGreater(cand_ic, pool_ic)              # the "new peak" is higher quality

        r = pool_redundancy(candidate, pool, None, labels, None, days[0], days[-1], _request())
        self.assertEqual(r['decision'], 'pass')
        self.assertEqual(r['reason_code'], 'high_corr_but_independent_residual_alpha')

    def test_residual_rank_ic_zero_variance_returns_empty(self):
        days = ['2024-01-01', '2024-01-02']
        stocks = ['A', 'B', 'C', 'D', 'E']
        scores = _grid(days, stocks, lambda i: float(i))
        labels = _Labels({d: {s: float(i) for i, s in enumerate(stocks)} for d in days})
        self.assertEqual(residual_rank_ic(scores, scores, None, labels, None, '2024-01-01', '2024-01-02'), [])


if __name__ == '__main__':
    unittest.main()
