"""
Shapley-Sortino Factor Search — pure single-objective (replaces NSGA-II Phase 4)
==============================================================================

Method: Monte-Carlo Permutation Shapley (Data Shapley, Ghorbani & Zou 2019)
  - Value function V(S) = annualized Sortino ratio of a top-N equal-weight
    portfolio formed by the SUM of z-scored factors in coalition S (training set).
    (Composite scale is irrelevant for top-N ranking, so unnormalized sum is used.)
  - For each factor i, MC estimate of its marginal contribution:
        phi_i = mean over random permutations of [V(S U {i}) - V(S)]
  - Drop factors with phi_i <= 0 (no contribution at all).
  - Unique optimal combination = grand coalition of all surviving (positive-
    contribution) factors, weighted by redundancy-discounted Shapley values.

No factor-count limit (user requirement). Output is a SINGLE optimal
combination, wrapped in the Pareto-compatible structure so downstream
Phase 4.5 decode / WFA keep working unchanged.

Advantages over the old 4-factor-restricted version:
  - No C(N,4) combinatorial explosion; MC sampling scales linearly in N.
  - Searches the FULL factor space, not just 4-factor slices.
  - Genuinely single-objective: returns the one best combination, not a
    multi-objective Pareto front that collapses into duplicates.
"""

import numpy as np
import pandas as pd
import sys
import time
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# Value function: Sortino of top-N portfolio from a composite
# ============================================================

def _V(composite, fwd_ret, top_n=30, min_valid_weeks=26):
    """
    composite : np.ndarray (weeks, stocks) -- sum of z-scores
    fwd_ret   : np.ndarray (weeks, stocks) -- next-week returns
    Returns annualized Sortino (clipped [-5, 5]); 0.0 if insufficient data.
    """
    weeks, stocks = composite.shape
    if weeks < min_valid_weeks:
        return 0.0
    # NaN-safe: treat non-finite composite as -inf so it is never selected
    comp_clean = np.where(np.isfinite(composite), composite, -np.inf)
    top_idx = np.argpartition(-comp_clean, top_n - 1, axis=1)[:, :top_n]
    port = np.nanmean(fwd_ret[np.arange(weeks)[:, None], top_idx], axis=1)
    valid = np.isfinite(port)
    if valid.sum() < min_valid_weeks:
        return 0.0
    port = port[valid]
    excess = port  # target = 0
    downside = np.minimum(excess, 0.0)
    dv = np.mean(downside ** 2)
    if dv <= 1e-12:
        return 5.0
    sortino = np.mean(excess) / np.sqrt(dv) * np.sqrt(52.0)
    return float(np.clip(sortino, -5.0, 5.0))


# ============================================================
# Main search: pure single-objective MC Shapley over ALL factors
# ============================================================

def shapley_sortino_search(
    factor_names, factor_dfs, factor_dfs_std,
    forward_returns, corr_matrix,
    ic_summary=None,
    mc_iterations=500,
    redundancy_threshold=0.7,
    redundancy_discount=0.8,
    contrib_threshold=0.0,        # drop factors with phi <= contrib_threshold * max_phi
    min_factors=1,
    top_n=30,
    seed=42,
    verbose=True,
):
    """
    Pure single-objective Shapley-Sortino search over ALL screened factors.

    Returns
    -------
    tuple : (population, objectives, pareto_front)
        Compatible with nsga2_run output format (but a single solution).
    """
    from ga.chromosome import Chromosome

    t0 = time.time()

    # ---- 1. Build aligned numpy universe (all factors that have data) ----
    factors = [n for n in factor_names
               if n in factor_dfs_std and n in corr_matrix.index]
    n = len(factors)
    if verbose:
        print(f"  [Shapley] Universe: {n} factors (ALL screened, no count limit)")
        sys.stdout.flush()

    # common columns across all factors + fwd_ret
    common_cols = None
    for name in factors:
        cols = factor_dfs_std[name].columns
        common_cols = cols if common_cols is None else common_cols.intersection(cols)
    common_cols = common_cols.intersection(forward_returns.columns)
    common_cols = list(common_cols)
    if verbose:
        print(f"  [Shapley] Aligned stock universe: {len(common_cols)} stocks")
        sys.stdout.flush()

    Z = []
    for name in factors:
        arr = factor_dfs_std[name].loc[:, common_cols].values.astype(np.float64)
        Z.append(arr)
    fwd = forward_returns.loc[:, common_cols].values.astype(np.float64)
    W, S = fwd.shape

    # correlation as numpy (factors x factors), aligned to `factors` order
    corr_np = corr_matrix.loc[factors, factors].values.astype(np.float64)

    # ---- 2. MC permutation Shapley ----
    if verbose:
        print(f"  [Shapley] MC permutation sampling: {mc_iterations} iters x {n} factors "
              f"= {mc_iterations * n:,} coalition evaluations ...")
        sys.stdout.flush()

    phi = np.zeros(n)
    rng = np.random.default_rng(seed)
    for it in range(mc_iterations):
        perm = rng.permutation(n)
        comp = np.zeros((W, S), dtype=np.float64)
        last_v = 0.0
        for k, i in enumerate(perm):
            comp += Z[i]
            v_after = _V(comp, fwd, top_n=top_n)
            phi[i] += (v_after - last_v)
            last_v = v_after
        if verbose and (it + 1) % 100 == 0:
            print(f"    MC iter {it + 1}/{mc_iterations} ({100 * (it + 1) // mc_iterations}%)")
            sys.stdout.flush()

    phi /= mc_iterations

    if verbose:
        order = np.argsort(-phi)
        print(f"  [Shapley] MC done in {time.time() - t0:.1f}s")
        print(f"    phi range: [{phi.min():.4f}, {phi.max():.4f}]")
        print(f"    Top contributors:")
        for i in order[:8]:
            print(f"      {factors[i]:30s} phi={phi[i]:+.4f}")
        sys.stdout.flush()

    # ---- 3. Drop zero / negative contributors ----
    max_phi = phi.max() if phi.max() > 0 else 1.0
    keep_mask = phi > contrib_threshold * max_phi
    survivors = [i for i in range(n) if keep_mask[i]]
    if len(survivors) < min_factors:
        # relax: keep top min_factors by phi
        survivors = list(order[:max(min_factors, n)])
    survivors = sorted(survivors, key=lambda i: -phi[i])

    if verbose:
        print(f"  [Shapley] Surviving factors (phi>0): {len(survivors)}/{n}")
        print(f"    {[factors[i] for i in survivors]}")
        sys.stdout.flush()

    if len(survivors) == 0:
        print("  [Shapley] WARNING: no positive-contribution factor found!")
        return [], np.empty((0, 3)), {
            'objectives': np.empty((0, 3)), 'chromosomes': [], 'weights': [], 'details': []
        }

    # ---- 4. Redundancy-discounted weights ----
    phi_adj = np.zeros(n)
    for rank, i in enumerate(survivors):
        adj = phi[i]
        for j in survivors[:rank]:
            rho = abs(corr_np[i, j])
            if rho > redundancy_threshold:
                disc = 1.0 - redundancy_discount * (rho - redundancy_threshold)
                disc = max(disc, 0.05)
                adj *= disc
        phi_adj[i] = max(adj, 0.0)

    total = phi_adj[survivors].sum()
    if total <= 0:
        w_arr = np.array([1.0 / len(survivors) if i in survivors else 0.0 for i in range(n)])
    else:
        w_arr = np.array([phi_adj[i] / total if i in survivors else 0.0 for i in range(n)])

    weights = {factors[i]: w_arr[i] for i in survivors}

    # ---- 5. Unique optimal combination: grand coalition of survivors ----
    comp_opt = np.zeros((W, S), dtype=np.float64)
    for i in survivors:
        comp_opt += Z[i]
    sortino_opt = _V(comp_opt, fwd, top_n=top_n)

    # ---- 6. Build Pareto-compatible output (single solution) ----
    svals = phi_adj[survivors]
    if len(svals) > 1 and svals.mean() > 0:
        cv = svals.std() / svals.mean()
        s_obj = float(np.clip(1.0 - cv, 0.0, 1.0))
    else:
        s_obj = 0.5
    q_obj = float(np.clip(sortino_opt / 3.0, 0.0, 1.0))
    c_obj = q_obj  # single-objective: cost-adjusted == quality (both from Sortino)

    objectives = np.array([[q_obj, s_obj, c_obj]])
    w_full = {factors[i]: w_arr[i] for i in range(n)}
    chromosomes = [Chromosome(np.array(
        [w_full.get(f, 0.0) for f in factor_names], dtype=float))]
    weights_list = [weights]   # survivors only
    details_list = [{
        'sortino': sortino_opt,
        'shapley_raw': {factors[i]: phi[i] for i in survivors},
        'shapley_adj': {factors[i]: phi_adj[i] for i in survivors},
        'total_score': float(phi_adj[survivors].sum()),
        'survivors': [factors[i] for i in survivors],
        'n_survivors': len(survivors),
    }]

    pareto_front = {
        'objectives': objectives,
        'chromosomes': chromosomes,
        'weights': weights_list,
        'details': details_list,
    }

    # safety dump for offline debugging
    try:
        import pickle
        with open('/e/quant/research/factor_alchemy/output/_shapley_mc_dump.pkl', 'wb') as f:
            pickle.dump({
                'phi': phi, 'factors': factors, 'survivors': survivors,
                'phi_adj': phi_adj, 'weights': weights, 'sortino_opt': sortino_opt,
                'mc_iterations': mc_iterations,
            }, f)
    except Exception:
        pass

    if verbose:
        print(f"  [Shapley] UNIQUE OPTIMAL COMBINATION ({len(survivors)} factors, "
              f"Sortino={sortino_opt:.3f}):")
        for i in survivors:
            print(f"      {factors[i]:30s} w={w_arr[i]:.4f}  phi={phi[i]:+.4f}")
        sys.stdout.flush()

    return chromosomes, objectives, pareto_front
