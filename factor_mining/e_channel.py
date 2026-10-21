"""
生产 E 道（E0-E4）标准移植到 97 因子宇宙（Path B）
==================================================
把 fuxi_v1 生产研究链的 E1-E4 门控逻辑（研究协议 history_research_v2）原样移植到
factor_mining 的 97 因子周频宇宙，回答：

    「NSGA-II / GFlowNet 产出的 JQ 候选，能否通过生产的 E1-E4 研究门？」

为何需要移植（架构事实）：
  生产 DSL = 10 个纯价量字段 + 17 算子；97 因子宇宙 = 5 维 97 个预计算因子，
  约 50+ 因子无字段映射 ⇒ 直接翻译进生产 DSL 不可行（Path A 会牺牲多样性）。
  因此走 Path B：把生产的 E 道判据（方向 / 年度稳定 / 双交易所 / block bootstrap
  下界 / 独立性 / 配对账户增量 / 成本容量压力）移植到 97 因子宇宙，用周频复刻。

口径忠实度（与原生产逻辑逐条对应，源文件 fuxi_v1/）：
  E1 = research_protocol.screen()        : 方向 + 年度稳定 + 沪深双交易所 + block bootstrap 下界>0
  E2 = research_evaluation.pool_redundancy(): 与已准入池 max|rank corr| < 0.7（独立）或残差 IC>0
  E3 = research_evaluation.paired_accounts(): Top-N 多头净收益 vs 等权市场基线，净增量 bootstrap 下界>0
  E4 = research_evaluation 成本/容量压力   : 双倍成本 + 容量摊薄，仍优于基线

周频适配（生产 horizon=5 日/期 ⇒ 5 日 = 1 周）：
  screen_min_sessions 120 日 ⇒ 24 周；其余阈值原样保留（policy.py）。
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

# ---------------------------------------------------------------------------
# 生产口径常量（policy.py calibration_request / research_protocol.screen）
# ---------------------------------------------------------------------------
ALPHA = 0.05
MAX_TRIALS = 30
BOOTSTRAP_REPLICATES = max(199, int(np.ceil(40 * MAX_TRIALS / ALPHA)))  # = 24000
BLOCK_GRID = [6, 12, 24]          # screen 的 block_grid（三个 block 长度都要下界>0）
BLOCK_LENGTH = 12                 # inference 用 block_length

SCREEN_MIN_PAIRS = 10             # 每个截面至少 10 只有效股票（本库 _cross_sectional_ic 用 50，更严）
SCREEN_MIN_SESSIONS_WEEKLY = 24   # 生产 120 日 ÷ horizon 5 = 24 周（数据支持下限）
SCREEN_PAIR_COVERAGE = 0.8        # 周截面覆盖度 ≥ 80%
SCREEN_POSITIVE_YEAR_FRACTION = 0.6   # ≥ 60% 年份 IC 为正
SCREEN_POSITIVE_EXCHANGE_COUNT = 2    # 沪深两所 IC 均为正

MINIMUM_NET_INCREMENT = 0.0       # 配对净增量下界（>0 才算 pass）
COST_BPS = 15.0                   # 15bp：3bp 佣金往返 + 5bp 印花税 + 7bp 冲击/滑点（cost_spec.py）
COST_STRESS_MULTIPLIER = 2.0      # E4 双倍成本
CAPACITY_STRESS_TOP_N = 60        # E4 容量压力 = top-30 → top-60 摊薄（模拟参与率减半）

E2_CORR_THRESHOLD = 0.7           # pool_redundancy 的 corr_threshold
TOP_N = 30
MIN_STOCKS = 50


# ---------------------------------------------------------------------------
# block bootstrap（忠实移植 research_protocol.block_inference_fast，保留时序缺口）
# ---------------------------------------------------------------------------
def block_inference_fast(values, *, block_length, alpha=ALPHA, replicates=BOOTSTRAP_REPLICATES, seed=0):
    """Circular moving blocks；缺测日期保持槽位、不压缩。返回 {status, mean, lower, upper, ...}。"""
    values = [None if (v is None or not np.isfinite(v)) else float(v) for v in values]
    n = len(values)
    ok = np.array([v is not None for v in values])
    valid = int(ok.sum())
    if valid < 2 * block_length:
        return {"status": "data_insufficient", "n_pairs": valid, "mean": None,
                "lower": None, "upper": None, "p_one_sided": None}
    x = np.array([0.0 if v is None else v for v in values])
    avg = float(x.sum() / valid)

    def sums(a, length):
        if not length:
            return np.zeros(n)
        extended = np.r_[a, a[:length - 1]]
        cs = np.r_[0.0, np.cumsum(extended)]
        return cs[length:] - cs[:-length]

    full, rem = divmod(n, block_length)
    bs = sums(x, block_length)
    bc = sums(ok, block_length)
    rs, rc = sums(x, rem), sums(ok, rem)
    rng = np.random.default_rng(seed)
    draws = []
    for lo in range(0, replicates, 128):
        size = min(128, replicates - lo)
        starts = rng.integers(0, n, size=(size, full + bool(rem)))
        numerator = bs[starts[:, :full]].sum(axis=1)
        denominator = bc[starts[:, :full]].sum(axis=1)
        if rem:
            numerator += rs[starts[:, -1]]
            denominator += rc[starts[:, -1]]
        if not bool((denominator > 0).all()):
            return {"status": "data_insufficient", "n_pairs": valid, "mean": avg,
                    "lower": None, "upper": None, "p_one_sided": None}
        draws.extend((numerator / denominator).tolist())
    arr = np.array(draws)
    p = (1 + int((arr - avg >= avg).sum())) / (replicates + 1)
    return {"status": "valid", "n_pairs": valid, "calendar_slots": n, "mean": avg,
            "lower": float(np.quantile(arr, alpha / 2)), "upper": float(np.quantile(arr, 1 - alpha / 2)),
            "p_one_sided": p, "block_length": block_length, "replicates": replicates}


# ---------------------------------------------------------------------------
# 截面工具
# ---------------------------------------------------------------------------
def _cross_sectional_ic(comp: np.ndarray, fr: np.ndarray, min_stocks: int = MIN_STOCKS) -> np.ndarray:
    """逐周截面 Spearman rank IC（float64）。comp/fr 形状 (n_weeks, n_stocks)。"""
    n_weeks, n_stocks = comp.shape
    ic_vals = np.full(n_weeks, np.nan)
    for w in range(n_weeks):
        c = comp[w]
        f = fr[w]
        valid = np.isfinite(c) & np.isfinite(f)
        if valid.sum() < min_stocks:
            continue
        rk_c = rankdata(c[valid])
        rk_f = rankdata(f[valid])
        n = len(rk_c)
        d2 = np.sum((rk_c - rk_f) ** 2)
        ic_vals[w] = 1.0 - 6.0 * d2 / (n * (n * n - 1))
    return ic_vals


def _cross_sectional_rank_matrix(arr: np.ndarray) -> np.ndarray:
    """逐行截面百分位 rank → [0,1]，NaN 保留。返回 float32 省内存。"""
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    for w in range(arr.shape[0]):
        row = arr[w]
        valid = np.isfinite(row)
        if valid.sum() < 2:
            continue
        rk = rankdata(row[valid])
        out[w, valid] = (rk.astype(np.float64) / valid.sum()).astype(np.float32)
    return out


def _pearson(a, b):
    """两个 1D 向量（对齐后）的 Pearson 相关，NaN 掩码对齐。"""
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3:
        return None
    a, b = a[valid].astype(np.float64), b[valid].astype(np.float64)
    ac = a - a.mean()
    bc = b - b.mean()
    denom = float(np.sqrt((ac @ ac) * (bc @ bc)))
    if denom < 1e-12:
        return None
    return float((ac @ bc) / denom)


def _mean_abs_rank_corr(cand_rank, pool_rank):
    """向量化：候选 rank 与池因子 rank 的「周均值 |Pearson|」。cand/pool (W,N) float32，NaN 掩码对齐。"""
    valid = np.isfinite(cand_rank) & np.isfinite(pool_rank)   # (W, N)
    a = np.where(valid, cand_rank, 0.0).astype(np.float64)
    b = np.where(valid, pool_rank, 0.0).astype(np.float64)
    cnt = valid.sum(axis=1).astype(np.float64)               # (W,)
    am = a.sum(axis=1) / np.maximum(cnt, 1.0)
    bm = b.sum(axis=1) / np.maximum(cnt, 1.0)
    ac = a - (am[:, None] * valid)
    bc = b - (bm[:, None] * valid)
    num = (ac * bc).sum(axis=1)
    denom = np.sqrt((ac * ac).sum(axis=1) * (bc * bc).sum(axis=1))
    corr = np.where((cnt >= 3) & (denom > 1e-12), num / np.maximum(denom, 1e-12), np.nan)
    corr = np.abs(corr[np.isfinite(corr)])
    return float(np.mean(corr)) if corr.size else None


# ---------------------------------------------------------------------------
# E1：screen（方向 + 年度稳定 + 双交易所 + block bootstrap 下界）
# ---------------------------------------------------------------------------
def e1_screen(comp_df, fr, *, block_grid=BLOCK_GRID, min_sessions=SCREEN_MIN_SESSIONS_WEEKLY):
    """E1 等价门。comp_df / fr 为对齐的 DataFrame（周 × 股）。

    与生产 screen 对应：
      - 方向：候选 IC 均值的符号（方向感知，短因子翻转后仍可做多）
      - 年度稳定：各年（≥ min_sessions 周）signed IC 均值 > 0，正年占比 ≥ 0.6
      - 双交易所：.SH / .SZ 两所各自 signed IC 均值 > 0（count = 2）
      - block bootstrap：block_grid 三个 block 长度的 signed IC 下界都要 > 0
    """
    idx = list(comp_df.index)
    cols = list(comp_df.columns)
    C = comp_df.astype(np.float64).values
    F = fr.astype(np.float64).values

    ic = _cross_sectional_ic(C, F)
    ic_f = ic[np.isfinite(ic)]
    n_weeks = int(np.isfinite(ic).sum())
    if n_weeks < min_sessions:
        return {"eval_status": "data_insufficient", "decision": "not_applicable",
                "reason_code": "research_support_inadequate", "n_weeks": n_weeks}
    avg_ic = float(np.mean(ic_f))
    direction = 1 if avg_ic >= 0 else -1
    signed_ic = ic * direction  # 方向归一化后，>0 = 在交易方向上有效

    # 周截面覆盖度
    coverage = np.nanmean([np.isfinite(C[w]).sum() / C.shape[1] for w in range(C.shape[0])]) \
        if C.shape[1] else 0.0
    data_ok = n_weeks >= min_sessions and coverage >= SCREEN_PAIR_COVERAGE

    # block bootstrap 下界（三个 block 长度）
    intervals = [block_inference_fast(signed_ic, block_length=b, seed=1729) for b in block_grid]
    bootstrap_ok = all(x["status"] == "valid" and (x["lower"] or 0) > 0 for x in intervals)

    # 年度稳定
    years = sorted({t.year for t in idx if hasattr(t, "year")})
    yearly = {}
    for y in years:
        mask = np.array([t.year == y for t in idx])
        vals = signed_ic[mask]
        vals = vals[np.isfinite(vals)]
        yearly[y] = {"n": int(len(vals)), "mean_signed_ic": float(np.mean(vals)) if len(vals) else None}
    eligible_years = [v for v in yearly.values() if v["n"] >= min_sessions]
    positive_years = sum(1 for v in eligible_years if v["mean_signed_ic"] is not None and v["mean_signed_ic"] > 0)
    year_fraction = positive_years / len(eligible_years) if eligible_years else 0.0

    # 双交易所
    exchanges = {}
    for exchange in ("XSHG", "XSHE"):
        suffix = ".SH" if exchange == "XSHG" else ".SZ"
        col_mask = np.array([c.endswith(suffix) for c in cols])
        n_cols = int(col_mask.sum())
        if n_cols == 0:
            exchanges[exchange] = {"n": 0, "mean_signed_ic": None}
            continue
        sub_ic = _cross_sectional_ic(C[:, col_mask], F[:, col_mask])
        sub_f = sub_ic[np.isfinite(sub_ic)]
        sub_mean = float(np.mean(sub_f) * direction) if len(sub_f) else None
        exchanges[exchange] = {"n": n_cols, "n_weeks": int(len(sub_f)), "mean_signed_ic": sub_mean}
    exchange_positive = sum(1 for v in exchanges.values()
                            if v["n"] > 0 and v["mean_signed_ic"] is not None and v["mean_signed_ic"] > 0)

    passed = (data_ok and bootstrap_ok and year_fraction >= SCREEN_POSITIVE_YEAR_FRACTION
              and exchange_positive >= SCREEN_POSITIVE_EXCHANGE_COUNT)

    return {
        "eval_status": "valid" if data_ok else "data_insufficient",
        "decision": "pass" if passed else ("reject" if data_ok else "not_applicable"),
        "reason_code": "research_v2_direction_and_stability_screen",
        "direction": direction,
        "ic_mean": avg_ic,
        "signed_ic_mean": float(np.nanmean(signed_ic)),
        "n_weeks": n_weeks,
        "coverage": float(coverage),
        "bootstrap": [{"block": b, "lower": x["lower"], "status": x["status"]} for b, x in zip(block_grid, intervals)],
        "bootstrap_ok": bool(bootstrap_ok),
        "yearly": yearly,
        "positive_year_fraction": float(year_fraction),
        "exchanges": exchanges,
        "exchange_positive": int(exchange_positive),
    }


# ---------------------------------------------------------------------------
# E2：pool_redundancy（独立性）
# ---------------------------------------------------------------------------
def e2_pool_redundancy(comp_df, fr, pool_ranks, pool_names, *, corr_threshold=E2_CORR_THRESHOLD):
    """E2 等价门。pool = 97 个原始因子（基础宇宙）；候选若只是重表达已有因子则冗余。

    与生产 pool_redundancy 对应：
      - max |rank corr| < 0.7 ⇒ 独立（pass）
      - 否则残差 IC：候选 rank 对最相关池因子 rank 正交化后，残差 IC 的 bootstrap 下界 > 0
        才 pass（仍携带正交 alpha），否则 reject（冗余）。
    """
    C = comp_df.astype(np.float64).values
    F = fr.astype(np.float64).values
    cand_rank = _cross_sectional_rank_matrix(C)

    if not pool_ranks:
        return {"decision": "pass", "reason_code": "empty_pool_no_reference",
                "pool_size": 0, "max_corr": None}

    # 找最相关的池因子（向量化）
    best = None
    for name in pool_names:
        pr = pool_ranks[name]
        mc = _mean_abs_rank_corr(cand_rank, pr)
        if mc is None:
            continue
        if best is None or mc > best["corr"]:
            best = {"name": name, "corr": mc, "ranks": pr}
    if best is None:
        return {"decision": "pass", "reason_code": "no_comparable_pool_member",
                "pool_size": len(pool_names), "max_corr": None}
    max_corr = best["corr"]
    if max_corr < corr_threshold:
        return {"decision": "pass", "reason_code": "independent_of_pool",
                "pool_size": len(pool_names), "max_corr": max_corr}

    # 残差 IC：候选 rank 对最相关池因子 rank 正交化
    residual_ic = []
    for w in range(C.shape[0]):
        a = cand_rank[w].astype(np.float64)
        b = best["ranks"][w].astype(np.float64)
        y = F[w]
        valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(y)
        if valid.sum() < 5:
            continue
        ra = rankdata(a[valid])
        rb = rankdata(b[valid])
        ry = rankdata(y[valid])
        ra_c, rb_c = ra - ra.mean(), rb - rb.mean()
        denom = float(np.dot(rb_c, rb_c))
        if denom == 0:
            continue
        beta = float(np.dot(ra_c, rb_c)) / denom
        resid = ra_c - beta * rb_c
        c = _pearson(resid, ry - ry.mean())
        if c is not None:
            residual_ic.append(c)
    if not residual_ic:
        return {"decision": "reject", "reject_class": "redundancy",
                "reason_code": "redundant_no_residual_support", "redundant_with": best["name"],
                "pool_size": len(pool_names), "max_corr": max_corr}
    interval = block_inference_fast(residual_ic, block_length=BLOCK_LENGTH, seed=20261018)
    if interval["status"] == "valid" and (interval["lower"] or 0) > 0:
        return {"decision": "pass", "reason_code": "high_corr_but_independent_residual_alpha",
                "redundant_with": best["name"], "pool_size": len(pool_names),
                "max_corr": max_corr, "residual_alpha": interval}
    return {"decision": "reject", "reject_class": "redundancy",
            "reason_code": "redundant_with_pool_factor", "redundant_with": best["name"],
            "pool_size": len(pool_names), "max_corr": max_corr, "residual_alpha": interval}


# ---------------------------------------------------------------------------
# Top-N 组合净收益（E3/E4 共用）
# ---------------------------------------------------------------------------
def top_n_net_returns(comp_df, fr, direction, *, top_n=TOP_N, cost_bps=COST_BPS):
    """方向感知 Top-N 等权多头，周净收益 = 毛收益 − 换手 × 成本。返回 (net, turnover_series)。"""
    C = comp_df.astype(np.float64).values
    F = fr.astype(np.float64).values
    n_weeks, n_stocks = C.shape

    top_idx = np.zeros((n_weeks, top_n), dtype=int)
    for w in range(n_weeks):
        row = C[w].copy()
        row[~np.isfinite(row)] = -np.inf
        order = np.argsort(-row) if direction >= 0 else np.argsort(row)
        top_idx[w] = order[:top_n]

    gross = np.full(n_weeks - 1, np.nan)
    for w in range(n_weeks - 1):
        row = F[w, top_idx[w]]
        if np.isfinite(row).sum() < 5:
            continue
        gross[w] = np.nanmean(row)

    mask = np.zeros((n_weeks, n_stocks), dtype=bool)
    for w in range(n_weeks):
        mask[w, top_idx[w]] = True
    overlap = (mask[:-1] & mask[1:]).sum(axis=1)
    turnover = 1.0 - overlap / top_n

    net = gross - turnover * cost_bps / 1e4
    return net, turnover


def equal_weight_baseline(fr):
    """等权市场基线：每周全部股票 forward return 的均值（成本忽略）。"""
    import warnings
    F = fr.astype(np.float64).values
    if F.shape[0] <= 1:
        return np.array([])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(F[:-1], axis=1)


def _paired_increment(net, baseline):
    """净增量 = 候选净收益 − 基线净收益；返回 block bootstrap 下界。"""
    n = min(len(net), len(baseline))
    delta = [net[i] - baseline[i] for i in range(n) if np.isfinite(net[i]) and np.isfinite(baseline[i])]
    if not delta:
        return {"status": "data_insufficient", "lower": None, "mean": None}
    return block_inference_fast(delta, block_length=BLOCK_LENGTH, seed=731)


# ---------------------------------------------------------------------------
# E3 / E4
# ---------------------------------------------------------------------------
def e3_paired_account(comp_df, fr, direction):
    """E3 等价门：Top-N 多头净收益 vs 等权市场基线，净增量下界 > 0。"""
    net, _ = top_n_net_returns(comp_df, fr, direction)
    baseline = equal_weight_baseline(fr)
    interval = _paired_increment(net, baseline)
    passed = interval["status"] == "valid" and (interval["lower"] or 0) > MINIMUM_NET_INCREMENT
    return {"eval_status": "valid" if interval["status"] == "valid" else "data_insufficient",
            "decision": "pass" if passed else ("reject" if interval["status"] == "valid" else "not_applicable"),
            "reason_code": "paired_account_increment",
            "paired_mean_increment": interval.get("mean"),
            "increment_lower": interval.get("lower"),
            "status": interval.get("status")}


def e4_stress(comp_df, fr, direction):
    """E4 等价门：双倍成本 + 容量摊薄（top-30→top-60）两个压力场景，配对增量下界都要 > 0。"""
    stress = []

    # 场景 1：双倍成本
    net_2x, _ = top_n_net_returns(comp_df, fr, direction, cost_bps=COST_BPS * COST_STRESS_MULTIPLIER)
    base_2x = equal_weight_baseline(fr)
    int_2x = _paired_increment(net_2x, base_2x)
    stress.append({"scenario": "double_cost", "increment": int_2x,
                   "passed": int_2x["status"] == "valid" and (int_2x["lower"] or 0) > MINIMUM_NET_INCREMENT})

    # 场景 2：容量摊薄（top-30 → top-60）
    net_60, _ = top_n_net_returns(comp_df, fr, direction, top_n=CAPACITY_STRESS_TOP_N)
    base_60 = equal_weight_baseline(fr)
    int_60 = _paired_increment(net_60, base_60)
    stress.append({"scenario": "reduced_capacity_top60", "increment": int_60,
                   "passed": int_60["status"] == "valid" and (int_60["lower"] or 0) > MINIMUM_NET_INCREMENT})

    valid = all(x["increment"]["status"] == "valid" for x in stress)
    passed = valid and all(x["passed"] for x in stress)
    return {"eval_status": "valid" if valid else "data_insufficient",
            "decision": "pass" if passed else ("reject" if valid else "not_applicable"),
            "reason_code": "cost_capacity_stress",
            "stress": stress}


# ---------------------------------------------------------------------------
# 编排：E0-E4
# ---------------------------------------------------------------------------
def evaluate_candidate_e_channel(comp_df, fr, pool_ranks, pool_names, *, run_e4=True):
    """对一个候选因子跑 E1-E4 等价门（漏斗短路）。返回 {evaluations:[e1..e4], passed_all, funnel}。

    漏斗语义（忠实生产）：E1 不过则后续 not_applicable；E2 只在 E1 通过后跑；
    E3 只在 E1∧E2 通过后跑；E4 只在 E1∧E2∧E3 通过后跑。
    """
    NA = lambda reason: {"decision": "not_applicable", "reason_code": reason}
    e1 = e1_screen(comp_df, fr)
    direction = e1.get("direction", 1)
    evals = {"E1": e1}

    if e1["decision"] != "pass":
        evals.update(E2=NA("e1_not_pass"), E3=NA("e1_not_pass"), E4=NA("e1_not_pass"))
        return {"evaluations": evals, "passed_e1": False, "passed_e2": False,
                "passed_e3": False, "passed_e4": False, "passed_all": False,
                "funnel": {"e1": False, "e2": False, "e3": False, "e4": False}}

    evals["E2"] = e2_pool_redundancy(comp_df, fr, pool_ranks, pool_names)
    if evals["E2"]["decision"] != "pass":
        evals.update(E3=NA("e2_not_pass"), E4=NA("e2_not_pass"))
        return {"evaluations": evals, "passed_e1": True, "passed_e2": False,
                "passed_e3": False, "passed_e4": False, "passed_all": False,
                "funnel": {"e1": True, "e2": False, "e3": False, "e4": False}}

    evals["E3"] = e3_paired_account(comp_df, fr, direction)
    if evals["E3"]["decision"] != "pass":
        evals.update(E4=NA("e3_not_pass"))
        return {"evaluations": evals, "passed_e1": True, "passed_e2": True,
                "passed_e3": False, "passed_e4": False, "passed_all": False,
                "funnel": {"e1": True, "e2": True, "e3": False, "e4": False}}

    evals["E4"] = e4_stress(comp_df, fr, direction) if run_e4 else NA("e4_disabled")
    passed4 = evals["E4"]["decision"] == "pass"
    return {
        "evaluations": evals,
        "passed_e1": True, "passed_e2": True, "passed_e3": True, "passed_e4": passed4,
        "passed_all": passed4,
        "funnel": {"e1": True, "e2": True, "e3": True, "e4": passed4},
    }
