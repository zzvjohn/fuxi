"""
修复最后 8 个 reserve 因子 — 全部用向量化重写, 消除 rolling().apply() lambda。
运行后增量追加到 existing_factor_pool.pkl。
"""
import sys, pickle, time, warnings, traceback, os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import pandas as pd, numpy as np
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'research' / 'factor_alchemy'))
from fri import _exec_formula, _cross_section_zscore

POOL_CSV = PROJECT_ROOT / 'data' / 'passed_factor_pool.csv'
CACHE_PATH = PROJECT_ROOT / 'data' / 'cache' / 'existing_factor_pool.pkl'
TIMEOUT_PER_FACTOR = 180  # 给向量化版本更多时间（首次加载数据）

# ═══════════════════════════════════════════════════════════════════
# 8 个因子的向量化重写
# ═══════════════════════════════════════════════════════════════════

FIXED_FORMULAS = {
    # ── 1. scc_network_centrality: 各股与市场均值的滚动相关性 ──
    # 原公式: ret.rolling(60).apply(lambda x: corrcoef(x, ret.iloc[...].mean(axis=1).loc[x.index])...)
    # 向量化: 预计算市场收益, 滚动协方差/(滚动标准差之积)
    "scc_network_centrality": """
ret = close.pct_change().fillna(0)
mkt = ret.mean(axis=1)
w, mp = 60, 30
ret_rm = ret.rolling(w, min_periods=mp).mean()
mkt_rm = mkt.rolling(w, min_periods=mp).mean()
ret_dm = ret.sub(ret_rm)
mkt_dm = mkt.sub(mkt_rm)
# 滚动协方差: sum[(x-μx)(y-μy)]
cov = ret_dm.mul(mkt_dm, axis=0).rolling(w, min_periods=mp).sum()
# 滚动方差
ret_var = (ret_dm ** 2).rolling(w, min_periods=mp).sum()
mkt_var = (mkt_dm ** 2).rolling(w, min_periods=mp).sum()
# 相关系数
denom = np.sqrt(ret_var.values * mkt_var.values[:, np.newaxis] + 1e-10)
result = pd.DataFrame(np.abs(cov.values / denom), index=cov.index, columns=cov.columns)
""",

    # ── 2. downside_beta_asymmetry: 数学等价于 rolling variance ──
    # 原公式: -cov(x, x.mean()-x)[0,1] = -(-var(x)) = var(x)
    # 向量化为纯 rolling var
    "downside_beta_asymmetry": """
ret = close.pct_change().fillna(0)
result = ret.rolling(60, min_periods=30).var()
""",

    # ── 3. earnings_season_vol_div: 波动率变化 / 成交量变化 ──
    # 原始就是向量化的, 但用了 volume_p 变量; 修复: 统一用 volume
    "earnings_season_vol_div": """
ret_1d = close.pct_change().fillna(0)
vol = ret_1d.rolling(20).std()
delta_vol = vol - vol.shift(20)
delta_dv = volume.rolling(20).mean().sub(volume.rolling(20).mean().shift(20))
result = delta_vol / delta_dv.abs().replace(0, np.nan)
""",

    # ── 4. roll_implied_spread_20d: 滞后1阶自协方差的平方根 ──
    # 原公式: apply(lambda x: -(x.iloc[:-1]*x.iloc[1:].values).mean())
    # 向量化: -(ret_dm * ret_dm.shift(1)).rolling(20).mean()
    "roll_implied_spread_20d": """
ret = close.pct_change().fillna(0)
w, mp = 20, 10
ret_rm = ret.rolling(w, min_periods=mp).mean()
ret_dm = ret - ret_rm
ret_lag = ret_dm.shift(1)
autocov = -(ret_dm * ret_lag).rolling(w, min_periods=mp).mean()
result = -np.sqrt(np.abs(-2.0 * autocov.clip(upper=0)))
""",

    # ── 5. tail_asymmetry_kelly_jiang: 正/负尾部90分位差异 ──
    # 原公式: rolling(60).apply(lambda x: np.percentile(x.values, 90), raw=True)
    # 向量化: rolling().quantile(0.90) — pandas 内置 C 实现
    "tail_asymmetry_kelly_jiang": """
ret_1d = close.pct_change().fillna(0)
w, mp = 60, 30
pos_ret = ret_1d.clip(lower=0)
neg_ret = -ret_1d.clip(upper=0)
pos_tail = pos_ret.rolling(w, min_periods=mp).quantile(0.90)
neg_tail = neg_ret.rolling(w, min_periods=mp).quantile(0.90)
result = -(pos_tail - neg_tail.fillna(0))
""",

    # ── 6. earnings_surprise_persistence: 季度动量持续性 ──
    # 原公式: rolling(63).apply(lambda x: x.iloc[-1]-x.iloc[0])
    # 向量化: diff - diff.shift(62)
    "earnings_surprise_persistence": """
ret_q = close.pct_change(63).fillna(0)
ret_q_lag = ret_q.shift(63)
diff = ret_q - ret_q_lag
# diff.rolling(63).apply(lambda x: x[-1]-x[0]) = diff - diff.shift(62)
result = diff - diff.shift(62)
""",

    # ── 7. regime_adaptive_sigma_ratio: Parkinson波动率 / 收盘价波动率 ──
    # 原公式: .apply(np.log) .apply(np.sqrt) — 慢但非致命
    # 向量化: 直接用 np.log() 和 np.sqrt() 操作整个 DataFrame
    "regime_adaptive_sigma_ratio": """
ln_hl = np.log(high / low.replace(0, np.nan))
parkinson_var = (ln_hl ** 2 / (4.0 * np.log(2.0))).rolling(20, min_periods=10).mean()
parkinson_vol = np.sqrt(parkinson_var)
close_vol = close.pct_change().rolling(20, min_periods=10).std()
result = parkinson_vol / close_vol.replace(0, np.nan)
""",

    # ── 8. liquidity_resilience_recovery_rate: 最大回撤/累计收益 ──
    # 原公式: rolling(60).apply(lambda x: x.prod()-1, raw=True)
    # 向量化: log(1+r).rolling().sum() → exp - 1
    "liquidity_resilience_recovery_rate": """
ret_1d = close.pct_change().fillna(0)
w, mp = 60, 30
# 对数累积收益 = sum(log(1+r)) 在窗口内
log_ret = np.log(1.0 + ret_1d.clip(lower=-0.99))
log_cumret = log_ret.rolling(w, min_periods=mp).sum()
cumret = np.exp(log_cumret) - 1.0
# cumret 的滚动最小值 = 窗口内最大回撤的累计收益谷底
max_dd = cumret.rolling(w, min_periods=mp).min()
result = -max_dd / (cumret.abs() + 0.001)
""",
}


def exec_one_factor(name, formula, close, open_df, high_df, low_df, vol_df):
    """单因子执行, 返回 (name, success, result_or_error)"""
    try:
        fv = _exec_formula(formula.strip(), close, open_df, high_df, low_df, vol_df)
        if fv is None or fv.empty:
            return (name, False, "Empty result")
        if fv.shape[1] < 10:
            return (name, False, f"Only {fv.shape[1]} stocks")
        fv = fv.fillna(0.0)
        fv = _cross_section_zscore(fv)
        return (name, True, fv)
    except Exception as e:
        return (name, False, f"{type(e).__name__}: {str(e)[:80]}")


def main():
    print("=" * 70)
    print("修复最后 8 个 reserve 因子 (全部向量化重写)")
    print(f"超时: {TIMEOUT_PER_FACTOR}s/因子")
    print("=" * 70)

    # ── 1. 加载数据 ──────────────────────────────
    print("\n[1/4] 加载数据...", flush=True)
    local_path = PROJECT_ROOT / 'data' / 'raw' / 'daily_prices.csv'
    daily = pd.read_csv(local_path, dtype={'ts_code': 'str', 'trade_date': 'str'},
                        usecols=['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol'])
    daily['trade_date'] = pd.to_datetime(daily['trade_date'], format='mixed')
    daily = daily.drop_duplicates(subset=['ts_code', 'trade_date'])
    daily = daily.sort_values(['ts_code', 'trade_date'])
    end_dt = daily['trade_date'].max()
    start_dt = end_dt - timedelta(days=270)
    daily = daily[(daily['trade_date'] >= pd.Timestamp(start_dt.strftime('%Y-%m-%d'))) &
                  (daily['trade_date'] <= pd.Timestamp(end_dt.strftime('%Y-%m-%d')))]
    close_df = daily.pivot(index='trade_date', columns='ts_code', values='close')
    min_days = 40
    close_df = close_df[close_df.columns[close_df.count() >= min_days]]
    open_df = daily.pivot(index='trade_date', columns='ts_code', values='open').reindex_like(close_df)
    high_df = daily.pivot(index='trade_date', columns='ts_code', values='high').reindex_like(close_df)
    low_df = daily.pivot(index='trade_date', columns='ts_code', values='low').reindex_like(close_df)
    vol_df = daily.pivot(index='trade_date', columns='ts_code', values='vol').reindex_like(close_df)
    print(f"   数据: {len(close_df)} 天 x {len(close_df.columns)} 只股票", flush=True)

    # ── 2. 加载缓存 ──────────────────────────────
    print("\n[2/4] 加载现有缓存...", flush=True)
    if CACHE_PATH.exists():
        with open(CACHE_PATH, 'rb') as fh:
            cache = pickle.load(fh)
    else:
        cache = {'_meta': {}}
    cached_before = sum(1 for k in cache if not k.startswith('_'))
    print(f"   已有 {cached_before} 个因子", flush=True)

    # ── 3. 逐个修复 ──────────────────────────────
    targets = list(FIXED_FORMULAS.keys())
    print(f"\n[3/4] 执行 {len(targets)} 个因子的向量化公式...", flush=True)
    t0 = time.time()
    ok_list = []
    fail_list = []

    for i, name in enumerate(targets):
        formula = FIXED_FORMULAS[name]
        t_start = time.time()

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(exec_one_factor, name, formula,
                                     close_df, open_df, high_df, low_df, vol_df)
            try:
                r_name, r_success, r_result = future.result(timeout=TIMEOUT_PER_FACTOR)
            except FutureTimeoutError:
                r_name, r_success, r_result = name, False, f"Timeout({TIMEOUT_PER_FACTOR}s)"
                future.cancel()
            except Exception as e:
                r_name, r_success, r_result = name, False, f"ThreadError: {e}"

        elapsed_i = time.time() - t_start

        if r_success and isinstance(r_result, pd.DataFrame) and r_result.shape[1] > 10:
            cache[name] = r_result
            ok_list.append((name, r_result.shape[1]))
            tag = f"OK  {r_result.shape[1]}stocks {elapsed_i:.1f}s"
        else:
            fail_list.append((name, str(r_result)))
            tag = f"FAIL  {r_result}"

        print(f"  [{i+1:2d}/{len(targets)}] {name:40s} {tag}", flush=True)

        # checkpoint every 3
        if (i + 1) % 3 == 0 and ok_list:
            cache['_meta'] = {
                'n_factors': cached_before + len(ok_list),
                'built_at': datetime.now().isoformat(),
            }
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(CACHE_PATH, 'wb') as fh:
                pickle.dump(cache, fh)
            print(f"        checkpoint: {cached_before + len(ok_list)} factors", flush=True)

    # ── 4. 最终保存 & 摘要 ─────────────────────────
    total_elapsed = time.time() - t0
    cache['_meta'] = {
        'n_factors': cached_before + len(ok_list),
        'built_at': datetime.now().isoformat(),
    }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, 'wb') as fh:
        pickle.dump(cache, fh)
    total_cached = sum(1 for k in cache if not k.startswith('_'))
    mb = CACHE_PATH.stat().st_size / 1024 / 1024

    print(f"\n{'='*70}")
    print(f"修复结果:")
    print(f"  成功: {len(ok_list)}/{len(targets)}")
    for name, nstocks in ok_list:
        print(f"    OK   {name} ({nstocks} stocks)")
    if fail_list:
        print(f"  失败: {len(fail_list)}/{len(targets)}")
        for name, err in fail_list:
            print(f"    FAIL {name}: {err}")
    print(f"  缓存总计: {total_cached} 因子 ({mb:.0f} MB)")
    print(f"  总耗时: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
    print("=" * 70)

    # 同时更新 CSV 中这 8 个因子的 formula 字段（保存修复后的公式）
    if ok_list:
        df = pd.read_csv(POOL_CSV, encoding='utf-8-sig')
        for name, _ in ok_list:
            mask = df['name'] == name
            if mask.any():
                df.loc[mask, 'formula'] = FIXED_FORMULAS[name].strip()
        df.to_csv(POOL_CSV, index=False, encoding='utf-8-sig')
        print("CSV formula 字段已更新")


if __name__ == '__main__':
    main()
