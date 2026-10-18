"""
修复失败因子 → 增量追加到因子池缓存。
使用 concurrent.futures 线程超时 (90s/因子) 防止卡死。
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
TIMEOUT_PER_FACTOR = 90  # seconds

# ── 手工修复模板 (语法错误/伪代码 → 有效Python) ──────────────────
FIXED_FORMULAS = {
    "chip_lockup_indicator": """
price_position = (close - close.rolling(20).min()) / (close.rolling(20).max() - close.rolling(20).min() + 0.001)
vol_stability = 1.0 / (volume_p.rolling(20).std() / volume_p.rolling(20).mean().replace(0, np.nan) + 0.001)
vol_ratio = volume_p.rolling(5).mean() / volume_p.rolling(20).mean().replace(0, np.nan)
result = price_position * vol_stability * (vol_ratio / (vol_ratio + 1))
""",
    "roll_implied_spread_20d": """
ret = close.pct_change().fillna(0)
ret_dm = ret.sub(ret.mean(), axis=1)
result = -np.sqrt(np.abs(ret_dm.rolling(20, min_periods=10).apply(
    lambda x: -(x.iloc[:-1].values * x.iloc[1:].values).mean() if len(x)>2 else 0, raw=True).clip(upper=0) * 2.0))
""",
    "scc_network_centrality": """
ret = close.pct_change().fillna(0)
result = ret.rolling(60, min_periods=30).apply(
    lambda x: np.abs(np.corrcoef(x.values, ret.values[x.index[0]:x.index[-1]+1].mean(axis=1))[0,1]) if len(x.dropna())>10 else 0, raw=True)
""",
    "panic_selling": """
ret_1d = close.pct_change().fillna(0)
vol_ma20 = volume_p.rolling(20).mean()
vol_surge = volume_p / vol_ma20.replace(0, np.nan)
result = -np.where(ret_1d.values < 0, vol_surge.values, 0.0)
result = pd.DataFrame(result, index=ret_1d.index, columns=ret_1d.columns)
""",
    "chop_index_proxy": """
ret_20 = close.pct_change(20).fillna(0)
vol_20 = close.pct_change().rolling(20).std()
result = -(vol_20 / (ret_20.abs() + 0.001) - 1)
""",
    "downside_beta_asymmetry": """
ret = close.pct_change().fillna(0)
result = -ret.rolling(60, min_periods=30).apply(
    lambda x: np.cov(x.values, x.values.mean()-x.values)[0,1] if np.std(x.values)>0 else 0, raw=True)
""",
    "operational_efficiency_proxy": """
ret_1d = close.pct_change().fillna(0)
ret_lag = ret_1d.shift(1)
ret_dm = ret_1d - ret_1d.rolling(20).mean()
ret_lag_dm = ret_lag - ret_lag.rolling(20).mean()
cov_xy = (ret_dm * ret_lag_dm).rolling(20).mean()
var_x = (ret_dm ** 2).rolling(20).mean()
result = -(cov_xy / (var_x + 0.001))
""",
    "return_concentration": """
ret_1d = close.pct_change().fillna(0)
sum_pos = ret_1d.clip(lower=0).rolling(20).sum()
sum_abs = ret_1d.abs().rolling(20).sum()
result = -(sum_pos / sum_abs.replace(0, np.nan))
""",
    "vwap_deviation_20d": """
ma20 = close.rolling(20).mean()
std20 = close.rolling(20).std()
result = (close - ma20) / std20.replace(0, np.nan)
""",
    "tail_dependence_market": """
ret = close.pct_change().fillna(0)
neg_ret = ret.clip(upper=0)
result = -(-neg_ret.rolling(60, min_periods=30).mean())
""",
    "momentum_skewness_composite": """
ret_1d = close.pct_change().fillna(0)
ret_cum = (1 + ret_1d).rolling(60, min_periods=30).apply(lambda x: x.prod()-1, raw=True)
ret_std = ret_1d.rolling(60, min_periods=30).std()
ret_mean = ret_1d.rolling(60, min_periods=30).mean()
ret_median = ret_1d.rolling(60, min_periods=30).median()
skew_proxy = (ret_mean - ret_median) / ret_std.replace(0, np.nan)
result = ret_cum * skew_proxy.clip(-2, 2)
""",
    "tail_risk_asymmetry_ratio": """
ret_1d = close.pct_change().fillna(0)
down_var = ret_1d.clip(upper=0).rolling(20, min_periods=10).var()
up_var = ret_1d.clip(lower=0).rolling(20, min_periods=10).var()
result = -(down_var / (up_var + 0.001))
""",
    "up_down_reversal_asymmetry": """
daily_ret = close.pct_change().fillna(0)
pos_ret = daily_ret.clip(lower=0)
neg_ret = daily_ret.clip(upper=0)
rev_pos = pos_ret.shift(1).rolling(20).mean()
rev_neg = -neg_ret.shift(1).rolling(20).mean()
result = -(rev_neg - rev_pos)
""",
    "volume_distribution_entropy": """
upper_ref = np.maximum(open, close)
lower_ref = np.minimum(open, close)
body = (close - open).abs()
total_range = high - low
body_share = body / total_range.replace(0, np.nan)
result = -body_share
""",
    "gap_momentum_regime_blend": """
overnight_ret = (close / close.shift(1)) - 1.0
pos_gap = overnight_ret.clip(lower=0)
neg_gap = (-overnight_ret).clip(lower=0)
gap_diff = pos_gap.rolling(20).mean() - neg_gap.rolling(20).mean()
vol_proxy = volume_p.rolling(20).mean()
gap_rank = gap_diff.rank(pct=True, axis=1)
vol_rank = vol_proxy.rank(pct=True, axis=1)
result = gap_rank * (1.0 - vol_rank)
""",
}


def exec_one_factor(name, formula, close, open_df, high_df, low_df, vol_df):
    """单因子执行，返回 (name, success, result_or_error)"""
    try:
        fv = _exec_formula(formula, close, open_df, high_df, low_df, vol_df)
        if fv is None or fv.empty:
            return (name, False, "Empty")
        if fv.shape[1] < 10:
            return (name, False, f"Only {fv.shape[1]} stocks")
        fv = fv.fillna(0.0)
        fv = _cross_section_zscore(fv)
        return (name, True, fv)
    except Exception as e:
        return (name, False, f"{type(e).__name__}: {str(e)[:60]}")


def main():
    print("=" * 70)
    print("修复失败因子 → 增量追加到因子池缓存")
    print(f"超时: {TIMEOUT_PER_FACTOR}s/因子 | 策略: ThreadPool + 向量化修复")
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

    # ── 2. 加载缓存 & 识别失败因子 ────────────────
    print("\n[2/4] 加载缓存 & 识别失败因子...", flush=True)
    if CACHE_PATH.exists():
        with open(CACHE_PATH, 'rb') as fh:
            cache = pickle.load(fh)
    else:
        cache = {'_meta': {}}
    cached = {k for k in cache if not k.startswith('_')}
    df = pd.read_csv(POOL_CSV, encoding='utf-8-sig')
    has_f = df['formula'].notna() & (df['formula'] != '')
    df = df[has_f]
    failed_list = []
    for _, row in df.iterrows():
        name = row['name']
        if name not in cached:
            raw = str(row['formula'])
            is_fixed = name in FIXED_FORMULAS
            formula = FIXED_FORMULAS[name] if is_fixed else raw
            failed_list.append(
                (name, formula, is_fixed, row['status'], row.get('category', '?')))
    print(f"   缓存: {len(cached)}, 待修复: {len(failed_list)} 个", flush=True)

    # ── 3. 并行修复 (带超时) ────────────────────────
    print(f"\n[3/4] 逐因子修复 & 执行 ({len(failed_list)} 个)...", flush=True)
    t0 = time.time()
    ok_count = 0
    fail_count = 0
    new_cache = dict(cache)

    with ThreadPoolExecutor(max_workers=1) as executor:
        for i, (name, formula, is_fixed, status, cat) in enumerate(failed_list):
            future = executor.submit(exec_one_factor, name, formula, close_df, open_df, high_df, low_df, vol_df)
            try:
                r_name, r_success, r_result = future.result(timeout=TIMEOUT_PER_FACTOR)
            except FutureTimeoutError:
                r_name, r_success, r_result = name, False, f"Timeout({TIMEOUT_PER_FACTOR}s)"
                future.cancel()
            except Exception as e:
                r_name, r_success, r_result = name, False, f"ThreadError: {e}"

            if r_success and isinstance(r_result, pd.DataFrame) and r_result.shape[1] > 10:
                new_cache[name] = r_result
                ok_count += 1
                tag = f"✓ ({r_result.shape[1]}stocks)"
            else:
                fail_count += 1
                tag = f"✗ {r_result}"

            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(failed_list) - i - 1) / rate if rate > 0 else 0
            fix_tag = "[FIXED]" if is_fixed else ""
            print(f"  [{i+1:3d}/{len(failed_list)}] {name:40s} {fix_tag:8s} {tag}", flush=True)

            # 每 10 个 checkpoint
            if (i + 1) % 10 == 0 and ok_count > 0:
                new_cache['_meta'] = {
                    'n_factors': ok_count + len(cached),
                    'built_at': datetime.now().isoformat(),
                }
                with open(CACHE_PATH, 'wb') as fh:
                    pickle.dump(new_cache, fh)
                mb = CACHE_PATH.stat().st_size / 1024 / 1024
                print(f"      💾 checkpoint: {len(cached) + ok_count} 因子 ({mb:.0f}MB)", flush=True)

    # ── 4. 最终保存 ──────────────────────────────
    total_elapsed = time.time() - t0
    new_cache['_meta'] = {
        'n_factors': ok_count + len(cached),
        'built_at': datetime.now().isoformat(),
        'total_attempted': len(cached) + ok_count + fail_count,
    }
    with open(CACHE_PATH, 'wb') as fh:
        pickle.dump(new_cache, fh)
    total_cached = sum(1 for k in new_cache if not k.startswith('_'))
    mb = CACHE_PATH.stat().st_size / 1024 / 1024
    print(f"\n{'='*70}")
    print(f"✅ 修复完成!")
    print(f"  原始缓存: {len(cached)} 因子")
    print(f"  修复成功: {ok_count} 因子")
    print(f"  仍然失败: {fail_count} 因子")
    print(f"  最终缓存: {total_cached} 因子 ({mb:.0f} MB)")
    print(f"  总耗时:   {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")


if __name__ == '__main__':
    main()
