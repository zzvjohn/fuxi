"""
因子池缓存构建器 — 全量版 (所有有公式的因子)

用法:
    python scripts/build_pool_cache.py             # 全量构建
    python scripts/build_pool_cache.py --force     # 强制重建

首次构建 ~10-15 分钟 (164 因子), 后续秒级加载。
支持增量追加 (factor_injector.py 注入后自动调用)。
"""
import sys
import pickle
import time
import warnings
import threading
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'research' / 'factor_alchemy'))

from fri import _exec_formula, _cross_section_zscore

POOL_CSV = PROJECT_ROOT / 'data' / 'passed_factor_pool.csv'
CACHE_PATH = PROJECT_ROOT / 'data' / 'cache' / 'existing_factor_pool.pkl'
CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

PER_FACTOR_TIMEOUT = 60  # 单因子超时秒数, 超过则跳过


def _save_checkpoint(existing: dict, n_dates: int, n_stocks: int, incremental: bool):
    """保存中间 checkpoint，防超时丢进度"""
    cache = dict(existing)
    cache['_meta'] = {
        'cache_key': {
            'csv_mtime': POOL_CSV.stat().st_mtime if POOL_CSV.exists() else 0,
            'n_dates': n_dates,
            'n_stocks': n_stocks,
        },
        'built_at': datetime.now().isoformat(),
        'n_factors': len(existing),
        'strategy': 'full_all_formulas',
        'incremental': incremental,
        'checkpoint': True,
    }
    try:
        with open(CACHE_PATH, 'wb') as f:
            pickle.dump(cache, f)
    except Exception as e:
        print(f"      ⚠️ checkpoint 写入失败: {e}")


def _exec_with_timeout(formula, close_df, open_df, high_df, low_df, vol_df, timeout=PER_FACTOR_TIMEOUT):
    """在独立线程中执行公式, 超时则返回 None"""
    result = [None]
    exception = [None]
    
    def _run():
        try:
            result[0] = _exec_formula(formula, close_df, open_df, high_df, low_df, vol_df)
        except Exception as e:
            exception[0] = e
    
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout)
    
    if t.is_alive():
        return None, TimeoutError(f"公式执行超时 ({timeout}s)")
    if exception[0]:
        return None, exception[0]
    return result[0], None


def build(incremental: bool = False) -> dict:
    """
    构建全量因子池缓存。
    
    incremental=True 时: 只计算缓存中尚不存在的因子, 追加进去。
    否则: 全量重建。
    """
    # 强制行缓冲, 确保后台运行时有即时输出
    sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
    
    # ── 加载数据 ──────────────────────────────────────────
    t_total = time.time()
    print("=" * 60)
    mode_str = "增量追加" if incremental else "全量构建"
    print(f"因子池缓存 — {mode_str}")
    print("=" * 60)

    local_path = PROJECT_ROOT / 'data' / 'raw' / 'daily_prices.csv'
    today = datetime.now()
    end_dt = pd.Timestamp(today.strftime('%Y-%m-%d'))
    start_dt = pd.Timestamp((today - timedelta(days=270)).strftime('%Y-%m-%d'))

    print("\n[1/3] 加载数据...")
    t_load = time.time()
    daily = pd.read_csv(local_path, dtype={'ts_code': str, 'trade_date': str},
                        usecols=['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol'])
    daily['trade_date'] = pd.to_datetime(daily['trade_date'], format='mixed')
    daily = daily[(daily['trade_date'] >= start_dt) & (daily['trade_date'] <= end_dt)]
    daily = daily.drop_duplicates(subset=['ts_code', 'trade_date'])
    daily = daily.sort_values(['ts_code', 'trade_date'])

    data_frames = {}
    for col in ['close', 'open', 'high', 'low', 'vol']:
        df = daily.pivot(index='trade_date', columns='ts_code', values=col)
        min_days = 40
        valid = df.columns[df.count() >= min_days]
        df = df[valid]
        data_frames[col] = df

    close_df = data_frames['close']
    open_df = data_frames['open']
    high_df = data_frames['high']
    low_df = data_frames['low']
    vol_df = data_frames['vol']
    n_dates, n_stocks = len(close_df), len(close_df.columns)
    print(f"   数据: {n_dates} 天 x {n_stocks} 只股票 (加载耗时 {time.time()-t_load:.1f}s)")

    # ── 读取因子池 ────────────────────────────────────────
    pool_df = pd.read_csv(POOL_CSV, encoding='utf-8-sig')
    pool_df = pool_df[pool_df['formula'].notna() & (pool_df['formula'] != '')]
    total_formulas = len(pool_df)
    print(f"\n[2/3] 因子池: {total_formulas} 个有公式的因子")

    # ── 增量模式: 读取已有缓存, 只计算新因子 ─────────────
    existing = {}
    if incremental and CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, 'rb') as f:
                cached = pickle.load(f)
            existing = {k: v for k, v in cached.items() if not k.startswith('_')}
            print(f"   已有缓存: {len(existing)} 个因子")
        except Exception as e:
            print(f"   缓存读取失败: {e}, 回退到全量构建")

    n_existing = len(existing)
    remaining = pool_df[~pool_df['name'].isin(existing)]
    n_remaining = len(remaining)
    
    if incremental and n_remaining == 0:
        print(f"   已是最新: 无需更新")
        return existing

    print(f"   需处理: {n_remaining} 个新因子 / {total_formulas} 总因子")

    # ── 执行所有待处理因子 (每 20 个 checkpoint) ──────────
    t0 = time.time()
    ok_count = 0
    fail_count = 0
    slow_formulas = []
    CHECKPOINT_EVERY = 20

    for i, (_, row) in enumerate(remaining.iterrows()):
        name = row['name']
        formula = row['formula']
        status = str(row.get('status', '?'))
        
        t_start = time.time()
        try:
            fv, err = _exec_with_timeout(formula, close_df, open_df, high_df, low_df, vol_df)
            dt = time.time() - t_start
            if err is not None:
                fail_count += 1
                ok = f'✗ ({str(err)[:40]})'
            elif fv is not None and not fv.empty and fv.shape[1] > 10:
                existing[name] = _cross_section_zscore(fv)
                ok_count += 1
                ok = '✓'
            else:
                fail_count += 1
                ok = '✗ (no data)'
        except Exception as e:
            dt = time.time() - t_start
            fail_count += 1
            ok = f'✗ ({str(e)[:40]})'
        
        if dt > 5.0:
            slow_formulas.append((name, dt))
        
        # 进度 + ETA
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (n_remaining - i - 1) / rate if rate > 0 else 0
        pct = 100 * (i + 1) / n_remaining
        print(f"  [{i+1:3d}/{n_remaining}] ({pct:5.1f}%) {name:35s} [{status}] {ok}  "
              f"({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

        # 定期 checkpoint (防超时丢进度)
        if (i + 1) % CHECKPOINT_EVERY == 0 and existing:
            _save_checkpoint(existing, n_dates, n_stocks, incremental)
            print(f"      💾 checkpoint @ {len(existing)} factor(s)", flush=True)

    total_elapsed = time.time() - t0
    print(f"\n   完成: +{ok_count} 成功 / +{fail_count} 失败 "
          f"(总耗时 {total_elapsed:.1f}s, {total_elapsed/n_remaining:.1f}s/因子)")

    if slow_formulas:
        print(f"   慢公式 (>5s):")
        for sname, sdt in sorted(slow_formulas, key=lambda x: -x[1])[:5]:
            print(f"      {sname}: {sdt:.1f}s")

    # ── 写入缓存 ──────────────────────────────────────────
    print(f"\n[3/3] 写入缓存 ({len(existing)} 个因子)...")
    cache = dict(existing)
    cache['_meta'] = {
        'cache_key': {
            'csv_mtime': POOL_CSV.stat().st_mtime if POOL_CSV.exists() else 0,
            'n_dates': n_dates,
            'n_stocks': n_stocks,
        },
        'built_at': datetime.now().isoformat(),
        'n_factors': len(existing),
        'strategy': 'full_all_formulas',
        'incremental': incremental,
    }
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(cache, f)

    size_mb = CACHE_PATH.stat().st_size / 1e6
    elapsed_total = time.time() - t_total
    print(f"   已保存: {CACHE_PATH} ({size_mb:.1f} MB)")
    print(f"\n✅ {mode_str}完成! ({elapsed_total:.1f}s, {len(existing)} 个因子)")
    return existing


def append_factors_to_cache(new_factors: dict) -> bool:
    """
    增量追加因子到已有缓存 (不重建已有数据)。
    
    参数:
        new_factors: {factor_name: z-scored DataFrame, ...}
    返回:
        True 如果成功追加
    """
    if not CACHE_PATH.exists():
        print("[append_cache] 缓存不存在, 跳过追加")
        return False

    if not new_factors:
        return False

    try:
        import pickle as _pickle
        with open(CACHE_PATH, 'rb') as f:
            cache = _pickle.load(f)

        added = 0
        for name, df in new_factors.items():
            if name not in cache:
                cache[name] = df
                added += 1

        if added > 0:
            # 更新 meta
            cache['_meta']['n_factors'] = len(cache) - 1  # 减 _meta 自己
            cache['_meta']['last_appended'] = datetime.now().isoformat()
            cache['_meta']['appended_count'] = cache['_meta'].get('appended_count', 0) + added

            with open(CACHE_PATH, 'wb') as f:
                _pickle.dump(cache, f)

            print(f"[append_cache] 已追加 {added} 个因子 → {CACHE_PATH} "
                  f"(总因子数: {cache['_meta']['n_factors']})")

        return True
    except Exception as e:
        print(f"[append_cache] 追加失败: {e}")
        return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="因子池缓存构建器")
    parser.add_argument('--force', action='store_true', help='强制全量重建 (忽略已有缓存)')
    parser.add_argument('--incremental', action='store_true', help='增量追加 (只计算缓存中不存在的因子)')
    args = parser.parse_args()

    if args.force:
        # 删除旧缓存, 全量重建
        if CACHE_PATH.exists():
            CACHE_PATH.unlink()
            print(f"已删除旧缓存: {CACHE_PATH}")
        build(incremental=False)
    elif args.incremental:
        build(incremental=True)
    else:
        # 默认: 有缓存则增量, 无则全量
        if CACHE_PATH.exists():
            build(incremental=True)
        else:
            build(incremental=False)
