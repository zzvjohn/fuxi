"""
Phase 7 ONLY — 独立 Walk-forward 稳定性验证 (v7.9)
=================================================
目的: 第四次完整 run_fa.py 在 Phase 7 (ProcessPool spawn OOM) 处被杀死。
      GA / Phase 5/5.5/6 结果已在 run_20260714_191646.log 中算出, 仅缺 Phase 7 WF。
      Phase 7 只依赖 factor_dfs / forward_returns / ic_summary, 与 GA 解无关,
      故单独跑本脚本补齐 WF 结果, 避免重跑 4h 流水线。

2026-07-15 修复:
  - walkforward_stability 改用 compute_ic_icir_fast (numpy向量化, 快5-10x),
    根治原慢版逐日期 set() 操作的"卡死"问题。
  - 本脚本加 pickle 缓存: 首次算完 Phase1-3 落盘, 之后 --use-cache 仅跑 WF(~1min),
    避免每次重跑浪费 7min 数据加载。

用法: python phase7_only.py [--use-cache]   (可选 FA_WF_WORKERS=N 控制线程数)
"""
import sys, time, os, json, traceback, pickle
from pathlib import Path
from datetime import datetime

# ★ 静默死亡诊断
import faulthandler as _fh
import signal as _signal
try:
    _fh.enable()
    _fh.register(_signal.SIGTERM)
    _fh.register(_signal.SIGINT)
    # 若进程在 200s 后仍活着(说明 WF 又卡死), dump 所有线程栈到 stderr(→日志) 定位卡点
    _fh.dump_traceback_later(200, exit=False)
except Exception:
    pass

import numpy as np
import pandas as pd

# sys.path (同 run_fa.py: factor_alchemy 的祖父目录 = e:/quant)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import FACTOR_DEFS
from factors import ALL_FACTORS
from data.loader import load_all_data
from evaluation.ic_analysis import compute_ic_summary
from evaluation.walkforward_stability import run_full_walkforward_validation

# ★ 文件日志 (tee): 即使进程被会话重建杀死, 也能从磁盘日志看到死在哪
import sys as _sys
_PHASE7_LOG = Path(__file__).parent / 'output' / 'icir_biased' / 'phase7_progress.log'
try:
    _PHASE7_LOG.parent.mkdir(parents=True, exist_ok=True)
    _flog = open(_PHASE7_LOG, 'a', encoding='utf-8')
except Exception:
    _flog = None
class _Tee:
    def __init__(self, *streams):
        self.streams = list(streams)
    def write(self, data):
        for s in self.streams:
            try:
                s.write(data); s.flush()
            except Exception:
                pass
    def flush(self):
        for s in self.streams:
            try: s.flush()
            except Exception: pass
_sys.stdout = _Tee(_sys.stdout, _flog) if _flog else _sys.stdout
_sys.stderr = _Tee(_sys.stderr, _flog) if _flog else _sys.stderr
def _exc_hook(exc_type, exc, tb):
    import traceback as _tb
    _tb.print_exception(exc_type, exc, tb)
_sys.excepthook = _exc_hook
print(f"[LOG] phase7 进度日志: {_PHASE7_LOG}")

# ★ pickle 缓存 (Phase1-3 结果), 重跑仅 WF
_CACHE = Path(__file__).parent / 'output' / 'icir_biased' / 'phase7_cache.pkl'
_USE_CACHE = ('--use-cache' in _sys.argv) or (os.environ.get('PHASE7_USE_CACHE') == '1')

print("=" * 70)
print("  Phase 7 ONLY — 独立 Walk-forward 稳定性验证 (v7.9)")
print(f"  启动: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  缓存: {'启用(--use-cache)' if _USE_CACHE else '关闭'}  路径={_CACHE}")
print("=" * 70)


def compute_inputs():
    """Phase 1-3: 加载数据 + 因子计算 + ICIR 汇总, 返回 (factor_dfs, forward_returns, ic_summary)"""
    # ---- Phase 1 ----
    print("\n" + "=" * 70)
    print("[Phase 1] 加载数据 (2021-01-01 ~ 2026-06-25)")
    print("=" * 70)
    t1 = time.time()
    data = load_all_data('2021-01-01', '2026-06-25')
    price_data = data['price_data']
    financial_data = data['financial_data']
    valuation_data = data['valuation_data']
    industry_map = data.get('industry_map', {})
    close = price_data.get('close')
    print(f"  价格矩阵: {close.shape}")

    close_weekly = close.resample('W').last()
    close_weekly.index = pd.to_datetime(close_weekly.index)
    forward_returns = close_weekly.shift(-1) / close_weekly - 1
    print(f"  周频收益: {forward_returns.shape} "
          f"({forward_returns.index[0].strftime('%Y-%m-%d')} ~ {forward_returns.index[-1].strftime('%Y-%m-%d')})")
    print(f"  [Phase 1] 耗时: {time.time()-t1:.0f}s")

    # ---- Phase 2 ----
    print("\n" + "=" * 70)
    print("[Phase 2] 因子计算")
    print("=" * 70)
    factor_dfs = {}
    t2 = time.time()
    for name in FACTOR_DEFS:
        if name not in ALL_FACTORS:
            continue
        factor_def = FACTOR_DEFS[name]
        factor_obj = ALL_FACTORS[name]()
        try:
            df = factor_obj.compute(
                price_data=price_data,
                financial_data=financial_data,
                valuation_data=valuation_data,
                industry_map=industry_map,
            )
            if df is not None and not df.empty:
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                df_weekly = df.resample('W').last()
                is_financial = df_weekly.shape[0] <= 10 and factor_def['category'] in \
                    ('profitability', 'value', 'growth')
                if is_financial:
                    df_weekly = df_weekly.reindex(close_weekly.index).ffill().bfill()
                else:
                    common_idx = close_weekly.index.intersection(df_weekly.index)
                    if len(common_idx) > 0:
                        df_weekly = df_weekly.reindex(common_idx)
                factor_dfs[name] = df_weekly
        except Exception as e:
            print(f"  {name:25s} FAIL ({str(e)[:60]})")
    print(f"\n  成功: {len(factor_dfs)}/{len(FACTOR_DEFS)} 个因子 ({time.time()-t2:.0f}s)")

    if len(factor_dfs) < 3:
        print("[FATAL] 有效因子不足3个, 无法继续")
        sys.exit(1)

    # ---- Phase 3 ----
    print("\n" + "=" * 70)
    print("[Phase 3] ICIR 汇总")
    print("=" * 70)
    ic_summary = compute_ic_summary(factor_dfs, forward_returns)
    print(f"  ICIR top5: {list(ic_summary.nlargest(5, 'ICIR').index)}")

    return factor_dfs, forward_returns, ic_summary


# =====================================================
# 加载 / 计算 Phase 1-3
# =====================================================
factor_dfs = forward_returns = ic_summary = None
if _USE_CACHE and _CACHE.exists():
    try:
        with open(_CACHE, 'rb') as f:
            factor_dfs, forward_returns, ic_summary = pickle.load(f)
        print(f"\n[CACHE] 从缓存加载 Phase1-3 ({len(factor_dfs)} 因子), 跳过加载/计算")
    except Exception as e:
        print(f"\n[CACHE] 加载失败, 重新计算: {e}")
        factor_dfs = forward_returns = ic_summary = None

if factor_dfs is None:
    factor_dfs, forward_returns, ic_summary = compute_inputs()
    # 保存缓存 (供下次 --use-cache 秒级重跑)
    try:
        with open(_CACHE, 'wb') as f:
            pickle.dump((factor_dfs, forward_returns, ic_summary), f, protocol=4)
        print(f"\n[缓存] 已保存 Phase1-3 结果: {_CACHE}")
    except Exception as e:
        print(f"\n[WARN] 缓存保存失败: {e}")

# =====================================================
# Phase 7: Walk-forward (ThreadPool, 无 spawn)
# =====================================================
print("\n" + "=" * 70)
print("[Phase 7] Walk-forward 因子稳定性 — ThreadPool (无 spawn)")
print("=" * 70)
t7 = time.time()
wf_results = run_full_walkforward_validation(factor_dfs, forward_returns, ic_summary)
print(f"\n[Phase 7] 总耗时: {time.time()-t7:.0f}s")

# =====================================================
# 保存结果 (与 run_fa.py 同路径/同格式)
# =====================================================
if wf_results:
    wf_summary = {}
    for method, result in wf_results.items():
        if result:
            freq = result.get('factor_freq', {})
            if hasattr(freq, 'most_common'):
                freq = dict(freq)
            wf_summary[method] = {
                'stability_score': result.get('stability_score', 0),
                'verdict': result.get('verdict', 'N/A'),
                'oos_icir_mean': result.get('oos_icir_mean', 0),
                'oos_icir_std': result.get('oos_icir_std', 0),
                'factor_freq': {str(k): int(v) for k, v in freq.items()},
            }
    out_dir = Path(__file__).parent / 'output' / 'icir_biased'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'walkforward_results.json'
    with open(out_path, 'w', encoding='utf-8') as jf:
        json.dump(wf_summary, jf, ensure_ascii=False, indent=2)
    print(f"\n=== PHASE7 DONE ===")
    print(f"结果已保存: {out_path}")
    print(json.dumps(wf_summary, ensure_ascii=False, indent=2))
else:
    print("\n[WARN] Phase 7 无结果")
