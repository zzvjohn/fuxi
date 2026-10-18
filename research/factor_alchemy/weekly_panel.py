# -*- coding: utf-8 -*-
"""
周频面板构建器 + 周频公式求值 (Weekly Panel Builder)
================================================================================
P-20260910-008 周频链路修复 — 数据层

背景 (根因):
  v0.7 双通道架构声明「生成层频率中立 × 裁决层 S1 分频 XOR × **执行层 S5/S6/JQ
  唯一周频口径**」(见 weekly_lane.py 头部)。但实现只落在 S1:
    - `weekly_lane.WeeklyLaneJudge` 在**周频宽表**上求值 (rolling(4)=4 周) ✅
    - `s5_joint_filter.S5JointFilter` 只有**日频长表**通道 (rolling(4)=4 日) ❌
  于是 natural_freq=="weekly" 的候选过了 τ_w 之后, S2~S6/S5 全部按日频判同一公式:
  `close.rolling(4)` 从「4 周」变成「4 日」, `pct_change(52)` 从「52 周」变成
  「52 日」。语义被破坏 → 22/22 过 τ_w 的周频种子无一能过 S5。

本模块提供:
  1. `build_weekly_panel(daily)` — 从日频长表重采样出周频宽表面板
     (index=周标签, columns=stock_code), 字段与 WeeklyLaneJudge._wide_env 对齐:
     close/open/high/low/volume/amount/returns/vwap/hl_ratio/
     momentum_12_1/vol_60/log_size (+ `_p` 别名)
  2. `eval_weekly_formula(formula, panel)` — 在周频宽表上 eval, 返回长表
  3. `align_weekly_to_daily(weekly_long, panel)` — 周信号 → 周内最后交易日 →
     前向填充到日频, 供既有「周频调仓 + 日频 P&L」回测直接消费
  4. `weekly_ic(panel, wide_result)` — 周频 Rank IC / ICIR (与 S1 同口径, 对拍用)

口径 (与 scripts/ap_batch_factor_test.py::load_price_weekly 逐条对齐, 已对拍):
  - 周标签 = W-SUN period end (Sunday 00:00), dayofweek==6 → 与
    weekly_prices.parquet 完全一致 (重叠窗口 close 精确相等 98.9%)
  - close=周内最后交易日 close, open=周内首日 open, high/low=周内 max/min,
    volume/amount=周内求和
  - momentum_12_1 = close.pct_change(52); vol_60 = 周收益 rolling(12,min_periods=6).std();
    log_size = log(|close| * clip(amount,1) + 1e-8)

差异 (相对缓存 weekly_prices.parquet, 有意为之):
  - 面板直接来自 **当轮日频面板**, 天然到最新交易日 (缓存止于 2026-06-21,
    本模块覆盖至 2026-08-28/30), S5 的 excess_2026 不再被截断
  - 缓存 vol/amount 为 2026-08-31 重采前的旧值 (中位数偏高 ~14%), 本模块用新值

回滚: 本模块为纯函数库, 不被调用即零影响; 调用方由
  FUXI_S5_WEEKLY_CHANNEL=off 逐层关闭。
"""

from __future__ import annotations

import os
import re
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# 与 WeeklyLaneJudge._wide_env 对齐的字段清单
WEEKLY_FIELDS = [
    "close", "open", "high", "low", "volume", "amount", "returns",
    "vwap", "hl_ratio", "momentum_12_1", "vol_60", "log_size",
]

_RAW_AGGS = {
    "open": "first", "high": "max", "low": "min",
    "close": "last", "volume": "sum", "amount": "sum",
}


# ── 公式清洗 (与 weekly_lane 保持同源; 优先复用其实现) ────────────────

def _clean_formula_local(formula: str) -> Optional[str]:
    """本地兜底实现 — 与 weekly_lane.WeeklyLaneJudge._clean_formula 逐行等价。"""
    clean = (formula or "").strip()
    if not clean or clean.startswith("# TODO"):
        return None
    lines = [ln.strip() for ln in clean.split("\n")
             if ln.strip() and not ln.strip().startswith("#")]
    if not lines:
        return None
    if len(lines) > 1:
        final_line = lines[-1]
        m = re.match(r'^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*(.+)$', final_line)
        if m:
            final_line = m.group(1).strip()
        clean = final_line
    else:
        if "=" in clean and not any(op in clean for op in ["<=", ">=", "==", "!="]):
            parts = clean.split("=", 1)
            if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', parts[0].strip()):
                clean = parts[1].strip()

    _roll_rank_re = re.compile(r'(\.rolling\([^)]*\))\.rank\(pct\s*=\s*True\)')
    _plain_rank_re = re.compile(r'\.rank\(pct\s*=\s*True\)')
    _holders = []
    clean = _roll_rank_re.sub(
        lambda m: _holders.append(m.group(0)) or f'{m.group(1)}.__RRK{len(_holders)-1}__',
        clean)
    clean = _plain_rank_re.sub('.rank(pct=True, axis=1)', clean)
    for _i, _h in enumerate(_holders):
        clean = clean.replace(f'__RRK{_i}__', 'rank(pct=True)')
    return clean


def clean_formula(formula: str) -> Optional[str]:
    """公式清洗 (裸 rank 轴修正 + 赋值剥离)。优先走 weekly_lane, 保证 S1/S5 同语义。"""
    try:
        from weekly_lane import WeeklyLaneJudge  # type: ignore
        return WeeklyLaneJudge._clean_formula(formula)
    except Exception:
        return _clean_formula_local(formula)


# ── 面板构建 ─────────────────────────────────────────────────────────

def build_weekly_panel(
    daily: pd.DataFrame,
    *,
    stock_col: str = "ts_code",
    date_col: str = "trade_date",
    price_start: Optional[str] = None,
    price_end: Optional[str] = None,
    verbose: bool = False,
) -> Dict:
    """日频长表 → 周频宽表面板。

    Parameters
    ----------
    daily : DataFrame
        需含 stock_col / date_col 与 open/high/low/close/volume/amount 中至少 close。
    price_start / price_end : 可选的日期裁剪 (含端点)。

    Returns
    -------
    dict:
      env        : {field: wide DataFrame (index=week_label, columns=stock)}
      universe   : list[str]  股票代码
      weeks      : DatetimeIndex  周标签 (W-SUN)
      week_last_day : Series  week_label → 该周最后交易日 (市场日历)
      fwd_ret    : wide DataFrame  下周收益 (close.pct_change(1).shift(-1))
      fields     : list[str]  实际构建成功的字段
    """
    if daily is None or len(daily) == 0:
        raise ValueError("daily 面板为空")

    if date_col not in daily.columns or stock_col not in daily.columns:
        raise ValueError(f"daily 缺少 {stock_col}/{date_col} 列")

    need = [stock_col, date_col] + [c for c in _RAW_AGGS if c in daily.columns]
    d = daily[need].copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col, stock_col])
    if price_start is not None:
        d = d[d[date_col] >= pd.Timestamp(price_start)]
    if price_end is not None:
        d = d[d[date_col] <= pd.Timestamp(price_end)]
    if len(d) == 0:
        raise ValueError("裁剪后日频面板为空")

    # 去重 (键唯一性保证 groupby agg 语义; S5 _price_df 可能含重复键)
    d = d.drop_duplicates(subset=[stock_col, date_col], keep="last")

    # 周标签 = W-SUN period end (与 weekly_prices.parquet 一致, dayofweek==6)
    d["_week"] = d[date_col].dt.to_period("W").dt.end_time.dt.normalize()

    # 周内按交易日排序 → first/last 语义正确 (open=周首日, close=周末日)
    d = d.sort_values([stock_col, "_week", date_col], kind="mergesort")

    agg_cols = {k: v for k, v in _RAW_AGGS.items() if k in d.columns}
    g = d.groupby([stock_col, "_week"], sort=True).agg(**{
        k: (k, v) for k, v in agg_cols.items()
    }).reset_index()
    if os.environ.get('FUXI_WEEKLY_MISSING_CONTRACT', 'enforce') == 'enforce':
        for field in ('volume', 'amount'):
            if field in agg_cols:
                counts = d.groupby([stock_col, '_week'], sort=True)[field].count().to_numpy()
                g.loc[counts == 0, field] = np.nan

    universe = sorted(g[stock_col].unique().tolist())
    weeks = pd.DatetimeIndex(sorted(g["_week"].unique()))

    def _pivot(col: str) -> Optional[pd.DataFrame]:
        if col not in g.columns:
            return None
        sub = g[[stock_col, "_week", col]].dropna(subset=[col])
        wide = sub.set_index(["_week", stock_col])[col].unstack()
        return wide.reindex(index=weeks, columns=universe)

    close_w = _pivot("close")
    if close_w is None:
        raise ValueError("daily 无 close 列, 无法构建周频面板")

    env: Dict[str, pd.DataFrame] = {}
    env["close"] = close_w
    for f in ("open", "high", "low", "volume", "amount"):
        w = _pivot(f)
        if w is not None:
            env[f] = w

    # 派生字段
    env["returns"] = close_w.pct_change(fill_method=None)
    if "amount" in env and "volume" in env:
        env["vwap"] = env["amount"] / env["volume"].clip(lower=1)
    if "high" in env and "low" in env:
        env["hl_ratio"] = env["high"] / env["low"]
    env["momentum_12_1"] = close_w.pct_change(52, fill_method=None)
    env["vol_60"] = env["returns"].rolling(12, min_periods=6).std()
    if "amount" in env:
        env["log_size"] = np.log(close_w.abs() * env["amount"].clip(lower=1) + 1e-8)

    # `_p` 别名 (与 WeeklyLaneJudge._wide_env 命名对齐)
    for f in list(env.keys()):
        env[f"{f}_p"] = env[f]

    fwd_ret = env["returns"].shift(-1)

    # 市场日历: 周标签 → 该周最后交易日
    week_last_day = (d.groupby("_week")[date_col].max()
                     .reindex(weeks).sort_index())

    if verbose:
        print(f"  [WeeklyPanel] {len(g):,} 行, {len(universe)} 股, "
              f"{len(weeks)} 周 ({weeks.min().date()} ~ {weeks.max().date()})")

    return {
        "env": env,
        "universe": universe,
        "weeks": weeks,
        "week_last_day": week_last_day,
        "fwd_ret": fwd_ret,
        "fields": [f for f in WEEKLY_FIELDS if f in env],
    }


# ── 周频公式求值 ─────────────────────────────────────────────────────

def _eval_context(env: Dict[str, pd.DataFrame]) -> Dict:
    """与 weekly_lane.judge 的 eval 环境逐项对齐。"""
    safe_builtins = {
        'range': range, 'len': len, 'int': int, 'float': float,
        'list': list, 'dict': dict, 'tuple': tuple, 'str': str, 'bool': bool,
        'True': True, 'False': False, 'None': None,
        'abs': abs, 'min': min, 'max': max, 'round': round,
        'sum': sum, 'zip': zip, 'sorted': sorted,
        'print': print, 'isinstance': isinstance,
        'Exception': Exception, 'ValueError': ValueError, 'TypeError': TypeError,
    }
    base_context = {
        "np": np, "pd": pd,
        "abs": np.abs, "sqrt": np.sqrt, "log": np.log,
        "log1p": np.log1p, "exp": np.exp, "sign": np.sign,
        "maximum": np.maximum, "minimum": np.minimum,
        "where": np.where, "clip": np.clip,
        "range": range, "len": len, "int": int, "float": float,
        "list": list, "dict": dict, "tuple": tuple,
    }
    return {"__builtins__": safe_builtins, **base_context, **env}


def eval_weekly_formula(
    formula: str,
    panel: Dict,
) -> Tuple[bool, Optional[pd.DataFrame], Optional[pd.DataFrame], str]:
    """在周频宽表上求值。

    Returns
    -------
    (ok, wide_result, weekly_long, reason)
      wide_result : DataFrame (index=week_label, cols=stock) — IC 计算用
      weekly_long : DataFrame [trade_date, stock_code, factor_value] — 周标签为索引
    """
    clean = clean_formula(formula)
    if not clean:
        return False, None, None, "公式不可解析"

    full_globals = _eval_context(panel["env"])
    import warnings
    warnings.filterwarnings("ignore")
    try:
        result = eval(clean, full_globals, {})
    except Exception as e:  # noqa: BLE001
        return False, None, None, f"周频 eval 失败: {type(e).__name__}: {str(e)[:120]}"

    if isinstance(result, pd.Series):
        result = result.to_frame().reindex(columns=panel["universe"])
    if not isinstance(result, pd.DataFrame):
        return False, None, None, f"周频 eval 结果类型不支持: {type(result).__name__}"

    result = result.replace([np.inf, -np.inf], np.nan)
    long = (result.rename_axis("trade_date")
            .rename_axis("stock_code", axis=1)
            .stack().rename("factor_value").reset_index())
    long = long.dropna(subset=["factor_value"])
    long = long[np.isfinite(long["factor_value"])]
    if len(long) == 0:
        return False, None, None, "周频因子值全 NaN"
    return True, result, long, ""


def weekly_ic(
    panel: Dict,
    wide_result: pd.DataFrame,
    *,
    min_stocks: int = 30,
) -> Dict:
    """周频 Rank IC / ICIR (与 WeeklyLaneJudge.judge 同口径, 用于 S1↔S5 对拍)。"""
    fwd = panel["fwd_ret"]
    cols = wide_result.columns.intersection(fwd.columns)
    a = wide_result[cols]
    b = fwd[cols]
    ics = []
    for d in a.index:
        x, y = a.loc[d], b.loc[d]
        m = x.notna() & y.notna()
        if int(m.sum()) < min_stocks:
            continue
        ic = x[m].rank().corr(y[m].rank())
        if pd.notna(ic):
            ics.append(float(ic))
    if not ics:
        return {"weekly_ic": 0.0, "weekly_icir": 0.0, "weekly_n": 0}
    arr = np.array(ics)
    mean_ic = float(np.nanmean(arr))
    std_ic = float(np.nanstd(arr))
    return {
        "weekly_ic": round(mean_ic, 6),
        "weekly_icir": round(abs(mean_ic / std_ic) if std_ic > 0 else 0.0, 4),
        "weekly_n": len(ics),
    }


# ── 周信号 → 日频对齐 ───────────────────────────────────────────────

def align_weekly_to_daily(
    weekly_long: pd.DataFrame,
    panel: Dict,
    daily_keys: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """周频信号 → 日频长表 (供「周频调仓 + 日频 P&L」回测消费)。

    🔴 关键: 周标签 (Sunday) 必须**先映射到该周最后交易日** (通常周五), 再对齐日频。
    否则 ffill 会让 rebalance 日 (周五) 拿到上一周的信号 → 系统性 1 周滞后。

    返回 DataFrame [ts_code, trade_date, factor_value], 仅含日频交易日。
    """
    if weekly_long is None or len(weekly_long) == 0:
        return pd.DataFrame(columns=["ts_code", "trade_date", "factor_value"])

    wk = weekly_long.rename(columns={"stock_code": "ts_code"}).copy()
    wk["trade_date"] = pd.to_datetime(wk["trade_date"])

    # ① 周标签 → 周内最后交易日
    wld = panel.get("week_last_day")
    if wld is not None and len(wld) > 0:
        wk["trade_date"] = wk["trade_date"].map(wld)
        wk = wk.dropna(subset=["trade_date"])

    wide = wk.pivot_table(index="trade_date", columns="ts_code",
                          values="factor_value", aggfunc="last")

    # ② 日频目标轴
    if daily_keys is not None and len(daily_keys) > 0:
        daily_axis = pd.DatetimeIndex(sorted(
            pd.to_datetime(daily_keys["trade_date"]).dropna().unique()))
        pairs = daily_keys.rename(columns={"ts_code": "ts_code"})[
            ["ts_code", "trade_date"]].drop_duplicates()
        pairs["trade_date"] = pd.to_datetime(pairs["trade_date"])
    else:
        daily_axis = pd.DatetimeIndex(sorted(
            pd.to_datetime(panel["env"]["close"].index).unique()))
        pairs = None

    # ③ 在 (周标签 ∪ 日频) 轴上 ffill, 再取日频截面
    axis = wide.index.union(daily_axis).sort_values()
    if os.environ.get('FUXI_WEEKLY_MISSING_CONTRACT', 'enforce') == 'legacy':
        filled = wide.reindex(axis).ffill().reindex(daily_axis)
    else:
        # A missing observation at the next market week expires the old signal.
        events = (pd.DatetimeIndex(pd.to_datetime(wld.dropna().values)).sort_values().unique()
                  if wld is not None else wide.index)
        wide = wide.reindex(events)
        filled = wide.reindex(daily_axis, method='ffill')

    out = (filled.stack().rename("factor_value").reset_index()
           .rename(columns={"level_0": "trade_date", "level_1": "ts_code"}))
    out["trade_date"] = pd.to_datetime(out["trade_date"])
    out = out.dropna(subset=["factor_value"])

    if pairs is not None:
        out = out.merge(pairs, on=["ts_code", "trade_date"], how="inner")
    return out
