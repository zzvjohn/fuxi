# -*- coding: utf-8 -*-
"""M11 (2026-09-27): S5 固定股数现金账本桥接 — 双口径与对拍.

背景 (reports/fuxi_alpha_generation_overhaul_20260926.md M11 / 证据 E1+E2):
  S5 现行口径 = 周频换成员 / 日频等权收益 (``weighting_contract`` 自述
  "daily_equal_weight_membership_weekly; not a fixed-share NAV ledger")。
  固定股数现金账本 ``portfolio_execution_ledger`` 已存在且经 A1 三臂封存
  对拍 (24 情景 7,680 账户日, 独立 Decimal 最大差 < 1e-7 元), 但完全隔离
  (账本自述 "Not wired to S5 or promotion")。

本模块把账本接入 S5 作为**可选正式口径**:
  ``FUXI_S5_LEDGER = proxy (默认, 100% 历史行为) | fixed_shares``

三件套:
  ``proxy_book``        — 等权周换仓代理口径 (复刻 S5 backtest_factor 的
                          enforce 语义, 供对拍与自查; 不用于生产替换)
  ``fixed_shares_book`` — 固定股数现金账本 (fail-closed: 未闭合 ⇒
                          data_insufficient + 缺数可枚举; 未决陈旧 mark 不授 valid)
  ``compare_books``     — 双口径对拍 (逐日 NAV 差 / 费用差 / 五态分类)

三个带窗装载器 (生产 wiring 用; 均为只读, 失败返回空 dict 由调用方披露):
  ``load_adj_checks`` / ``load_limits`` / ``load_halt_states``

口径纪律:
  · proxy 的日收益 = 官方 ``close/pre_close - 1`` (pre_close 已含除权调整,
    与 S5 ``close_p/pre_close_p`` 同构; ``*_p`` 在 S5 加载层只是原列别名);
  · 缺失报价**不缩分母、不补零** — 当日整组记 None (等价 S5 min_count=top_n);
  · adj 守卫: |ret| > 板块幅度 + 0.5pp ⇒ 丢弃 (与 S5 一致, 挡新股/恢复上市);
  · 成本口径: proxy = 换手名义 × bps (持有期首个有效日扣除);
    账本 = 每笔成交 notional × bps (逐笔, 含卖出腿);
  · 五态: 账本 complete ⇒ valid; 未闭合 ⇒ data_insufficient (reasons 可枚举);
    异常输入/运行 ⇒ error。任何读数不得被包装为 S5 通过或 JQ 资格。
"""
from __future__ import annotations

import math
import os
from collections import Counter
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

LEDGER_MODES = ("proxy", "fixed_shares")
_COMPLETE = "book_complete_under_declared_assumptions"


def ledger_mode() -> str:
    """S5 账本口径开关。非法值 fail-fast (不静默回落)。

    ``proxy``        — 历史等权代理口径 (默认; 与改动前逐位一致)
    ``fixed_shares`` — 固定股数现金账本 (实验口径, 未闭合即 data_insufficient)
    """
    raw = (os.environ.get("FUXI_S5_LEDGER", "proxy") or "proxy").strip().lower()
    if raw not in LEDGER_MODES:
        raise ValueError(
            "FUXI_S5_LEDGER must be proxy|fixed_shares, got %r" % raw)
    return raw


def board_limit(code) -> float:
    """按 ts_code 前缀返回涨跌停幅度 (小数; 与 S5 _board_limit_vector 一致)。

    主板 10% / 创业板·科创板 (30x·68x) 20% / 北交所 (.BJ 或 4·8 开头) 30%。
    """
    s = str(code)
    if s.endswith(".BJ") or s[:1] in ("4", "8"):
        return 0.30
    if s[:2] in ("30", "68"):
        return 0.20
    return 0.10


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


# ═══════════════════════════════════════════════════════════════
# 代理口径 (复刻 S5 backtest_factor(enforce) 语义)
# ═══════════════════════════════════════════════════════════════

def proxy_book(days, bars, targets, *, top_n, cost_bps=15.0,
               entry_mode="signal_close"):
    """等权周换仓代理口径 —— 供对拍, 不用于生产替换。

    targets[signal_date] = {code: weight} (仅取成员集合; 权重按 1/top_n)。
    持有期 = (signal_date, next_signal_date]; 日收益 = close/pre_close - 1
    (缺失/越界 ⇒ None, 分母固定 top_n 不缩); 成本 = 换手 × bps, 持有期首个
    有效收益日一次性扣除。返回 dict(status/nav_by_day/daily_returns/...)。

    ``entry_mode`` (时点分解用; 不改变默认口径):
      ``signal_close`` (默认) — 复刻 S5 adj 口径切片: 首个持有日 = 建仓日次日
        全天 close/pre_close-1 (等价"信号收盘建仓", 含隔夜跳空);
      ``next_open`` — 首日按"次交易日**开盘**建仓"计 (close/open-1), 用于把
        时点差从总差中分离; **不建模**递延/涨跌停检查, 仅时点变体。
    """
    if entry_mode not in ("signal_close", "next_open"):
        raise ValueError("entry_mode must be signal_close|next_open, got %r"
                         % entry_mode)
    dates = list(days)
    order = sorted(d for d in targets if d in set(dates))
    rets = []          # [(date, ret_or_None)]
    total_cost = 0.0
    turns = []
    prev_codes = None
    n_periods = 0
    for k in range(len(order) - 1):
        entry, exit_ = order[k], order[k + 1]
        codes = sorted(targets[entry].keys())
        held_days = [d for d in dates if entry < d <= exit_]
        daily = []
        for j, d in enumerate(held_days):
            vals = []
            for c in codes:
                bar = bars.get((d, c), {})
                close, pre = bar.get("close"), bar.get("pre_close")
                ref = pre
                if entry_mode == "next_open" and j == 0:
                    ref = bar.get("open")    # 首日 = 开盘建仓 (账本时点)
                if _finite(close) and _finite(ref) and close > 0 and ref > 0:
                    r = close / ref - 1.0
                    if abs(r) > board_limit(c) + 0.005:   # adj 守卫 (同 S5)
                        r = None
                else:
                    r = None
                vals.append(r)
            n_ok = sum(1 for v in vals if v is not None)
            if n_ok >= top_n and top_n > 0:
                daily.append((d, sum(v for v in vals if v is not None) / top_n))
            else:
                daily.append((d, None))       # 缺报价 ⇒ 整组不产出 (不缩分母)
        if prev_codes is None:
            turn = 1.0
        else:
            cur, prev = set(codes), set(prev_codes)
            turn = (len(prev - cur) + len(cur - prev)) / float(top_n)
        cost = turn * cost_bps / 1e4
        total_cost += cost
        turns.append(turn)
        n_periods += 1
        first = next((i for i, (_d, v) in enumerate(daily) if v is not None), None)
        if first is not None:
            d0, v0 = daily[first]
            daily[first] = (d0, v0 - cost)
        rets.extend(daily)
        prev_codes = codes

    nav, nav_by_day = 1.0, {}
    for d, v in sorted(rets):
        if v is not None:
            nav *= (1.0 + v)
        nav_by_day[d] = nav
    usable = sum(1 for _d, v in rets if v is not None)
    status = "proxy_ok" if (n_periods > 0 and usable > 0) else "data_insufficient"
    return {
        "status": status,
        "weighting_contract": "daily_equal_weight_membership_weekly",
        "entry_mode": entry_mode,
        "n_periods": n_periods,
        "n_usable_day_returns": usable,
        "n_target_days": len(order),
        "total_cost_frac": total_cost,
        "turnover": turns,
        "daily_returns": rets,
        "nav_by_day": nav_by_day,
        "final_nav": nav,
        "formal_eligibility": False,
    }


# ═══════════════════════════════════════════════════════════════
# 账本口径 (fail-closed)
# ═══════════════════════════════════════════════════════════════

def fixed_shares_book(days, bars, targets, *, initial_cash=10_000_000.0,
                      cost_bps=15.0, rules=None, actions=(),
                      delist_evidence=None, share_exchanges=(),
                      fractional_share_policy="block",
                      gross_cash_tax_rate=0.0, delist_mark_fraction=1.0,
                      missing_limit_policy="defer",
                      halted_session_check_policy="unavailable",
                      unresolved_action_policy="block",
                      reference_only_events=(),
                      include_ledger=False):
    """固定股数现金账本 (portfolio_execution_ledger, fail-closed)。

    未闭合 ⇒ status=data_insufficient, reasons/missing 全部可枚举;
    未决陈旧 mark 不授 valid (账本自身的不变量)。
    ``missing_limit_policy="defer"`` 为**声明策略**: 无涨跌停证据的订单
    递延次日 (订单留痕、可枚举), 而非凭推导价格成交。
    """
    try:
        from portfolio_execution_ledger import (assess_economic_book,
                                                run_ledger)
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "reasons": ["import_failed"],
                "error": f"{type(exc).__name__}: {exc}"[:240],
                "weighting_contract": "fixed_share_cash_ledger",
                "formal_eligibility": False}

    try:
        ledger = run_ledger(
            days, bars, targets, initial_cash=initial_cash, cost_bps=cost_bps,
            rules=rules or {}, actions=actions,
            fractional_share_policy=fractional_share_policy,
            delist_evidence=delist_evidence,
            missing_limit_policy=missing_limit_policy,
            unresolved_action_policy=unresolved_action_policy,
            halted_session_check_policy=halted_session_check_policy,
            reference_only_events=reference_only_events,
            gross_cash_tax_rate=gross_cash_tax_rate,
            delist_mark_fraction=delist_mark_fraction,
            share_exchanges=share_exchanges)
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "reasons": ["ledger_exception"],
                "error": f"{type(exc).__name__}: {exc}"[:240],
                "weighting_contract": "fixed_share_cash_ledger",
                "formal_eligibility": False}

    assessment = assess_economic_book(ledger, days, initial_cash)
    complete = (assessment["status"] == _COMPLETE
                and all(math.isfinite(row.get("nav", math.nan))
                        for row in ledger.days))
    reasons = list(assessment.get("reasons") or [])
    if not complete and not reasons:
        reasons = ["ledger_incomplete"]

    total_fees = sum(float(e.get("fee") or 0.0) for e in ledger.journal
                     if e.get("kind") == "fill")
    n_fills = sum(1 for e in ledger.journal if e.get("kind") == "fill")
    unfilled = sum(1 for o in ledger.orders if not o.get("filled"))
    missing = {
        "n_errors": len(ledger.errors),
        "error_counts": dict(Counter(e.get("reason") for e in ledger.errors)),
        "error_samples": ledger.errors[:12],
        "n_excluded_security_days": len(ledger.excluded),
        "n_unfilled_orders": unfilled,
        "order_reasons": dict(Counter(
            o.get("reason") for o in ledger.orders if o.get("reason"))),
        "n_collapsed_events": len(ledger.collapsed_events),
    }

    out = {
        "status": _COMPLETE if complete else "data_insufficient",
        "assessment": assessment,
        "reasons": reasons,
        "n_days": len(ledger.days),
        "initial_cash": float(initial_cash),
        "final_nav": float(ledger.days[-1]["nav"]) if ledger.days else None,
        "net_return": (float(ledger.days[-1]["nav"]) / float(initial_cash) - 1.0
                       if complete and ledger.days else None),
        "total_fees": total_fees,
        "total_fees_frac": total_fees / float(initial_cash),
        "n_fills": n_fills,
        "n_orders": len(ledger.orders),
        "missing": missing,
        "weighting_contract": "fixed_share_cash_ledger_under_declared_assumptions",
        "formal_eligibility": False,
    }
    if complete:
        rets, prev = [], float(initial_cash)
        for row in ledger.days:
            nav = float(row["nav"])
            rets.append((row["date"], (nav / prev - 1.0) if prev else None))
            prev = nav
        out["daily_returns"] = rets
        out["nav_by_day"] = {row["date"]: float(row["nav"]) / float(initial_cash)
                             for row in ledger.days}
    else:
        out["daily_returns"] = None
        out["nav_by_day"] = None
    if include_ledger:
        out["ledger_object"] = ledger
    return out


# ═══════════════════════════════════════════════════════════════
# 双口径对拍
# ═══════════════════════════════════════════════════════════════

def classify_five_state(book: dict) -> str:
    """五态分类 (valid / data_insufficient / error); 未产出即不可裁决。"""
    st = book.get("status")
    if st == _COMPLETE or st == "proxy_ok":
        return "valid"
    if st == "error":
        return "error"
    return "data_insufficient"


def compare_books(proxy: dict, ledger: dict, *, label: str = ""):
    """双口径对拍 —— 逐日 NAV 差 / 费用差 / 五态分类。"""
    out = {
        "label": label,
        "proxy_status": proxy.get("status"),
        "ledger_status": ledger.get("status"),
        "five_state": {
            "proxy": classify_five_state(proxy),
            "ledger": classify_five_state(ledger),
        },
        "ledger_reasons": list(ledger.get("reasons") or []),
        "proxy_weighting_contract": proxy.get("weighting_contract"),
        "ledger_weighting_contract": ledger.get("weighting_contract"),
    }
    pnav = proxy.get("nav_by_day") or {}
    lnav = ledger.get("nav_by_day") or {}
    common = sorted(set(pnav) & set(lnav))
    if common:
        diffs = [pnav[d] - lnav[d] for d in common]
        absd = [abs(x) for x in diffs]
        out.update({
            "n_days_compared": len(common),
            "first_common_day": common[0],
            "last_common_day": common[-1],
            "max_abs_nav_diff": max(absd),
            "mean_abs_nav_diff": sum(absd) / len(absd),
            "final_nav_diff": diffs[-1],
            "final_nav_proxy": pnav[common[-1]],
            "final_nav_ledger": lnav[common[-1]],
        })
    else:
        out.update({"n_days_compared": 0})
    pc = proxy.get("total_cost_frac")
    lc = ledger.get("total_fees_frac")
    out["cost"] = {
        "proxy_total_cost_frac": pc,
        "ledger_total_fees_frac": lc,
        "diff_frac": (pc - lc) if (pc is not None and lc is not None) else None,
        "proxy_cost_model": "turnover_notional_x_bps",
        "ledger_cost_model": "per_fill_notional_x_bps",
    }
    return out


# ═══════════════════════════════════════════════════════════════
# 带窗装载器 (生产 wiring 用; 只读; 失败返回空 dict 由调用方披露)
# ═══════════════════════════════════════════════════════════════

def load_adj_checks(codes, days, *, raw_dir=None):
    """data/raw/adj_factor_daily.csv → {(day, code): (available, required)}。

    与 A1 strict probe 同口径: available = 前后两行 adj_factor 均有效;
    required = 相邻会话 adj_factor 变化 (存在除权事件, 未供 action 即阻断)。
    """
    import pandas as pd  # 局部导入: 生产 S5 已依赖 pandas
    root = Path(raw_dir) if raw_dir else RAW_DIR
    path = root / "adj_factor_daily.csv"
    codes = set(codes)
    lo, hi = min(days), max(days)
    parts = []
    for piece in pd.read_csv(path, usecols=["ts_code", "trade_date", "adj_factor"],
                             dtype={"trade_date": str}, chunksize=500_000):
        piece = piece[piece.ts_code.isin(codes)]
        if not piece.empty:
            parts.append(piece)
    if not parts:
        return {}
    frame = pd.concat(parts, ignore_index=True)
    frame = frame.sort_values(["ts_code", "trade_date"])
    frame["previous"] = frame.groupby("ts_code").adj_factor.shift()
    out = {}
    for r in frame.itertuples():
        d = r.trade_date
        if not (lo <= d <= hi):
            continue
        ok = bool(pd.notna(r.previous) and r.previous > 0 and r.adj_factor > 0)
        req = bool(ok and not math.isclose(float(r.previous), float(r.adj_factor),
                                           rel_tol=1e-7, abs_tol=0.0))
        out[(d, r.ts_code)] = (ok, req)
    return out


def load_limits(codes, days, *, raw_dir=None):
    """data/raw/stk_limit.csv → {(day, code): {"up_limit": x, "down_limit": y}}。"""
    import pandas as pd
    root = Path(raw_dir) if raw_dir else RAW_DIR
    path = root / "stk_limit.csv"
    codes = set(codes)
    lo, hi = min(days), max(days)
    parts = []
    for piece in pd.read_csv(path, usecols=["ts_code", "trade_date", "up_limit", "down_limit"],
                             dtype={"trade_date": str}, chunksize=500_000):
        piece = piece[piece.ts_code.isin(codes)]
        if not piece.empty:
            piece["trade_date"] = pd.to_datetime(piece.trade_date, format="mixed")
            piece = piece[piece.trade_date.between(lo, hi)]
            if not piece.empty:
                parts.append(piece)
    if not parts:
        return {}
    frame = pd.concat(parts, ignore_index=True)
    out = {}
    for r in frame.itertuples():
        out[(r.trade_date.strftime("%Y-%m-%d"), r.ts_code)] = {
            "up_limit": float(r.up_limit), "down_limit": float(r.down_limit)}
    return out


def load_halt_states(codes, days, *, raw_dir=None):
    """data/raw/suspend_d.csv → {(day, code): "full"|"partial"} (仅 suspend_type=S)。"""
    import pandas as pd
    root = Path(raw_dir) if raw_dir else RAW_DIR
    path = root / "suspend_d.csv"
    codes = set(codes)
    lo, hi = min(days), max(days)
    frame = pd.read_csv(path, dtype=str).fillna("")
    frame = frame[frame.ts_code.isin(codes)]
    if frame.empty:
        return {}
    frame["date"] = pd.to_datetime(frame.trade_date, format="mixed").dt.strftime("%Y-%m-%d")
    frame = frame[(frame.date >= lo) & (frame.date <= hi)]
    frame = frame[frame.suspend_type == "S"]
    out = {}
    for r in frame.itertuples():
        out[(r.date, r.ts_code)] = "full" if not r.suspend_timing else "partial"
    return out


# ═══════════════════════════════════════════════════════════════
# 生产 wiring: 完整 bars 契约装载 (与 A1 strict probe load_bars 同构)
# ═══════════════════════════════════════════════════════════════

def load_ledger_bars(codes, start, end, *, raw_dir=None):
    """data/raw 四表 → ``(days, bars, coverage)``, 供 S5 账本路径使用。

    与 A1 strict probe ``load_bars`` 同源同过滤 (东财同批 CSV):
      · 价格 = daily_prices.csv (open/close/pre_close 未复权真实价);
      · 涨跌停 = stk_limit.csv → up_limit/down_limit;
      · 事件检查 = adj_factor_daily.csv 相邻两行均有效 ⇒ available,
        相邻 adj_factor 变化 ⇒ required (除权未供 action 即阻断);
      · 停牌 = suspend_d.csv (S 类): 全日 ⇒ halted(+evidence),
        半日 ⇒ partial_halt; 有报价且在停牌表 ⇒ state=unknown (冲突, 账本 fail-closed)。
    ``bars.get((day, code))`` 缺失返回 ``{"state": "unknown"}`` (默认拒绝侧)。
    窗口 [start, end] 为 ISO 字符串; 日历由调用方与 S5 实际截面窗口对齐。
    """
    import pandas as pd
    root = Path(raw_dir) if raw_dir else RAW_DIR
    codes = set(str(c) for c in codes)
    lo, hi = str(start), str(end)

    def _panel(name, fields):
        parts = []
        for piece in pd.read_csv(root / name,
                                 usecols=["ts_code", "trade_date", *fields],
                                 dtype={"trade_date": str}, chunksize=500_000):
            piece = piece[piece.ts_code.isin(codes)]
            if not piece.empty:
                piece = piece.copy()
                piece["date"] = (pd.to_datetime(piece.trade_date, format="mixed")
                                 .dt.strftime("%Y-%m-%d"))
                parts.append(piece[piece.date.between(lo, hi)])
        if not parts:
            return pd.DataFrame(columns=["date", "ts_code", *fields])
        frame = pd.concat(parts, ignore_index=True)
        if frame.duplicated(["date", "ts_code"]).any():
            raise RuntimeError("duplicate source keys: " + name)
        return frame

    prices = _panel("daily_prices.csv", ["open", "close", "pre_close"])
    quote = {(r.date, r.ts_code): {"open": r.open, "close": r.close,
                                   "pre_close": r.pre_close}
             for r in prices.itertuples()}
    days = sorted(prices.date.unique().tolist())

    limits = _panel("stk_limit.csv", ["up_limit", "down_limit"])
    boundaries = {(r.date, r.ts_code): {"up_limit": r.up_limit,
                                        "down_limit": r.down_limit}
                  for r in limits.itertuples()}

    factors = _panel("adj_factor_daily.csv", ["adj_factor"])
    if len(factors):
        factors = factors.sort_values(["ts_code", "date"])
        factors["previous"] = factors.groupby("ts_code").adj_factor.shift()
        checks = {(r.date, r.ts_code):
                  (bool(pd.notna(r.previous) and r.previous > 0 and r.adj_factor > 0),
                   bool(pd.notna(r.previous) and r.previous > 0 and r.adj_factor > 0
                        and not math.isclose(float(r.previous), float(r.adj_factor),
                                             rel_tol=1e-7, abs_tol=0.0)))
                  for r in factors.itertuples()}
    else:
        checks = {}

    susp = pd.read_csv(root / "suspend_d.csv", dtype=str).fillna("")
    susp = susp[susp.ts_code.isin(codes)].copy()
    if len(susp):
        susp["date"] = (pd.to_datetime(susp.trade_date, format="mixed")
                        .dt.strftime("%Y-%m-%d"))
        susp = susp[(susp.date >= lo) & (susp.date <= hi)]
    full = {(r.date, r.ts_code) for r in susp.itertuples()
            if r.suspend_type == "S" and not r.suspend_timing}
    partial = {(r.date, r.ts_code) for r in susp.itertuples()
               if r.suspend_type == "S" and r.suspend_timing}

    class Bars:
        def get(self, key, default=None):
            available, required = checks.get(key, (False, False))
            if key in quote:
                state = ("unknown" if key in full else
                         "partial_halt" if key in partial else "trading")
                return {**quote[key], **boundaries.get(key, {}), "state": state,
                        "action_check_available": available,
                        "action_required": required}
            if key in full:
                return {"state": "halted",
                        "evidence": "suspend_d.csv/full-day-S",
                        "action_check_available": available,
                        "action_required": required}
            return default if default is not None else {"state": "unknown"}

    coverage = {"quote_keys": len(quote), "limit_keys": len(boundaries),
                "adjustment_checks": len(checks), "full_halt_keys": len(full)}
    return days, Bars(), coverage
