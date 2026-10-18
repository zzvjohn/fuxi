# -*- coding: utf-8 -*-
"""M18: 数据更新链新鲜度监控扩展 (fail-closed 闸门), 2026-09-28

背景 (P2-3 = M6/M18 批; 参考 reports/fuxi_alpha_generation_overhaul_20260926.md §M18):
- 已有单个守卫: scripts/weekly_cache_refresh.py --check (仅覆盖 weekly_prices/ohlc 两个 parquet)。
- 历史教训 (静默截断事件):
    · stk_limit 冻结于 2026-04-30 (格式错致更新链断裂, 09-26 才修)
    · sw_daily 停更 (冻结至 09 月中, 09-26 恢复)
    · 000985/000905 指数冻结于 2026-09-09 (09-26 才纳入日更)
    · weekly 缓存冻结 12 周 (P-20260915-008 修)
    · hk_hold 冻结于 2024-12-31 (北向披露规则变更后源停更 — 属「源停更」非「采集故障」)
- 本模块把 --check 式守卫扩展为「通道注册表 + 交易日 lag + 分档报警 + fail-closed 闸门」,
  覆盖: 指数 / 行业 / 涨跌停 / 停牌 / 事件表 (预告/分红/财报) 与既有行情通道。

设计纪律 (对齐项目铁律):
- 只读监控: 默认(--check / --json)绝不写任何数据文件, 仅落盘报告到 data/。
- fail-closed 闸门是可开关的: FUXI_FRESHNESS_GATE = off | shadow(默认) | on。
    · off: 不动作
    · shadow: 跑读数 + 打印「本应阻断 ...」+ 落盘, 返回值不变 (零行为影响)
    · on: 若存在 block 级 stale 通道 ⇒ 调用方 (流水线) 阻断 (raise RuntimeError)
- 「源停更」与「采集停更」分开: retired 通道只报告不判 stale (例如 hk_hold)。
- lag 用交易日口径 (trade_calendar SSE 开市日), 不用自然日。
- 判据: lag<=1 合规 (对齐 M16 验收「数据新鲜度 lag ≤1 交易日」);
        lag>=2 warning; lag>=3 block (fail-closed 候选)。

CLI:
  --check               打印全通道读数 + 分档报警 (exit 0=全新鲜 / 1=有 stale)
  --json [PATH]         额外落盘结构化报告 (默认 data/freshness_guard_report.json)
  --asof YYYY-MM-DD HH:MM   以指定时刻判定「应有数据的最近交易日」(测试/复算用)
  --simulate-stale ch=n 注入模拟 (ch 落后 n 个交易日), 只影响本次读数, 不碰真实文件
  --quiet               精简输出 (供流水线调用)

解释器: C:/Users/Zhong/.workbuddy/binaries/python/envs/default/Scripts/python.exe -X utf8
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time
from bisect import bisect_right
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
CAL_PATH = ROOT / "data" / "raw" / "trade_calendar.csv"
REPORT_PATH = ROOT / "data" / "freshness_guard_report.json"

# ── 期望采集时刻: 本地时间此后当日数据视为「应到」 ──
EXPECTED_CUTOFF_HOUR = 21

# ── 通道注册表 ─────────────────────────────────────────────────────────────
# kind:
#   trading_day  — 日频行情/执行类; 以交易日 lag 判定 (warn/block 为交易日数)
#   weekly_cache — 周频缓存 (parquet); 以周差判定
#   event        — 事件/公告类; 以 mtime 摄入年龄判定 (max_ingest_age_days)
#   on_demand    — 按需更新; 只报告, 不判 stale
#   retired      — 数据源本体停更 (只报告, 附 retired_reason)
# on_stale: block -> 参与 fail-closed 闸门; warn -> 仅报警; info -> 仅记录
CHANNELS: List[Dict[str, Any]] = [
    # ── 日频行情/执行类 ──
    dict(channel="daily_prices", file="data/raw/daily_prices.csv", date_col="trade_date",
         kind="trading_day", warn_lag_td=2, block_lag_td=3, on_stale="block",
         consumers=["ralph_pipeline", "s5_eval", "date_ceil"], note="全市场日行情; S1-S5 基础"),
    dict(channel="daily_basic", file="data/raw/daily_basic.csv", date_col="trade_date",
         kind="trading_day", warn_lag_td=2, block_lag_td=3, on_stale="block",
         consumers=["s1_eval", "factor_ic"], note="估值面板; 因子 IC 计算"),
    dict(channel="adj_factor", file="data/raw/adj_factor_daily.csv", date_col="trade_date",
         kind="trading_day", warn_lag_td=2, block_lag_td=3, on_stale="block",
         consumers=["s5_eval", "tradable_gate"], note="复权因子; 收益对齐"),
    dict(channel="moneyflow", file="data/raw/moneyflow_daily.csv", date_col="trade_date",
         kind="trading_day", warn_lag_td=2, block_lag_td=3, on_stale="block",
         consumers=["ralph_pipeline"], note="资金流族因子输入"),
    dict(channel="stk_limit", file="data/raw/stk_limit.csv", date_col="trade_date",
         kind="trading_day", warn_lag_td=2, block_lag_td=3, on_stale="block",
         consumers=["tradable_gate", "s5_eval"], note="涨跌停价; 可成交性门 (09-26 前曾静默截断)"),
    dict(channel="suspend_d", file="data/raw/suspend_d.csv", date_col="trade_date",
         kind="trading_day", warn_lag_td=2, block_lag_td=3, on_stale="block",
         consumers=["tradable_gate", "s5_eval"], note="停牌表 (曾 04-30 截断)"),
    dict(channel="top_list", file="data/raw/top_list.csv", date_col="trade_date",
         kind="trading_day", warn_lag_td=3, block_lag_td=6, on_stale="warn",
         consumers=["longhu_bonus"], note="龙虎榜; 加分机制 (非核心评价链)"),
    dict(channel="margin_detail", file="data/raw/margin_detail.csv", date_col="trade_date",
         kind="trading_day", warn_lag_td=3, block_lag_td=6, on_stale="warn",
         consumers=["margin_family"], note="两融明细"),
    # ── 指数 / 行业 (M16-① 修复对象) ──
    dict(channel="index_000300", file="data/raw/index_000300_daily.csv", date_col="trade_date",
         kind="trading_day", warn_lag_td=2, block_lag_td=3, on_stale="block",
         consumers=["date_ceil", "s5_eval"], note="沪深300; 基准/评价窗 (曾冻结 09-10)"),
    dict(channel="index_000905", file="data/raw/index_000905_daily.csv", date_col="trade_date",
         kind="trading_day", warn_lag_td=2, block_lag_td=3, on_stale="block",
         consumers=["date_ceil"], note="中证500 (曾冻结 09-09)"),
    dict(channel="index_000985", file="data/raw/index_000985_daily.csv", date_col="trade_date",
         kind="trading_day", warn_lag_td=2, block_lag_td=3, on_stale="block",
         consumers=["date_ceil"], note="中证全指 (曾冻结 09-09)"),
    dict(channel="sw_daily", file="data/raw/sw_daily.csv", date_col="trade_date",
         kind="trading_day", warn_lag_td=2, block_lag_td=3, on_stale="block",
         consumers=["industry_neutral", "s5_eval"], note="申万行业 (曾停更)"),
    # ── 周频缓存 ──
    dict(channel="weekly_cache", file="output/ap_batch/cache/weekly_prices.parquet",
         date_col="trade_date", kind="weekly_cache", warn_lag_wk=2, block_lag_wk=3,
         on_stale="block", consumers=["forge", "weekly_lane", "tau_w_calib"],
         note="Forge/WeeklyLane/τ_w 共用; 曾冻结 12 周 (P-20260915-008)"),
    # ── 事件/公告类 ──
    # 注意: 事件表非按日期追加 (实测 forecast 尾行 ann_date=20260130 但全表 max=20260807;
    # fina_indicator/namechange 同类) ⇒ read_mode="full" 强制全表扫描取 max。
    dict(channel="forecast", file="data/raw/forecast.csv", date_col="ann_date",
         kind="event", max_ingest_age_days=100, on_stale="warn", read_mode="full",
         consumers=["pit_event_lib"], note="业绩预告; 采集器排班 ~90 天间隔"),
    dict(channel="dividend", file="data/raw/kuaidao_gaps/dividend.csv", date_col="ann_date",
         kind="event", max_ingest_age_days=40, on_stale="warn", read_mode="full",
         consumers=["s5_event_ledger"], note="分红全周期 (P2-2b 补采; 采集器 kuaidao_gap_update)"),
    dict(channel="fina_indicator", file="data/raw/fina_indicator.csv", date_col="ann_date",
         kind="on_demand", on_stale="info", read_mode="full",
         consumers=["factor_ic", "s5_eval"], note="财务指标; stage0 日更跳过, 按需批量采集"),
    dict(channel="namechange", file="data/raw/namechange_2023_2026.csv", date_col="ann_date",
         kind="on_demand", on_stale="info", read_mode="full",
         consumers=["kuaidao_gate"], note="更名史; 按需更新"),
    # ── 源停更 (只报告) ──
    dict(channel="hk_hold", file="data/raw/hk_hold.csv", date_col="trade_date",
         kind="retired", on_stale="info",
         retired_reason="北向持股披露规则 2024-08 变更后源停更; 冻结于 2024-12-31",
         consumers=["north_family"], note="北向持股 (源停更)"),
]


# ──────────────────────────────────────────────────────────────────────────
# 基础工具
# ──────────────────────────────────────────────────────────────────────────

def norm_date(value: str) -> Optional[str]:
    """任意形态日期 -> 'YYYYMMDD' (无法解析返回 None)。

    覆盖: '20260924' / '20260924.0' / '2026-09-24' / '2026/09/24'
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None
    if "." in s:
        s = s.split(".")[0]
    s = s.replace("-", "").replace("/", "").strip()
    if not s.isdigit() or len(s) != 8:
        return None
    return s


def load_calendar() -> List[str]:
    """SSE 开市日 (YYYYMMDD) 升序列表。"""
    opens: List[str] = []
    with io.open(CAL_PATH, "r", encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            ex = (row.get("exchange") or "").strip()
            if ex and ex != "SSE":
                continue
            d = norm_date(row.get("cal_date") or "")
            if d and (row.get("is_open") or "").strip() == "1":
                opens.append(d)
    opens = sorted(set(opens))
    return opens


def prev_open_on_or_before(opens: List[str], day: str) -> Optional[str]:
    idx = bisect_right(opens, day)
    return opens[idx - 1] if idx > 0 else None


def expected_open_day(opens: List[str], now: datetime,
                      cutoff_hour: int = EXPECTED_CUTOFF_HOUR) -> Optional[str]:
    """当前时刻「应有数据的最近开市日」。

    cutoff 前: 当日数据尚未到采集时刻 ⇒ 期望 = 上一开市日;
    cutoff 后: 当日 (若开市) 数据应已到 ⇒ 期望 = 当日或最近开市日。
    """
    d = now.strftime("%Y%m%d")
    cand = prev_open_on_or_before(opens, d)
    if cand is None:
        return None
    if cand == d and now.hour < cutoff_hour:
        cand = prev_open_on_or_before(opens, opens[bisect_right(opens, d) - 2])
    return cand


def lag_trading_days(opens: List[str], end_day: str, expected_day: str) -> int:
    """end_day 相对 expected_day 落后的开市日数 (end>=expected 时为 0)。"""
    if end_day >= expected_day:
        return 0
    i_end = bisect_right(opens, end_day)
    i_exp = bisect_right(opens, expected_day)
    return max(0, i_exp - i_end)


# ──────────────────────────────────────────────────────────────────────────
# 通达道读数 (tail 快速读 + full 校准)
# ──────────────────────────────────────────────────────────────────────────

def _header_and_idx(path: Path, date_col: str) -> Optional[int]:
    """读首行取 date_col 的列索引 (0-based)。"""
    with io.open(path, "r", encoding="utf-8-sig", newline="") as f:
        first = f.readline()
    if not first:
        return None
    parts = next(csv.reader([first.rstrip("\r\n")])) if first else []
    try:
        return parts.index(date_col)
    except ValueError:
        return None


def tail_max_date(path: Path, date_col: str,
                  block: int = 262144, max_rounds: int = 4) -> Optional[str]:
    """从文件尾部块解析「最后一条可解析行的日期」。

    前提: 表按时间追加 (投资链常规)。若块内解析不到有效日期, 逐次扩大块。
    仅读到边缘块, 大表 (GB 级) 亚秒完成。
    """
    size = path.stat().st_size
    if size == 0:
        return None
    col_idx = _header_and_idx(path, date_col)
    if col_idx is None:
        return None
    cur_block = block
    for _ in range(max_rounds):
        start = max(0, size - cur_block)
        with io.open(path, "rb") as f:
            f.seek(start)
            chunk = f.read()
        if start > 0:
            nl = chunk.find(b"\n")
            if nl < 0:
                cur_block *= 4
                continue
            chunk = chunk[nl + 1:]
        text = chunk.decode("utf-8-sig", errors="replace")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = [ln for ln in text.split("\n") if ln.strip()]
        for line in reversed(lines):
            try:
                parts = next(csv.reader([line]))
            except Exception:
                continue
            if len(parts) <= col_idx:
                continue
            d = norm_date(parts[col_idx])
            if d:
                return d
        cur_block *= 4
    return None


def full_max_date(path: Path, date_col: str) -> Optional[str]:
    """全表扫描取 max (慢; 校准/对拍用)。pandas 快速路径 + csv 回退。"""
    col_idx = _header_and_idx(path, date_col)
    if col_idx is None:
        return None
    try:
        import pandas as pd
        best: Optional[str] = None
        chunks = pd.read_csv(path, usecols=[date_col], dtype=str,
                             chunksize=1_000_000, low_memory=False)
        for ch in chunks:
            v = ch[date_col].dropna()
            if not len(v):
                continue
            normed = v.map(norm_date).dropna()
            if len(normed):
                m = normed.max()
                if best is None or m > best:
                    best = m
        return best
    except Exception:
        pass
    best: Optional[str] = None
    with io.open(path, "r", encoding="utf-8-sig", newline="") as f:
        rd = csv.reader(f)
        next(rd, None)
        for parts in rd:
            if len(parts) <= col_idx:
                continue
            d = norm_date(parts[col_idx])
            if d and (best is None or d > best):
                best = d
    return best


def channel_max_date(ch: Dict[str, Any], path: Path) -> Optional[str]:
    """按通道注册的 read_mode 读取末端日期。"""
    if ch.get("read_mode") == "full":
        return full_max_date(path, ch["date_col"])
    return tail_max_date(path, ch["date_col"])


def _parquet_max_date(path: Path, date_col: str) -> Optional[str]:
    try:
        import pandas as pd
        d = pd.read_parquet(path, columns=[date_col])
        t = pd.to_datetime(d[date_col], errors="coerce").dropna()
        if not len(t):
            return None
        return t.max().strftime("%Y%m%d")
    except Exception:
        return None


def _mtime(path: Path) -> Optional[datetime]:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime)


# ──────────────────────────────────────────────────────────────────────────
# 单通道判定
# ──────────────────────────────────────────────────────────────────────────

def read_channel(ch: Dict[str, Any], opens: List[str],
                 now: datetime) -> Dict[str, Any]:
    """读取单通道的当前末端/年龄 + 计算 lag 与档位 (不做 gate 汇总)。"""
    path = ROOT / ch["file"]
    out: Dict[str, Any] = {
        "channel": ch["channel"], "file": ch["file"], "kind": ch["kind"],
        "consumers": ch.get("consumers", []), "on_stale": ch.get("on_stale", "warn"),
        "note": ch.get("note", ""),
    }
    if not path.exists():
        out.update(exists=False, level="block", reason="文件缺失",
                   max_date=None, lag=None)
        return out
    out["exists"] = True
    out["bytes"] = path.stat().st_size
    mt = _mtime(path)
    out["mtime"] = mt.strftime("%Y-%m-%d %H:%M") if mt else None
    out["ingest_age_days"] = (now.date() - mt.date()).days if mt else None

    if ch["kind"] == "trading_day":
        md = channel_max_date(ch, path)
        exp = expected_open_day(opens, now)
        lag = lag_trading_days(opens, md, exp) if (md and exp) else None
        out.update(max_date=md, expected_day=exp, lag=lag,
                   lag_unit="trading_day")
        if lag is None:
            out.update(level="warn", reason="日期读取失败")
        elif lag >= ch.get("block_lag_td", 3):
            out.update(level="block", reason=f"落后 {lag} 个交易日 (>= {ch.get('block_lag_td',3)})")
        elif lag >= ch.get("warn_lag_td", 2):
            out.update(level="warn", reason=f"落后 {lag} 个交易日 (>= {ch.get('warn_lag_td',2)})")
        else:
            out.update(level="ok", reason=f"落后 {lag} 个交易日")
    elif ch["kind"] == "weekly_cache":
        md = _parquet_max_date(path, ch["date_col"])
        # 期望: daily_prices 末端的上一个完整周 (与 daily_prices 通道同口径读取)
        dp = ROOT / "data/raw/daily_prices.csv"
        _dp_ch = next((c for c in CHANNELS if c["channel"] == "daily_prices"), None)
        dmax = (channel_max_date(_dp_ch, dp)
                if (_dp_ch is not None and dp.exists()) else None)
        exp_wk = None
        if dmax:
            import datetime as _dt
            d = _dt.datetime.strptime(dmax, "%Y%m%d")
            this_week = d + _dt.timedelta(days=6 - d.weekday())
            exp_wk = (this_week - _dt.timedelta(days=7)).strftime("%Y%m%d")
        lag_w = None
        if md and exp_wk:
            import datetime as _dt
            lag_w = max(0, (_dt.datetime.strptime(exp_wk, "%Y%m%d")
                            - _dt.datetime.strptime(md, "%Y%m%d")).days // 7)
        out.update(max_date=md, expected_day=exp_wk, lag=lag_w, lag_unit="week")
        if lag_w is None:
            out.update(level="warn", reason="周缓存日期读取失败")
        elif lag_w >= ch.get("block_lag_wk", 3):
            out.update(level="block", reason=f"落后 {lag_w} 周 (>= {ch.get('block_lag_wk',3)})")
        elif lag_w >= ch.get("warn_lag_wk", 2):
            out.update(level="warn", reason=f"落后 {lag_w} 周 (>= {ch.get('warn_lag_wk',2)})")
        else:
            out.update(level="ok", reason=f"落后 {lag_w} 周")
    elif ch["kind"] == "event":
        md = channel_max_date(ch, path) if ch.get("date_col") else None
        age = out["ingest_age_days"]
        max_age = ch.get("max_ingest_age_days", 30)
        out.update(max_date=md, max_ingest_age_days=max_age, lag=age,
                   lag_unit="ingest_age_day")
        if age is None:
            out.update(level="warn", reason="mtime 读取失败")
        elif age > max_age * 2:
            out.update(level="block", reason=f"摄入年龄 {age} 天 (> 2x{max_age}, 采集器疑似停跑)")
        elif age > max_age:
            out.update(level="warn", reason=f"摄入年龄 {age} 天 (> {max_age}, 采集器疑似放缓)")
        else:
            out.update(level="ok", reason=f"摄入年龄 {age} 天")
    elif ch["kind"] == "on_demand":
        md = channel_max_date(ch, path) if ch.get("date_col") else None
        out.update(max_date=md, level="info", lag=None,
                   reason="按需更新; 只报告不判 stale")
    elif ch["kind"] == "retired":
        md = channel_max_date(ch, path) if ch.get("date_col") else None
        out.update(max_date=md, level="retired", lag=None,
                   reason=ch.get("retired_reason", "数据源已停更"))
    else:
        out.update(level="warn", reason=f"未知 kind: {ch['kind']}")
    return out


def apply_simulation(readings: List[Dict[str, Any]],
                     sim: Dict[str, int], opens: List[str]) -> List[Dict[str, Any]]:
    """--simulate-stale 注入: 把某通道末端前移 n 个交易日后再判档 (不动真实文件)。"""
    if not sim:
        return readings
    out = []
    for r in readings:
        n = sim.get(r["channel"])
        if not n:
            out.append(r); continue
        r = dict(r)
        # 按交易日前移 n 天
        md = r.get("max_date")
        if md and md in opens:
            i = opens.index(md)
            new_md = opens[max(0, i - n)]
        elif md:
            i = bisect_right(opens, md)
            new_md = opens[max(0, i - 1 - n)]
        else:
            new_md = md
        kind = r["kind"]
        if kind == "trading_day":
            lag = lag_trading_days(opens, new_md, r["expected_day"])
        elif kind == "weekly_cache":
            import datetime as _dt
            a = _dt.datetime.strptime(new_md, "%Y%m%d")
            b = _dt.datetime.strptime(r["expected_day"], "%Y%m%d")
            lag = max(0, (b - a).days // 7)
        else:
            lag = (r.get("lag") or 0) + n
        r.update(max_date=new_md, lag=lag, simulated=True)
        ch = next(c for c in CHANNELS if c["channel"] == r["channel"])
        if kind == "trading_day":
            if lag >= ch.get("block_lag_td", 3):
                r.update(level="block", reason=f"[simulated] 落后 {lag} 个交易日")
            elif lag >= ch.get("warn_lag_td", 2):
                r.update(level="warn", reason=f"[simulated] 落后 {lag} 个交易日")
            else:
                r.update(level="ok", reason=f"[simulated] 落后 {lag} 个交易日")
        elif kind == "weekly_cache":
            if lag >= ch.get("block_lag_wk", 3):
                r.update(level="block", reason=f"[simulated] 落后 {lag} 周")
            elif lag >= ch.get("warn_lag_wk", 2):
                r.update(level="warn", reason=f"[simulated] 落后 {lag} 周")
            else:
                r.update(level="ok", reason=f"[simulated] 落后 {lag} 周")
        out.append(r)
    return out


def evaluate_gate(readings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """fail-closed 闸门判定: block 级 stale + on_stale=block ⇒ 阻断其 consumers。"""
    blocked_ch = [r["channel"] for r in readings
                  if r["level"] == "block" and r["on_stale"] == "block"]
    warn_ch = [r["channel"] for r in readings if r["level"] == "warn"]
    chains: List[str] = []
    for r in readings:
        if r["channel"] in blocked_ch:
            chains.extend(r.get("consumers", []))
    return {
        "blocked": bool(blocked_ch),
        "blocked_channels": blocked_ch,
        "warn_channels": warn_ch,
        "blocked_chains": sorted(set(chains)),
    }


def gate_mode(env: Optional[Dict[str, str]] = None) -> str:
    """FUXI_FRESHNESS_GATE 解析: off | shadow(默认) | on。"""
    e = env if env is not None else os.environ
    m = str(e.get("FUXI_FRESHNESS_GATE", "shadow")).strip().lower()
    return m if m in ("off", "shadow", "on") else "shadow"


# ──────────────────────────────────────────────────────────────────────────
# 报告
# ──────────────────────────────────────────────────────────────────────────

_LEVEL_ICON = {"ok": "✅", "warn": "🟡", "block": "🔴", "info": "ℹ️", "retired": "⚪"}


def run_check(now: Optional[datetime] = None,
              sim: Optional[Dict[str, int]] = None,
              quiet: bool = False) -> Dict[str, Any]:
    now = now or datetime.now()
    opens = load_calendar()
    readings = [read_channel(ch, opens, now) for ch in CHANNELS]
    readings = apply_simulation(readings, sim or {}, opens)
    gate = evaluate_gate(readings)
    report = {
        "schema": "fuxi_freshness_guard/v1",
        "checked_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "expected_open_day": expected_open_day(opens, now),
        "gate_mode": gate_mode(),
        "simulated": bool(sim),
        "readings": readings,
        "gate": gate,
        "summary": {
            "total": len(readings),
            "ok": sum(1 for r in readings if r["level"] == "ok"),
            "warn": len(gate["warn_channels"]),
            "block": len(gate["blocked_channels"]),
            "retired": sum(1 for r in readings if r["level"] == "retired"),
            "info": sum(1 for r in readings if r["level"] == "info"),
        },
    }
    if not quiet:
        print("=" * 74)
        print(f"  数据新鲜度守卫 (M18)   判定时刻 {report['checked_at']}   "
              f"应有数据至 {report['expected_open_day']}   gate={report['gate_mode']}")
        print("=" * 74)
        for r in readings:
            icon = _LEVEL_ICON.get(r["level"], "?")
            lag = r.get("lag")
            lag_s = ("-" if lag is None else str(lag))
            unit = {"trading_day": "td", "week": "wk", "ingest_age_day": "d"}.get(
                r.get("lag_unit", ""), "")
            print(f"  {icon} {r['channel']:16s} 末端={str(r.get('max_date')):>10s} "
                  f"lag={lag_s:>3s}{unit:2s} {r['reason']}")
        print("-" * 74)
        s = report["summary"]
        print(f"  小结: ok={s['ok']} warn={s['warn']} block={s['block']} "
              f"info={s['info']} retired={s['retired']} / 共 {s['total']}")
        if gate["blocked"]:
            print(f"  🔴 fail-closed 候选通道: {', '.join(gate['blocked_channels'])}")
            print(f"     受影响链: {', '.join(gate['blocked_chains'])}")
        print("=" * 74)
    return report


def verify_tail(only: Optional[List[str]] = None) -> int:
    """对所有 tail 模式通道做 full 对拍 (一次性校准; 大表较慢)。"""
    print("=" * 74)
    print("  tail vs full 一致性校验 (候选 tail 通道)")
    print("=" * 74)
    bad: List[str] = []
    for ch in CHANNELS:
        if ch.get("read_mode") == "full":
            continue
        if only and ch["channel"] not in only:
            continue
        p = ROOT / ch["file"]
        if not p.exists() or p.suffix == ".parquet":
            continue
        t0 = time.time()
        t = tail_max_date(p, ch["date_col"])
        f = full_max_date(p, ch["date_col"])
        ok = (t == f)
        print(f"  {'OK  ' if ok else 'FAIL'} {ch['channel']:16s} "
              f"tail={t} full={f} ({time.time() - t0:.1f}s)")
        if not ok:
            bad.append(ch["channel"])
    print("-" * 74)
    if bad:
        print(f"  FAIL: {len(bad)} 个通道 tail 读 != full 读 -> 需 read_mode='full': {bad}")
    else:
        print("  OK: 全部 tail 通道与 full 一致")
    return 1 if bad else 0


def write_report(report: Dict[str, Any], path: Optional[Path] = None) -> Path:
    p = path or REPORT_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(text + "\n")
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description="数据更新链新鲜度守卫 (M18)")
    ap.add_argument("--check", action="store_true", help="打印读数 (默认动作)")
    ap.add_argument("--json", nargs="?", const=str(REPORT_PATH), default=None,
                    help="落盘结构化报告 (默认 data/freshness_guard_report.json)")
    ap.add_argument("--asof", default=None, help="判定时刻 'YYYY-MM-DD HH:MM'")
    ap.add_argument("--simulate-stale", default=None,
                    help="注入模拟 'channel=n' (逗号分隔; 只影响本次读数)")
    ap.add_argument("--quiet", action="store_true", help="精简输出")
    ap.add_argument("--verify-tail", action="store_true",
                    help="tail vs full 全通道对拍 (慢, 校准用)")
    ap.add_argument("--only", default=None, help="--verify-tail 限定通道 (逗号分隔)")
    a = ap.parse_args()

    if a.verify_tail:
        only = [x.strip() for x in a.only.split(",")] if a.only else None
        return verify_tail(only)

    now = None
    if a.asof:
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                now = datetime.strptime(a.asof, fmt); break
            except ValueError:
                continue
        if now is None:
            print(f"无法解析 --asof: {a.asof}"); return 2
    sim: Dict[str, int] = {}
    if a.simulate_stale:
        for item in a.simulate_stale.split(","):
            item = item.strip()
            if not item:
                continue
            k, _, v = item.partition("=")
            sim[k.strip()] = int(v)

    report = run_check(now=now, sim=sim, quiet=a.quiet)
    if a.json is not None:
        p = write_report(report, Path(a.json))
        if not a.quiet:
            print(f"  报告已落盘: {p}")
    # 退出码: 0=全 ok/warn; 1=存在 block 级 (fail-closed 候选); 2=参数错
    return 1 if report["gate"]["blocked"] else 0


if __name__ == "__main__":
    sys.exit(main())
