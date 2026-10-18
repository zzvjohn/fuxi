# -*- coding: utf-8 -*-
"""P-20260918-002 (收缩版) 成本敏感性影子诊断。

🔴🔴 用途边界 (硬约束, 修改本文件前必读)
==========================================
本模块**只输出标量读数**, 且**只用于成本敏感性观测**。
严禁用于:
    × 因子准入 / S1·S2·S5 裁决输入
    × 候选改写 / 因子中性化 / 生成器上下文注入
    × 取代 S5 内已有的 P0-g 成本计费与 TRADABLE_GATE
理由: 本模块用的是**本地 5 日 fwd 代理面板**, 不是 S5 引擎 ⇒
      绝对收益不可与 S5/JQ 读数比较, **只有跨成本档位的梯度**有意义。

为何是「收缩版」(2026-09-18 用户钦定)
=====================================
提案原文称「交易门(换手/冲击成本)无显式硬门禁」。逐行核对实现后发现该表述
**高估了缺口** —— 以下均已存在, 本模块**不重建**:
  1. 换手已实现: s5_joint_filter.py L2269-2280 P0-g 每期按
     (退出腿+进入腿)/top_n 计换手, 首期全额 1.0, 落盘 avg_turnover/total_cost。
  2. 交易成本已计费: _s5_cost_bps() 默认 FUXI_S5_COST_BPS=15 (v0.7.26 翻默认)。
  3. 可成交性已闸: TRADABLE_GATE (v0.7.27 与 ADJ_PRICE 捆绑)。
  4. 逻辑门已有雏形: SemanticVerifier.verify 已输出 scores.economic_logic + h2e_pass。
⇒ 真实残余缺口只剩三项, 本模块**只做第 2 项**(实证支撑最强):
  (a) 换手仅 S5 末期、组合级, S1/S2 层零读数  -> 本模块顺带输出 turnover 标注
      (仅标注, 不作门槛 —— 换手是组合构造的产物, 随 top_n/调仓频率/TIE_RULE 变化)
  (b) 成本为「已扣除」单档 15bp, **无档位敏感性**  <== 本模块主体
  (c) 容量仅 qlib 侧一次性实验, 伏羲管线内零容量维度  -> 本次**不做**
      (个人资金 1000 万下 capacity 1319 亿 = 1319x, 投产比差, 留待未来机构规模)

实证依据 (库内已存在, 非外部孤证)
==================================
research/qlib_daily/p4_cost_capacity_report.json 的 cost 敏感性扫描:
  日频 cost x1 / x2 / x3 -> 年化 18.95% / 9.51% / 0.82%   (turnover 33%)
  周频 cost x1 / x2 / x3 -> 年化 10.72% / 8.79% / 6.91%   (turnover 7.1%)
⇒ 日频高换手对成本近乎脆断; 周频稳健。这正是本模块要提前标出的性质。
  成本口径 open_cost=0.0005*mult / close_cost=0.0015*mult / min_cost=5。

换手口径与 S5 对齐 (可比性)
===========================
本模块 turnover = (len(prev - cur) + len(cur - prev)) / top_n
与 s5_joint_filter.py P0-g 完全同式, 避免两套口径静默分叉。

开关
====
env `FUXI_TRADE_GATE`:
    shadow | on | true | 1 (默认) -> 生效, 零裁决, 落 data/trade_gate_shadow.json
    off | 0 | false               -> 完全禁用
默认 shadow: 本模块不改变任何裁决, 故无「凭空造出 PASS」风险 (零裁决类可安全默认开)。

成本敏感性门候选 (本次唯一交付的「门」, 仅提案, 未生效)
====================================================
候选内容: 在 S1 通过后, 对 `高拖累` 因子**不硬拒**, 而是在 S2/S5 阶段
          把成本档位从单档 15bp 改为 **15bp/30bp 双档对账**, 要求
          net(30bp) 仍为正 才允许进入 JQ 队列 (即 `cost_robust_30bp` 门)。

证据 (库内实测, 非假设):
  1. 本模块 2026-09-18 冒烟: fast_mom_2 (2 日动量) gross 17.7% / turn 1.92
     -> 15bp +3.2% (过), 30bp -11.3% (不过) = 脆断。若只按 15bp 单档, 该因子
     会被判「可入库」, 但实盘成本稍高即转亏。
  2. research/qlib_daily/p4_cost_capacity_report.json: 日频 cost x1/x2/x3
     -> 年化 18.95% / 9.51% / 0.82% (turnover 33%); 周频 10.72% / 8.79% / 6.91%。
     ⇒ 高换手因子对成本的响应是**非线性脆断**, 不是线性折价。
  3. 本模块对 forge_gen4_r1 (turn 1.23) 读出 drag30 = 224% of |gross| ⇒
     即使毛 alpha 为正, 成本吃掉全部且倒亏。

为何只是「候选」而非直接生效:
  - 该门会**削减**候选池 (单向收紧), 理论上可翻; 但成本档位选择 (30bp 是否过严)
    目前只有 3 个样本因子支撑, 样本太薄 ⇒ 必须先影子跑满若干轮, 用
    `data/trade_gate_shadow.json` 的 n_high_drag / n_checked 比例决定阈值,
    再走「翻默认四关」(getter 实测 / A/B 反事实 / 不变量 / 端到端对账)。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent          # e:/quant
DATA_DIR = BASE / "data"
OUT_PATH = DATA_DIR / "trade_gate_shadow.json"

FORBIDDEN_USE = (
    "仅成本敏感性诊断: 本模块输出(含本地 fwd 代理收益)严禁用于因子准入/S1·S2·S5 "
    "裁决输入/候选改写/生成器注入, 亦不得取代 S5 的 P0-g 成本计费与 TRADABLE_GATE。"
    "本地代理面板的绝对收益不可与 S5/JQ 读数比较, 仅跨成本档位梯度有意义。"
)

# ── 参数 (env 可覆盖, 全部有安全默认) ─────────────────────
DEFAULT_BPS: Tuple[float, ...] = (15.0, 30.0, 45.0)
MIN_STOCKS = int(os.environ.get("FUXI_TRADE_MIN_STOCKS", "50") or 50)
MAX_FORMULAS = int(os.environ.get("FUXI_TRADE_MAX_FORMULAS", "8") or 8)
MAX_PERIODS = int(os.environ.get("FUXI_TRADE_MAX_PERIODS", "120") or 120)
TOP_N = int(os.environ.get("FUXI_TRADE_TOP_N", "30") or 30)
REBAL_H = int(os.environ.get("FUXI_TRADE_H", "5") or 5)
FWD = 5                     # 与 REBAL_H 对齐; 改 H 须同步改 fwd 派生
PPY = 252.0 / REBAL_H       # 年化期数
# 🔴 判据分两类, 缺一不可 (2026-09-18 修正):
#  (1) 多头脆断 fragile_30bp: 15bp 档为正、30bp 档转为非正 ⇒ 成本吃掉全部 alpha。
#      仅对「多头方向正确」的因子有意义 (first > 0 才可能触发)。
#  (2) 成本拖累 high_drag_30bp: **方向无关**。drag = gross_ann - net_ann,
#      drag_ratio_30 = drag30 / |gross_ann| > 该比例 ⇒ 换手过高。
#      这是 (1) 的补充: 对**反向后**因子 (gross<0) 或 alpha 偏弱的因子, (1) 恒不触发,
#      但成本敏感性依然存在 —— 只看 (1) 会漏报 (fail-open)。
FRAGILE_30 = float(os.environ.get("FUXI_TRADE_FRAGILE_30", "0.0") or 0.0)
STEEP_RATIO = float(os.environ.get("FUXI_TRADE_STEEP", "0.5") or 0.5)
# 成本拖累判据: drag30 / |gross_ann| 超过该比例 ⇒ 换手驱动的高成本敏感性
DRAG_RATIO = float(os.environ.get("FUXI_TRADE_DRAG", "0.5") or 0.5)
# 期数下限: 低于此值只记录读数, 不出「脆断/拖累」结论 (样本太薄 ⇒ 假阳性)
MIN_PERIODS_REPORT = int(os.environ.get("FUXI_TRADE_MIN_PERIODS", "12") or 12)

_CTX: Dict[str, Any] = {}


def get_mode() -> str:
    """读取开关。默认 shadow (零裁决)。"""
    raw = str(os.environ.get("FUXI_TRADE_GATE", "shadow") or "shadow").strip().lower()
    if raw in ("off", "0", "false", "no", "disable", "disabled", "none"):
        return "off"
    return "shadow"


def _bps_levels() -> Tuple[float, ...]:
    raw = os.environ.get("FUXI_TRADE_BPS", "")
    if raw:
        try:
            lv = tuple(float(x) for x in str(raw).split(",") if str(x).strip())
            if lv:
                return lv
        except ValueError:
            pass
    return DEFAULT_BPS


def _get_computer():
    if "computer" not in _CTX:
        from factor_ic_computer import FactorICComputer
        _CTX["computer"] = FactorICComputer(lookback_days=int(
            os.environ.get("FUXI_TRADE_LOOKBACK_DAYS", "750") or 750))
        _CTX["computer"]._load_data()
    return _CTX["computer"]


def attach_computer(comp) -> None:
    """复用 ralph_loop 已加载的 IC 计算器 (避免重复读 ~560MB 面板)。"""
    if comp is not None:
        _CTX["computer"] = comp


def _price_frame() -> pd.DataFrame:
    """从面板派生 (ts_code, trade_date, close, fwd) —— 进程内缓存。

    fwd = 该行后 FWD 个交易日的收益 (按个股 groupby shift(-FWD))。
    🔴 只取 close 一列派生, 不做任何中性化/残差处理 (本模块纯诊断)。
    """
    if "px" in _CTX:
        return _CTX["px"]
    comp = _get_computer()
    px = comp._price_df
    cols = [c for c in ("ts_code", "trade_date", "close") if c in px.columns]
    px = (px[cols].drop_duplicates(subset=["ts_code", "trade_date"])
          .sort_values(["ts_code", "trade_date"]).reset_index(drop=True))
    g = px.groupby("ts_code", sort=False)["close"]
    px["fwd"] = (g.shift(-FWD) / px["close"].clip(lower=1e-9) - 1.0).astype("float32")
    px = px[["ts_code", "trade_date", "close", "fwd"]]
    _CTX["px"] = px
    return px


def _period_returns(fdf: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """按 REBAL_H 期调仓的等权组合: 毛收益序列 + 逐期换手序列。"""
    dates = np.sort(fdf["trade_date"].unique())
    if len(dates) < REBAL_H * 3:
        return None
    reb = dates[::REBAL_H]
    if len(reb) > MAX_PERIODS:
        idx = np.unique(np.linspace(0, len(reb) - 1, MAX_PERIODS).round().astype(int))
        reb = reb[idx]

    by_date = {d: g for d, g in fdf.groupby("trade_date", sort=False)}
    gross: List[float] = []
    turns: List[float] = []
    prev: Optional[set] = None
    for d in reb:
        g = by_date.get(d)
        if g is None or len(g) < MIN_STOCKS:
            continue
        sel = g.nlargest(TOP_N, "factor_value")
        cur = set(sel["ts_code"].tolist())
        if prev is None:
            turn = 1.0                       # 首期全额建仓 (与 S5 P0-g 同式)
        else:
            turn = (len(prev - cur) + len(cur - prev)) / float(TOP_N)
        r = pd.to_numeric(sel["fwd"], errors="coerce").dropna()
        if len(r) == 0:
            continue
        gross.append(float(r.mean()))
        turns.append(float(turn))
        prev = cur

    if len(gross) < 5:
        return None
    return {"gross": np.asarray(gross, dtype="float64"),
            "turns": np.asarray(turns, dtype="float64")}


def factor_check(formula: str, name: str, paradigm: str = "",
                 verbose: bool = False) -> Optional[Dict[str, Any]]:
    """单因子成本敏感性 (只返回标量读数)。"""
    comp = _get_computer()
    lb = int(getattr(comp, "lookback_days", 0) or 0)
    fdf = comp._eval_formula(formula, str(name)[:40])
    if fdf is None or len(fdf) == 0:
        return None
    fdf = fdf.dropna(subset=["factor_value"])
    if len(fdf) == 0:
        return None
    df = fdf.merge(_price_frame(), on=["ts_code", "trade_date"], how="inner")
    df = df.dropna(subset=["factor_value", "fwd"])
    if len(df) < MIN_STOCKS * 5:
        return None

    pr = _period_returns(df)
    if pr is None:
        return None
    gross, turns = pr["gross"], pr["turns"]

    levels = _bps_levels()
    nets: Dict[str, float] = {}
    for bps in levels:
        net = gross - turns * (bps / 1e4)
        nets[f"{bps:g}"] = float(np.mean(net) * PPY)

    g_ann = float(np.mean(gross) * PPY)
    keys = [f"{b:g}" for b in levels]
    first = nets[keys[0]]
    second = nets[keys[1]] if len(keys) > 1 else None
    fragile = bool(first > FRAGILE_30 and second is not None and second <= FRAGILE_30)
    denom = abs(first) if abs(first) > 1e-9 else None
    steep = bool(denom and ((first - second) / denom) > STEEP_RATIO) if second is not None else False

    # 🔴 方向无关的成本拖累 (见文件头判据说明): drag = gross - net = turnover * bps
    drag = {k: float(g_ann - nets[k]) for k in keys}
    d_denom = abs(g_ann) if abs(g_ann) > 1e-9 else None
    k30 = keys[1] if len(keys) > 1 else keys[0]
    drag_ratio_30 = float(drag[k30] / d_denom) if d_denom else None
    reliable = int(len(gross)) >= MIN_PERIODS_REPORT
    high_drag = bool(reliable and drag_ratio_30 is not None
                     and drag_ratio_30 > DRAG_RATIO)

    return {
        "name": str(name)[:40],
        "paradigm": paradigm,
        "n_periods": int(len(gross)),
        "reliable": reliable,
        "lookback_days": lb,
        "top_n": TOP_N,
        "rebal_h": REBAL_H,
        "gross_ann": g_ann,
        "turnover_mean": float(np.mean(turns)),
        "net_ann": nets,
        "cost_drag": drag,
        "drag_ratio_30": drag_ratio_30,
        "fragile_30bp": bool(fragile and reliable),
        "steep_sensitivity": bool(steep and reliable),
        "high_drag_30bp": high_drag,
    }


def _pick_formulas(candidates: List[Dict], results: List[Any],
                   tier: str = "s1") -> List[Dict[str, str]]:
    """挑出公式齐备的候选 (默认只取 S1 通过者)。"""
    out: List[Dict[str, str]] = []
    for i, c in enumerate(candidates or []):
        if not isinstance(c, dict):
            continue
        f = c.get("formula") or c.get("expression") or ""
        if isinstance(f, dict):
            f = f.get("content", "")
        if not f or not str(f).strip():
            continue
        res = results[i] if results and i < len(results) else None
        if res is not None and tier != "any":
            if tier == "s1" and not getattr(res, "s1_passed", False):
                continue
            if tier == "jq" and not getattr(res, "eligible_for_jq", False):
                continue
        out.append({
            "formula": str(f).strip(),
            "name": str(c.get("name") or c.get("factor_name") or f"cand{i}")[:40],
            "paradigm": str(c.get("paradigm", "")),
        })
        if len(out) >= MAX_FORMULAS:
            break
    return out


def evaluate(candidates: List[Dict], results: List[Any],
             verbose: bool = True) -> Dict[str, Any]:
    """对候选批量做成本敏感性诊断, 汇总并落盘 (零裁决)。"""
    mode = get_mode()
    rep: Dict[str, Any] = {
        "mode": mode,
        "generated_at": datetime.now().isoformat(),
        "caliber": {
            "panel": "local proxy (S1 fwd5)", "top_n": TOP_N, "rebal_h": REBAL_H,
            "standalone_lookback_days": int(os.environ.get("FUXI_TRADE_LOOKBACK_DAYS", "750") or 750),
            "lookback_note": "🔴 读数是窗口敏感的 (750d vs 270d 毛年化可反号) ⇒ 跨轮比较必须同窗",
            "turnover_formula": "(len(prev-cur)+len(cur-prev))/top_n",
            "bps_levels": list(_bps_levels()), "ppy": PPY,
            "criteria": {
                "fragile_30bp": f"net(15bp)>0 且 net(30bp)<=0 (仅多头方向有效)",
                "steep_sensitivity": f"(net15-net30)/|net15| > {STEEP_RATIO}",
                "high_drag_30bp": f"drag30/|gross_ann| > {DRAG_RATIO} (方向无关)",
                "min_periods_report": MIN_PERIODS_REPORT,
            },
            "evidence": "research/qlib_daily/p4_cost_capacity_report.json",
        },
        "forbidden_use": FORBIDDEN_USE,
        "n_checked": 0, "n_fragile": 0, "n_steep": 0, "n_high_drag": 0,
        "n_unreliable": 0,
        "median_turnover": 0.0, "median_net_15": 0.0, "median_net_30": 0.0,
        "median_drag_ratio_30": None,
        "rows": [],
    }
    if mode == "off":
        return rep

    tier = str(os.environ.get("FUXI_TRADE_TIER", "s1") or "s1").strip().lower()
    items = _pick_formulas(candidates, results, tier=tier)
    rows: List[Dict[str, Any]] = []
    for it in items:
        try:
            r = factor_check(it["formula"], it["name"], it["paradigm"], verbose=False)
        except Exception as e:
            if verbose:
                print(f"  [TradeGate] {it['name']} 诊断失败 (跳过): {type(e).__name__}: {e}")
            continue
        if r:
            rows.append(r)

    rep["n_checked"] = len(rows)
    if rows:
        rep["n_fragile"] = sum(1 for r in rows if r["fragile_30bp"])
        rep["n_steep"] = sum(1 for r in rows if r["steep_sensitivity"])
        rep["n_high_drag"] = sum(1 for r in rows if r["high_drag_30bp"])
        rep["n_unreliable"] = sum(1 for r in rows if not r["reliable"])
        rep["median_turnover"] = float(np.median([r["turnover_mean"] for r in rows]))
        for k in ("15", "30"):
            vals = [r["net_ann"].get(k) for r in rows if r["net_ann"].get(k) is not None]
            rep[f"median_net_{k}"] = float(np.median(vals)) if vals else 0.0
        drv = [r["drag_ratio_30"] for r in rows if r.get("drag_ratio_30") is not None]
        rep["median_drag_ratio_30"] = float(np.median(drv)) if drv else None
    rep["rows"] = rows

    if verbose:
        lv = ",".join(f"{b:g}" for b in _bps_levels())
        print(f"  [TradeGate] {mode} 模式 (仅成本敏感性, 零裁决): 检查 {len(rows)} 因子 | "
              f"脆断 {rep['n_fragile']} | 陡降 {rep['n_steep']} | 高拖累 {rep['n_high_drag']}"
              + (f" | 样本不足 {rep['n_unreliable']}" if rep['n_unreliable'] else ""))
        print(f"  [TradeGate] turnover 中位 {rep['median_turnover']:.2f} | "
              f"drag30 中位 {('%.2f' % rep['median_drag_ratio_30']) if rep['median_drag_ratio_30'] is not None else 'n/a'}")
        for r in rows:
            ann = " / ".join(f"{k}bp:{v*100:+.1f}%" for k, v in r["net_ann"].items())
            dr = r.get("drag_ratio_30")
            print(f"    {r['name'][:34]:34s} turn={r['turnover_mean']:.2f} "
                  f"n={r['n_periods']:3d} gross={r['gross_ann']*100:+.1f}% | {ann}"
                  f" drag30={('%+.0f%%' % (dr*100)) if dr is not None else 'n/a'}"
                  + ("  [脆断]" if r["fragile_30bp"] else "")
                  + ("  [陡降]" if r["steep_sensitivity"] else "")
                  + ("  [高拖累]" if r["high_drag_30bp"] else "")
                  + ("  [样本不足]" if not r["reliable"] else ""))
        print(f"  [TradeGate] 档位 {lv} | 报告 → {OUT_PATH.name}")

    try:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        # 🔴 data/ 下 JSON 约定 CRLF (用 newline='' 会落裸 LF)
        with open(OUT_PATH, "w", encoding="utf-8", newline="\r\n") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"  [TradeGate] ⚠️ 落盘失败 (非阻塞): {e}")
    return rep


def _cli() -> int:
    ap = argparse.ArgumentParser(description="成本敏感性影子诊断 (P-20260918-002 收缩版)")
    ap.add_argument("--formula", help="直接指定公式")
    ap.add_argument("--name", default="adhoc", help="公式名 (默认 adhoc)")
    ap.add_argument("--replay", type=int, default=0, help="重放库内因子数量")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.formula:
        r = factor_check(args.formula, args.name)
        print(json.dumps(r, ensure_ascii=False, indent=2) if r else "无读数")
        return 0
    print(f"[TradeGate] mode={get_mode()} | 需从 ralph_loop 调用 evaluate() 或等 replay 接入")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
