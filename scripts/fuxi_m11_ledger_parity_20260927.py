"""M11 双口径对拍 (2026-09-27): 等权周换仓 proxy vs 固定股数现金账本.

用途 (reports/fuxi_alpha_generation_overhaul_20260926.md §M11):
  同候选 (冻结三臂 v2/v3/amount_top80)、同窗 (2025-01-02~2026-04-30, 320 日),
  对拍「等权周换仓代理口径」与「固定股数现金账本」:
    · 逐日 NAV 差 (归一化后)
    · 费用差 (proxy 换手成本 vs 账本逐笔费)
    · 五态分布差异 (valid / data_insufficient / error 分类)
  账本侧 8 情景 (零碎股 floor/ceil × 股息税 0/20% × 退市计价 0/100%),
  每情景独立 Decimal 复算, 断言最大日净值绝对差 ≤ 1e-7 元。

声明 (不越界):
  · 账本 = 声明模型情景 (gross-cash stress), formal_eligibility=False;
    不是券商/聚宽/真实账户对拍;
  · proxy = S5 backtest_factor(enforce) 语义复刻 (close/pre_close-1, 缺报价
    不缩分母, 板块 adj 守卫, 成本在持有期首个有效日);
  · 本产物不产生 S5 通过、JQ 资格或经济赢家结论; 2025/2026 为研究已暴露窗口.

用法:
  python scripts/fuxi_m11_ledger_parity_20260927.py            # 全量三臂 (封存目录)
  python scripts/fuxi_m11_ledger_parity_20260927.py --smoke    # v2 单情景冒烟
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research/factor_alchemy"))
sys.path.insert(0, str(ROOT / "scripts"))

from portfolio_execution_ledger import CorporateAction, LotRule  # noqa: E402
from s5_ledger_bridge import (compare_books, fixed_shares_book,  # noqa: E402
                              proxy_book)
from fuxi_a1_three_arm_strict_probe_20260926 import (  # noqa: E402
    ADJUDICATION, OUT, load, load_bars, targets_for)
from fuxi_a1_actual_holdings_trace_20260926 import DELIST_TRACE_FACTS  # noqa: E402
from fuxi_a1_model_account_replay_20260926 import (  # noqa: E402
    EXCHANGE, independent_calculation)
from run_contract import FuxiWriterLock  # noqa: E402

ARMS = ("v2", "v3", "amount_top80")
TOP_N = 80
INITIAL_CASH = 10_000_000.0
COST_BPS = 15.0
FRACTIONS = ("floor", "ceil")
TAXES = (0.0, 0.2)
DELIST_MARKS = (0.0, 1.0)
COMPLETE = "book_complete_under_declared_assumptions"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_actions(arm: str, codes: set, days: set):
    """封存事件 → CorporateAction 表 (与 A1 v4 完全同源同过滤)。"""
    planned = load(ADJUDICATION)
    extension = load(OUT / "held_extension_audit_v5.json")
    if planned["counts"] != {"approved_for_gross_cash_stress_only": 229}:
        raise AssertionError("planned action seal changed")
    if extension["union_keys"] != 24:
        raise AssertionError("held-event seal changed")
    actions_by_key = {}
    for item in planned["events"] + extension["events"]:
        if item.get("status",
                    "approved_for_gross_cash_stress_only") != "approved_for_gross_cash_stress_only":
            continue
        action = item.get("action")
        if action is None or action["code"] not in codes:
            continue
        if action["record_date"] not in days or action["ex_date"] not in days:
            continue
        key = (action["code"], action["ex_date"])
        previous = actions_by_key.get(key)
        if previous and (previous.cash_per_share_net != action["cash_per_share_net"]
                         or previous.share_multiplier != action["share_multiplier"]):
            raise AssertionError(f"conflicting event rights: {key}")
        actions_by_key[key] = CorporateAction(**action)
    if arm == "amount_top80":
        item = load(OUT / "actual_holdings_extension_v1.json")
        action = CorporateAction(**item["action"])
        actions_by_key[(action.code, action.ex_date)] = action
    return actions_by_key


def timing_decomposition(proxy: dict, proxy_next_open: dict, book: dict) -> dict:
    """把 proxy vs ledger 的净值总差分解为「时点敏感度」与「残差」两部分.

    构造 (恒等式):
        total_gap    = ledger_norm_final_nav - proxy(signal_close).final_nav
        timing_gap   = proxy(next_open).final_nav - proxy(signal_close).final_nav
        residual_gap = ledger_norm_final_nav - proxy(next_open).final_nav
        total_gap    = timing_gap + residual_gap
    残差包含: 固定股数 vs 等权日再平衡、整手取整/拒单/递延、逐笔费用结构、
    股息税与退市计价情景。仅作分解陈述, 不是归因结论; 账本未闭合时只报
    时点敏感度, 其余置 None (不得用未闭合账本读数做差)。
    """
    out = {
        "proxy_signal_close_final_nav": proxy.get("final_nav"),
        "proxy_next_open_final_nav": proxy_next_open.get("final_nav"),
        "timing_gap_nav": (None if proxy.get("final_nav") is None
                           or proxy_next_open.get("final_nav") is None
                           else proxy_next_open["final_nav"] - proxy["final_nav"]),
    }
    if book.get("status") == COMPLETE and book.get("final_nav") is not None \
            and proxy.get("final_nav") is not None:
        ledger_norm = book["final_nav"] / INITIAL_CASH
        total = ledger_norm - proxy["final_nav"]
        out.update({
            "ledger_final_nav_norm": ledger_norm,
            "total_gap_nav": total,
            "residual_gap_nav": ledger_norm - proxy_next_open["final_nav"],
            "timing_share_of_abs_total": (abs(out["timing_gap_nav"]) / abs(total)
                                          if total else None),
        })
    else:
        out.update({"ledger_final_nav_norm": None, "total_gap_nav": None,
                    "residual_gap_nav": None, "timing_share_of_abs_total": None})
    return out


def run_arm(arm: str, *, smoke: bool = False) -> dict:
    target_path, targets = targets_for(arm)
    codes = {c for weights in targets.values() if weights for c in weights}
    if arm == "amount_top80":
        codes.add("600150.SH")
    days, bars, coverage = load_bars(codes)
    if len(days) != 320:
        raise AssertionError("frozen calendar changed")
    day_set = set(days)

    t0 = time.time()
    proxy = proxy_book(days, bars, targets, top_n=TOP_N, cost_bps=COST_BPS)
    t_proxy = time.time() - t0

    # 时点敏感度变体: 首个持有日按 close/open-1 计价 (账本为次日开盘成交),
    # 仅用于把总差分解为「时点敏感度 + 残差(股数/费用/整手/拒单口径差)」,
    # 不改变默认口径, 也不产生任何晋级读数。
    proxy_next_open = proxy_book(days, bars, targets, top_n=TOP_N,
                                 cost_bps=COST_BPS, entry_mode="next_open")

    actions_by_key = build_actions(arm, codes, day_set)
    rules = {c: LotRule(200, 1) if c.startswith(("688", "689")) else
             LotRule(100, 1) if c.endswith(".BJ") else LotRule()
             for c in codes}
    delisted = {code: fact for code, fact in DELIST_TRACE_FACTS.items()
                if code in codes}
    model_exchanges = [EXCHANGE] if arm == "amount_top80" else []

    scenarios = []
    combos = [(FRACTIONS[0], TAXES[0], DELIST_MARKS[1])] if smoke else [
        (f, t, d) for f in FRACTIONS for t in TAXES for d in DELIST_MARKS]
    for fraction, tax, delist_mark in combos:
        scenario = f"{fraction}_tax{int(tax*100)}_delist{int(delist_mark*100)}"
        ts = time.time()
        book = fixed_shares_book(
            days, bars, targets, initial_cash=INITIAL_CASH, cost_bps=COST_BPS,
            rules=rules, actions=list(actions_by_key.values()),
            delist_evidence=delisted, share_exchanges=model_exchanges,
            fractional_share_policy=fraction, gross_cash_tax_rate=tax,
            delist_mark_fraction=delist_mark,
            missing_limit_policy="defer",
            halted_session_check_policy="vacuous",
            reference_only_events={("002822.SZ", "2025-12-30")},
            include_ledger=True)
        parity_calc = None
        if book["status"] == COMPLETE:
            try:
                calc = independent_calculation(
                    book["ledger_object"], bars,
                    list(actions_by_key.values()), delisted,
                    fractional_policy=fraction, tax_rate=tax,
                    delist_mark_fraction=delist_mark,
                    share_exchanges=model_exchanges)
                parity_calc = {
                    "compared_days": calc["compared_days"],
                    "checked_fills": calc["checked_fills"],
                    "checked_rights": calc["checked_rights"],
                    "checked_live_marks": calc["checked_live_marks"],
                    "checked_model_marks": calc["checked_model_marks"],
                    "max_nav_abs_gap": calc["max_nav_abs_gap"],
                    "independent_data_source": calc["independent_data_source"],
                    "broker_or_jq_parity": calc["broker_or_jq_parity"],
                }
            except AssertionError as exc:
                parity_calc = {"error": str(exc)[:240]}
        cmp = compare_books(proxy, book, label=scenario)
        timing = timing_decomposition(proxy, proxy_next_open, book)
        ledger_obj = book.pop("ledger_object", None)
        del ledger_obj
        scenarios.append({
            "scenario": scenario,
            "ledger": {
                "status": book["status"],
                "reasons": book["reasons"],
                "n_fills": book["n_fills"],
                "n_orders": book["n_orders"],
                "total_fees": book["total_fees"],
                "total_fees_frac": book["total_fees_frac"],
                "final_nav": book["final_nav"],
                "net_return": book["net_return"],
                "missing": book["missing"],
                "nav_daily": (sorted(book["nav_by_day"].items())
                              if book.get("nav_by_day") else None),
                "assessment_reasons": book["assessment"].get("reasons"),
            },
            "independent_calculation": parity_calc,
            "comparison": cmp,
            "timing_decomposition": timing,
            "elapsed_sec": round(time.time() - ts, 2),
        })
        print(f"    [{arm}] {scenario}: ledger={book['status']} "
              f"nav={book['final_nav']:.0f} fees={book['total_fees']:.1f} "
              f"gap={cmp.get('max_abs_nav_diff')} "
              f"parity_gap={(parity_calc or {}).get('max_nav_abs_gap')}",
              flush=True)

    return {
        "schema": "fuxi_m11_ledger_parity/v1",
        "arm": arm,
        "window": [days[0], days[-1]],
        "expected_days": len(days),
        "signal_dates": len(targets),
        "proxy": {
            "status": proxy["status"],
            "n_periods": proxy["n_periods"],
            "n_usable_day_returns": proxy["n_usable_day_returns"],
            "total_cost_frac": proxy["total_cost_frac"],
            "final_nav": proxy["final_nav"],
            "nav_daily": sorted(proxy["nav_by_day"].items()),
            "weighting_contract": proxy["weighting_contract"],
        },
        "proxy_next_open": {
            "status": proxy_next_open["status"],
            "n_periods": proxy_next_open["n_periods"],
            "n_usable_day_returns": proxy_next_open["n_usable_day_returns"],
            "total_cost_frac": proxy_next_open["total_cost_frac"],
            "final_nav": proxy_next_open["final_nav"],
            "nav_daily": sorted(proxy_next_open["nav_by_day"].items()),
            "entry_mode": "next_open",
        },
        "scenarios": scenarios,
        "quote_coverage": coverage,
        "action_keys": len(actions_by_key),
        "proxy_sec": round(t_proxy, 2),
        "input_sha256": {str(p.relative_to(ROOT)): digest(p) for p in (
            target_path, ADJUDICATION, OUT / "held_extension_audit_v5.json",
            ROOT / "data/raw/daily_prices.csv",
            ROOT / "data/raw/stk_limit.csv",
            ROOT / "data/raw/adj_factor_daily.csv",
            ROOT / "data/raw/suspend_d.csv",
            ROOT / "research/factor_alchemy/portfolio_execution_ledger.py",
            ROOT / "research/factor_alchemy/s5_ledger_bridge.py",
            ROOT / "scripts/fuxi_m11_ledger_parity_20260927.py",
        )},
        "boundary": {
            "declared_model_scenarios": True,
            "formal_eligibility": False,
            "research_exposed_window": "2025/2026 已暴露, 不得包装为样本外",
            "not_broker_or_jq_parity": True,
        },
    }


def summarize(payload: list) -> list:
    rows = []
    for item in payload:
        for sc in item["scenarios"]:
            cmp = sc["comparison"]
            rows.append({
                "arm": item["arm"],
                "scenario": sc["scenario"],
                "ledger_status": sc["ledger"]["status"],
                "n_days_compared": cmp.get("n_days_compared"),
                "max_abs_nav_diff": cmp.get("max_abs_nav_diff"),
                "final_nav_diff": cmp.get("final_nav_diff"),
                "proxy_cost_frac": cmp["cost"]["proxy_total_cost_frac"],
                "ledger_fees_frac": cmp["cost"]["ledger_total_fees_frac"],
                "cost_diff_frac": cmp["cost"]["diff_frac"],
                "five_state_proxy": cmp["five_state"]["proxy"],
                "five_state_ledger": cmp["five_state"]["ledger"],
                "parity_max_nav_abs_gap": (sc["independent_calculation"] or {}).get(
                    "max_nav_abs_gap"),
                "timing_gap_nav": sc["timing_decomposition"]["timing_gap_nav"],
                "total_gap_nav": sc["timing_decomposition"]["total_gap_nav"],
                "residual_gap_nav": sc["timing_decomposition"]["residual_gap_nav"],
                "timing_share_of_abs_total": sc["timing_decomposition"][
                    "timing_share_of_abs_total"],
                "elapsed_sec": sc["elapsed_sec"],
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    outdir = ROOT / ("reports/fuxi_m11_ledger_parity_20260927_smoke" if args.smoke
                     else "reports/fuxi_m11_ledger_parity_20260927")
    if outdir.exists():
        raise RuntimeError("sealed parity directory already exists: " + str(outdir))
    arms = ("v2",) if args.smoke else ARMS
    with FuxiWriterLock(run_id="m11_ledger_parity"):
        payload = [run_arm(arm, smoke=args.smoke) for arm in arms]
        outdir.mkdir(parents=True, exist_ok=False)
        for item in payload:
            (outdir / f"{item['arm']}.json").write_text(
                json.dumps(item, ensure_ascii=False, indent=1) + "\n",
                encoding="utf-8", newline="\n")
        rows = summarize(payload)
        (outdir / "summary.json").write_text(
            json.dumps({"schema": "fuxi_m11_ledger_parity_summary/v1",
                        "rows": rows}, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8", newline="\n")
        print(json.dumps({"outdir": str(outdir), "rows": len(rows)},
                         ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
