# -*- coding: utf-8 -*-
"""
P-20260825-004: 拥挤度三阶段衰减模型 — 离线回放校准 + 每日扫描

两种模式:
  1. python scripts/calibrate_crowding_decay.py --daily
     → 每日族级拥挤衰减扫描 (读 factor_level_crowding.json + library_orthogonality_state.json,
       结果归档 crowding_decay_history.json; 校准完成前影子模式只记录不报警)

  2. python scripts/calibrate_crowding_decay.py --calibrate
     → 离线回放校准: 网格搜索 (corr_threshold × ic_drop_threshold),
       评估"触发拥挤衰减预警后该族 IC 是否继续下滑"的命中率/误报率,
       选最优阈值写入 crowding_decay_config.json 并启用告警

数据要求: crowding_decay_history.json 至少积累 20 个交易日快照
(影子模式运行 4 周后执行校准)
"""

import argparse
import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
CONFIG_PATH = DATA_DIR / "crowding_decay_config.json"
HISTORY_PATH = DATA_DIR / "crowding_decay_history.json"

# 校准网格 (P-004 初始建议值附近展开)
CORR_GRID = [0.15, 0.20, 0.25, 0.30, 0.35]
DROP_GRID = [0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
CONFIRM_GRID = [1, 2, 3]
MIN_SNAPSHOTS = 20          # 校准所需最少历史快照数
HORIZON = 5                 # 预测窗口: 触发后 N 日内族 IC 是否继续下滑

# ── P-20260826-004: 影子转正判据 ──
PROMOTION_CORR_RANGE = (0.15, 0.50)     # (a) 阈值合理区间
PROMOTION_DROP_RANGE = (0.15, 0.50)
PROMOTION_MIN_F1 = 0.25                 # (b) 校准质量
PROMOTION_MIN_PREC = 0.45
PROMOTION_MIN_TP = 5
PROMOTION_EVENT_LEAD = 5                # (c) 提前命中交易日数
EVENT_ANCHOR = '2026-07-15'             # 2026-07 行业量化回撤事件锚点 (中证网/好买报道窗口)
EVENT_REPLAY_START = '2026-06-20'
EVENT_REPLAY_END = '2026-08-10'


def _load_history() -> list:
    if not HISTORY_PATH.exists():
        return []
    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_config(cfg: dict) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def run_daily():
    """每日扫描: 读族级拥挤快照 + 族级 IC 快照 → scan_crowding_decay"""
    sys.path.insert(0, str(Path(__file__).parent.parent / "research" / "factor_alchemy"))
    from decay_monitor import scan_crowding_decay, aggregate_family_ic

    crowding_path = DATA_DIR / "factor_level_crowding.json"
    if not crowding_path.exists():
        print("[P-004] ⚠️ factor_level_crowding.json 不存在, 跳过 (先跑 factor_level_crowding.py)")
        return

    with open(crowding_path, "r", encoding="utf-8") as f:
        crowding = json.load(f)
    family_corr = crowding.get("family_max_ratio", {})
    if isinstance(family_corr, list):  # 兼容 list 形态
        family_corr = dict(family_corr)

    family_ic = aggregate_family_ic()
    result = scan_crowding_decay(
        family_corr_map=family_corr,
        family_ic_map=family_ic,
        verbose=True,
    )
    print(f"\n[P-004] 归档完成: {result['date']} "
          f"触发 {len(result['triggered'])} 族 | 影子模式={result['shadow_mode']}")
    return result


def _build_labels(history: list, corr_th: float, drop_th: float, confirm: int) -> tuple:
    """回放: 用历史快照构造 (预测触发, 实际标签) 对

    预测: corr > corr_th 且 drop_pct > drop_th 且连续 >= confirm 天
    标签: 该族在随后 HORIZON 个快照内 ic_now 均值 < 触发时 ic_now (持续下滑)
    """
    preds, labels = [], []
    for i, snap in enumerate(history[:-HORIZON]):
        families = snap.get("families", {})
        for fam, d in families.items():
            if not d.get("triggered"):
                continue
            if d.get("consecutive", 0) < confirm:
                continue
            # 未来窗口族级 IC
            future_ics = []
            for j in range(i + 1, min(i + 1 + HORIZON, len(history))):
                fd = history[j].get("families", {}).get(fam, {})
                ic = fd.get("ic_now")
                if ic is not None:
                    future_ics.append(ic)
            if not future_ics or d.get("ic_now") is None:
                continue
            preds.append(1)
            labels.append(1 if (sum(future_ics) / len(future_ics)) < d["ic_now"] else 0)

    # 负样本: 未触发族在后续窗口的走向 (采样等量以平衡)
    neg_cnt = 0
    for i, snap in enumerate(history[:-HORIZON]):
        families = snap.get("families", {})
        for fam, d in families.items():
            if d.get("triggered") or d.get("ic_now") is None:
                continue
            future_ics = []
            for j in range(i + 1, min(i + 1 + HORIZON, len(history))):
                fd = history[j].get("families", {}).get(fam, {})
                ic = fd.get("ic_now")
                if ic is not None:
                    future_ics.append(ic)
            if not future_ics:
                continue
            preds.append(0)
            labels.append(1 if (sum(future_ics) / len(future_ics)) < d["ic_now"] else 0)
            neg_cnt += 1
            if neg_cnt >= len(preds) - neg_cnt:
                break
        if neg_cnt >= 200:
            break

    return preds, labels


def _precision_recall(preds, labels):
    tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1, tp, fp, fn


def _replay_event_lead():
    """P-20260826-004 (c): 2026-07 事件回放 — 拥挤度侧提前预警验证

    方法: 每族取当前多空波动率比最高的 1 个代表因子, 用 daily_prices.csv
    回填 2026-06-20~2026-08-10 的日度 ls_vol_ratio 序列, 检查是否存在族
    在事件锚点 (2026-07-15) 前 >=5 交易日出现连续触发 (ratio>1.5, confirm>=2)。

    返回 {"passed": bool, "best_lead": int, "hit_families": [...], "reason": str}
    """
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent))
    try:
        import numpy as _np
        import pandas as _pd
        from factor_level_crowding import (
            DAILY_PRICES, LIB_STATE, STATUSES, TH_LS_RATIO,
            load_price_panels, eval_expression, ls_vol_ratio_series,
        )
    except Exception as e:
        return {"passed": False, "best_lead": 0, "hit_families": [],
                "reason": f"事件回放模块导入失败: {str(e)[:80]}"}

    # 每族代表因子 (当前 ls_vol_ratio 最高)
    if not LIB_STATE.exists():
        return {"passed": False, "best_lead": 0, "hit_families": [],
                "reason": "library_orthogonality_state.json 缺失"}
    with open(LIB_STATE, 'r', encoding='utf-8') as f:
        lib = json.load(f)
    sels = [x for x in lib.get('factors', []) if x.get('status') in STATUSES]
    if not sels:
        return {"passed": False, "best_lead": 0, "hit_families": [],
                "reason": "无 candidate/jq_done 因子"}
    rep = {}
    for x in sels:
        p = x.get('paradigm') or '未标注'
        # 族内收集全部候选 (代表取首个可求值者, 避免 Forge 风格表达式导致全族跳过)
        rep.setdefault(p, []).append(x)

    # 回放区间面板 (数据窗口前置 150 自然日, 供 rolling(60/120) 预热; 检测窗口不变)
    df = _pd.read_csv(
        DAILY_PRICES,
        dtype={'ts_code': str, 'trade_date': str},
        usecols=['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount'],
    )
    df['trade_date'] = _pd.to_datetime(df['trade_date'], format='mixed')
    load_start = str((_pd.Timestamp(EVENT_REPLAY_START) - _pd.Timedelta(days=150)).date())
    df = df[(df['trade_date'] >= load_start) & (df['trade_date'] <= EVENT_REPLAY_END)]
    df = df.drop_duplicates(subset=['ts_code', 'trade_date'])
    cols = {'close': 'close_p', 'open': 'open_p', 'high': 'high_p',
            'low': 'low_p', 'vol': 'volume_p', 'amount': 'amount_p'}
    panels = {}
    for src, dst in cols.items():
        panels[dst] = df.pivot(index='trade_date', columns='ts_code', values=src)
    ret_df = panels['close_p'].pct_change(fill_method=None)
    anchor = _pd.Timestamp(EVENT_ANCHOR)

    best_lead = 0
    hit_families = []
    n_eval_ok = 0
    for p, fam_factors in rep.items():
        ratio = None
        for x in fam_factors:
            expr = str(x.get('expression', ''))
            if not expr:
                continue
            try:
                fv = eval_expression(expr, panels, {})
            except Exception:
                continue
            ratio, _ = ls_vol_ratio_series(fv, ret_df)
            if ratio is not None and not ratio.empty:
                n_eval_ok += 1
                break  # 族内首个可求值者即代表
        if ratio is None or ratio.empty:
            continue
        # 连续触发检测 (ratio > TH_LS_RATIO, confirm=2), 记录首次触发日
        # 检测窗口限制在回放区间内 (预热期数据不参与判定)
        consec = 0
        first_hit = None
        for dt, v in ratio.items():
            if dt < _pd.Timestamp(EVENT_REPLAY_START):
                continue
            if v > TH_LS_RATIO:
                consec += 1
                if first_hit is None:
                    first_hit = dt
            else:
                consec = 0
                first_hit = None
            if consec >= 2 and first_hit is not None:
                lead = (anchor - first_hit).days
                # 自然日 → 交易日近似 (/1.4)
                lead_td = lead / 1.4
                if lead_td >= PROMOTION_EVENT_LEAD and first_hit < anchor:
                    hit_families.append({'family': p, 'first_hit': str(first_hit.date()),
                                         'lead_days': round(lead_td, 1)})
                    best_lead = max(best_lead, round(lead_td, 1))
                    break
    if hit_families:
        return {"passed": True, "best_lead": best_lead,
                "hit_families": hit_families[:5], "reason": ""}
    return {"passed": False, "best_lead": best_lead, "hit_families": [],
            "reason": (f"事件回放: 无族在锚点 {EVENT_ANCHOR} 前 >= {PROMOTION_EVENT_LEAD} "
                       f"交易日提前触发 (最佳 lead={best_lead})")}


def _evaluate_promotion(history: list, best: dict) -> tuple:
    """P-20260826-004: 影子转正判据 (三条件全满足才允许 alert_enabled=True)

    (a) 阈值合理区间: corr∈[0.15,0.50] 且 drop∈[0.15,0.50]
    (b) 校准质量: F1>=0.25 且 precision>=0.45 且 TP>=5
    (c) 事件回放: 2026-07 行业回撤事件前 >=5 交易日提前预警

    返回 (passed, checks, reasons)
    """
    checks = {}
    reasons = []

    corr_ok = PROMOTION_CORR_RANGE[0] <= best['corr'] <= PROMOTION_CORR_RANGE[1]
    drop_ok = PROMOTION_DROP_RANGE[0] <= best['drop'] <= PROMOTION_DROP_RANGE[1]
    checks['a_threshold_reasonable'] = bool(corr_ok and drop_ok)
    if not checks['a_threshold_reasonable']:
        reasons.append(f"最优阈值 corr={best['corr']}/drop={best['drop']} 超出合理区间")

    qual_ok = (best['f1'] >= PROMOTION_MIN_F1 and best['prec'] >= PROMOTION_MIN_PREC
               and best['tp'] >= PROMOTION_MIN_TP)
    checks['b_calibration_quality'] = bool(qual_ok)
    if not qual_ok:
        reasons.append(f"校准质量不足: F1={best['f1']} P={best['prec']} TP={best['tp']}")

    event = _replay_event_lead()
    checks['c_event_replay'] = event
    if not event['passed']:
        reasons.append(event['reason'])

    passed = bool(checks['a_threshold_reasonable'] and qual_ok and event['passed'])
    return passed, checks, reasons


def run_calibrate():
    """网格搜索回放校准"""
    history = _load_history()
    if len(history) < MIN_SNAPSHOTS:
        print(f"[P-004] ⚠️ 历史快照不足: {len(history)} < {MIN_SNAPSHOTS}")
        print(f"  影子模式需再运行 {MIN_SNAPSHOTS - len(history)} 个交易日后才可校准")
        print("  (每日执行: python scripts/calibrate_crowding_decay.py --daily)")
        return

    print(f"[P-004] 回放校准: {len(history)} 个快照, 网格 "
          f"{len(CORR_GRID)}x{len(DROP_GRID)}x{len(CONFIRM_GRID)}")

    best = None
    results = []
    for corr_th in CORR_GRID:
        for drop_th in DROP_GRID:
            for confirm in CONFIRM_GRID:
                preds, labels = _build_labels(history, corr_th, drop_th, confirm)
                if len(preds) < 10:
                    continue
                prec, rec, f1, tp, fp, fn = _precision_recall(preds, labels)
                results.append({
                    "corr": corr_th, "drop": drop_th, "confirm": confirm,
                    "prec": round(prec, 3), "rec": round(rec, 3),
                    "f1": round(f1, 3), "tp": tp, "fp": fp, "fn": fn,
                    "n": len(preds),
                })
                if best is None or f1 > best["f1"]:
                    best = results[-1]

    results.sort(key=lambda r: -r["f1"])
    print(f"\n  最优组合 (按 F1): corr>{best['corr']} 且 IC降幅>{best['drop']:.0%} "
          f"确认{best['confirm']}天")
    print(f"  F1={best['f1']} | precision={best['prec']} | recall={best['rec']} "
          f"| TP={best['tp']} FP={best['fp']} FN={best['fn']}")
    print("\n  网格 Top10:")
    for r in results[:10]:
        print(f"    corr>{r['corr']}  drop>{r['drop']:.0%}  confirm={r['confirm']}  "
              f"F1={r['f1']}  P={r['prec']}  R={r['rec']}  n={r['n']}")

    # ── P-20260826-004: 转正判据 (满足才写回启用) ──
    passed, checks, reasons = _evaluate_promotion(history, best)
    print(f"\n  [P-004 转正判据] {'✅ 全部通过' if passed else '❌ 未通过 → 维持影子模式'}")
    for k, v in checks.items():
        if isinstance(v, dict):
            v = f"passed={v['passed']}" + (f" best_lead={v.get('best_lead', 0)}" if v.get('best_lead') else "")
        print(f"    {k}: {v}")
    for r in reasons:
        print(f"    - {r}")

    cfg = _load_config()
    import datetime as _dt
    if not passed:
        cfg.update({
            "alert_enabled": False,
            "promotion_check": {
                "checked_at": _dt.datetime.now().isoformat()[:19],
                "passed": False,
                "checks": {k: (v if not isinstance(v, dict) else v.get('passed')) for k, v in checks.items()},
                "reasons": reasons,
                "best_candidate": best,
            },
        })
        _save_config(cfg)
        print(f"\n  ⏸️ 判据未满足 → 维持影子模式 (再观察 10 交易日)")
        return

    # 写回配置并启用告警
    cfg.update({
        "corr_threshold": best["corr"],
        "ic_drop_threshold": best["drop"],
        "confirm_days": best["confirm"],
        "alert_enabled": True,
        "calibrated_at": _dt.datetime.now().isoformat()[:19],
        "calibration": best,
        "promotion_check": {
            "checked_at": _dt.datetime.now().isoformat()[:19],
            "passed": True,
            "checks": {k: (v if not isinstance(v, dict) else v.get('passed')) for k, v in checks.items()},
            "event_replay": checks.get('c_event_replay', {}),
        },
    })
    _save_config(cfg)
    print(f"\n  ✅ 阈值已写入 {CONFIG_PATH.name}, 告警已启用 (转正判据通过)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily", action="store_true", help="每日扫描+归档 (影子模式)")
    ap.add_argument("--calibrate", action="store_true", help="离线回放校准阈值")
    args = ap.parse_args()
    if args.calibrate:
        run_calibrate()
    else:
        run_daily()
