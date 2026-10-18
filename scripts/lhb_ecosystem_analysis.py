# -*- coding: utf-8 -*-
"""P-20260902-004: 龙虎榜生态变异应对 — 席位分类统计 + lhb 分段 IC 重验

① 席位类型分类 (机构专用/陆股通/游资营业部/其他) → 2024/2025/2026H1 成交占比漂移
② lhb_freq_20d 因子 2025 前/后分段 IC 检验 (结构性断点检测)
③ 分层席位表现影子基线: 各类型席位上榜后 5 日收益 (v3 分层评分影子输入)

输出: data/lhb_ecosystem_analysis.json
用法: python scripts/lhb_ecosystem_analysis.py
"""
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
OUT_FILE = DATA_DIR / 'lhb_ecosystem_analysis.json'


def classify_seat(name: str) -> str:
    """席位类型分类 (按名称规则)"""
    if not isinstance(name, str):
        return "其他"
    if "机构专用" in name:
        return "机构专用"
    if re.search(r"陆股通|深股通|沪股通|香港中央结算|港股通", name):
        return "陆股通"
    if re.search(r"证券(营业部|分公司)", name) or "营业部" in name:
        return "游资营业部"
    if "机构" in name:
        return "机构专用"
    return "其他"


def compute_seat_share():
    """① 席位类型成交占比漂移"""
    d = pd.read_csv(
        DATA_DIR / 'raw' / 'top_inst.csv',
        dtype={'trade_date': str, 'ts_code': str},
        usecols=['trade_date', 'exalter', 'buy', 'sell'],
    )
    d['trade_date'] = pd.to_datetime(d['trade_date'], format='mixed')
    d['se_type'] = d['exalter'].apply(classify_seat)
    d['turnover'] = d['buy'] + d['sell']
    d['year'] = d['trade_date'].dt.year

    periods = {'2024': [2024, 2024], '2025': [2025, 2025], '2026H1': [2026, 2026]}
    rows = {}
    for label, (y0, y1) in periods.items():
        sub = d[(d['year'] >= y0) & (d['year'] <= y1)]
        total = sub['turnover'].sum()
        share = sub.groupby('se_type')['turnover'].sum() / total
        rows[label] = {k: round(float(v), 4) for k, v in share.items()}
        print(f"[席位占比] {label}: 总成交 {total/1e12:.2f} 万亿 | " +
              "  ".join(f"{k}={v:.1%}" for k, v in share.sort_values(ascending=False).items()))
    return rows


def compute_lhb_segment_ic():
    """② lhb_freq_20d 分段 IC (2024 / 2025 / 2026H1)"""
    tl = pd.read_csv(
        DATA_DIR / 'raw' / 'top_list.csv',
        dtype={'trade_date': str, 'ts_code': str},
        usecols=['trade_date', 'ts_code'],
    )
    tl['trade_date'] = pd.to_datetime(tl['trade_date'], format='mixed')
    tl['flag'] = 1
    tl = tl.drop_duplicates(subset=['trade_date', 'ts_code'])

    px = pd.read_csv(
        DATA_DIR / 'raw' / 'daily_prices.csv',
        dtype={'ts_code': str, 'trade_date': str},
        usecols=['ts_code', 'trade_date', 'close'],
    )
    px['trade_date'] = pd.to_datetime(px['trade_date'], format='mixed')
    px = px.drop_duplicates(subset=['ts_code', 'trade_date'])

    # 全交易日 × 全股票矩阵 (上市股票全集)
    dates = sorted(px['trade_date'].unique())
    stocks = sorted(px['ts_code'].unique())
    idx = pd.MultiIndex.from_product([dates, stocks], names=['trade_date', 'ts_code'])
    flag_full = tl.set_index(['trade_date', 'ts_code'])['flag'].reindex(idx).fillna(0)
    # 上市后才有意义: 未上市 (close NaN) 置 NaN
    px_idx = px.set_index(['trade_date', 'ts_code'])['close']
    close_full = px_idx.reindex(idx)
    flag_full = flag_full.where(close_full.notna())

    freq20 = flag_full.groupby('ts_code').rolling(20, min_periods=5).sum().reset_index(level=0, drop=True)
    freq20 = freq20.sort_index()

    merged = pd.DataFrame({'ts_code': idx.get_level_values(1), 'trade_date': idx.get_level_values(0),
                           'freq': freq20.values, 'close': close_full.values})
    merged = merged.sort_values(['ts_code', 'trade_date'])
    merged['fwd5'] = merged.groupby('ts_code')['close'].shift(-5) / merged['close'] - 1
    merged = merged.dropna(subset=['freq', 'fwd5'])
    merged['year'] = merged['trade_date'].dt.year

    seg = {}
    for label, cond in [('2024', merged['year'] == 2024),
                        ('2025', merged['year'] == 2025),
                        ('2026H1', (merged['year'] == 2026))]:
        sub = merged[cond]
        ics = []
        for date, g in sub.groupby('trade_date'):
            if len(g) < 100:
                continue
            ic = g['freq'].rank().corr(g['fwd5'].rank())
            if not np.isnan(ic):
                ics.append(ic)
        if ics:
            mean_ic = float(np.mean(ics))
            icir = abs(mean_ic / np.std(ics)) if np.std(ics) > 0 else 0.0
            seg[label] = {"ic": round(mean_ic, 5), "icir": round(icir, 4), "n_days": len(ics)}
            print(f"[分段IC] {label}: IC={mean_ic:+.5f}  ICIR={icir:.3f}  n={len(ics)}")
        else:
            seg[label] = {"ic": None, "icir": None, "n_days": 0}
    return seg


def compute_seat_performance():
    """③ 分层席位表现影子基线: 各类型席位净买入后 5 日收益"""
    d = pd.read_csv(
        DATA_DIR / 'raw' / 'top_inst.csv',
        dtype={'trade_date': str, 'ts_code': str},
        usecols=['trade_date', 'ts_code', 'exalter', 'net_buy'],
    )
    d['trade_date'] = pd.to_datetime(d['trade_date'], format='mixed')
    d['se_type'] = d['exalter'].apply(classify_seat)

    px = pd.read_csv(
        DATA_DIR / 'raw' / 'daily_prices.csv',
        dtype={'ts_code': str, 'trade_date': str},
        usecols=['ts_code', 'trade_date', 'close'],
    )
    px['trade_date'] = pd.to_datetime(px['trade_date'], format='mixed')
    px = px.drop_duplicates(subset=['ts_code', 'trade_date']).sort_values(['ts_code', 'trade_date'])
    px['fwd5'] = px.groupby('ts_code')['close'].shift(-5) / px['close'] - 1

    d = d.merge(px[['ts_code', 'trade_date', 'fwd5']], on=['ts_code', 'trade_date'], how='left')
    d = d.dropna(subset=['fwd5'])

    out = {}
    for st, g in d.groupby('se_type'):
        # 净买入方向收益: net_buy>0 视为做多信号
        pos = g[g['net_buy'] > 0]['fwd5']
        neg = g[g['net_buy'] <= 0]['fwd5']
        out[st] = {
            "n_events": int(len(g)),
            "fwd5_mean_all": round(float(g['fwd5'].mean()), 5),
            "fwd5_mean_netbuy_pos": round(float(pos.mean()), 5) if len(pos) else None,
            "fwd5_mean_netbuy_neg": round(float(neg.mean()), 5) if len(neg) else None,
        }
        print(f"[席位表现] {st}: n={len(g)}  上榜后5日均收益={g['fwd5'].mean():+.4%}  "
              f"(净买={pos.mean() if len(pos) else 0:+.4%} / 净卖={neg.mean() if len(neg) else 0:+.4%})")
    return out


def main():
    print("=" * 64)
    print("[P-004] ① 席位类型成交占比漂移")
    seat_share = compute_seat_share()
    print()
    print("[P-004] ② lhb_freq_20d 分段 IC (2025 断点检验)")
    seg_ic = compute_lhb_segment_ic()
    print()
    print("[P-004] ③ 分层席位表现影子基线 (v3 分层评分输入)")
    seat_perf = compute_seat_performance()

    # 断点结论
    ic24 = seg_ic.get('2024', {}).get('ic')
    ic25 = seg_ic.get('2025', {}).get('ic')
    ic26 = seg_ic.get('2026H1', {}).get('ic')
    breakpoint = "unknown"
    if ic24 is not None and ic25 is not None:
        delta = abs(ic25 - ic24)
        flip = (ic24 > 0) != (ic25 > 0)
        breakpoint = ("flip" if flip else f"shift_{delta:.4f}") if (flip or delta > 0.02) else "stable"
    print()
    print(f"[断点结论] 2024→2025 IC 变化: {ic24} → {ic25}, 断点类型={breakpoint}")

    out = {
        "generated_at": datetime.now().isoformat(),
        "proposal": "P-20260902-004",
        "seat_share_by_type": seat_share,
        "lhb_freq_20d_segment_ic": seg_ic,
        "breakpoint": breakpoint,
        "seat_performance_shadow": seat_perf,
        "note": "v3 分层评分为影子并行, v2 评分不动",
    }
    OUT_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"  → {OUT_FILE}")


if __name__ == '__main__':
    main()
