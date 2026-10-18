"""
标注历史 regime 标签, 验证划分合理性

用法: python label_regimes.py
输出: data/regime_labels.csv (日期+标签)
      data/regime_stats.csv   (分布统计)
      终端打印 regime timeline + JQ策略对照
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from pathlib import Path
from regime import RegimeDetector, load_and_label

DATA_RAW = Path(__file__).parent.parent.parent / 'data' / 'raw'


def main():
    # 1. 加载 + 标注
    print("=" * 60)
    print("Regime 历史标注")
    print("=" * 60)

    detector, labels = load_and_label(
        DATA_RAW, start='2021-01-01', end='2026-07-17',
        detail=False  # MVP: base regimes only
    )

    # 2. 保存
    out_dir = Path(__file__).parent / 'data'
    out_dir.mkdir(exist_ok=True)
    labels.to_csv(out_dir / 'regime_labels.csv', header=True)
    detector.stats().to_csv(out_dir / 'regime_stats.csv')

    print(f"\n  标签已保存: {out_dir / 'regime_labels.csv'}")
    print(f"  统计已保存: {out_dir / 'regime_stats.csv'}")

    # 3. 分布统计
    stats = detector.stats()
    print(f"\n{'='*60}")
    print("Regime 分布统计")
    print(f"{'='*60}")
    print(stats.to_string())

    # 4. 逐年分布
    print(f"\n{'='*60}")
    print("逐年 Regime 占比")
    print(f"{'='*60}")
    yearly = labels.groupby(labels.index.year).apply(
        lambda x: dict(x.value_counts())
    )
    for yr, dist_dict in yearly.items():
        if not isinstance(dist_dict, dict):
            continue
        total = sum(dist_dict.values())
        parts = [f"  {reg}: {cnt}周({cnt/total*100:.0f}%)"
                 for reg, cnt in sorted(dist_dict.items())]
        print(f"  {yr}: {' | '.join(parts)}")

    # 5. Regime 切换 timeline (每年一个摘要行)
    print(f"\n{'='*60}")
    print("关键切换点")
    print(f"{'='*60}")
    prev = None
    switches = []
    for dt, reg in labels.items():
        if reg != prev and prev is not None:
            switches.append((dt, prev, reg))
        prev = reg

    # 只显示切换超过8周的
    for dt, old, new in switches:
        print(f"  {dt.strftime('%Y-%m-%d')}: {old} → {new}")

    # 6. 与 JQ 回测结果对照
    print(f"\n{'='*60}")
    print("JQ 回测结果对照 (已知事实)")
    print(f"{'='*60}")

    # 分 regime 看各时间段
    for regime_name in ['small_bull', 'small_bear', 'large_dominant', 'neutral']:
        r_dates = labels[labels == regime_name].index
        if len(r_dates) == 0:
            continue
        periods = f"{r_dates[0].strftime('%Y-%m')} → {r_dates[-1].strftime('%Y-%m')}"
        print(f"\n  [{regime_name}] {len(r_dates)}周 | {periods}")
        print(f"    high_quality 预期表现: ", end='')
        if regime_name == 'small_bull':
            print("↑↑ 最佳 (流动性因子主场)")
        elif regime_name == 'small_bear':
            print("↓ 不确定 (风格矛盾)")
        elif regime_name == 'large_dominant':
            print("↓ 弱 (大盘风格下流动性因子失效)")
        else:
            print("→ 中性")

    # 7. 对比: 2024年前 vs 2024年后
    print(f"\n{'='*60}")
    print("2024年前后对比 (high_quality 的 regime 切换)")
    print(f"{'='*60}")
    pre_2024 = labels[labels.index < '2024-01-01']
    post_2024 = labels[labels.index >= '2024-01-01']

    for period_name, period_labels in [("2021-2023", pre_2024), ("2024-2026", post_2024)]:
        if len(period_labels) == 0:
            continue
        counts = period_labels.value_counts()
        total = len(period_labels)
        print(f"\n  {period_name} ({total}周):")
        for reg, cnt in counts.items():
            pct = cnt / total * 100
            bar = '█' * int(pct / 2)
            print(f"    {reg:20s} {cnt:3d}周 ({pct:4.0f}%) {bar}")

    # 8. V2 精细版 (加波动率)
    print(f"\n{'='*60}")
    print("V2 精细版 (base + vol)")
    print(f"{'='*60}")
    _, labels_detail = load_and_label(
        DATA_RAW, start='2021-01-01', end='2026-07-17',
        detail=True
    )
    detail_stats = labels_detail.value_counts()
    for reg, cnt in detail_stats.items():
        pct = cnt / len(labels_detail) * 100
        print(f"  {reg:25s} {cnt:3d}周 ({pct:4.1f}%)")

    # 9. 数据充足性检查
    print(f"\n{'='*60}")
    print("训练充足性检查 (每 regime 需 ≥ 50 周)")
    print(f"{'='*60}")
    all_ok = True
    for reg, cnt in detail_stats.items():
        ok = "✓" if cnt >= 50 else "✗ 不足!"
        if cnt < 50:
            all_ok = False
            print(f"  {ok} {reg:25s} 仅 {cnt} 周")
    if all_ok:
        print("  全部 regime ≥ 50 周 ✓ 足够 GA 训练")
    else:
        print("  ⚠ 部分 regime 样本不足, 需合并或降低阈值")


if __name__ == '__main__':
    main()
