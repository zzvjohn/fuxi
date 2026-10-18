"""
专用: 测试新加入的 3 个研究提炼因子 (ARBR_26 / trend_alignment_ma / momentum_fragility)
来源: 2026-06-28 每日研究 → 东方财富/广发期货/CSDN
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import importlib
import daily_factor_hypothesis as dfh
from datetime import datetime

# ✅ 确保新因子已在库中: novel_114, 115, 116
target_names = ['ARBR_26', 'trend_alignment_ma', 'momentum_fragility']

# 验证因子在库中
lib_names = [f['name'] for f in dfh.NOVEL_FACTOR_LIBRARY]
for tn in target_names:
    if tn not in lib_names:
        print(f"❌ {tn} 不在 NOVEL_FACTOR_LIBRARY 中!")
        sys.exit(1)

# 提取目标因子定义
target_factors = [f for f in dfh.NOVEL_FACTOR_LIBRARY if f['name'] in target_names]
for tf in target_factors:
    tf['source'] = 'research_extracted'

print("=" * 70)
print(f"  研究提炼因子专用测试 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"  来源: 2026-06-28 每日研究 (东方财富ARBR / 广发期货CTA / CSDN动量崩溃)")
print("=" * 70)

# Step 1: 加载数据
print("\n[Step 1] 加载数据...")
close_df, open_df, high_df, low_df, vol_df = dfh.load_data(n_days=270)
forward_5d = close_df.shift(-5) / close_df - 1

# Step 2: 测试每个因子
print(f"\n[Step 2] IC评估 ({len(target_factors)} 个因子)...")
results = []

for i, hyp in enumerate(target_factors):
    print(f"\n  [{i+1}/{len(target_factors)}] {hyp['label']} ({hyp['name']})")
    print(f"  类别: {hyp['category']}")
    print(f"  假设: {hyp['hypothesis']}")
    print(f"  逻辑: {hyp['logic'][:120]}...")
    print(f"  公式: {hyp['formula'][:100]}...")

    result = dfh.evaluate_factor(hyp, close_df, open_df, high_df, low_df, vol_df, forward_5d)
    results.append(result)

    # 标记已测
    dfh.mark_as_tested(hyp['name'], {
        'label': hyp['label'],
        'category': hyp.get('category', ''),
        'icir': result.get('icir', float('nan')),
        'ic_mean': result.get('ic_mean', float('nan')),
        '+ic_pct': result.get('+ic%', float('nan')),
        'n_dates': result.get('n_dates', 0),
    })

    if 'error' in result:
        print(f"    [ERROR] {result['error']}")
        continue

    icir_ok = not (result['icir'] is None or (isinstance(result['icir'], float) and result['icir'] != result['icir']))
    if icir_ok:
        passed = (result['icir'] >= dfh.ICIR_THRESHOLD and result['+ic%'] >= dfh.IC_PCT_THRESHOLD)
    else:
        passed = False

    status = "✅ PASS → candidate (触发FA)" if passed else "📦 RESERVE → 储备池"
    print(f"    IC_mean: {result['ic_mean']:+.4f}  |  ICIR: {result['icir']:+.3f}  |  +IC%: {result['+ic%']:.1%}  |  n={result.get('n_dates', 0)}截面")
    print(f"    结论: {status}")

    # 全量入库
    dfh.add_to_pool(result, passed=passed)
    if passed:
        result['fa_trigger'] = True

# Step 3: 汇总
print("\n" + "=" * 70)
print("[汇总] 研究提炼因子测试结果")
print("=" * 70)

valid = [r for r in results if 'error' not in r and 'icir' in r]
print(f"\n{'因子':28s} {'类别':20s} {'ICIR':>7s} {'IC_mean':>8s} {'+IC%':>7s} {'结果':>8s}")
print("-" * 85)
for r in valid:
    if r['icir'] is None or (isinstance(r['icir'], float) and r['icir'] != r['icir']):
        continue
    passed = r['icir'] >= dfh.ICIR_THRESHOLD and r['+ic%'] >= dfh.IC_PCT_THRESHOLD
    status = '✅ PASS' if passed else '📦 RESERVE'
    print(f"{r['label']:28s} {r.get('category', ''):20s} {r['icir']:+7.3f} {r['ic_mean']:+7.4f} {r['+ic%']:6.1%} {status:>8s}")

n_pass = sum(1 for r in valid if r['icir'] >= dfh.ICIR_THRESHOLD and r['+ic%'] >= dfh.IC_PCT_THRESHOLD)
n_reserve = len(valid) - n_pass
print(f"\n通过: {n_pass}  |  入库储备: {n_reserve}  |  总计: {len(valid)}")

# Step 4: 保存报告
date_str = datetime.now().strftime('%Y%m%d')
report_file = dfh.DAILY_REPORT_DIR / f'research_extracted_{date_str}.csv'
import pandas as pd
clean = []
for r in valid:
    cr = {k: v for k, v in r.items() if k != 'ic_series'}
    cr['date'] = date_str
    clean.append(cr)
pd.DataFrame(clean).to_csv(report_file, index=False, encoding='utf-8-sig')
print(f"\n报告: {report_file}")

if n_pass > 0:
    print("\n🎉 新通过因子 → 需要注册到FA系统!")
    for r in valid:
        if r['icir'] >= dfh.ICIR_THRESHOLD and r['+ic%'] >= dfh.IC_PCT_THRESHOLD:
            print(f"  ✅ {r['label']} (ICIR={r['icir']:.3f}, +IC%={r['+ic%']:.1%})")
else:
    print("\n📦 全部入库储备池 — 暂无新candidate触发FA")
