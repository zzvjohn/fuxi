"""
Pipeline修复脚本: 同步 Experience Memory + injected_factors.json + experiment_state.json + 缓存重建

修复内容:
1. 将7个新注入因子的FRI/Ψ结果写入 Experience Memory
2. 更新 injected_factors.json (追加7条新记录)
3. 同步 experiment_state.json (injection_countdown, injected_factors_count, last_evolution)
4. 因子池缓存对齐重建 (解决213@5531 + 6@5710的混搭问题)
"""
import sys, os, json, pickle, time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'research' / 'factor_alchemy'))

# ── 加载注入轮次结果 ──────────────────────────────────────
with open(PROJECT_ROOT / 'data' / 'injection_round_20260803.json', 'r', encoding='utf-8') as f:
    injection_results = json.load(f)

print("=" * 60)
print("Pipeline 修复脚本 — 2026-08-03")
print("=" * 60)

# ── Step 1: 更新 Experience Memory ────────────────────────
print("\n[Step 1] 更新 Experience Memory...")
from experience_memory import get_memory

CAT2PAR = {
    'volume_structure': '流动性', 'liquidity_micro': '微观结构',
    'return_decomposition': '隔夜', 'momentum_dynamics': '趋势/动量',
    'price_pattern': '反转', 'volatility_structure': '波动率',
    'return_distribution': '尾部风险', 'behavioral': '行为金融',
    'information_flow': '微观结构', 'extreme_events': '尾部风险',
    'event_driven': '基本面/成长', 'multi_period': '趋势/动量',
    'relative_value': '基本面/成长', 'trend_quality': '趋势/动量',
    'growth_quality': '基本面/成长', 'mid_report_divergence': '趋势/动量',
    '学术异象库': '尾部风险', 'ml_inspired_construction': '趋势/动量',
}

# 读取CSV获取因子元数据 (category, icir, formula, direction)
pool_df = pd.read_csv(PROJECT_ROOT / 'data' / 'passed_factor_pool.csv', encoding='utf-8-sig')

mem = get_memory()
new_mem_records = 0
for r in injection_results:
    name = r['name']
    if r['status'] != 'PASS':
        continue
    
    # 获取CSV元数据
    csv_row = pool_df[pool_df['name'] == name]
    if csv_row.empty:
        print(f"  ⚠️ {name}: CSV中未找到, 跳过")
        continue
    
    row = csv_row.iloc[0]
    category = str(row.get('category', ''))
    paradigm = CAT2PAR.get(category, '通用')
    icir_val = float(row.get('icir', row.get('daily_icir', 0))) if pd.notna(row.get('icir', row.get('daily_icir', np.nan))) else 0.0
    formula = str(row.get('formula', ''))
    direction = row.get('direction', 'long')
    
    fri_score = r['fri']
    outcome = "PASS" if fri_score >= 0.4 else "WEAK"
    
    mem.record(
        factor_name=name,
        formula=formula,
        paradigm=paradigm,
        category=category,
        fri_score=fri_score,
        fri_grade=r['grade'],
        fri_precision=r['precision'],
        fri_persistence=r['persistence'],
        fri_consistency=r['consistency'],
        fri_novelty=r['novelty'],
        icir=icir_val,
        outcome=outcome,
        psi_r_squared=r['psi_r2'],
        psi_independence=1.0 - r['psi_r2'],
        max_corr_factor=f"r={r.get('max_corr', 0):.3f}",
    )
    new_mem_records += 1
    print(f"  ✅ {name}: paradigm={paradigm} outcome={outcome}")

s = mem.get_summary()
print(f"  Memory Summary: total={s['total_attempts']} pass_rate={s['pass_rate']} "
      f"toxic={s['toxic_patterns']} successful={s['successful_patterns']}")

# ── Step 2: 更新 injected_factors.json ────────────────────
print("\n[Step 2] 更新 injected_factors.json...")
inj_path = PROJECT_ROOT / 'data' / 'injected_factors.json'
with open(inj_path, 'r', encoding='utf-8') as f:
    existing_inj = json.load(f)

existing_names = {e['factor_name'] for e in existing_inj['injected']}
new_injected = 0

for r in injection_results:
    name = r['name']
    if r['status'] != 'PASS' or name in existing_names:
        continue
    
    csv_row = pool_df[pool_df['name'] == name]
    if csv_row.empty:
        continue
    row = csv_row.iloc[0]
    
    injected_record = {
        'factor_name': name,
        'factor_label': str(row.get('label', name)),
        'category': str(row.get('category', '')),
        'date': '2026-08-03',
        'icir': float(row.get('icir', row.get('daily_icir', 0))) if pd.notna(row.get('icir', row.get('daily_icir', np.nan))) else 0.0,
        'ic_pct': float(row.get('+ic%', row.get('+ic_pct', 0))) if pd.notna(row.get('+ic%', row.get('+ic_pct', np.nan))) else 0.0,
        'fri': r['fri'],
        'fri_grade': r['grade'],
        'fri_precision': r['precision'],
        'fri_persistence': r['persistence'],
        'fri_consistency': r['consistency'],
        'fri_novelty': r['novelty'],
        'max_corr_with_existing': r.get('max_corr', 0),
        'max_corr_factor': '',
        'psi_r_squared': r['psi_r2'],
        'psi_independence': 1.0 - r['psi_r2'],
        'psi_n_orthogonalized_against': 0,
        'psi_orthogonalized_factor_names': [],
        'psi_rejection_recommended': False,
        'injected_at': '2026-08-03T16:50:00.000000',
    }
    existing_inj['injected'].append(injected_record)
    new_injected += 1
    print(f"  ✅ {name}: FRI={r['fri']:.3f}/{r['grade']}")

existing_inj['total_injected'] = len(existing_inj['injected'])
existing_inj['last_updated'] = datetime.now().isoformat()
inj_path.parent.mkdir(parents=True, exist_ok=True)
with open(inj_path, 'w', encoding='utf-8') as f:
    json.dump(existing_inj, f, ensure_ascii=False, indent=2)
print(f"  injected_factors.json: {existing_inj['total_injected']} 总注入 (+{new_injected})")

# ── Step 3: 同步 experiment_state.json ───────────────────
print("\n[Step 3] 同步 experiment_state.json...")
state_path = PROJECT_ROOT / 'research' / 'factor_alchemy' / 'experiment_state.json'
with open(state_path, 'r', encoding='utf-8') as f:
    state = json.load(f)

state['injected_factors_count'] = existing_inj['total_injected']

# 更新 evolution_state
evo = state.setdefault('evolution_state', {})
evo['new_factors_since_last_round'] = new_injected + evo.get('new_factors_since_last_round', 0)
evo['injected_factors_count'] = existing_inj['total_injected']

# 累积触发倒计时: 之前=3 (因上次已触发), 现在应调整为注入总数
# 上次触发是 2026-08-02, 之后新因子=7
old_countdown = evo.get('trigger_countdown', 3)
evo['trigger_countdown'] = old_countdown + new_injected
evo['_note'] = f"2026-08-03 修复: +{new_injected}因子注入, countdown={evo['trigger_countdown']}"

state['_updated'] = datetime.now().isoformat()

with open(state_path, 'w', encoding='utf-8') as f:
    json.dump(state, f, ensure_ascii=False, indent=2)
print(f"  experiment_state: injected={state['injected_factors_count']} "
      f"countdown={evo['trigger_countdown']} new_since_last={evo['new_factors_since_last_round']}")

# ── Step 4: 重建因子池缓存 (解决形状混搭) ────────────────
print("\n[Step 4] 重建因子池缓存 (解决 213@5531 + 6@5710 混搭)...")
cache_path = PROJECT_ROOT / 'data' / 'cache' / 'existing_factor_pool.pkl'

# 加载数据
daily = pd.read_csv(
    PROJECT_ROOT / 'data' / 'raw' / 'daily_prices.csv',
    dtype={'ts_code': str, 'trade_date': str},
    usecols=['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol']
)
daily['trade_date'] = pd.to_datetime(daily['trade_date'], format='mixed')
today = datetime.now()
end_dt = pd.Timestamp(today.strftime('%Y-%m-%d'))
start_dt = pd.Timestamp((today - pd.Timedelta(days=270)).strftime('%Y-%m-%d'))
daily = daily[(daily['trade_date'] >= start_dt) & (daily['trade_date'] <= end_dt)]
daily = daily.drop_duplicates(subset=['ts_code', 'trade_date']).sort_values(['ts_code', 'trade_date'])

close_df = daily.pivot(index='trade_date', columns='ts_code', values='close')
open_df = daily.pivot(index='trade_date', columns='ts_code', values='open')
high_df = daily.pivot(index='trade_date', columns='ts_code', values='high')
low_df = daily.pivot(index='trade_date', columns='ts_code', values='low')
vol_df = daily.pivot(index='trade_date', columns='ts_code', values='vol')

min_days = 40
valid = close_df.columns[close_df.count() >= min_days]
close_df = close_df[valid]; open_df = open_df[valid]
high_df = high_df[valid]; low_df = low_df[valid]; vol_df = vol_df[valid]

print(f"  数据: {close_df.shape} (日期 x 股票)")

# 重建缓存
from fri import build_existing_pool_from_csv

pool_csv = PROJECT_ROOT / 'data' / 'passed_factor_pool.csv'
existing = build_existing_pool_from_csv(pool_csv, close_df, open_df, high_df, low_df, vol_df)

if existing:
    cache_key = {
        'csv_mtime': pool_csv.stat().st_mtime,
        'n_dates': len(close_df),
        'n_stocks': len(close_df.columns),
    }
    cached = dict(existing)
    cached['_meta'] = {'cache_key': cache_key, 'built_at': datetime.now().isoformat()}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(cached, f)
    
    # 验证形状一致性
    from collections import Counter
    shapes = Counter()
    for k, v in existing.items():
        if hasattr(v, 'shape'):
            shapes[v.shape] += 1
    print(f"  ✅ 缓存重建: {len(existing)} 因子, 形状: {dict(shapes)}")
    print(f"     文件大小: {cache_path.stat().st_size / 1024 / 1024:.0f}MB")
else:
    print("  ❌ 缓存重建失败")

# ── Step 5: 运行 fixed factor_injector.py 验证 ─────────────
print("\n[Step 5] 运行 factor_injector.py (验证修复)...")

# 直接调用 (避免 subprocess)
import runpy
# 先静默检查: 是否还有 candidate
pool_check = pd.read_csv(pool_csv, encoding='utf-8-sig')
candidates = pool_check[pool_check['status'] == 'candidate']
candidate_names = set(candidates['name'].values)
already_inj = {e['factor_name'] for e in existing_inj['injected']}
pending = candidate_names - already_inj

print(f"  Candidate: {len(candidates)} 个 → {list(candidate_names)}")
print(f"  已注入: {len(already_inj)} 个")
print(f"  待注入: {len(pending)} 个 → {list(pending) if pending else '无'}")

if pending:
    print("  ⚠️ 仍有待注入因子, 运行 factor_injector.py...")
    os.chdir(PROJECT_ROOT)
    # 要运行一次完整注入
    from scripts.factor_injector import run_injection, load_data, load_regime_labels
    close_df2, open_df2, high_df2, low_df2, vol_df2 = load_data()
    regime_labels = load_regime_labels()
    
    result = run_injection(
        pool_csv_path=pool_csv,
        close_df=close_df2,
        open_df=open_df2,
        high_df=high_df2,
        low_df=low_df2,
        vol_df=vol_df2,
        regime_labels=regime_labels,
        dry_run=False,
        all_factors=False,
        verbose=True,
    )
    print(f"\n  注入结果: {result.get('injected_count', 0)}/{result.get('candidates_found', 0)}")
else:
    print("  ✅ 所有 candidate 因子已注入, 无需重复运行")
    # 但还是要确认注入记录中有最新的7个
    final_inj_count = len(existing_inj['injected'])
    print(f"  最终注入总数: {final_inj_count}")
    
    # 如果有7个新注入且 factor_injector 还没跑, 现在补跑一次让它更新 augmenter 状态
    if new_injected > 0:
        print("  补跑 factor_injector.py 以更新 ParadigmAugmenter 状态...")
        from scripts.factor_injector import run_injection, load_data, load_regime_labels
        close_df2, open_df2, high_df2, low_df2, vol_df2 = load_data()
        regime_labels = load_regime_labels()
        
        result = run_injection(
            pool_csv_path=pool_csv,
            close_df=close_df2,
            open_df=open_df2,
            high_df=high_df2,
            low_df=low_df2,
            vol_df=vol_df2,
            regime_labels=regime_labels,
            dry_run=False,
            all_factors=False,
            verbose=True,
        )
        print(f"  结果: {result.get('status', '?')} - {result.get('message', str(result.get('injected_count', '?')))}")
        if result.get('should_trigger_evolution'):
            print(f"  ⚡ should_trigger_evolution = TRUE")

print(f"\n{'='*60}")
print("Pipeline 修复完成!")
print(f"{'='*60}")
