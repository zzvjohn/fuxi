"""Stage 1 Factor Proposal Generator - 2026-07-21"""
import json, re
from pathlib import Path
_PROJ = Path(__file__).resolve().parent.parent

# ============ DEDUP: load all existing factor names ============
all_existing = set()

# 1. tested_factors.json
try:
    with open(_PROJ / 'data' / 'tested_factors.json') as f:
        tested = json.load(f)
    names = tested.get('names', tested) if isinstance(tested, dict) else tested
    if isinstance(names, list):
        all_existing.update(names)
    elif isinstance(names, dict):
        all_existing.update(names.keys())
except:
    pass

# 2. passed_factor_pool.csv
try:
    import pandas as pd
    df = pd.read_csv(_PROJ / 'data' / 'passed_factor_pool.csv')
    if 'name' in df.columns:
        all_existing.update(df['name'].values)
except:
    pass

# 3. factors/__init__.py
try:
    with open(_PROJ / 'factor_alchemy' / 'factors' / '__init__.py') as f:
        content = f.read()
    refs = re.findall(r'(?:"|\')([a-z][a-z0-9_]{4,})(?:"|\')\s*:', content)
    all_existing.update(refs)
except:
    pass

# 4. existing proposals
try:
    with open(_PROJ / 'data' / 'stage1_factor_proposals.json') as f:
        props = json.load(f)
    all_existing.update(p['name'] for p in props.get('proposals', []))
except:
    pass

print(f"Total existing unique names: {len(all_existing)}")

# ============ NEW FACTORS ============
new_factors = [
    {
        "name": "earnings_hl_range_widening",
        "label": "业绩预告窗口HL价差扩张",
        "category": "中报预增分化因子",
        "hypothesis": "业绩预告前5日HL价差收敛的股票隐含信息不对称低，预告后信息冲击小超预期概率大；反之HL扩张则信息提前泄露利好兑现。",
        "logic": "计算5日窗口内High-Low价差变化率，聚合为5日累积HL扩张得分。5日窗口聚合降低单日噪声。",
        "formula": """import numpy as np

# 5日HL价差累积变化: (high-low)/prev_close
hl_spread = (high_p - low_p) / close_p.shift(1).replace(0, np.nan)
hl_spread_5d = hl_spread.rolling(5).mean()
hl_spread_20d_ma = hl_spread.rolling(20).mean()

# 价差扩张 = 当前5日均值超过20日均值
hl_widening = hl_spread_5d - hl_spread_20d_ma

# 5日价格位置: 处于区间高位+价差扩张=利好兑现风险
price_pos_5d = (close_p - low_p.rolling(5).min()) / (high_p.rolling(5).max() - low_p.rolling(5).min() + 1e-6)

# 合成: 价差扩张且在价格高位的股票做空(利好兑现)
factor = -(hl_widening * (price_pos_5d - 0.5))""",
        "direction": "long",
        "source": "stage1_exploration",
        "exploration_basis": "中报预增分化因子"
    },
    {
        "name": "volume_concentration_hhi",
        "label": "成交量价格区间集中度HHI",
        "category": "筹码分层ai因子",
        "hypothesis": "成交量集中在特定价格区间的股票筹码锁定度高、机构持仓稳定，未来上涨概率大；均匀分布则筹码分散。",
        "logic": "将20日收盘价分为上中下三等分区间，计算成交量在各区间的Herfindahl指数。HHI高=成交量集中=筹码锁定，但加底部偏向修正。",
        "formula": """import numpy as np

lookback = 20
roll_max = close_p.rolling(lookback).max()
roll_min = close_p.rolling(lookback).min()
price_pos = (close_p - roll_min) / (roll_max - roll_min + 1e-8)

# 三等分区间
in_top = (price_pos > 0.67).astype(float)
in_mid = ((price_pos > 0.33) & (price_pos <= 0.67)).astype(float)
in_bot = (price_pos <= 0.33).astype(float)

vol_top = (volume_p * in_top).rolling(lookback).sum()
vol_mid = (volume_p * in_mid).rolling(lookback).sum()
vol_bot = (volume_p * in_bot).rolling(lookback).sum()
vol_total = volume_p.rolling(lookback).sum().replace(0, np.nan)

# Herfindahl-Hirschman Index
hhi = (vol_top / vol_total) ** 2 + (vol_mid / vol_total) ** 2 + (vol_bot / vol_total) ** 2

# 修正: 集中在底部区间加分，集中在顶部减分
bottom_concentration = vol_bot / vol_total
factor = hhi * (bottom_concentration - 0.33)""",
        "direction": "long",
        "source": "stage1_exploration",
        "exploration_basis": "筹码分层AI因子"
    },
    {
        "name": "corwin_schultz_spread",
        "label": "Corwin-Schultz买卖价差估计",
        "category": "学术异象库",
        "hypothesis": "Corwin-Schultz(2012, JF)从日度High-Low估计买卖价差。价差大=流动性差=未来收益补偿(Liquidity Premium)；价差小=已充分定价。",
        "logic": "两日High-Low比率方法: beta=sum of [ln(H/L)]^2 over 2 days, gamma=[ln(H_2d/L_2d)]^2, alpha推导, spread=2(e^alpha-1)/(1+e^alpha)。20日均值平滑。",
        "formula": """import numpy as np

# Step 1: daily High-Low log ratio squared
hl_sq = (np.log(high_p / low_p.replace(0, np.nan))) ** 2

# Step 2: 2-day sum (beta)
beta = hl_sq.rolling(2).sum()

# Step 3: 2-day max high / min low (gamma)
h_2d = high_p.rolling(2).max()
l_2d = low_p.rolling(2).min()
gamma = (np.log(h_2d / l_2d.replace(0, np.nan))) ** 2

# Step 4: alpha (Corwin-Schultz 2012 equation A.5-A.8)
denom = 3.0 - 2.0 * np.sqrt(2.0)
sqrt_beta = np.sqrt(beta.clip(lower=0))
term1 = (np.sqrt(2.0) * sqrt_beta - sqrt_beta) / denom
term2 = np.sqrt(gamma.clip(lower=0) / denom)
alpha = term1 - term2

# Step 5: spread estimate
exp_alpha = np.exp(alpha.clip(upper=10))
spread = 2.0 * (exp_alpha - 1.0) / (1.0 + exp_alpha)
spread = spread.clip(lower=0)

# 20-day smoothed for stability
factor = spread.rolling(20).mean()""",
        "direction": "long",
        "source": "stage1_exploration",
        "exploration_basis": "学术异象库"
    },
    {
        "name": "harvey_siddique_coskew",
        "label": "Harvey-Siddique市场协偏度",
        "category": "学术异象库",
        "hypothesis": "Harvey-Siddique(2000, JF)协偏度：与市场下行共偏的股票(负协偏度)承担系统性崩溃风险，应获更高风险补偿；正协偏度股票含彩票特征，未来收益低。",
        "logic": "计算个股收益与等权市场收益平方的标准化协方差(60日滚窗)。负协偏度=crash risk premium=做多；正协偏度=lottery-like=做空。",
        "formula": """import numpy as np
import pandas as pd

ret = close_p.pct_change().fillna(0.0)
mkt_ret_raw = ret.mean(axis=1)
mkt_ret_sq = (mkt_ret_raw ** 2).values.reshape(-1, 1)

lookback = 60

# Denominator: std(ret_i) * mean(mkt_ret^2)
ri_std = ret.rolling(lookback).std()

# mean(mkt_ret^2) - same for all stocks per day, broadcast
mkt_idx = pd.Series(mkt_ret_sq.flatten(), index=ret.index)
rm_sq_mean = mkt_idx.rolling(lookback).mean()
# Broadcast to all columns
rm_sq_mean_df = pd.DataFrame(
    np.tile(rm_sq_mean.values.reshape(-1, 1), (1, ret.shape[1])),
    index=ret.index, columns=ret.columns
)

# Numerator: mean(ret_i * mkt_ret^2)
cross_prod = ret.mul(pd.Series(mkt_ret_sq.flatten(), index=ret.index), axis=0)
num = cross_prod.rolling(lookback).mean()

# Co-skewness
denom = (ri_std * rm_sq_mean_df).replace(0, np.nan)
coskew = num / denom

# Negative coskew = crash risk premium -> long
factor = -coskew.rolling(5).mean()""",
        "direction": "long",
        "source": "stage1_exploration",
        "exploration_basis": "学术异象库"
    },
    {
        "name": "roll_implied_spread",
        "label": "Roll隐含买卖价差",
        "category": "学术异象库",
        "hypothesis": "Roll(1984, JF)从收益率负一阶自协方差推断有效买卖价差。价差大=交易成本高=流动性补偿需求=预期收益高。",
        "logic": "S=2*sqrt(-cov(r_t, r_{t-1}))，仅协方差负时有效(买卖价差的反弹效应)。20日滚动窗口，按收盘价归一化。",
        "formula": """import numpy as np

ret = close_p.pct_change().fillna(0.0)
lookback = 20

# Lag-1 autocovariance: cov(r_t, r_{t-1})
ret_lag = ret.shift(1)
acov = ret.rolling(lookback).cov(ret_lag)

# Only negative autocovariance indicates bid-ask bounce
neg_acov = (-acov).clip(lower=0)

# Roll spread: S = 2 * sqrt(-acov)
spread_raw = 2.0 * np.sqrt(neg_acov)

# Normalize by price level and smooth
factor = (spread_raw / close_p.replace(0, np.nan)).rolling(5).mean()""",
        "direction": "long",
        "source": "stage1_exploration",
        "exploration_basis": "学术异象库"
    }
]

# ============ VALIDATION ============
new_names = {f['name'] for f in new_factors}
dupes = new_names & all_existing
if dupes:
    print(f"ERROR: Duplicate names found: {dupes}")
else:
    print(f"Dedup check PASSED: 0 duplicates with existing {len(all_existing)} names")

required = ['name', 'label', 'category', 'hypothesis', 'logic', 'formula', 'direction', 'source', 'exploration_basis']
for f in new_factors:
    missing = [k for k in required if k not in f]
    if missing:
        print(f"ERROR: {f['name']} missing: {missing}")
    else:
        print(f"  OK: {f['name']} ({f['label']}) [{f['category']}]")

cats = set(f['category'] for f in new_factors)
print(f"Categories covered: {cats} (need >= 3)")

# ============ MERGE AND SAVE ============
# 2026-08-21 并发覆盖防护: 锁内闭环读改写 (原 try/except 裸读写会被并发进程覆盖)
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from proposals_io import load_and_modify as _lmod, save as _psave

def _merge(existing):
    existing['proposals'].extend(new_factors)
    return existing

try:
    merged = _lmod(_merge)
except FileNotFoundError:
    merged = {'proposals': list(new_factors)}
    _psave(merged)

print(f"Total proposals after merge: {len(merged['proposals'])}")
print("Saved to data/stage1_factor_proposals.json")

# Re-validate
with open(_PROJ / 'data' / 'stage1_factor_proposals.json', encoding='utf-8') as f:
    json.load(f)
print("JSON validation PASSED")
