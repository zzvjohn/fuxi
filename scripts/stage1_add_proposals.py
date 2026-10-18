"""Stage 1: Add new factor proposals to the pool"""
import json
import sys

# Load existing proposals
with open('e:/quant/data/stage1_factor_proposals.json', 'r') as f:
    existing = json.load(f)

existing_names = {p['name'] for p in existing.get('proposals', [])}
print(f'Existing names: {len(existing_names)}')

new_factors = [
    {
        'name': 'small_cap_turnover_price_divergence',
        'label': '小盘换手-价格背离因子',
        'category': 'small_bull_breakthrough',
        'hypothesis': '小盘股中，成交量放大但价格未同步上涨，反映主力资金在底部吸筹，后续将补涨',
        'logic': '小盘股换手率加速(短窗/长窗)但价格涨幅滞后(滞后期涨幅<同期换手涨幅)，捕捉主力隐蔽吸筹期。仅作用于小市值(top 30% by cap)，低换手时平仓。',
        'formula': 'import numpy as np\nsmall_cap_mask = close_p.rank(axis=1, pct=True) <= 0.3\nturnover_proxy = volume_p / close_p\nto_ratio = turnover_proxy.rolling(10, min_periods=5).mean() / turnover_proxy.rolling(60, min_periods=20).mean().replace(0, np.nan)\nprice_ret_5d = close_p.pct_change(5)\nprice_ret_20d = close_p.pct_change(20)\nto_accel = to_ratio > to_ratio.rolling(5, min_periods=3).mean()\nprice_lag = (price_ret_5d < price_ret_5d.median()) & (to_ratio > 1.2)\nfactor = to_ratio.where(small_cap_mask & to_accel & price_lag, 0)',
        'direction': 'long',
        'source': 'stage1_exploration',
        'exploration_basis': '小盘突围因子-筹码吸筹量价背离',
        'target_problem': 'small_bull_regime_no_solution',
        'compatible_regime': 'small_bull',
        'expected_turnover': 'medium',
        'related_blacklist': '无',
        'academic_reference': 'Chordia, Subrahmanyam & Anshuman (2001) Trading activity and expected stock returns, JFE'
    },
    {
        'name': 'monthly_value_momentum_hybrid',
        'label': '月频价值动量混合因子',
        'category': 'low_turnover_mid_freq',
        'hypothesis': '月频视角下，低估值的动量股票有持续性alpha——价值提供安全边际，动量提供趋势确认，天然低换手',
        'logic': '每月计算value得分(1/近端价格相对远端均价) * momentum得分(6个月收益/波动率)，两者共振时做多。月频天然降换手至<0.3。',
        'formula': 'import numpy as np\nma_120 = close_p.rolling(120, min_periods=60).mean()\nvalue_score = 1 / (close_p / ma_120.replace(0, np.nan)).replace(0, np.nan)\nret_6m = close_p.pct_change(126).fillna(0)\nvol_6m = close_p.pct_change().rolling(126, min_periods=40).std().replace(0, np.nan)\nmomentum_score = ret_6m / (vol_6m + 1e-6)\nv_rank = value_score.rank(axis=1, pct=True)\nm_rank = momentum_score.rank(axis=1, pct=True)\nfactor = v_rank * m_rank',
        'direction': 'long',
        'source': 'stage1_exploration',
        'exploration_basis': '低换手中低频因子-月频价值动量',
        'target_problem': 'high_turnover',
        'compatible_regime': 'both',
        'expected_turnover': 'low',
        'related_blacklist': '无',
        'academic_reference': 'Asness, Moskowitz & Pedersen (2013) Value and Momentum Everywhere, JF'
    },
    {
        'name': 'hl_volatility_spread_regime_stable',
        'label': '高低价波动率价差因子',
        'category': 'vm_diff_replacement',
        'hypothesis': 'HL波动率与收盘波动率的价差在regime切换时更稳定——HL波动包含日内全路径信息，比仅用收盘价的vm_diff对市场结构变化更robust',
        'logic': 'Parkinson(HL)波动率估计与收盘-收盘波动率的差异。当HL波动率系统性高于收盘波动率（日内振幅大但收盘回归），反映市场分歧加大但未定价，后续有方向性行情。替代vm_diff的量价差思路但更regime-stable。',
        'formula': 'import numpy as np\nhl_ratio = np.log(high_p / low_p.replace(0, np.nan))\nparkinson_vol = np.sqrt(1 / (4 * np.log(2)) * hl_ratio**2).rolling(20, min_periods=10).mean()\nclose_vol = close_p.pct_change().rolling(20, min_periods=10).std()\nvol_spread = (parkinson_vol - close_vol) / (parkinson_vol + close_vol + 1e-6)\nvol_spread_smooth = vol_spread.rolling(5, min_periods=3).mean()\nfactor = vol_spread_smooth',
        'direction': 'long',
        'source': 'stage1_exploration',
        'exploration_basis': 'vm_diff替代-regime稳定波动率价差',
        'target_problem': 'vm_diff_replacement',
        'compatible_regime': 'both',
        'expected_turnover': 'medium',
        'related_blacklist': 'vm_diff(替代)',
        'academic_reference': 'Parkinson (1980) The extreme value method for estimating the variance of the rate of return, JB'
    },
    {
        'name': 'sector_relative_strength_consistency',
        'label': '行业相对强度一致性因子',
        'category': 'regime_bridging',
        'hypothesis': '行���内持续跑赢同业的个股，具有良好的alpha持续性——行业中性化使因子在small_bull和large_dominant均有效',
        'logic': '计算股票在行业内的超额收益(20日)，并计算该超额收益滚动一致性(连续跑赢的窗口比例)。行业中性化排除了板块轮动的噪声，专注选股能力。',
        'formula': 'import numpy as np\nret_1d = close_p.pct_change()\nret_20d = close_p.pct_change(20)\ncap_rank = close_p.rank(axis=1, pct=True)\nsector_group = (cap_rank * 10).astype(int)\nsector_ret = ret_20d.copy()\nfor g in range(10):\n    mask = sector_group == g\n    if mask.sum().sum() > 0:\n        ranked = ret_20d.where(mask, np.nan).rank(axis=1, pct=True)\n        sector_ret = sector_ret.where(~mask, ranked.where(mask, sector_ret))\npos_frac = (ret_1d > 0).rolling(20, min_periods=10).mean()\nconsistency = sector_ret.rolling(5, min_periods=3).apply(lambda x: (x > 0).mean(), raw=True)\nfactor = pos_frac * consistency',
        'direction': 'long',
        'source': 'stage1_exploration',
        'exploration_basis': 'regime桥接-行业中性化相对强度',
        'target_problem': 'regime_concentrated',
        'compatible_regime': 'both',
        'expected_turnover': 'medium',
        'related_blacklist': '无',
        'academic_reference': 'Moskowitz & Grinblatt (1999) Do industries explain momentum?, JF'
    },
    {
        'name': 'volume_pressure_asymmetry_score',
        'label': '量压非对称性得分因子',
        'category': 'ml_inspired',
        'hypothesis': '上涨日与下跌日的成交量分布不对称——涨多跌少(上涨日放量/下跌日缩量)的股票是机构在控盘吸筹，alpha持续性强',
        'logic': '将日成交量按涨跌日分组，计算上涨日平均量/下跌日平均量的比率及比率稳定性。类似对比学习中的正负样本对比思想——上涨量和下跌量的对比度反映资金方向。',
        'formula': 'import numpy as np\nret_1d = close_p.pct_change()\nvol = volume_p\nup_mask = ret_1d > 0\ndown_mask = ret_1d < 0\nup_vol = vol.where(up_mask, np.nan).rolling(20, min_periods=5).mean()\ndown_vol = vol.where(down_mask, np.nan).rolling(20, min_periods=5).mean()\nvol_ratio = up_vol / down_vol.replace(0, np.nan)\nup_days = up_mask.rolling(20, min_periods=10).sum()\ntotal_days = (~ret_1d.isna()).rolling(20, min_periods=10).sum()\nup_pct = up_days / total_days.replace(0, np.nan)\nfactor = vol_ratio * up_pct',
        'direction': 'long',
        'source': 'stage1_exploration',
        'exploration_basis': 'ML启发-对比学习启发的量价非对称',
        'target_problem': 'phantom_high',
        'compatible_regime': 'both',
        'expected_turnover': 'medium',
        'related_blacklist': '无',
        'academic_reference': None
    },
    {
        'name': 'momentum_skewness_composite',
        'label': '动量偏度复合因子',
        'category': 'academic_anomaly',
        'hypothesis': '正偏度的动量(稳步上涨)优于负偏度的动量(脉冲式暴涨后回调)——偏度作为动量质量过滤器，可降低V2的MaxDD问题',
        'logic': '动量 * 收益偏度：正偏度(收盘价稳步上升，日收益分布右偏)的动量 = 机构持续吸筹；负偏度(少数几天暴涨拉高动量) = 游资脉冲易回调。偏度过滤掉假动量信号。',
        'formula': 'import numpy as np\nret_1d = close_p.pct_change()\nret_60d = close_p.pct_change(60)\nret_std = ret_1d.rolling(60, min_periods=30).std().replace(0, np.nan)\nskew_60d = ret_1d.rolling(60, min_periods=30).skew()\nmom_norm = ret_60d / (ret_std * np.sqrt(60) + 1e-6)\nskew_clipped = skew_60d.clip(-1, 2)\nfactor = mom_norm * (1 + skew_clipped)',
        'direction': 'long',
        'source': 'stage1_exploration',
        'exploration_basis': '学术异象-动量偏度复合',
        'target_problem': 'maxdd_reduction',
        'compatible_regime': 'both',
        'expected_turnover': 'medium',
        'related_blacklist': '无',
        'academic_reference': 'Harvey & Siddique (2000) Conditional skewness in asset pricing tests, JF; Barroso & Santa-Clara (2015) Momentum has its moments, JFE'
    }
]

# Check for duplicates
all_conflicts = []
for f in new_factors:
    if f['name'] in existing_names:
        all_conflicts.append(f['name'])

if all_conflicts:
    print(f'CONFLICTS: {all_conflicts}')
    sys.exit(1)

print('All new factors are unique!')

# Add new proposals
existing['proposals'].extend(new_factors)
existing['_updated'] = '2026-07-28T13:20:00'
existing['_total'] = len(existing['proposals'])

# Validate JSON
json_str = json.dumps(existing, ensure_ascii=False, indent=2)
json.loads(json_str)  # re-parse to validate

# Write back
with open('e:/quant/data/stage1_factor_proposals.json', 'w', encoding='utf-8') as f:
    f.write(json_str)

print(f'Total proposals now: {len(existing["proposals"])}')
print('JSON validation: OK')
for f in new_factors:
    print(f'  + {f["name"]} [{f["category"]}] -> {f["target_problem"]}')
