#!/usr/bin/env python3
"""Merge new factor proposals into stage1_factor_proposals.json"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proposals_io import load_and_modify   # 2026-08-21 并发覆盖防护: 锁内闭环读写

new_proposals = [
    {
        "name": "amihud_illiquidity",
        "label": "Amihud非流动性(2002)",
        "category": "academic_anomalies",
        "hypothesis": "Amihud(2002, JFE)非流动性(|return|/dollar_volume)高的股票承担流动性风险应获风险溢价。小盘股Amihud值天然更高→small_bull regime内截面分化强",
        "logic": "经典学术度量(未被本库覆盖)。日频|ret|/成交额,20日均值。高Amihud=低流动性=预期收益补偿。尤其适合small_bull：小盘间流动性差异大→截面rank更有效",
        "formula": "import numpy as np\n\nret_daily = close_p.pct_change().fillna(0)\ndollar_vol = volume_p.mul(close_p, axis=0).replace(0, np.nan)\nilliquidity = ret_daily.abs() / dollar_vol.clip(lower=1e6)\nilliquidity_20 = illiquidity.rolling(20).mean()\nq99 = illiquidity_20.quantile(0.99, axis=1)\nfactor = illiquidity_20.clip(upper=q99.max())",
        "direction": "long",
        "source": "stage1_exploration",
        "exploration_basis": "学术异象库",
        "target_problem": "small_bull_regime_no_solution",
        "compatible_regime": "small_bull",
        "expected_turnover": "low",
        "related_blacklist": "无",
        "academic_reference": "Amihud(2002, JFE) Illiquidity and stock returns"
    },
    {
        "name": "idiosyncratic_momentum",
        "label": "特质动量(Beta剥离)",
        "category": "ml_inspired_construction",
        "hypothesis": "Blitz-Huijman-Swinkels(2011)残差动量：剔除市场Beta后的纯特质动量比原始动量ICIR高50%+。解决基因库中'量价因子small_bull全负'：残差化后不受大小盘风格污染",
        "logic": "用20日滚动Beta剥离市场收益，残差累积=纯特质动量。本质ML思路(降维/特征解耦)但传统OLS实现。与contrastive_sector_pairwise_rank同为去均值类但维度不同(市场vs行业)",
        "formula": "import numpy as np\n\nret = close_p.pct_change().fillna(0)\nmkt = ret.mean(axis=1)\n\nlookback = 20\nprod_sum = ret.mul(mkt, axis=0).rolling(lookback, min_periods=10).sum()\nmkt_sq_sum = (mkt ** 2).rolling(lookback, min_periods=10).sum()\nbeta = prod_sum.div(mkt_sq_sum.replace(0, np.nan), axis=0)\nexpected = beta.mul(mkt, axis=0)\nresidual = ret - expected\nidio_mom = residual.rolling(lookback, min_periods=10).sum()\ncross_std = idio_mom.std(axis=1).replace(0, 1e-6)\ncross_mean = idio_mom.mean(axis=1)\nfactor = idio_mom.sub(cross_mean, axis=0).div(cross_std, axis=0).clip(-3, 3)",
        "direction": "long",
        "source": "stage1_exploration",
        "exploration_basis": "ML-inspired构造因子",
        "target_problem": "small_bull_regime_no_solution",
        "compatible_regime": "both",
        "expected_turnover": "medium",
        "related_blacklist": "无",
        "academic_reference": "Blitz, Huijman & Swinkels(2011, JPM) Residual Momentum"
    },
    {
        "name": "overnight_gap_durability",
        "label": "隔夜跳空持续性",
        "category": "return_decomposition",
        "hypothesis": "跳空幅度大但日内未被回补→机构信息定价有效→趋势可信。跳空大但日内完全回补→假突破(散户追高)。比overnight_intraday_ratio更精细",
        "logic": "持稳度=1-|日内回补/跳空幅度|。跳空+2%且收盘+1.8%(守住)→高分；跳空+2%但收盘-0.5%(几乎回补)→低分。5日平均",
        "formula": "import numpy as np\n\novernight_gap = open_p / close_p.shift(1) - 1\nintraday_ret = close_p / open_p - 1\ngap_mag = overnight_gap.abs()\ndurability = 1.0 - (intraday_ret.abs() / gap_mag.add(1e-4)).clip(0, 1)\nmeaningful = (gap_mag > 0.005).astype(float)\ndurability_weighted = durability.mul(meaningful, axis=0)\ndurability_5 = durability_weighted.rolling(5, min_periods=2).mean()\ngap_direction = overnight_gap.rolling(5, min_periods=2).mean()\nfactor = durability_5.clip(0, 1).mul(gap_direction.clip(-0.02, 0.02).add(0.1), axis=0)",
        "direction": "long",
        "source": "stage1_exploration",
        "exploration_basis": "ML-inspired构造因子",
        "target_problem": "phantom_overfitting",
        "compatible_regime": "both",
        "expected_turnover": "medium",
        "related_blacklist": "无",
        "academic_reference": "Lou, Polk & Skouras(2019, JFE) overnight-intraday tug of war"
    },
    {
        "name": "turnover_cycle_position",
        "label": "换手率周期位置",
        "category": "low_turnover_mid_freq",
        "hypothesis": "换手刚从高位回落到均值附近的股票(收缩阶段)比持续高换手股票有更低交易成本和更强信号稳定性。直接解决SUSPENDED基因高换手(190%/174%)成本杀",
        "logic": "20日换手代理Volume(归一化)Z-score相对60日均值偏离；Z-score从高位(-1→-0.3区间)回落=脱离亢奋→信号可信度回升+低换手",
        "formula": "import numpy as np\n\nvol_turnover = volume_p / volume_p.rolling(20).mean().replace(0, np.nan)\nvol_mean_60 = vol_turnover.rolling(60).mean()\nvol_std_60 = vol_turnover.rolling(60).std().replace(0, 1e-6)\nvol_z = (vol_turnover - vol_mean_60) / vol_std_60\nvol_z_max_20 = vol_z.rolling(20).max()\nvol_z_min_20 = vol_z.rolling(20).min()\nvol_z_range = (vol_z_max_20 - vol_z_min_20).replace(0, 1e-6)\ncycle_pos = (vol_z - vol_z_max_20) / vol_z_range\nsettling = ((cycle_pos > -0.35) & (cycle_pos < -0.05)).astype(float)\nsettling_score = settling.rolling(5, min_periods=2).mean()\nret_5 = close_p.pct_change(5).clip(-0.15, 0.15)\nmom_confirm = (ret_5 > 0).astype(float)\nfactor = settling_score.mul(mom_confirm, axis=0)",
        "direction": "long",
        "source": "stage1_exploration",
        "exploration_basis": "低换手中低频因子",
        "target_problem": "high_turnover_cost_kill",
        "compatible_regime": "both",
        "expected_turnover": "low",
        "related_blacklist": "无",
        "academic_reference": "Chordia, Subrahmanyam & Anshuman(2001) turnover; Lo & MacKinlay(1990)"
    },
    {
        "name": "max_drawdown_risk_premium",
        "label": "最大回撤风险溢价",
        "category": "failure_mode_inverse",
        "hypothesis": "过往20日最大回撤深度是风险定价信号(与vm_diff的波动率差分正交)。近期经历大回撤的股票=尾部风险暴露→应获补偿。用路径依赖回撤代替无条件波动率差分，降低regime脆弱",
        "logic": "与vm_diff同为风险维度但用回撤(投资者真实感知)代替统计波动率。回撤深度×恢复确认=风险溢价信号。去重：不同于volatility_of_volatility和regime_conditional_volatility_ratio",
        "formula": "import numpy as np\n\nlookback = 20\ncumret = (1 + close_p.pct_change().fillna(0)).rolling(lookback, min_periods=10).apply(lambda x: x.prod(), raw=True)\nrolling_max = cumret.rolling(lookback, min_periods=10).max()\ndrawdown = (cumret / rolling_max - 1).clip(upper=0)\nmax_dd = drawdown.rolling(lookback, min_periods=10).min()\nret_5 = close_p.pct_change(5).clip(-0.2, 0.2)\nrecovering = (ret_5 > -0.05).astype(float)\nmax_dd_premium = (-max_dd).clip(0, 0.3)\nfactor = max_dd_premium.mul(recovering, axis=0)",
        "direction": "long",
        "source": "stage1_exploration",
        "exploration_basis": "失败模式逆向因子",
        "target_problem": "vm_diff_toxic_combo_replacement",
        "compatible_regime": "both",
        "expected_turnover": "low",
        "related_blacklist": "vm_diff(风险维度替代)",
        "academic_reference": "Ang, Chen & Xing(2006, JF) downside risk"
    },
    {
        "name": "market_cap_volume_extreme_ratio",
        "label": "小盘极端量比(质押风险代理)",
        "category": "risk_event_driven",
        "hypothesis": "小市值股票出现极端成交量(相对自身均值>1.5x)伴随重大事件(质押爆仓/重组/ST)。异常暴露后过度反应→定价修正→risk premium。A股小盘特色因子",
        "logic": "A股大量小市值壳股(<50亿)的极端放量是专属风险事件。20日均量vs 60日均量极端度×市值惩罚(小市值=高风险溢价)。方向确认避免抄底飞刀",
        "formula": "import numpy as np\n\nvol_20 = volume_p.rolling(20).mean()\nvol_60 = volume_p.rolling(60).mean()\nvol_ratio = vol_20 / vol_60.replace(0, np.nan)\nvol_extreme = (vol_ratio - 1.5).clip(0, 3)\nmcap_proxy = close_p.rolling(20).mean().mul(volume_p.rolling(60).mean(), axis=0)\nmcap_rank = mcap_proxy.rank(pct=True, axis=1)\nsmall_cap_weight = 1.0 - mcap_rank\nrisk_premium = vol_extreme.rolling(5, min_periods=2).mean().mul(small_cap_weight, axis=0)\nret_10 = close_p.pct_change(10).clip(-0.25, 0.25)\ndirection = (ret_10 > -0.1).astype(float)\nfactor = risk_premium.clip(0, 2).mul(direction, axis=0)",
        "direction": "long",
        "source": "stage1_exploration",
        "exploration_basis": "A股特色风险因子",
        "target_problem": "small_bull_regime_no_solution",
        "compatible_regime": "small_bull",
        "expected_turnover": "medium",
        "related_blacklist": "无",
        "academic_reference": "null(A股微观结构特色, 无直接学术引用)"
    }
]

def _merge(data):
    existing_names = set(p['name'] for p in data['proposals'])
    for p in new_proposals:
        if p['name'] in existing_names:
            print(f'DUPLICATE: {p["name"]}')
        else:
            data['proposals'].append(p)
            print(f'ADDED: {p["name"]} → [{p["target_problem"]}] {p["expected_turnover"]} turnover | {p["compatible_regime"]}')
    return data

# 锁内闭环: 读最新 → 去重合并 → 原子写 (2026-08-21 防护)
data = load_and_modify(_merge)

print(f'\nTotal proposals: {len(data["proposals"])}')
print(f'JSON valid: True')

# Check all have required metadata
required_fields = ['target_problem', 'compatible_regime', 'expected_turnover', 'related_blacklist', 'academic_reference']
missing = 0
for p in data['proposals']:
    for f in required_fields:
        if f not in p:
            print(f'MISSING {f} in {p["name"]}')
            missing += 1

if missing == 0:
    print(f'All {len(data["proposals"])} proposals have complete metadata')
else:
    print(f'{missing} metadata fields missing')

# Bottleneck coverage summary
from collections import Counter
targets = Counter(p.get('target_problem', 'unknown') for p in data['proposals'])
print('\nBottleneck coverage:')
for t, c in targets.most_common():
    print(f'  {t}: {c} proposals')
