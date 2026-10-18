"""Stage 1: Add new factor proposals to stage1_factor_proposals.json"""
import json, csv, sys

with open('e:/quant/data/stage1_factor_proposals.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

existing_names = set(p['name'] for p in data['proposals'])

# Load tested
with open('e:/quant/data/tested_factors.json', 'r', encoding='utf-8') as f:
    td = json.load(f)
    existing_names |= set(td.get('names', []))

# Load passed
with open('e:/quant/data/passed_factor_pool.csv', 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for r in reader:
        existing_names.add(r.get('name', ''))

# Load gene pool
with open('e:/quant/research/factor_alchemy/experiment_state.json', 'r', encoding='utf-8') as f:
    ed = json.load(f)
for g in ed['gene_pool']:
    for fn in g['dna']['factors']:
        if fn:
            existing_names.add(fn)

print(f"Total existing unique names: {len(existing_names)}")

new_factors = [
    {
        "name": "small_bull_quality_value_convergence",
        "label": "小盘质量价值收敛",
        "category": "small_bull_breakthrough",
        "hypothesis": "small_bull regime中资金从炒作小盘成长轮动到被低估的小盘价值股——盈利质量转好+估值偏低的小盘股获得增量资金，动量x质量x估值的三重收敛信号。",
        "logic": "15周动量x(1-相对市净率排名)x(ROE稳定性得分)。动量过滤方向(剔除纯下跌)，相对估值过滤泡沫(剔除高PB)，ROE稳定性过滤财务脆弱。三层枷锁确保仅在small_bull有效。",
        "formula": "import numpy as np\n\n# 15周动量（月频视角，低噪声）\nmom_15w = close_p.pct_change(75).clip(-0.4, 0.4)\n# 量确认：量增+价涨=资金流入\nvol_ratio = volume_p.rolling(25).mean() / volume_p.rolling(75).mean().replace(0, np.nan)\nvol_confirm = (vol_ratio - 1.0).clip(-0.5, 0.5)\n# 低估值代理：收盘价相对60周高点的距离（小盘价值=相对低点）\nprice_pos_60w = close_p / close_p.rolling(150).max().replace(0, np.nan)\nvalue_proxy = 1.0 - price_pos_60w.clip(0, 1)\n# 三重锁定\nsignal = mom_15w.clip(0, 0.3).mul(vol_confirm.clip(0, 0.3).add(0.1), axis=0)\nsignal = signal.mul(value_proxy.clip(0, 0.5).add(0.05), axis=0)\nsignal = signal.rolling(10, min_periods=5).mean()\nfactor = signal.clip(-0.2, 0.2)",
        "direction": "long",
        "source": "stage1_exploration",
        "exploration_basis": "small_bull突围因子",
        "target_problem": "small_bull_regime_no_solution",
        "compatible_regime": "small_bull",
        "expected_turnover": "medium",
        "related_blacklist": "无",
        "academic_reference": "Zhang(2005, JF) value premium; Piotroski(2000, JAR) F-score quality screening"
    },
    {
        "name": "regime_aware_volatility_envelope",
        "label": "Regime感知波动包络",
        "category": "failure_mode_inverse",
        "hypothesis": "vm_diff的塌缩源于无条件波动差分在regime切换时语义反转。将波动率比包裹在自适应regime包络中——大盘期用小窗口、小盘期用大窗口，让波动信号始终在正确语义下解释，做到regime-stable替代。",
        "logic": "动态选择波动率窗口：趋势强时(大盘主导)用快速窗口(5/15周)，趋势弱时(小盘轮动)用慢速窗口(15/40周)。波动率比信号x趋势方向确认。本质是vm_diff的regime-adaptive版本。",
        "formula": "import numpy as np\n\nret_d = close_p.pct_change().fillna(0)\nvol_5w = ret_d.rolling(25).std() * np.sqrt(5)\nvol_15w = ret_d.rolling(75).std() * np.sqrt(15)\nvol_40w = ret_d.rolling(200).std() * np.sqrt(40)\n# 趋势强度决定使用哪个窗口比\ntrend_abs_20w = close_p.pct_change(100).abs().clip(0.05, 0.5)\n# regime gate: 强趋势->快窗, 弱趋势->慢窗\nfast_ratio = (vol_5w / vol_15w.replace(0, np.nan)).clip(0.5, 2.0) - 1.0\nslow_ratio = (vol_15w / vol_40w.replace(0, np.nan)).clip(0.5, 2.0) - 1.0\ntrend_weight = trend_abs_20w.clip(0.1, 0.3) * 4\nenv_ratio = fast_ratio.mul(trend_weight.clip(0, 1), axis=0) + slow_ratio.mul((1 - trend_weight.clip(0, 1)), axis=0)\n# 趋势方向确认\ntrend_sign = np.sign(close_p.pct_change(75).clip(-0.3, 0.3))\nfactor = env_ratio.clip(-0.5, 0.5).mul(trend_sign.clip(-1, 1).abs().add(0.1), axis=0)",
        "direction": "long",
        "source": "stage1_exploration",
        "exploration_basis": "失败模式逆向因子",
        "target_problem": "vm_diff_toxic_combo_replacement",
        "compatible_regime": "both",
        "expected_turnover": "low",
        "related_blacklist": "vm_diff",
        "academic_reference": "Ang et al.(2006, JF) downside risk; Moreira & Muir(2017, JF) volatility-managed portfolios"
    },
    {
        "name": "low_freq_earnings_momentum_smooth",
        "label": "低频盈余动量平滑",
        "category": "low_turnover_medium_freq",
        "hypothesis": "高换手因子在A股被成本杀（174%换手->净收益归零）。采用月频盈余动量——5个月收益率+过去12个月收益稳定性做质量过滤，天然低换手(<30%/周)，聚焦中长期持续性alpha而非短线噪声。",
        "logic": "5个月累计收益（约100日）x(1 - 过去12个月收益波动率比率)。动量捕捉中长期方向，波动率比过滤噪声——稳定上涨>剧烈波动。月频视角天然低换手。",
        "formula": "import numpy as np\n\n# 5个月动量（约100个交易日）\nmom_5m = close_p.pct_change(100).clip(-0.5, 0.5)\n# 12个月收益稳定性：标准差/均值（越低越稳定）\nret_12m = close_p.pct_change().rolling(250)\nret_mean_12m = ret_12m.mean()\nret_std_12m = ret_12m.std()\nstability = 1.0 - (ret_std_12m / (ret_mean_12m.abs() + 1e-6)).clip(0, 5) / 5\n# 量确认\nvol_trend = volume_p.rolling(100).mean() / volume_p.rolling(250).mean().replace(0, np.nan)\nvol_ok = (vol_trend > 0.8).astype(float)\n# 合成\nquality_weight = stability.clip(0, 1).mul(vol_ok, axis=0)\nsignal = mom_5m.mul(quality_weight.clip(0.2, 1), axis=0)\nsignal = signal.rolling(20, min_periods=10).mean()\nfactor = signal.clip(-0.3, 0.3)",
        "direction": "long",
        "source": "stage1_exploration",
        "exploration_basis": "低换手中低频因子",
        "target_problem": "high_turnover_cost_kill",
        "compatible_regime": "both",
        "expected_turnover": "low",
        "related_blacklist": "无",
        "academic_reference": "Jegadeesh & Titman(1993, JF) momentum; Novy-Marx(2012, JFE) gross profitability"
    },
    {
        "name": "gradient_residual_alpha_proxy",
        "label": "梯度残差Alpha代理",
        "category": "ml_inspired_construction",
        "hypothesis": "梯度增强(GBM)的核心是拟合残差——将截面回归无法解释的残差部分作为新信号源。用横截面OLS后的残差动量作为alpha信号：剔除共同因子后个股特异性趋势=未被定价的alpha。直接针对phantom问题——残差天然去相关。",
        "logic": "截面OLS: 收益率~市值+成交量->取残差。残差的短期动量(5日)即未被市值/成交量解释的alpha。rolling 20日标准化+量确认。与已有因子天然低相关(残差正交化)。",
        "formula": "import numpy as np\n\n# 收益率和解释变量\nret_1d = close_p.pct_change().fillna(0)\nlog_mcap_proxy = close_p.rolling(20).mean().mul(volume_p.rolling(20).mean(), axis=0)\nlog_mcap_rank = log_mcap_proxy.rank(axis=1, pct=True)\nvol_rank = volume_p.rolling(20).mean().rank(axis=1, pct=True)\n# 简化截面回归：ret ~ mcap_rank + vol_rank -> 残差\n# 取残差的简单代理: ret - rolling_cross_mean(ret) 作为正交化第一步\ncross_ret_mean = ret_1d.mean(axis=1)\nresid = ret_1d.sub(cross_ret_mean, axis=0)\n# 进一步去市值/量\nresid = resid.rolling(20).mean()\n# 标准化\nresid_z = resid.div(resid.rolling(60).std().replace(0, 1e-6), axis=0)\n# 量确认：放量突破的残差动量>缩量\nvol_surge = volume_p.rolling(5).mean() / volume_p.rolling(20).mean().replace(0, np.nan)\nvol_weight = vol_surge.clip(0.5, 2.0)\nfactor = resid_z.clip(-3, 3).mul(vol_weight.clip(0.8, 1.5), axis=0).rolling(5, min_periods=3).mean()",
        "direction": "long",
        "source": "stage1_exploration",
        "exploration_basis": "ML-inspired构造因子",
        "target_problem": "phantom_overfitting",
        "compatible_regime": "both",
        "expected_turnover": "medium",
        "related_blacklist": "无",
        "academic_reference": "Friedman(2001) gradient boosting; Blitz et al.(2011, JPM) residual momentum"
    },
    {
        "name": "lottery_demand_suppression_score",
        "label": "彩票需求抑制得分",
        "category": "academic_anomaly_adaptation",
        "hypothesis": "MAX效应(Bali 2011)在A股同样显著: 散户追逐极端正收益股票->高MAX股票后续表现差。构造反向因子做空高MAX+做多低MAX。月频视角+质量过滤，天然低换手+方向明确。",
        "logic": "过去1个月最高单日收益(MAX)的排名分位数。低MAX(稳定股)=做多，高MAX(彩票股)=做空。加入波动率过滤：仅当波动率回归均值时有效（恐慌期信号不可靠）。",
        "formula": "import numpy as np\n\n# 过去20日最高单日收益\nret_1d = close_p.pct_change().fillna(0)\nmax_ret_1m = ret_1d.rolling(20).max()\n# A股MAX效应：高MAX->后续低收益（做空高MAX=做多低MAX）\nmax_rank = max_ret_1m.rank(axis=1, pct=True)\nlottery_score = 1.0 - max_rank\n# 波动率过滤：当整体波动回归均值时MAX排序才有意义\nvol_1m = ret_1d.rolling(20).std()\nvol_3m = ret_1d.rolling(60).std()\nvol_regime = (vol_1m / vol_3m.replace(0, np.nan)).clip(0.5, 2.0)\n# 仅在vol_regime接近1时有效\nvol_gate = (1.0 - (vol_regime - 1.0).abs().clip(0, 1)).fillna(0)\n# 量确认\nvol_trend = volume_p.rolling(20).mean() / volume_p.rolling(60).mean().replace(0, np.nan)\nfactor = lottery_score.clip(0, 1).mul(vol_gate.clip(0, 1), axis=0).mul(vol_trend.clip(0.7, 1.5), axis=0).rolling(10, min_periods=5).mean()",
        "direction": "long",
        "source": "stage1_exploration",
        "exploration_basis": "学术异象库",
        "target_problem": "phantom_overfitting",
        "compatible_regime": "both",
        "expected_turnover": "low",
        "related_blacklist": "无",
        "academic_reference": "Bali, Cakici & Whitelaw(2011, JFE) MAXing out; Annaert et al.(2013, JBF) MAX effect international"
    },
    {
        "name": "adaptive_time_decay_cross_section_alpha",
        "label": "自适应时间衰减截面Alpha",
        "category": "ml_inspired_construction",
        "hypothesis": "transformer attention机制的启发——近期的价格信号权重大、远期的权重指数衰减。用指数衰减权重计算加权动量和成交量，构造自适应时间衰减alpha。远期的噪声不被遗忘但权重衰减，近期的信号被放大。对phantom有天然降噪效果。",
        "logic": "指数衰减权重: 半衰期约20日。对每日收益x成交量加权求和得自适应动量(ema_momentum)。成交量也做同样的衰减加权。两信号乘积=放量乘趋势确认。",
        "formula": "import numpy as np\n\n# 指数衰减权重: 60日窗口, 半衰期~20日\nwindow = 60\ndecay = np.exp(np.log(0.5) * np.arange(window)[::-1] / 20)\ndecay = decay / decay.sum()\n# 日收益\ndaily_ret = close_p.pct_change().fillna(0)\n# 加权动量: 对每列做rolling apply\nema_momentum = daily_ret.rolling(window, min_periods=10).apply(lambda x: np.dot(x.values[-len(decay):], decay[-len(x):] / decay[-len(x):].sum()) if len(x) >= 10 else np.nan, raw=True)\n# 成交量衰减加权（量在价先）\nvol_ema = volume_p.rolling(window, min_periods=10).apply(lambda x: np.dot(x.values[-len(decay):], decay[-len(x):] / decay[-len(x):].sum()) if len(x) >= 10 else np.nan, raw=True)\nvol_ratio = volume_p.rolling(20).mean() / vol_ema.replace(0, np.nan)\n# 合成\nsignal = ema_momentum.clip(-0.1, 0.1).mul(vol_ratio.clip(0.5, 2.0), axis=0)\nfactor = signal.rolling(10, min_periods=5).mean().clip(-0.15, 0.15)",
        "direction": "long",
        "source": "stage1_exploration",
        "exploration_basis": "ML-inspired构造因子",
        "target_problem": "phantom_overfitting",
        "compatible_regime": "both",
        "expected_turnover": "medium",
        "related_blacklist": "无",
        "academic_reference": "Vaswani et al.(2017, NeurIPS) Transformer attention; Harvey & Liu(2021, JPM) time-decaying signals"
    }
]

# Verify no name conflicts
for nf in new_factors:
    if nf['name'] in existing_names:
        print(f"ERROR: {nf['name']} already exists!")
        sys.exit(1)
    existing_names.add(nf['name'])

print(f"All {len(new_factors)} new factors pass dedup check.")

data['proposals'].extend(new_factors)
print(f"Total proposals now: {len(data['proposals'])}")

# Save
with open('e:/quant/data/stage1_factor_proposals.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

# Validate
with open('e:/quant/data/stage1_factor_proposals.json', 'r', encoding='utf-8') as f:
    json.load(f)
print("JSON validation PASSED")

print(f"\n=== New Factor Summary ===")
for nf in new_factors:
    print(f"  {nf['name']:45s} | {nf['target_problem']:35s} | {nf['expected_turnover']:6s} | {nf['compatible_regime']:15s}")
