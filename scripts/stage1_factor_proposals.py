"""Stage 1 Factor Proposal Builder — dedup + validate + append"""
import json, os, ast

# === STEP 1: Collect all existing factor names ===
existing = set()

# 1a. tested_factors.json
tf = json.load(open('data/tested_factors.json'))
existing.update(tf['tested'].keys())

# 1b. stage1_factor_proposals.json
sf = 'data/stage1_factor_proposals.json'
if os.path.exists(sf):
    sp = json.load(open(sf))
    existing.update(p['name'] for p in sp.get('proposals', []))

# 1c. experiment_state.json gene_pool
es = json.load(open('research/factor_alchemy/experiment_state.json'))
for gene in es['gene_pool']:
    existing.update(gene['dna']['factors'].keys())

# 1d. factors/__init__.py (rough extraction)
with open('research/factor_alchemy/factors/__init__.py') as f:
    tree = ast.parse(f.read())
for node in ast.walk(tree):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        if '_' in node.value and len(node.value) > 5:
            existing.add(node.value)

print(f'Total existing names collected: {len(existing)}')

# === STEP 2: New factor proposals ===
new_proposals = [
    {
        'name': 'small_cap_momentum_quality',
        'label': '小盘动量质量共振',
        'category': 'regime_specific',
        'hypothesis': '小盘牛市regime下，小市值+短期动量+盈利质量三因子共振可产生独立于大盘质量策略的alpha',
        'logic': '对小市值池(市值<中位数)同时要求：1)近期正动量(20日收益>0)，2)收盘/开盘稳定性(QoQ增长代理)，3)量价共振(量增价涨)。取三者排名等权平均。',
        'formula': 'import numpy as np\nimport pandas as pd\n# 1. Size factor: small cap bias (negative market cap score)\nmarket_cap = close_p.rolling(20).mean() * volume_p.rolling(20).mean()\nsize_rank = market_cap.rank(axis=1, pct=True)\nsize_score = -size_rank\n# 2. Momentum: 20-day return\nmom_20 = close_p.pct_change(20)\nmom_rank = mom_20.rank(axis=1, pct=True)\n# 3. Quality proxy: close/open ratio stability\nco_ratio = close_p / open_p.replace(0, np.nan)\nco_stability = -co_ratio.rolling(20).std()\nquality_rank = co_stability.rank(axis=1, pct=True)\n# 4. Volume-price resonance\nvol_chg = volume_p.pct_change(5)\nret_5 = close_p.pct_change(5)\nresonance = vol_chg.rolling(20).corr(ret_5)\nresonance_rank = resonance.rank(axis=1, pct=True)\nfactor = pd.concat([size_score, mom_rank, quality_rank, resonance_rank], axis=1).mean(axis=1).unstack()',
        'direction': 'long',
        'source': 'stage1_exploration',
        'exploration_basis': 'small_bull突围因子',
        'target_problem': 'small_bull_regime_no_solution',
        'compatible_regime': 'small_bull',
        'expected_turnover': 'medium',
        'related_blacklist': '无',
        'academic_reference': 'Fama-French (1993) SMB; Carhart (1997) MOM; Asness et al. (2019) QMJ'
    },
    {
        'name': 'regime_adaptive_sigma_ratio',
        'label': 'Regime自适应波动率比率',
        'category': 'volatility_risk',
        'hypothesis': '用regime条件化的波动率比率替代vm_diff，避免组合层regime脆弱',
        'logic': 'Garman-Klass波动率/实现波动率比(捕捉盘中vs收盘波动异象)+上行/下行半方差比(捕捉非对称风险)。双比率互补，替代vm_diff的单一波动差异。',
        'formula': 'import numpy as np\n# Garman-Klass volatility estimator\nlog_hl = np.log(high_p.replace(0, np.nan) / low_p.replace(0, np.nan))\nlog_co = np.log(close_p.replace(0, np.nan) / open_p.replace(0, np.nan))\ngk_vol = np.sqrt(0.5 * log_hl**2 - (2*np.log(2)-1) * log_co**2)\ngk_vol_20 = gk_vol.rolling(20).mean()\n# Realized volatility\nret_daily = close_p.pct_change()\nreal_vol_20 = ret_daily.rolling(20).std()\n# GK/Real ratio\ngk_real_ratio = gk_vol_20 / real_vol_20.replace(0, np.nan)\n# Upside/Downside semi-variance ratio\nret_up = ret_daily.clip(lower=0)\nret_down = ret_daily.clip(upper=0)\nup_vol = ret_up.rolling(20).std()\ndown_vol = ret_down.rolling(20).std()\nud_ratio = up_vol / (down_vol.replace(0, np.nan) + 1e-6)\n# Composite: 50% GK/Real + 50% Up/Down\ngk_rank = gk_real_ratio.rank(axis=1, pct=True)\nud_rank = ud_ratio.rank(axis=1, pct=True)\nfactor = gk_rank.mul(0.5, axis=0).add(ud_rank.mul(0.5, axis=0))',
        'direction': 'long',
        'source': 'stage1_exploration',
        'exploration_basis': '失败模式逆向因子·vm_diff替代',
        'target_problem': 'vm_diff_toxic_combo',
        'compatible_regime': 'both',
        'expected_turnover': 'low',
        'related_blacklist': 'vm_diff (替代因子)',
        'academic_reference': 'Garman-Klass (1980); Sortino & Price (1994)'
    },
    {
        'name': 'biweekly_price_momentum_smooth',
        'label': '双周平滑价格动量',
        'category': 'momentum_reversal',
        'hypothesis': 'EMA平滑斜率天然低换手，中低频空间有独立alpha（未被浅层高频挖掘覆盖）',
        'logic': 'EMA(12)斜率保留趋势方向但过滤日间噪声，双周调仓视角。差异化于短期反转和高频动量（拥挤）。',
        'formula': 'import numpy as np\n# EMA-12 of close price\nema_vals = close_p.ewm(span=12, adjust=False).mean()\n# EMA slope: (EMA_t - EMA_t-10) / |EMA_t-10|\nema_slope = (ema_vals - ema_vals.shift(10)) / (ema_vals.shift(10).replace(0, np.nan).abs() + 1e-6)\n# Smooth the slope with 5-day MA\nsmooth_slope = ema_slope.rolling(5).mean()\nfactor = smooth_slope.rank(axis=1, pct=True)',
        'direction': 'long',
        'source': 'stage1_exploration',
        'exploration_basis': '低换手中低频因子',
        'target_problem': 'high_turnover_suspended_genes',
        'compatible_regime': 'both',
        'expected_turnover': 'low',
        'related_blacklist': '无',
        'academic_reference': 'Moskowitz & Grinblatt (1999); Jegadeesh & Titman (2001)'
    },
    {
        'name': 'earnings_forecast_consensus_gap',
        'label': '盈利预测共识分歧',
        'category': 'fundamental_analyst',
        'hypothesis': '分析师盈利预测分歧度高=信息不对称=不确定性溢价(正alpha)，结合近期价格方向确认',
        'logic': '用OHLCV代理：日内波动率/VWAP偏离度作为信息分歧代理(分歧度高的股票被低估)。分歧度高+近期价格上涨=买入信号。',
        'formula': 'import numpy as np\n# Information disagreement proxy\nvwap_proxy = (high_p + low_p + close_p) / 3.0\nvwap_dev = (close_p - vwap_proxy).abs() / (vwap_proxy.replace(0, np.nan) + 1e-6)\n# Daily range as dispersion\ndaily_range_pct = (high_p - low_p) / (close_p.shift(1).replace(0, np.nan) + 1e-6)\n# 20-day disagreement\ndisagreement = (daily_range_pct * vwap_dev).rolling(20).mean()\ndis_z = (disagreement - disagreement.mean()) / (disagreement.std() + 1e-6)\n# Direction: recent 5-day return\nret_5d = close_p.pct_change(5)\ndirection = np.sign(ret_5d)\nfactor_raw = dis_z * direction\nfactor = factor_raw.rank(axis=1, pct=True)',
        'direction': 'long',
        'source': 'stage1_exploration',
        'exploration_basis': '学术异象库(分析师情绪)',
        'target_problem': 'phantom_4.6x',
        'compatible_regime': 'large_dominant',
        'expected_turnover': 'low',
        'related_blacklist': '无',
        'academic_reference': 'Diether, Malloy & Scherbina (2002); Stickel (1991)'
    },
    {
        'name': 'gap_momentum_regime_blend',
        'label': 'Gap动量Regime融合',
        'category': 'regime_adaptive',
        'hypothesis': 'Gap动量+反转过滤自适应组合：大盘追trend、小盘接反转，避免small_bull假突破',
        'logic': '向上跳空动量(gap magnitude*hit rate)为主信号(60%权重)，短期反转强度为辅助过滤(40%权重)。大盘时gap动量主导，小盘时反转过滤防止追高。',
        'formula': 'import numpy as np\n# Positive gap momentum\novernight_ret = close_p / open_p.replace(0, np.nan) - 1.0\npos_gap = overnight_ret.clip(lower=0)\ngap_hit_rate = (overnight_ret > 0).rolling(20).mean()\ngap_mag = pos_gap.rolling(20).mean()\ngap_mom = gap_mag * gap_hit_rate\n# Reversal filter: 5-day reversal\nret_5d = close_p.pct_change(5)\nrev_strength = -ret_5d\n# Blend\ngap_rank = gap_mom.rank(axis=1, pct=True)\nrev_rank = rev_strength.rank(axis=1, pct=True)\nfactor = gap_rank.mul(0.6, axis=0).add(rev_rank.mul(0.4, axis=0))',
        'direction': 'long',
        'source': 'stage1_exploration',
        'exploration_basis': 'small_bull突围因子',
        'target_problem': 'small_bull_regime_no_solution',
        'compatible_regime': 'both',
        'expected_turnover': 'medium',
        'related_blacklist': '无',
        'academic_reference': 'Hamilton (1989) Regime Switching; Kelly & Pruitt (2013)'
    },
    {
        'name': 'volume_distribution_entropy',
        'label': '成交量分布熵',
        'category': 'microstructure_ml',
        'hypothesis': '成交量日内分布熵反映信息质量(高熵=机构参与/知情交易)，可用OHLCV日频代理获取差异化alpha',
        'logic': '用High-Open和Close-Low的相对范围构建日内成交分布代理。二进制熵值归一化，高熵=均匀分布(机构化)。熵×收益方向=聪明钱信号。',
        'formula': 'import numpy as np\n# Intraday distribution proxy\nupper_ref = np.maximum(open_p, close_p)\nlower_ref = np.minimum(open_p, close_p)\nup_range = (high_p - upper_ref) / (high_p - low_p).replace(0, np.nan)\ndown_range = (lower_ref - low_p) / (high_p - low_p).replace(0, np.nan)\n# Binary entropy\np_up = up_range / (up_range + down_range + 1e-6)\np_up_clip = p_up.clip(1e-6, 1-1e-6)\nentropy = -(p_up_clip * np.log(p_up_clip) + (1-p_up_clip) * np.log(1-p_up_clip))\nentropy_norm = entropy / np.log(2)\n# 20-day rolling\nentropy_20 = entropy_norm.rolling(20).mean()\n# Direction\nret_5d = close_p.pct_change(5)\ndirection = np.where(ret_5d > 0, 1.0, -1.0)\nfactor_raw = entropy_20.rank(axis=1, pct=True) * direction\nfactor = factor_raw.rank(axis=1, pct=True)',
        'direction': 'long',
        'source': 'stage1_exploration',
        'exploration_basis': 'ML-inspired构造因子',
        'target_problem': 'regime_concentrated',
        'compatible_regime': 'both',
        'expected_turnover': 'medium',
        'related_blacklist': '无',
        'academic_reference': 'Easley, Kiefer & OHara (1997) PIN; Shannon (1948)'
    }
]

# === STEP 3: Validate uniqueness ===
new_names = [p['name'] for p in new_proposals]
conflicts = [n for n in new_names if n in existing]
if conflicts:
    print(f'CONFLICTS FOUND: {conflicts}')
    raise SystemExit(1)
else:
    print('All 6 new factor names are unique!')

# === STEP 4: Load existing, append, save ===
existing_proposals = []
if os.path.exists(sf):
    sp = json.load(open(sf))
    existing_proposals = sp.get('proposals', [])

all_proposals = existing_proposals + new_proposals

output = {'proposals': all_proposals}
with open(sf, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

# Validate
json.load(open(sf))
print(f'Saved {len(all_proposals)} proposals to {sf}')
print(f'Newly added ({len(new_proposals)}):')
for p in new_proposals:
    print(f'  + {p["name"]} [{p["target_problem"]}] [{p["compatible_regime"]}]')

# === STEP 5: Print bottleneck-problem mapping ===
print('\n--- Bottleneck-Problem Mapping ---')
problems = set(p['target_problem'] for p in new_proposals)
for prob in problems:
    factors = [p['name'] for p in new_proposals if p['target_problem'] == prob]
    print(f'{prob}: {factors}')
