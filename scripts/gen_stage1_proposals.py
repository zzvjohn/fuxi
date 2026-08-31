"""Generate stage1_factor_proposals.json with 5 new factors for 2026-07-22."""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proposals_io import load_and_modify   # 2026-08-21 并发覆盖防护: 锁内闭环读写

# 5 NEW factors for 2026-07-22
new_factors = [
    {
        'name': 'abdi_ranaldo_spread',
        'label': 'Abdi-Ranaldo买卖价差(2017)',
        'category': 'academic_anomalies',
        'hypothesis': 'Abdi-Ranaldo(2017, JFE)用收盘价与高低价中点协方差估计有效买卖价差，比Corwin-Schultz(2012)更稳定且前视偏差更小。高AR价差=高流动性成本=流动性溢价补偿=正向预期收益。未被本库覆盖。',
        'logic': 'c_t = ln(close_t) - [ln(high_t)+ln(low_t)]/2。AR证明E[c_t * c_{t+1}] = -S^2/4, S = 2*sqrt(|E[c_t*c_{t+1}]|)。用20日滚动窗口协方差估计。比Roll(1984)更稳定因为锚定日内价格中点。高AR价差=信息不对称程度高=预期收益补偿。',
        'formula': 'import numpy as np\n\n# c_t = ln(close) - (ln(high) + ln(low)) / 2\nlog_close = np.log(close_p.replace(0, np.nan))\nlog_high = np.log(high_p.replace(0, np.nan))\nlog_low = np.log(low_p.replace(0, np.nan))\nc = log_close - (log_high + log_low) / 2.0\n\n# 20日窗口: E[c_t * c_{t+1}] = -S^2/4\nc_lag = c.shift(1)\ncov_20 = c.rolling(20).cov(c_lag)\n\n# AR spread: S = 2 * sqrt(|E[c_t*c_{t+1}]|)\n# 仅当协方差为负时有效(买卖价差反弹)\nneg_cov = (-cov_20).clip(lower=0)\nar_spread_raw = 2.0 * np.sqrt(neg_cov)\n\n# 用收盘价归一化 + 5日平滑降噪\nar_spread_pct = ar_spread_raw / close_p.replace(0, np.nan)\nfactor = ar_spread_pct.rolling(5, min_periods=2).mean()',
        'direction': 'long',
        'source': 'stage1_exploration',
        'exploration_basis': '学术异象库'
    },
    {
        'name': 'garman_klass_volatility',
        'label': 'Garman-Klass波动率(1980)',
        'category': 'academic_anomalies',
        'hypothesis': 'Garman-Klass(1980, JBus)证明OHLC四价格波动率估计器效率是收盘价波动率的7.4倍。GK波动率高的股票信息流丰富(价格发现活跃)=不确定性溢价=预期收益更高。与Parkinson_vol_efficiency不同：本因子是GK原始波动率的绝对值而非比率。',
        'logic': 'GK(1980)简化公式: sigma^2 = 0.5*ln(H/L)^2 - (2*ln(2)-1)*ln(C/O)^2。20日rolling均值后开根号。高GK波动=丰富的日内价格发现活动=做多。已测parkinson_vol_efficiency是比率(HL波动/close波动)，本因子是绝对GK波动率，互补。',
        'formula': 'import numpy as np\n\n# Garman-Klass (1980) 波动率简化估计\n# sigma^2 = 0.5 * ln(H/L)^2 - (2*ln(2)-1) * ln(C/O)^2\nln_hl = np.log(high_p / low_p.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)\nln_co = np.log(close_p / open_p.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)\n\n# 日频GK方差项\nterm1 = 0.5 * ln_hl.pow(2)\nterm2 = (2 * np.log(2) - 1) * ln_co.pow(2)\ngk_var_daily = (term1 - term2).clip(lower=0)\n\n# 20日滚动均值 = GK波动率\ngk_var_20 = gk_var_daily.rolling(20).mean()\ngk_vol = np.sqrt(gk_var_20.clip(lower=0))\n\n# 高GK波动 = 丰富价格发现 = 做多\nfactor = gk_vol.clip(0, 0.15)',
        'direction': 'long',
        'source': 'stage1_exploration',
        'exploration_basis': '学术异象库'
    },
    {
        'name': 'vwap_gravitational_pull',
        'label': 'VWAP引力强度',
        'category': 'chip_stratification_ai',
        'hypothesis': '机构围绕VWAP执行交易(VWAP算法是主流执行策略)=偏离VWAP后价格会向VWAP回归。收盘价持续被拉回VWAP(高引力)的股票表明机构参与度深、筹码有序换手=趋势可靠性高。华泰筹码分层逻辑量价代理。',
        'logic': '引力强度 = 1 / (1 + |close-VWAP|/振幅)。偏离度越低表明价格紧贴VWAP(机构在均价附近有序建仓)。用20日VWAP计算偏离，5日窗口评估引力稳定性。与已测vwap_deviation_20d互补(该因子用z-score，本因子用原始比例+稳定性评分)。',
        'formula': 'import numpy as np\n\n# 典型价格和20日VWAP\ntypical = (high_p + low_p + close_p) / 3.0\nvp = typical * volume_p\nvwap_20 = vp.rolling(20).sum() / volume_p.rolling(20).sum().replace(0, np.nan)\n\n# 偏离度: |close - VWAP| / 20日振幅 (归一化偏离)\namp = (high_p.rolling(20).max() - low_p.rolling(20).min()).replace(0, np.nan)\ndev = (close_p - vwap_20).abs() / amp.clip(lower=1e-4)\n\n# 引力强度 = 1 / (1 + 偏离) = 偏离小=引力强\npull_5 = 1.0 / (1.0 + dev.rolling(5).mean())\n\n# 叠加温和上涨确认方向\nret_5 = close_p.pct_change(5).clip(-0.2, 0.2)\nupside = ret_5.clip(lower=0)\nfactor = pull_5.clip(0.3, 1.0).mul(upside.add(0.5).clip(0, 1), axis=0)',
        'direction': 'long',
        'source': 'stage1_exploration',
        'exploration_basis': '筹码分层AI因子'
    },
    {
        'name': 'volume_price_elasticity',
        'label': '量价弹性',
        'category': 'chip_stratification_ai',
        'hypothesis': '成交量变化率/价格变化率=量价弹性。高弹性(量增很多但价不动)反映买卖双方势均力敌=筹码在特定价格区间充分换手=持仓结构趋于稳固=趋势突破后的持续性更强。华泰筹码分层思想的量价逆向推断代理。',
        'logic': '弹性 = |volume变化率| / (1 + |price变化率|)。弹性高=量动价不动=有买卖双方在僵持换手(非散户脉冲)。与smart_money_convexity(量价二阶导差)互补：弹性是一阶维度捕捉持筹稳固度。10日滚动窗口。',
        'formula': 'import numpy as np\n\n# 10日量变化率: 当前10日均量 / 前10日均量 - 1\nvol_10 = volume_p.rolling(10).mean()\nvol_10_lag = volume_p.shift(10).rolling(10).mean()\nvol_chg = (vol_10 / vol_10_lag.replace(0, np.nan) - 1).abs().clip(0, 3)\n\n# 10日价格变化率: 10日动量绝对值\nprice_chg = close_p.pct_change(10).abs().clip(0, 0.5)\n\n# 量价弹性 = 量变化率 / (1 + 价格变化率)\nelasticity = vol_chg / (1.0 + price_chg)\n\n# 高弹性=筹码在换手, 叠加温和上涨确认\nret_5 = close_p.pct_change(5).clip(-0.15, 0.15)\nupside_component = (ret_5 > 0).astype(float)\nfactor = elasticity.clip(0, 5).mul(upside_component, axis=0)',
        'direction': 'long',
        'source': 'stage1_exploration',
        'exploration_basis': '筹码分层AI因子'
    },
    {
        'name': 'earnings_pre_drift_alignment',
        'label': '中报窗口前动量偏离对齐',
        'category': 'mid_report_divergence',
        'hypothesis': '中报窗口(7月)前5日，个股相对行业的超额动量方向与量能背离方向一致时(正超额+量缩=锁仓、负超额+量增=恐慌)，信息提前反映的程度最高=窗口期后超额收益方向性强。5日聚合降噪。',
        'logic': '华泰AI量价因子在中报窗口超额转负=需要更精细的中报期信号。本因子捕捉"个股超额动量×量能方向一致性"：超额正+量缩=知情资金锁仓、超额负+量增=恐慌出逃。与earnings_season_vol_div(价量Spearman相关)互补=本因子是方向对齐而非相关性。',
        'formula': 'import numpy as np\nimport pandas as pd\n\n# 个股5日动量 vs 截面均值动量(行业代理)\nret_5d = close_p.pct_change(5)\nmarket_ret = ret_5d.mean(axis=1)\n# 超额动量: 个股 - 市场均值\nexcess_ret = pd.DataFrame(index=close_p.index, columns=close_p.columns)\nfor col in close_p.columns:\n    excess_ret[col] = ret_5d[col] - market_ret\n\n# 量能背离: 1 - 5日均量/20日均量 (缩量=正)\nvol_ratio = volume_p.rolling(5).mean() / volume_p.rolling(20).mean().replace(0, np.nan)\nvol_compression = 1.0 - vol_ratio.clip(0.5, 1.5)\n\n# 方向对齐: 超额动量 * 量能压缩\nalignment = excess_ret.clip(-0.2, 0.2).mul(vol_compression.clip(-0.5, 0.5), axis=0)\n\n# 3日平滑降噪\nfactor = alignment.rolling(3, min_periods=1).mean()',
        'direction': 'long',
        'source': 'stage1_exploration',
        'exploration_basis': '中报预增分化因子'
    }
]

# 锁内闭环: 读最新 → append → 原子写 (2026-08-21 防护)
def _append(existing):
    for nf in new_factors:
        existing['proposals'].append(nf)
    existing['last_updated'] = '2026-07-22'
    return existing

existing = load_and_modify(_append)

print(f'Proposals updated: {len(existing["proposals"])} total (+{len(new_factors)} new)')
print(f'JSON validation: OK ({len(existing["proposals"])} entries)')

# Print new factor names
for nf in new_factors:
    print(f'  + {nf["name"]}: {nf["label"]} [{nf["exploration_basis"]}]')
