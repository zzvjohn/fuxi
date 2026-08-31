"""
因子炼金术 (Factor Alchemy) — 全局配置
========================================
独立于 v5b 的全新因子研究体系。
遗传算法驱动因子发现 + 多维度验证。
"""
from pathlib import Path

# ==================== 路径 ====================
ROOT = Path(__file__).parent
DATA_DIR = ROOT / 'data'
FACTOR_DIR = ROOT / 'factors'
EVAL_DIR = ROOT / 'evaluation'
GA_DIR = ROOT / 'ga'
PORTFOLIO_DIR = ROOT / 'portfolio'
UTILS_DIR = ROOT / 'utils'
OUTPUT_DIR = ROOT / 'output'

# 确保输出目录存在
for d in [OUTPUT_DIR]:
    d.mkdir(exist_ok=True)

# ==================== 数据 ====================
START_DATE = '2021-01-01'
END_DATE   = '2026-06-25'

# Tushare 凭据 (从工作区 credentials 导入)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from credentials import get_tushare_api

# ==================== 因子计算参数 ====================
FREQUENCY = 'W'               # 周频
REBALANCE_WEEKDAY = 1         # 周一调仓

# 后复权价格
PRICE_FQ = 'hfq'

# 停牌处理
SUSPEND_MOMENTUM_FFILL = True   # 动量类: ffill
SUSPEND_VOLATILITY_NAN  = True  # 波动类: NaN
SUSPEND_MIN_SAMPLE_RATIO = 2/3  # 有效样本门槛

# 异常值缩尾
WINSORIZE_PCT = (0.01, 0.99)

# 中性化
NEUTRALIZE_INDUSTRY = True
NEUTRALIZE_MCAP     = True

# 净资产过滤
FILTER_NEGATIVE_EQUITY = True

# ==================== 因子池定义 ====================
FACTOR_DEFS = {
    # ── 规模因子 ──
    'ln_mcap':               {'category': 'size',          'label': '对数市值'},
    'ln_circulating_mcap':   {'category': 'size',          'label': '对数流通市值'},

    # ── 价值因子 ──
    'ep':                    {'category': 'value',         'label': 'E/P'},
    'bp':                    {'category': 'value',         'label': 'B/P'},
    'sp':                    {'category': 'value',         'label': 'S/P'},
    'cfp':                   {'category': 'value',         'label': 'CF/P'},
    'dp':                    {'category': 'value',         'label': '股息率'},

    # ── 盈利因子 (Piotroski F-score 为核心) ──
    'f_score':               {'category': 'profitability', 'label': 'Piotroski F-score'},
    'roe':                   {'category': 'profitability', 'label': 'ROE'},
    'roa':                   {'category': 'profitability', 'label': 'ROA'},
    'roic':                  {'category': 'profitability', 'label': 'ROIC'},
    'gross_margin':          {'category': 'profitability', 'label': '毛利率'},
    'net_margin':            {'category': 'profitability', 'label': '净利率'},
    'accruals':              {'category': 'profitability', 'label': '应计利润'},

    # ── 换手率因子 ──
    'avg_turnover_1m':       {'category': 'turnover',      'label': '月均换手'},
    'avg_turnover_3m':       {'category': 'turnover',      'label': '季均换手'},
    'turnover_std':          {'category': 'turnover',      'label': '换手波动'},
    'turnover_change':       {'category': 'turnover',      'label': '换手变化'},
    'abnormal_turnover':     {'category': 'turnover',      'label': '异常换手'},

    # ── 动量因子 ──
    'ret_1m':                {'category': 'momentum',      'label': '1月收益'},
    'ret_3m':                {'category': 'momentum',      'label': '3月收益'},
    'ret_6m':                {'category': 'momentum',      'label': '6月收益'},
    'ret_12m':               {'category': 'momentum',      'label': '12月收益'},
    'ret_1m_skip1m':         {'category': 'momentum',      'label': '1月收益(跳1月)'},
    'max_ret_1m':            {'category': 'momentum',      'label': '月最大日收益'},

    # ── 波动率因子 ──
    'vol_1m':                {'category': 'volatility',     'label': '1月波动'},
    'vol_3m':                {'category': 'volatility',     'label': '3月波动'},
    'downside_vol':          {'category': 'volatility',     'label': '下行波动'},
    'beta':                  {'category': 'volatility',     'label': 'Beta'},
    'idio_vol':              {'category': 'volatility',     'label': '特质波动'},

    # ── 流动性因子 ──
    'amihud_illiq':          {'category': 'liquidity',     'label': 'Amihud非流动'},
    'dollar_vol_20d':        {'category': 'liquidity',     'label': '日均成交额'},
    'dollar_vol_stability':  {'category': 'liquidity',     'label': '成交额稳定性'},
    'turnover_cv_20d':       {'category': 'liquidity',     'label': '换手率变异系数'},
    'turnover_to_vol':       {'category': 'liquidity',     'label': '换手/波动比'},

    # ── 成长因子 ──
    'rev_growth_yoy':         {'category': 'growth',        'label': '营收增长'},
    'earnings_growth_yoy':    {'category': 'growth',        'label': '盈利增长'},
    'asset_growth':           {'category': 'growth',        'label': '资产增长'},
    
    # ── jqfactor 启发: 技术/量价因子 ──
    'streak':                 {'category': 'momentum',      'label': '连涨天数'},
    'overnight_5d':           {'category': 'momentum',      'label': '隔夜收益5D'},
    'min_ret_1m':             {'category': 'momentum',      'label': '月最低日收益'},
    'rsi_14':                 {'category': 'momentum',      'label': 'RSI反转'},
    'high_low_range':         {'category': 'volatility',    'label': '高低价振幅'},
    'vpt':                    {'category': 'liquidity',     'label': '量价趋势'},
    'short_rev_5d':           {'category': 'momentum',      'label': '5日反转'},
    'boll_pct_b':             {'category': 'volatility',    'label': '布林带位置'},
    'volume_ratio':           {'category': 'turnover',      'label': '量比反转'},
    
    # ── XQuant Ch9 通过因子 (第1-2轮) ──
    'gap_up':                 {'category': 'momentum',      'label': '跳空缺口持续性'},
    'opening_gap_momentum':   {'category': 'momentum',      'label': '开盘跳空动量'},
    'max_drawdown_duration':  {'category': 'extreme_events', 'label': '最大回撤持续时间'},
    # ── 日频因子实验 v2 通过 (2026-07-01) ──
    'accrual_quality_proxy':   {'category': 'fundamental_quality_proxy', 'label': '应计质量代理'},
    'panic_selling':          {'category': 'behavioral',    'label': '恐慌抛售代理'},
    
    # ── 日频因子实验 v2 通过 (2026-06-15) ──
    'trend_persistence_score': {'category': 'trend_quality', 'label': '趋势持续性评分'},
    
    # ── 日频因子实验 v2 通过 (2026-06-20) ──
    'attention_decay':          {'category': 'behavioral',    'label': '注意力衰减'},
    
    # ── 日频因子实验 v2 通过 (2026-06-21) ──
    'volume_stability':              {'category': 'volume_structure', 'label': '成交量稳定性'},
    
    # ── 吴先兴五维成长因子 (2026-06-20) ──
    'earnings_quality_proxy':      {'category': 'growth_quality', 'label': '盈利质量代理'},
    'cashflow_matching_proxy':     {'category': 'growth_quality', 'label': '现金流匹配度代理'},
    'capital_efficiency_proxy':    {'category': 'growth_quality', 'label': '资本投入效率代理'},
    'operational_efficiency_proxy': {'category': 'growth_quality', 'label': '营运效率代理'},
    'bargaining_power_proxy':      {'category': 'growth_quality', 'label': '议价能力代理'},
    
    # ── 网络拓扑因子 ──
    'scc_network_centrality':      {'category': 'network',       'label': 'SCC网络中心度'},
    
    # ── Ridge中性化因子 (C级→变废为宝) ──
    'dollar_vol_20d_neut':  {'category': 'neutralized', 'label': 'dollar_vol_20d_neut'},
    'davol_20_neut':        {'category': 'neutralized', 'label': 'davol_20_neut'},
    'abnormal_turnover_neut': {'category': 'neutralized', 'label': 'abnormal_turnover_neut'},
    'bp_neut':              {'category': 'neutralized', 'label': 'bp_neut'},
    'atr_14_neut':          {'category': 'neutralized', 'label': 'atr_14_neut'},
    'vol_1m_neut':          {'category': 'neutralized', 'label': 'vol_1m_neut'},
    'ret_3m_neut':          {'category': 'neutralized', 'label': 'ret_3m_neut'},
    'avg_turnover_1m_neut': {'category': 'neutralized', 'label': 'avg_turnover_1m_neut'},
    'vol_3m_neut':          {'category': 'neutralized', 'label': 'vol_3m_neut'},
    'max_ret_1m_neut':      {'category': 'neutralized', 'label': 'max_ret_1m_neut'},
    'downside_vol_neut':    {'category': 'neutralized', 'label': 'downside_vol_neut'},
    
    # ── jqfactor 二次补充: 趋势延伸 ──
    'bias_20':                {'category': 'momentum',      'label': '乖离率'},
    'roc_20':                 {'category': 'momentum',      'label': '变动速率'},
    'cci_14':                 {'category': 'momentum',      'label': '顺势指标'},
    'plrc_12':                {'category': 'momentum',      'label': '价格回归斜率'},
    'price_1m':               {'category': 'momentum',      'label': '价格月度偏离'},
    
    # ── jqfactor 二次补充: 情绪因子 ──
    'vr_26':                  {'category': 'sentiment',     'label': 'VR量比'},
    'vroc_12':                {'category': 'sentiment',     'label': '量变动速率'},
    'psy_12':                 {'category': 'sentiment',     'label': '心理线'},
    'money_flow_20':          {'category': 'sentiment',     'label': '资金流量'},
    'davol_20':               {'category': 'sentiment',     'label': '换手异动'},
    
    # ── jqfactor 二次补充: 收益分布 ──
    'skewness_20':            {'category': 'distribution',  'label': '收益偏度'},
    'kurtosis_20':            {'category': 'distribution',  'label': '收益峰度'},
    'sharpe_20':              {'category': 'distribution',  'label': '夏普比率'},
    'atr_14':                 {'category': 'distribution',  'label': '均幅指标'},
    
    # ── jqfactor 二次补充: 形态因子 ──
    'bull_power':             {'category': 'pattern',       'label': '多头力道'},
    'bear_power':             {'category': 'pattern',       'label': '空头力道'},
    'high_52w_rank':          {'category': 'pattern',       'label': '52周高点接近'},
    'rank_1m':                {'category': 'pattern',       'label': '收益排名反转'},
    
    # ── jqfactor 二次补充: 成交量结构 ──
    'vm_diff':                {'category': 'volume_structure', 'label': '量MACD差值'},
    'tvma_20':                {'category': 'volume_structure', 'label': '成交额均线比'},

    # ── Phase 2 独立因子补充 ──
    'high_52w_dist':          {'category': 'distribution',  'label': '52周高点距离'},
    'skew_1m':                {'category': 'distribution',  'label': '1月收益偏度'},

    # ── v5.1 基本面 Alpha 因子 ──
    'ocf_quality':            {'category': 'quality',      'label': '现金流质量'},
    'debt_coverage':          {'category': 'quality',      'label': '债务覆盖'},
    'earnings_stability':     {'category': 'quality',      'label': '盈利稳定性'},
    'asset_turnover':         {'category': 'quality',      'label': '资产周转率'},

    # ── 日频因子实验 v2 积压批量注册 (2026-07-03) ──
    # 6 历史积压 + 2 今日新 candidate
    'intraday_reversal':           {'category': 'return_decomposition',       'label': '日内反转'},
    'range_consistency':           {'category': 'price_pattern',              'label': '波幅一致性'},
    'volatility_of_volatility':    {'category': 'volatility_structure',       'label': '波动率的波动率'},
    'relative_spread_proxy':       {'category': 'liquidity_micro',            'label': '相对价差代理'},
    'trend_smoothness':            {'category': 'trend_quality',              'label': '趋势平滑度'},
    'volume_climax_reversal':      {'category': 'volume_structure',           'label': '天量反转'},
    'earnings_consistency_proxy':  {'category': 'fundamental_quality_proxy',  'label': '盈利一致性代理'},
    'ret_open_2d_proxy':           {'category': 'market_microstructure',      'label': '开盘动量2日代理'},
    # ★ 日频因子实验 v2 通过 (2026-07-13)
    'earnings_season_vol_div':     {'category': 'earnings_season_divergence',  'label': '中报窗口量价背离度'},
    # ★ 日频因子实验 v2 通过 (2026-07-17)
    'post_earnings_stability':    {'category': 'earnings_anomaly',           'label': '业绩后稳定性'},
    'earnings_volume_drift':      {'category': 'earnings_anomaly',           'label': '业绩窗量价漂移'},
    # ── Stage 2 通过 (2026-07-31) ──
    'harvey_siddique_coskew':     {'category': 'market_microstructure',     'label': 'Harvey-Siddique市场协偏度'},

    # ── 跨周期稳健因子: 交叉信号源复合因子 (2026-07-27) ──
    # rank-percentile 乘积: 两个不相关信号源相乘, 需两者同向才触发。
    # 单一 regime 更难同时让两个独立信号源失效 → 比单一因子更跨周期稳健。
    'efficiency_quality':      {'category': 'composite',     'label': '资本效率×质量'},
    'bargain_stability':       {'category': 'composite',     'label': '议价能力×换手稳定'},
    'active_quality':          {'category': 'composite',     'label': '成交额×ROE'},
    'value_momentum':          {'category': 'composite',     'label': '估值×动量'},
    'lowvol_trend':            {'category': 'composite',     'label': '低波动×趋势持续'},
    'efficient_growth':        {'category': 'composite',     'label': '资本效率×盈利成长'},
}

# ====================================================
# 跨周期稳健复合因子定义 (rank-percentile 乘积)
#   factors: [因子A, 因子B]  (必须已在 FACTOR_DEFS / 缓存中)
#   invert : 需要取反的因子 (lower=better 的信号源)
#   复合 = rank_pct(A) × rank_pct(B), 每截面(周)独立 rank→[0,1]
# ====================================================
COMPOSITE_FACTORS = {
    'efficiency_quality': {
        'factors': ['capital_efficiency_proxy', 'f_score'],
        'invert': [],
        'label': '资本效率×质量',
    },
    'bargain_stability': {
        'factors': ['bargaining_power_proxy', 'turnover_std'],
        'invert': ['turnover_std'],  # 换手波动越低越好
        'label': '议价能力×换手稳定',
    },
    'active_quality': {
        'factors': ['dollar_vol_20d', 'roe'],
        'invert': [],  # 成交额越高越好, ROE越高越好
        'label': '成交额×ROE',
    },
    'value_momentum': {
        'factors': ['bp', 'ret_3m'],
        'invert': [],  # bp越高越便宜越好, ret_3m越高越好
        'label': '估值×动量',
    },
    'lowvol_trend': {
        'factors': ['vol_3m', 'trend_persistence_score'],
        'invert': ['vol_3m'],  # 波动越低越好
        'label': '低波动×趋势持续',
    },
    'efficient_growth': {
        'factors': ['capital_efficiency_proxy', 'earnings_growth_yoy'],
        'invert': [],
        'label': '资本效率×盈利成长',
    },
}

# ==================== 因子评估参数 ====================
IC_MIN_SAMPLES = 30           # IC 计算最低股票数
DECILE_GROUPS = 10            # 十分位
DOUBLE_SORT_SIZE_GROUPS = 5   # 双重排序市值分组
DOUBLE_SORT_FACTOR_GROUPS = 5 # 双重排序因子分组
SIZE_SINGLE_SORT_GROUPS = 10  # 规模因子单变量排序分组数

# 因子通过标准
# 两个 ICIR 阈值服务于不同用途:
#   QUALITY: 因子预测力质量判断 (行业标准: |ICIR| >= 0.5 即有稳定预测力)
#   NSGA2:   GA 适应度 ratio 评分校准 (需更高值以防止评分饱和, 纯校准参数不表示因子质量)
ICIR_QUALITY_THRESHOLD = 0.5   # 因子质量: |ICIR| >= 0.5 → 预测力充分 (行业共识)
ICIR_THRESHOLD = 1.0           # v7.2fix: 降到1.0配合惩罚链, 原1.5太保守致Obj1全零 (v6=1.5)

# NSGA-II Obj1 (ICIR) 打分模式
#   'raw'   — 直接用 |ICIR| 原始值, 保留完整信号强度差异
#   'clip'  — clip(|icir|/threshold, 0, 1), 可解释性最好, 防止ICIR饱和
#   'tanh'  — tanh(|icir|/scale), [0,1] 有界 + 中间段梯度大
ICIR_OBJ_MODE = 'clip'           # ★ v7.2: clip模式 Obj1∈[0,1] 与Obj2同量纲, 叠加加法链惩罚区分优劣
ICIR_TANH_SCALE = 0.5           # tanh 尺度参数 (越小越陡)
MONOTONICITY_P_THRESHOLD = 0.05  # 单调性 t-test p < 0.05
CORRELATION_THRESHOLD = 0.5   # 与其他因子最大相关性 < 0.5
DOUBLE_SORT_P_THRESHOLD = 0.05  # 双重排序 t-test p < 0.05
STABILITY_THRESHOLD = 0.5     # ★ 时序稳定性综合分数 >= 0.5

# ==================== GA 参数 ====================
GA_POPULATION = 200
GA_GENERATIONS = 100
GA_TOURNAMENT_SIZE = 5
GA_CROSSOVER_PROB = 0.7
GA_MUTATION_PROB = 0.2
GA_MUTATION_SIGMA = 0.1      # 高斯变异标准差
GA_ELITISM = 5               # 精英保留数

# GA 适应度权重 (五维度)
GA_WEIGHT_ICIR        = 0.30    # (原0.35) 预测力
GA_WEIGHT_MONOTONICITY = 0.15   # (原0.20) 分层单调性
GA_WEIGHT_CORRELATION  = 0.10   # (原0.15) 低冗余性
GA_WEIGHT_DOUBLE_SORT  = 0.25   # (原0.30) 独立于市值的Alpha
GA_WEIGHT_STABILITY    = 0.20   # ★ 新增: 时序稳定性

# GA 通过标准: 4选3
GA_PASS_MIN_DIMENSIONS = 3

# ==================== NSGA-II 约束参数 (v5) ====================
# 品类集中度上限 — 同一品类族权重之和 ≤ 40%，打破同质化
MAX_CATEGORY_WEIGHT = 0.40
CATEGORY_CONCENTRATION_PENALTY_MULT = 0.5  # 超额部分打五折

# 品类族映射 (合并高相关品类, 每个族内权重之和上限 40%)
CONCENTRATION_GROUPS = {
    '换手/量价': ['turnover', 'sentiment', 'volume_structure'],
    '波动/分布': ['volatility', 'distribution'],
    '动量/形态': ['momentum', 'pattern'],
}
# 不设上限的品类族: size, value, profitability, growth, liquidity, quality
# (基本面品类不受集中度约束)

# 基本面品类 (用于 NSGA-II 偏好加分)
FUNDAMENTAL_CATEGORIES = {'value', 'profitability', 'growth', 'size', 'quality'}
FUNDAMENTAL_BONUS = 1.08     # 含 >=2 个基本面因子时 stability 乘 1.08
FUNDAMENTAL_MIN_COUNT = 2    # 触发加分的最少基本面因子数

# ==================== NSGA-II 约束参数 (v7: 三目标 + 加法链 + Top30对齐) ====================
# ★ v7 (06-26): 
#   Obj1 = icir_score × (1 - Σpenalties) × (1+bonus)  净预测力质量 (加法链)
#   Obj2 = stab_norm                                   纯时序稳定性
#   Obj3 = top30等权成本后年化Sharpe                    真实可执行收益
#   惩罚项从 Obj2 乘法链移至 Obj1 加法链, 对齐 top30 执行口径
#   MAX_FACTORS 从 8 降为 4 (惩罚起点与硬约束统一)
MAX_FACTORS = 4                  # ★ v7: 惩罚起点+硬约束统一为4
COMPLEXITY_PENALTY_CAP  = 0.50   # 惩罚上限 50%
COMPLEXITY_PENALTY_MODE = 'linear'  # ★ 线性惩罚

# 加法链总惩罚硬上限
MAX_TOTAL_PENALTY = 0.50         # v7.2fix: 配合clip模式减半 (原0.80, 太激进)

# 因子间相关性惩罚 — 鼓励低相关, 各因子独立贡献
MAX_ACCEPTABLE_CORR = 0.3       # 平均 pairwise |corr| 超过此值开始惩罚
CORRELATION_PENALTY_MULT = 0.8  # 惩罚乘数 (avg_corr/MAX_ACCEPTABLE_CORR)*MULT
CORRELATION_PENALTY_MAX = 0.25  # v7.2fix: 减半配合clip模式 (原0.40, 太激进)

N_OBJECTIVES = 3            # ★ v7: 三目标 NSGA-II (净质量/纯稳定/成本后夏普)

# === OOS 验证 (v6.1 Step 3) ===
OOS_SPLIT_DATE = '2025-01-01'   # v6.1: 2021-2024训练(含牛熊周期), 2025-2026测试

# === Meta Learner 元验证集 (2026-07-15) ===
# "无接触"数据段: 完全不被任何因子挖掘、策略进化使用。
# 仅 Meta Learner 的最终方法论验证可访问, 且系统生命周期内仅一次。
# META_HOLDOUT env var 控制是否在 pipeline 中隔离此段:
#   - META_HOLDOUT=1: run_fa.py 训练数据截断于 META_VALIDATION_START 前
#   - META_HOLDOUT=0 (默认): run_fa.py 使用全量数据 (v7.9 默认行为)
META_VALIDATION_START = '2025-07-01'
META_VALIDATION_END   = '2026-06-25'  # 当前数据末端
META_HOLDOUT_ENABLED  = False         # 默认关闭, 设 env META_HOLDOUT=1 开启

# 滚动窗口 (WFA)
GA_TRAIN_WINDOW_WEEKS = 156   # 3年
GA_TEST_WINDOW_WEEKS  = 52    # 1年外样本

# ==================== 组合模拟参数 ====================
PORTFOLIO_TOP_N = 30          # 持仓数
PORTFOLIO_WEIGHTING = 'equal' # 等权
PORTFOLIO_MAX_TURNOVER = 0.50 # 单边最大换手
PORTFOLIO_COST = {
    'commission': 0.0001,
    'stamp_tax':  0.001,
    'slippage':   0.003,
}

# ==================== 因果发现 Top20 因子 (Double ML, 按 |ATE| 排序, 2026-07-23) ====================
# 这些因子在因果推断中 ATE 效应量显著，但部分因子 ICIR 低（被传统预测筛选低估）。
# 用于 Composer 搜索空间扩容: 作为 base 因子强制进入配对池，释放被 ICIR 筛选屏蔽的信号源。
# 依据: 14 个因果独有因子已被 ICIR 低估 (市值/盈利增长/52周高点/资本效率等)。
CAUSAL_FACTORS_TOP20 = [
    'vol_1m',                     # ATE -0.0027, 波动率
    'vol_3m',                     # ATE +0.0027, 长期波动
    'dollar_vol_20d',             # ATE +0.0024, 日均成交额
    'high_52w_dist',              # ATE +0.0023, 52周高点距离
    'high_52w_rank',              # ATE +0.0023, 52周高点接近
    'capital_efficiency_proxy',   # ATE +0.0020, 资本效率
    'bargaining_power_proxy',     # ATE +0.0020, 议价能力 (⚠️ base-5成员, 禁投作为独立因子但应允许进入composer配对)
    'earnings_growth_yoy',        # ATE +0.0019, 盈利增长
    'ln_mcap',                    # ATE +0.0019, 对数市值
    'max_drawdown_duration',      # ATE -0.0017, 最大回撤持续期
    'ln_circulating_mcap',        # ATE +0.0016, 对数流通市值
    'rev_growth_yoy',             # ATE +0.0016, 营收增长
    'skewness_20',                # ATE +0.0015, 20日收益偏度
    'psy_12',                     # ATE -0.0014, 心理线
    'high_low_range',             # ATE -0.0013, 高低价振幅
    'skew_1m',                    # ATE +0.0013, 1月收益偏度
    'amihud_illiq',               # ATE +0.0013, Amihud非流动性
    'vpt',                        # ATE -0.0013, 量价趋势
    'ret_3m',                     # ATE +0.0012, 3月收益
    'turnover_std',               # ATE +0.0012, 换手波动
]

# ==================== 调试 ====================
VERBOSE = True
RANDOM_SEED = 42

# ==================== v0.6 实验性升级开关 ====================
# 2026-08-28 用户钦定: 默认全开 = v0.6 行为 (评价宪法生效态)。
# 回退 v0.5.2: 全置 False 即可 (基线: _versioning/v0.5.2_baseline/)
V06_EXPERIMENTAL = {
    # P0-3 多重检验: DSR/PBO/BH-FDR (影子=False只输出, 生效=硬门禁)
    "multiple_testing_shadow": True,   # 影子模式: 计算并输出 DSR/PBO/FDR, 不改判定
    "multiple_testing_gate": True,     # 生效: DSR<阈值 或 PBO>0.40 或 FDR 不过 → S6 拒绝
    # v0.6.1 (2026-08-29) 口径校准: 原 0.90 在 S5 组合收益口径下全杀 —
    # 实测 JQ 王者 overnight_raw(JQ+131.4%)/overnight5(JQ+173%组合) 在
    # S5 top-80/200 等权组合口径下 DSR(trials=10)=0.095~0.122, 而纯噪声
    # 序列 DSR≈0.16~0.40 — 该口径下王者与噪声不可区分 (5 轮实证: S5 通过
    # 2/2 全被 S6 拒 → 流水线死锁)。裁决力移交配对增量门禁 (control/treatment
    # 对称口径), DSR 降级为"极端负 alpha 排渣": 仅 DSR<0.01 (显著劣于折损基准)
    # 才硬拒。符合铁则: local 仅否决不排序, JQ 唯一真相源。
    "dsr_min_probability": 0.01,
    # DSR trials 批次级: 多重检验次数=本批候选数 (原用全库 248 attempts 过度折损)
    "dsr_trials_batch_level": True,
    "pbo_max": 0.40,
    "fdr_alpha": 0.10,
    # P0-2 行为聚类同质性门禁
    "behavior_homogeneity_enabled": True,    # 行为指纹聚类 + 准入门禁 + 生成前指令
    "behavior_similarity_reject": 0.92,      # 在线最近邻相似度硬拒
    "behavior_similarity_substitute": 0.82,  # 替代品标记
    "behavior_cluster_threshold": 0.74,      # 层次聚类边界
    "behavior_crowded_cluster_size": 8,      # 拥挤簇阈值
    # P0-1 密封 holdout + 盲评边界
    "holdout_enabled": True,                 # local 隐藏区 + JQ verdict 分级
    "holdout_local_start": "2025-07-01",     # local 面板隐藏区起点 (与 META_VALIDATION_START 对齐)
    # P1-1 增量边际门禁 (组合层)
    "incremental_margin_enabled": True,
    "incremental_net_ir_min": 0.10,
    "incremental_dd_deterioration_max": 0.02,
    # P1-2 方向微战役
    "direction_campaign_enabled": True,
    "campaign_max_attempts": 3,
    "campaign_early_stop_misses": 3,
    "campaign_cooldown": 2,
    # P1-3 G 阶段提示词宪法约束注入 (build_prompt + system_prompt)
    "g_prompt_constitution": True,           # 评价宪法条款: 增量边际/行为冗余/预算/holdout不透明
    # P2 预算账本 / 污染账本
    "budget_ledger_enabled": True,           # 族级+世代配额账本
    "budget_max_per_family": 100,
    "budget_max_per_generation": 500,
    "manual_contamination_enabled": True,    # 人工回测污染登记
    # P2 策略规格对象化 (交付物规格, 不影响流水线判定)
    "strategy_spec_enabled": True,
}

# ==================== v0.7 频率对称双通道 (2026-08-29) ====================
# 架构 (v07_forge_s1_calibration_plan_20260829.md v3):
#   生成层频率中立 (LLM/gp_breed 双语境) × 裁决层 S1 分频 XOR 路由 × 执行层 S5/S6/JQ 唯一周频口径
# enabled=False = 影子/关闭: 全部候选走 daily lane (现状), 零回归。
# 开启后: natural_freq=="weekly" 的候选走周频裁决器 (weekly_prices.parquet, 周频 ICIR ≥ τ_w),
#         其余走日频 FactorICComputer fwd5 (现状)。一因子一通道, 不重复裁决。
# v0.7 P4 (2026-08-29): enabled=True 转正 (P2 影子轮次零回归 + P3 冒烟 22/22 后)。
#   τ_w 动态读取: lane_calibration.tau_w_effective (JQ 锚点滚动确认) 优先于本文件基线值。
V07_DUAL_LANE = {
    "enabled": True,
    # τ_w 周频 ICIR 门槛 — pandas 周频口径 (rolling min_periods=全窗口, 与 JQ 执行口径一致)。
    # 🔴 校准锚定 (2026-08-29 实测, 非 Forge nanstd 口径):
    #   Forge fitness 口径 ts_std=0.587 / tsmin=0.563 (np.nanstd 宽松窗口);
    #   pandas 周频口径同公式实测 ts_std=0.183(JQ PASS +91.91%) / tsmin=0.230(JQ MARGINAL +38.65%)
    #   → τ_w 必须 ≤0.183 否则误杀 JQ PASS 因子, 取 0.15 留余量。
    #   第五个口径错配: Forge nanstd(min_periods宽松) vs pandas rolling(30).std(min_periods=30)。
    "weekly_icir_threshold": 0.15,
    # 周频 IC 绝对值下限 (排渣, 保守从宽; 裁决主力是 ICIR)
    "weekly_ic_min": 0.005,
    # 活性校验 (影子): 近 N 年周频 ICIR 仅输出不拦截。
    # 🔴 实测 ts_std/tsmin 近 2 年 ICIR 仅 0.074/0.140 仍全期 JQ PASS →
    #    硬拦截 0.2 会误杀已验证 PASS 因子; P1 阶段降级为影子, P3 校准集充足以再定拦截线。
    "weekly_activity_years": 2.0,
    "weekly_activity_min_icir": 0.05,
    "weekly_activity_gate": False,
    # 周频裁决逐周最小股票数
    "weekly_min_stocks": 30,
    # 校准集路径 (P3/P4: JQ 反馈 → lane_calibration 滚动更新 τ_w)
    "calibration_path": "data/lane_calibration.json",
    # 周频数据源 (weekly_prices.parquet, 与 Forge 建模同源)
    "weekly_parquet": "output/ap_batch/cache/weekly_prices.parquet",
}
