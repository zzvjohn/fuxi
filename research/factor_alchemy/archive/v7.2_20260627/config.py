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
    'rsi_14':                 {'category': 'momentum',      'label': 'RSI反转'},
    'high_low_range':         {'category': 'volatility',    'label': '高低价振幅'},
    'vpt':                    {'category': 'liquidity',     'label': '量价趋势'},
    'short_rev_5d':           {'category': 'momentum',      'label': '5日反转'},
    'boll_pct_b':             {'category': 'volatility',    'label': '布林带位置'},
    'volume_ratio':           {'category': 'turnover',      'label': '量比反转'},
    
    # ── XQuant Ch9 通过因子 (第1-2轮) ──
    'gap_up':                 {'category': 'momentum',      'label': '跳空缺口持续性'},
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

# ==================== 调试 ====================
VERBOSE = True
RANDOM_SEED = 42
