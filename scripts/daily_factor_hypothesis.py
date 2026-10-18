"""
每日因子假设实验 — XQuant Ch9 日频版 (v2: 原创去重)
==================================================
核心原则: 每天提 ~20 个**全新**因子假设，与现有因子库、FA系统、已测因子全部不同。
入库策略: 全量入库 — 通过阈值(PASS)→status=candidate触发FA; 未通过→status=reserve作为储备。

去重保护:
  1. 排除 FA 系统已有 71 个因子 (factors/__init__.py)
  2. 排除因子池已收录因子 (data/passed_factor_pool.csv)
  3. 排除历史已测因子 (data/tested_factors.json)
  4. 每次运行后标记已测，永不重复

原创来源:
  - 学术论文异象 (Anomaly Library): 全球量化文献已验证异象的A股适配版
  - 市场微观结构: 订单流/知情交易/流动性供给
  - 行为金融: 认知偏差/注意力/情绪
  - 最新研报驱动的动态构造
  - LLM生成因子 (v3): 由每日策略研究产出的LLM挖掘因子，优先测试

来源: 《XQuant: 人人都是量化交易员》第9章 + 原创因子图书馆
创建: 2026-06-15 | v2: 2026-06-15 (原创去重机制) | v3: 2026-06-21 (LLM槽位) | v4: 2026-06-22 (113) | v5: 2026-06-30 (116, +研究提炼)
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import json
import random

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 扩展数据加载 (基本面 + 订单流 + 两融 + 龙虎榜 + 北向)
try:
    from scripts.data_loader_ext import (
        load_fundamental_wide, load_moneyflow_wide, load_margin_wide,
        load_analyst_forecast_wide, load_top_list_wide, load_hk_hold_wide,
        load_industry_wide,
        load_fina_growth_wide, load_valuation_wide, load_industry_l1_wide,
        load_feature_peer_wide, load_limit_up_wide,
        load_lion_anchor_wide,
        load_kuaidao_fund_wide,
    )
    _EXT_DATA_AVAILABLE = True
except ImportError:
    _EXT_DATA_AVAILABLE = False

# ============================================================
# 路径配置
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent
REPORT_DIR = PROJECT_ROOT / 'reports'
DAILY_REPORT_DIR = REPORT_DIR / 'factor_daily'
DAILY_REPORT_DIR.mkdir(exist_ok=True, parents=True)

POOL_CSV = PROJECT_ROOT / 'data' / 'passed_factor_pool.csv'
TESTED_JSON = PROJECT_ROOT / 'data' / 'tested_factors.json'

# Stage 1 (每日策略研究) 报告目录与闸门
STAGE1_DAILY_DIR = REPORT_DIR / 'daily'

# Stage 1 → Stage 2 因子扩充交接文件
# Stage 1 探索产出全新因子提案 (data/stage1_factor_proposals.json), Stage 2 自动消费验证
STAGE1_PROPOSALS_FILE = PROJECT_ROOT / 'data' / 'stage1_factor_proposals.json'


def stage1_report_path(today: datetime = None) -> Path:
    """返回当日 Stage 1 策略研究报告的路径 (reports/daily/YYYY-MM-DD_strategy_research.md)。"""
    today = today or datetime.now()
    return STAGE1_DAILY_DIR / f"{today.strftime('%Y-%m-%d')}_strategy_research.md"


def check_stage1_gate(require_stage1: bool, today: datetime = None) -> bool:
    """Stage 2 前置闸门: 检查当日 Stage 1 报告是否存在。

    返回 True = 放行; 返回 False = 拦截。
    require_stage1=False 时永远放行 (手动/历史兼容模式)。
    """
    if not require_stage1:
        return True
    p = stage1_report_path(today)
    if p.exists():
        print(f"  [闸门✅] 当日 Stage 1 报告存在: {p.name}")
        return True
    print(f"  [闸门🚧] 拦截: 当日 Stage 1 报告不存在 ({p.name})")
    print(f"  Stage 2 中止 — 不消耗因子库存、不写报告。")
    print(f"  (如需跳过闸门手动运行, 去掉 --require-stage1)")
    return False

# 通过阈值 (XQuant Ch9 双重门坎，不可事后修改)
ICIR_THRESHOLD = 0.3
IC_PCT_THRESHOLD = 0.55

# ── v0.7.29 跨截面样本量门槛 (防「稀疏截面假 PASS」) ─────────────────────────
# 背景: ICIR = |mean(IC)/std(IC)| 的统计可信度建立在「IC 序列足够长」之上, 但原实现
#   只对**单截面股票数**设了门槛 (len(common) < 30 → 跳过该截面), 对**跨截面数量**
#   没有任何下限。实测 (2026-09-16, P-20260916-003/004 复评):
#     `large_order_flow_zscore_20d` 在 NaN 诚信口径下**仅 37 个有效截面**
#     (占面板 267 的 13.9%), 却算出 ICIR 0.401 → 会被判 PASS。
#   ⇒ 若不先补此门槛就翻 FUXI_MF_MISSING=nan, 会放进一批「稀疏截面假 PASS」。
# 判据锚点 (🔴 锚在实现而非命名): 用**相对覆盖率** + 绝对地板, 而非绝对 n_dates。
#   理由: 绝对门槛会随面板长度漂移 —— tested_factors.json 实测默认 270 日面板中位
#   n_dates 仅 155, 而 400 日复评面板为 256; 用绝对数会在面板变短时误杀、变长时放松。
#   env FUXI_MIN_IC_DATE_RATIO / FUXI_MIN_IC_DATES_FLOOR 可覆盖; ratio=0 关闭相对维度。
MIN_IC_DATE_RATIO = 0.5   # n_dates ≥ 面板可用截面数 × 0.5
MIN_IC_DATES_FLOOR = 60   # 且 n_dates ≥ 60 (防极短面板下 ratio 失效)


def _ic_dates_gate(n_dates, n_panel):
    """跨截面样本量门槛。返回 (是否达标, 需要的最少截面数)。

    达标才有资格判 PASS; 不达标时 icir 置 NaN (但保留 icir_raw 供诊断)。
    属「单向收紧」型判据 —— 只会把 PASS 变 FAIL, 不会凭空造出 PASS。
    """
    try:
        ratio = float(os.environ.get('FUXI_MIN_IC_DATE_RATIO', MIN_IC_DATE_RATIO))
    except (TypeError, ValueError):
        ratio = MIN_IC_DATE_RATIO
    try:
        floor = int(os.environ.get('FUXI_MIN_IC_DATES_FLOOR', MIN_IC_DATES_FLOOR))
    except (TypeError, ValueError):
        floor = MIN_IC_DATES_FLOOR
    if ratio > 0:
        need = max(floor, int(np.ceil(ratio * max(int(n_panel), 1))))
    else:
        need = floor
    return bool(int(n_dates) >= need), int(need)

# LLM因子注册槽位 (由每日策略研究产出填充)
LLM_FACTOR_REGISTRY = [
    {
        'name': 'qmj_profitability_sharpe',
        'label': 'QMJ盈利质量-收益稳定性(QMJ Profitability Sharpe)',
        'category': 'fundamental_quality_proxy',
        'hypothesis': 'Li et al.(2026) QMJ: A股盈利能力是质量因子唯一有效支柱。60日收益Sharpe代理盈利质量，高质量=高Sharpe(稳定正向收益)',
        'logic': 'QMJ论文四支柱仅Profitability有效(成长/安全/分红无效)。量价代理: 滚动Sharpe=盈利质量—高Sharpe→稳定盈利能力→正向Alpha',
        'formula': "close_p.pct_change().rolling(60).mean() / (close_p.pct_change().rolling(60).std() + 1e-6)",
        'direction': 'long',
    },
    {
        'name': 'qmj_earnings_consistency',
        'label': 'QMJ盈利质量-收益一致性(QMJ Earnings Consistency)',
        'category': 'fundamental_quality_proxy',
        'hypothesis': 'Li et al.(2026) QMJ: 盈利能力唯一有效。60日正收益占比=盈利一致性代理，高一致性→盈利可预测→低风险Alpha',
        'logic': '盈利质量高的公司收益稳定、可预测。量价代理: 60日正收益天数占比越高=盈利越一致=质量越高',
        'formula': "(close_p.pct_change() > 0).astype(float).rolling(60).mean()",
        'direction': 'long',
    },
    {
        'name': 'qmj_gross_profitability_proxy',
        'label': 'QMJ盈利质量-毛利率代理(QMJ Gross Profit Proxy)',
        'category': 'fundamental_quality_proxy',
        'hypothesis': 'Li et al.(2026)+Novy-Marx(2013): 毛利率(GPOA)是最强盈利因子。60日累计收益/累计振幅=毛利率量价代理',
        'logic': '高毛利率公司=定价权强=盈利质量高。量价代理: 累计收益/(高-低振幅)→高比值=低波动高收益=毛利率高→质量高',
        'formula': "close_p.pct_change().rolling(60).sum() / (close_p.rolling(60).max() - close_p.rolling(60).min()).div(close_p.rolling(60).mean()).replace(0, np.nan)",
        'direction': 'long',
    },
]  # List[dict]: 每项含 name/label/category/hypothesis/logic/formula/direction


# ============================================================
# Part A: 去重基础设施
# ============================================================

def load_fa_factor_names():
    """从 FA 系统 __init__.py 读取全部 71 个因子名称"""
    init_path = PROJECT_ROOT / 'research' / 'factor_alchemy' / 'factors' / '__init__.py'
    names = set()
    if init_path.exists():
        content = init_path.read_text(encoding='utf-8')
        for line in content.split('\n'):
            # 匹配 "'name': ClassName" 行
            stripped = line.strip()
            if stripped.startswith("'") and "':" in stripped:
                name = stripped.split("'")[1]
                if name and len(name) > 1 and name[0].isalpha():
                    names.add(name)
    return names


def load_pool_names():
    """从因子池 CSV 读取已收录因子名称"""
    if POOL_CSV.exists():
        df = pd.read_csv(POOL_CSV)
        return set(df['name'].tolist())
    return set()


def load_tested_history():
    """加载已测因子历史"""
    if TESTED_JSON.exists():
        with open(TESTED_JSON, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('tested', {}), data.get('names', set())
    return {}, set()


def save_tested_history(tested_dict, tested_names):
    """保存已测因子历史"""
    TESTED_JSON.parent.mkdir(exist_ok=True, parents=True)
    with open(TESTED_JSON, 'w', encoding='utf-8') as f:
        json.dump({
            'tested': tested_dict,
            'names': list(tested_names),
            'last_updated': datetime.now().isoformat(),
            'total_tested': len(tested_names),
        }, f, ensure_ascii=False, indent=2)


def get_blocked_names():
    """获取全部已占用因子名（FA + 因子池 + 已测）"""
    blocked = set()
    blocked |= load_fa_factor_names()
    blocked |= load_pool_names()
    _, tested_names = load_tested_history()
    blocked |= set(tested_names)
    return blocked


# ============================================================
# Part A2: Stage 1 → Stage 2 因子扩充交接
# ============================================================
# ── P-20260910-002: 可证伪提案前置门 ──────────────────────
# 因子须带 (hypothesis/mechanism/predicted_direction/refutation_conditions) 四元组。
# 缺 refutation_conditions 时自动补默认模板 (宽松); FUXI_FALSIFIABLE_GATE=enforce
# 时四元组任一缺失直接剔除。方向证伪比对在 run_daily Step 3 执行。
DEFAULT_REFUTATION = ("若截面 IC 符号与预测方向相反或 |ICIR|<0.05, 则假设被证伪, "
                      "以本地实证 IC 方向为准 (方向本地实证翻转铁律)")

_DIRECTION_SIGN = {"long": 1, "positive": 1, "short": -1, "negative": -1}


def _ensure_falsifiable(hyp: dict, source: str = "") -> dict:
    """补齐可证伪四元组字段。返回 (hyp, ok)。FUXI_FALSIFIABLE_GATE=enforce 时
    四元组 (hypothesis/logic/direction/refutation_conditions) 任一缺失 → ok=False。"""
    enforce = os.environ.get("FUXI_FALSIFIABLE_GATE") == "enforce"
    hyp = dict(hyp)
    hyp['direction'] = hyp.get('direction') or 'long'
    hyp['refutation_conditions'] = hyp.get('refutation_conditions') or DEFAULT_REFUTATION
    ok = True
    if enforce:
        missing = [k for k in ('hypothesis', 'logic', 'direction', 'refutation_conditions')
                   if not str(hyp.get(k, '')).strip()]
        if missing:
            print(f"  ⛔ [可证伪门] {hyp.get('name', '?')} ({source}) 四元组缺失: {missing}")
            ok = False
    return hyp, ok


def _record_falsification_event(factor_name: str, predicted_direction: str,
                                ic_mean: float, icir: float, date_str: str):
    """P-20260910-002: 方向证伪事件写入 Experience Memory falsification_log。"""
    try:
        mem_path = PROJECT_ROOT / 'data' / 'experience_memory.json'
        data = json.loads(mem_path.read_text(encoding='utf-8')) if mem_path.exists() else {}
        log = data.setdefault('falsification_log', [])
        log.append({
            'factor_name': factor_name,
            'date': date_str,
            'predicted_direction': predicted_direction,
            'ic_mean': float(ic_mean),
            'icir': float(icir),
            'note': '截面 IC 符号与预测方向相反, 以本地实证方向为准',
        })
        mem_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception as e:
        print(f"    [falsification_log] 写入失败 (不阻塞): {e}")


def load_stage1_proposals():
    """从 Stage 1 因子扩充探索产出读取待验证因子提案。

    自动排除已测/已入池/FA 已有因子 (与 get_blocked_names 一致)。
    返回 List[dict], 每项已归一化为标准字段 (name/label/formula/hypothesis/logic/category/direction/refutation_conditions)。
    """
    if not STAGE1_PROPOSALS_FILE.exists():
        return []
    try:
        data = json.loads(STAGE1_PROPOSALS_FILE.read_text(encoding='utf-8'))
    except Exception:
        return []
    proposals = data.get('proposals', []) if isinstance(data, dict) else data
    blocked = get_blocked_names()  # FA + 因子池 + 已测

    normalized = []
    n_rejected = 0
    for p in proposals:
        fname = p.get('factor_name') or p.get('name')
        if not fname or fname in blocked:
            continue
        # 归一化到标准字段名 (兼容 evaluate_factor + mark_as_tested 的访问)
        np_norm = dict(p)
        np_norm['name'] = fname
        np_norm['label'] = fname
        np_norm['formula'] = p.get('formula_pandas', p.get('formula', ''))
        np_norm['hypothesis'] = p.get('economic_rationale', p.get('hypothesis', ''))
        np_norm['logic'] = p.get('economic_rationale', p.get('logic', ''))
        np_norm['category'] = p.get('paradigm', p.get('target_problem', ''))
        np_norm['direction'] = p.get('direction', p.get('predicted_direction', 'long'))
        np_norm['predicted_direction'] = np_norm['direction']

        # 🆕 P-20260910-002: 可证伪四元组补齐 / enforce 剔除
        np_norm, _ok = _ensure_falsifiable(np_norm, source='stage1')
        if not _ok:
            n_rejected += 1
            continue

        # 🆕 P-021: 公式预检 — 提前拒绝不可执行的公式, 避免浪费 S1-S5 评估时间
        is_valid, warnings, _ = validate_and_preprocess_formula(
            np_norm['formula'], fname
        )
        if not is_valid:
            n_rejected += 1
            print(f"  ⛔ 预检拒绝 {fname}: {'; '.join(warnings)}")
            continue
        if warnings:
            for w in warnings:
                print(f"  ⚠️ {fname}: {w}")

        normalized.append(np_norm)

    if n_rejected:
        print(f"  [预检] {n_rejected} 个提案因子公式不可执行, 已跳过")
    return normalized


# ═══════════════════════════════════════════════════════════
# 公式预检 & 自动修复 (P-021: pandas 属性冲突防护)
# ═══════════════════════════════════════════════════════════

# Forge 风格函数调用 — 在 pandas eval/exec 中不可执行
# 使用 (?<!\.) 负向后顾避免误匹配 pandas 方法 (.rank()/.abs()/.div() 等)
_FORGE_FUNCTION_PATTERNS = [
    # 明确的 Forge 原语 (永远不会是 pandas 方法)
    r'\bcs_rank\s*\(', r'\bts_mean\s*\(', r'\bts_std\s*\(', r'\bts_min\s*\(',
    r'\bts_max\s*\(', r'\bts_sum\s*\(', r'\bts_corr\s*\(', r'\bts_cov\s*\(',
    r'\bneg\s*\(', r'\bscale\s*\(', r'\bdelta\s*\(', r'\bdelay\s*\(',
    r'\bsigned_power\s*\(', r'\brolling_min\s*\(', r'\brolling_max\s*\(',
    # 可能冲突的函数 → 仅匹配独立调用 (前面无 . → 非 pandas 方法链)
    r'(?<!\.)\brank\s*\(', r'(?<!\.)\bdiv\s*\(', r'(?<!\.)\bsub\s*\(',
    r'(?<!\.)\badd\s*\(', r'(?<!\.)\bmul\s*\(', r'(?<!\.)\babs\s*\(',
    r'(?<!\.)\blog\s*\(', r'(?<!\.)\bsqrt\s*\(', r'(?<!\.)\bsign\s*\(',
]

# 虚构列名 — 不在 OHLCV + 资金流数据中
_GHOST_COLUMNS = [
    'rd_expense', 'rd_capitalized', 'rd_staff', 'revenue', 'goodwill',
    'net_profit', 'total_assets', 'north_money',
    'concept_label', 'industry_code',
    # 注意: buy_elg_vol 已于 P-20260814-002 接入 data_loader_ext (load_moneyflow_wide 返回
    # 第4/5个字段), 2026-09-07 从幻影列名中移除 (P-20260907-001 落地), 防公式预检误杀。
]

# 预检错误提示中的真实可用字段清单 (与 data_loader_ext 加载保持同步)
_AVAILABLE_COLS_HINT = (
    "OHLCV(open/high/low/close/vol/amount/pre_close) + "
    "衍生(returns/overnight/amplitude/turnover/hl_ratio, 与 FactorICComputer 同口径) + "
    "资金流(buy_sm_vol/sell_sm_vol/buy_lg_vol/sell_lg_vol/buy_elg_vol/sell_elg_vol/net_mf_amount) + "
    "龙虎榜(lhb_flag/lhb_net_amount/lhb_net_rate/lhb_inst_*/lhb_north_*/lhb_yz_*) + "
    "两融(rzye/rqye/rzmre/rqyl/rqchl/rzche/rqmcl/rzrqye) + 北向(north_vol/north_ratio) + "
    "基本面(roe/eps/eps_qoq/netprofit_yoy/roe_dt/tr_yoy) + 分析师(eps_disp/tp_disp/rating_disp/n_cover) + "
    "估值(circ_mv/total_mv/pb/ps_ttm/pe_ttm/dv_ratio/dv_ttm/kd_ps_pct_own) + "
    "快刀基本面(kd_total_assets/kd_money_cap/kd_trad_asset/kd_accounts_receiv/kd_notes_receiv/"
    "kd_goodwill/kd_st_borr/kd_lt_borr/kd_bond_payable/kd_adv_receipts/kd_contract_liab/"
    "kd_revenue_ttm/kd_n_income_ttm/kd_revenue_ann/kd_n_income_ann/"
    "kd_ncf_act_ttm/kd_c_fr_sale_sg_ttm/kd_capex_ttm/kd_profit_dedt_ttm/"
    "kd_holder_num/kd_holder_num_yoy/kd_top10_hold_sum/kd_top10_hold_ratio/kd_cash_div_12m/"
    "kd_holder_net_ratio_3m/kd_holder_in_ratio_3m/kd_holder_de_ratio_3m/"
    "kd_repurchase_flag_3m/kd_repurchase_price_3m/kd_repurchase_amt_12m)"
)

# Python 内置冲突 — 需要预处理的列名
# 'open' 是 Python 内置函数 open(); 在 eval() 中如果 locals 未显式覆盖,
# 名字解析会穿透到 builtins 导致 TypeError
_PYTHON_BUILTIN_CONFLICTS = {
    'open': 'open',   # 公式中 `open` 已在 local_ns 中显式覆盖, 此处仅做日志告警
}


def validate_and_preprocess_formula(formula: str, factor_name: str = "") -> tuple:
    """预检因子公式的语法安全性和列名有效性。

    返回 (is_valid: bool, warnings: list[str], fixed_formula: str)
    - is_valid=False 时, 公式不可执行 (Forge 语法 / 幻觉列名)
    - 警告: Python 内置冲突 (已自动兼容) / 可疑模式
    """
    import re

    warnings = []
    if not formula or not isinstance(formula, str):
        return False, ["公式为空或非字符串"], ""

    # ── 1. 检测 Forge 风格函数调用 ──
    forge_hits = []
    for pat in _FORGE_FUNCTION_PATTERNS:
        if re.search(pat, formula):
            # 从正则提取可读函数名 (去掉 lookbehind/lookahead/转义)
            clean_name = pat.replace(r'(?<!\.)', '').replace(r'\b', '').replace(r'\s*\(', '(')
            forge_hits.append(clean_name)
    if forge_hits:
        warnings.append(
            f"公式含 Forge 风格函数调用 {forge_hits}, "
            f"应使用 pandas .rolling()/.rank(pct=True) 语法"
        )
        return False, warnings, formula

    # ── 2. 检测幻觉列名 ──
    ghost_hits = []
    for col in _GHOST_COLUMNS:
        # 使用词边界匹配, 避免 'amount' 匹配到 'amount_ratio' 的前缀误判
        if re.search(r'\b' + re.escape(col) + r'\b', formula):
            ghost_hits.append(col)
    if ghost_hits:
        warnings.append(
            f"公式引用了不存在的列名 {ghost_hits}, "
            f"数据仅含 {_AVAILABLE_COLS_HINT}"
        )
        return False, warnings, formula

    # ── 3. 检测 Python 内置冲突 (open 已在 local_ns 中覆盖, 仅告警) ──
    for conflict_name, _ in _PYTHON_BUILTIN_CONFLICTS.items():
        if re.search(r'\b' + re.escape(conflict_name) + r'\b', formula):
            warnings.append(
                f"公式使用 '{conflict_name}' (Python 内置), "
                f"已在 eval 上下文中覆盖为 DataFrame, 可正常执行"
            )

    # ── 4. Python 语法检查 (括号配平 / 表达式合法性) ──
    # P-20260901-007: ast.parse 预检, 提前拒绝括号不配平等语法错误公式,
    # 避免 eval 阶段 SyntaxError 后误剪枝有效提案。
    # 实证: margin_flow_volume_ratio_zscore 公式 (A-A).rolling(...) 缺括号配平,
    #       预检漏过 → eval 报 unmatched ')' → 被 prune_stage1_proposals 剪枝。
    import ast as _ast
    try:
        _ast.parse(formula, mode='eval')
    except SyntaxError:
        try:
            _ast.parse(formula, mode='exec')
        except SyntaxError as _se:
            warnings.append(f"公式语法错误 (括号不配平或非法表达式): {_se.msg}")
            return False, warnings, formula

    # ── 5. pandas Rolling 对象非法链检测 ──
    # P-20260904: Ralph LLM 生成 .rolling(N).pct_change() → eval AttributeError
    # (Rolling 无 pct_change; 正确: series.pct_change(N) 或 .rolling().mean().pct_change())
    _roll_bad = re.search(r'\.rolling\([^)]*\)\s*\)?\s*\.\s*(pct_change|pct_chg)\s*\(', formula)
    if _roll_bad:
        warnings.append(
            f"非法链: Rolling 对象无 {_roll_bad.group(1)} 方法 "
            f"(应改用 series.{_roll_bad.group(1)}(N) 或 .rolling().mean() 后再调用)"
        )
        return False, warnings, formula

    # ── 6. P-20260910-001: 静态因果闭包校验 (shadow 默认: 只记录不拦截) ──
    try:
        import sys as _sys
        from pathlib import Path as _P
        _fa_dir = str(_P(__file__).resolve().parents[1] / "research" / "factor_alchemy")
        if _fa_dir not in _sys.path:
            _sys.path.insert(0, _fa_dir)
        from causal_closure_check import check_causal_closure, record_batch, should_block
        cc = check_causal_closure(formula)
        if cc["verdict"] != "ok":
            record_batch([{
                "factor": factor_name, "formula_preview": formula[:80],
                "verdict": cc["verdict"], "violations": cc["violations"],
                "warnings": cc["warnings"],
            }])
            if should_block(cc["verdict"]):
                warnings.append(
                    f"因果闭包违规 (前视): {'; '.join(cc['violations'])}"
                )
                return False, warnings, formula
            elif cc["verdict"] == "fatal":
                warnings.append(
                    f"[shadow] 因果闭包违规 (未拦截): {'; '.join(cc['violations'])}"
                )
            else:
                warnings.append(f"[shadow] 因果闭包警告 (未拦截): {'; '.join(cc['warnings'])}")
    except Exception:
        pass  # 校验器不可用时静默降级, 不阻塞 Stage2

    return True, warnings, formula


def prune_stage1_proposals(tested_name):
    """测试后将因子从提案文件移除 (状态已移交 tested_factors.json 管理)。"""
    if not STAGE1_PROPOSALS_FILE.exists():
        return
    try:
        data = json.loads(STAGE1_PROPOSALS_FILE.read_text(encoding='utf-8'))
    except Exception:
        return
    proposals = data.get('proposals', []) if isinstance(data, dict) else data
    new_proposals = [p for p in proposals if (p.get('factor_name') or p.get('name')) != tested_name]
    if len(new_proposals) != len(proposals):
        # 2026-09-07 修复: 保留 dict 中的其他键 (如 pending_fina), 旧逻辑 {'proposals': ...}
        # 重写会静默丢失 pending_fina 待财务数据条目。
        if isinstance(data, dict):
            out = dict(data)
            out['proposals'] = new_proposals
        else:
            out = new_proposals
        STAGE1_PROPOSALS_FILE.write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')


# ============================================================
# Part B: 原创因子图书馆 (78个，v3扩充版 2026-06-22)
# ============================================================
# 设计原则:
#   1. 每个因子名不得与 FA 71 / 因子池 / 已测 任何已有因子重名
#   2. 构造公式必须完全独立，不与其他因子高相关 (>0.6) 的设计思路
#   3. 来源: 学术文献异象 + 市场微观结构 + 行为金融 + 券商研报
#   4. 全部公式仅依赖 price/volume (OHLCV)，不需要基本面数据
# v3扩充(06-22): +25因子, 来源: 清华PBCSF 208异象/广发Level-2/华泰微观结构/研报驱动
#   新增类别: market_microstructure/tail_risk/supply_demand/sentiment_crowding

NOVEL_FACTOR_LIBRARY = [
    # ============= 收益分解类 (Overnight/Intraday) =============
    # 基于 中信建投 (2025.11) 隔夜-日内异象研究报告
    {
        'id': 'novel_001', 'name': 'overnight_intraday_ratio',
        'label': '隔夜/日内收益比',
        'category': 'return_decomposition',
        'hypothesis': '隔夜收益占比高的股票(信息在非交易时段消化)有更纯净的趋势',
        'logic': '中信建投(2025)隔夜-日内异象: 隔夜收益反映机构信息优势; 比率高→趋势稳定',
        'formula': '(open_p / close_p.shift(1) - 1) / (close_p / open_p - 1).abs().where((close_p / open_p - 1).abs() > 0.001, 1.0)',
        'direction': 'long',
    },
    {
        'id': 'novel_002', 'name': 'overnight_momentum',
        'label': '隔夜动量(5日累计)',
        'category': 'return_decomposition',
        'hypothesis': '连续多日隔夜正收益的股票有持续动量',
        'logic': '隔夜收益反映机构定价权，连续多日隔夜正收益=机构持续买入',
        'formula': '(open_p / close_p.shift(1) - 1).rolling(5).mean()',
        'direction': 'long',
    },
    {
        'id': 'novel_003', 'name': 'intraday_reversal',
        'label': '日内反转(开盘→收盘)',
        'category': 'return_decomposition',
        'hypothesis': '连续多日日内负收益(开盘高走低)的股票短期反弹',
        'logic': '日内收益反映散户情绪; 连续日内下跌=散户恐慌→均值回归',
        'formula': '-(close_p / open_p - 1).rolling(5).mean()',
        'direction': 'long',
    },

    # ============= 成交量结构类 =============
    {
        'id': 'novel_004', 'name': 'volume_stability',
        'label': '成交量稳定性',
        'category': 'volume_structure',
        'hypothesis': '成交量波动率低的股票(筹码稳定)未来收益更高',
        'logic': '成交量稳定→筹码锁定好 → 信息噪音低 → 趋势持续性高',
        'formula': '-(volume_p.rolling(20).std() / volume_p.rolling(20).mean())',
        'direction': 'long',
    },
    {
        'id': 'novel_005', 'name': 'up_down_volume_ratio',
        'label': '涨跌量比',
        'category': 'volume_structure',
        'hypothesis': '上涨日成交占比高的股票(资金偏好买入)有动量',
        'logic': '涨跌量比=资金流向代理: 上涨日放量→资金偏好买入→趋势延续',
        'formula': '(volume_p * (close_p.pct_change() > 0).astype(float)).rolling(20).sum() / '
                   '(volume_p.rolling(20).sum() + 1)',
        'direction': 'long',
    },
    {
        'id': 'novel_006', 'name': 'volume_price_trend_div',
        'label': '量价趋势背离',
        'category': 'volume_structure',
        'hypothesis': '价格趋势向上但成交量趋势向下的股票(量价背离)短期反转',
        'logic': '量价背离经典技术信号: 价格上涨=量缩 = 趋势衰竭',
        'formula': '-(close_p.pct_change(20) * (volume_p.rolling(10).mean() / volume_p.rolling(30).mean() - 1))',
        'direction': 'long',
    },

    # ============= 价格形态类 =============
    {
        'id': 'novel_007', 'name': 'close_position',
        'label': '收盘位置(日内)',
        'category': 'price_pattern',
        'hypothesis': '收盘价在当日高位区的股票有动量(强势收盘)',
        'logic': '收盘位置=日内买卖力量对比: 收在高位→买方主导→次日惯性',
        'formula': '(close_p - low_p) / (high_p - low_p + 0.001)',
        'direction': 'long',
    },
    {
        'id': 'novel_008', 'name': 'close_position_momentum',
        'label': '强势收盘动量',
        'category': 'price_pattern',
        'hypothesis': '连续多日收在当日高位的股票短期动量延续',
        'logic': '连续强势收盘=持续的买方主导→趋势延续信号',
        'formula': '((close_p - low_p) / (high_p - low_p + 0.001)).rolling(5).mean()',
        'direction': 'long',
    },
    {
        'id': 'novel_009', 'name': 'gap_fill_tendency',
        'label': '缺口回补倾向',
        'category': 'price_pattern',
        'hypothesis': '历史上缺口回补率高的股票(缺口不可靠)动量弱',
        'logic': '缺口回补率=缺口信号的质量: 回补率高=缺口信息含量低',
        'formula': '-((open_p - close_p.shift(1)).abs() > 0.02 * close_p.shift(1)).astype(float)'
                   '.rolling(60).mean()',
        'direction': 'long',
    },
    {
        'id': 'novel_010', 'name': 'range_consistency',
        'label': '波幅一致性',
        'category': 'price_pattern',
        'hypothesis': '日内波幅稳定的股票有更持续的趋势(非暴涨暴跌)',
        'logic': '波幅一致性=价格行为有序: 波幅波动大→分歧大→预测力弱',
        'formula': '-(high_p / low_p - 1).rolling(20).std()',
        'direction': 'long',
    },

    # ============= 动量加速度类 =============
    {
        'id': 'novel_011', 'name': 'momentum_acceleration',
        'label': '动量加速度',
        'category': 'momentum_dynamics',
        'hypothesis': '动量在加速的股票(短动量>长动量)有超预期趋势',
        'logic': '动量加速度=二阶动量: 趋势不是匀速而是加速→超预期信号',
        'formula': 'close_p.pct_change(5) - close_p.pct_change(20)',
        'direction': 'long',
    },
    {
        'id': 'novel_012', 'name': 'momentum_jerk',
        'label': '动量加加速度(三阶)',
        'category': 'momentum_dynamics',
        'hypothesis': '动量二阶导数(加速度变化率)极端时预示趋势反转',
        'logic': '三阶动量=jerk: 速度变化过快→不可持续→均值回归',
        'formula': '-(close_p.pct_change(5) - 2 * close_p.pct_change(10) + close_p.pct_change(20)).abs()',
        'direction': 'long',
    },
    {
        'id': 'novel_013', 'name': 'fractal_efficiency',
        'label': '分形效率(路径比)',
        'category': 'momentum_dynamics',
        'hypothesis': '价格路径更直的股票(高方向性)有动量',
        'logic': '分形效率=(终点-起点)/路径总长度; 效率高→噪音低→趋势可靠',
        'formula': 'close_p.pct_change(20).abs() / close_p.pct_change().abs().rolling(20).sum()',
        'direction': 'long',
    },

    # ============= 收益分布类 =============
    {
        'id': 'novel_014', 'name': 'hit_rate_20d',
        'label': '20日胜率',
        'category': 'return_distribution',
        'hypothesis': '过去20日日胜率高的股票(稳定盈利)有动量',
        'logic': '胜率>幅度: 稳定正收益的股票信号质量高于偶尔暴涨的',
        'formula': '(close_p.pct_change() > 0).astype(float).rolling(20).mean()',
        'direction': 'long',
    },
    {
        'id': 'novel_015', 'name': 'return_concentration',
        'label': '收益集中度',
        'category': 'return_distribution',
        'hypothesis': '收益集中在少数几天的股票(不稳定)未来走弱',
        'logic': '收益集中度=风险信号: 少数天贡献大部分收益→信号质量差',
        'formula': '-(close_p.pct_change().rolling(20).apply(lambda x: (x.nlargest(3).sum() / x.abs().sum()) '
                   'if x.abs().sum() > 0 else 0.5, raw=True))',
        'direction': 'long',
    },
    {
        'id': 'novel_016', 'name': 'upside_downside_capture',
        'label': '涨跌捕获比',
        'category': 'return_distribution',
        'hypothesis': '上涨日涨幅/下跌日跌幅比值高的股票有上行偏好',
        'logic': '涨跌捕获比=非对称性: 涨得多跌得少的股票有上行动量',
        'formula': '(close_p.pct_change() * (close_p.pct_change() > 0).astype(float)).rolling(20).mean() / '
                   '(close_p.pct_change().abs() * (close_p.pct_change() < 0).astype(float)).rolling(20).mean().abs()',
        'direction': 'long',
    },

    # ============= 波动率结构类 =============
    {
        'id': 'novel_017', 'name': 'volatility_of_volatility',
        'label': '波动率的波动率',
        'category': 'volatility_structure',
        'hypothesis': '波动率本身波动大的股票(不确定性大)未来收益低',
        'logic': 'Vol-of-Vol = 不确定性中的不确定性 → 价格信号的信噪比极低',
        'formula': '-close_p.pct_change().rolling(5).std().rolling(20).std()',
        'direction': 'long',
    },
    {
        'id': 'novel_018', 'name': 'volatility_regime_change',
        'label': '波动率区制转换',
        'category': 'volatility_structure',
        'hypothesis': '波动率刚突破历史均值的股票短期有超额(新信息进入)',
        'logic': '波动率区制转换=新信息冲击: 突破历史均值→信息含量高→短期超额',
        'formula': 'close_p.pct_change().rolling(5).std() / close_p.pct_change().rolling(60).std()',
        'direction': 'long',
    },

    # ============= 流动性微观结构类 =============
    {
        'id': 'novel_019', 'name': 'dollar_vol_stability',
        'label': '成交额稳定性',
        'category': 'liquidity_micro',
        'hypothesis': '成交额稳定的股票(流动性供给稳定)流动性溢价低、收益高',
        'logic': '成交额稳定→流动性提供者稳定→交易成本低→预期收益高',
        'formula': '-(volume_p * close_p).rolling(20).std() / (volume_p * close_p).rolling(20).mean()',
        'direction': 'long',
    },
    {
        'id': 'novel_020', 'name': 'relative_spread_proxy',
        'label': '相对价差代理',
        'category': 'liquidity_micro',
        'hypothesis': '日内相对价差(high-low/close)小的股票流动性好、预期收益高',
        'logic': '日内价差=买卖价差代理: 价差小→交易成本低→流动性溢价为正',
        'formula': '-(high_p - low_p) / (close_p + 0.001)',
        'direction': 'long',
    },
    {
        'id': 'novel_021', 'name': 'amihud_change',
        'label': '非流动性变化率',
        'category': 'liquidity_micro',
        'hypothesis': '非流动性快速下降的股票(流动性改善)短期有超额',
        'logic': 'Amihud变化率=流动性改善信号: 改善中的股票有增量资金',
        'formula': '-(abs(close_p.pct_change()) / (volume_p * close_p) * 1e8).pct_change(20)',
        'direction': 'long',
    },

    # ============= 行为金融类 =============
    {
        'id': 'novel_022', 'name': 'disposition_effect_proxy',
        'label': '处置效应代理',
        'category': 'behavioral',
        'hypothesis': '接近成本价(52周均价)的股票抛压大(处置效应)→短期回调',
        'logic': '处置效应: 投资者倾向卖出回到成本价的股票; close=avg→抛压→回调',
        'formula': '-(close_p / close_p.rolling(252).mean() - 1).abs()',  # 远离52周均价=反弹
        'direction': 'long',
    },
    {
        'id': 'novel_023', 'name': 'attention_decay',
        'label': '注意力衰减',
        'category': 'behavioral',
        'hypothesis': '极端成交量事件后第N天是反转窗口',
        'logic': '注意力驱动(Barinov 2014): 极端放量=注意力事件→吸引散户→高估后反转',
        'formula': '-(volume_p.rolling(5).max() / volume_p.rolling(20).mean() - 1).shift(3)',  # 3天后反转
        'direction': 'long',
    },
    {
        'id': 'novel_024', 'name': 'anchoring_20d_high',
        'label': '锚定效应(20日高点)',
        'category': 'behavioral',
        'hypothesis': '股票在20日高点附近有阻力(锚定效应)→做空信号，远离高点=反弹',
        'logic': '短锚定(20日): 投资者锚定近期高点; 远离高点=超卖修复',
        'formula': 'close_p / close_p.rolling(20).max()',  # 值越低=离高点越远=反弹潜力
        'direction': 'long',
    },

    # ============= 相关性/联动类 =============
    {
        'id': 'novel_025', 'name': 'market_correlation_change',
        'label': '市场相关性变化',
        'category': 'correlation_linkage',
        'hypothesis': '与市场相关性快速下降的股票(独立行情)有Alpha',
        'logic': 'Beta下降=个股独立行情=信息驱动而非大盘Beta→Alpha信号',
        'formula': '-(close_p.pct_change().rolling(60).corr(close_p.mean(axis=1).pct_change()))',
        'direction': 'long',
    },
    {
        'id': 'novel_026', 'name': 'co_movement_intensity',
        'label': '同涨同跌强度',
        'category': 'correlation_linkage',
        'hypothesis': '与市场同涨同跌强度高的股票(高Beta跟随)Alpha低',
        'logic': '高同动性=被动跟随市场→无独立Alpha→做空(低配)',
        'formula': '-((close_p.pct_change() * close_p.mean(axis=1).pct_change() > 0).astype(float)'
                   '.rolling(20).mean())',
        'direction': 'long',
    },

    # ============= 信息离散度类 =============
    {
        'id': 'novel_027', 'name': 'information_discreteness',
        'label': '信息离散度',
        'category': 'information_flow',
        'hypothesis': '连续同向涨跌幅(非锯齿)的股票信息流更连续→趋势可靠',
        'logic': '信息离散度=连续同向天数: 同向多→信息逐步释放→趋势可信',
        'formula': '((close_p.pct_change() * close_p.pct_change().shift(1)) > 0).astype(float).rolling(10).mean()',
        'direction': 'long',
    },
    {
        'id': 'novel_028', 'name': 'reversal_depth',
        'label': '反转深度',
        'category': 'information_flow',
        'hypothesis': '从近期低点反弹幅度小的股票(仍在底部)有持续修复空间',
        'logic': '反转深度=底部确认程度: 反弹小→修复未完成→持续上行空间',
        'formula': '-(close_p / close_p.rolling(60).min() - 1)',  # 越小=离低点越近=修复空间大
        'direction': 'long',
    },

    # ============= 极端事件类 =============
    {
        'id': 'novel_029', 'name': 'crash_recovery_speed',
        'label': '暴跌修复速度',
        'category': 'extreme_events',
        'hypothesis': '经历暴跌后恢复快的股票(强韧性)有Alpha',
        'logic': '暴跌后修复速度=股票韧性: 修复快→基本面支撑强→选股信号',
        'formula': 'close_p.pct_change(20) - close_p.pct_change(20).shift(20)',  # 后20日vs前20日
        'direction': 'long',
    },
    {
        'id': 'novel_030', 'name': 'tail_co_movement',
        'label': '尾部联动',
        'category': 'extreme_events',
        'hypothesis': '市场大跌日跟跌幅度小的股票(低尾部Beta)防御性高→Alpha',
        'logic': '尾部Beta=极端风险暴露: 低尾部Beta=熊市防御→长期Alpha',
        'formula': '-close_p.pct_change().rolling(60).apply('
                   'lambda x: np.mean(x[x < np.percentile(x, 20)]) if len(x[x < np.percentile(x, 20)]) > 0 else np.mean(x), raw=True)',
        'direction': 'long',
    },

    # ============= 新异象/事件驱动类 =============
    {
        'id': 'novel_031', 'name': 'earnings_momentum_proxy',
        'label': '盈利动量代理(量价)',
        'category': 'event_driven',
        'hypothesis': '财报窗口期前后量价异常的股票可能有信息泄露',
        'logic': '财报季量价异常=信息提前反映: 异常量价→有未公开信息→趋势',
        'formula': '(volume_p / volume_p.rolling(60).mean() - 1) * close_p.pct_change(5)',
        'direction': 'long',
    },
    {
        'id': 'novel_032', 'name': 'limit_up_proximity',
        'label': '涨停板邻近效应',
        'category': 'event_driven',
        'hypothesis': '距离涨停板(9.5%+)越近的股票有动量(涨停磁吸)',
        'logic': 'A股涨停磁吸效应: 接近涨停→资金追逐→涨停概率上升→动量',
        'formula': 'close_p.pct_change()',  # 简化: A股涨跌幅限制附近
        'direction': 'long',
    },

    # ============= 多周期共振类 =============
    {
        'id': 'novel_033', 'name': 'multi_timescale_alignment',
        'label': '多周期共振',
        'category': 'multi_period',
        'hypothesis': '多周期(5/10/20/60日)动量方向一致的股票趋势更可靠',
        'logic': '多周期共振=多空力量在多个时间尺度上一致→最强趋势信号',
        'formula': '(np.sign(close_p.pct_change(5)) + np.sign(close_p.pct_change(10)) + '
                   'np.sign(close_p.pct_change(20)) + np.sign(close_p.pct_change(60))) / 4',
        'direction': 'long',
    },
    {
        'id': 'novel_034', 'name': 'timescale_divergence',
        'label': '多周期背离',
        'category': 'multi_period',
        'hypothesis': '短期涨但长期跌的股票(背离)可能短期补跌→反转',
        'logic': '多周期背离=反向信号: 短多长空→多空分歧→短期方向不确定→反转',
        'formula': '-(close_p.pct_change(5) * close_p.pct_change(60) < 0).astype(float) '
                   '* close_p.pct_change(5).abs()',
        'direction': 'long',
    },

    # ============= 相对价值类 =============
    {
        'id': 'novel_035', 'name': 'relative_value_zscore',
        'label': '相对价值Z-score',
        'category': 'relative_value',
        'hypothesis': '价格相对自身历史位置低的股票(便宜)有均值回归',
        'logic': '价格Z-score=(C-μ)/σ: Z<-2→极度便宜→均值回归',
        'formula': '-(close_p - close_p.rolling(120).mean()) / close_p.rolling(120).std()',
        'direction': 'long',
    },
    {
        'id': 'novel_036', 'name': 'drawdown_depth',
        'label': '回撤深度',
        'category': 'relative_value',
        'hypothesis': '从52周高点回撤深的股票有反弹(超卖修复)',
        'logic': '回撤深度=超卖程度: 回撤>30%→恐慌性抛售→均值回归',
        'formula': 'close_p / close_p.rolling(252).max() - 1',  # 负值大=回撤深→反弹
        'direction': 'long',
    },

    # ============= 趋势质量类 =============
    {
        'id': 'novel_037', 'name': 'trend_smoothness',
        'label': '趋势平滑度',
        'category': 'trend_quality',
        'hypothesis': '趋势平滑(R²高)的股票比趋势锯齿的股票动量更可靠',
        'logic': '趋势平滑度=回归R²: 高R²→线性趋势→可信; 低R²→锯齿→噪音',
        'formula': 'close_p.rolling(20).apply('
                   'lambda x: np.corrcoef(range(len(x)), x)[0,1]**2 if len(x) > 5 else 0, raw=True)',
        'direction': 'long',
    },
    {
        'id': 'novel_038', 'name': 'trend_persistence_score',
        'label': '趋势持续性评分',
        'category': 'trend_quality',
        'hypothesis': '多维度评估趋势持续性: 效率×胜率×方向一致性',
        'logic': '综合趋势质量评分=分形效率×正收益概率×多周期方向一致',
        'formula': '(close_p.pct_change(20).abs() / close_p.pct_change().abs().rolling(20).sum()) * '
                   '(close_p.pct_change() > 0).astype(float).rolling(20).mean()',
        'direction': 'long',
    },

    # ============= 最新研报驱动 (2026-06-15) =============
    {
        'id': 'novel_039', 'name': 'style_rotation_signal',
        'label': '风格轮动信号(动量vs价值差)',
        'category': 'research_driven',
        'hypothesis': '短期动量(5日)与中期动量(60日)方向相反时, 市场正在风格切换',
        'logic': '6/15研报: 红利vs科技拔河; 风格Z-score极端预示均值回归; '
                '5日vs60日动量方向差=风格轮动信号',
        'formula': '-(close_p.pct_change(5) - close_p.pct_change(60)).abs()',  # 差大→切换中→混乱
        'direction': 'long',
    },
    {
        'id': 'novel_040', 'name': 'sector_relative_rotation',
        'label': '行业相对轮动强度',
        'category': 'research_driven',
        'hypothesis': '个股相对截面均值的超额动量反映行业内部轮动',
        'logic': '6月流动性因子占优: 个股超额动量=行业轮动alpha载体',
        'formula': 'close_p.pct_change(10).sub(close_p.mean(axis=1).pct_change(10), axis=0)',  # 超额收益
        'direction': 'long',
    },
    {
        'id': 'novel_041', 'name': 'dispersion_anomaly',
        'label': '横截面离散度异象',
        'category': 'research_driven',
        'hypothesis': '高横截面离散度时期, 基本面质量因子表现更好(flight to quality)',
        'logic': '摩根大通(6/7): 横截面波动率98%分位; '
                '极端离散→资金流向质量→质量因子占优',
        'formula': 'pd.DataFrame({c: close_p.pct_change().rolling(20).std().mean(axis=1) for c in close_p.columns}, index=close_p.index)',  # 市场离散度的个股暴露，保持DataFrame格式
        'direction': 'long',
    },

    # ============= 补充: 非对称性类 =============
    {
        'id': 'novel_042', 'name': 'gain_loss_asymmetry',
        'label': '涨跌不对称性',
        'category': 'asymmetry',
        'hypothesis': '上涨天数多于下跌天数但总收益为负的股票(假强势)未来走弱',
        'logic': '不对称性陷阱: 频繁小涨+偶尔大跌→净负收益→无法持续',
        'formula': '(close_p.pct_change() > 0).astype(float).rolling(20).mean() * '
                   'close_p.pct_change(20)',  # 胜率×总收益
        'direction': 'long',
    },
    {
        'id': 'novel_043', 'name': 'resistance_distance',
        'label': '阻力位距离',
        'category': 'asymmetry',
        'hypothesis': '突破20日均线的股票(阻力变支撑)有动量',
        'logic': '均线突破信号: close>MA20→多头排列→趋势上行',
        'formula': 'close_p / close_p.rolling(20).mean() - 1',
        'direction': 'long',
    },

    # ============= 吴先兴五维成长因子框架 (2026-06-14) =============
    {
        'id': 'novel_044', 'name': 'earnings_quality_proxy',
        'label': '盈利质量代理(OHLCV)',
        'category': 'growth_quality',
        'hypothesis': '盈利质量高的股票(收益稳定性好)长期Alpha更高',
        'logic': '吴先兴五维之一: 盈利质量。量价代理=收益波动率低+正收益占比高',
        'formula': '-(close_p.pct_change().rolling(20).std() / (close_p.pct_change().rolling(20).mean().abs() + 0.001)) * (close_p.pct_change() > 0).astype(float).rolling(20).mean()',
        'direction': 'long',
    },
    {
        'id': 'novel_045', 'name': 'cashflow_matching_proxy',
        'label': '现金流匹配度代理',
        'category': 'growth_quality',
        'hypothesis': '量价趋势与成交额趋势一致的股票(量价匹配)趋势更可靠',
        'logic': '吴先兴五维之二: 现金流匹配度。量价代理=价格趋势*成交额趋势同向性',
        'formula': '(close_p.pct_change(20) * (volume_p * close_p).pct_change(20) > 0).astype(float).rolling(20).mean()',
        'direction': 'long',
    },
    {
        'id': 'novel_046', 'name': 'capital_efficiency_proxy',
        'label': '资本投入效率代理',
        'category': 'growth_quality',
        'hypothesis': '单位成交额产生的价格变动效率高的股票资金利用效率高',
        'logic': '吴先兴五维之三: 资本投入效率。量价代理=收益/成交额比(Amihud的反面)',
        'formula': 'close_p.pct_change(20).abs() / (volume_p * close_p).rolling(20).mean().replace(0, np.nan) * 1e8',
        'direction': 'long',
    },
    {
        'id': 'novel_047', 'name': 'operational_efficiency_proxy',
        'label': '营运效率代理',
        'category': 'growth_quality',
        'hypothesis': '价格调整效率高的股票(快速消化信息)长期收益更高',
        'logic': '吴先兴五维之四: 营运效率。量价代理=价格自相关低(信息消化快)',
        'formula': '-close_p.pct_change().rolling(20).apply(lambda x: x.autocorr(lag=1) if len(x) > 5 else 0, raw=True)',
        'direction': 'long',
    },
    {
        'id': 'novel_048', 'name': 'bargaining_power_proxy',
        'label': '议价能力代理',
        'category': 'growth_quality',
        'hypothesis': '在行业下跌时跌幅更小的股票(定价权强)质量更高',
        'logic': '吴先兴五维之五: 议价能力。量价代理=下行捕获率(跌时跌得少)',
        'formula': '-close_p.pct_change().clip(upper=0).rolling(20).mean()',
        'direction': 'long',
    },

    # ============= 2026-06-21 扩充: 全新未覆盖概念 =============
    {
        'id': 'novel_049', 'name': 'breakout_strength',
        'label': '突破强度',
        'category': 'price_pattern',
        'hypothesis': '价格突破20日高点同时放量的股票(真实突破)有持续动量',
        'logic': '技术分析经典: 突破+放量=真实突破; 突破+缩量=假突破。组合信号过滤假信号',
        'formula': '(close_p / close_p.rolling(20).max() - 1) * (volume_p / volume_p.rolling(20).mean())',
        'direction': 'long',
    },
    {
        'id': 'novel_050', 'name': 'downside_rarity',
        'label': '下行稀缺性',
        'category': 'return_distribution',
        'hypothesis': '过去20日下跌天数极少的股票(极强趋势)动量延续',
        'logic': '胜率极端化: 20日中仅1-3天下跌=极强买入力量→趋势延续而非反转(与反转因子相反)',
        'formula': '-(close_p.pct_change() < 0).astype(float).rolling(20).sum()',
        'direction': 'long',
    },
    {
        'id': 'novel_051', 'name': 'sma_divergence',
        'label': '均线发散度',
        'category': 'trend_quality',
        'hypothesis': '短期均线在长期均线之上的股票(多头排列)有动量，发散度越大=趋势越强',
        'logic': '均线系统: SMA5/SMA20 > 1=多头排列; 发散度量化均线距离，与价格位置(close/MA)互补',
        'formula': 'close_p.rolling(5).mean() / close_p.rolling(20).mean() - 1',
        'direction': 'long',
    },
    {
        'id': 'novel_052', 'name': 'volume_climax_reversal',
        'label': '天量反转',
        'category': 'volume_structure',
        'hypothesis': '天量(>3倍均量)次日缩量的股票短期反转(天量耗尽买方力量)',
        'logic': '成交量极端事件后: 天量=买方情绪顶点→次日买方枯竭→反转。与attention_decay互补(天量次日vs高峰后3日)',
        'formula': '-(volume_p.shift(1) / volume_p.rolling(20).mean().shift(1) - 1).clip(lower=0)',
        'direction': 'long',
    },
    {
        'id': 'novel_053', 'name': 'intraday_reversal_depth',
        'label': '日内反转深度',
        'category': 'price_pattern',
        'hypothesis': '低开高走的股票(日内反转幅度大)短期有动能，反映买方盘中主导',
        'logic': '日内反转=盘中力量逆转: open<prev_close但close>open→空头开盘后被多头压制→次日惯性',
        'formula': '(close_p / open_p - 1) * (open_p < close_p.shift(1)).astype(float)',
        'direction': 'long',
    },

    # ================================================================
    # v3 扩充因子库 (2026-06-22) — 三路并行研究产出
    # 来源: 清华PBCSF 208异象 + 广发金工Level-2 + 华泰金工微观结构 + 研报驱动
    # ================================================================

    # --- 成交量自相关 & 微观结构 (华泰金工2026 分钟级因子日频代理) ---
    {
        'id': 'novel_054', 'name': 'volume_autocorr_5d',
        'label': '成交量自相关(5日)',
        'category': 'market_microstructure',
        'hypothesis': '成交量自相关结构被打破(自相关降低)的股票有主力资金介入→短期动量',
        'logic': '华泰金工(2026): 成交笔数自相关降低=主力主动介入打破原有交易节奏→资金驱动上行; 日频用成交量自相关代理',
        'formula': '-volume_p.pct_change().rolling(20).apply(lambda x: x.autocorr(lag=5) if len(x) > 10 else 0, raw=True)',
        'direction': 'long',
    },
    {
        'id': 'novel_055', 'name': 'volume_price_distance',
        'label': '量价欧氏距离代理',
        'category': 'market_microstructure',
        'hypothesis': '成交量与成交额关系出现极端背离(欧氏距离大)的股票有流动性结构断裂→反转信号',
        'logic': '华泰金工(2026): 成交笔数与成交额常态高度正相关; 欧氏距离大=微观结构极端背离→反转; 日频用volume与amount标准化距离代理',
        'formula': '-(((volume_p - volume_p.rolling(20).mean()) / volume_p.rolling(20).std()).fillna(0)**2 + '
                   '((volume_p * close_p - (volume_p * close_p).rolling(20).mean()) / (volume_p * close_p).rolling(20).std()).fillna(0)**2)',
        'direction': 'long',
    },
    {
        'id': 'novel_056', 'name': 'trade_intensity_proxy',
        'label': '单笔交易强度代理',
        'category': 'market_microstructure',
        'hypothesis': '单位换手率的价格变动大(单笔冲击大)的股票有信息不对称→短期Alpha',
        'logic': '华泰金工(2026)单笔成交回归截距的日频代理: 剥离系统性量价波动后的个股内生交易信号; 高单笔冲击=知情交易',
        'formula': '(high_p - low_p) / (volume_p / volume_p.rolling(60).mean() + 0.001)',
        'direction': 'long',
    },

    # --- 隔夜与日内结构 (清华PBCSF Day-Night anomaly + 广发Level-2) ---
    {
        'id': 'novel_057', 'name': 'overnight_volume_concentration',
        'label': '隔夜成交量集中度',
        'category': 'market_microstructure',
        'hypothesis': '开盘成交量占全天比高的股票(信息在开盘集中释放)有短期动量',
        'logic': '广发金工Level-2 ret_overnight因子(月频RankIC 3.11%): 隔夜信息反映机构定价; 开盘量集中=信息冲击大→趋势延续',
        'formula': 'volume_p.rolling(5).apply(lambda x: x[0] / x.sum() if x.sum() > 0 else 0.2, raw=True)',
        'direction': 'long',
    },
    {
        'id': 'novel_058', 'name': 'day_night_return_gap',
        'label': '日夜收益差',
        'category': 'market_microstructure',
        'hypothesis': '隔夜收益远大于日内收益的股票(机构主导定价)有持续Alpha',
        'logic': 'Qiu&Huang(2025) Day-Night anomaly: 中国A股异象收益50%集中在公司新闻日/6倍在财报日; 日夜差=信息消化渠道差异',
        'formula': '(open_p / close_p.shift(1) - 1) - (close_p / open_p - 1)',
        'direction': 'long',
    },

    # --- 换手率异象 (清华PBCSF 208异象库) ---
    {
        'id': 'novel_059', 'name': 'abnormal_turnover_5d',
        'label': '异常换手率(5日)',
        'category': 'liquidity_micro',
        'hypothesis': '换手率异常升高(相对自身历史)的股票短期有动量(关注度效应)',
        'logic': '清华PBCSF异常换手率异象: 换手率偏离历史均值=投资者关注事件→价格发现→短期动量',
        'formula': '(volume_p / volume_p.rolling(60).mean() - 1) * (close_p.pct_change() > 0).astype(float).rolling(5).mean()',
        'direction': 'long',
    },
    {
        'id': 'novel_060', 'name': 'turnover_cv_20d',
        'label': '换手率变异系数',
        'category': 'liquidity_micro',
        'hypothesis': '换手率波动大的股票(流动性不稳定)未来收益低(风险溢价)',
        'logic': '清华PBCSF换手率波动异象: 高换手波动=流动性风险高→补偿性收益预期，但A股散户主导→高波动=噪音→低收益',
        'formula': '-(volume_p.rolling(20).std() / volume_p.rolling(20).mean())',
        'direction': 'long',
    },
    {
        'id': 'novel_061', 'name': 'zero_volume_days',
        'label': '零成交量天数',
        'category': 'liquidity_micro',
        'hypothesis': '近期零成交量天数多的股票(流动性枯竭)未来收益低',
        'logic': '清华PBCSF零成交量异象(Liu 2006): 零成交量反映极端非流动性→流动性折价→低收益; A股涨停/停牌可能干扰',
        'formula': '-(volume_p == 0).astype(float).rolling(20).sum()',
        'direction': 'long',
    },

    # --- 波动率微观结构 ---
    {
        'id': 'novel_062', 'name': 'realized_semivariance_ratio',
        'label': '已实现半方差比',
        'category': 'volatility_structure',
        'hypothesis': '下行波动/上行波动比高的股票(非对称风险)未来收益低',
        'logic': 'Barndorff-Nielsen(2010)已实现半方差: 上下行波动非对称=投资者对负面信息更敏感→下行波动高=风险溢价',
        'formula': '-(close_p.pct_change().clip(upper=0).rolling(20).std() / '
                   '(close_p.pct_change().clip(lower=0).rolling(20).std() + 0.001))',
        'direction': 'long',
    },
    {
        'id': 'novel_063', 'name': 'jump_intensity',
        'label': '跳跃强度',
        'category': 'volatility_structure',
        'hypothesis': '近期跳空次数多的股票(信息冲击频繁)短期有反转(过度反应)',
        'logic': '跳跃扩散模型: 跳空=意外信息冲击→短期过度反应→反转; 3σ阈值识别跳跃',
        'formula': '-((close_p.pct_change().abs() > 3 * close_p.pct_change().rolling(60).std()).astype(float).rolling(20).sum())',
        'direction': 'long',
    },

    # --- 尾部依赖 & 风险 ---
    {
        'id': 'novel_064', 'name': 'tail_dependence_market',
        'label': '尾部市场依赖',
        'category': 'tail_risk',
        'hypothesis': '市场大跌时跟跌幅度大的股票(高尾部Beta)长期Alpha低',
        'logic': 'Van Oordt&Zhou(2016)尾部Beta: 市场极端下跌时个股表现反映系统性尾部风险暴露→高尾部Beta=低长期收益',
        'formula': '-close_p.pct_change().rolling(60).apply('
                   'lambda x: np.mean(x[x < np.percentile(x, 10)]) if len(x[x < np.percentile(x, 10)]) > 0 else 0, raw=True)',
        'direction': 'long',
    },
    {
        'id': 'novel_065', 'name': 'downside_correlation',
        'label': '下行相关性',
        'category': 'tail_risk',
        'hypothesis': '市场下跌时相关性高的股票(系统性熊市暴露)防御性差→低收益',
        'logic': 'Ang et al. downside beta: 仅在下行市场计算的相关性能更纯净地捕捉系统性尾部风险',
        'formula': '-close_p.pct_change().rolling(60).apply('
                   'lambda x: np.corrcoef(x, x.mean())[0,1] if np.std(x) > 0 else 0, raw=True)',
        'direction': 'long',
    },

    # --- 价格效率 & 信息传播 ---
    {
        'id': 'novel_066', 'name': 'price_efficiency_ratio',
        'label': '价格效率比',
        'category': 'information_flow',
        'hypothesis': '价格调整快的股票(信息效率高)长期收益高(低信息不对称)',
        'logic': '市场微观结构文献: 价格效率=信息融入速度快→低信息不对称→低逆向选择成本→高收益',
        'formula': '1 - close_p.pct_change().rolling(20).apply('
                   'lambda x: abs(x.autocorr(lag=1)) if len(x) > 5 else 0.5, raw=True)',
        'direction': 'long',
    },
    {
        'id': 'novel_067', 'name': 'news_absorption_speed',
        'label': '信息吸收速度',
        'category': 'information_flow',
        'hypothesis': '大幅波动后快速回归常态的股票(信息吸收快)有选股Alpha',
        'logic': '信息冲击后的恢复速度=市场处理信息效率; 快速吸收=基本面投资者多→信号质量高',
        'formula': '-(close_p.pct_change().abs().rolling(5).mean() / close_p.pct_change().abs().rolling(20).mean() - 1)',
        'direction': 'long',
    },

    # --- 供需结构 ---
    {
        'id': 'novel_068', 'name': 'volume_price_pressure',
        'label': '量价压力比',
        'category': 'supply_demand',
        'hypothesis': '上涨需要成交量支撑，量价压力比异常的股票(量不足但价涨)短期回调',
        'logic': '供需微观结构: 价格上涨需买方量能支撑; 价涨量缩=买方枯竭→短期反转',
        'formula': '-(close_p.pct_change(5) / (volume_p.rolling(5).mean() / volume_p.rolling(20).mean() + 0.001))',
        'direction': 'long',
    },
    {
        'id': 'novel_069', 'name': 'cumulative_volume_trend',
        'label': '累计量趋势',
        'category': 'supply_demand',
        'hypothesis': '上涨日累计量/下跌日累计量比高的股票(买方力量强)持续跑赢',
        'logic': 'OBV(On-Balance Volume)变体: 累计量趋势=买方vs卖方力量累积差异→趋势可靠性指标',
        'formula': '((volume_p * (close_p.pct_change() > 0).astype(float)).rolling(20).sum() / '
                   '((volume_p * (close_p.pct_change() < 0).astype(float)).abs().rolling(20).sum() + 1))',
        'direction': 'long',
    },

    # --- 情绪 & 羊群 ---
    {
        'id': 'novel_070', 'name': 'herding_intensity',
        'label': '羊群效应强度',
        'category': 'sentiment_crowding',
        'hypothesis': '个股收益与市场收益绝对偏差小(接近市场)的股票羊群效应强→反转',
        'logic': 'Christie&Huang(1995)羊群效应: 市场极端波动时个股收益趋同→羊群; 低偏差=跟风→后续反转',
        'formula': '-(1 - (close_p.pct_change().sub(close_p.mean(axis=1).pct_change(), axis=0)).abs().rolling(20).mean())',
        'direction': 'long',
    },
    {
        'id': 'novel_071', 'name': 'dispersion_beta',
        'label': '离散度Beta',
        'category': 'sentiment_crowding',
        'hypothesis': '个股对横截面离散度敏感的股票(跟风型)Alpha低',
        'logic': '广发金工(2026): 横截面离散度极端时资金流向质量; 高离散Beta=跟风型→选股能力弱',
        'formula': '-(close_p.pct_change().rolling(60).corr(close_p.pct_change().std(axis=1)))',
        'direction': 'long',
    },

    # --- 动量增强 & 衰减 ---
    {
        'id': 'novel_072', 'name': 'momentum_decay_rate',
        'label': '动量衰减速率',
        'category': 'momentum_dynamics',
        'hypothesis': '动量衰减快的股票(短周期信号主导)不适合中长期持有→短期反转信号',
        'logic': '学术文献: 动量衰减半衰期=信息持久性指标; 衰减快=噪音主导→缺乏持续Alpha',
        'formula': '-(close_p.pct_change(5).abs() / (close_p.pct_change(20).abs() + 0.001) - 1)',
        'direction': 'long',
    },
    {
        'id': 'novel_073', 'name': 'residual_momentum',
        'label': '残差动量',
        'category': 'momentum_dynamics',
        'hypothesis': '剔除市场和行业收益后的残差动量比原始动量更纯净(不受Beta干扰)',
        'logic': 'Blitz et al.(2011)残差动量: 残差收益反映个股特有Alpha→比总收益动量更稳定的选股信号',
        'formula': '(close_p / close_p.shift(20) - 1).sub(close_p.mean(axis=1) / close_p.mean(axis=1).shift(20) - 1, axis=0)',
        'direction': 'long',
    },

    # --- 隔夜反转 & 开盘效应 ---
    {
        'id': 'novel_074', 'name': 'opening_gap_momentum',
        'label': '开盘跳空动量',
        'category': 'asymmetry',
        'hypothesis': '连续跳空高开的股票(强势开盘)短期动量延续',
        'logic': '广发Level-2 ret_open2A系列因子: 开盘跳空=隔夜信息在开盘集中反映→延续信号; 连续跳空=持续信息流入',
        'formula': '(open_p / close_p.shift(1) - 1).rolling(3).mean()',
        'direction': 'long',
    },
    {
        'id': 'novel_075', 'name': 'closing_auction_strength',
        'label': '尾盘强度代理',
        'category': 'asymmetry',
        'hypothesis': '尾盘价格高于当日均价(VWAP)的股票有主力尾盘买入→次日动量',
        'logic': '广发金工Level-2尾盘因子: 尾盘买入=机构在收盘价附近建仓→避免日内波动→次日延续',
        'formula': 'close_p / ((high_p + low_p + close_p) / 3 + 0.001) - 1',
        'direction': 'long',
    },

    # --- 极端事件 & 恢复 ---
    {
        'id': 'novel_076', 'name': 'gap_survival_rate',
        'label': '缺口存活率',
        'category': 'extreme_events',
        'hypothesis': '跳空后不回补的股票(缺口存活)趋势强→动量延续',
        'logic': '技术分析经典: 突破性缺口不回补=趋势确认; 回补率低=买方力量持续→趋势延续信号',
        'formula': '((open_p.shift(5) / close_p.shift(6) - 1).abs() > 0.03).astype(float) * '
                   '(close_p > close_p.shift(6)).astype(float)',
        'direction': 'long',
    },
    {
        'id': 'novel_077', 'name': 'max_drawdown_duration',
        'label': '最大回撤持续时间',
        'category': 'extreme_events',
        'hypothesis': '已从长期回撤中修复的股票(韧性信号)有选股Alpha',
        'logic': '回撤持续时间=市场压力测试: 从长期回撤恢复=基本面强→Alpha信号',
        'formula': '(close_p / close_p.rolling(120).max()).rolling(20).mean()',
        'direction': 'long',
    },

    # --- 收益分解增强 ---
    {
        'id': 'novel_078', 'name': 'return_asymmetry_ratio',
        'label': '收益非对称比',
        'category': 'return_distribution',
        'hypothesis': '正收益日平均收益/负收益日平均收益(绝对值)比高的股票上行趋势强',
        'logic': '非对称收益=买方力量不对称: 涨时涨得多+跌时跌得少→机构主导→趋势延续',
        'formula': '(close_p.pct_change().clip(lower=0).rolling(20).mean() / '
                   '(close_p.pct_change().clip(upper=0).abs().rolling(20).mean() + 0.001))',
        'direction': 'long',
    },

    # ================================================================
    # v4扩充 (06-22): +35因子
    # 方向1: 基本面质量OHLCV代理 (财务季7-8月) + 吴先兴五维框架V2
    # 方向2: 最新学术论文(arXiv/SSRN 2026) + 券商研报(广发Level-2/国联民生/长江金工)
    # 方向3: 高级信号处理 + 行为金融扩展 + 游资微观结构
    # 新增类别: fundamental_quality_proxy/earnings_behavior/price_efficiency/
    #           alpha_dynamics/cf_microstructure/behavioral_v2/regime_adaptive
    # ================================================================

    # --- 基本面质量代理 (OHLCV财务季版) ---
    {
        'id': 'novel_079', 'name': 'capital_efficiency_proxy_v2',
        'label': '资本效率代理V2(含波动调整)',
        'category': 'fundamental_quality_proxy',
        'hypothesis': '单位波动产生更高收益的股票(隐含高ROIC)持续跑赢',
        'logic': '吴先兴五维框架: 资本效率=ROIC/WACC; 价格效率(收益/波动率)是有效代理; V2加入60日窗口提升稳定性',
        'formula': 'close_p.pct_change(60) / (close_p.pct_change().rolling(60).std() + 0.001)',
        'direction': 'long',
    },
    {
        'id': 'novel_080', 'name': 'earnings_consistency_proxy',
        'label': '盈利一致性代理',
        'category': 'fundamental_quality_proxy',
        'hypothesis': '正收益窗口占比高的股票(隐含盈利稳定)未来Alpha更高',
        'logic': 'Hsu et al.(2018)盈利能力因子: 持续正收益=隐藏的盈利增长; 60日正收益占比->基本面质量信号',
        'formula': 'close_p.pct_change(5).rolling(60).apply(lambda x: (x > 0).mean(), raw=True)',
        'direction': 'long',
    },
    {
        'id': 'novel_081', 'name': 'accrual_quality_proxy',
        'label': '应计质量代理',
        'category': 'fundamental_quality_proxy',
        'hypothesis': '价格趋势稳定(低应计噪音)的股票更可靠,未来收益更持续',
        'logic': 'Sloan(1996)应计异象: 高应计=盈余管理->未来反转; 价格趋势平滑=低应计噪音->财务质量好',
        'formula': '-(close_p.pct_change().rolling(20).std() / (close_p.pct_change().abs().rolling(20).mean() + 0.001))',
        'direction': 'long',
    },
    {
        'id': 'novel_082', 'name': 'growth_quality_composite',
        'label': '增长质量综合',
        'category': 'fundamental_quality_proxy',
        'hypothesis': '多周期收益均为正且稳定的股票(隐含高质量增长)持续跑赢',
        'logic': '吴先兴五维: 增长质量=多维增长(g/roe/fcf); OHLCV代理: 5/10/20/60日正收益一致性',
        'formula': '(close_p.pct_change(5) > 0).astype(float) + (close_p.pct_change(10) > 0).astype(float) + '
                   '(close_p.pct_change(20) > 0).astype(float) + (close_p.pct_change(60) > 0).astype(float)',
        'direction': 'long',
    },
    {
        'id': 'novel_083', 'name': 'leverage_proxy_v2',
        'label': '杠杆风险代理V2',
        'category': 'fundamental_quality_proxy',
        'hypothesis': '下跌波动/上涨波动比低的股票(隐含低杠杆)下行风险小',
        'logic': '财务杠杆->放大下行波动; 从价格行为逆向推断: 下跌波动小=低杠杆->安全边际高',
        'formula': '-(close_p.pct_change().clip(upper=0).rolling(20).std() / '
                   '(close_p.pct_change().clip(lower=0).rolling(20).std() + 0.001))',
        'direction': 'long',
    },

    # --- 财报季行为信号 ---
    {
        'id': 'novel_084', 'name': 'earnings_anticipation_proxy',
        'label': '业绩预期代理',
        'category': 'earnings_behavior',
        'hypothesis': '财报前成交量异常萎缩(信息等待)后价格有方向性突破',
        'logic': '华泰金工(2025)财报效应: 信息不对称在财报前达峰值->量缩后爆发; 量缩+价稳=高质量等待',
        'formula': '(2 * (close_p.pct_change(20) > 0).astype(int) - 1) * '
                   '(-volume_p.rolling(5).mean() / volume_p.rolling(20).mean())',
        'direction': 'long',
    },
    {
        'id': 'novel_085', 'name': 'post_earnings_drift_proxy',
        'label': '盈余公告后漂移代理',
        'category': 'earnings_behavior',
        'hypothesis': '单日大涨幅后持续多日正收益(PEAD效应)的股票有动量',
        'logic': 'Ball&Brown(1968)PEAD: 盈余公告后漂移; 代理=检测单日大涨(模拟盈余超预期)+后续延续',
        'formula': '(close_p.pct_change() > 0.05).astype(float).rolling(5).sum() * '
                   'close_p.pct_change(5)',
        'direction': 'long',
    },

    # --- 广发Level-2日频代理 (广发金工2025量化精选) ---
    {
        'id': 'novel_086', 'name': 'ret_open_2d_proxy',
        'label': '开盘动量2日代理',
        'category': 'market_microstructure',
        'hypothesis': '连续2日开盘跳空高开的股票(持续信息流入)短期动量强',
        'logic': '广发Level-2 ret_open2A系列: 开盘跳空=隔夜信息集中反映; 2日连续跳空=信息持续性->强趋势',
        'formula': '(open_p / close_p.shift(1) - 1).rolling(2).mean()',
        'direction': 'long',
    },
    {
        'id': 'novel_087', 'name': 'vwap_deviation_20d',
        'label': 'VWAP偏离20日',
        'category': 'market_microstructure',
        'hypothesis': '收盘价持续高于VWAP代理(均价)的股票有机构持续买入',
        'logic': '广发金工Level-2 VWAP因子: 机构交易围绕VWAP执行->收盘持续高于VWAP=机构买入痕迹->Alpha',
        'formula': 'close_p.rolling(20).apply(lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0, raw=False)',
        'direction': 'long',
    },
    {
        'id': 'novel_088', 'name': 'intraday_reversal_depth_v2',
        'label': '日内反转深度V2',
        'category': 'market_microstructure',
        'hypothesis': '高开低走幅度大(散户主导)的股票短期反转修复',
        'logic': '广发高频因子: 日内价格路径反映投资者结构; 高开-低走深度=散户情绪释放->次日修复',
        'formula': '-(open_p - low_p) / (open_p + 0.001)',
        'direction': 'long',
    },
    {
        'id': 'novel_089', 'name': 'closing_momentum_3d',
        'label': '尾盘动能3日',
        'category': 'market_microstructure',
        'hypothesis': '连续3日尾盘(收盘-均线)走强的股票有机构建仓',
        'logic': '广发尾盘因子: 收盘价-日内均价反映尾盘买卖力量; 连续3日走强=机构在收盘价附近吸筹',
        'formula': '(close_p / ((high_p + low_p) / 2 + 0.001) - 1).rolling(3).mean()',
        'direction': 'long',
    },

    # --- 国联民生/长江金工 预测信号代理 ---
    {
        'id': 'novel_090', 'name': 'rating_momentum_proxy',
        'label': '评级动量代理',
        'category': 'research_driven',
        'hypothesis': '价格趋势+成交量收敛的股票(隐含评级上调预期)Alpha持续',
        'logic': '国联民生(2026): 评级上下调差因子超额34.58%; OHLCV代理=趋势强度*量收敛(隐含信息质量)',
        'formula': 'close_p.pct_change(20) * (1 - volume_p.rolling(20).std() / (volume_p.rolling(20).mean() + 0.001))',
        'direction': 'long',
    },
    {
        'id': 'novel_091', 'name': 'roa_trend_proxy',
        'label': 'ROA趋势代理',
        'category': 'research_driven',
        'hypothesis': '价格加速度(二阶导数)正的股票隐含ROA改善',
        'logic': '国联民生(2026): 单季度ROA同比差值因子超额33.35%; 价格加速度->隐含盈利改善->选股信号',
        'formula': 'close_p.pct_change(10) - close_p.pct_change(20)',
        'direction': 'long',
    },
    {
        'id': 'novel_092', 'name': 'analyst_revision_proxy',
        'label': '分析师修正代理',
        'category': 'research_driven',
        'hypothesis': '短期趋势逆转+成交量配合的股票(隐含预期修正)有Alpha',
        'logic': '长江金工: 分析师一致预期修正=最强Alpha来源之一; OHLCV代理=趋势转折点+量确认',
        'formula': '(close_p.pct_change(10) - close_p.pct_change(30)) * '
                   '(volume_p.rolling(5).mean() / volume_p.rolling(20).mean())',
        'direction': 'long',
    },

    # --- 民生金工: 现金流质量代理 ---
    {
        'id': 'novel_093', 'name': 'operating_cf_proxy',
        'label': '经营现金流代理',
        'category': 'fundamental_quality_proxy',
        'hypothesis': '上涨日量比/下跌日量比持续>1的股票(隐含强劲经营现金流)有质量Alpha',
        'logic': '民生金工(2024): CFOR分解->经营现金流是核心质量指标; 代理: 上涨日放量/下跌日缩量=买方有力',
        'formula': '((volume_p * (close_p.pct_change() > 0).astype(float)).rolling(20).mean() / '
                   '((volume_p * (close_p.pct_change() < 0).astype(float)).rolling(20).mean() + 1))',
        'direction': 'long',
    },
    {
        'id': 'novel_094', 'name': 'free_cf_margin_proxy',
        'label': '自由现金流利润率代理',
        'category': 'fundamental_quality_proxy',
        'hypothesis': '短期收益/长期收益比高的股票(隐含高FCF利润率)Alpha可持续',
        'logic': 'FCF利润率=自由现金流/收入; OHLCV代理: 短期动量/长期动量->收益加速=高FCF利润率->选股信号',
        'formula': 'close_p.pct_change(5) / (close_p.pct_change(20).abs() + 0.001)',
        'direction': 'long',
    },

    # --- 高级价格效率因子 ---
    {
        'id': 'novel_095', 'name': 'price_efficiency_ratio',
        'label': '价格效率比(Kaufman)',
        'category': 'price_efficiency',
        'hypothesis': '价格路径更直(趋势效率高)的股票信息传递快->动量可靠',
        'logic': 'Kaufman(1995)效率比+吴先兴框架: 价格效率=趋势距离/路径长度; 高效率=信息无摩擦->可持续',
        'formula': 'abs(close_p - close_p.shift(20)) / (abs(close_p - close_p.shift(1)).rolling(20).sum() + 0.001)',
        'direction': 'long',
    },
    {
        'id': 'novel_096', 'name': 'path_smoothness',
        'label': '路径平滑度',
        'category': 'price_efficiency',
        'hypothesis': '价格路径波动小的股票(不跳跃)有更纯净的趋势信号',
        'logic': '价格跳跃=信息冲击; 平滑路径=信息渐进->趋势更可预测; 平滑度=日均收益绝对值的波动',
        'formula': '-(close_p.pct_change().abs().rolling(20).std())',
        'direction': 'long',
    },
    {
        'id': 'novel_097', 'name': 'trend_efficiency_direction',
        'label': '趋势效率带方向',
        'category': 'price_efficiency',
        'hypothesis': '高效率且方向向上的股票(既快又准)是最优选股信号',
        'logic': '效率+方向=信息质量*动量方向; 高效率向上=好信息快速定价->最强Alpha',
        'formula': 'abs(close_p - close_p.shift(20)) / (abs(close_p - close_p.shift(1)).rolling(20).sum() + 0.001) * '
                   'pd.Series(np.sign(close_p.pct_change(20)), index=close_p.index)',
        'direction': 'long',
    },

    # --- Alpha动力学 & 机器学习启发 ---
    {
        'id': 'novel_098', 'name': 'alpha_decay_rate',
        'label': 'Alpha衰减速率',
        'category': 'alpha_dynamics',
        'hypothesis': 'Alpha衰减快的股票(短期信号)不适合中长期->Alpha衰减慢的选股更好',
        'logic': 'ML因子研究: Alpha半衰期=策略有效期; 衰减慢=信号持久->更稳健的超额收益',
        'formula': '-(close_p.pct_change(5).abs() / (close_p.pct_change(60).abs() + 0.001))',
        'direction': 'long',
    },
    {
        'id': 'novel_099', 'name': 'idiosyncratic_trend',
        'label': '异质趋势(残差动量V2)',
        'category': 'alpha_dynamics',
        'hypothesis': '剔除市场因素后5日/60日相对加速的股票(特质动量增强)Alpha更显著',
        'logic': 'Blitz et al.(2011)残差动量V2: 短期特质收益加速/长期=信号增强; 短期主导长期=Alpha积累',
        'formula': '(close_p.pct_change(5).sub(close_p.pct_change().mean(axis=1).rolling(5).sum(), axis=0)) / '
                   '((close_p.pct_change(60).sub(close_p.pct_change().mean(axis=1).rolling(60).sum(), axis=0)).abs() + 0.001)',
        'direction': 'long',
    },
    {
        'id': 'novel_100', 'name': 'signal_consensus',
        'label': '信号共识度',
        'category': 'alpha_dynamics',
        'hypothesis': '多时间尺度趋势方向一致(共振)的股票Alpha更强',
        'logic': '多因子模型: 信号共识=维度共振; 5/10/20/60日均为正=多周期趋势一致->最强信号',
        'formula': '(np.sign(close_p.pct_change(5)) + np.sign(close_p.pct_change(10)) + '
                   'np.sign(close_p.pct_change(20)) + np.sign(close_p.pct_change(60))) / 4',
        'direction': 'long',
    },

    # --- 游资/资金跟随微观结构 ---
    {
        'id': 'novel_101', 'name': 'stealth_accumulation_proxy',
        'label': '隐形吸筹代理',
        'category': 'cf_microstructure',
        'hypothesis': '缩量小涨(暗吸)的股票游资建仓->后续放量拉升',
        'logic': '游资打板量化: 隐形吸筹=量缩+价稳涨; 主力在低成交量环境下悄悄建仓->避免暴露',
        'formula': 'close_p.pct_change(5) * (-(volume_p.rolling(5).mean() / (volume_p.rolling(20).mean() + 0.001)) + 1)',
        'direction': 'long',
    },
    {
        'id': 'novel_102', 'name': 'volume_dry_up_reversal',
        'label': '量竭反转',
        'category': 'cf_microstructure',
        'hypothesis': '放量后急剧缩量(买方枯竭)的股票短期反转',
        'logic': 'CFv3.7: 天量次日缩量=买方力量耗尽->反转信号; 量缩比=今日量/5日最大量',
        'formula': '-(volume_p / (volume_p.rolling(5).max() + 0.001))',
        'direction': 'long',
    },
    {
        'id': 'novel_103', 'name': 'price_volume_rhythm',
        'label': '量价节奏',
        'category': 'cf_microstructure',
        'hypothesis': '上涨放量+下跌缩量的交替节奏(量价配合好)Alpha稳定',
        'logic': '游资策略核心: 量价配合节奏=健康上涨模式; 涨放量跌缩量=买方控盘->趋势可持',
        'formula': '((volume_p * (close_p.pct_change() > 0).astype(float)).rolling(10).sum() / '
                   '(volume_p.rolling(10).sum() + 1)) - '
                   '((volume_p * (close_p.pct_change() < 0).astype(float)).rolling(10).sum() / '
                   '(volume_p.rolling(10).sum() + 1))',
        'direction': 'long',
    },

    # --- 行为金融V2 (2026新视角) ---
    {
        'id': 'novel_104', 'name': 'reference_price_anchoring',
        'label': '参考价锚定(远离高点)',
        'category': 'behavioral_v2',
        'hypothesis': '远离20日高点的股票(锚定效应卖压小)上行空间大',
        'logic': '行为金融: 投资者锚定近期高点->接近时卖出->短期反转; 远离高点=锚定压力小->上行空间大',
        'formula': '1 - close_p / (close_p.rolling(20).max() + 0.001)',
        'direction': 'long',
    },
    {
        'id': 'novel_105', 'name': 'round_number_proximity',
        'label': '整数关口效应',
        'category': 'behavioral_v2',
        'hypothesis': '远离整数价格关口(心理阻力小)的股票更易上涨',
        'logic': '整数关口效应: 投资者以整数价格为目标; 接近整数=卖出压力->远离=无阻力->更容易涨',
        'formula': '-(close_p % 1).abs() / (close_p + 0.001)',
        'direction': 'long',
    },
    {
        'id': 'novel_106', 'name': 'disposition_proxy_v2',
        'label': '处置效应代理V2',
        'category': 'behavioral_v2',
        'hypothesis': '浮盈率高(卖压大)的股票短期回调->价格位置居中最佳',
        'logic': 'Odean(1998)处置效应V2: 浮盈->兑现冲动(卖压); 基于20日高低价格位置衡量浮盈浮亏',
        'formula': '(close_p - close_p.rolling(20).min()) / (close_p.rolling(20).max() - close_p.rolling(20).min() + 0.001)',
        'direction': 'long',
    },

    # --- 波动率结构高级因子 ---
    {
        'id': 'novel_107', 'name': 'volatility_term_structure',
        'label': '波动率期限结构',
        'category': 'volatility_structure',
        'hypothesis': '短期波动/长期波动比低的股票(波动率收敛)趋势可靠性高',
        'logic': '波动率期限结构: 短/长期波动比->波动预期方向; 比率低=波动收敛->趋势稳定',
        'formula': '-(close_p.pct_change().rolling(5).std() / (close_p.pct_change().rolling(60).std() + 0.001))',
        'direction': 'long',
    },
    {
        'id': 'novel_108', 'name': 'realized_vol_persistence',
        'label': '实现波动持续度',
        'category': 'volatility_structure',
        'hypothesis': '波动聚集效应强的股票(高波动持续)有波动溢价',
        'logic': '波动率聚类(volatility clustering)效应: 高波动持续->被高估的期权价值->波动溢价->做多标的',
        'formula': 'close_p.pct_change().abs().rolling(5).mean() / (close_p.pct_change().abs().rolling(20).mean() + 0.001)',
        'direction': 'long',
    },

    # --- 尾部风险 & 市场状态 ---
    {
        'id': 'novel_109', 'name': 'downside_beta_asymmetry',
        'label': '下行Beta非对称',
        'category': 'tail_risk',
        'hypothesis': '市场下跌时抗跌(低下行Beta)的股票是高质量信号',
        'logic': 'Ang et al.(2006)下行Beta: 下行Beta<上行Beta的股票(非对称风险)有正Alpha',
        'formula': '-(close_p.rolling(60).apply(lambda x: np.corrcoef(x, x.mean() - x)[0,1] if np.std(x) > 0 else 0, raw=True))',
        'direction': 'long',
    },
    {
        'id': 'novel_110', 'name': 'stress_resilience',
        'label': '压力韧性',
        'category': 'tail_risk',
        'hypothesis': '市场大跌日表现优于大盘的股票有基本面韧性',
        'logic': '压力测试: 极端下跌日收益=隐含尾部风险暴露; 抗跌=低尾部风险->高质量信号',
        'formula': 'close_p.pct_change().rolling(60).apply('
                   'lambda x: x[x < np.percentile(x, 20)].mean() - x.mean() if len(x) > 10 else 0, raw=True)',
        'direction': 'long',
    },
    {
        'id': 'novel_111', 'name': 'regime_adaptive_momentum',
        'label': '市场状态自适应动量',
        'category': 'regime_adaptive',
        'hypothesis': '波动率调整后的动量(市场状态自适应)比原始动量更稳定',
        'logic': 'Moreira&Muir(2017)波动管理: 动量/波动率->低波动时多投高波动时少投->提升夏普',
        'formula': 'close_p.pct_change(20) / (close_p.pct_change().rolling(60).std() + 0.001)',
        'direction': 'long',
    },
    {
        'id': 'novel_112', 'name': 'chop_index_proxy',
        'label': '震荡指数代理',
        'category': 'regime_adaptive',
        'hypothesis': '震荡指数低的股票(处于趋势市而非震荡市)趋势信号更可靠',
        'logic': '市场状态识别: Chop Index=价格方向性指标; 低值=趋势市->趋势策略有效; 高值=震荡市->反转策略有效',
        'formula': '-(close_p.rolling(20).apply(lambda x: x.std() / (abs(x.iloc[-1]-x.iloc[0]) + 0.001) - 1, raw=False))',
        'direction': 'long',
    },

    # --- 时间序列信号处理 ---
    {
        'id': 'novel_113', 'name': 'hurricane_eye_effect',
        'label': '台风眼效应',
        'category': 'price_efficiency',
        'hypothesis': '大波动后波动急速收缩(台风眼平静区)的股票方向性突破在即',
        'logic': '波动率均值回归: 极端波动->波动衰减->价格方向明确; 波动收缩率=未来动量信号',
        'formula': '-(close_p.pct_change().abs().rolling(5).std() - close_p.pct_change().abs().rolling(20).std()) / '
                   '(close_p.pct_change().abs().rolling(20).std() + 0.001)',
        'direction': 'long',
    },

    # ================================================================
    # v5 扩充 (2026-06-30) — 每日策略研究提炼因子
    # 来源: 6/28 每日报告前沿研究 (东方财富ARBR情绪/广发期货CTA/CSDN动量崩溃)
    # ================================================================

    # --- 情绪技术指标 (东方财富 ARBR情绪因子+小市值 6/22) ---
    {
        'id': 'novel_114', 'name': 'ARBR_26',
        'label': '人气意愿指标(26日)',
        'category': 'sentiment_technical',
        'hypothesis': 'AR(人气,买方力量)/BR(意愿,卖方力量)均衡的股票有更好动量; AR过强=人气过旺→回调',
        'logic': '东方财富(2026): ARBR情绪维度+小市值交叉验证; AR=26日(High-Open)之和/(Open-Low)之和→买方人气; BR=26日(High-PrevClose)之和/(PrevClose-Low)之和→卖方意愿; ARBR=(AR+BR)/2综合情绪',
        'formula': '((high_p - open_p).rolling(26).sum() / ((open_p - low_p).rolling(26).sum() + 0.001) + '
                   '(high_p - close_p.shift(1)).rolling(26).sum() / ((close_p.shift(1) - low_p).rolling(26).sum() + 0.001)) / 2',
        'direction': 'long',
    },
    # --- 多周期趋势一致性 (广发期货 2026半年报) ---
    {
        'id': 'novel_115', 'name': 'trend_alignment_ma',
        'label': '均线多头排列度',
        'category': 'multi_period',
        'hypothesis': '价格同时站上短/中/长均线的股票(均线多头排列)趋势最强',
        'logic': '广发期货(2026)CTA半年报: 多周期趋势组合贡献主要收益; 价格>MA5,MA20,MA60三者同时满足→多周期共振→最强趋势信号; 与多周期动量符号共振(033)互补(基于价格位置而非变化方向)',
        'formula': '((close_p > close_p.rolling(5).mean()).astype(float) + '
                   '(close_p > close_p.rolling(20).mean()).astype(float) + '
                   '(close_p > close_p.rolling(60).mean()).astype(float)) / 3',
        'direction': 'long',
    },
    # --- 动量脆弱性 (CSDN ETF动量崩溃风险 5/12) ---
    {
        'id': 'novel_116', 'name': 'momentum_fragility',
        'label': '动量脆弱性',
        'category': 'momentum_dynamics',
        'hypothesis': '近期频繁触及回撤阈值(>5%)的股票动量结构脆弱→即将崩溃概率高→应低配',
        'logic': 'CSDN(2026): 动量策略存在崩溃风险,需波动率止损+因子对冲; 频繁小幅回撤=持仓结构不稳→一次大回撤的预兆; 脆弱性=20日内回撤>5%的天数占比',
        'formula': '-(close_p / close_p.rolling(20).max() < 0.95).astype(float).rolling(20).mean()',
        'direction': 'long',
    },

    {
        'id': 'novel_117',
        'name': 'style_crowding_smallcap_reversal',
        'label': '小盘超跌修复(20日)',
        'category': 'style_crowding',
        'hypothesis': '小市值因子拥挤度达历史极值后,超跌小盘股进入修复窗口,未来收益为正',
        'logic': '国泰海通(2026-08)小市值拥挤度-1.00历史极值;微盘12次深调后50日平均反弹+19.2%(概率>80%);小盘代理=低60日量能,超跌=20日负收益,乘积捕捉修复弹性',
        'formula': '(1.0 / (volume_p.rolling(60).mean() + 1e-9)) * (-(close_p / close_p.shift(20) - 1))',
        'direction': 'long',
    },
    {
        'id': 'novel_118',
        'name': 'smallcap_recovery_window_50d',
        'label': '小盘50日深度超跌修复',
        'category': 'style_crowding',
        'hypothesis': '50日深度超跌的小盘股(拥挤度极值后的核心受损群体)反弹弹性最大',
        'logic': '开源证券复盘:微盘深调后50日反弹概率>80%平均+19.2%;用50日收益clip只保留下跌幅度,叠加小盘代理',
        'formula': '(1.0 / (volume_p.rolling(120).mean() + 1e-9)) * (-(close_p / close_p.shift(50) - 1).clip(upper=0.0))',
        'direction': 'long',
    },
    {
        'id': 'novel_119',
        'name': 'volume_retreat_stabilize',
        'label': '缩量下跌企稳反转',
        'category': 'style_crowding',
        'hypothesis': '缩量下跌(拥挤资金撤退后卖压衰竭)的股票短期反弹',
        'logic': '拥挤度下降过程伴随成交萎缩;量比<1且5日负收益=筹码交换充分,反弹买点',
        'formula': '((volume_p / volume_p.rolling(20).mean() - 1) * (close_p / close_p.shift(5) - 1))',
        'direction': 'long',
    },
    {
        'id': 'novel_120',
        'name': 'amplitude_expansion_ratio',
        'label': '振幅扩张比(20日/60日)',
        'category': 'volatility_structure',
        'hypothesis': '近期振幅相对历史扩张的股票进入高波动活跃期,趋势延续性强',
        'logic': '中金Loop Engineering(2026-08):振幅扩张与跳空溢价因子年化超额14.1%/超额Sharpe 2.64;振幅扩张=筹码活跃度提升',
        'formula': '((high_p - low_p) / close_p).rolling(20).mean() / (((high_p - low_p) / close_p).rolling(60).mean() + 1e-9)',
        'direction': 'long',
    },
    {
        'id': 'novel_121',
        'name': 'amplitude_expansion_zscore',
        'label': '振幅扩张zscore',
        'category': 'volatility_structure',
        'hypothesis': '振幅扩张的zscore越高,资金关注度上升越显著,动量持续性越强',
        'logic': '中金报告信号源:波动率回归与振幅扩张为独立盈利逻辑;zscore标准化避免量纲依赖',
        'formula': '(((high_p - low_p) / close_p).rolling(20).mean() - ((high_p - low_p) / close_p).rolling(60).mean()) / (((high_p - low_p) / close_p).rolling(60).std() + 1e-9)',
        'direction': 'long',
    },
    {
        'id': 'novel_122',
        'name': 'highvol_smallcap_oversold',
        'label': '高波动超跌修复',
        'category': 'style_crowding',
        'hypothesis': '高波动+20日超跌的股票(小微盘高Beta群体)在风格修复期弹性最大',
        'logic': '7月低波动因子+38%后风格均值回归;高波动超跌=修复beta最高群体;两条件乘积放大排序区分度',
        'formula': '((high_p - low_p) / close_p).rolling(20).mean() * (-(close_p / close_p.shift(20) - 1))',
        'direction': 'long',
    },
    {
        'id': 'novel_123',
        'name': 'chip_cost_premium_inversion',
        'label': '价格低于量能加权成本(套牢反转)',
        'category': 'chip_distribution',
        'hypothesis': '现价低于20日量能加权成本(套牢盘)的股票抛压已释放,后续反弹',
        'logic': 'P-20260817-002:筹码分布最高UCB方向新表达;VWAP成本=成交额加权均价(量价近似);现价低于成本=套牢区,超卖反转;避开已警告motif(开盘价30日最低zscore)',
        'formula': '-((close_p * volume_p).rolling(20).sum() / (volume_p.rolling(20).sum() + 1e-9) / (close_p + 1e-9) - 1)',
        'direction': 'long',
    },
    {
        'id': 'novel_124',
        'name': 'chip_cost_dispersion_low',
        'label': '筹码分歧度低(锁定好)',
        'category': 'chip_distribution',
        'hypothesis': '量能加权振幅小=筹码分歧度低,持仓结构稳定,后续上涨阻力小',
        'logic': '筹码锁定度用(高低价差×成交量)加权和代理;分歧度低=获利盘与套牢盘均少=拉升容易',
        'formula': '-(((high_p - low_p) * volume_p).rolling(20).sum() / (volume_p.rolling(20).sum() + 1e-9) / (close_p + 1e-9))',
        'direction': 'long',
    },
    {
        'id': 'novel_125',
        'name': 'chip_turnover_completion',
        'label': '20日换手完成度(洗盘充分)',
        'category': 'chip_distribution',
        'hypothesis': '20日累计换手占60日比重高=近期筹码充分交换,浮筹清理完成,趋势启动前兆',
        'logic': '筹码换手完成度=近期/长期换手比;洗盘充分后拉升阻力小;纯量能结构,与价格因子低相关',
        'formula': 'volume_p.rolling(20).sum() / (volume_p.rolling(60).sum() + 1e-9)',
        'direction': 'long',
    },
    {
        'id': 'novel_126',
        'name': 'chip_anchor_proximity',
        'label': '回踩成本线企稳',
        'category': 'chip_distribution',
        'hypothesis': '现价贴近20日量能成本线的股票获得成本支撑,回踩确认后上行概率高',
        'logic': '成本线=主力建仓均价;回踩不破=筹码支撑有效;abs距离越小得分越高',
        'formula': '-(close_p / ((close_p * volume_p).rolling(20).sum() / (volume_p.rolling(20).sum() + 1e-9) + 1e-9) - 1).abs()',
        'direction': 'long',
    },
    {
        'id': 'novel_127',
        'name': 'chip_breakout_volume_confirm',
        'label': '突破成本线+量能确认',
        'category': 'chip_distribution',
        'hypothesis': '放量突破20日成本线的股票解放全部套牢盘,开启上行空间',
        'logic': '突破成本线(溢价为正)且量比>1(资金确认)=主力拉升信号;乘积结构过滤无量假突破',
        'formula': '(close_p / ((close_p * volume_p).rolling(20).sum() / (volume_p.rolling(20).sum() + 1e-9) + 1e-9) - 1) * (volume_p / volume_p.rolling(20).mean())',
        'direction': 'long',
    },
    {
        'id': 'novel_128',
        'name': 'chip_profit_stability',
        'label': '收盘价持续高于成本线占比',
        'category': 'chip_distribution',
        'hypothesis': '收盘价连续高于20日量能成本线的天数占比高=持仓盈利稳定,趋势健康',
        'logic': '获利盘稳定的股票无抛压累积;占比=20日内收盘>VWAP成本的天数/20;稳定性优于单点价格',
        'formula': '((close_p * volume_p).rolling(20).sum() / (volume_p.rolling(20).sum() + 1e-9) < close_p).astype(float).rolling(20).mean()',
        'direction': 'long',
    },
    {
        'id': 'novel_129',
        'name': 'block_discount_aftermath',
        'label': '放量阴线后的修复(大宗折价效应)',
        'category': 'block_trade_proxy',
        'hypothesis': '大宗折价成交后次日往往放量阴线,抛压一次性释放后股价修复',
        'logic': 'P-20260817-005大宗范式冷启动前代理;大宗折价→二级市场跟卖→放量阴线→卖压出清→反转;真大宗折溢价因子待block_trade修复后替换',
        'formula': '-((volume_p / volume_p.rolling(20).mean()) * (close_p < open_p).astype(float)).rolling(5).mean()',
        'direction': 'long',
    },
    {
        'id': 'novel_130',
        'name': 'block_premium_momentum_proxy',
        'label': '放量阳线动量(大宗溢价效应)',
        'category': 'block_trade_proxy',
        'hypothesis': '大宗溢价成交反映买方意愿强烈,对应放量阳线,趋势延续',
        'logic': '溢价大宗=机构抢筹信号,二级市场放量阳线为代理形态;量价齐升确认',
        'formula': '((volume_p / volume_p.rolling(20).mean()) * (close_p > open_p).astype(float)).rolling(5).mean()',
        'direction': 'long',
    },
    {
        'id': 'novel_131',
        'name': 'block_frequency_activity_proxy',
        'label': '大额换手活跃度',
        'category': 'block_trade_proxy',
        'hypothesis': '大额换手(量比>1.5)频繁的股票机构调仓活跃,信息含量高,后续有趋势',
        'logic': '大宗频发=机构调仓期;代理:20日内量比>1.5的天数占比;真实大宗频率因子待数据接入',
        'formula': '(volume_p / volume_p.rolling(60).mean() > 1.5).astype(float).rolling(20).mean()',
        'direction': 'long',
    },
    {
        'id': 'novel_132',
        'name': 'intraday_close_position',
        'label': '收盘价在当日振幅中的位置',
        'category': 'microstructure_proxy',
        'hypothesis': '收盘位置高的股票(尾盘强势)次日动量延续',
        'logic': '高频微观结构未覆盖范式的日频代理;收盘位置=日内多空力量终局;位置高=尾盘买方控盘',
        'formula': '((close_p - low_p) / (high_p - low_p + 1e-9)).rolling(5).mean()',
        'direction': 'long',
    },
    {
        'id': 'novel_133',
        'name': 'overnight_gap_retreat',
        'label': '连续跳空后的回补压力',
        'category': 'microstructure_proxy',
        'hypothesis': '连续3日跳空高开消耗做多动能,缺口回补压力上升,短期反转',
        'logic': '中金报告:跳空溢价为独立盈利信号源但过度跳空后反转;3日累计缺口为正且过大则回补',
        'formula': '-(open_p / close_p.shift(1) - 1).rolling(3).mean()',
        'direction': 'long',
    },
    {
        'id': 'novel_134',
        'name': 'high_open_low_close_reversal',
        'label': '高开低走反转',
        'category': 'microstructure_proxy',
        'hypothesis': '高开低走(正缺口+收阴)反映日内抛压强,短期继续走弱,应低配',
        'logic': '高开低走=冲高回落,日内微观结构弱势;公式负号实现低配;5日平滑降低噪声',
        'formula': '-((open_p / close_p.shift(1) - 1) * (close_p < open_p).astype(float)).rolling(5).mean()',
        'direction': 'long',
    },
    {
        'id': 'novel_135',
        'name': 'gap_fill_failure',
        'label': '缺口盘中回补(做多失败)',
        'category': 'microstructure_proxy',
        'hypothesis': '上跳空缺口当日被回补(低点跌破前收)说明做多力量失败,后续走弱',
        'logic': '缺口回补=买方无法守住跳空成果;与未回补缺口(强势)形成镜像;负号低配',
        'formula': '-((open_p / close_p.shift(1) - 1) * (low_p < close_p.shift(1)).astype(float)).rolling(5).mean()',
        'direction': 'long',
    },
    {
        'id': 'novel_136',
        'name': 'price_efficiency_ratio',
        'label': '价格效率比(Kaufman)',
        'category': 'microstructure_proxy',
        'hypothesis': '价格效率比高的股票趋势顺畅(净变动大而路径短),动量质量高',
        'logic': 'Kaufman效率比=20日净变动/日间路径长度;高效=趋势驱动,低效=震荡;高频微观结构日频代理',
        'formula': '(close_p - close_p.shift(20)).abs() / (close_p.diff().abs().rolling(20).sum() + 1e-9)',
        'direction': 'long',
    },
    {
        'id': 'novel_137',
        'name': 'intraday_reversal_volume_confirm',
        'label': '日内反转强度×量能确认',
        'category': 'microstructure_proxy',
        'hypothesis': '低开高走(日内反转)且放量的股票买方力量真实,短期延续',
        'logic': '低开高走=日内反转结构;放量=资金真实参与而非无量修复;乘积过滤虚假反弹',
        'formula': '((close_p / open_p - 1) * (volume_p / volume_p.rolling(20).mean())).rolling(5).mean()',
        'direction': 'long',
    },
    {
        'id': 'novel_138',
        'name': 'crash_loss_reversal',
        'label': '深跌累计反转(截断-7%)',
        'category': 'momentum_dynamics',
        'hypothesis': '20日内单日跌幅超7%的累计损失越大,超跌反转动能越强',
        'logic': '动量崩溃风险(CSDN)的镜像:极端下跌后的均值回归;clip截断-7%聚焦深跌日,排除温和阴跌',
        'formula': '-(close_p / close_p.shift(1) - 1).clip(lower=-0.07).rolling(20).sum()',
        'direction': 'long',
    },
    {
        'id': 'novel_139',
        'name': 'dual_timeframe_momentum_gap',
        'label': '短长动量差(加速)',
        'category': 'momentum_dynamics',
        'hypothesis': '5日动量显著高于60日动量=动量加速期,趋势最强段',
        'logic': '短长动量差捕捉加速拐点;与单纯动量因子互补(二阶信息)',
        'formula': '(close_p / close_p.shift(5) - 1) - (close_p / close_p.shift(60) - 1)',
        'direction': 'long',
    },
    {
        'id': 'novel_140',
        'name': 'momentum_quality_20d',
        'label': '优质动量(动量/波动)',
        'category': 'momentum_dynamics',
        'hypothesis': '动量相同下波动更低的股票趋势质量高,持续性强',
        'logic': '7月动量因子-37%崩盘教训:纯动量脆弱;动量/波动比=质量过滤;与低波动风格形成交叉',
        'formula': '(close_p / close_p.shift(20) - 1) / (close_p.pct_change().rolling(20).std() + 1e-9)',
        'direction': 'long',
    },
    {
        'id': 'novel_141',
        'name': 'reversal_shrink_volume',
        'label': '缩量下跌反转',
        'category': 'momentum_dynamics',
        'hypothesis': '10日下跌且缩量的股票卖压衰竭,反转概率高',
        'logic': '缩量=主动抛盘减少;下跌+缩量=自然回落而非出货;量价背离的反转结构',
        'formula': '-(close_p / close_p.shift(10) - 1) * (1.0 - volume_p / (volume_p.rolling(20).mean() + 1e-9))',
        'direction': 'long',
    },
    {
        'id': 'novel_142',
        'name': 'momentum_acceleration_2nd',
        'label': '动量加速度(二阶)',
        'category': 'momentum_dynamics',
        'hypothesis': '2×20日动量-40日动量>0=近期动量快于前半程,加速上涨',
        'logic': '加速度=2*ret20-ret40;捕捉动量二阶变化;加速期趋势惯性最强',
        'formula': '2 * (close_p / close_p.shift(20) - 1) - (close_p / close_p.shift(40) - 1)',
        'direction': 'long',
    },
    {
        'id': 'novel_143',
        'name': 'max_return_reversal_20d',
        'label': 'MAX效应反转(20日)',
        'category': 'behavioral_genuine',
        'hypothesis': '20日内最大单日涨幅越高的股票(博彩偏好)后续收益越低',
        'logic': 'MAX效应(博彩偏好)的20日窗变体;已PASS的limited_attention_max_return_5d为5日窗,此为中长窗变体;行为金融SHAP 58%预测力佐证',
        'formula': '-(close_p.pct_change().rolling(20).max())',
        'direction': 'long',
    },
    {
        'id': 'novel_144',
        'name': 'overconfidence_volume_reversal',
        'label': '放量上涨过度自信反转',
        'category': 'behavioral_genuine',
        'hypothesis': '放量上涨(过度自信交易)的股票后续反转',
        'logic': '过度自信=交易量随盈利上升;放量×正动量=过度自信特征;均值回归预期;与已验证overconfidence变体互补',
        'formula': '-((close_p.pct_change().rolling(20).mean()) * (volume_p / volume_p.rolling(60).mean()))',
        'direction': 'long',
    },
    {
        'id': 'novel_145',
        'name': 'herding_chase_reversal',
        'label': '羊群追涨反转',
        'category': 'behavioral_genuine',
        'hypothesis': '放量暴涨(羊群追涨)的股票短期超买反转',
        'logic': '羊群效应=放量+单日大涨>2%;散户集中追涨后回调;5日平滑捕捉追涨集群',
        'formula': '-(volume_p / volume_p.rolling(20).mean() * (close_p.pct_change() > 0.02).astype(float)).rolling(5).mean()',
        'direction': 'long',
    },
    {
        'id': 'novel_146',
        'name': 'attention_decay_recovery',
        'label': '注意力衰减后修复',
        'category': 'behavioral_genuine',
        'hypothesis': '量比从5日峰值大幅回落的股票情绪退潮,筹码冷静后修复',
        'logic': '注意力衰减=关注度从峰值回落;情绪退潮则错杀修复;峰值-当前量比差为衰减幅度',
        'formula': '-((volume_p / volume_p.rolling(20).mean()).rolling(5).max() - volume_p / volume_p.rolling(20).mean())',
        'direction': 'long',
    },
    {
        'id': 'novel_147',
        'name': 'sentiment_pendulum_5d',
        'label': '情绪钟摆反转(5日)',
        'category': 'behavioral_genuine',
        'hypothesis': '5日累计涨幅过大(情绪过热)的股票反转',
        'logic': '情绪钟摆:短期极端涨幅透支预期;5日累计涨幅反转;与20日动量互为短长窗',
        'formula': '-(close_p.pct_change().rolling(5).sum())',
        'direction': 'long',
    },
    {
        'id': 'novel_148',
        'name': 'lowvol_lowturnover_defensive',
        'label': '低波动×低换手防御',
        'category': 'volatility_structure',
        'hypothesis': '低波动且低换手的股票在风格轮动期表现稳健',
        'logic': 'UBS 2026-07:低波动因子单月+38%;低波动×低换手=防御质量双条件;7月风格反转后防御因子重获溢价',
        'formula': '-(close_p.pct_change().rolling(20).std()) * (1.0 - volume_p / (volume_p.rolling(60).mean() + 1e-9))',
        'direction': 'long',
    },
    {
        'id': 'novel_149',
        'name': 'volatility_squeeze_ratio',
        'label': '波动率收缩比(20/60)',
        'category': 'volatility_structure',
        'hypothesis': '波动率从60日水平收缩的股票酝酿突破,风险收益比改善',
        'logic': '波动收缩=能量积蓄;20/60波动比<1为收缩;与振幅扩张(120)镜像,覆盖波动周期两端',
        'formula': '-(close_p.pct_change().rolling(20).std() / (close_p.pct_change().rolling(60).std() + 1e-9))',
        'direction': 'long',
    },
    {
        'id': 'novel_150',
        'name': 'up_down_volatility_asymmetry',
        'label': '上行波动/下行波动比',
        'category': 'volatility_structure',
        'hypothesis': '上涨日波动>下跌日波动的股票多方主导,防御性强势',
        'logic': '涨跌波动不对称反映资金态度;上行波动占比高=跌时有承接;纯波动结构因子',
        'formula': '(close_p.pct_change().clip(lower=0).rolling(20).std()) / (close_p.pct_change().clip(upper=0).rolling(20).std() + 1e-9)',
        'direction': 'long',
    },
    {
        'id': 'novel_151',
        'name': 'risk_adjusted_momentum_60d',
        'label': '60日风险调整动量',
        'category': 'volatility_structure',
        'hypothesis': '60日收益/波动比高的股票是长周期优质趋势',
        'logic': '低波动风格溢价的长窗验证;60日窗覆盖完整季度;UBS数据显示低波动长期有效',
        'formula': '(close_p / close_p.shift(60) - 1) / (close_p.pct_change().rolling(60).std() + 1e-9)',
        'direction': 'long',
    },
    {
        'id': 'novel_152',
        'name': 'low_amplitude_trend',
        'label': '低振幅趋势股',
        'category': 'volatility_structure',
        'hypothesis': '日内振幅低且20日趋势向上的股票为机构控盘型慢牛,持续性强',
        'logic': '低振幅=筹码锁定+机构控盘;正动量=方向确认;双条件乘积筛选慢牛股',
        'formula': '-((high_p - low_p).rolling(20).mean() / (close_p.rolling(20).mean() + 1e-9)) * (close_p / close_p.shift(20) - 1)',
        'direction': 'long',
    },
    {
        'id': 'novel_153',
        'name': 'mean_price_reversion_120d',
        'label': '120日均线偏离反转',
        'category': 'mean_reversion',
        'hypothesis': '价格显著低于120日均线的股票(长期滞涨)均值回归',
        'logic': 'UBS 2026-07价值+20%:估值洼地修复;120日均线偏离为价格维度的价值代理;与60日版本区分长周期',
        'formula': '-(close_p / close_p.rolling(120).mean())',
        'direction': 'long',
    },
    {
        'id': 'novel_154',
        'name': 'laggard_catchup_120d',
        'label': '120日滞涨补涨',
        'category': 'mean_reversion',
        'hypothesis': '120日涨幅垫底的股票在风格轮动中补涨',
        'logic': '价值/低波动轮动期滞涨股受益;120日收益反转捕捉长周期轮动;与20日反转区分时间尺度',
        'formula': '-(close_p / close_p.shift(120) - 1)',
        'direction': 'long',
    },
    {
        'id': 'novel_155',
        'name': 'stable_value_proxy',
        'label': '低价稳定(负动均/波动)',
        'category': 'mean_reversion',
        'hypothesis': '长期涨幅均值低且波动小的股票是价格维度的价值股,风格修复期受益',
        'logic': '负120日动均/120日波动=价格便宜且稳定;价值风格+20%下的双条件筛选',
        'formula': '-(close_p.pct_change().rolling(120).mean()) / (close_p.pct_change().rolling(120).std() + 1e-9)',
        'direction': 'long',
    },
    {
        'id': 'novel_156',
        'name': 'near_high_pullback_buy',
        'label': '高位回踩买点(趋势向上)',
        'category': 'mean_reversion',
        'hypothesis': '60日趋势向上且回踩距20日高点不超过20%的股票为趋势中的回踩买点',
        'logic': '趋势过滤(60日向上)+回踩幅度限制(<20%)=洗盘而非破位;close/20日max越接近1越强',
        'formula': '(close_p / close_p.rolling(20).max()).clip(lower=0.8) * (close_p > close_p.shift(60)).astype(float)',
        'direction': 'long',
    },

    {
        'id': 'novel_157',
        'name': 'close_position_consistency_5d',
        'label': '收盘位置一致性5日',
        'category': 'behavioral_genuine',
        'hypothesis': '收盘价持续收在当日振幅高位(买方控盘行为一致)的股票短期延续强势',
        'logic': '国盛WPE行为指纹(2026-08-15):行为一致性是核心维度;日线代理=收盘位置(close-low)/(high-low)的5日均值;位置持续高=买方行为一致',
        'formula': '((close_p - low_p) / (high_p - low_p + 1e-9)).rolling(5).mean()',
        'direction': 'long',
    },
    {
        'id': 'novel_158',
        'name': 'close_position_dispersion_5d',
        'label': '收盘位置离散度反转',
        'category': 'behavioral_genuine',
        'hypothesis': '收盘位置剧烈波动(行为不一致)的股票情绪紊乱,短期反转走弱',
        'logic': '行为一致性镜像:位置离散度=多空拉锯无主导;行为金融CGO(资本利得突出)负相关确认,分歧大=方向不明',
        'formula': '-((close_p - low_p) / (high_p - low_p + 1e-9)).rolling(5).std()',
        'direction': 'long',
    },
    {
        'id': 'novel_159',
        'name': 'gap_close_confirmation_divergence',
        'label': '跳空方向与收盘确认分歧',
        'category': 'behavioral_genuine',
        'hypothesis': '高开但收阴(开盘乐观被日内行为证伪)的股票短期走弱,分歧度越大越弱',
        'logic': '国盛WPE预训练目标含委托激进度与撤单压力;日线代理=跳空方向×(开-收)符号;行为证伪=情绪透支',
        'formula': '-((open_p / close_p.shift(1) - 1) * (close_p - open_p) / (high_p - low_p + 1e-9)).rolling(5).mean()',
        'direction': 'long',
    },
    {
        'id': 'novel_160',
        'name': 'upper_shadow_aggression',
        'label': '上影线攻击失败反转',
        'category': 'behavioral_genuine',
        'hypothesis': '连续出现长上影线(买方攻击被击退)的股票抛压聚集,短期走弱',
        'logic': '影线不对称=买卖双方攻击行为代理;上影长=冲高被卖盘压制;5日累积识别持续攻击失败',
        'formula': '-((high_p - close_p) / (high_p - low_p + 1e-9)).rolling(5).mean()',
        'direction': 'long',
    },
    {
        'id': 'novel_161',
        'name': 'lower_shadow_support',
        'label': '下影线承接买入',
        'category': 'behavioral_genuine',
        'hypothesis': '连续出现长下影线(卖方打压被买方承接)的股票底部支撑强,短期反弹',
        'logic': '下影长=盘中砸盘被承接,买方护盘行为;与上影线攻击失败镜像;行为金融承接=错杀修复',
        'formula': '((close_p - low_p) / (high_p - low_p + 1e-9) - (high_p - close_p) / (high_p - low_p + 1e-9)).rolling(5).mean()',
        'direction': 'long',
    },
    {
        'id': 'novel_162',
        'name': 'volume_asymmetry_updown',
        'label': '涨跌日成交量不对称',
        'category': 'behavioral_genuine',
        'hypothesis': '上涨日放量多于下跌日(买方行为积极)的股票趋势延续',
        'logic': '成交量在涨跌日的分布不对称=资金态度指纹;涨日量/跌日量比>1=买方主导;5日累计平滑',
        'formula': '((volume_p * (close_p > close_p.shift(1)).astype(float)).rolling(10).mean() / (volume_p * (close_p < close_p.shift(1)).astype(float) + 1e-9).rolling(10).mean())',
        'direction': 'long',
    },
    {
        'id': 'novel_163',
        'name': 'intraday_path_kurtosis',
        'label': '日内路径突变计数反转',
        'category': 'behavioral_genuine',
        'hypothesis': '近期多次出现极端日内反转路径(冲高回落或砸盘拉起)的股票行为紊乱,方向不明',
        'logic': 'WPE波动状态房间:路径突变为行为切换信号;代理=日内振幅与实体比值极端的天数计数;次数多=无共识',
        'formula': '-(((high_p - low_p) / (close_p.abs() + 1e-9) > 0.08).astype(float)).rolling(10).mean()',
        'direction': 'long',
    },
    {
        'id': 'novel_164',
        'name': 'behavior_herding_tail_chase',
        'label': '尾盘追涨羊群反转',
        'category': 'behavioral_genuine',
        'hypothesis': '收盘价逼近当日最高价(尾盘追涨)且放量的股票短期超买反转',
        'logic': '已PASS herding_industry_deviation 的个股级变体;尾盘追涨=收盘位置>0.9且量比>1.5;羊群追涨后回吐',
        'formula': '-(((close_p - low_p) / (high_p - low_p + 1e-9) > 0.9).astype(float) * (volume_p / volume_p.rolling(20).mean() > 1.5).astype(float)).rolling(5).mean()',
        'direction': 'long',
    },
    {
        'id': 'novel_165',
        'name': 'volume_price_trend_entropy',
        'label': '量价趋势一致性熵反转',
        'category': 'flow_microstructure',
        'hypothesis': '量与价的日度符号分歧(量价背离)累积的股票资金分歧大,反转',
        'logic': '流动性×微观结构MAB最高UCB方向新表达;量价符号一致性熵=资金与价格同向天数占比;背离多=分歧',
        'formula': '-((volume_p.pct_change() * close_p.pct_change() > 0).astype(float)).rolling(10).mean()',
        'direction': 'long',
    },
    {
        'id': 'novel_166',
        'name': 'dollar_turnover_acceleration',
        'label': '量价乘积二阶加速',
        'category': 'flow_microstructure',
        'hypothesis': '成交活跃度(量×价代理)二阶加速的股票资金持续涌入,动量延续',
        'logic': '成交额无法直接求值(无amount_p),用volume×close代理;二阶差分捕捉资金流入加速;流动性动量',
        'formula': '((volume_p * close_p).rolling(5).mean() / (volume_p * close_p).rolling(20).mean() - 1)',
        'direction': 'long',
    },
    {
        'id': 'novel_167',
        'name': 'liquidity_shock_absorption',
        'label': '放量冲击后快速企稳',
        'category': 'flow_microstructure',
        'hypothesis': '单日放量冲击(量比>2)后价格未崩(跌幅小)的股票承接力强,后续走强',
        'logic': '国盛量价淘金(十四)流动性冲击事件思路;冲击=放量,吸收=跌幅可控;承接力=买方资金雄厚',
        'formula': '((volume_p / volume_p.rolling(20).mean() > 2).astype(float) * (close_p.pct_change() > -0.03).astype(float)).rolling(10).mean()',
        'direction': 'long',
    },
    {
        'id': 'novel_168',
        'name': 'microstructure_turnover_price_drift',
        'label': '换手价格漂移比',
        'category': 'flow_microstructure',
        'hypothesis': '低换手驱动的上涨(筹码锁定型上涨)质量高于放量型,持续性强',
        'logic': '价格漂移/换手比=单位换手的价格弹性;低换手高涨幅=惜售锁仓;量价关系范式已验证BEST方向',
        'formula': '(close_p / close_p.shift(20) - 1) / (volume_p.rolling(20).sum() / (volume_p.rolling(20).mean() + 1e-9) + 1e-9)',
        'direction': 'long',
    },
    {
        'id': 'novel_169',
        'name': 'vwap_bias_convergence',
        'label': '价格向20日量价成本收敛',
        'category': 'flow_microstructure',
        'hypothesis': '价格偏离20日量能加权均价过大后向成本收敛,偏离向下则反弹',
        'logic': 'VWAP锚定效应:资金成本引力;负偏离(低于成本)反转修复;与novel_123成本溢价反演互补',
        'formula': '-((close_p / ((close_p * volume_p).rolling(20).sum() / (volume_p.rolling(20).sum() + 1e-9) + 1e-9) - 1))',
        'direction': 'long',
    },
    {
        'id': 'novel_170',
        'name': 'orderflow_imbalance_proxy',
        'label': '买卖失衡代理(收盘力)',
        'category': 'flow_microstructure',
        'hypothesis': '收盘位置与量能方向一致(收阳放量/收阴缩量)反映订单流失衡方向,延续',
        'logic': '订单流不平衡的日线代理=收盘位置×(收阳放量-收阴放量);失衡正向=买方订单主导',
        'formula': '(((close_p - low_p) / (high_p - low_p + 1e-9) - 0.5) * ((volume_p * (close_p > open_p).astype(float) - volume_p * (close_p < open_p).astype(float)) / (volume_p + 1e-9))).rolling(10).mean()',
        'direction': 'long',
    },
    {
        'id': 'novel_171',
        'name': 'intraday_volatility_share',
        'label': '日内波动贡献比',
        'category': 'flow_microstructure',
        'hypothesis': '日内振幅占区间总波动比低的股票走势平稳,机构控盘,趋势质量高',
        'logic': '微观结构平稳性=日内振幅/区间累计波动;低比值=盘中无剧烈博弈;与低波动范式交叉',
        'formula': '-(((high_p - low_p) / (close_p + 1e-9)).rolling(10).mean() / (close_p.pct_change().abs().rolling(30).mean() + 1e-9))',
        'direction': 'long',
    },
    {
        'id': 'novel_172',
        'name': 'active_buy_flow_proxy',
        'label': '主动买入代理(收阳量)',
        'category': '资金流',
        'hypothesis': '收阳日成交量占比高(主动买入占优)的股票资金持续流入,动量延续',
        'logic': '华泰行业资金流向图个股级代理:主动净买入=收阳量-收阴量;行业级真实因子走stage1提案路径;此处为量价代理',
        'formula': '((volume_p * (close_p > open_p).astype(float) - volume_p * (close_p < open_p).astype(float)) / (volume_p.rolling(20).mean() + 1e-9)).rolling(5).mean()',
        'direction': 'long',
    },
    {
        'id': 'novel_173',
        'name': 'flow_persistence_days',
        'label': '资金流入持续天数',
        'category': '资金流',
        'hypothesis': '连续收阳天数多的股票资金流入持续性强,趋势健康',
        'logic': '华泰微观信息维度:净流入持续性;代理=连续收阳计数;持续性>单日强度',
        'formula': '((close_p > open_p).astype(float).rolling(5).sum() - 2.5)',
        'direction': 'long',
    },
    {
        'id': 'novel_174',
        'name': 'flow_volatility_inverse',
        'label': '资金流波动率倒数',
        'category': '资金流',
        'hypothesis': '量能波动率低的股票资金进出平稳,机构型资金主导,防御性强',
        'logic': '华泰微观信息:资金流波动率;平稳流入=机构配置而非游资博弈;波动高=投机资金进出',
        'formula': '-(volume_p.pct_change().rolling(20).std())',
        'direction': 'long',
    },
    {
        'id': 'novel_175',
        'name': 'large_order_intensity_proxy',
        'label': '大单强度代理(量价冲击)',
        'category': '资金流',
        'hypothesis': '单日价格冲击(涨幅×量比)累积大的股票大资金活动频繁,趋势延续',
        'logic': '大单净流入的无金额代理:价格冲击=abs(ret)×量比;大资金活动=冲击大且方向正;5日累积',
        'formula': '(close_p.pct_change() * volume_p / (volume_p.rolling(20).mean() + 1e-9)).rolling(5).sum()',
        'direction': 'long',
    },
    {
        'id': 'novel_176',
        'name': 'industry_leader_follower_gap',
        'label': '龙头跟随者价差(个股级)',
        'category': '资金流',
        'hypothesis': '短期涨幅与市场宽度(自身量能趋势)背离的个股补涨',
        'logic': 'leader_follower_spread(attention类RESERVE)的价量变体:价格弱于量能趋势=资金已进价格未动=补涨',
        'formula': '((volume_p.rolling(10).mean() / volume_p.rolling(60).mean()) - (close_p / close_p.shift(20)))',
        'direction': 'long',
    },
    {
        'id': 'novel_177',
        'name': 'block_discount_shock_repair',
        'label': '放量阴线修复(大宗折价效应)',
        'category': 'block_trade_proxy',
        'hypothesis': '大宗折价成交在二级市场的放量阴线冲击后,抛压释放完毕,修复反弹',
        'logic': 'P-005大宗折价效应代理:折价大宗→次日放量阴线→卖压出清→反转;真实折价因子见stage1提案(block_discount_20d_weighted)',
        'formula': '-((volume_p / volume_p.rolling(20).mean() * (close_p < open_p).astype(float)).rolling(3).mean())',
        'direction': 'long',
    },
    {
        'id': 'novel_178',
        'name': 'block_buyer_concentration_proxy',
        'label': '集中放量吸筹代理',
        'category': 'block_trade_proxy',
        'hypothesis': '放量集中在少数交易日(机构集中吸筹)的股票筹码集中,后续上涨',
        'logic': '大宗买方席位集中度的量价代理:量能集中度=单日量/10日总量 max;集中放量=集中吸筹',
        'formula': '(volume_p / (volume_p.rolling(10).sum() + 1e-9)).rolling(10).max()',
        'direction': 'long',
    },
    {
        'id': 'novel_179',
        'name': 'block_discount_reversal_proxy',
        'label': '折价深度×反转交互代理',
        'category': 'block_trade_proxy',
        'hypothesis': '下跌深度与放量强度交互(折价甩货)的股票超跌,机构接盘后反转',
        'logic': '折价甩货=下跌+放量;机构接盘=跌后缩量;用下跌×量比识别甩货日,5日反转预期',
        'formula': '-((close_p.pct_change().clip(upper=0)) * (volume_p / volume_p.rolling(20).mean())).rolling(5).mean()',
        'direction': 'long',
    },
    {
        'id': 'novel_180',
        'name': 'overnight_sentiment_accumulation',
        'label': '隔夜情绪累积',
        'category': '情绪×日内',
        'hypothesis': '连续高开(隔夜乐观情绪累积)的股票情绪过热,反转',
        'logic': '隔夜情绪=跳空方向累积;连续跳空高开=乐观透支;情绪×日内范式扩充',
        'formula': '-(open_p / close_p.shift(1) - 1).rolling(5).sum()',
        'direction': 'long',
    },
    {
        'id': 'novel_181',
        'name': 'intraday_sentiment_reversal',
        'label': '日内情绪反转(低开高走)',
        'category': '情绪×日内',
        'hypothesis': '低开高走(日内情绪从悲观转向乐观)的股票情绪修复,延续性强',
        'logic': '情绪反转点:低开=隔夜悲观,高走=日内修复;日内情绪路径的纯OHLC刻画',
        'formula': '(((close_p - open_p) / (open_p + 1e-9)) * (open_p < close_p.shift(1)).astype(float)).rolling(5).mean()',
        'direction': 'long',
    },
    {
        'id': 'novel_182',
        'name': 'sentiment_pendulum_range_position',
        'label': '价格在20日区间位置反转',
        'category': '情绪×日内',
        'hypothesis': '价格处于20日区间下沿的股票情绪冰点,均值回归反弹',
        'logic': '区间位置=(close-20日min)/(20日max-20日min);位置低=情绪冰点;与RSI类反转逻辑同源',
        'formula': '-((close_p - close_p.rolling(20).min()) / (close_p.rolling(20).max() - close_p.rolling(20).min() + 1e-9) - 0.5)',
        'direction': 'long',
    },
    {
        'id': 'novel_183',
        'name': 'gap_sentiment_dispersion',
        'label': '跳空方向离散度',
        'category': '情绪×日内',
        'hypothesis': '跳空方向来回切换(隔夜情绪不稳定)的股票方向不明,短期走弱',
        'logic': '隔夜情绪稳定性:跳空方向符号的5日离散;离散大=情绪无共识;弱覆盖范式新表达',
        'formula': '-((open_p / close_p.shift(1) - 1).rolling(5).std())',
        'direction': 'long',
    },
    {
        'id': 'novel_184',
        'name': 'mean_reversion_volume_weighted',
        'label': '量权均线偏离反转',
        'category': 'mean_reversion',
        'hypothesis': '价格低于量能加权均线的股票(放量区被套)反转,量权均线偏离反转力更强',
        'logic': '均值回复BEST范式的量权升级:放量区的价格锚更强;负偏离=套牢于高量区,修复弹性大',
        'formula': '-((close_p / ((close_p * volume_p).rolling(30).sum() / (volume_p.rolling(30).sum() + 1e-9) + 1e-9) - 1))',
        'direction': 'long',
    },
    {
        'id': 'novel_185',
        'name': 'double_bottom_proxy',
        'label': '双底形态代理',
        'category': 'mean_reversion',
        'hypothesis': '价格两次触及相近低点后企稳(双底)的股票反转概率高',
        'logic': '形态反转:20日内两次低点相近(差<3%)且现价高于第二次低点=底部确认',
        'formula': '((close_p.rolling(10).min() / close_p.rolling(20).min() - 1).abs() < 0.03).astype(float) * (close_p / close_p.rolling(10).min() - 1)',
        'direction': 'long',
    },
    {
        'id': 'novel_186',
        'name': 'reversion_streak_break',
        'label': '连跌后首阳反转',
        'category': 'mean_reversion',
        'hypothesis': '连续下跌后出现放量首阳(卖压衰竭+买盘进场)的股票反转',
        'logic': '连跌3日+今日收阳=反转信号;放量确认;与均值回复BEST范式正交',
        'formula': '((close_p < close_p.shift(1)).astype(float).rolling(3).sum() == 3).astype(float) * (close_p > open_p).astype(float) * (volume_p / volume_p.rolling(20).mean())',
        'direction': 'long',
    },
    {
        'id': 'novel_187',
        'name': 'volume_price_confirmation_10d',
        'label': '量价确认10日',
        'category': '价量关系',
        'hypothesis': '10日上涨伴随温和放量(非天量)的股票趋势健康',
        'logic': '价量关系BEST范式:温和放量上涨=健康;天量=分歧;量比上界裁剪过滤天量',
        'formula': '(close_p / close_p.shift(10) - 1) * (volume_p.rolling(10).mean() / (volume_p.rolling(60).mean() + 1e-9)).clip(upper=2.0)',
        'direction': 'long',
    },
    {
        'id': 'novel_188',
        'name': 'volume_dry_up_turn',
        'label': '地量转折',
        'category': '价量关系',
        'hypothesis': '成交量收缩至60日最低(地量)且价格企稳的股票变盘向上',
        'logic': '地量=抛压枯竭;60日量比<0.6且5日价格不创新低=转折;经典量价底部信号',
        'formula': '((volume_p / volume_p.rolling(60).mean() < 0.6).astype(float) * (close_p > close_p.rolling(5).min()).astype(float)).rolling(3).mean()',
        'direction': 'long',
    },
    {
        'id': 'novel_189',
        'name': 'price_volume_divergence_bullish',
        'label': '价涨量稳背离',
        'category': '价量关系',
        'hypothesis': '价格创新高但成交量未同步放大(惜售)的股票筹码锁定,继续上涨',
        'logic': '量价背离的牛市形态:价格20日新高+量比<1.2=筹码锁定拉升;与天量出货镜像',
        'formula': '(close_p == close_p.rolling(20).max()).astype(float) * (volume_p / (volume_p.rolling(20).mean() + 1e-9) < 1.2).astype(float)',
        'direction': 'long',
    },
    {
        'id': 'novel_190',
        'name': 'stable_growth_lowvol',
        'label': '低波动稳定增长',
        'category': 'volatility_structure',
        'hypothesis': '低波动且60日趋势温和向上的股票是慢牛,收益风险比最优',
        'logic': '低波动×正动量(温和)=质量型慢牛;60日收益clip(0,0.15)裁剪暴涨股;7月低波+38%风格延续',
        'formula': '(close_p / close_p.shift(60) - 1).clip(lower=0, upper=0.15) / (close_p.pct_change().rolling(60).std() + 1e-9)',
        'direction': 'long',
    },
    {
        'id': 'novel_191',
        'name': 'volatility_regime_stability',
        'label': '波动率区制稳定性',
        'category': 'volatility_structure',
        'hypothesis': '波动率长期稳定(60日波动/120日波动≈1)的股票风格风险低',
        'logic': '波动区制稳定=无突然放大;比值偏离1越大说明近期波动异常;稳定者防御',
        'formula': '-((close_p.pct_change().rolling(60).std() / (close_p.pct_change().rolling(120).std() + 1e-9) - 1).abs())',
        'direction': 'long',
    },
    {
        'id': 'novel_192',
        'name': 'momentum_convexity',
        'label': '动量凸性(加速-减速)',
        'category': '动量反转',
        'hypothesis': '动量加速阶段(5日>20日)的股票趋势惯性最强,减速阶段反转',
        'logic': '动量二阶结构:加速=凸性正;5日动量-20日动量为加速度;与纯动量正交',
        'formula': '(close_p / close_p.shift(5) - 1) - (close_p / close_p.shift(20) - 1)',
        'direction': 'long',
    },
    {
        'id': 'novel_193',
        'name': 'reversal_after_gap_exhaustion',
        'label': '跳空衰竭反转',
        'category': '动量反转',
        'hypothesis': '连续跳空上涨后动量衰竭(5日动能转负)的股票反转',
        'logic': '跳空=动能爆发,连续跳空=衰竭前兆;5日累计跳空为正但今日高开低走=衰竭点;反转',
        'formula': '-((open_p / close_p.shift(1) - 1).rolling(5).sum() * (close_p < open_p).astype(float)).rolling(3).mean()',
        'direction': 'long',
    },
    {
        'id': 'novel_194',
        'name': 'chip_release_through_volume',
        'label': '放量突破压力位',
        'category': 'chip_distribution',
        'hypothesis': '放量突破60日高点的股票解放长期套牢盘,上行空间打开',
        'logic': '筹码压力位=60日高点;放量突破=套牢盘释放;突破+量确认双条件',
        'formula': '(close_p > close_p.rolling(60).max()).astype(float) * (volume_p / volume_p.rolling(20).mean())',
        'direction': 'long',
    },
    {
        'id': 'novel_195',
        'name': 'range_breakout_pullback',
        'label': '箱体突破回踩',
        'category': 'chip_distribution',
        'hypothesis': '突破20日箱体上沿后回踩不破的股票确认突破,趋势启动',
        'logic': '箱体=20日高低点区间;突破后回踩(现价接近突破位)=洗盘确认;回踩深度小=强势',
        'formula': '(close_p.rolling(20).max() / close_p.rolling(20).min() - 1 < 0.15).astype(float) * (close_p / close_p.rolling(20).min() - 1) * (close_p > close_p.rolling(5).mean()).astype(float)',
        'direction': 'long',
    },
    {
        'id': 'novel_196',
        'name': 'structure_break_gap_hold',
        'label': '突破缺口不回补',
        'category': '结构突变',
        'hypothesis': '向上跳空突破且缺口5日未回补的股票趋势强劲,动量延续',
        'logic': '结构突变+缺口未回补=多方力量持续;low>跳空前收确认缺口成立;5日保持',
        'formula': '((open_p / close_p.shift(1) - 1 > 0.03).astype(float) * (low_p.rolling(5).min() > close_p.shift(6)).astype(float)).rolling(3).mean()',
        'direction': 'long',
    },

    # ============= 路径加权动量 (P-20260826-001, 华福 CMM 简化版) =============
    # 核心: 打破"每天同等重要"假设, 用日间信息量(|ret|/振幅/量能)生成权重,
    #       对过去 60/120 日收益做加权和。权重=该信息量在滚动窗口内的时间序列分位
    #       (rolling(N).rank(pct=True), 窗口内排名无前视; 权重只依赖过去信息)。
    {
        'id': 'novel_197', 'name': 'vol_info_weighted_momentum_60',
        'label': '波动信息权重动量(60日)',
        'category': 'momentum',
        'hypothesis': '信息量大(单日波动剧烈)的交易日对未来趋势的预测力更强, 其收益应被赋予更高权重',
        'logic': '华福CMM路径加权动量: 日|ret|越高=信息含量越高; 用窗口内波动分位加权后动量的信噪比优于等权动量',
        'formula': '(close_p.pct_change() * close_p.pct_change().abs().rolling(60).rank(pct=True)).rolling(60).sum()',
        'direction': 'long',
    },
    {
        'id': 'novel_198', 'name': 'energy_weighted_momentum_120',
        'label': '能量权重动量(120日)',
        'category': 'momentum',
        'hypothesis': '波动×成交量能量(信息冲击)大的交易日对趋势的确认力更强, 加权后长期动量更稳健',
        'logic': '华福CMM简化: |ret|×volume=当日信息能量; 能量分位高=机构博弈激烈日, 其方向性收益权重应更高',
        'formula': '(close_p.pct_change() * (close_p.pct_change().abs() * volume_p).rolling(120).rank(pct=True)).rolling(60).sum()',
        'direction': 'long',
    },
    {
        'id': 'novel_199', 'name': 'amplitude_weighted_momentum_60',
        'label': '振幅信息权重动量(60日)',
        'category': 'momentum',
        'hypothesis': '振幅大的交易日承载更多定价博弈信息, 对收益方向贡献的权重应高于窄幅震荡日',
        'logic': '华福CMM路径加权: 振幅(high-low)/close=日内信息量; 高振幅日收益方向更可信, 加权后可提升动量因子ICIR',
        'formula': '(close_p.pct_change() * ((high_p - low_p) / close_p).rolling(60).rank(pct=True)).rolling(60).sum()',
        'direction': 'long',
    },
    {
        'id': 'novel_200', 'name': 'close_strength_weighted_momentum_60',
        'label': '收盘强度权重动量(60日)',
        'category': 'momentum',
        'hypothesis': '收盘强度(|收盘收益|/振幅)高的交易日=方向共识日, 其收益在动量中的权重应更高',
        'logic': '华福CMM简化: |ret|/振幅=日内方向强度(趋势日vs十字星); 共识日的收益信号更纯, 加权可降噪',
        'formula': '(close_p.pct_change() * (close_p.pct_change().abs() / ((high_p - low_p) / close_p + 0.001)).rolling(60).rank(pct=True)).rolling(60).sum()',
        'direction': 'long',
    },
    {
        'id': 'novel_201', 'name': 'volume_surprise_weighted_momentum_60',
        'label': '量能异动权重动量(60日)',
        'category': 'momentum',
        'hypothesis': '成交量相对均量放大的交易日=资金关注日, 其收益对动量的贡献应被放大',
        'logic': '华福CMM简化: vol/20日均量=量能异动; 放量日=新信息入场, 该日收益权重更高, 缩量日权重压低',
        'formula': '(close_p.pct_change() * (volume_p / (volume_p.rolling(20).mean() + 1)).rolling(60).rank(pct=True)).rolling(60).sum()',
        'direction': 'long',
    },
    {
        'id': 'novel_202',
        'name': 'ts_decay_ewm_halflife10',
        'label': '指数衰减加权收益(ewm 半衰期10日)',
        'category': 'ts_decay',
        'hypothesis': '近端收益信息权重大于远端, 衰减加权比等权动量响应更快',
        'logic': 'WQ101 decay_linear 族: ewm 指数衰减权重替代等权 rolling mean (纯 pandas, 无 np 依赖)',
        'formula': 'close_p.pct_change().ewm(halflife=10).mean()',
        'direction': 'long',
    },
    {
        'id': 'novel_203',
        'name': 'mkt_neutralized_momentum_20',
        'label': '市场中性化动量(20日超额市场动量, 行业中性待 P-001 升级)',
        'category': 'indneutralize',
        'hypothesis': '剔除市场 Beta 后的动量是个股特有信息, 拥挤度更低',
        'logic': 'WQ101 indneutralize 族降级版: 宽表 .mean(axis=1) 减全市场截面均值; 行业级中性化待 P-20260908-001 sw 行业映射接入后升级',
        'formula': '(close_p.pct_change(20).sub(close_p.pct_change(20).mean(axis=1), axis=0))',
        'direction': 'long',
    },
    {
        'id': 'novel_204',
        'name': 'signedpower_ret_20_15',
        'label': '符号幂收益(20日收益^1.5保符号)',
        'category': 'signedpower',
        'hypothesis': '保符号幂放大大幅波动股的信号强度, 小波动股被压缩',
        'logic': 'WQ101 signedpower 族: sign(x)*|x|^a 对极端收益敏感化 (x/|x| 实现符号, 无 np 依赖)',
        'formula': '((close_p.pct_change(20) / (close_p.pct_change(20).abs() + 1e-8)) * close_p.pct_change(20).abs() ** 1.5)',
        'direction': 'long',
    },

    # ============= P-20260910-004: 多尺度复杂性 (分形粗糙度/长记忆 × 回撤-恢复几何) =============
    # 库普查: fractal/hurst/multi_scale/geometry 命中 0 — 唯一完全空白范式。
    # 方向均为假设断言 (P-20260910-002 可证伪四元组), 以本地 S1 IC 实证符号为准。
    {
        'id': 'novel_205', 'name': 'hurst_vr_60_20',
        'label': '变差比近似Hurst(20步/1步, 60窗)',
        'category': 'multi_scale_complexity',
        'hypothesis': '分形市场假说: VR>1 长记忆正自相关(Hurst>0.5)趋势延续, VR<1 均值回归',
        'logic': 'Lo-MacKinlay 变差比简化: var(r_20)/(20*var(r_1)); >1 趋势, <1 反转',
        'formula': '(close_p.pct_change(20).rolling(60).var() / 20) / (close_p.pct_change().rolling(60).var() + 1e-8)',
        'direction': 'long',
        'refutation_conditions': '若 IC 符号为负(VR<1 反转占优)或 |ICIR|<0.05 则方向证伪, 以实证符号为准',
    },
    {
        'id': 'novel_206', 'name': 'multi_scale_vol_ratio_5_20',
        'label': '多尺度波动比(5日/20日, 粗糙度)',
        'category': 'multi_scale_complexity',
        'hypothesis': '短窗波动相对长窗越高=路径越粗糙(Hurst 低), 粗糙路径噪声大未来收益弱',
        'logic': 'CogAlpha Level V: 多尺度波动率比值谱度量粗糙度; 高比值=白噪声主导',
        'formula': 'close_p.pct_change().rolling(5).std() / (close_p.pct_change().rolling(20).std() + 1e-8)',
        'direction': 'short',
        'refutation_conditions': '若 IC 符号为正或 |ICIR|<0.05 则方向证伪',
    },
    {
        'id': 'novel_207', 'name': 'multi_scale_vol_ratio_5_60',
        'label': '多尺度波动比(5日/60日, 长尺度粗糙度)',
        'category': 'multi_scale_complexity',
        'hypothesis': '极短窗 vs 长窗波动比捕捉长记忆衰减速度; 比值高=短记忆白噪声',
        'logic': 'Hurst 指数与波动尺度律: 尺度比值谱斜率≈粗糙度; 5/60 比谱比 5/20 更平稳',
        'formula': 'close_p.pct_change().rolling(5).std() / (close_p.pct_change().rolling(60).std() + 1e-8)',
        'direction': 'short',
        'refutation_conditions': '若 IC 符号为正或 |ICIR|<0.05 则方向证伪',
    },
    {
        'id': 'novel_208', 'name': 'rough_path_ratio_20',
        'label': '路径粗糙度(20步路径长度/1步路径长度和)',
        'category': 'multi_scale_complexity',
        'hypothesis': '长步累计波动/短步累计波动=路径粗糙度代理; 越粗糙(Hurst 越低)未来噪声越大',
        'logic': '分形路径长度随尺度幂律增长; 比值偏离 1/sqrt(k) 度量粗糙度',
        'formula': 'close_p.pct_change(20).abs().rolling(60).sum() / (close_p.pct_change().abs().rolling(60).sum() * (20 ** 0.5) + 1e-8)',
        'direction': 'short',
        'refutation_conditions': '若 IC 符号为正或 |ICIR|<0.05 则方向证伪',
    },
    {
        'id': 'novel_209', 'name': 'hurst_vr_60_step60',
        'label': '变差比近似Hurst(60步/1步, 60窗)',
        'category': 'multi_scale_complexity',
        'hypothesis': '更长步长变差比对长记忆更敏感; VR60>1 强趋势延续',
        'logic': 'Lo-MacKinlay VR 长步长版: var(r_60)/(60*var(r_1)); 长记忆成分在长步长更显著',
        'formula': '(close_p.pct_change(60).rolling(60).var() / 60) / (close_p.pct_change().rolling(60).var() + 1e-8)',
        'direction': 'long',
        'refutation_conditions': '若 IC 符号为负或 |ICIR|<0.05 则方向证伪; 窗口不足(样本<150截面)时不裁决',
    },
    {
        'id': 'novel_210', 'name': 'vol_spectrum_slope',
        'label': '波动率谱斜率(长窗扩张-短窗扩张差)',
        'category': 'multi_scale_complexity',
        'hypothesis': '谱斜率为正(长窗波动扩张快于短窗)=波动聚集上升=风险积聚, 未来收益受压',
        'logic': '多尺度谱: (60/20 波动比)-(20/5 波动比) 度量谱斜率, 捕捉波动聚集的非对称演化',
        'formula': '(close_p.pct_change().rolling(60).std() / (close_p.pct_change().rolling(20).std() + 1e-8)) - (close_p.pct_change().rolling(20).std() / (close_p.pct_change().rolling(5).std() + 1e-8))',
        'direction': 'short',
        'refutation_conditions': '若 IC 符号为正或 |ICIR|<0.05 则方向证伪',
    },
    {
        'id': 'novel_211', 'name': 'dd_path_asymmetry',
        'label': '回撤路径对称性(下跌斜率/反弹斜率)',
        'category': 'multi_scale_complexity',
        'hypothesis': '下跌斜率>反弹斜率(急跌慢涨)的股票含负偏信息, 未来继续走弱; 慢跌急涨(V型)则反转修复',
        'logic': '回撤-恢复几何: 负收益均值/正收益均值度量路径非对称; 急跌慢涨=派发结构',
        'formula': 'close_p.pct_change().where(close_p.pct_change() < 0).rolling(20, min_periods=5).mean().abs() / (close_p.pct_change().where(close_p.pct_change() > 0).rolling(20, min_periods=5).mean() + 1e-8)',
        'direction': 'short',
        'refutation_conditions': '若 IC 符号为正或 |ICIR|<0.05 则方向证伪',
    },
    {
        'id': 'novel_212', 'name': 'dd_max_depth_20_60',
        'label': '20日最深回撤(相对60日高点)',
        'category': 'multi_scale_complexity',
        'hypothesis': '近 20 日最深回撤深的股票(近高点大幅回落)短期均值回归修复',
        'logic': '回撤深度维度: 相对 60 日高点的 20 日极值深度; 深度极端=超卖',
        'formula': '-((close_p / close_p.rolling(60).max()) - 1).rolling(20).min()',
        'direction': 'long',
        'refutation_conditions': '若 IC 符号为负或 |ICIR|<0.05 则方向证伪',
    },
    {
        'id': 'novel_213', 'name': 'dd_recovery_speed_5',
        'label': '回撤恢复速度(5日相对修复幅度)',
        'category': 'multi_scale_complexity',
        'hypothesis': '近 5 日从局部低点快速修复的股票有承接力量, 动量延续',
        'logic': '恢复速度维度: 相对 60 日高点序列的 5 日修复幅度; 修复快=买盘主动',
        'formula': '(close_p / close_p.rolling(60).max() - (close_p / close_p.rolling(60).max()).rolling(5).min()) * 100',
        'direction': 'long',
        'refutation_conditions': '若 IC 符号为负或 |ICIR|<0.05 则方向证伪',
    },
    {
        'id': 'novel_214', 'name': 'dd_recurrence_freq',
        'label': '回撤复发频率(20日内跌破95%高位线天数)',
        'category': 'multi_scale_complexity',
        'hypothesis': '频繁跌破 95% 高位线的股票处于脆弱结构, 未来走弱',
        'logic': '回撤复发维度: 高位线反复失守=支撑薄弱=派发; 频率高=脆弱性',
        'formula': '((close_p / close_p.rolling(60).max()) < 0.95).astype(float).rolling(20).sum()',
        'direction': 'short',
        'refutation_conditions': '若 IC 符号为正或 |ICIR|<0.05 则方向证伪',
    },
    {
        'id': 'novel_215', 'name': 'dd_geometry_curvature',
        'label': '回撤-恢复曲率(相对高位序列二阶差分10日均)',
        'category': 'multi_scale_complexity',
        'hypothesis': '相对高位序列曲率为正(修复加速)的股票形态转强',
        'logic': '几何曲率维度: 相对高位序列的加速度(二阶差分)度量恢复形态凸性',
        'formula': '(close_p / close_p.rolling(60).max()).diff().diff().rolling(10).mean()',
        'direction': 'long',
        'refutation_conditions': '若 IC 符号为负或 |ICIR|<0.05 则方向证伪',
    },
    {
        'id': 'novel_216', 'name': 'dd_highline_skew',
        'label': '相对高位序列偏度(60窗)',
        'category': 'multi_scale_complexity',
        'hypothesis': '相对高位序列正偏(长期低于高位后急拉)的股票修复动能强',
        'logic': '形态偏度维度: 相对高位序列三阶矩刻画高位停留-急杀-急拉的非对称形态',
        'formula': '(close_p / close_p.rolling(60).max()).rolling(60).skew()',
        'direction': 'long',
        'refutation_conditions': '若 IC 符号为负或 |ICIR|<0.05 则方向证伪',
    },

    # ============================================================
    # A1 条件化因子族 (novel_217~226, 2026-09-10 注入)
    # 结构: 信号 × 状态权重(中心化截面 rank), 乘性无掩码 → 规避 0 平局陷阱
    # 实证依据: crowded_volume_share_reversal_5d (ICIR 0.328 PASS) 与
    # peer_dev_* 同伴效应族 (09-06 PASS) 均属"状态条件化"结构
    # ============================================================
    {
        'id': 'novel_217', 'name': 'crowd_turnover_reversal_10d',
        'label': '拥挤度加权10日反转',
        'category': 'conditional_state',
        'hypothesis': '高成交拥挤的股票10日收益反转更强, 低拥挤股反转效应弱',
        'logic': '拥挤度(10日均量截面rank)中心化后加权反转信号, 与 crowded_volume_share_reversal_5d 同族换状态变量(成交额→成交量)',
        'formula': '-(close_p.pct_change(10)) * ((volume_p.rolling(10).mean().rank(axis=1, pct=True)) - 0.5)',
        'direction': 'long',
        'refutation_conditions': '若高拥挤组 10 日收益动量延续(IC 符号与反转假设相反)或 |ICIR|<0.05 则证伪',
    },
    {
        'id': 'novel_218', 'name': 'low_amp_momentum_5d',
        'label': '低振幅加权5日动量',
        'category': 'conditional_state',
        'hypothesis': '低振幅(窄幅整理)状态下5日动量延续, 高波动股动量噪声大',
        'logic': '振幅(高低价比-1)5日均值截面rank中心化, 低振幅股获得动量权重, 高振幅股反向下注',
        'formula': 'close_p.pct_change(5) * (0.5 - ((high_p / low_p - 1).rolling(5).mean().rank(axis=1, pct=True) - 0.5))',
        'direction': 'long',
        'refutation_conditions': '若低振幅股5日收益反转(IC为负)或 |ICIR|<0.05 则证伪',
    },
    {
        'id': 'novel_219', 'name': 'vol_expand_momentum_20d',
        'label': '波动扩张加权20日动量',
        'category': 'conditional_state',
        'hypothesis': '波动率扩张状态(20日std>60日std)下的20日动量延续更强',
        'logic': '波动扩张差(20日std-60日std)截面rank中心化加权20日收益, 刻画趋势启动期的量价共振',
        'formula': 'close_p.pct_change(20) * ((close_p.pct_change().rolling(20).std() - close_p.pct_change().rolling(60).std()).rank(axis=1, pct=True) - 0.5)',
        'direction': 'long',
        'refutation_conditions': '若波动扩张期动量反而反转(IC为负)或 |ICIR|<0.05 则证伪',
    },
    {
        'id': 'novel_220', 'name': 'turnover_shock_reversal_5d',
        'label': '量比突增加权5日反转',
        'category': 'conditional_state',
        'hypothesis': '量比突增(当日量/20日均量)的股票5日收益反转, 放量后短期过度反应',
        'logic': '量比偏离度截面rank中心化加权5日反转, 放量程度越极端反转权重越大',
        'formula': '-(close_p.pct_change(5)) * (((volume_p / volume_p.rolling(20).mean().shift(1)) - 1).rank(axis=1, pct=True) - 0.5)',
        'direction': 'long',
        'refutation_conditions': '若放量股5日动量延续(IC符号与反转相反)或 |ICIR|<0.05 则证伪',
    },
    {
        'id': 'novel_221', 'name': 'quiet_accum_breakout_20d',
        'label': '低振幅加权20日相对强度',
        'category': 'conditional_state',
        'hypothesis': '窄幅整理(低振幅)后的20日相对均线超额预示向上突破',
        'logic': '低振幅状态权重 × 20日相对均线超额, 刻画安静吸筹后的方向选择',
        'formula': '(close_p / close_p.rolling(20).mean() - 1) * (0.5 - ((high_p / low_p - 1).rolling(20).mean().rank(axis=1, pct=True) - 0.5))',
        'direction': 'long',
        'refutation_conditions': '若窄幅后20日相对强度反转(IC为负)或 |ICIR|<0.05 则证伪',
    },
    {
        'id': 'novel_222', 'name': 'crowd_fade_momentum_60d',
        'label': '高拥挤加权60日动量反转',
        'category': 'conditional_state',
        'hypothesis': '高成交拥挤股票60日中期动量反转, 拥挤透支后均值回归',
        'logic': '60日平均量截面rank中心化加权60日反转, 与 lhb 上榜频次负信号(拥挤=负向)呼应',
        'formula': '-(close_p.pct_change(60)) * ((volume_p.rolling(60).mean().rank(axis=1, pct=True)) - 0.5)',
        'direction': 'long',
        'refutation_conditions': '若高拥挤股60日动量延续(IC符号与反转相反)或 |ICIR|<0.05 则证伪',
    },
    {
        'id': 'novel_223', 'name': 'liq_state_reversal_10d',
        'label': '低流动性加权10日反转',
        'category': 'conditional_state',
        'hypothesis': '低流动性(冷门)股票的10日反转效应更强, 冷门股定价效率低',
        'logic': '流动性(20日均量rank)反向下注: 低流动性股获得反转权重, 高流动性股反向下注',
        'formula': '-(close_p.pct_change(10)) * (0.5 - (volume_p.rolling(20).mean().rank(axis=1, pct=True) - 0.5))',
        'direction': 'long',
        'refutation_conditions': '若低流动性股10日动量延续(IC符号与反转相反)或 |ICIR|<0.05 则证伪',
    },
    {
        'id': 'novel_224', 'name': 'hl_squeeze_cont_5d',
        'label': '高低价挤压加权5日动量',
        'category': 'conditional_state',
        'hypothesis': '5日高低价区间收窄(挤压)后的短期动量延续',
        'logic': '区间收窄度(5日max/min比)截面rank反向下注 × 5日动量, 与 CF 窄幅建仓逻辑同源',
        'formula': 'close_p.pct_change(5) * (0.5 - (((high_p.rolling(5).max() / low_p.rolling(5).min()) - 1).rank(axis=1, pct=True) - 0.5))',
        'direction': 'long',
        'refutation_conditions': '若挤压后5日收益反转(IC为负)或 |ICIR|<0.05 则证伪',
    },
    {
        'id': 'novel_225', 'name': 'dual_crowd_contrarian_20d',
        'label': '拥挤波动双高加权20日反转',
        'category': 'conditional_state',
        'hypothesis': '成交拥挤+波动率双高状态的20日反转最强, 双高=投机过热',
        'logic': '拥挤rank与波动rank等权复合后中心化加权20日反转, 双高股获得最强反转权重',
        'formula': '-(close_p.pct_change(20)) * (((volume_p.rolling(20).mean().rank(axis=1, pct=True)) + (close_p.pct_change().rolling(20).std().rank(axis=1, pct=True))) * 0.5 - 0.5)',
        'direction': 'long',
        'refutation_conditions': '若双高状态股20日动量延续(IC符号与反转相反)或 |ICIR|<0.05 则证伪',
    },
    {
        'id': 'novel_226', 'name': 'calm_momentum_20d',
        'label': '低波动加权20日动量',
        'category': 'conditional_state',
        'hypothesis': '低波动状态下的20日动量延续更强(低波质量动量)',
        'logic': '波动率(20日std)截面rank反向下注 × 20日动量, 与 P3 红利策略低波=alpha 实证呼应',
        'formula': 'close_p.pct_change(20) * (0.5 - (close_p.pct_change().rolling(20).std().rank(axis=1, pct=True) - 0.5))',
        'direction': 'long',
        'refutation_conditions': '若低波动股20日收益反转(IC为负)或 |ICIR|<0.05 则证伪',
    },
    {
        'id': 'kd_bp',
        'name': 'kuaidao_bp',
        'label': '快刀·账面市值比(1/PB)',
        'category': 'kuaidao_fundamental',
        'hypothesis': '低 PB 在全A截面有正向超额',
        'logic': '快刀估值层核心。实测 ICIR +0.341 / t=+2.16 (31月, 全A)',
        'formula': '(1.0 / pb).where(pb > 0)',
        'direction': 'long',
    },
    {
        'id': 'kd_ep_ttm',
        'name': 'kuaidao_ep_ttm',
        'label': '快刀·盈利收益率(1/PE_TTM)',
        'category': 'kuaidao_fundamental',
        'hypothesis': '低 PE_TTM 有正向超额',
        'logic': '实测 ICIR +0.173。亏损股 PE 为空 → 自动剔除',
        'formula': '(1.0 / pe_ttm).where(pe_ttm > 0)',
        'direction': 'long',
    },
    {
        'id': 'kd_sp_ttm',
        'name': 'kuaidao_sp_ttm',
        'label': '快刀·销售收益率(1/PS_TTM)',
        'category': 'kuaidao_fundamental',
        'hypothesis': '低 PS 有正向超额 (非稳定盈利公司的主估值口径)',
        'logic': '实测 ICIR +0.260; 净现金调整版 adj_sp_inv 更强 +0.282',
        'formula': '(1.0 / ps_ttm).where(ps_ttm > 0)',
        'direction': 'long',
    },
    {
        'id': 'kd_ps_own_pct_low',
        'name': 'kuaidao_ps_own_pct_low',
        'label': '快刀·PS自身历史分位取负',
        'category': 'kuaidao_fundamental',
        'hypothesis': 'PS 处于自身 250 日低位 = 相对历史便宜',
        'logic': '把「非稳定盈利门控」整体因子化。实测 ICIR +0.4125 / t=+2.18 (全表第3强)。分位在加载器内用完整历史预计算 (kd_ps_pct_own)',
        'formula': '-kd_ps_pct_own',
        'direction': 'long',
    },
    {
        'id': 'kd_dv_yield',
        'name': 'kuaidao_dv_yield',
        'label': '快刀·股息率(TTM)',
        'category': 'kuaidao_fundamental',
        'hypothesis': '高股息有正向超额 (股东回报层)',
        'logic': '实测 ICIR +0.117。替代原 div_yield 自算口径 (口径更标准)',
        'formula': 'dv_ttm',
        'direction': 'long',
    },
    {
        'id': 'kd_payout_ratio',
        'name': 'kuaidao_payout_ratio',
        'label': '快刀·股利支付率',
        'category': 'kuaidao_fundamental',
        'hypothesis': '适度分红有正向超额, 异常分红(>100%)是套现嫌疑',
        'logic': '实测 ICIR +0.149 (非单调)。恒等式: payout = dv_ratio*pe/100',
        'formula': '(dv_ratio * pe_ttm / 100.0).where((dv_ratio > 0) & (pe_ttm > 0))',
        'direction': 'long',
    },
    {
        'id': 'kd_net_cash_to_mv',
        'name': 'kuaidao_net_cash_to_mv',
        'label': '快刀·净现金占市值比',
        'category': 'kuaidao_fundamental',
        'hypothesis': '净现金厚(能覆盖负债后的冗余现金)的公司更抗跌',
        'logic': '给调整后市值口径提供暴露。实测 ICIR +0.045 (弱, 但为 PR 结构件)',
        'formula': '_nc = kd_money_cap.fillna(0) + kd_trad_asset.fillna(0) - kd_st_borr.fillna(0) - kd_lt_borr.fillna(0) - kd_bond_payable.fillna(0)\nfactor = _nc / (total_mv * 1e4)',
        'direction': 'long',
    },
    {
        'id': 'kd_est_np_yield_mv',
        'name': 'kuaidao_est_np_yield_mv',
        'label': '快刀·预估净利/市值',
        'category': 'kuaidao_fundamental',
        'hypothesis': '增长市盈率的倒数 (PR 分子端), 不清洗现金',
        'logic': '实测 ICIR +0.214 / t=+1.19。用 TTM 净利作预估净利代理',
        'formula': '(kd_n_income_ttm / (total_mv * 1e4)).where(kd_n_income_ttm > 0)',
        'direction': 'long',
    },
    {
        'id': 'kd_adj_sp_inv',
        'name': 'kuaidao_adj_sp_inv',
        'label': '快刀·营收/调整后市值',
        'category': 'kuaidao_fundamental',
        'hypothesis': '净现金调整后的销售收益率 (PS 口径剔除冗余现金)',
        'logic': '实测 ICIR +0.282, 强于未调整的 1/PS_TTM (+0.260) → 销售口径上净现金调整加分',
        'formula': '_nc = kd_money_cap.fillna(0) + kd_trad_asset.fillna(0) - kd_st_borr.fillna(0) - kd_lt_borr.fillna(0) - kd_bond_payable.fillna(0)\nfactor = (kd_revenue_ttm / (total_mv * 1e4 - _nc)).where((total_mv * 1e4 - _nc) > 0)',
        'direction': 'long',
    },
    {
        'id': 'kd_growth_pe_inv',
        'name': 'kuaidao_growth_pe_inv',
        'label': '快刀·预估净利/调整后市值',
        'category': 'kuaidao_fundamental',
        'hypothesis': '增长市盈率的倒数, 清洗冗余现金',
        'logic': '实测 ICIR +0.192 / t=+1.03 (盈利口径上净现金调整小幅减分)',
        'formula': '_nc = kd_money_cap.fillna(0) + kd_trad_asset.fillna(0) - kd_st_borr.fillna(0) - kd_lt_borr.fillna(0) - kd_bond_payable.fillna(0)\nfactor = (kd_n_income_ttm / (total_mv * 1e4 - _nc)).where((kd_n_income_ttm > 0) & ((total_mv * 1e4 - _nc) > 0))',
        'direction': 'long',
    },
    {
        'id': 'kd_ar_low',
        'name': 'kuaidao_ar_low',
        'label': '快刀·应收账款占比取负',
        'category': 'kuaidao_fundamental',
        'hypothesis': '应收账款占营收越低越好 (收入质量)',
        'logic': '快刀 2.3a: 应收/营收 <=20% 为 +1 条件之一。实测原始占比 ICIR -0.073 (弱)',
        'formula': '-(kd_accounts_receiv / kd_revenue_ttm).where(kd_revenue_ttm > 0)',
        'direction': 'long',
    },
    {
        'id': 'kd_advance_receipt_ratio',
        'name': 'kuaidao_advance_receipt_ratio',
        'label': '快刀·(合同负债+预收)/营收',
        'category': 'kuaidao_fundamental',
        'hypothesis': '预收类负债多 = 下游景气 / 议价权强',
        'logic': '快刀 2.3a 后半。⚠ 全A 实测 IC -0.374 与判定反向, 行业中性后归零 → 实为行业暴露, 接入供系统自行判定',
        'formula': '((kd_contract_liab.fillna(0) + kd_adv_receipts.fillna(0)) / kd_revenue_ttm).where(kd_revenue_ttm > 0)',
        'direction': 'long',
    },
    {
        'id': 'kd_goodwill_low',
        'name': 'kuaidao_goodwill_low',
        'label': '快刀·商誉占比取负',
        'category': 'kuaidao_fundamental',
        'hypothesis': '商誉占资产越低越好 (减值暴雷风险)',
        'logic': '快刀 2.3c。实测原始占比 ICIR -0.006 (近零)',
        'formula': '-(kd_goodwill.fillna(0) / kd_total_assets).where(kd_total_assets > 0)',
        'direction': 'long',
    },
    {
        'id': 'kd_asset_heaviness',
        'name': 'kuaidao_asset_heaviness',
        'label': '快刀·资产重资产度',
        'category': 'kuaidao_fundamental',
        'hypothesis': '经营性资产占总资产比重 (剔除现金后的真实经营资产密度)',
        'logic': '实测 ICIR +0.225, 行业中性后增强至 +0.372',
        'formula': '((kd_total_assets - kd_money_cap.fillna(0) - kd_trad_asset.fillna(0)) / kd_total_assets).where(kd_total_assets > 0)',
        'direction': 'long',
    },
    {
        'id': 'kd_cash_receipt_ratio',
        'name': 'kuaidao_cash_receipt_ratio',
        'label': '快刀·收现比',
        'category': 'kuaidao_fundamental',
        'hypothesis': '销售商品收到的现金 / 营收 > 1 = 收入真实回款',
        'logic': '快刀 2.4a。⭐ 数据 2026-09-11 补采 c_fr_sale_sg 后首次可算。未实测',
        'formula': '(kd_c_fr_sale_sg_ttm / kd_revenue_ttm).where(kd_revenue_ttm > 0)',
        'direction': 'long',
    },
    {
        'id': 'kd_cashflow_to_profit',
        'name': 'kuaidao_cashflow_to_profit',
        'label': '快刀·净现比',
        'category': 'kuaidao_fundamental',
        'hypothesis': '经营现金流净额 / 净利润 > 1 = 利润有现金支撑 (Sloan 应计异象直接代理)',
        'logic': '快刀 2.4b。⭐ 首次可算。未实测',
        'formula': '(kd_ncf_act_ttm / kd_n_income_ttm).where(kd_n_income_ttm > 0)',
        'direction': 'long',
    },
    {
        'id': 'kd_capex_low',
        'name': 'kuaidao_capex_low',
        'label': '快刀·资本开支占比取负',
        'category': 'kuaidao_fundamental',
        'hypothesis': '维持性资本开支 / 经营现金流越低越好 (轻资产 / 造血强)',
        'logic': '快刀 2.4c (阈值 <=10%)。⭐ 首次可算。未实测',
        'formula': '-(kd_capex_ttm / kd_ncf_act_ttm).where(kd_ncf_act_ttm > 0)',
        'direction': 'long',
    },
    {
        'id': 'kd_fcf_yield',
        'name': 'kuaidao_fcf_yield',
        'label': '快刀·自由现金流收益率',
        'category': 'kuaidao_fundamental',
        'hypothesis': '(经营现金流 - 资本开支) / 总市值',
        'logic': '快刀 2.4d。⭐ 首次可算。未实测',
        'formula': '((kd_ncf_act_ttm - kd_capex_ttm.fillna(0)) / (total_mv * 1e4)).where(kd_ncf_act_ttm.notna())',
        'direction': 'long',
    },
    {
        'id': 'kd_nonrecurring_ratio',
        'name': 'kuaidao_nonrecurring_ratio',
        'label': '快刀·扣非净利占净利比',
        'category': 'kuaidao_fundamental',
        'hypothesis': '扣非净利 / 净利润越高 = 盈利越干净 (非经常性损益越少)',
        'logic': '快刀 2.1 财务洗澡分支的代理。⭐ 首次可算 (fina_indicator.profit_dedt)。未实测',
        'formula': '(kd_profit_dedt_ttm / kd_n_income_ttm).where(kd_n_income_ttm > 0)',
        'direction': 'long',
    },
    {
        'id': 'kd_insider_net_buy',
        'name': 'kuaidao_insider_net_buy',
        'label': '快刀·大股东净增持比例(近3月)',
        'category': 'kuaidao_fundamental',
        'hypothesis': '近 3 月大股东/高管净增持为正 = 内部人看多',
        'logic': '实测 ICIR +0.160。阈值侧: 净增持 >0.03% 为 +1, 减持 <-0.03% 为 -1',
        'formula': 'kd_holder_net_ratio_3m',
        'direction': 'long',
    },
    {
        'id': 'kd_buyback_to_mv',
        'name': 'kuaidao_buyback_to_mv',
        'label': '快刀·回购金额/市值(近12月)',
        'category': 'kuaidao_fundamental',
        'hypothesis': '回购力度大 = net share issuance 异象',
        'logic': '🏆 全表最强: 实测 ICIR +0.465 / t=+2.59 / 胜率 74.2% (两轮一致)',
        'formula': '(kd_repurchase_amt_12m / (total_mv * 1e4)).where(total_mv > 0)',
        'direction': 'long',
    },
    {
        'id': 'kd_buyback_discount',
        'name': 'kuaidao_buyback_discount',
        'label': '快刀·现价相对回购价折价',
        'category': 'kuaidao_fundamental',
        'hypothesis': '(回购均价 - 现价)/现价 > 0 = 现价低于公司自己的回购成本',
        'logic': '快刀 2.8a 原意「当前股价低于回购价10%以上」。未实测',
        'formula': '((kd_repurchase_price_3m - close) / close).where((kd_repurchase_price_3m > 0) & (close > 0))',
        'direction': 'long',
    },
    {
        'id': 'kd_holder_num_yoy_low',
        'name': 'kuaidao_holder_num_yoy_low',
        'label': '快刀·股东户数同比取负',
        'category': 'kuaidao_fundamental',
        'hypothesis': '股东户数下降 = 筹码集中 = 好',
        'logic': '实测 ICIR +0.203 (取负后)。同比在加载器内于季频层预计算 (kd_holder_num_yoy)。⚠ full_top10 曾仅近2期, 2026-09-11 已回补至多期',
        'formula': '-kd_holder_num_yoy',
        'direction': 'long',
    },
    {
        'id': 'kd_per_capita_value',
        'name': 'kuaidao_per_capita_value',
        'label': '快刀·散户人均持股市值',
        'category': 'kuaidao_fundamental',
        'hypothesis': '(总市值 - 前十大持股市值) / 股东户数, 30万以上为 +1',
        'logic': '快刀 2.9 核心。实测原始值 ICIR -0.211 (门控池内翻转为 +0.055)',
        'formula': '((total_mv * 1e4 - kd_top10_hold_sum.fillna(0) * close) / kd_holder_num).where(kd_holder_num > 0)',
        'direction': 'long',
    },
    {
        'id': 'kd_top10_ratio',
        'name': 'kuaidao_top10_ratio',
        'label': '快刀·前十大股东持股比例',
        'category': 'kuaidao_fundamental',
        'hypothesis': '前十大持股比例高 = 筹码集中',
        'logic': '快刀 2.9 的辅助口径。数据 2026-09-11 回补至 21 期',
        'formula': 'kd_top10_hold_ratio',
        'direction': 'long',
    },
    {
        'id': 'kd_s2_score',
        'name': 'kuaidao_s2_score',
        'label': '快刀·十项打分第2项(ROE)',
        'category': 'kuaidao_fundamental',
        'hypothesis': 'ROE >=15% 记 +1, <10% 记 -1',
        'logic': '实测 ICIR -0.3564 / t=-1.98: 负向几乎全来自「低 ROE 扣分」支路 (2024-26 杀估值窗口)',
        'formula': '(roe >= 15).astype(float) - (roe < 10).astype(float)',
        'direction': 'long',
    },
    {
        'id': 'kd_s8_score',
        'name': 'kuaidao_s8_score',
        'label': '快刀·十项打分第8项(股权变动)',
        'category': 'kuaidao_fundamental',
        'hypothesis': '回购 或 大额增持 记 +1, 大额减持记 -1',
        'logic': '🏆 十项里唯一强项, 两轮一致: 实测 ICIR +0.438 / t=+2.44',
        'formula': '((kd_repurchase_amt_12m > 0) | (kd_holder_net_ratio_3m > 0.03)).astype(float) - (kd_holder_net_ratio_3m < -0.03).astype(float)',
        'direction': 'long',
    },
    {
        'id': 'kd_gate_pass',
        'name': 'kuaidao_gate_pass',
        'label': '快刀·价格门控(0/1)',
        'category': 'kuaidao_fundamental',
        'hypothesis': '稳定盈利走 PR<=1.5 路径, 非稳定盈利走 PS 自身分位<=70% 路径',
        'logic': '实测 ICIR +0.2585 / t=+1.42 (门控整体当因子须用 0/1; 连续 PR 值是零信号)。门控池 1349 只 = 全A 23.8%',
        'formula': '_nc = kd_money_cap.fillna(0) + kd_trad_asset.fillna(0) - kd_st_borr.fillna(0) - kd_lt_borr.fillna(0) - kd_bond_payable.fillna(0)\n_pe = (total_mv * 1e4 - _nc) / kd_n_income_ttm.where(kd_n_income_ttm > 0)\n_pr = _pe / (roe.where(roe > 0) / 100.0)\nfactor = ((_pr <= 1.5) | (kd_ps_pct_own <= 0.70)).astype(float)',
        'direction': 'long',
    },
    {
        'id': 'kd_quality_value_inter',
        'name': 'kuaidao_quality_value_inter',
        'label': '快刀·质量价值交互',
        'category': 'kuaidao_fundamental',
        'hypothesis': '便宜且高质量 (bp 分位 + ROE 分位) 的组合暴露',
        'logic': '把快刀「好公司 + 好价格」双条件合成为一个连续因子。未实测',
        'formula': '_bp = (1.0 / pb).where(pb > 0)\nfactor = _bp.rank(pct=True).fillna(0.5) + roe.rank(pct=True).fillna(0.5)',
        'direction': 'long',
    },
    # ============= P-20260914-001 Spread-Compression 拥挤诊断 (探索性, 未实测) =============
    {
        'id': 'novel_sc001',
        'name': 'spread_compression_20d',
        'label': 'Spread-Compression 拥挤侵蚀代理(20日相对溢价斜率, 取负)',
        'category': 'sentiment_crowding',
        'hypothesis': '相对截面溢价被持续压缩(斜率<0, 拥挤侵蚀)的股票短期出现反转修复; 溢价仍在扩张的股票动量延续',
        'logic': 'P-20260914-001 (QuanterLab spread-compression 口径 + McLean&Pontiff 2016 发表后衰减58%): 股票级代理 = 20日日收益(vs 截面均值)的 OLS 时间斜率取负; 与 factor_level_crowding 的 corr 口径互补。未实测',
        'formula': '_ret = close_p / close_p.shift(1) - 1\n_rel = _ret.sub(_ret.mean(axis=1), axis=0)\n_t = np.arange(20, dtype=float)\n_dev = _t - _t.mean()\n_den = (_dev ** 2).sum()\nfactor = -(_rel.rolling(20).apply(lambda s: (_dev * s).sum(), raw=True) / _den)',
        'direction': 'long',
    },
    {
        'id': 'novel_227',
        'name': 'lion_bias',
        'label': '涨停启动点贴近度(结构锚点)',
        'category': 'behavioral_anchor',
        'hypothesis': '现价贴近「最近一次涨停日之前最后一根阴线低点」的股票处于主力成本区, 下行空间有限且修复概率高',
        'logic': '事件锚点(非固定窗口): 最近涨停=主力发力信号; 其前最后一根阴线低点=启动点/成本下沿; 现价/启动点越低=越贴近成本区→结构型反转。来源: 聚宽社区《变种小狮子 V3.0》bias 构造, 口径改为后复权比值。实测 RankIC -0.0250 / 日ICIR -0.178 (过闸); 但与本库 rev20/rev60 截面相关 0.40~0.48, 正交后 IC 仅 +0.0090 → 主要是反转代理',
        'formula': 'lion_bias',
        'direction': 'short',
    },
    {
        'id': 'novel_228',
        'name': 'lion_lu_days',
        'label': '距最近涨停天数',
        'category': 'event_decay',
        'hypothesis': '距最近一次涨停越久(沉寂越深)的股票未来收益越高 —— 涨停后的注意力溢价随时间衰减',
        'logic': '涨停=注意力事件, 其后接力资金与情绪溢价随时间衰减; 距涨停远=情绪已出清未被炒作。实测 RankIC +0.0439 / 日ICIR +0.410, 五分位单调性 +0.900 (Q1 -0.168% → Q5 +0.227%), 小/中/大市值三池 t = +5.65 / +8.35 / +2.64 全部显著。与本库 rev20/rev60 相关仅 +0.11/+0.21, 与 size -0.04 → 非反转/市值代理; 但与同源 lion_lu_cnt 相关 -0.655',
        'formula': 'lion_lu_days',
        'direction': 'long',
    },
    {
        'id': 'novel_229',
        'name': 'lion_lu_cnt',
        'label': '760日涨停次数(涨停基因)',
        'category': 'event_decay',
        'hypothesis': '过去 760 交易日涨停次数越多的股票未来收益越低 —— 与社区「涨停基因」选股方向相反',
        'logic': '涨停次数=历史被炒作程度; 炒作过的票套牢盘重、筹码分散。聚宽社区策略取涨停次数前 10% 当优势股池, 实测方向相反 (RankIC -0.0195, 五分位单调性 -1.000)。与 lion_lu_days 相关 -0.655 (同源两测度), 二者择一即可',
        'formula': 'lion_lu_cnt',
        'direction': 'short',
    },

    # ============= 多源分通道建模类 (P-20260916-001 落地, 2026-09-16) =============
    # 来源: 国泰海通量化选股系列(五)《让深度学习因子适应不同市场环境》
    #   实证: 多粒度深度学习因子与市值因子相关 -0.3, 根因是成交额/换手等风格高相关
    #   特征在同一 GRU 注意力机制中被赋过高权重; 改为「分通道建模 + 相关性惩罚层」
    #   后风格分位回归 50%、回撤降近 10%。
    # 落地代理: 各通道独立构建截面 rank 信号后再组合 (而非同式混用异类字段),
    #   以此在结构上避免「某类特征在共享参数中被吞并」。
    {
        'id': 'novel_230',
        'name': 'mc_reversal_lgflow_equalweight',
        'label': '分通道: 量价反转 × 大单资金流(等权)',
        'category': 'multi_channel',
        'hypothesis': '量价反转通道与主动大单资金流通道是低相关的两类信息, 各自截面 rank 后等权组合, 可避免单一通道在混合建模中被吞并',
        'logic': '通道A=20日收益反转的截面分位; 通道B=大单净买占比5日均值的截面分位; 组合=0.5A+0.5B。两通道独立构建后才相加, 不在同一表达式内混用 close 与资金流字段。来源 P-20260916-001 / 国泰海通量化选股系列(五)',
        'formula': "(-close_p.pct_change(20)).rank(pct=True, axis=1) * 0.5 + ((buy_lg_vol - sell_lg_vol) / (buy_lg_vol + sell_lg_vol + 1e-8)).rolling(5).mean().rank(pct=True, axis=1) * 0.5",
        'direction': 'long',
    },
    {
        'id': 'novel_231',
        'name': 'mc_momentum_elgflow_gate',
        'label': '分通道: 量价动量 × 超大单资金流(门控)',
        'category': 'multi_channel',
        'hypothesis': '仅当量价动量通道与超大单资金流通道同时处于高分位时才触发(相乘门控), 两个独立通道同时失效的概率相乘 → 组合对单一 regime 更稳健',
        'logic': '通道A=20日动量的截面分位; 通道B=超大单净买占比10日均值的截面分位; 组合=A×B 门控式, 要求两个独立信号源同向才得高分。来源 P-20260916-001',
        'formula': "(close_p.pct_change(20)).rank(pct=True, axis=1) * ((buy_elg_vol - sell_elg_vol) / (buy_elg_vol + sell_elg_vol + 1e-8)).rolling(10).mean().rank(pct=True, axis=1)",
        # 落地日探针(270d, 不复权, fwd5): RankIC -0.0294 / ICIR 0.232 → 实证为负,
        # 即「动量×超大单门控」在 A 股是反向指标(与库内「纯动量不可用」结论一致)。
        # 按方向本地实证翻转铁律, 方向标 short 而非 long。
        'direction': 'short',
    },
    {
        'id': 'novel_232',
        'name': 'mc_inst_minus_retail_divergence',
        'label': '分通道: 机构通道 − 散户通道(资金结构分歧)',
        'category': 'multi_channel',
        'hypothesis': '机构通道与散户通道方向背离的股票未来收益更高 —— 机构净买而散户净卖代表筹码由弱手转向强手',
        'logic': '通道A=大单+超大单净买占比5日均值的截面分位(机构通道); 通道B=小单净买占比5日均值的截面分位(散户通道); 组合=A−B。两通道各自独立构建 rank 后才相减, 避免在同一表达式内混用同源量纲。来源 P-20260916-001',
        'formula': "((buy_lg_vol + buy_elg_vol - sell_lg_vol - sell_elg_vol) / (buy_lg_vol + buy_elg_vol + sell_lg_vol + sell_elg_vol + 1e-8)).rolling(5).mean().rank(pct=True, axis=1) - ((buy_sm_vol - sell_sm_vol) / (buy_sm_vol + sell_sm_vol + 1e-8)).rolling(5).mean().rank(pct=True, axis=1)",
        'direction': 'long',
    },
    {
        'id': 'novel_233',
        'name': 'mc_dynamic_lowcorr_weight',
        'label': '分通道: 动态低相关加权(三通道)',
        'category': 'multi_channel',
        'hypothesis': '按通道间 60 日滚动相关的绝对值倒数动态加权(低相关者高权), 可在通道相关性上升(风格趋同)时自动降权 —— 直接对应国泰海通「相关性惩罚层」',
        'logic': '三通道=20日反转分位 / 大单净买分位 / 成交额集中分位; 权重 w=∏(1−|rolling_corr|), 逐通道乘权后相加。通道间相关性升高(两类信息趋同)时该项权重自动衰减, 等价于对风格暴露施加惩罚。多行赋值块实现。来源 P-20260916-001 / 国泰海通量化选股系列(五) 相关性惩罚层',
        'formula': """ch_a = (-close_p.pct_change(20)).rank(pct=True, axis=1).fillna(0.5)
ch_b = ((buy_lg_vol - sell_lg_vol) / (buy_lg_vol + sell_lg_vol + 1e-8)).rolling(5).mean().rank(pct=True, axis=1).fillna(0.5)
ch_c = (amount / amount.rolling(20).sum()).rank(pct=True, axis=1).fillna(0.5)
c_ab = ch_a.rolling(60).corr(ch_b).fillna(0.0).abs().clip(0.0, 1.0)
c_ac = ch_a.rolling(60).corr(ch_c).fillna(0.0).abs().clip(0.0, 1.0)
c_bc = ch_b.rolling(60).corr(ch_c).fillna(0.0).abs().clip(0.0, 1.0)
w_a = (1.0 - c_ab) * (1.0 - c_ac)
w_b = (1.0 - c_ab) * (1.0 - c_bc)
w_c = (1.0 - c_ac) * (1.0 - c_bc)
factor = w_a * ch_a + w_b * ch_b + w_c * ch_c
""",
        'direction': 'long',
    },
    # ============= 龙虎榜三路资金合力 (P-20260917-022, 2026-09-17) =============
    # 来源: 2026-09-17 前沿扫描 D 维度 —— 9/16 光迅科技(002281)涨停, 深股通净买
    #       4.61亿 + 3家机构净买 2.53亿 + 游资(紫阳东路/环城北路)净买 4.09亿, 三路同日合力超10亿。
    # 三路定义: lhb_north_net_buy(深股通/沪股通专用席位) / lhb_inst_net_buy(机构专用)
    #          / lhb_yz_net_buy(排除机构与北向后的游资营业部)。
    # 口径约定: load_top_list_wide 已 reindex+fillna(0) ⇒ **未上榜 = 0**。故三路同日
    #          同为正才构成合力, 缺任意一路即不成立 —— 这是定义本身, 不是 fillna 伪装信号。
    {
        'id': 'novel_234',
        'name': 'lhb_triple_resonance_5d',
        'label': '龙虎榜三路资金合力强度(5日累积)',
        'category': 'lhb_resonance',
        'hypothesis': '龙虎榜深股通、机构专用、游资营业部三路资金同日同向净买入的股票, 后续收益显著高于单路资金买入的股票',
        'logic': '三路资金代表三类独立信息源(外资定价/机构研究/游资情绪); 同日同向净买=多源信息共振, 是稀缺强信号(单路买入可能仅是风格暴露)。合力强度=三路净额之和, 5日累积后取截面分位。',
        'formula': """_n = lhb_north_net_buy.fillna(0.0)
_i = lhb_inst_net_buy.fillna(0.0)
_y = lhb_yz_net_buy.fillna(0.0)
_res = ((_n > 0) & (_i > 0) & (_y > 0)).astype(float)
factor = ((_n + _i + _y) * _res).rolling(5).sum().rank(pct=True, axis=1).fillna(0.5)""",
        'direction': 'long',
        'source': 'P-20260917-022',
    },
    {
        'id': 'novel_235',
        'name': 'lhb_triple_resonance_persist_20d',
        'label': '龙虎榜三路资金合力持续性(20日频次)',
        'category': 'lhb_resonance',
        'hypothesis': '近20日内三路资金合力出现频次高的股票(反复被多路资金共同买入)比一次性合力的股票更有持续性',
        'logic': '单次合力可能是事件驱动(涨停/公告); 20日内反复合力=资金持续建仓, 属筹码由弱手转强手的持续过程。用频次而非强度, 与 novel_234 的强度维度正交。',
        'formula': """_both = ((lhb_north_net_buy.fillna(0.0) > 0) & (lhb_inst_net_buy.fillna(0.0) > 0) & (lhb_yz_net_buy.fillna(0.0) > 0)).astype(float)
factor = _both.rolling(20).sum().rank(pct=True, axis=1).fillna(0.5)""",
        'direction': 'long',
        'source': 'P-20260917-022',
    },
    {
        'id': 'novel_236',
        'name': 'lhb_triple_agree_minus_disagree_10d',
        'label': '龙虎榜三路资金方向净一致性(10日累积)',
        'category': 'lhb_resonance',
        'hypothesis': '三路资金方向净一致(同买次数减同卖次数)高的股票未来收益更高 —— 方向一致性比净额大小更能区分信息共振与单路主导',
        'logic': 'novel_234 用净额强度, 会被单路巨量主导(如某游资单笔巨额); 本因子只数方向一致性: 三路同买=+1, 三路同卖=−1, 其余=0, 10日累积。剔除金额量纲影响, 是纯"共识"度量。',
        'formula': """_n = lhb_north_net_buy.fillna(0.0)
_i = lhb_inst_net_buy.fillna(0.0)
_y = lhb_yz_net_buy.fillna(0.0)
_dir = ((_n > 0) & (_i > 0) & (_y > 0)).astype(float) - ((_n < 0) & (_i < 0) & (_y < 0)).astype(float)
factor = _dir.rolling(10).sum().rank(pct=True, axis=1).fillna(0.5)""",
        'direction': 'long',
        'source': 'P-20260917-022',
    },
    # ============= 龙虎榜三路资金**分歧度** (P-20260918-001, 2026-09-18) =============
    # 🔴 与 novel_234/235/236 的关键差异: 那三条用 `fillna(0)` 后直接 rolling ⇒
    #    非上榜日三路全为 0, 「三路同买」恒不成立 ⇒ 序列里绝大多数样本是「平局 0」,
    #    截面 rank 被大量 0 值占位稀释, 分歧/共识的**区分度被摊平**。
    #    本组改为 **先锚在上榜日** (lhb_flag > 0 或任一路非零), 只在真实上榜样本上
    #    度量分歧度, 再对「上榜样本」做时序聚合 ⇒ 不引入 fillna(0) 的假信号。
    # 🔴 待验风险 (须在 S5 复核): 本组在宽表 (index=日期) 下 rank(pct,axis=1) 是
    #    截面分位; 而 S5 `_build_series_context` 是**逐股 Series** 上下文, axis=1 语义
    #    不同 (novel_234/235/236 存在同样问题, 属既有口径, 本组沿用不扩大差异)。
    {
        'id': 'novel_237',
        'name': 'lhb_3way_divergence_20d',
        'label': '龙虎榜三路资金方向分歧度(20日累积, 仅上榜样本)',
        'category': 'lhb_divergence',
        'hypothesis': '上榜日三路资金方向不一致(有买有卖)的股票, 后续收益显著低于三路方向一致的股票 —— 分歧是"缺乏共识"的信号',
        'logic': '三路各取符号 sign(净额), 分歧度 = 1 - |Σsign|/3 ∈ [0, 1]: 三路同向=0, 两同一异=1/3, 一买一卖互斥=2/3~1。只在上榜日累积 (未上榜日不计入分子也不计分母) ⇒ 度量的纯是"有多分歧"而非"上没上榜"。取负向 (分歧高=看空)。',
        'formula': """_n = lhb_north_net_buy.fillna(0.0)
_i = lhb_inst_net_buy.fillna(0.0)
_y = lhb_yz_net_buy.fillna(0.0)
_on = (lhb_flag.fillna(0.0) > 0) | (_n != 0) | (_i != 0) | (_y != 0)
_sgn = np.sign(_n) + np.sign(_i) + np.sign(_y)
_div = (1.0 - (_sgn.abs() / 3.0)).where(_on, 0.0)
factor = -_div.rolling(20).mean().rank(pct=True, axis=1).fillna(0.5)""",
        'direction': 'negative',
        'source': 'P-20260918-001',
    },
    {
        'id': 'novel_238',
        'name': 'lhb_3way_concentration_20d',
        'label': '龙虎榜三路资金单路主导度(20日均值, 仅上榜样本)',
        'category': 'lhb_divergence',
        'hypothesis': '上榜日由单一资金路径主导(其余两路几乎无参与)的股票, 后续收益低于三路均衡参与的股票 —— 单路主导多为事件性炒作, 缺乏多源背书',
        'logic': '主导度 = max(|n|,|i|,|y|) / (|n|+|i|+|y|) ∈ [1/3, 1]: 三路完全均衡=1/3, 单路独揽=1。与 novel_237 正交 —— 237 看**方向**分歧, 238 看**量能**集中度 (三路可能同向但量能全在游资一路)。仅上榜样本。',
        'formula': """_n = lhb_north_net_buy.fillna(0.0).abs()
_i = lhb_inst_net_buy.fillna(0.0).abs()
_y = lhb_yz_net_buy.fillna(0.0).abs()
_tot = _n + _i + _y
_on = _tot > 0
_conc = np.maximum(np.maximum(_n, _i), _y) / _tot.replace(0.0, np.nan)
_conc = _conc.where(_on, np.nan)
factor = (-_conc).rolling(20, min_periods=1).mean().rank(pct=True, axis=1).fillna(0.5)""",
        'direction': 'negative',
        'source': 'P-20260918-001',
    },
    {
        'id': 'novel_239',
        'name': 'lhb_yz_vs_smart_20d',
        'label': '龙虎榜游资相对机构+北向净额差(20日累积, 仅上榜样本)',
        'category': 'lhb_divergence',
        'hypothesis': '上榜日游资净买入相对「机构+北向」更强(游资主导)的股票, 后续收益低于机构+北向主导的股票 —— 游资资金更短视',
        'logic': '定义 _smart = _i + _n (机构+北向=信息型), _hot = _y (游资=情绪型), 差值 _hot - _smart 只在上榜日累积。这是**分路归因**而非 net 合计: 同样是净买入 1 亿, 由机构买入 vs 由游资买入含义不同。novel_234 的合力强度把两路人当成同质的, 本因子做异质拆分。',
        'formula': """_n = lhb_north_net_buy.fillna(0.0)
_i = lhb_inst_net_buy.fillna(0.0)
_y = lhb_yz_net_buy.fillna(0.0)
_on = (lhb_flag.fillna(0.0) > 0) | (_n != 0) | (_i != 0) | (_y != 0)
_hot = (_y - (_i + _n)).where(_on, 0.0)
factor = (-_hot.rolling(20).sum()).rank(pct=True, axis=1).fillna(0.5)""",
        'direction': 'negative',
        'source': 'P-20260918-001',
    },
    {
        'id': 'novel_240',
        'name': 'lhb_smart_agree_hot_dissent_20d',
        'label': '龙虎榜「机构+北向同向 / 游资反向」分歧(20日频次)',
        'category': 'lhb_divergence',
        'hypothesis': '机构与北向同向、游资反向的股票, 后续收益高于三者同向的股票 —— 信息型资金与情绪型资金对立时, 跟信息型资金胜率更高',
        'logic': '典型「聪明钱 vs 游资」对立场景。_smart_ag = (sign(_i)==sign(_n)) & (_i!=0) & (_n!=0); _hot_opp = sign(_y) == -sign(_i)。同时满足且 _smart_ag 方向为正(买入)则 +1, 为负(卖出)则 -1 ⇒ 20日累积。这是方向**结构**条件, 而非金额大小, 与金额类因子正交。',
        'formula': """_n = lhb_north_net_buy.fillna(0.0)
_i = lhb_inst_net_buy.fillna(0.0)
_y = lhb_yz_net_buy.fillna(0.0)
_sa = (np.sign(_i) == np.sign(_n)) & (_i != 0) & (_n != 0)
_opp = (_y != 0) & (np.sign(_y) != np.sign(_i))
_hit = (_sa & _opp).astype(float) * np.sign(_i)
factor = _hit.rolling(20).sum().rank(pct=True, axis=1).fillna(0.5)""",
        'direction': 'long',
        'source': 'P-20260918-001',
    },
]



# ============================================================
# Part C: 核心逻辑
# ============================================================

def compute_ic_icir(factor_values, forward_returns):
    """计算截面 Rank IC 和 ICIR"""
    ic_series = []
    common_dates = factor_values.index.intersection(forward_returns.index)

    for dt in sorted(common_dates):
        fv = factor_values.loc[dt].dropna()
        fr = forward_returns.loc[dt].dropna()
        common = fv.index.intersection(fr.index)
        if len(common) < 30:
            continue
        ic = pd.Series(fv[common]).rank().corr(pd.Series(fr[common]).rank())
        if not np.isnan(ic):
            ic_series.append((dt, ic))

    n_panel = len(common_dates)
    if not ic_series:
        return {'ic_mean': np.nan, 'ic_std': np.nan, 'icir': np.nan,
                'icir_raw': np.nan, '+ic%': np.nan,
                'ic_series': pd.Series(dtype=float),
                'n_dates': 0, 'n_panel_dates': n_panel,
                'n_dates_enough': False, 'n_dates_required': 0}

    ic_vals = pd.Series([x[1] for x in ic_series],
                        index=[x[0] for x in ic_series])
    ic_mean = ic_vals.mean()
    ic_std = ic_vals.std()
    icir_raw = abs(ic_mean / ic_std) if ic_std > 0 else np.nan

    # ── v0.7.29 跨截面样本量门槛 (防稀疏截面假 PASS) ──
    #   不达标 → icir 置 NaN ⇒ 下游全部 5 处 `icir >= ICIR_THRESHOLD` 判定自动收紧,
    #   无需改裁决点(零遗漏); icir_raw / n_dates_enough 保留供诊断与审计。
    _enough, _need = _ic_dates_gate(len(ic_series), n_panel)

    return {
        'ic_mean': ic_mean, 'ic_std': ic_std,
        'icir': icir_raw if _enough else np.nan,
        'icir_raw': icir_raw,
        'n_dates': len(ic_series), 'n_panel_dates': n_panel,
        'n_dates_enough': _enough, 'n_dates_required': _need,
        '+ic%': (ic_vals > 0).mean(), 'ic_series': ic_vals,
    }


def load_data(n_days=270):
    """加载本地日频数据，构建 OHLCV 面板 + 扩展数据 (基本面/订单流)"""
    local_path = PROJECT_ROOT / 'data' / 'raw' / 'daily_prices.csv'
    today = datetime.now()
    end_dt = pd.Timestamp(today.strftime('%Y-%m-%d'))
    start_dt = pd.Timestamp((today - timedelta(days=n_days)).strftime('%Y-%m-%d'))

    daily = pd.read_csv(
        local_path,
        dtype={'ts_code': str, 'trade_date': str},
        usecols=['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'vol', 'amount']
    )
    # 统一日期格式: CSV中混有YYYYMMDD和YYYY-MM-DD两种格式
    daily['trade_date'] = pd.to_datetime(daily['trade_date'], format='mixed')
    daily = daily[(daily['trade_date'] >= start_dt) & (daily['trade_date'] <= end_dt)]
    daily = daily.drop_duplicates(subset=['ts_code', 'trade_date'])
    daily = daily.sort_values(['ts_code', 'trade_date'])

    close_df = daily.pivot(index='trade_date', columns='ts_code', values='close')
    open_df = daily.pivot(index='trade_date', columns='ts_code', values='open')
    high_df = daily.pivot(index='trade_date', columns='ts_code', values='high')
    low_df = daily.pivot(index='trade_date', columns='ts_code', values='low')
    vol_df = daily.pivot(index='trade_date', columns='ts_code', values='vol')
    amount_df = daily.pivot(index='trade_date', columns='ts_code', values='amount')
    pre_close_df = daily.pivot(index='trade_date', columns='ts_code', values='pre_close')

    min_days = 40
    valid_codes = close_df.columns[close_df.count() >= min_days]
    for nm in ['close_df', 'open_df', 'high_df', 'low_df', 'vol_df', 'amount_df',
               'pre_close_df']:
        locals()[nm] = locals()[nm][valid_codes]

    daily_index = close_df.index
    daily_cols = list(valid_codes)

    print(f"  数据: {len(daily)} 条日线, {len(valid_codes)} 只有效股票 (≥{min_days}日)")

    # 加载扩展数据 (基本面 + 订单流)
    extra = {}
    # P-20260815-002: 成交额接入 (交易行为双因子需要 amount 算 VWAP/大单占比)
    extra['amount'] = amount_df
    # P-20260906-003: pre_close 接入 (涨停判定 close/pre_close - 1 用, 未复权口径)
    extra['pre_close'] = pre_close_df
    # P-20260907-007: 衍生字段接入 (与 FactorICComputer 派生口径一致)
    # 修复 S1 过 → Stage2 NameError: FD 翻译白名单含 returns/amplitude 等, 但
    # load_data 此前不提供 → 9 条 fd 因子 (含 fd_skewness S1 ICIR 0.7415) 静默错误。
    # 语义与 factor_ic_computer._load_data 派生字段完全一致 (wide 口径):
    extra['returns'] = close_df.pct_change()
    extra['overnight'] = open_df / close_df.shift(1) - 1
    extra['amplitude'] = (high_df - low_df) / close_df.shift(1)
    extra['turnover'] = vol_df / close_df.clip(lower=0.01)
    extra['hl_ratio'] = high_df / low_df.clip(lower=0.01)
    if _EXT_DATA_AVAILABLE:
        try:
            roe_df, eps_df = load_fundamental_wide(daily_index, daily_cols)
            if roe_df is not None:
                extra['roe'] = roe_df
                extra['eps'] = eps_df
                # 计算 eps_qoq (quarter-over-quarter EPS change)
                extra['eps_qoq'] = eps_df.diff(1)
        except Exception as e:
            print(f'  [基本面] 加载失败: {e}')

        try:
            mf_data = load_moneyflow_wide(daily_index, daily_cols)
            if mf_data[0] is not None:
                extra['buy_sm_vol'] = mf_data[0]
                extra['sell_sm_vol'] = mf_data[1]
                extra['buy_lg_vol'] = mf_data[2]
                extra['sell_lg_vol'] = mf_data[3]
                extra['buy_elg_vol'] = mf_data[4]
                extra['sell_elg_vol'] = mf_data[5]
                extra['net_mf_amount'] = mf_data[6]
        except Exception as e:
            print(f'  [订单流] 加载失败: {e}')

        try:
            mg_data = load_margin_wide(daily_index, daily_cols)
            if mg_data[0] is not None:
                extra['rzye'] = mg_data[0]
                extra['rqye'] = mg_data[1]
                extra['rzmre'] = mg_data[2]
                extra['rqyl'] = mg_data[3]
                extra['rqchl'] = mg_data[4]
                # P-20260911-margin (2026-09-11): 新增 3 列 (融资偿还额/融券卖出量/两融余额)
                if len(mg_data) > 7:
                    extra['rzche'] = mg_data[5]
                    extra['rqmcl'] = mg_data[6]
                    extra['rzrqye'] = mg_data[7]
        except Exception as e:
            print(f'  [两融] 加载失败: {e}')

        # P-20260812-026: 龙虎榜字段接入 (未上榜=0, 公式用 rolling 聚合)
        try:
            tl_data = load_top_list_wide(daily_index, daily_cols)
            if tl_data[0] is not None:
                extra['lhb_flag'] = tl_data[0]
                extra['lhb_net_amount'] = tl_data[1]
                extra['lhb_net_rate'] = tl_data[2]
                extra['lhb_amount'] = tl_data[3]
                extra['lhb_inst_net_buy'] = tl_data[4]
                extra['lhb_inst_buy'] = tl_data[5]
                extra['lhb_inst_sell'] = tl_data[6]
                # P-20260823-001: 北向专用席位 (深股通/沪股通专用)
                if len(tl_data) > 7:
                    extra['lhb_north_net_buy'] = tl_data[7]
                    extra['lhb_north_buy'] = tl_data[8]
                    extra['lhb_north_sell'] = tl_data[9]
                # P-20260904-004: 游资营业部 (排除机构专用/北向专用后的营业部)
                if len(tl_data) > 10:
                    extra['lhb_yz_net_buy'] = tl_data[10]
                    extra['lhb_yz_buy'] = tl_data[11]
                    extra['lhb_yz_sell'] = tl_data[12]
        except Exception as e:
            print(f'  [龙虎榜] 加载失败: {e}')

        # P-20260814-002: 北向持股接入 (2018-2024 历史窗口, 近端 NaN)
        try:
            nh_data = load_hk_hold_wide(daily_index, daily_cols)
            if nh_data[0] is not None:
                extra['north_vol'] = nh_data[0]
                extra['north_ratio'] = nh_data[1]
        except Exception as e:
            print(f'  [北向] 加载失败: {e}')

        # P-20260819-002: 分析师预测分歧接入 (report_rc 中间表)
        try:
            an_data = load_analyst_forecast_wide(daily_index, daily_cols)
            if an_data[0] is not None:
                extra['eps_disp'] = an_data[0]
                extra['tp_disp'] = an_data[1]
                extra['rating_disp'] = an_data[2]
                extra['n_cover'] = an_data[3]
        except Exception as e:
            print(f'  [分析师] 加载失败: {e}')

        # P-20260827-002: 财务成长质量接入 (netprofit_yoy/roe_dt/tr_yoy, ann_date asof 无前视)
        try:
            fina_data = load_fina_growth_wide(daily_index, daily_cols)
            if fina_data[0] is not None:
                extra['netprofit_yoy'] = fina_data[0]
                extra['roe_dt'] = fina_data[1]
                extra['tr_yoy'] = fina_data[2]
        except Exception as e:
            print(f'  [财务成长] 加载失败: {e}')

        # P-20260827-004 + P-20260911-kd: 估值宽表 + 行业 L1 代码接入
        # P-20260911-kd 起 load_valuation_wide 返回 7 字段 (日频直出, 原月末快照会
        # 让同月内因子值恒定 → 高估 IC 自相关); 解包必须同步扩展。
        try:
            val_data = load_valuation_wide(daily_index, daily_cols)
            if val_data[0] is not None:
                for nm, w in zip(['circ_mv', 'total_mv', 'pb', 'ps_ttm', 'pe_ttm',
                                  'dv_ratio', 'dv_ttm', 'kd_ps_pct_own'], val_data):
                    if w is not None:
                        extra[nm] = w
        except Exception as e:
            print(f'  [估值] 加载失败: {e}')
        try:
            ind_data = load_industry_l1_wide(daily_index, daily_cols)
            if ind_data[0] is not None:
                extra['industry_l1'] = ind_data[0]
        except Exception as e:
            print(f'  [行业] 加载失败: {e}')

        # P-20260911-kd: 快刀斩股基本面通道 (2026-09-11 接入)
        # 资产负债 / 利润TTM / 现金流TTM(收现比·净现比·资本开支比·FCF) / 扣非 /
        # 分红 / 大股东增减持 / 股东户数 / 前十大合计 / 回购。
        # 数据源 data/raw/kuaidao_gaps/; 季频先在季层做 TTM 再 ann_date asof(+1日)。
        try:
            kd_extra = load_kuaidao_fund_wide(daily_index, daily_cols)
            if kd_extra:
                extra.update(kd_extra)
        except Exception as e:
            print(f'  [快刀基本面] 加载失败: {e}')

        # P-20260819-003: 申万 L1 行业 peer 收益接入 (Salience DS)
        try:
            ind_peer = load_industry_wide(daily_index, daily_cols)
            if ind_peer is not None:
                extra['industry_ret_peer'] = ind_peer
        except Exception as e:
            print(f'  [行业] 加载失败: {e}')

        # P-20260906-001: 任意量价特征行业均值宽表 (双重同伴效应分解)
        try:
            fp_data = load_feature_peer_wide(daily_index, daily_cols)
            if fp_data[0] is not None:
                extra['ind_avg_amount'] = fp_data[0]
                extra['ind_avg_turnover'] = fp_data[1]
                extra['ind_avg_amplitude'] = fp_data[2]
                extra['ind_avg_returns'] = fp_data[3]
        except Exception as e:
            print(f'  [行业peer] 加载失败: {e}')

        # P-20260906-003: 涨停生态市场级指标 (涨停家数/开板率/溢价, 截面广播)
        try:
            lu_data = load_limit_up_wide(daily_index, daily_cols)
            if lu_data[0] is not None:
                extra['limit_up_count'] = lu_data[0]
                extra['limit_up_open_rate'] = lu_data[1]
                extra['limit_up_premium'] = lu_data[2]
        except Exception as e:
            print(f'  [涨停生态] 加载失败: {e}')

        # P-20260915-lion: 狮子锚点 (最近涨停日→启动点结构锚点, 后复权比值口径)
        # 来源: 聚宽社区《变种小狮子 V3.0》bias 构造, 审计见
        # reports/lion_v3_strategy_audit_20260915.md
        try:
            lion_data = load_lion_anchor_wide(daily_index, daily_cols)
            if lion_data[0] is not None:
                extra['lion_bias'] = lion_data[0]
                extra['lion_lu_days'] = lion_data[1]
                extra['lion_lu_cnt'] = lion_data[2]
        except Exception as e:
            print(f'  [狮子锚点] 加载失败: {e}')
    else:
        print('  [扩展数据] 不可用 (data_loader_ext 未安装)')

    return close_df, open_df, high_df, low_df, vol_df, extra


def select_novel_hypotheses(n=4, llm_slot=False, stage1_proposals=None):
    """
    从因子来源中选择尚未测试的因子。
    保证: 每个因子与 FA/因子池/历史已测 全部不同。

    来源优先级 (库存耗尽时 Stage1 提案成为主驱动):
      1) Stage 1 扩充提案 (stage1_exploration) — 最多填满 n
      2) LLM 槽位 (llm_generated, 每日策略研究产出) — 最多 1 个
      3) 原创因子图书馆 (novel_library) — 补足到 n
    返回: List[dict] — n 个全新因子假设
    """
    blocked = get_blocked_names()
    available = [f for f in NOVEL_FACTOR_LIBRARY if f['name'] not in blocked]

    stage1_proposals = stage1_proposals or []
    stage1_avail = [p for p in stage1_proposals
                    if p.get('name') and p['name'] not in blocked]

    selected = []

    # 1) Stage 1 扩充提案优先 (填满 n)
    n_stage1 = min(n, len(stage1_avail))
    for p in stage1_avail[:n_stage1]:
        pp = dict(p)  # 不污染原始提案
        pp['source'] = 'stage1_exploration'
        selected.append(pp)
    if n_stage1:
        print(f"  🔬 Stage1扩充提案: 选用 {n_stage1} 个")

    # 2) LLM 槽位 (最多1个, 未用满时)
    llm_factors = []
    if llm_slot and LLM_FACTOR_REGISTRY and len(selected) < n:
        for llm_f in LLM_FACTOR_REGISTRY:
            if llm_f['name'] not in blocked:
                llm_pp = dict(llm_f)
                llm_pp['source'] = 'llm_generated'
                llm_factors.append(llm_pp)
                break
        if llm_factors:
            print(f"  🤖 LLM槽位: {llm_factors[0]['label']} (策略研究产出)")

    # 3) 原创图书馆补足到 n
    remaining = n - len(selected) - len(llm_factors)
    if remaining > 0 and available:
        today = datetime.now()
        weekday = today.weekday()
        categories_in_order = sorted(set(f['category'] for f in available))
        primary_cat = categories_in_order[weekday % len(categories_in_order)]
        primary_available = [f for f in available if f['category'] == primary_cat]
        other_available = [f for f in available if f['category'] != primary_cat]
        n_primary = min(max(remaining // 2 + 1, 1), len(primary_available)) if primary_available else 0
        n_other = remaining - n_primary
        novel_selected = random.sample(primary_available, n_primary) if primary_available and n_primary > 0 else []
        if other_available and n_other > 0:
            novel_selected += random.sample(other_available, min(n_other, len(other_available)))
        seen = set(f['name'] for f in selected + llm_factors)
        for f in novel_selected:
            ff = dict(f); ff['source'] = 'novel_library'; selected.append(ff); seen.add(f['name'])
        for f in available:
            if len(selected) >= n:
                break
            if f['name'] not in seen:
                ff = dict(f); ff['source'] = 'novel_library'; selected.append(ff); seen.add(f['name'])

    combined = (selected + llm_factors)[:n]
    if not combined:
        print("  ⚠️  所有来源已耗尽! 无可用因子 (请 Stage 1 扩充提案或扩充 NOVEL_FACTOR_LIBRARY)")
        return []
    print(f"  测试因子 ({len(combined)}): {[f.get('source','?')[:4]+':'+f['label'] for f in combined]}")
    return combined


def _compute_rre(factor_z, lookback=60, n_bins=10):
    """P-20260815-005: 相对秩熵 RRE — 因子截面分布形态的时间稳定性 (纯监控, 不干预)。

    RRE = 1 / (1 + mean KL(S_t || S_{t-1}))
      S_t = t 日截面 z 值在固定边界 [-3.5, 3.5] 上的等距直方图
      分布形态逐日漂移越大 → KL 越大 → RRE 越低 (排序结构不稳定)
      AlphaEval 框架: RRE 量化因子排序稳定性, 低于 0.3 提示短寿风险。

    返回 (rre: float, kl_mean: float) — NaN 表示数据不足无法计算。
    """
    if factor_z is None or len(factor_z) < 10:
        return np.nan, np.nan
    z = factor_z.iloc[-lookback:]
    z = z.dropna(how="all")
    lo, hi = -3.5, 3.5
    bins = np.linspace(lo, hi, n_bins + 1)  # z 值固定边界分箱 (形态分布)
    kl_list = []
    dates = list(z.index)
    for i in range(1, len(dates)):
        prev = z.loc[dates[i - 1]].dropna().clip(lo, hi)
        curr = z.loc[dates[i]].dropna().clip(lo, hi)
        common = prev.index.intersection(curr.index)
        if len(common) < 50:
            continue
        p = pd.cut(prev[common], bins, labels=False).value_counts().sort_index().reindex(
            range(n_bins), fill_value=0).astype(float)
        q = pd.cut(curr[common], bins, labels=False).value_counts().sort_index().reindex(
            range(n_bins), fill_value=0).astype(float)
        p = (p + 1e-6)
        q = (q + 1e-6)
        p = p / p.sum()
        q = q / q.sum()
        kl_list.append(float((p * np.log(p / q)).sum()))
    if not kl_list:
        return np.nan, np.nan
    kl_mean = float(np.mean(kl_list))
    return 1.0 / (1.0 + kl_mean), kl_mean


def evaluate_factor(hyp, close_df, open_df, high_df, low_df, vol_df, forward_5d, extra=None):
    """评估单个因子的截面IC。extra: dict of {name: wide DataFrame} — 基本面/订单流扩展数据"""
    try:
        extra = extra or {}
        close_p = close_df
        open_p = open_df
        high_p = high_df
        low_p = low_df
        volume_p = vol_df

        # ── v0.9.4 rank 语义修正 (2026-08-21 rmb 因子审计) ──
        # 宽表 DataFrame 上裸 .rank(pct=True) 默认 axis=0 = 沿时间轴排名,
        # 用整个回看窗口(含未来)计算每天分位 → 前视偏差。
        # 实证: rmb_appreciation_sensitive_spread 裸 rank IC=0.1625/ICIR=1.526 假阳性,
        # 修正 axis=1 后 IC=0.0215/ICIR=0.174。横截面因子语义必须 axis=1。
        # 例外: .rolling(N).rank(pct=True) (Rolling.rank) 是窗口内时间序列分位, 保持原样。
        # v0.9.5: 负向后顾变长 \d+ 会让 re.compile 抛异常 (定宽限制),
        # 改为「先保护 rolling(N).rank 占位 → 裸 rank 改 axis=1 → 还原」三段式。
        import re as _re
        _formula_src = hyp.get('formula', '')
        _roll_rank_re = _re.compile(r'\.rolling\((\d+)\)\.rank\(pct\s*=\s*True\)')
        _plain_rank_re = _re.compile(r'\.rank\(pct\s*=\s*True\)')
        _holders = []
        _formula_fixed = _roll_rank_re.sub(
            lambda m: _holders.append(m.group(0)) or f'.rolling({m.group(1)}).__RRK{len(_holders)-1}__',
            _formula_src)
        _formula_fixed = _plain_rank_re.sub('.rank(pct=True, axis=1)', _formula_fixed)
        for _i, _h in enumerate(_holders):
            _formula_fixed = _formula_fixed.replace(f'__RRK{_i}__', 'rank(pct=True)')
        if _formula_fixed != _formula_src:
            hyp = dict(hyp, formula=_formula_fixed)
        # 构造 df dict (兼容 formula_pandas 的 df["col"] 语法) — 含扩展数据
        df_dict = {'close': close_p, 'open': open_p, 'high': high_p,
                   'low': low_p, 'volume': volume_p}
        # 添加扩展数据到 df dict
        for key, df_val in extra.items():
            if df_val is not None:
                df_dict[key] = df_val

        # 安全执行因子公式
        # ⚠️ 'open' 是 Python 内置函数, 必须显式覆盖才能让公式中的裸 `open` 解析为 DataFrame
        # eval()/exec() 中的名字解析顺序: locals → globals → builtins
        # 将 'open' (不带下划线) 放入 local_ns 即可在 eval/exec 上下文中遮蔽内置 open()
        local_ns = {
            'close_p': close_p, 'open_p': open_p, 'high_p': high_p,
            'low_p': low_p, 'volume_p': volume_p, 'np': np, 'pd': pd,
            # Stage1提案兼容别名 — 支持裸名 (close/high/low/open/volume) 与 _p 后缀
            'close': close_p, 'open': open_p, 'open_': open_p,
            'high': high_p, 'low': low_p, 'volume': volume_p,
            # 扩展数据直接注入
            **{k: v for k, v in extra.items() if v is not None},
            # df dict 兼容
            'df': df_dict,
        }
        try:
            # 先尝试单表达式求值 (兼容 NOVEL/LLM 库原有单行公式)
            factor_raw = eval(compile(hyp['formula'], '<factor>', 'eval'), local_ns)
        except SyntaxError:
            # 多行赋值块: exec 后取 factor/result 变量 (兼容不同提案命名)
            exec(compile(hyp['formula'], '<factor>', 'exec'), local_ns)
            factor_raw = local_ns.get('factor')
            if factor_raw is None:
                factor_raw = local_ns.get('result')
            if factor_raw is None:
                raise ValueError("公式未定义 factor 或 result 变量")

        # 缩尾 (1%/99%)
        for col in factor_raw.columns:
            col_data = factor_raw[col].dropna()
            if len(col_data) > 50:
                lo, hi = col_data.quantile(0.01), col_data.quantile(0.99)
                factor_raw[col] = factor_raw[col].clip(lo, hi)

        # 截面标准化 (z-score)
        factor_z = factor_raw.subtract(factor_raw.mean(axis=1), axis=0) \
                             .divide(factor_raw.std(axis=1), axis=0)

        result = compute_ic_icir(factor_z, forward_5d)

        # P-20260815-005: RRE 时间稳定性监控 (纯提醒, 不降级不过滤)
        rre_val, rre_kl = _compute_rre(factor_z)
        result['rre_60d'] = rre_val
        if not np.isnan(rre_val):
            rre_flag = " ⚠️ <0.9 分布形态剧变" if rre_val < 0.9 else ""
            print(f"    [RRE监控] 排序稳定性 RRE={rre_val:.3f} (KL={rre_kl:.4f}){rre_flag}")

        return {**hyp, **result, 'n_dates': len(result.get('ic_series', []))}

    except Exception as e:
        return {**hyp, 'error': str(e)}


def mark_as_tested(factor_name, factor_info):
    """标记因子为已测（无论通过与否）"""
    tested_dict, tested_names = load_tested_history()
    tested_names = set(tested_names)  # 确保是set
    factor_info['tested_date'] = datetime.now().strftime('%Y-%m-%d')
    factor_info['tested_time'] = datetime.now().isoformat()
    tested_dict[factor_name] = factor_info
    tested_names.add(factor_name)
    save_tested_history(tested_dict, tested_names)


def add_to_pool(factor_info, passed=False):
    """将所有测试因子加入因子储备池。passed=True→candidate, False→reserve。返回是否为新因子"""
    new_entry = {
        'name': factor_info['name'],
        'label': factor_info['label'],
        'category': factor_info.get('category', 'novel'),
        'date': datetime.now().strftime('%Y-%m-%d'),
        'icir': factor_info['icir'],
        'ic_mean': factor_info['ic_mean'],
        '+ic_pct': factor_info['+ic%'],
        'decay_lambda': factor_info.get('decay_lambda', None),  # XQuant: α(t)=K/(1+λt), 需多窗口数据
        'hypothesis': factor_info['hypothesis'],
        'logic': factor_info['logic'],
        'formula': factor_info['formula'],
        'direction': factor_info.get('direction', 'long'),
        'status': 'candidate' if passed else 'reserve',
        'round': 'daily',
        'source': factor_info.get('source', 'novel_library'),
    }

    if POOL_CSV.exists():
        existing = pd.read_csv(POOL_CSV)
        if factor_info['name'] in existing['name'].values:
            print(f"    [INFO] {factor_info['name']} 已在因子池中, 更新记录")
            # 2026-09-07 修复: 复评/校正重跑时 status 必须同步更新
            # (candidate↔reserve), 旧逻辑只更新 IC 数值 → 校正后 PASS 仍被记 reserve。
            existing.loc[existing['name'] == factor_info['name'],
                        ['icir', 'ic_mean', '+ic_pct', 'date', 'status']] = [
                factor_info['icir'], factor_info['ic_mean'],
                factor_info['+ic%'], new_entry['date'], new_entry['status']
            ]
            existing.to_csv(POOL_CSV, index=False, encoding='utf-8-sig')
            return False
        existing = pd.concat([existing, pd.DataFrame([new_entry])], ignore_index=True)
    else:
        existing = pd.DataFrame([new_entry])

    existing.to_csv(POOL_CSV, index=False, encoding='utf-8-sig')
    print(f"    [POOL] {factor_info['name']} 已加入因子池 ({POOL_CSV})")
    return True


def run_daily(llm_slot=False, require_stage1=False, n=20):
    """执行日频因子假设实验 — 原创去重版 + LLM槽位

    require_stage1=True 时, 开头先检查当日 Stage 1 策略研究报告是否存在;
    若不存在则闸门拦截, 直接返回 (None, 0), 不消耗因子库存、不写报告。
    """
    today = datetime.now()
    date_str = today.strftime('%Y%m%d')
    version_tag = 'v3 (LLM slot)' if llm_slot and LLM_FACTOR_REGISTRY else 'v2'

    print("=" * 70)
    print(f"  每日因子假设实验 {version_tag} — {today.strftime('%Y-%m-%d')} ({today.strftime('%A')})")
    print(f"  原创去重: FA 71因子 + 因子池 + 已测历史 = 全量排除")
    if llm_slot:
        print(f"  🤖 LLM因子槽位: 已启用 ({len(LLM_FACTOR_REGISTRY)} 个LLM因子待测)")
    print("=" * 70)

    # ---- P-20260903-001: natural_freq 审计前置 (铁证才修正, 失败不阻塞) ----
    try:
        import sys as _sys
        from pathlib import Path as _P
        _fa_dir = str(_P(__file__).resolve().parents[1] / "research" / "factor_alchemy")
        if _fa_dir not in _sys.path:
            _sys.path.insert(0, _fa_dir)
        from freq_audit import run_freq_audit
        run_freq_audit(verbose=True)
    except Exception:
        pass

    # ---- 前置闸门: 当日 Stage 1 报告 ----
    if not check_stage1_gate(require_stage1, today):
        return None, 0

    # ---- Step 0: 去重信息 ----
    print("\n[Step 0] 去重检查...")
    blocked = get_blocked_names()
    available_count = len([f for f in NOVEL_FACTOR_LIBRARY if f['name'] not in blocked])
    stage1_proposals = load_stage1_proposals()
    print(f"  已占用: {len(blocked)} 个 (FA {len(load_fa_factor_names())} + "
          f"因子池 {len(load_pool_names())} + 已测 {len(load_tested_history()[1])})")
    print(f"  可测原创: {available_count}/{len(NOVEL_FACTOR_LIBRARY)}")
    print(f"  🔬 Stage1扩充提案待验证: {len(stage1_proposals)} 个")

    # ---- Step 1: 加载数据 ----
    print("\n[Step 1] 加载数据...")
    close_df, open_df, high_df, low_df, vol_df, extra = load_data(n_days=270)
    forward_5d = close_df.shift(-5) / close_df - 1

    # ---- Step 2: 选择未测因子 (含 Stage1 提案) ----
    print("\n[Step 2] 选择未测因子 (Stage1提案优先)...")
    hypotheses = select_novel_hypotheses(n=n, llm_slot=llm_slot,
                                         stage1_proposals=stage1_proposals)

    if not hypotheses:
        print("  ❌ 无可测的原创因子，请扩充 NOVEL_FACTOR_LIBRARY")
        return [], 0

    # 🆕 P-20260910-002: 可证伪四元组补齐 (NOVEL/LLM 库因子缺 refutation_conditions 自动补默认)
    _fals_hypotheses = []
    for h in hypotheses:
        h2, _ok = _ensure_falsifiable(h, source=h.get('source', 'novel'))
        if _ok:
            _fals_hypotheses.append(h2)
    hypotheses = _fals_hypotheses
    if not hypotheses:
        print("  ❌ 可证伪门拦截后无可测因子 (FUXI_FALSIFIABLE_GATE=enforce)")
        return [], 0

    # ---- Step 3: IC评估 ----
    print(f"\n[Step 3] IC评估 ({len(hypotheses)} 个全新因子)...")
    results = []
    for i, hyp in enumerate(hypotheses):
        print(f"\n  [{i+1}/{len(hypotheses)}] {hyp['label']} — {hyp['hypothesis']}")
        print(f"  逻辑: {hyp['logic'][:100]}...")

        result = evaluate_factor(hyp, close_df, open_df, high_df, low_df, vol_df, forward_5d, extra=extra)
        results.append(result)

        # 🆕 P-20260910-002: 方向证伪比对 (predicted_direction vs 本地实证 IC 符号)
        pred_dir = str(hyp.get('predicted_direction', hyp.get('direction', ''))).strip().lower()
        sign = _DIRECTION_SIGN.get(pred_dir)
        ic_mean = result.get('ic_mean')
        if sign is not None and 'error' not in result and not np.isnan(ic_mean):
            if ic_mean * sign < 0:
                result['direction_falsified'] = True
                print(f"    🔄 [方向证伪] 预测 {pred_dir} (IC 预期 {'+' if sign > 0 else '-'}) "
                      f"vs 实证 IC {ic_mean:+.4f} → 方向被证伪, 以实证符号为准")
                _record_falsification_event(hyp['name'], pred_dir, ic_mean,
                                            result.get('icir', np.nan), date_str)
            else:
                result['direction_falsified'] = False
        else:
            result['direction_falsified'] = None

        # 标记已测 (无论通过与否)
        mark_as_tested(hyp['name'], {
            'label': hyp['label'],
            'category': hyp.get('category', ''),
            'icir': result.get('icir', np.nan),
            'ic_mean': result.get('ic_mean', np.nan),
            '+ic_pct': result.get('+ic%', np.nan),
            'n_dates': result.get('n_dates', 0),
            'source': hyp.get('source', 'novel_library'),
        })
        # 若来自 Stage1 提案: 通过者保留条目 (供 Step5.5 回写 PASS → Stage3 桥接消费),
        # 未通过者剪枝 (状态移交 tested_factors.json 管理)
        # 修复 P-018: 旧逻辑在 passed 判定前剪枝 → PASS 提案被删 → 回写落空 → 桥接 miss

        if 'error' in result:
            print(f"    [ERROR] {result['error']}")
            prune_stage1_proposals(hyp['name'])
            continue

        passed = (not np.isnan(result['icir']) and
                 result['icir'] >= ICIR_THRESHOLD and
                 result['+ic%'] >= IC_PCT_THRESHOLD)

        if not passed:
            prune_stage1_proposals(hyp['name'])

        status = "[PASS] 通过 → 加入因子池 + 触发FA重跑" if passed else "[RESERVE] 未通过 → 加入因子储备池"
        print(f"    IC_mean: {result['ic_mean']:+.4f}  "
              f"ICIR: {result['icir']:+.3f}  "
              f"+IC%: {result['+ic%']:.1%}  "
              f"n={result.get('n_dates', 0)}截面")
        print(f"    结论: {status}")

        # 所有有效因子都入库 (passed→candidate, 未通过→reserve)
        is_new = add_to_pool(result, passed=passed)
        if passed:
            result['fa_trigger'] = is_new

    # ---- Step 4: 汇总 ----
    print("\n" + "=" * 70)
    print("[Step 4] 日频汇总")
    print("=" * 70)

    valid_results = [r for r in results if 'icir' in r and not np.isnan(r['icir'])]

    if valid_results:
        print(f"\n{'假设':24s} {'类别':14s} {'ICIR':>7s} {'+IC%':>7s} {'RRE':>6s} {'结果':>6s}")
        print("-" * 80)
        for r in sorted(valid_results, key=lambda x: -x['icir']):
            passed = r['icir'] >= ICIR_THRESHOLD and r['+ic%'] >= IC_PCT_THRESHOLD
            status = 'PASS' if passed else 'RESERVE'
            rre_s = f"{r['rre_60d']:.2f}" if r.get('rre_60d') is not None and not np.isnan(r.get('rre_60d', np.nan)) else "—"
            print(f"{r['label']:24s} {r.get('category', ''):14s} {r['icir']:+7.3f} "
                  f"{r['+ic%']:6.1%} {rre_s:>6s} {status:>6s}")

        n_passed = sum(1 for r in valid_results
                       if r['icir'] >= ICIR_THRESHOLD and r['+ic%'] >= IC_PCT_THRESHOLD)
        n_reserve = len(valid_results) - n_passed
        n_new_fa = sum(1 for r in valid_results if r.get('fa_trigger', False))
        print(f"\n  通过: {n_passed} / 入库储备: {n_reserve} / 总计: {len(valid_results)}")
        print(f"  需重跑FA: {n_new_fa} 个新因子 (仅通过阈值的candidate因子)")

        if n_passed > 0:
            print(f"\n  🎉 新通过因子!")
            for r in valid_results:
                if r['icir'] >= ICIR_THRESHOLD and r['+ic%'] >= IC_PCT_THRESHOLD:
                    print(f"    ✓ {r['label']} (ICIR={r['icir']:.3f}, +IC%={r['+ic%']:.1%})")
                    print(f"      → 已加入因子池 → 需要注册FA并运行 run_fa.py --full")

            # P-20260908-006: PASS→S5/JQ 自动桥接 (修复入池后桥接悬空)。
            # env FUXI_PASS_JQ_BRIDGE=on 启用 (默认 off 只盘点不生成);
            # 独立 subprocess 失败不阻塞主流程; 幂等 (已在队列/同名文件跳过)。
            try:
                if os.environ.get('FUXI_PASS_JQ_BRIDGE') == 'on':
                    import subprocess
                    bridge = PROJECT_ROOT / 'scripts' / 'pass_to_jq_bridge.py'
                    print(f"\n  [Bridge] FUXI_PASS_JQ_BRIDGE=on → 触发 PASS→S5/JQ 桥接...")
                    r = subprocess.run(
                        [sys.executable, str(bridge), '--execute', '--max-codegen', '10'],
                        capture_output=True, text=True, timeout=1800, cwd=str(PROJECT_ROOT))
                    tail = (r.stdout or '').strip().splitlines()[-8:]
                    for line in tail:
                        print(f"    {line}")
                    if r.returncode != 0:
                        print(f"    [Bridge] 退出码 {r.returncode} (不阻塞主流程)")
            except Exception as _be:
                print(f"  [Bridge] 桥接失败 (不阻塞): {_be}")

    # ---- Step 5: 保存报告 ----
    print("\n[Step 5] 保存报告...")
    report_file = DAILY_REPORT_DIR / f'daily_factor_{date_str}.csv'

    clean_results = []
    for r in results:
        clean_r = {k: v for k, v in r.items() if k != 'ic_series'}
        clean_r['date'] = date_str
        clean_r['weekday'] = today.strftime('%A')
        clean_results.append(clean_r)

    pd.DataFrame(clean_results).to_csv(report_file, index=False, encoding='utf-8-sig')
    print(f"  报告: {report_file}")

    json_file = DAILY_REPORT_DIR / f'daily_factor_{date_str}.json'
    n_passed = sum(1 for r in valid_results
                   if r['icir'] >= ICIR_THRESHOLD and r['+ic%'] >= IC_PCT_THRESHOLD)
    n_reserve = len(valid_results) - n_passed
    n_new_fa = sum(1 for r in valid_results if r.get('fa_trigger', False))
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            'date': date_str,
            'version': 'v2_novel_dedup',
            'n_available': available_count,
            'n_tested_today': len(hypotheses),
            'n_passed': n_passed,
            'n_reserve': n_reserve,
            'n_new_fa': n_new_fa,
            'results': [{k: v for k, v in r.items() if k != 'ic_series'}
                       for r in valid_results],
        }, f, ensure_ascii=False, indent=2)
    print(f"  JSON: {json_file}")

    # ── Step 5.5: 回写测试结果到 stage1_factor_proposals.json (Stage3 桥接) ──
    _sync_test_results_to_proposals(valid_results, date_str)

    # ── Step 5.6: (P-20260925-001) Stage2 PASS → 完整 S1~S5 核验链路 ──
    # 刻意放在日报落盘与提案池回写**之后**: 即便真实 S5 耗时或超时,
    # 日报、回写、MEMORY 均已是安全状态, 不会因 S5 未完成而丢失主产物。
    #   第一次调用(恒做): 刷新「待核验队列」→ 未做 S5 的 PASS 因子在流水线里必然可见;
    #   第二次调用(仅 FUXI_STAGE2_S5_CHAIN=enforce, 历史别名 run): 对队列前 N 条执行真实 S5。
    _run_stage2_s5_chain(execute=False)
    _run_stage2_s5_chain(execute=True)

    # 库存警告
    if available_count - len(hypotheses) < 5:
        print(f"\n  ⚠️  原创因子库存不足! ({available_count - len(hypotheses)} 剩余)")
        print(f"  建议: 扩充 NOVEL_FACTOR_LIBRARY (当前{len(NOVEL_FACTOR_LIBRARY)}个)")

    return results, n_new_fa


# ============================================================
# P-20260925-001: Stage2 PASS → 完整 S1~S5 核验链路
# ============================================================
def _run_stage2_s5_chain(execute: bool = False):
    """把 Stage2 判 PASS 的因子送进完整 S1~S5 核验链路。

    立项背景（2026-09-25 审计）:
      scripts/s135_candidate_check.py 自述「把 status=candidate 的因子重跑完整
      S1-S5 管线」，通道本身**可跑**，但全仓库调用方 **0**（上次运行 2026-09-08
      13:32）⇒ Stage2 PASS 入池后永久滞留 candidate，即「入池即终点」。
      本函数就是那个缺失的调用方。

    参数:
      execute=False → 只刷新待核验队列（--queue-only，零回测成本，秒级）
      execute=True  → 额外对队列前 N 条执行**真实** S5（--limit，成本高）

    开关:
      FUXI_STAGE2_S5_CHAIN = off | annotate(默认) | audit | enforce(历史别名 run)
      FUXI_STAGE2_S5_LIMIT = 单次核验上限（默认 10）

    失败语义:
      子进程异常一律**不阻塞**主流程；真实 S5 未就绪时链路内部 fail-closed
      （退出码 2）并留痕，**绝不静默降级**为 IC 年拆分轻量代理。
    """
    mode = str(os.environ.get('FUXI_STAGE2_S5_CHAIN', '') or '').strip().lower()
    if mode not in ('off', 'annotate', 'audit', 'run', 'enforce'):
        mode = 'annotate'
    if mode == 'off':
        return
    if execute and mode not in ('run', 'enforce'):
        return

    chain = PROJECT_ROOT / 'scripts' / 'stage2_s5_chain.py'
    if not chain.exists():
        print(f"\n  [S5链] 未找到 {chain}（跳过）")
        return

    cmd = [sys.executable, str(chain)]
    cmd += (['--limit', str(os.environ.get('FUXI_STAGE2_S5_LIMIT', '10'))]
            if execute else ['--queue-only'])
    print(f"\n  [S5链] FUXI_STAGE2_S5_CHAIN={mode} → "
          f"{'执行真实 S1~S5' if execute else '刷新待核验队列'}...")
    try:
        import subprocess
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=7200 if execute else 300, cwd=str(PROJECT_ROOT))
        for line in (r.stdout or '').strip().splitlines()[-12:]:
            print(f"    {line}")
        if execute and r.returncode == 2:
            print("    [S5链] S5 引擎未就绪 → 已 fail-closed 留痕（未做轻量降级）")
        elif r.returncode != 0:
            print(f"    [S5链] 退出码 {r.returncode}，不阻塞主流程")
    except Exception as _ce:
        print(f"  [S5链] 触发失败，不阻塞主流程: {_ce}")


# ============================================================
# Stage2 → Stage3 桥接: 回写测试结果
# ============================================================
def _sync_test_results_to_proposals(valid_results: list, date_str: str):
    """将测试结果 (ICIR/+IC%/status) 回写到 stage1_factor_proposals.json。
    
    这是 Stage2→Stage3 桥接的关键步骤:
      Stage2 测试完毕 → 回写结果 → Stage3 桥接脚本读取 → 自动注入
    
    按 factor_name 匹配，覆盖 status/icir/+ic%/tested_at 字段。
    """
    proposals_file = PROJECT_ROOT / 'data' / 'stage1_factor_proposals.json'
    if not proposals_file.exists():
        return
    
    try:
        with open(proposals_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return

    # 兼容两种格式: {'proposals': [...]} 或裸 list（8-23 事故清理后为 list 格式）
    if isinstance(data, dict):
        proposals = data.get('proposals', [])
    elif isinstance(data, list):
        proposals = data
    else:
        return
    if not proposals:
        return
    
    # 建结果索引 (按 factor_name)
    result_map = {}
    for r in valid_results:
        name = r.get('factor_name', r.get('name', ''))
        if not name:
            # 尝试从 label 反推
            label = r.get('label', '')
            for p in proposals:
                if p.get('label', '') == label:
                    name = p.get('factor_name', '')
                    break
        if name:
            result_map[name] = {
                'status': 'PASS' if (r.get('icir', 0) or 0) >= 0.3 and (r.get('+ic%', 0) or 0) >= 0.5 else 'RESERVE',
                'icir': round(float(r.get('icir', 0) or 0), 4),
                'plus_ic_pct': round(float(r.get('+ic%', 0) or 0), 4),
                'ic_mean': round(float(r.get('ic_mean', 0) or 0), 4),
                'tested_at': date_str,
            }
    
    updated = 0
    for p in proposals:
        name = p.get('factor_name', '')
        if name in result_map and not p.get('status'):
            p.update(result_map[name])
            updated += 1
    
    if updated > 0:
        if isinstance(data, dict):
            data['proposals'] = proposals
        else:
            data = proposals
        with open(proposals_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  🔗 回写测试结果到 proposals: {updated} 条 → {proposals_file}")


# ============================================================
# 信息工具
# ============================================================
def show_library_stats():
    """展示因子库统计"""
    blocked = get_blocked_names()
    available = [f for f in NOVEL_FACTOR_LIBRARY if f['name'] not in blocked]

    print("=" * 70)
    print("  原创因子图书馆 — 统计")
    print("=" * 70)
    print(f"  库容量: {len(NOVEL_FACTOR_LIBRARY)} 个因子")
    print(f"  已验证: {len(blocked & set(f['name'] for f in NOVEL_FACTOR_LIBRARY))} 个")
    print(f"  可测: {len(available)} 个")

    print(f"\n  按类别分布:")
    from collections import Counter
    cat_counts = Counter(f['category'] for f in NOVEL_FACTOR_LIBRARY)
    for cat, n in cat_counts.most_common():
        avail = len([f for f in available if f['category'] == cat])
        print(f"    {cat:25s} {avail:3d}/{n:3d} 可用")


if __name__ == '__main__':
    from pathlib import Path as _FuxiPath
    import sys as _FuxiSys
    _FuxiSys.path.insert(0, str(_FuxiPath(__file__).resolve().parents[1]))
    from fuxi_v1.legacy_writers import legacy_cli_guard as _FuxiGuard
    _fuxi_guard = _FuxiGuard(__file__, read_only='--stats' in sys.argv)
    import argparse
    parser = argparse.ArgumentParser(description='每日因子假设实验 v2 (原创去重)')
    parser.add_argument('--stats', action='store_true',
                       help='展示因子库统计')
    parser.add_argument('--n', type=int, default=20,
                       help='每日测试因子数 (默认20, 加速库存消耗)')
    parser.add_argument('--all', action='store_true',
                       help='测试所有未测因子 (慎用! 会耗尽库存)')
    parser.add_argument('--llm-slot', action='store_true',
                       help='预留1个槽位给LLM生成因子 (由每日策略研究产出)')
    parser.add_argument('--llm-add', type=str, nargs='*',
                       help='注册LLM因子: --llm-add name label category hypothesis logic formula direction')
    parser.add_argument('--require-stage1', action='store_true',
                       help='前置闸门: 当日 Stage 1 策略研究报告必须存在, 否则拦截本次运行')

    args = parser.parse_args()

    # 处理LLM因子注册
    if args.llm_add:
        name, label, cat, hyp, logic, formula = args.llm_add[:6]
        direction = args.llm_add[6] if len(args.llm_add) > 6 else 'long'
        LLM_FACTOR_REGISTRY.append({
            'name': name, 'label': label,
            'category': cat, 'hypothesis': hyp,
            'logic': logic, 'formula': formula,
            'direction': direction,
        })
        print(f"  ✅ LLM因子已注册: {label} ({name})")

    if args.stats:
        show_library_stats()
    else:
        results, n_new_fa = run_daily(llm_slot=args.llm_slot,
                                      require_stage1=args.require_stage1,
                                      n=args.n)

        if n_new_fa and n_new_fa > 0:

            print("\n" + "=" * 70)
            print("[ACTION REQUIRED] 有新因子通过! 需要集成到FA系统:")
            print("  1. 在 research/factor_alchemy/factors/advanced_technical.py 添加因子类")
            print("  2. 在 research/factor_alchemy/config.py FACTOR_DEFS 添加条目")
            print("  3. 在 research/factor_alchemy/factors/__init__.py ALL_FACTORS 注册")
            print("  4. 运行 research/factor_alchemy/run_fa.py --full")
            print("=" * 70)
