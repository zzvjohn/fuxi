"""
Factor Injector — PASS因子自动注入桥接

从 passed_factor_pool.csv 读取 PASS 因子,
计算 FRI, 注册到范式增强器,
准备 MetaController 摄入。

替代旧 GA 流水线中的因子注册环节。
Stage 2 PASS因子 → FRI评估 → 范式增强 → MetaController基因库

用法:
    python scripts/factor_injector.py [--dry-run] [--all]

自动化管线:
    1. 读取 passed_factor_pool.csv, 找出 status=candidate 的未注入因子
    2. 对每个因子计算 FRI (需要日频数据)
    3. 注册到 ParadigmAugmenter
    4. 写入 injected_factors.json (MetaController 摄入接口)
    5. 如果 ParadigmAugmenter 触发阈值 → 建议运行 AlphaAgent v3 进化轮
"""

import sys
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# ── 路径 ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.factor_alchemy.fri import (
    compute_fri, FRIResult,
    _exec_formula, _cross_section_zscore,
    build_existing_pool_from_csv,
)
from research.factor_alchemy.paradigm_augmenter import (
    ParadigmAugmenter, ParadigmAugmentation,
)
from research.factor_alchemy.psi_orthogonalizer import (
    psi_orthogonalize, PsiResult, MIN_R_SQUARED_REJECT,
)
from research.factor_alchemy.experience_memory import get_memory
from build_pool_cache import append_factors_to_cache

# ── 状态文件 ──────────────────────────────────────────────
INJECTED_STATE_PATH = PROJECT_ROOT / "data" / "injected_factors.json"
POOL_CACHE_PATH = PROJECT_ROOT / "data" / "cache" / "existing_factor_pool.pkl"

# ── 类别→维度关键词 映射 (用于自动生成 rationale + dimensions) ────
# KEY: category (来自 passed_factor_pool.csv)
# VALUE: (dimensions_list, rationale_template)  —— template 中用 {label} 占位
CATEGORY_RATIONALE_MAP = {
    "extreme_events": (
        ["尾部风险/下行", "趋势/动量"],
        "{label}捕捉个股在极端下跌后回撤持续时间中隐含的尾部风险与下行保护溢价：回撤持续时间越长，下行脆弱性越高。与趋势/动量因子形成天然互补（尾部风险×动量配对），有助于降低策略在市场急跌中的最大回撤。"
    ),
    "学术异象库": (
        ["尾部风险/下行", "波动率", "微观结构"],
        "{label}是经典学术异象的A股验证：个股收益与市场波动的协变尾部风险信号。与传统波动率因子正交，与微观结构因子互补（隔夜跳空对崩盘风险的重新定价），提供独特的下行维度。"
    ),
    "mid_report_divergence": (
        ["趋势/动量", "资金流", "基本面/成长"],
        "{label}捕捉财报窗口前价格动量与资金流趋势中蕴含的盈余预期信息泄露：量价趋势偏离反映机构预判行为。与资金流因子互补（量价配合验证信息可信度），提供不同于传统60日动量的短窗口基本面/成长预期信号。"
    ),
    "growth_quality": (
        ["基本面/成长", "趋势/动量", "波动率"],
        "{label}从资本投入产出效率评估企业基本面质量与成长可持续性：高资本效率=更强盈利增长，避免伪成长陷阱。与技术面因子（趋势/动量、资金流、波动率）高度互补，提供V3池稀缺的盈利质量维度。"
    ),
    "failure_mode_inverse": (
        ["反转/均值回归", "资金流", "流动性"],
        "{label}利用历史上验证失败的因子模式的反向信号：失败模式逆转为正alpha源，捕捉被市场过度定价的行为偏差。与资金流和流动性因子互补（量价关系验证反转信号的可靠性），提供反共识alpha。"
    ),
    "small_bull_breakthrough": (
        ["流动性", "微观结构", "趋势/动量"],
        "{label}捕捉小盘股在牛市突破行情中的流动性-质量共振：高流动性质量的小盘股在突破时弹性更强。与微观结构和趋势/动量因子互补，适配A股小盘风格的独特alpha源。"
    ),
    "vm_diff_replacement": (
        ["波动率", "趋势/动量", "市场宽度/结构"],
        "{label}作为vm_diff毒因子的替代方案：用高低价波动率价差替代成交量波动率，规避vm_diff的JQ灾难。与趋势/动量因子互补（波动率收敛→趋势启动），提供波动率维度的安全替代信号。"
    ),
    "ml_inspired_construction": (
        ["趋势/动量", "波动率", "市场宽度/结构"],
        "{label}采用机器学习启发的扩散指数构造方法：多信号交叉确认动量趋势，减少单一技术指标噪声。与传统趋势/动量因子互补（扩散确认提升信号信噪比），提供ML构造范式的alpha增强。"
    ),
    "academic_anomaly_adaptation": (
        ["尾部风险/下行", "情绪/行为", "反转/均值回归"],
        "{label}是学术文献中行为金融异象的A股适配：MAX效应/彩票偏好等行为偏差的A股验证。与尾部风险和反转因子互补（行为偏差→过度反应→均值回归），捕获被散户交易行为驱动的错误定价。"
    ),
    "volume_structure": (
        ["流动性", "资金流", "筹码分布"],
        "{label}从成交量结构角度评估筹码集中度与资金动向：量能结构的截面差异反映机构与散户行为分化。与筹码分布和资金流因子互补，提供A股独特的量能alpha信号。"
    ),
    "momentum_dynamics": (
        ["趋势/动量", "资金流"],
        "{label}从动量生命周期角度捕捉趋势加速与衰减阶段：动量加速→溢价累积，减速→溢价消退。与资金流因子互补（量价配合验证动量持续性），提供超传统价格动量的时序alpha。"
    ),
    "volatility_structure": (
        ["波动率", "反转/均值回归"],
        "{label}从波动率期限结构捕捉波动率聚集与均值回归：波动率异常→风险补偿→均值回归。与传统反转因子互补（波动率加权增强反转信号稳定性），提供波动率维度的纯alpha。"
    ),
    "price_pattern": (
        ["反转/均值回归", "微观结构"],
        "{label}从价格形态识别超卖超买区域：技术形态+截面排名，捕捉短期过度反应的均值回归。与微观结构因子互补（日内路径信息增强形态信号），适配A股散户主导的价格形态。"
    ),
    "liquidity_microstructure": (
        ["流动性", "微观结构", "筹码分布"],
        "{label}从流动性微观结构捕捉市场微观摩擦：买卖价差、市场深度等隐含的交易成本补偿。与筹码分布因子互补（微观结构与宏观筹码的交叉验证），提供流动性溢价的多维信号。"
    ),
    "behavioral": (
        ["情绪/行为", "反转/均值回归"],
        "{label}从投资者行为偏差中提取alpha：过度自信、锚定效应、损失厌恶等认知偏差的系统性利用。与反转因子互补（行为偏差驱动→过度反应→均值回归），行为金融在A股的实证验证。"
    ),
}

# ── 类别→范式 映射 (用于 Experience Memory 记录) ────
CAT2PAR = {
    'volume_structure': '流动性', 'liquidity_micro': '微观结构',
    'return_decomposition': '隔夜', 'momentum_dynamics': '趋势/动量',
    'price_pattern': '反转', 'volatility_structure': '波动率',
    'return_distribution': '尾部风险', 'behavioral': '行为金融',
    'information_flow': '微观结构', 'extreme_events': '尾部风险',
    'event_driven': '基本面/成长', 'multi_period': '趋势/动量',
    'relative_value': '基本面/成长', 'trend_quality': '趋势/动量',
    'growth_quality': '基本面/成长', 'mid_report_divergence': '趋势/动量',
    '学术异象库': '尾部风险',
    'failure_mode_inverse': '反转',
    'small_bull_breakthrough': '流动性',
    'vm_diff_replacement': '波动率',
    'ml_inspired_construction': '趋势/动量',
    'academic_anomaly_adaptation': '行为金融',
    'liquidity_microstructure': '微观结构',
}


def _generate_factor_rationale(name: str, label: str, category: str,
                               fri: float, fri_grade: str, icir: float) -> str:
    """为注入因子自动生成含维度关键词的丰富 rational。

    优先用 CATEGORY_RATIONALE_MAP 模板，fallback 用通用模板。
    结构: {经济机制+维度关键词} [FRI={score}/{grade}, ICIR={icir}]
    """
    info = CATEGORY_RATIONALE_MAP.get(category)
    if info is not None:
        _, template = info
        # 安全的 label
        safe_label = str(label) if label and str(label) != 'nan' else name
        body = template.format(label=safe_label)
    else:
        # 通用 fallback: 嵌入 category 名 + 所有可能维度
        dimensions, _ = _infer_dimensions_from_category(category, name, label)
        dim_str = "、".join(dimensions) if dimensions else "alpha信号"
        body = (
            f"{str(label) if label and str(label) != 'nan' else name}（类型={category}）: "
            f"Stage2 FRI验证因子，关注{dim_str}维度的alpha信号。"
        )
    return f"{body} [FRI={fri:.2f}/{fri_grade}, ICIR={icir:.3f}]"


def _infer_dimensions_from_category(category: str, name: str = "",
                                     label: str = "") -> tuple:
    """从 category 推断因子经济维度。

    Returns: (dimensions_list, rationale_template_or_None)
    """
    info = CATEGORY_RATIONALE_MAP.get(category)
    if info is not None:
        return info  # (dimensions, template)
    # 无法识别的 category: 用通用维度
    return (["趋势/动量", "资金流", "波动率"], None)


def _generate_factor_dimensions(rationale: str, category: str) -> List[str]:
    """根据 rationale 文本 + category 自动推断因子的经济维度。

    优先级: CATEGORY_RATIONALE_MAP > ECON_DIMENSIONS 关键词扫描
    """
    info = CATEGORY_RATIONALE_MAP.get(category)
    if info is not None:
        return info[0]  # predefined dimensions list

    # Fallback: 扫描 rationale 中的维度关键词
    from research.factor_alchemy.alpha_agent_pipeline_v3 import ECON_DIMENSIONS
    text = rationale
    found = []
    for dim, keywords in ECON_DIMENSIONS.items():
        for kw in keywords:
            if kw in text:
                found.append(dim)
                break
    return found if found else ["趋势/动量"]


def _verify_injected_rationale(injected_path: Path = INJECTED_STATE_PATH) -> dict:
    """校验 injected_factors.json 中所有已注入因子是否都有 rationale + dimensions。

    Returns: {"ok": bool, "missing_rationale": [...], "missing_dimensions": [...]}
    """
    if not injected_path.exists():
        return {"ok": True, "missing_rationale": [], "missing_dimensions": []}

    with open(injected_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    missing_r = []
    missing_d = []
    for item in data.get('injected', []):
        fn = item.get('factor_name', '?')
        if not item.get('rationale'):
            missing_r.append(fn)
        if not item.get('dimensions'):
            missing_d.append(fn)

    return {
        "ok": len(missing_r) == 0 and len(missing_d) == 0,
        "total": len(data.get('injected', [])),
        "missing_rationale": missing_r,
        "missing_dimensions": missing_d,
    }


def build_cached_pool(
    pool_csv_path: Path,
    close_df: pd.DataFrame,
    open_df: pd.DataFrame,
    high_df: pd.DataFrame,
    low_df: pd.DataFrame,
    vol_df: pd.DataFrame,
    force_rebuild: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    构建并缓存已有因子池。
    
    首次运行或数据变化时重建 (~5-10分钟), 后续从 pickle 加载 (~1秒)。
    缓存按 pool_csv 的修改时间和数据 shape 做版本校验。
    """
    cache_key = {
        'csv_mtime': pool_csv_path.stat().st_mtime if pool_csv_path.exists() else 0,
        'n_dates': len(close_df),
        'n_stocks': len(close_df.columns),
    }
    
    if not force_rebuild and POOL_CACHE_PATH.exists():
        try:
            import pickle
            with open(POOL_CACHE_PATH, 'rb') as f:
                cached = pickle.load(f)
            if cached.get('_meta', {}).get('cache_key') == cache_key:
                existing = {k: v for k, v in cached.items() if not k.startswith('_')}
                if existing:
                    print(f"  [cache] 加载已有因子池: {len(existing)} 个因子 (from {POOL_CACHE_PATH})")
                    return existing
        except Exception:
            pass
    
    print(f"  [build] 构建已有因子池 (首次运行, 约2-5分钟)...")
    existing = build_existing_pool_from_csv(
        pool_csv_path, close_df, open_df, high_df, low_df, vol_df
    )
    
    # 缓存
    if existing:
        POOL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        cached = dict(existing)
        cached['_meta'] = {'cache_key': cache_key, 'built_at': datetime.now().isoformat()}
        import pickle
        with open(POOL_CACHE_PATH, 'wb') as f:
            pickle.dump(cached, f)
        print(f"  [cache] 已缓存 {len(existing)} 个因子 → {POOL_CACHE_PATH}")
    
    return existing


def load_data():
    """加载日频 OHLCV 数据 (与 daily_factor_hypothesis.py 一致)"""
    local_path = PROJECT_ROOT / 'data' / 'raw' / 'daily_prices.csv'
    today = datetime.now()
    end_dt = pd.Timestamp(today.strftime('%Y-%m-%d'))
    start_dt = pd.Timestamp((today - timedelta(days=270)).strftime('%Y-%m-%d'))

    daily = pd.read_csv(
        local_path,
        dtype={'ts_code': str, 'trade_date': str},
        usecols=['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol']
    )
    daily['trade_date'] = pd.to_datetime(daily['trade_date'], format='mixed')
    daily = daily[(daily['trade_date'] >= start_dt) & (daily['trade_date'] <= end_dt)]
    daily = daily.drop_duplicates(subset=['ts_code', 'trade_date'])
    daily = daily.sort_values(['ts_code', 'trade_date'])

    close_df = daily.pivot(index='trade_date', columns='ts_code', values='close')
    open_df = daily.pivot(index='trade_date', columns='ts_code', values='open')
    high_df = daily.pivot(index='trade_date', columns='ts_code', values='high')
    low_df = daily.pivot(index='trade_date', columns='ts_code', values='low')
    vol_df = daily.pivot(index='trade_date', columns='ts_code', values='vol')

    min_days = 40
    valid_codes = close_df.columns[close_df.count() >= min_days]
    close_df = close_df[valid_codes]
    open_df = open_df[valid_codes]
    high_df = high_df[valid_codes]
    low_df = low_df[valid_codes]
    vol_df = vol_df[valid_codes]

    return close_df, open_df, high_df, low_df, vol_df


def load_regime_labels() -> Optional[pd.Series]:
    """尝试加载 regime 标签"""
    regime_path = PROJECT_ROOT / 'data' / 'regime_labels.csv'
    if regime_path.exists():
        df = pd.read_csv(regime_path, parse_dates=['date'], index_col='date')
        return df['regime']
    
    # 尝试从 experiment_state.json 推断
    from research.factor_alchemy.meta_controller import MetaController
    mc = MetaController()
    if hasattr(mc, 'regime_labels') and mc.regime_labels is not None:
        return mc.regime_labels
    
    return None


def load_injected_history() -> List[str]:
    """加载已注入因子历史"""
    if INJECTED_STATE_PATH.exists():
        with open(INJECTED_STATE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [item['factor_name'] for item in data.get('injected', [])]
    return []


def save_injected_history(injected: List[Dict]):
    """保存注入历史"""
    INJECTED_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {
        'version': '1.2.0',
        'last_updated': datetime.now().isoformat(),
        'total_injected': len(injected),
        '_rationale_version': 'v3 — 自动生成 rationale + dimensions (CATEGORY_RATIONALE_MAP), 注入后自动校验',
        'injected': injected,
    }
    with open(INJECTED_STATE_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def run_injection(
    pool_csv_path: Path,
    close_df: pd.DataFrame,
    open_df: pd.DataFrame,
    high_df: pd.DataFrame,
    low_df: pd.DataFrame,
    vol_df: pd.DataFrame,
    regime_labels: Optional[pd.Series] = None,
    dry_run: bool = False,
    all_factors: bool = False,
    verbose: bool = True,
) -> Dict:
    """
    执行因子注入流程。

    Parameters
    ----------
    pool_csv_path: passed_factor_pool.csv 路径
    close_df, open_df, high_df, low_df, vol_df: OHLCV 面板
    regime_labels: regime标签
    dry_run: 仅预览, 不写入
    all_factors: 注入所有 status=candidate 因子 (无视已注入)
    verbose: 详细输出
    """
    if not pool_csv_path.exists():
        return {"error": f"{pool_csv_path} 不存在"}

    pool_df = pd.read_csv(pool_csv_path, encoding='utf-8-sig')
    candidates = pool_df[pool_df['status'] == 'candidate'].copy()

    if candidates.empty:
        return {"status": "no_candidates", "message": "无 status=candidate 因子"}

    # 去重: 排除已注入
    injected_history = load_injected_history()
    if not all_factors:
        injected_names = set(injected_history)
        candidates = candidates[~candidates['name'].isin(injected_names)]
    
    if candidates.empty:
        return {"status": "all_injected", "message": f"所有 PASS 因子已注入 ({len(injected_history)} 个)"}

    if verbose:
        print(f"\n{'='*60}")
        print(f"Factor Injector — 待注入: {len(candidates)} 个因子")
        print(f"{'='*60}")

    # 构建已有因子池 (用于 novelty + Ψ正交化, 首次缓存后秒级加载)
    existing_pool = build_cached_pool(
        pool_csv_path, close_df, open_df, high_df, low_df, vol_df
    )

    # 前向收益
    forward_5d = close_df.shift(-5) / close_df - 1

    # 范式增强器
    augmenter = ParadigmAugmenter()

    # 逐因子处理
    injected = []
    new_for_pool = []
    fri_results = []

    for idx, (_, row) in enumerate(candidates.iterrows()):
        name = row['name']
        formula = row.get('formula', '')
        direction = row.get('direction', 'long')
        
        if not formula or pd.isna(formula):
            if verbose:
                print(f"  [{idx+1}/{len(candidates)}] {name}: 无公式, 跳过")
            continue
        
        if verbose:
            print(f"  [{idx+1}/{len(candidates)}] {name} ...", end=' ', flush=True)
        
        try:
            # 计算因子值
            fv = _exec_formula(formula, close_df, open_df, high_df, low_df, vol_df)
            if fv is None or fv.empty:
                if verbose:
                    print("公式执行无结果, 跳过")
                continue
            
            # 方向调整
            if direction == 'short':
                fv = -fv
            
            # 标准化
            fv = _cross_section_zscore(fv)
            
            # ── Ψ 正交化 (FactorMiner 核心) ─────────────────
            psi_result = psi_orthogonalize(
                factor_name=name,
                factor_values=fv,
                existing_pool=existing_pool,
                verbose=False,
            )
            
            if psi_result.rejection_recommended:
                if verbose:
                    print(f"Ψ-REJECT R²={psi_result.r_squared:.3f} "
                          f"(> {MIN_R_SQUARED_REJECT}), 跳过")
                continue
            
            fv_ortho = psi_result.orthogonalized_values
            
            # ── 计算 FRI (用正交化后的因子) ──────────────────
            fr = compute_fri(
                factor_name=f"{name}_ortho",
                factor_values=fv_ortho,
                forward_returns=forward_5d,
                regime_labels=regime_labels,
                existing_factor_pool=existing_pool,
                verbose=False,
            )
            
            fri_results.append(fr)
            
            if verbose:
                psi_flag = f"Ψ(R²={psi_result.r_squared:.2f})" if psi_result.r_squared > 0.1 else "Ψ(clean)"
                print(f"FRI={fr.fri:.3f} ({fr.grade}) "
                      f"P={fr.precision:.3f} S={fr.persistence:.3f} "
                      f"C={fr.consistency:.3f} N={fr.novelty:.3f} "
                      f"{psi_flag}")
            
            # ── 自动生成 rationale + dimensions ────────────
            label = str(row.get('label', name))
            category = str(row.get('category', ''))
            icir_val = float(row.get('icir', row.get('daily_icir', 0))) if pd.notna(row.get('icir', row.get('daily_icir', np.nan))) else 0.0
            factor_rationale = _generate_factor_rationale(
                name, label, category, fr.fri, fr.grade, icir_val)
            factor_dimensions = _generate_factor_dimensions(factor_rationale, category)

            # 记录注入
            injected_record = {
                'factor_name': name,
                'factor_label': label,
                'category': category,
                'date': datetime.now().strftime('%Y-%m-%d'),
                'icir': icir_val,
                'ic_pct': float(row.get('+ic%', row.get('+ic_pct', 0))) if pd.notna(row.get('+ic%', row.get('+ic_pct', np.nan))) else 0.0,
                'fri': fr.fri,
                'fri_grade': fr.grade,
                'fri_precision': fr.precision,
                'fri_persistence': fr.persistence,
                'fri_consistency': fr.consistency,
                'fri_novelty': fr.novelty,
                'max_corr_with_existing': fr.max_corr_with_existing,
                'max_corr_factor': fr.max_corr_factor,
                # ── 维度关键词 (Step 0 直接读取, 用于互补配对评分) ──
                'rationale': factor_rationale,
                'dimensions': factor_dimensions,
                # Ψ 正交化字段
                'psi_r_squared': psi_result.r_squared,
                'psi_independence': psi_result.independence,
                'psi_n_orthogonalized_against': psi_result.n_orthogonalized_against,
                'psi_orthogonalized_factor_names': psi_result.orthogonalized_factor_names,
                'psi_rejection_recommended': psi_result.rejection_recommended,
                'injected_at': datetime.now().isoformat(),
            }
            injected.append(injected_record)

            # ── 追加到因子池缓存 (供后续 Ψ 正交化使用) ────────
            # 注意: 缓存的是原始 z-scored 因子 (fv), 不是正交化后的,
            # 因为池缓存用于检测未来因子与已有因子的冗余度。
            if not dry_run:
                append_factors_to_cache({name: fv})

            # ── 记录到经验记忆库 ────────────────────────────
            if not dry_run:
                paradigm = CAT2PAR.get(row.get('category', ''), '通用')
                mem = get_memory()
                mem.record(
                    factor_name=name,
                    formula=formula,
                    paradigm=paradigm,
                    category=str(row.get('category', '')),
                    fri_score=fr.fri,
                    fri_grade=fr.grade,
                    fri_precision=fr.precision,
                    fri_persistence=fr.persistence,
                    fri_consistency=fr.consistency,
                    fri_novelty=fr.novelty,
                    icir=float(row.get('icir', row.get('daily_icir', 0))) if pd.notna(row.get('icir', row.get('daily_icir', np.nan))) else 0.0,
                    outcome="PASS" if fr.fri >= 0.4 else "WEAK",
                    psi_r_squared=psi_result.r_squared,
                    psi_independence=psi_result.independence,
                    max_corr_factor=fr.max_corr_factor,
                )
            
            # 注册到范式增强器 (不在 dry_run 模式下)
            if not dry_run:
                new_for_pool.append(row.to_dict())
            
        except Exception as e:
            if verbose:
                print(f"失败: {e}")
    
    # ── 更新状态 ──────────────────────────────────────────
    result = {
        "date": datetime.now().strftime('%Y-%m-%d'),
        "candidates_found": len(candidates),
        "injected_count": len(injected),
        "fri_summary": {
            "avg_fri": round(np.mean([r.fri for r in fri_results]), 4) if fri_results else 0,
            "grade_dist": {},
        },
    }
    
    # FRI 等级分布
    for r in fri_results:
        result["fri_summary"]["grade_dist"][r.grade] = \
            result["fri_summary"]["grade_dist"].get(r.grade, 0) + 1
    
    if dry_run:
        result["mode"] = "dry_run"
        result["injected"] = injected
        return result
    
    # 写入注入历史: 加载完整记录(非仅name列表)
    existing_injected_records = []
    if INJECTED_STATE_PATH.exists():
        try:
            with open(INJECTED_STATE_PATH, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            existing_injected_records = existing_data.get('injected', [])
        except Exception:
            existing_injected_records = []
    
    all_injected = list(existing_injected_records)
    
    # 转换已有记录
    existing_names = {h['factor_name'] for h in all_injected if isinstance(h, dict)}
    for ir in injected:
        if ir['factor_name'] not in existing_names:
            all_injected.append(ir)
    
    if injected:
        save_injected_history(all_injected)

        # ── 注入后校验: 确保所有因子都有 rationale + dimensions ──
        verify = _verify_injected_rationale(INJECTED_STATE_PATH)
        result["rationale_verification"] = verify
        if verbose:
            if not verify["ok"]:
                print(f"\n⚠️  注入后校验: 缺 rationale {len(verify['missing_rationale'])} 个, "
                      f"缺 dimensions {len(verify['missing_dimensions'])} 个!")
                if verify["missing_rationale"]:
                    print(f"   无rationale: {verify['missing_rationale']}")
                if verify["missing_dimensions"]:
                    print(f"   无dimensions: {verify['missing_dimensions']}")
            else:
                print(f"✅ 注入后校验通过: {verify['total']}/{verify['total']} 全部包含 rationale + dimensions")
    
    # 范式增强器
    if new_for_pool:
        # 临时写入一个 cleanup CSV 供 augmenter 读取
        augmenter.augment(pool_csv_path, since_date="2026-08-01")
    
    # 检查触发
    augment_status = augmenter.get_status()
    result["augmenter_status"] = augment_status
    result["should_trigger_evolution"] = augment_status["ready_to_trigger"]
    
    if augment_status["ready_to_trigger"]:
        result["llm_context"] = augmenter.get_llm_context()
    
    result["mode"] = "live"
    result["injected"] = injected
    
    # ── 输出摘要 ──────────────────────────────────────────
    if verbose:
        print(f"\n{'='*60}")
        print(f"注入完成: {len(injected)}/{len(candidates)} 个因子")
        print(f"FRI 均值: {result['fri_summary']['avg_fri']:.3f}")
        print(f"FRI 分布: {result['fri_summary']['grade_dist']}")
        if result["should_trigger_evolution"]:
            print(f"\n⚡ 范式增强器已触发!")
            print(f"   累积新因子: {augment_status['current_round_factors']}")
            print(f"   建议运行: AlphaAgent v3 进化轮")
        print(f"{'='*60}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Factor Injector — PASS因子自动注入')
    parser.add_argument('--dry-run', action='store_true', help='仅预览, 不写入')
    parser.add_argument('--all', action='store_true', help='注入所有PASS因子 (无视已注入)')
    parser.add_argument('--quiet', action='store_true', help='安静模式')
    args = parser.parse_args()
    
    pool_path = PROJECT_ROOT / 'data' / 'passed_factor_pool.csv'
    
    if not pool_path.exists():
        print(f"错误: {pool_path} 不存在")
        sys.exit(1)
    
    # 加载数据
    print("加载数据...")
    close_df, open_df, high_df, low_df, vol_df = load_data()
    regime_labels = load_regime_labels()
    
    print(f"  数据: {len(close_df)} 天 × {len(close_df.columns)} 只股票")
    
    # 执行注入
    result = run_injection(
        pool_csv_path=pool_path,
        close_df=close_df,
        open_df=open_df,
        high_df=high_df,
        low_df=low_df,
        vol_df=vol_df,
        regime_labels=regime_labels,
        dry_run=args.dry_run,
        all_factors=args.all,
        verbose=not args.quiet,
    )
    
    if not args.quiet:
        print(json.dumps(
            {k: v for k, v in result.items() if k not in ('injected', 'llm_context')},
            ensure_ascii=False, indent=2, default=str
        ))


if __name__ == "__main__":
    main()
