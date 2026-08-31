# -*- coding: utf-8 -*-
"""
V4 管道统一入口: paradigm_v4 → library_orthogonality → factor_model_cooptim
=============================================================================

端到端集成脚本，串联 v4 架构三大新模块:

  1. paradigm_v4.py        — 21范式 + A股算子库 + 多频率支持
  2. library_orthogonality.py — 因子族聚类 / 禁止区域 / Red Sea监控 / 库级准入
  3. factor_model_cooptim.py  — MAB调度 / 因子-模型联合优化 / LLM上下文生成

模式:
  python run_v4_pipeline.py --diagnose    # 全量诊断报告
  python run_v4_pipeline.py --explore     # 生成探索上下文 (→ AlphaAgent v3)
  python run_v4_pipeline.py --view        # 快速状态快照(默认)
  python run_v4_pipeline.py --ralph       # Ralph Loop 自进化 (v0.2 新增)

设计原则:
  - JQ是唯一真相源
  - Local仅做否决不排序
  - LLM仅接触schema级信息, 不接触raw data

v0.2 增强 (2026-08-05):
  ✅ --ralph 模式: 集成 Ralph Loop (Retrieve→Generate→Evaluate→Distill)
  ✅ 集成 Multi-Stage Validator (四阶段验证)
  ✅ 集成 Semantic Verifier (语义一致性校验)

与旧架构的关系:
  ✅ 替代: Stage 2/3 自动化中的旧 FactorMiner 注入管线
  ✅ 兼容: AlphaAgent v3 的 LLM context 生成接口
  ✅ 增强: Experience Memory 的禁止区域维度
  ❌ 不依赖: factor_injector.py / --evo-context / ENABLE_AUTO_EVOLUTION
"""

import sys, os, io, json, csv, pickle, gc, time
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Set
from collections import Counter, defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # 项目根目录

# ── 项目根目录 ──────────────────────────────────────────
ALCHEMY_DIR = Path(__file__).resolve().parent  # research/factor_alchemy/
RESEARCH_DIR = ALCHEMY_DIR.parent              # research/
PROJECT_ROOT = RESEARCH_DIR.parent             # 项目根目录
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"

# ── v4 模块 ─────────────────────────────────────────────
from paradigm_v4 import PARADIGMS_V4, FREQUENCY_CONFIGS, A_SHARE_SPECIFIC_OPERATORS, STANDARD_OPERATORS
from library_orthogonality import (
    LibraryOrthogonalityManager, FactorEntry, ForbiddenRegion, FactorCluster,
    create_from_existing_factors,
)
from factor_model_cooptim import (
    FactorModelCoOptimizer, MABScheduler, ResearchDirection, ModelConfig,
    create_default_cooptimizer, print_system_status,
)


# ═══════════════════════════════════════════════════════════
# 一、Category → Paradigm 映射 (基于 keyword_triggers)
# ═══════════════════════════════════════════════════════════

# 构建关键词→范式映射表
_KEYWORD_TO_PARADIGM: Dict[str, str] = {}
for paradigm_name, info in PARADIGMS_V4.items():
    for kw in info.get("keyword_triggers", []):
        kw_lower = kw.lower()
        if kw_lower not in _KEYWORD_TO_PARADIGM:
            _KEYWORD_TO_PARADIGM[kw_lower] = paradigm_name

# 手动补充 category 名称 → 范式 的精确映射 (优先级高于关键词)
_CATEGORY_PARADIGM_MAP: Dict[str, str] = {
    # v3 范式
    "liquidity_micro": "流动性×微观结构",
    "market_microstructure": "流动性×微观结构",
    "microstructure_liquidity": "流动性×微观结构",
    "microstructure_proxy": "流动性×微观结构",
    "microstructure_ml": "流动性×微观结构",
    "cf_microstructure": "流动性×微观结构",
    "money_flow": "资金流",
    "volume_structure": "资金流",
    "volume_clock_efficiency": "资金流",
    "momentum": "动量反转",
    "momentum_dynamics": "动量反转",
    "momentum_reversal": "动量反转",
    "behavioral": "行为金融",
    "behavioral_v2": "行为金融",
    "tail_risk": "尾部风险",
    "tail_risk_academic": "尾部风险",
    "higher_moment": "尾部风险",
    "asymmetry": "尾部风险",
    "sentiment_technical": "投资者情绪",
    "sentiment_crowding": "投资者情绪",
    "price_efficiency": "截面异常",
    "relative_value": "截面异常",
    "trend_quality": "趋势",
    "return_decomposition": "趋势",
    "volatility_structure": "下行保护",
    "volatility_risk": "波动率适应",
    "regime_adaptive": "波动率适应",
    "regime_bridging": "波动率适应",
    "chip_stratification": "筹码分布",
    "chip_stratification_ai": "筹码分布",
    "chip_structure": "筹码分布",
    "chip_structure_ai": "筹码分布",
    "extreme_events": "结构突变",
    "correlation_linkage": "市场宽度",
    "multi_period": "市场宽度",
    # v4 新增范式
    "event_driven": "事件驱动",
    "earnings_season_divergence": "事件驱动",
    "earnings_anomaly": "事件驱动",
    "earnings_behavior": "事件驱动",
    "earnings_drift": "事件驱动",
    "mid_report_divergence": "事件驱动",
    "sector_rotation": "行业轮动",
    "northbound_flow": "北向资金",
    "margin_trading": "两融信号",
    "block_trade": "大宗交易",
    "intraday_patterns": "高频微观结构",
    "cross_asset": "跨资产联动",
    "market_breadth": "市场情绪综合",
    # 综合类别 — 映射到最接近的范式
    "academic_anomalies": "学术异象",
    "academic_anomaly": "学术异象",
    "academic_anomaly_adaptation": "学术异象",
    "fundamental_quality_proxy": "基本面质量",
    "fundamental_analyst": "基本面质量",
    "growth_quality": "基本面质量",
    "small_bull_breakthrough": "流动性×微观结构",  # small_bull多用微观结构因子
    "ml_inspired_construction": "动量反转",  # ML构造多为动量变体
    "ml_inspired": "动量反转",
    "failure_mode_inverse": "尾部风险",  # 失败模式逆向→尾部保护
    "low_turnover_mid_freq": "流动性×微观结构",
    "low_turnover_medium_freq": "流动性×微观结构",
    "mid_freq_signal": "趋势",
    "research_driven": "学术异象",
    "information_flow": "流动性×微观结构",
    "price_pattern": "动量反转",
    "return_distribution": "尾部风险",
    "supply_demand": "资金流",
    "alpha_dynamics": "趋势",
    "network": "市场宽度",
    "calendar_anomaly": "事件驱动",
    "causal_discovery_driven": "截面异常",
    "new_data_source": "市场情绪综合",
    "risk_management": "下行保护",
    "risk_event_driven": "下行保护",
    "vm_diff_replacement": "波动率适应",
}


def map_category_to_paradigm(category: str, hypothesis: str = "", logic: str = "") -> str:
    """
    将因子 category 映射到 paradigm_v4 中的范式名。

    优先级: 精确映射 > 关键词匹配 > 默认
    """
    # 1. 精确映射
    if category in _CATEGORY_PARADIGM_MAP:
        return _CATEGORY_PARADIGM_MAP[category]

    # 2. 关键词匹配 (category + hypothesis + logic)
    all_text = f"{category} {hypothesis} {logic}".lower()
    scores: Dict[str, int] = defaultdict(int)
    for kw, paradigm in _KEYWORD_TO_PARADIGM.items():
        if kw in all_text:
            scores[paradigm] += 1

    if scores:
        return max(scores, key=scores.get)

    # 3. 默认 — 流动性（A股最主源）
    return "流动性×微观结构"


# ═══════════════════════════════════════════════════════════
# 二、因子池加载与范式映射
# ═══════════════════════════════════════════════════════════

def load_factor_metadata(csv_path: Path = None) -> List[Dict]:
    """从 passed_factor_pool.csv 加载因子元数据"""
    if csv_path is None:
        csv_path = DATA_DIR / "passed_factor_pool.csv"

    factors = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            factors.append(row)
    return factors


def build_factor_entries(metadata: List[Dict]) -> Tuple[List[FactorEntry], Dict[str, str]]:
    """
    将 CSV 元数据转换为 FactorEntry 对象。
    同时返回 category→paradigm 的最终映射供报告使用。
    """
    entries = []
    cat_map = {}

    for m in metadata:
        name = m.get('name', '').strip()
        category = m.get('category', '').strip()
        hypothesis = m.get('hypothesis', '')
        logic = m.get('logic', '')

        paradigm = map_category_to_paradigm(category, hypothesis, logic)
        cat_map[category] = paradigm

        try:
            icir = float(m.get('icir', 0) or m.get('daily_icir', 0) or 0)
        except (ValueError, TypeError):
            icir = 0.0
        try:
            ic = float(m.get('ic', 0) or m.get('ic_mean', 0) or m.get('daily_ic', 0) or 0)
        except (ValueError, TypeError):
            ic = 0.0

        fe = FactorEntry(
            name=name,
            expression=m.get('formula', ''),
            paradigm=paradigm,
            category=category,
            hypothesis=hypothesis,       # v0.5: LLM释义
            logic=logic,                 # v0.5: 经济逻辑
            dimensions=[hypothesis[:60]] if hypothesis else [],
            source=m.get('source', 'unknown'),
            ic=ic,
            icir=icir,
            status=m.get('status', 'reserve'),
            created_at=m.get('date', ''),
        )
        entries.append(fe)

    # 补充: 如果有"学术异象"或"基本面质量"范式不在PARADIGMS_V4中，
    # 合并到最近的标准范式
    for fe in entries:
        if fe.paradigm not in PARADIGMS_V4:
            # 回退匹配
            fe.paradigm = "学术异象" if "学术" in fe.paradigm or "academic" in (fe.category or "").lower() else \
                          "基本面质量" if "fundamental" in (fe.category or "").lower() or "quality" in (fe.category or "").lower() else \
                          "截面异常"

    # 再次确保所有范式都在PARADIGMS_V4中
    known = set(PARADIGMS_V4.keys())
    for fe in entries:
        if fe.paradigm not in known:
            fe.paradigm = "截面异常"  # 最后的回退

    return entries, cat_map


# ═══════════════════════════════════════════════════════════
# 三、相关性矩阵计算 (从 pickle 缓存)
# ═══════════════════════════════════════════════════════════

def compute_correlation_matrix(
    entries: List[FactorEntry],
    pkl_path: Path = None,
    sample_period: Tuple[str, str] = ("2025-01-01", "2025-12-31"),
    max_factors: int = 500,
) -> Tuple[Optional[np.ndarray], List[str]]:
    """
    从 pickle 缓存加载因子值并计算截面相关性矩阵。

    由于因子池很大 (217因子, ~3000+股票), 我们采用采样策略:
    - 只在指定日期范围内计算
    - 取每个因子的均值截面, 计算 factor×factor 相关性
    """
    if pkl_path is None:
        pkl_path = CACHE_DIR / "existing_factor_pool.pkl"

    if not pkl_path.exists():
        print(f"  ⚠️ 因子缓存不存在: {pkl_path}")
        return None, []

    print(f"  加载因子池缓存: {pkl_path}")
    try:
        with open(pkl_path, 'rb') as f:
            pool = pickle.load(f)
    except Exception as e:
        print(f"  ⚠️ 加载失败: {e}")
        return None, []

    if not isinstance(pool, dict):
        print(f"  ⚠️ 缓存格式异常: {type(pool)}")
        return None, []

    # 获取所有因子在缓存中的名称
    pool_names = set(pool.keys())
    matched_entries = [e for e in entries if e.name in pool_names]

    if len(matched_entries) < 3:
        print(f"  ⚠️ 匹配因子数不足: {len(matched_entries)}")
        return None, []

    # 限制因子数量以加速
    if len(matched_entries) > max_factors:
        # 优先保留非reserve状态的因子
        priority_entries = sorted(
            matched_entries,
            key=lambda e: (0 if e.status == 'candidate' else 1, e.icir or 0),
            reverse=False,
        )
        matched_entries = priority_entries[:max_factors]

    print(f"  计算 {len(matched_entries)} 个因子的相关性矩阵...")
    factor_series = {}
    dropped = 0

    # v0.5.4: 缓存格式检测 — 新缓存为元数据 dict {name: {formula, paradigm, ...}}
    #   旧缓存为因子值 DataFrame。dict 有 .values 属性但无 .shape,
    #   旧代码 `df.shape[1]` 在此直接 AttributeError 崩溃。
    sample_val = next(iter(pool.values()), None)
    pool_is_metadata = isinstance(sample_val, dict) and not isinstance(sample_val, pd.DataFrame)

    if pool_is_metadata:
        # 元数据格式 → 用 FactorICComputer 按公式现场计算因子值
        try:
            from factor_ic_computer import FactorICComputer
            ic_comp = FactorICComputer()
            ic_comp._load_data()  # _eval_formula 依赖已加载的价格数据
        except Exception as e:
            print(f"  ⚠️ IC 计算机初始化失败: {e}")
            return None, []

        with_formula = [fe for fe in matched_entries
                        if fe.expression and str(fe.expression).strip()]
        # 现场计算有成本 (~0.5-1s/因子), 元数据路径硬性上限 200 个
        cap = min(max_factors, 200)
        if len(with_formula) > cap:
            with_formula.sort(key=lambda e: abs(e.icir or 0), reverse=True)
            with_formula = with_formula[:cap]

        print(f"  缓存为元数据格式 → 现场计算 {len(with_formula)} 个因子值 "
              f"(top |ICIR|, 上限 {cap})...")
        t_fx = time.time()
        for fe in with_formula:
            try:
                fdf = ic_comp._eval_formula(str(fe.expression), fe.name)
                if fdf is None or len(fdf) < 100:
                    dropped += 1
                    continue
                # 最近60个交易日 → 每股均值向量 (对齐旧格式语义)
                fdf = fdf.sort_values("trade_date")
                recent = fdf.groupby("ts_code").tail(60)
                factor_series[fe.name] = recent.groupby("ts_code")["factor_value"].mean()
            except Exception:
                dropped += 1
                continue
        print(f"  因子值计算完成 (耗时 {time.time()-t_fx:.1f}s)")
    else:
        # 旧格式: 因子值 DataFrame
        for fe in matched_entries:
            df = pool[fe.name]
            if not isinstance(df, pd.DataFrame) or df.shape[1] < 10:
                dropped += 1
                continue
            try:
                # 取最近的截面均值作为因子代表向量
                recent = df.iloc[-60:]  # 最近60天
                factor_series[fe.name] = recent.mean(axis=0)  # 每个股票的均值
            except Exception:
                dropped += 1
                continue

    if dropped > 0:
        print(f"    ⚠️ {dropped} 个因子数据异常, 跳过")

    factor_names = list(factor_series.keys())
    n = len(factor_names)

    if n < 3:
        print(f"  ⚠️ 可用因子数不足: {n}")
        return None, []

    # 构建 n×n 相关性矩阵
    print(f"  构建 {n}×{n} 相关性矩阵...")
    corr_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                try:
                    # 取交集非NaN值
                    si = factor_series[factor_names[i]]
                    sj = factor_series[factor_names[j]]
                    valid = ~(np.isnan(si) | np.isnan(sj))
                    if valid.sum() < 20:
                        corr = 0.0
                    else:
                        # v0.5.4: 显式对齐索引 — 不同因子可用股票集合不同,
                        # 直接 si[valid]/sj[valid] 长度不一致会让 corrcoef 抛
                        # ValueError 被静默吞掉 → 矩阵全 0
                        si_v = si[valid]
                        sj_v = sj[valid]
                        common = si_v.index.intersection(sj_v.index)
                        if len(common) < 20:
                            corr = 0.0
                        else:
                            a = si_v.loc[common].to_numpy(dtype=float)
                            b = sj_v.loc[common].to_numpy(dtype=float)
                            a = np.nan_to_num(a, nan=0.0)
                            b = np.nan_to_num(b, nan=0.0)
                            if np.std(a) < 1e-12 or np.std(b) < 1e-12:
                                corr = 0.0  # 常数因子, 无相关
                            else:
                                with np.errstate(invalid='ignore', divide='ignore'):
                                    corr = np.corrcoef(a, b)[0, 1]
                    if np.isnan(corr):
                        corr = 0.0
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr
                except Exception:
                    pass

    print(f"  相关性矩阵完成")
    return corr_matrix, factor_names


# ═══════════════════════════════════════════════════════════
# 四、管道诊断报告
# ═══════════════════════════════════════════════════════════

def run_diagnose(
    csv_path: Path = None,
    pkl_path: Path = None,
    with_correlation: bool = True,
):
    """
    完整诊断模式 — 加载因子池、范式映射、聚类分析、探索建议。
    """
    print("=" * 70)
    print("  V4 管道诊断报告")
    print("  paradigm_v4 → library_orthogonality → factor_model_cooptim")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Step 1: 加载因子元数据
    print("\n[Step 1/5] 加载因子池元数据...")
    t0 = time.time()
    metadata = load_factor_metadata(csv_path)
    print(f"  加载 {len(metadata)} 个因子 (耗时 {time.time()-t0:.1f}s)")

    if not metadata:
        print("  ⚠️ 因子池为空, 中止")
        return

    # Step 2: 范式映射
    print("\n[Step 2/5] 范式映射 (category → paradigm_v4)...")
    entries, cat_map = build_factor_entries(metadata)

    # 统计范式覆盖
    paradigm_counts = Counter(fe.paradigm for fe in entries)
    status_counts = Counter(fe.status for fe in entries)

    print(f"  因子: {len(entries)} | candidate: {status_counts.get('candidate', 0)}")
    print(f"  范式覆盖: {len(paradigm_counts)}/{len(PARADIGMS_V4)}")

    uncovered = [p for p in PARADIGMS_V4 if p not in paradigm_counts]
    if uncovered:
        print(f"  ⚠️ 未覆盖范式 ({len(uncovered)}):")
        for p in uncovered:
            info = PARADIGMS_V4[p]
            print(f"      - {info['id']}. {p}: {info['a_share_relevance'][:60]}")

    # Step 3: 初始化 LibraryOrthogonalityManager
    print("\n[Step 3/5] 初始化库级正交性管理器...")
    mgr = LibraryOrthogonalityManager(data_dir=DATA_DIR)
    # v0.5.4: 记录持久化状态中的矩阵因子数 (注册会追加, 不能拿注册后的数比较)
    persisted_n = len(set(mgr._factor_names or []))

    # 注册所有因子
    for fe in entries:
        mgr.factors[fe.name] = fe
        if fe.name not in mgr._factor_names:
            mgr._factor_names.append(fe.name)

    # Step 4: 相关性矩阵 + 聚类
    if with_correlation:
        print("\n[Step 4/5] 计算相关性矩阵与聚类...")
        corr_matrix, factor_names = compute_correlation_matrix(entries, pkl_path)

        if corr_matrix is not None and len(factor_names) >= 3:
            # v0.5.4: 保护已持久化的更大矩阵 — 现场计算覆盖因子少时禁止覆盖
            if persisted_n > 0 and len(factor_names) * 2 < persisted_n:
                print(f"  ⚠️ 新矩阵仅 {len(factor_names)} 因子, 不足持久化状态 {persisted_n} 的 50%"
                      f" → 跳过 recluster, 保留现有状态")
            else:
                mgr.recluster(corr_matrix, factor_names)
        else:
            print("  ⚠️ 跳过聚类 (相关性数据不足)")
            # 即使没有corr也能输出范式层面统计
    else:
        print("\n[Step 4/5] 跳过相关性计算 (--no-corr)")

    mgr.save_state()

    # Step 5: 初始化 MAB 联合优化器
    print("\n[Step 5/5] 初始化因子-模型联合优化器...")
    t5 = time.time()
    cooptim = create_default_cooptimizer()
    # 同步 MAB 状态
    library_stats = mgr.get_library_stats()

    # 生成 specification
    spec = cooptim.specify(library_stats)
    print(f"  联合优化器就绪 ({time.time()-t5:.1f}s)")
    print(f"  研究方向: {len(cooptim.mab.directions)}")
    print(f"  模型配置: {len(cooptim.models)}")

    # ── 输出报告 ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  诊断结果汇总")
    print("=" * 70)

    # 范式覆盖柱状图
    print("\n  📊 范式覆盖 (V4: 21个范式):")
    for p in PARADIGMS_V4:
        count = paradigm_counts.get(p, 0)
        bar_len = min(count, 30)
        bar = "█" * bar_len + "░" * max(0, 30 - bar_len)
        info = PARADIGMS_V4[p]
        tag = ""
        if count == 0:
            tag = " 🔴 未覆盖"
        elif count >= 5:
            tag = " 🟡 近饱和"
        elif count >= 8:
            tag = " 🔴 过饱和"
        print(f"    {info['id']:2d}. {p:16s} {bar} {count:3d}{tag}")

    # Red Sea 状态
    rs = mgr.get_red_sea_status()
    level_icon = {"green": "🟢", "elevated": "🟡", "warning": "🟠", "critical": "🔴", "unknown": "⚪", "no_matrix": "📊"}
    print(f"\n  🌊 Correlation Red Sea: {level_icon.get(rs['level'], '⚪')} {rs['level'].upper()}")
    if rs.get("reason"):
        print(f"     📝 {rs['reason']}")
    if rs['level'] not in ('unknown', 'no_matrix'):
        print(f"     中位相关性: {rs['median_corr']:.3f}  |  超阈值比例: {rs['pct_above_threshold']:.1f}%")
        print(f"     最大相关性: {rs['max_corr']:.3f}  |  因子族: {rs['n_clusters']}  |  阈值: {rs['threshold']}")

    # P-018: Red Sea 健康检查 (P-20260831-001: 传真实池计数, 修复空池导致 delta 恒负)
    auto_result = mgr.schedule_recluster(factor_pool=mgr.factors, max_age_hours=24, min_factor_change=5)
    health = auto_result.get("details", {})
    if auto_result["triggered"]:
        print(f"\n  RS Health: NEEDS_REBUILD - {auto_result['reason']}")
    else:
        last_t = mgr._last_recluster_time
        age_h = health.get("age_hours")
        age_str = f"{age_h:.0f}h前" if age_h else "unknown"
        delta = health.get("factor_delta", 0)
        print(f"\n  RS Health: OK ({age_str}, delta={delta}, need>=5 or >24h)")

    # P-018: 拥挤度 Top-5
    if mgr._corr_matrix is not None:
        try:
            top5 = mgr.get_crowding_top_n(5)
            if top5:
                print(f"\n  Crowding Top-5:")
                for name, score in top5:
                    bar = "=" * int(score * 20) + "-" * max(0, 20 - int(score * 20))
                    print(f"    {name[:30]:30s} {bar} {score:.3f}")
        except Exception:
            pass

    # 禁止区域
    print(f"\n  🚫 禁止区域: {len(mgr.forbidden_regions)}")
    for fr in mgr.forbidden_regions:
        icon = "🔴" if fr.severity == "hard" else "🟡"
        print(f"    {icon} {fr.description} ({fr.severity})")

    # 探索建议
    suggestions = mgr.get_suggested_exploration_directions()
    high_priority = [s for s in suggestions if s["priority"] == "high"]
    if high_priority:
        print(f"\n  🎯 高优先级探索方向 ({len(high_priority)}):")
        for s in high_priority[:5]:
            print(f"    → {s['paradigm']}: {s['reason']}")

    # MAB 状态
    report = cooptim.mab.get_exploration_report()
    active_dirs = [d for d in report["directions"] if d["status"] == "active"]
    print(f"\n  🎰 MAB调度器: {report['total_pulls']} 次选择, {len(active_dirs)} 活跃方向")

    # ── P-022: S6 衰减监控 v2 ──────────────────────────
    print("\n" + "=" * 70)
    print("  S6 DecayMonitor v2 — 因子衰减健康检查")
    print("=" * 70)
    try:
        from research.factor_alchemy.decay_monitor import get_decay_monitor
        dm = get_decay_monitor()
        alert_report = dm.get_decay_alert_for_diagnose()

        health_icon = {"green": "🟢", "yellow": "🟡", "red": "🔴"}
        print(f"\n  📉 整体健康度: {health_icon.get(alert_report['overall_health'], '⚪')} {alert_report['overall_health'].upper()}")
        print(f"     结构性衰减(P-022 hyperbolic): {alert_report['structural_count']}")
        print(f"     线性告警(P-004): {alert_report['linear_count']}")

        # 范式级告警
        sys_alerts = alert_report.get("systemic_paradigm_alerts", [])
        if sys_alerts:
            print(f"\n  🏭 范式级衰减告警 ({len(sys_alerts)}):")
            for sa in sys_alerts:
                icon = "🔴" if sa["level"] == "red" else "🟡"
                print(f"    {icon} {sa['msg']}")
        else:
            print(f"\n  ✅ 无范式级衰减告警")

        # Top 风险因子
        top_risk = alert_report.get("top_risk_factors", [])
        if top_risk:
            print(f"\n  ⚡ 高风险因子 (Top-{len(top_risk)}):")
            for rf in top_risk:
                hl_str = f"HL={rf['half_life']:.0f}步" if rf['half_life'] else "HL=∞"
                tz_str = f", 归零≈{rf['time_to_zero']:.0f}步" if rf['time_to_zero'] else ""
                mode_str = rf['decay_mode']
                print(f"    📉 {rf['name'][:35]:35s} {mode_str:12s} {hl_str}{tz_str}")
        else:
            print(f"\n  ℹ️ IC 缓存为空或数据不足 — 需 S5 评估后填充")
    except Exception as e:
        print(f"\n  ⚠️ DecayMonitor 加载失败: {e}")

    # ── AlphaAgent v3 集成说明 ──────────────────────────
    print("\n" + "=" * 70)
    print("  AlphaAgent v3 集成")
    print("=" * 70)
    print(f"""
  触发方式:
    python run_v4_pipeline.py --diagnose     ← 当前诊断
    python run_v4_pipeline.py --explore      ← 生成探索上下文

  集成到自动化:
    Stage 1 (每日研究):
      → run_v4_pipeline.py --explore → AlphaAgent v3 LLM context
    
    Stage 2 (因子实验):
      → AlphaAgent v3 生成 → 三重约束 → JQ候选
    
    Stage 3 (监控):
      → run_v4_pipeline.py --view → 快速状态快照

  关键变更 (vs 旧架构):
    ✅ 不再依赖 factor_injector.py
    ✅ 不再依赖 --evo-context
    ✅ 不再依赖 ENABLE_AUTO_EVOLUTION
    ❌ 不运行任何因子注入或自动进化
""")

    return {
        "entries": entries,
        "paradigm_counts": paradigm_counts,
        "uncovered": uncovered,
        "library_stats": library_stats,
        "spec": spec,
        "manager": mgr,
        "cooptim": cooptim,
    }


def run_explore(
    csv_path: Path = None,
    pkl_path: Path = None,
    output_path: Path = None,
):
    """
    探索模式 — 生成 AlphaAgent v3 的 LLM 探索上下文。

    输出: JSON 文件 + 文本描述，用于 AlphaAgent v3 的 generate_novel_factors_v3()。
    """
    print("=" * 60)
    print("  V4 Pipeline: 探索上下文生成")
    print("=" * 60)

    # 先运行诊断获取基础数据
    print("初始化...")
    metadata = load_factor_metadata(csv_path)
    entries, cat_map = build_factor_entries(metadata)

    mgr = LibraryOrthogonalityManager(data_dir=DATA_DIR)
    for fe in entries:
        mgr.factors[fe.name] = fe
        mgr._factor_names.append(fe.name)

    if pkl_path and pkl_path.exists():
        print("计算相关性...")
        corr_matrix, factor_names = compute_correlation_matrix(entries, pkl_path)
        if corr_matrix is not None and len(factor_names) >= 3:
            mgr.recluster(corr_matrix, factor_names)
    mgr.save_state()

    library_stats = mgr.get_library_stats()
    cooptim = create_default_cooptimizer()
    spec = cooptim.specify(library_stats)
    cooptim.mab.save_state()  # 持久化 MAB 选择方向
    llm_context = cooptim.synthesize_context_for_llm(spec, library_stats)

    # 额外信息
    suggestions = mgr.get_suggested_exploration_directions()
    high_priority = [s for s in suggestions if s["priority"] == "high"]

    # 频率多策略
    freq_info = []
    for freq, config in FREQUENCY_CONFIGS.items():
        freq_info.append({
            "name": config["name"],
            "frequency": freq,
            "windows": config.get("typical_windows", []),
            "suitable_for": config.get("suitable_for", ""),
        })

    # 构建完整上下文
    context = {
        "meta": {
            "version": "v4",
            "generated_at": datetime.now().isoformat(),
            "architecture": "paradigm_v4 + library_orthogonality + factor_model_cooptim",
        },
        "library_status": {
            "total_factors": library_stats["total_factors"],
            "n_clusters": library_stats["n_clusters"],
            "red_sea_level": library_stats["red_sea"]["level"],
            "median_corr": library_stats["red_sea"]["median_corr"],
            "n_forbidden_regions": library_stats["n_forbidden_regions"],
        },
        "paradigm_coverage": {
            p: count for p, count in library_stats["paradigm_coverage"].items()
        },
        "uncovered_paradigms": high_priority,
        "forbidden_regions": [
            {
                "description": fr.description,
                "severity": fr.severity,
                "paradigms": fr.paradigms_involved,
            }
            for fr in mgr.forbidden_regions
        ],
        "exploration_targets": spec,
        "frequency_configs": freq_info,
        "a_share_operators": {
            cat: list(ops.keys()) if isinstance(ops, dict) else ops
            for cat, ops in A_SHARE_SPECIFIC_OPERATORS.items()
        },
        "standard_operators": STANDARD_OPERATORS,
        "llm_prompt_context": llm_context,
    }

    # 输出
    if output_path is None:
        output_path = DATA_DIR / "v4_exploration_context.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(context, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n✅ 探索上下文 → {output_path}")

    # 打印 LLM 可用的 prompt 片段
    print("\n" + "=" * 60)
    print("  AlphaAgent v3 LLM Context (摘要)")
    print("=" * 60)
    print(llm_context[:2000])
    if len(llm_context) > 2000:
        print(f"\n... (共 {len(llm_context)} 字符)")

    return context


def run_view():
    """快速状态快照 — 读取最新状态,不重新计算"""
    state_path = DATA_DIR / "library_orthogonality_state.json"
    mab_path = DATA_DIR / "mab_scheduler_state.json"

    print("=" * 60)
    print("  V4 Pipeline: 快速状态")
    print("=" * 60)

    if state_path.exists():
        with open(state_path, 'r', encoding='utf-8') as f:
            state = json.load(f)
        stats = state.get("stats", {})
        print(f"\n  因子总数:    {stats.get('total_factors', '?')}")
        print(f"  因子族:      {stats.get('n_clusters', '?')}")
        print(f"  禁止区域:    {stats.get('n_forbidden_regions', '?')}")

        rs = stats.get("red_sea", {})
        level = rs.get("level", "unknown")
        icon = {"green": "🟢", "elevated": "🟡", "warning": "🟠", "critical": "🔴", "no_matrix": "📊"}.get(level, "⚪")
        label = "N/A (无矩阵)" if level == "no_matrix" else level.upper()
        print(f"  Red Sea:     {icon} {label} (中位corr={rs.get('median_corr', 0):.3f})")

        # 未覆盖范式
        pc = stats.get("paradigm_coverage", {})
        uncovered = [p for p, c in pc.items() if c == 0]
        if uncovered:
            print(f"  未覆盖范式:  {len(uncovered)}")
            for p in uncovered[:5]:
                print(f"    → {p}")
        print(f"\n  最后更新: {state.get('updated_at', '?')}")
    else:
        print("\n  ⚠️ 尚未运行诊断 (无状态文件)")
        print("  请先运行: python run_v4_pipeline.py --diagnose")

    if mab_path.exists():
        with open(mab_path, 'r', encoding='utf-8') as f:
            mab_state = json.load(f)
        print(f"\n  MAB 总选择: {mab_state.get('total_pulls', 0)}")
        directions = mab_state.get("directions", {})
        if isinstance(directions, list):
            active = [d for d in directions if d.get("status") == "active"]
            cooling = [d for d in directions if d.get("status") == "cooling"]
        else:
            active = [d for d in directions.values()
                      if d.get("status") == "active"]
            cooling = [d for d in directions.values()
                       if d.get("status") == "cooling"]
        print(f"  研究方向:    {len(active)} 活跃, {len(cooling)} 冷却中")

    print("=" * 60)


def _load_jq_generator():
    """v0.9: 动态加载 scripts/gen_jq_generic.py (避免 sys.path 耦合)。"""
    import importlib.util
    p = Path(__file__).parent.parent.parent / "scripts" / "gen_jq_generic.py"
    spec = importlib.util.spec_from_file_location("gen_jq_generic", str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_jq_codegen_queue() -> dict:
    """v0.9: JQ 代码自动生成队列 (data/jq_codegen_queue.json)。"""
    p = DATA_DIR / "jq_codegen_queue.json"
    if p.exists():
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_jq_codegen_queue(queue: dict) -> None:
    p = DATA_DIR / "jq_codegen_queue.json"
    try:
        json.dump(queue, open(p, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2, default=str)
    except Exception:
        pass


def _auto_generate_jq_code(result: Dict, entry_map: Dict, ralph) -> List:
    """v0.9: 对所有 S5 通过 (eligible_for_jq) 的候选自动生成 JQ 单因子回测代码。

    - 新生成候选: 用 run() 返回的 jq_candidate_details (公式/含义)
    - 库内种子重检: 公式为翻译后的 pandas 口径 (原式保存在 _formula_original)
    - 生成成功 → 队列落盘 + 种子状态回写; 失败 → 记录原因 (不再静默)
    - 醒目打印通知块 (自动化日报/工作台可见)

    Returns: [(factor_name, status("OK"|"SKIP"), detail), ...]
    """
    details = result.get("jq_candidate_details", [])
    if not details:
        return []
    gen = _load_jq_generator()
    queue = _load_jq_codegen_queue()
    now = datetime.now().isoformat()

    report = []
    for d in details:
        name = d.get("factor_name", "")
        formula = d.get("formula", "")
        if not formula:
            entry = entry_map.get(name)
            formula = getattr(entry, "expression", "") if entry else ""
        if not formula:
            msg = "无公式"
            report.append((name, "SKIP", msg))
            continue
        # ── 2026-08-22 修复: 队列数据卫生 ──
        # 因子已 jq_run_done (用户已回测, 结果在 Memory) 或已生成过代码
        # (pending_jq_run) 时跳过重复生成 — 旧逻辑无条件覆盖会重置
        # jq_run_done → pending, 用户重复回测已测因子 (forge_gen2 两日两次事故)。
        _prev = queue.get(name)
        if isinstance(_prev, dict):
            _st = _prev.get("status", "")
            if _st.startswith("jq_run_done") or _st == "pending_jq_run":
                report.append((name, "SKIP", f"队列已有状态 {_st}, 跳过重复生成"))
                continue
        is_seed = bool(d.get("seed_recheck"))
        meaning = d.get("hypothesis", "") or ""
        local_note = (
            "S5: grade=%s calmar=%.2f | 来源=%s%s"
            % (d.get("grade", ""), d.get("s5_calmar", 0) or 0,
               d.get("source", "generated"), " (种子重检)" if is_seed else "")
        )
        try:
            out_path, meta = gen.generate_standalone(
                name, formula, meaning=meaning, local_note=local_note,
            )
        except Exception as e:
            out_path, meta = None, {"reason": f"生成异常: {type(e).__name__}: {str(e)[:80]}"}
        if out_path:
            queue[name] = {
                "file": str(out_path),
                "formula": formula[:300],
                "meaning": meaning[:200],
                "generated_at": now,
                "status": "pending_jq_run",
                "seed_recheck": is_seed,
                "lookback": meta.get("lookback"),
                # v0.7 频率对称: S1 裁决口径 (weekly/daily), D+ 蒸馏按频率归因
                "natural_freq": d.get("natural_freq", "daily"),
            }
            if is_seed:
                ralph.record_seed_jq_code(name, str(out_path), True, "")
            report.append((name, "OK", str(out_path)))
        else:
            reason = (meta or {}).get("reason", "未知错误")
            queue[name] = {
                "file": "", "formula": formula[:300], "meaning": meaning[:200],
                "generated_at": now, "status": "skipped", "reason": reason,
                "seed_recheck": is_seed,
            }
            if is_seed:
                ralph.record_seed_jq_code(name, "", False, reason)
            report.append((name, "SKIP", reason))

    _save_jq_codegen_queue(queue)

    # ── 醒目通知块 (自动化 stdout / 工作台日报可见) ──
    if report:
        print("\n" + "=" * 62)
        print("  【JQ 代码自动生成】S5 通过因子 → 单因子回测代码 (v0.9)")
        print("=" * 62)
        for name, status, msg in report:
            if status == "OK":
                print(f"  [OK]   {name} → {Path(msg).name}")
            else:
                print(f"  [SKIP] {name} — {msg}")
        print("  已生成代码请复制到聚宽研究环境回测; 结果反馈后自动触发 D+ 蒸馏")
        print("=" * 62)
    return report


def run_ralph(
    csv_path: Path = None,
    pkl_path: Path = None,
    inject_seeds: bool = True,
    use_mmr: bool = True,
    generator: str = "gp_breed",
    use_llm: bool = False,
    evo_turns: int = 5,
    freq: str = "",  # P-20260830-001: "weekly" → LLM 周频语境
):
    """
    Ralph Loop 统一自进化流水线 (v0.5) — R→G→E→D + MAB + MMR
    
    流程:
      种子注入 → MAB方向选择 → R(检索) → G(GP育种+QualityGate+可选LLM) → E(S1-S5) → MMR选择 → JQ候选 → D(蒸馏)
      
    双引擎:
      - GP 育种 (默认): 表达式树交叉/变异, 从 Memory 模板进化, 即时可用
      - LLM 生成 (可选): 基于 Memory priors 构建 prompt, 写入文件等待手动提交
    """
    print("=" * 60)
    print("  V4 Pipeline: Ralph Loop 统一自进化 (v0.5)")
    print("  双引擎: GP育种 + LLM生成 | MAB方向选择 | MMR复合选择")
    print("=" * 60)

    from ralph_loop import RalphLoop
    from seed_injector import run_full_injection
    from mmr_selector import MMRSelector
    from llm_generator import LLMGenerator

    # ── Step 0: 种子注入 ──
    if inject_seeds:
        print("\n[Step 0] 王者种子注入 + 因子库合并...")
        try:
            inj_result = run_full_injection(verbose=True)
            print(f"  因子库: {inj_result['pool']['total']} 因子")
            print(f"  Memory: {inj_result['memory'].get('total_templates', '?')} 模板")
        except Exception as e:
            print(f"  [警告] 种子注入跳过: {e}")

    # ── Step 1: 加载因子库 ──
    print("\n[Step 1] 加载统一因子库 (含LLM释义)...")
    csv_path = csv_path or DATA_DIR / "unified_factor_pool.csv"
    if not csv_path.exists():
        csv_path = DATA_DIR / "passed_factor_pool.csv"

    metadata = load_factor_metadata(csv_path)
    entries, cat_map = build_factor_entries(metadata)

    mgr = LibraryOrthogonalityManager(data_dir=DATA_DIR)
    for fe in entries:
        mgr.factors[fe.name] = fe
        mgr._factor_names.append(fe.name)

    library_stats = mgr.get_library_stats()
    
    # 统计标注覆盖率
    has_hyp = sum(1 for fe in entries if getattr(fe, 'hypothesis', ''))
    has_logic = sum(1 for fe in entries if getattr(fe, 'logic', ''))
    has_paradigm = sum(1 for fe in entries if getattr(fe, 'paradigm', ''))
    print(f"  因子: {len(entries)} | 族: {library_stats.get('n_clusters', '?')} "
          f"| Red Sea: {library_stats.get('red_sea', {}).get('level', '?')}")
    print(f"  标注: hyp={has_hyp}/{len(entries)} logic={has_logic}/{len(entries)} paradigm={has_paradigm}/{len(entries)}")

    # ── Step 2: 候选准备 ──
    candidates = []
    entry_map = {fe.name: fe for fe in entries}
    for fe in entries:
        if getattr(fe, 'status', '') == 'champion_seed' or getattr(fe, 'source', '') == 'alpha_agent_v3_champion':
            candidates.append({
                "factor_name": fe.name,
                "formula": fe.expression,
                "paradigm": fe.paradigm,
                "hypothesis": getattr(fe, 'hypothesis', ''),
                "logic": getattr(fe, 'logic', ''),
                "ic": fe.ic,
                "icir": fe.icir,
                "source": fe.source,
                "status": fe.status,
            })
    if len(candidates) < 20:
        for fe in entries[-20:]:
            if fe.name not in {c["factor_name"] for c in candidates}:
                candidates.append({
                    "factor_name": fe.name,
                    "formula": fe.expression,
                    "paradigm": fe.paradigm,
                    "hypothesis": getattr(fe, 'hypothesis', ''),
                    "logic": getattr(fe, 'logic', ''),
                    "ic": fe.ic,
                    "icir": fe.icir,
                    "source": fe.source,
                    "status": fe.status,
                })

    print(f"  候选: {len(candidates)} 个 (含 {sum(1 for c in candidates if c.get('status') == 'champion_seed')} 王者种子)")

    # ── Step 3: Ralph Loop ──
    gen_label = f"GP育种 + LLM生成" if use_llm else generator
    print(f"\n[Step 3] Ralph Loop 自进化 ({gen_label})...")
    ralph = RalphLoop(data_dir=DATA_DIR)

    result = ralph.run(
        candidates=candidates,
        library_state=library_stats,
        generator=generator,
        max_candidates=10,
        evo_turns=evo_turns,
        freq=freq,
    )
    
    # ── Step 3b: LLM 生成 (如果启用, 并行产出 prompt 文件) ──
    llm_file = None
    if use_llm:
        print(f"\n[Step 3b] LLM 因子生成 (基于 Experience Memory 反馈)...")
        # 从检索结果构建 LLM prompt
        retrieve_context = result.get("phases", {}).get("retrieve", {})
        llm_gen = LLMGenerator()
        llm_gen.receive_context(
            success_templates=retrieve_context.get("success_templates", []),
            forbidden_directions=[],
            fsa_forbidden="",
            library_stats=library_stats,
            mab_direction=result.get("phases", {}).get("generate", {}).get("mab_direction", ""),
        )
        mab_dir = result.get("phases", {}).get("generate", {}).get("mab_direction", "")
        llm_file = llm_gen.write_prompt_file(target_paradigm=mab_dir)
        print(f"  📝 LLM prompt → {llm_file}")
        print(f"  请将上述文件内容提交给 LLM, 用 ralph.jq_feedback() 注入结果")

    # ── Step 4: MMR 复合选择 ──
    jq_candidates_raw = result.get("jq_candidates", [])
    if use_mmr and len(jq_candidates_raw) > 4:
        print(f"\n[Step 4] MMR 复合选择 (从 {len(jq_candidates_raw)} 个 JQ 候选中选最优子集)...")
        mmr = MMRSelector()
        # 丰富候选信息
        enriched = []
        for c in jq_candidates_raw:
            if isinstance(c, str):
                entry = entry_map.get(c)
                if entry:
                    enriched.append({
                        "factor_name": c,
                        "formula": getattr(entry, 'expression', ''),
                        "paradigm": getattr(entry, 'paradigm', ''),
                        "icir": getattr(entry, 'icir', 0) or 0,
                        "hypothesis": getattr(entry, 'hypothesis', ''),
                    })
            elif isinstance(c, dict):
                enriched.append(c)
        
        if len(enriched) >= 4:
            selected = mmr.select(enriched, top_k=4)
            jq_candidates = selected
            print(f"  MMR 选择: {len(selected)} 个互补因子")
            for s in selected:
                print(f"    - {s.name} [{s.paradigm}]")
        else:
            jq_candidates = jq_candidates_raw
    else:
        jq_candidates = jq_candidates_raw

    # ── v0.9: S5 通过 → 自动生成 JQ 单因子回测代码 + 队列落盘 ──
    jq_gen_report = _auto_generate_jq_code(result, entry_map, ralph)

    # ── 摘要 ──
    print("\n" + "=" * 60)
    print("  Ralph Loop v0.5 结果摘要")
    print("=" * 60)
    print(f"\n  检索 priors: {result['phases']['retrieve']['n_retrieved']}")
    print(f"  候选因子:   {result['phases']['generate']['n_candidates']}")
    mab_dir = result['phases'].get('generate', {}).get('mab_direction', 'default')
    print(f"  MAB 方向:   {mab_dir}")
    print(f"  JQ 候选:    {len(jq_candidates)} (MMR精选)")
    n_gen_ok = sum(1 for _, st, _ in jq_gen_report if st == "OK")
    if jq_gen_report:
        print(f"  JQ 代码:    自动生成 {n_gen_ok}/{len(jq_gen_report)} 个 (队列: data/jq_codegen_queue.json)")
    print(f"  Memory:     {ralph.memory.get_summary()}")
    if llm_file:
        print(f"  LLM prompt: {llm_file}")

    # ── 保存结果 ──
    output_path = DATA_DIR / f"ralph_loop_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary = result.get("summary", result)
    summary["jq_candidates_mmr"] = []
    for c in (jq_candidates if isinstance(jq_candidates, list) else []):
        if isinstance(c, str):
            summary["jq_candidates_mmr"].append({"name": c, "paradigm": "", "formula": ""})
        elif isinstance(c, dict):
            summary["jq_candidates_mmr"].append({
                "name": c.get("factor_name", c.get("name", "")),
                "paradigm": c.get("paradigm", ""),
                "formula": c.get("formula", ""),
            })
    summary["jq_codegen"] = [
        {"name": n, "status": st, "detail": msg} for n, st, msg in jq_gen_report
    ]
    if llm_file:
        summary["llm_prompt_file"] = str(llm_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  结果 → {output_path}")

    return result


# ═══════════════════════════════════════════════════════════
# CLI 入口
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="V4 Pipeline: paradigm_v4 → library_orthogonality → factor_model_cooptim",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_v4_pipeline.py --view          # 快速状态快照
  python run_v4_pipeline.py --diagnose      # 全量诊断报告
  python run_v4_pipeline.py --explore       # 生成探索上下文
  python run_v4_pipeline.py --diagnose --no-corr  # 诊断(跳过相关性,更快)
        """,
    )
    parser.add_argument(
        "--diagnose", action="store_true",
        help="全量诊断: 加载因子池 → 范式映射 → 聚类 → 探索建议"
    )
    parser.add_argument(
        "--explore", action="store_true",
        help="生成 AlphaAgent v3 探索上下文 (JSON)"
    )
    parser.add_argument(
        "--view", action="store_true",
        help="快速状态快照 (默认)"
    )
    parser.add_argument(
        "--no-corr", action="store_true",
        help="跳过相关性矩阵计算 (更快, 仅用于范式层面分析)"
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="因子池 CSV 路径 (默认: data/passed_factor_pool.csv)"
    )
    parser.add_argument(
        "--pkl", type=str, default=None,
        help="因子池 Pickle 路径 (默认: data/cache/existing_factor_pool.pkl)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="探索上下文输出路径 (默认: data/v4_exploration_context.json)"
    )
    parser.add_argument(
        "--ralph", action="store_true",
        help="Ralph Loop 自进化模式 (R→G→E→D)"
    )
    parser.add_argument(
        "--generator", type=str, default="gp_breed",
        choices=["gp_breed", "gp_evolve", "llm", "forge", "hybrid", "custom"],
        help="因子生成器 (默认: gp_breed, EvoTraj用 gp_evolve)"
    )
    parser.add_argument(
        "--evo-turns", type=int, default=5,
        help="EvoTraj 最大进化轮数 (仅 gp_evolve 模式, 默认: 5)"
    )
    parser.add_argument(
        "--freq", type=str, default="",
        choices=["", "weekly", "daily"],
        help="P-20260830-001: 分频语境生成 (weekly = LLM 周频语境 + weekly lane 裁决)"
    )

    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else None
    pkl_path = Path(args.pkl) if args.pkl else None
    output_path = Path(args.output) if args.output else None

    if args.ralph:
        run_ralph(csv_path, pkl_path, generator=args.generator, evo_turns=args.evo_turns,
                  freq=args.freq)
    elif args.explore:
        run_explore(csv_path, pkl_path, output_path)
    elif args.diagnose:
        run_diagnose(csv_path, pkl_path, with_correlation=not args.no_corr)
    else:
        run_view()
