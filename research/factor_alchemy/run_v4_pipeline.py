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

import sys, os, io, json, csv, pickle, gc, time, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Set
from collections import Counter, defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # e:/quant/

# ── 项目根目录 ──────────────────────────────────────────
ALCHEMY_DIR = Path(__file__).resolve().parent  # research/factor_alchemy/
RESEARCH_DIR = ALCHEMY_DIR.parent              # research/
PROJECT_ROOT = RESEARCH_DIR.parent             # e:/quant/
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

FREEZE_JSON_PATH = DATA_DIR / "library_freeze.json"
RETIRED_JSON_PATH = DATA_DIR / "library_retired.json"


def load_retired_names(retired_path: Path = None) -> Set[str]:
    """
    P-20260905-002: 读取退役因子名单 (低价值库存人工批量退役)。

    退役状态唯一真相源 = data/library_retired.json
    (retire_lowvalue_volamt_20260905.py 落盘)。
    回捞由人工指定并重新走 S1, 移出即恢复可见。
    """
    fp = retired_path or RETIRED_JSON_PATH
    if not fp.exists():
        return set()
    try:
        with open(fp, encoding="utf-8") as f:
            d = json.load(f)
        return {it.get("name", "") for it in d.get("retired", []) if it.get("name")}
    except Exception as e:
        print(f"  ⚠️ 退役区读取失败 ({e}), 按空名单处理 (不排除任何因子)")
        return set()


def load_frozen_names(freeze_path: Path = None) -> Set[str]:
    """
    P-20260905-001: 读取冷冻区因子名单。

    冻结状态唯一真相源 = data/library_freeze.json (library_eviction.py 落盘)。
    回捞由 eviction 脚本执行 (移出 freeze json + 恢复 status), 本函数只读名单;
    移出即自动恢复可见, 无需此处处理。
    """
    fp = freeze_path or FREEZE_JSON_PATH
    if not fp.exists():
        return set()
    try:
        with open(fp, encoding="utf-8") as f:
            d = json.load(f)
        return {it.get("name", "") for it in d.get("frozen", []) if it.get("name")}
    except Exception as e:
        print(f"  ⚠️ 冷冻区读取失败 ({e}), 按空名单处理 (不排除任何因子)")
        return set()


def load_factor_metadata(csv_path: Path = None, exclude_frozen: bool = True) -> List[Dict]:
    """
    从 passed_factor_pool.csv 加载因子元数据。

    P-20260905-001: exclude_frozen=True 时排除 library_freeze.json 中的冷冻因子 —
    ralph 候选准备 / corr 矩阵 / 正交性聚类 / MMR 上下文统一不再看见它们
    (冷冻因子不占容量、不当正交性参照)。
    """
    if csv_path is None:
        csv_path = DATA_DIR / "passed_factor_pool.csv"

    frozen_names = load_frozen_names() if exclude_frozen else set()
    retired_names = load_retired_names() if exclude_frozen else set()
    excluded = frozen_names | retired_names

    factors = []
    n_excluded = 0
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('name', '').strip() in excluded:
                n_excluded += 1
                continue
            factors.append(row)
    if n_excluded:
        print(f"  🧊 已排除冷冻/退役因子: {n_excluded} 个 (frozen {len(frozen_names)} + retired {len(retired_names)})")
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

# P-20260905-001: 因子值持久化缓存 (60日每股均值向量)
# 首次现场求值后按因子落盘 parquet, 后续轮次 O(1) 加载 — 把每轮固定成本
# (~0.5-1s/因子) 变成增量成本, 支撑 corr 矩阵 200 → 500 扩容。
FACTOR_VALUE_CACHE_DIR = DATA_DIR / "factor_value_cache"
FACTOR_VALUE_META = FACTOR_VALUE_CACHE_DIR / "meta.json"
_META_DATA_END_KEY = "__data_end__"

# 截断排序的 status 优先级 (数字越小越优先保留):
# candidate/champion_seed 是链上/种子因子优先; jq_* 已有裁决次之; reserve 最低。
# P-20260905-001 修复: 原排序键只按 |ICIR|, 新入库因子 ICIR≈0 会被优先踢出矩阵,
# 使最新鲜的因子反而对 DivDisc/聚类不可见 (静默故障)。
_STATUS_PRIORITY = {
    "candidate": 0, "champion_seed": 0, "recheck_s1_passed": 1,
    "s5_passed": 1, "jq_passed": 1, "jq_done": 1, "jq_run_done": 1,
    "jq_run_done_v2": 1, "jq_marginal": 2, "jq_failed_direction": 2,
    "reserve": 3,
}


def _factor_cache_path(name: str) -> Path:
    """因子名 → 缓存文件路径 (md5 短哈希, 规避非法文件名字符)"""
    safe = hashlib.md5(name.encode("utf-8")).hexdigest()[:16]
    return FACTOR_VALUE_CACHE_DIR / f"{safe}.parquet"


def _load_factor_value_cache_meta() -> Dict[str, str]:
    """返回缓存元数据 {因子名: data_end_date} (+ __data_end__ 全局校验键)"""
    if not FACTOR_VALUE_META.exists():
        return {}
    try:
        return json.loads(FACTOR_VALUE_META.read_text(encoding="utf-8"))
    except Exception:
        return {}


def purge_frozen_from_manager(mgr, frozen_names: Set[str]) -> int:
    """
    P-20260905-001: 从 LibraryOrthogonalityManager 清除 frozen 因子。

    mgr 从持久化状态 (library_orthogonality_state.json) 恢复时可能带 frozen 因子的
    旧注册/旧聚类/旧 corr 矩阵; entries 过滤后这些残留会让 stats/Red Sea/聚类
    继续"看见"它们 (冷冻因子不占容量、不当正交性参照)。返回清除计数。
    """
    n = 0
    for name in list(mgr.factors.keys()):
        if name in frozen_names:
            mgr.factors.pop(name, None)
            n += 1
    names = list(mgr._factor_names or [])
    keep_idx = [i for i, x in enumerate(names) if x not in frozen_names]
    n += len(names) - len(keep_idx)
    mgr._factor_names = [names[i] for i in keep_idx]
    # 旧 corr 矩阵与 _factor_names 对齐时同步过滤行/列 (跳过 recluster 的场景仍生效)
    if (mgr._corr_matrix is not None and len(names) > 0
            and mgr._corr_matrix.shape[0] == len(names)
            and len(keep_idx) < len(names)):
        mgr._corr_matrix = mgr._corr_matrix[np.ix_(keep_idx, keep_idx)]
    if mgr.clusters:
        new_clusters = []
        for c in mgr.clusters:
            c.factors = [f for f in (c.factors or []) if f not in frozen_names]
            if c.factors:
                new_clusters.append(c)
        mgr.clusters = new_clusters
    return n


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

    # 限制因子数量以加速 (P-20260905-001: 排序键统一 — status 优先, 同状态按 |ICIR|)
    if len(matched_entries) > max_factors:
        matched_entries = sorted(
            matched_entries,
            key=lambda e: (_STATUS_PRIORITY.get(e.status, 3), -abs(e.icir or 0)),
        )[:max_factors]

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
        # P-20260905-001: 现场计算上限 200 → 500 (配合因子值持久化缓存 + 向量化 corr)。
        # 排序键修复: status 优先 (candidate/champion_seed/jq_* 保留), 同状态按 |ICIR| —
        # 原实现只按 |ICIR| 降序, 新入库因子 ICIR≈0 会被优先截断踢出矩阵。
        cap = max_factors
        if len(with_formula) > cap:
            with_formula.sort(
                key=lambda e: (_STATUS_PRIORITY.get(e.status, 3), -abs(e.icir or 0)))
            with_formula = with_formula[:cap]

        print(f"  缓存为元数据格式 → 现场计算 {len(with_formula)} 个因子值 "
              f"(status优先+top|ICIR|, 上限 {cap}, 值缓存增量)...")
        t_fx = time.time()
        # P-20260905-001: 因子值持久化缓存 —
        #   60日窗口随数据滚动, 以当前数据截止日 data_end 为全局校验键;
        #   data_end 变化 → 整个缓存失效重算 (语义正确), 否则旧因子 O(1) 加载。
        data_end = str(ic_comp._price_df["trade_date"].max().date())
        cache_meta = _load_factor_value_cache_meta()
        cache_valid = cache_meta.get(_META_DATA_END_KEY) == data_end
        if not cache_valid:
            cache_meta = {_META_DATA_END_KEY: data_end}
        n_cache_hit = 0
        n_cache_new = 0
        for fe in with_formula:
            try:
                cp = _factor_cache_path(fe.name)
                if (cache_valid and cache_meta.get(fe.name) == data_end
                        and cp.exists()):
                    s = pd.read_parquet(cp)["factor_value"]
                    n_cache_hit += 1
                else:
                    fdf = ic_comp._eval_formula(str(fe.expression), fe.name)
                    if fdf is None or len(fdf) < 100:
                        dropped += 1
                        continue
                    # 最近60个交易日 → 每股均值向量 (对齐旧格式语义)
                    fdf = fdf.sort_values("trade_date")
                    recent = fdf.groupby("ts_code").tail(60)
                    s = recent.groupby("ts_code")["factor_value"].mean()
                    try:
                        FACTOR_VALUE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                        pd.DataFrame({"factor_value": s}).to_parquet(cp)
                        cache_meta[fe.name] = data_end
                        n_cache_new += 1
                    except Exception:
                        pass  # 缓存写失败不阻塞主流程 (下次重算)
                factor_series[fe.name] = s
            except Exception:
                dropped += 1
                continue
        if n_cache_new:
            try:
                FACTOR_VALUE_META.write_text(
                    json.dumps(cache_meta, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass
        print(f"  因子值计算完成 (耗时 {time.time()-t_fx:.1f}s, "
              f"缓存命中 {n_cache_hit}, 新增 {n_cache_new}, 异常 {dropped})")
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
    # P-20260905-001: 向量化 df.corr 替换 O(n²) 纯 Python 双循环 —
    # n=200 约 2 万对(分钟级), n=500 约 12.5 万对(原实现 10-40 分钟 → 秒级)。
    # 语义对齐: pairwise 完整观测, min_periods=20 (原 valid<20 → corr=0);
    # 常数因子 std<1e-12 → 原置 0, df.corr 得 NaN → nan_to_num 置 0, 等价。
    print(f"  构建 {n}×{n} 相关性矩阵 (向量化 df.corr)...")
    t_corr = time.time()
    try:
        # P-20260905-001 防御: 个别异常因子的 Series index 可能混入非 str 类型
        # (如把 trade_date 当股票键的公式), DataFrame 构造时索引排序会抛
        # 'Timestamp' vs 'str' 比较异常 → 统一 str 化 + 去重后再对齐。
        std_series = {}
        for k, v in factor_series.items():
            idx = pd.Index([str(x) for x in v.index])
            s = pd.Series(v.to_numpy(dtype=float), index=idx)
            if idx.has_duplicates:
                s = s[~idx.duplicated(keep='last')]
            std_series[k] = s
        align_df = pd.DataFrame(std_series)
        corr_matrix = align_df.corr(min_periods=20).to_numpy(dtype=float)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        np.fill_diagonal(corr_matrix, 1.0)
    except Exception as e:
        print(f"  ⚠️ 向量化 corr 失败 ({e}), 回退双循环")
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

    print(f"  相关性矩阵完成 (耗时 {time.time()-t_corr:.1f}s)")
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
    # P-20260905-001: 持久化状态可能含 frozen 残留, 清除后再注册
    _n_purged = purge_frozen_from_manager(mgr, load_frozen_names() | load_retired_names())
    if _n_purged:
        print(f"  🧊 正交性管理器已清除 frozen 残留: {_n_purged} 处")
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
    # P-20260905-001: 清除持久化状态中的 frozen 残留
    _n_purged = purge_frozen_from_manager(mgr, load_frozen_names() | load_retired_names())
    if _n_purged:
        print(f"  🧊 正交性管理器已清除 frozen 残留: {_n_purged} 处")
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
    return _load_script_module("gen_jq_generic")


def _load_script_module(name: str):
    """动态加载 scripts/<name>.py 为独立模块 (importlib, 无 sys.path 污染)。"""
    import importlib.util
    p = Path(__file__).parent.parent.parent / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, str(p))
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

    # ── P-20260909-001: S5 换手率成本敏感性影子检查 (纯影子零裁决) ──
    # 高换手候选标记金融毒性 (cost_drag > 50% excess), 只标记不否决;
    # 阈值观测 2 周后评估升级硬门禁 (PandaAI 金融毒性 × N&V 交易成本过滤器互证)。
    try:
        import sys as _sys
        from pathlib import Path as _Path
        _scripts_dir = str(_Path(__file__).resolve().parents[1] / "scripts")
        if _scripts_dir not in _sys.path:
            _sys.path.insert(0, _scripts_dir)
        from turnover_cost_probe import probe_candidates
        _cost_report = probe_candidates(details)
        if _cost_report:
            print("\n  [CostShadow] S5 候选换手成本影子标记 "
                  f"({len(_cost_report)} 个, 纯影子零裁决):")
            for _c in _cost_report:
                if _c.get("status") != "ok":
                    print(f"    [skip] {_c['factor_name']} — {_c['status']}")
                    continue
                _flag = "⚠ COST_SENSITIVE" if _c["cost_sensitive"] else "ok"
                print(
                    f"    [{_flag}] {_c['factor_name']} "
                    f"cost_drag={_c['cost_drag_annual']*100:.2f}%/yr "
                    f"vs excess={_c['excess_ref']*100:.2f}% "
                    f"(rho={_c['signal_rho']:.3f}, ann_turnover={_c['annual_turnover']:.1f}x)"
                )
    except Exception as _ce:
        print(f"  [CostShadow] 影子检查失败 (非阻塞): {_ce}")
    gen = _load_jq_generator()
    queue = _load_jq_codegen_queue()
    now = datetime.now().isoformat()

    report = []
    # v0.7.23 P0-f: 统计"S5 未通过但因 eligible_for_jq 入队"的候选 (泄漏可见化)
    _s5_fail_enq = []
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
        # ── v0.7.23 P0-f: S5 状态如实透传 ─────────────────────────────
        # 入队条件是 eligible_for_jq (默认 4/5 阶段), 在
        # FUXI_ELIGIBLE_REQUIRE_S5=off 时**不等价于 S5 通过** ⇒ 生成的
        # jq_s5_pass_*.py 会自相矛盾 (文件头写 grade=B / docstring 写"S5通过")。
        # 现把真实 s5_passed 透传, 由生成器决定前缀 (pass→jq_s5_pass_ /
        # fail→jq_s5_fail_) 并在注释头显式标注, 使产物**自证状态**。
        _s5_ok = bool(d.get("s5_passed", False))
        local_note = (
            "S5: grade=%s calmar=%.2f | 来源=%s%s"
            % (d.get("grade", ""), d.get("s5_calmar", 0) or 0,
               d.get("source", "generated"), " (种子重检)" if is_seed else "")
        )
        # ── P-20260917-026: natural_freq → window_freq (窗口语义如实化) ──
        # 病灶: generate_standalone 的 window_freq 取默认 "daily", 而候选
        #   natural_freq 可能是 "weekly" (Forge 在**周频宽表**上内生演化) ⇒
        #   ① 生成的文件头写「原始验证频率: daily」是**假话**;
        #   ② 同一字面窗口在 周线(生成层/S1 周频 lane) vs 日线(JQ 回测) 上语义
        #      差 5x (rolling(60): 60 周 vs 60 日) —— max_drawdown_duration
        #      JQ -39.9% 事故同型。
        # 现把真实频率透传; 由生成器按 FUXI_JQ_WEEKLY_SEMANTICS (默认 auto)
        # 决定是否启用「周频宽表求值」, 并在注释头如实标注语义模式。
        _nf = str(d.get("natural_freq", "") or "").strip().lower()
        _wf = "weekly" if _nf == "weekly" else "daily"
        try:
            out_path, meta = gen.generate_standalone(
                name, formula, meaning=meaning, local_note=local_note,
                s5_passed=_s5_ok, window_freq=_wf,
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
                # v0.7.23 P0-f: 真实 S5 状态留痕 (文件名前缀 + 队列元数据一致)
                "s5_passed": _s5_ok,
                "grade": d.get("grade", ""),
                # P-20260917-026: 窗口语义模式留痕 (weekly_panel / daily)
                # — D+ 回填与 JQ 结果对比时必须按此口径解读读数
                "window_freq": _wf,
                "window_semantics": (meta or {}).get("window_semantics", "daily"),
            }
            if not _s5_ok:
                _s5_fail_enq.append(name)
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

    # ── 2026-08-31: JQ 单因子 IC 快验批量脚本 (第二道证据门, 影子模式) ──
    # 每周频候选入队后自动生成批量快验脚本 (纯因子 IC 计算, 无组合模拟),
    # 用户可先跑快验 (秒级~分钟级) 再决定是否跑完整组合回测。
    try:
        _fic = _load_script_module("gen_jq_fast_ic")
        _fic_out = _fic.build_batch_for_weekly_pending()
        if _fic_out:
            print(f"\n  [FAST-IC] 周频候选快验批量脚本已生成: {Path(_fic_out).name}")
            print(f"  [FAST-IC] 先跑快验拿 JQ 周频 IC/ICIR (第二道门, "
                  f"ENFORCE={_fic.FAST_IC_GATE_ENFORCE} 影子模式), "
                  f"再过线者跑完整组合回测")
    except Exception as _fic_e:
        print(f"  [FAST-IC] 批量生成失败 (非阻塞): {_fic_e}")

    # ── 醒目通知块 (自动化 stdout / 工作台日报可见) ──
    if report:
        print("\n" + "=" * 62)
        print("  【JQ 代码自动生成】eligible_for_jq 因子 → 单因子回测代码 (v0.9)")
        print("=" * 62)
        for name, status, msg in report:
            if status == "OK":
                _tag = "  ⚠️ S5未过" if name in _s5_fail_enq else ""
                print(f"  [OK]   {name} → {Path(msg).name}{_tag}")
            else:
                print(f"  [SKIP] {name} — {msg}")
        if _s5_fail_enq:
            print("-" * 62)
            print(f"  ⚠️ 其中 {len(_s5_fail_enq)} 个候选 **S5 未通过** 仍入队 "
                  f"(因 eligible_for_jq 默认 4/5 阶段):")
            for _n in _s5_fail_enq:
                print(f"       - jq_s5_fail_{_n}_standalone.py")
            print("     真实 S5 通过数 = 0 时, 这些文件的 JQ 结果**不能**当作"
                  "S5 证据;")
            print("     要彻底切断该路径: export FUXI_ELIGIBLE_REQUIRE_S5=on "
                  "(注意: 会使 JQ 产出归零)")
        print("  已生成代码请复制到聚宽研究环境回测; 结果反馈后自动触发 D+ 蒸馏")
        print("=" * 62)
    return report


def _ensure_weekly_cache_fresh() -> None:
    """P-20260915-008: 周频缓存新鲜度守卫 (Forge / WeeklyLane / τ_w 校准集 共用数据源)。

    背景 (2026-09-15 实测):
      output/ap_batch/cache/weekly_prices.parquet 冻结于 2026-06-21 (落后日频 12 周),
      weekly_ohlc.parquet 同为 2026-06-21。根因 = scripts/ap_batch_factor_test.py::
      load_price_weekly() 命中缓存即 return, 无 TTL / 无日期校验 ⇒ 缓存一旦生成永不刷新。

    该缓存被三处消费:
      - ralph_loop 的 Forge 引擎 (约 60% 生成量) 的进化地形
      - weekly_lane.WeeklyLaneJudge (v0.7 周频裁决器)
      - lane_calibration (τ_w 周频门槛的校准集)

    策略: 陈旧即自动重建 (scripts/weekly_cache_refresh.py --rebuild, 纯 CSV 聚合约 13s,
    无 Tushare/LLM 调用, 且自带对拍与备份)。失败不阻塞主流程。
    关闭: env FUXI_WEEKLY_AUTO_REFRESH=off
    """
    if str(os.environ.get("FUXI_WEEKLY_AUTO_REFRESH", "on")).strip().lower() \
            in ("off", "0", "false", "no", "disable"):
        return
    try:
        import importlib.util
        _p = Path(__file__).resolve().parent.parent.parent / "scripts" / "weekly_cache_refresh.py"
        if not _p.exists():
            print("  [WeeklyCache] ⚠️ 未找到 weekly_cache_refresh.py ⇒ 跳过新鲜度守卫")
            return
        # importlib 绝对路径加载: 规避 scripts/ 与 factor_alchemy/ 同名模块的 sys.path[0] 循环导入
        _spec = importlib.util.spec_from_file_location("_weekly_cache_refresh_mod", _p)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)

        _d_max = _mod._daily_max_date()
        _w_max = _mod._parquet_max_week(_mod.WK_PRICES)
        if _d_max is None or _w_max is None:
            print("  [WeeklyCache] ⚠️ 无法判定 (日频或缓存缺失) ⇒ 跳过")
            return
        _lag = int((_mod._last_complete_week(_d_max) - _w_max).days // 7)
        if _lag >= _mod.STALE_WARN_WEEKS:
            print(f"  [WeeklyCache] 🔴 周频缓存陈旧 {_lag} 周 "
                  f"(缓存至 {_w_max.date()} / 日频至 {_d_max.date()}) ⇒ 自动重建 ...", flush=True)
            _mod.cmd_rebuild()
        else:
            print(f"  [WeeklyCache] ✅ 新鲜 (缓存至 {_w_max.date()}, 落后 {_lag} 周)")
    except Exception as _e:  # noqa: BLE001
        print(f"  [WeeklyCache] ⚠️ 新鲜度守卫失败 (非阻塞): {_e}")


def _ensure_data_freshness() -> None:
    """M18 (2026-09-28): 数据更新链新鲜度守卫 (fail-closed 闸门, 默认 shadow)。

    覆盖 18 通道: 行情/估值/复权/资金流/涨跌停/停牌/龙虎榜/两融/指数×3/行业/
    周频缓存/事件表 (预告/分红/财报/更名) / 源停更 (hk_hold)。
    读数与分档: scripts/data_freshness_guard.py (lag<=1 合规; >=2 warn; >=3 block)。

    FUXI_FRESHNESS_GATE = off | shadow (默认) | on
      · off   : 不动作
      · shadow: 打印「本应阻断」+ 落盘报告, 不阻断 (零行为影响)
      · on    : 存在 block 级 stale 通道 ⇒ fail-closed 阻断 (RuntimeError 冒泡)

    守卫自身异常不阻塞主流程 (仅告警)。历史教训: stk_limit 04-30 截断 /
    sw_daily 停更 / 指数冻结 09-09 / weekly 缓存冻结 12 周均为静默截断,
    本守卫的目标是「静默截断事件 = 0」(先报警, 再按开关阻断)。
    """
    _mode = str(os.environ.get("FUXI_FRESHNESS_GATE", "shadow")).strip().lower()
    if _mode not in ("off", "shadow", "on"):
        _mode = "shadow"
    if _mode == "off":
        return
    try:
        import importlib.util
        _p = Path(__file__).resolve().parent.parent.parent / "scripts" / "data_freshness_guard.py"
        if not _p.exists():
            print("  [Freshness] ⚠️ 未找到 data_freshness_guard.py ⇒ 跳过")
            return
        _spec = importlib.util.spec_from_file_location("_freshness_guard_mod", _p)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        # 演练钩子: FUXI_FRESHNESS_SIMULATE="channel=n,..." 注入模拟 (不改真实数据)
        _sim: dict = {}
        _sim_raw = os.environ.get("FUXI_FRESHNESS_SIMULATE", "").strip()
        if _sim_raw:
            for _item in _sim_raw.split(","):
                _k, _, _v = _item.strip().partition("=")
                try:
                    _sim[_k.strip()] = int(_v)
                except ValueError:
                    pass
        _report = _mod.run_check(quiet=True, sim=_sim)
        try:
            _mod.write_report(_report)
        except Exception:  # noqa: BLE001
            pass
        _summary = _report["summary"]
        print(f"  [Freshness] 通道读数: ok={_summary['ok']} warn={_summary['warn']} "
              f"block={_summary['block']} info={_summary['info']} "
              f"retired={_summary['retired']} (共 {_summary['total']})")
        if _report["gate"]["warn_channels"]:
            print(f"  [Freshness] 🟡 warn 通道: "
                  f"{', '.join(_report['gate']['warn_channels'])}")
        _blocked = _report["gate"]["blocked_channels"]
        if _blocked:
            _chains = ", ".join(_report["gate"]["blocked_chains"])
            if _mode == "shadow":
                print(f"  [Freshness/shadow] 🔴 本应阻断: {', '.join(_blocked)} "
                      f"(受影响链: {_chains}; FUXI_FRESHNESS_GATE=on 时 fail-closed)")
            else:
                _msg = (f"数据新鲜度 fail-closed: {', '.join(_blocked)} 超阈值 "
                        f"(受影响链: {_chains})")
                print(f"  [Freshness] 🔴 {_msg}")
                raise RuntimeError(_msg)
    except RuntimeError:
        raise
    except Exception as _e:  # noqa: BLE001
        print(f"  [Freshness] ⚠️ 守卫执行失败 (非阻塞): {_e}")


def run_ralph(
    csv_path: Path = None,
    pkl_path: Path = None,
    inject_seeds: bool = True,
    use_mmr: bool = True,
    generator: str = "gp_breed",
    use_llm: bool = False,
    evo_turns: int = 5,
    freq: str = "",  # P-20260830-001: "weekly" → LLM 周频语境
    # ── 2026-09-12 (W1) 产量旋钮 ──
    # 原实现: max_candidates 硬编码 10 且 CLI 无参数; forge 的 n_generations=5 /
    # top_k=1 / mine_top_n=0 全部硬编码在 ralph_loop 内 ⇒ forge 每轮产量被钉死在 1~5。
    #   max_candidates    : 全局候选上限 (llm 路径由 prompt 软约束, forge 路径为截断上界)
    #   forge_generations : forge 进化代数 (原 5)
    #   forge_topk        : forge 每代记录条数 (原 1 ⇒ 5 代只出 5 条)
    #   forge_mine_top_n  : 额外并入 forge.final_population 前列条数 (该资产原从未被流水线消费)
    max_candidates: int = 10,
    forge_generations: int = 5,
    forge_topk: int = 1,
    forge_mine_top_n: int = 0,
    run_id: str = "",
    experiment_id: str = "",
    resume_run: bool = False,
    paradigm: str = "",
    run_contract=None,
):
    """
    Ralph Loop 统一自进化流水线 (v0.5) — R→G→E→D + MAB + MMR
    
    流程:
      种子注入 → MAB方向选择 → R(检索) → G(GP育种+QualityGate+可选LLM) → E(S1-S5) → MMR选择 → JQ候选 → D(蒸馏)
      
    双引擎:
      - GP 育种 (默认): 表达式树交叉/变异, 从 Memory 模板进化, 即时可用
      - LLM 生成 (可选): 基于 Memory priors 构建 prompt, 写入文件等待手动提交
    """
    # R0 (2026-09-22): every mutating Ralph invocation, including Python callers,
    # enters the same single-writer and atomic lineage contract.
    if run_contract is None:
        from run_contract import RunContract
        _params = {
            "generator": generator,
            "paradigm": str(paradigm or ""),
            "freq": freq,
            "max_candidates": int(max_candidates),
            "evo_turns": int(evo_turns),
            "forge_generations": int(forge_generations),
            "forge_topk": int(forge_topk),
            "forge_mine_top_n": int(forge_mine_top_n),
            "inject_seeds": bool(inject_seeds),
            "use_mmr": bool(use_mmr),
            "resolved_caliber": {
                "s5_date_ceil": os.environ.get("FUXI_S5_DATE_CEIL", "auto"),
                "s5_cost_bps": os.environ.get("FUXI_S5_COST_BPS", "15"),
                "s5_min_coverage": os.environ.get("FUXI_S5_MIN_COVERAGE", "0.9"),
                "s5_tradable_gate": os.environ.get("FUXI_S5_TRADABLE_GATE", "on"),
                "s5_adj_price": os.environ.get("FUXI_S5_ADJ_PRICE", "on"),
                "s4_oos_mode": os.environ.get("FUXI_S4_OOS_MODE", "audit"),
                "forge_fitness": os.environ.get(
                    "FUXI_FORGE_FITNESS", "legacy"
                ),
            },
        }
        with RunContract(
            parameters=_params,
            run_id=run_id,
            experiment_id=experiment_id,
            resume=resume_run,
        ) as _contract:
            return run_ralph(
                csv_path=csv_path,
                pkl_path=pkl_path,
                inject_seeds=inject_seeds,
                use_mmr=use_mmr,
                generator=generator,
                use_llm=use_llm,
                evo_turns=evo_turns,
                freq=freq,
                max_candidates=max_candidates,
                forge_generations=forge_generations,
                forge_topk=forge_topk,
                forge_mine_top_n=forge_mine_top_n,
                run_id=run_id,
                experiment_id=experiment_id,
                resume_run=resume_run,
                paradigm=paradigm,
                run_contract=_contract,
            )

    run_contract.phase("setup", "running")
    print("=" * 60)
    print("  V4 Pipeline: Ralph Loop 统一自进化 (v0.5)")
    print("  双引擎: GP育种 + LLM生成 | MAB方向选择 | MMR复合选择")
    print("=" * 60)

    # P-20260915-008: Forge/WeeklyLane/τ_w 校准集共用的周频缓存新鲜度守卫
    # (陈旧即重建; env FUXI_WEEKLY_AUTO_REFRESH=off 关闭; 失败不阻塞)
    _ensure_weekly_cache_fresh()

    # M18 (2026-09-28): 数据更新链新鲜度守卫 (18 通道; 默认 shadow 只读报警;
    # FUXI_FRESHNESS_GATE=on 时 fail-closed 阻断; off 关闭)
    _ensure_data_freshness()

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
    # P-20260905-001: 清除持久化状态中的 frozen 残留 (旧注册/旧聚类/旧矩阵)
    _n_purged = purge_frozen_from_manager(mgr, load_frozen_names() | load_retired_names())
    if _n_purged:
        print(f"  🧊 正交性管理器已清除 frozen 残留: {_n_purged} 处")
    for fe in entries:
        mgr.factors[fe.name] = fe
        mgr._factor_names.append(fe.name)

    library_stats = mgr.get_library_stats()
    run_contract.phase(
        "setup", "completed", n_library_factors=len(entries),
        library_stats={
            "n_clusters": library_stats.get("n_clusters"),
            "red_sea_level": library_stats.get("red_sea", {}).get("level"),
        },
    )
    
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
    # S1-a a2 (2026-09-12): 单实例 MAB 注入。
    # 原实现 RalphLoop.__init__ 与 FactorModelCoOptimizer.__init__ 各自 new 一个
    # MABScheduler, 却共用 data/mab_scheduler_state.json ⇒ 后写者用自己内存里的
    # 旧值盲覆盖前者 (实测 total_pulls=317 而 sum(pulls)=484, 两本账不一致)。
    # 现改为二者共用同一个调度器实例。
    _cooptim = create_default_cooptimizer()
    print(f"  [MAB] 单实例注入: {len(_cooptim.mab.directions)} 臂, "
          f"pull_semantics={getattr(_cooptim.mab, 'pull_semantics', 'n/a')}")
    ralph = RalphLoop(data_dir=DATA_DIR, mab=_cooptim.mab)

    run_contract.phase("ralph", "running")
    result = ralph.run(
        candidates=candidates,
        library_state=library_stats,
        generator=generator,
        paradigm=paradigm or None,
        max_candidates=max_candidates,
        evo_turns=evo_turns,
        freq=freq,
        # 2026-09-12 (W1): forge 产量旋钮 (原为 ralph_loop 内硬编码 5/1/0)
        forge_generations=forge_generations,
        forge_topk=forge_topk,
        forge_mine_top_n=forge_mine_top_n,
    )
    _generated = result.get("phases", {}).get("generate", {}).get("candidates", [])
    _evaluation = result.get("phases", {}).get("evaluate", {})
    run_contract.record_candidates(_generated)
    run_contract.record_evaluation(
        _evaluation.get("summary", {}),
        _evaluation.get("results", []),
    )
    run_contract.phase(
        "ralph", "completed",
        n_candidates=len(_generated),
        n_s1_passed=int(result.get("phases", {}).get("evaluate", {})
                        .get("summary", {}).get("stage1_passed", 0) or 0),
        n_s5_passed=int(result.get("phases", {}).get("evaluate", {})
                        .get("summary", {}).get("stage5_passed", 0) or 0),
        evaluation_status=_evaluation.get("summary", {}).get("evaluation_status", {}),
        n_jq_candidates=int(result.get("summary", {}).get("n_jq_candidates", 0) or 0),
    )
    
    # ── Step 3b: LLM 生成 (如果启用, 并行产出 prompt 文件) ──
    llm_file = None
    if use_llm:
        print(f"\n[Step 3b] LLM 因子生成 (基于 Experience Memory 反馈)...")
        # 从检索结果构建 LLM prompt
        retrieve_context = result.get("phases", {}).get("retrieve", {})
        llm_gen = LLMGenerator()
        # 2026-09-12 (W4a 修复): 原实现把 forbidden_directions 传 []、fsa_forbidden 传 ""，
        # 使这条独立的 prompt-file 路径完全丢失禁止方向约束 (主路径 ralph_loop._phase_generate
        # 走的是 retrieve_result.get("forbidden_directions") 正确通道)。此处对齐主路径。
        _fd_ctx = retrieve_context.get("forbidden_directions", [])
        _fsa_ctx = retrieve_context.get("fsa_forbidden_context", "")
        if _fd_ctx or _fsa_ctx:
            print(f"  [Step 3b] 禁止上下文注入: forbidden_directions={len(_fd_ctx)} 条, "
                  f"fsa_forbidden={'有' if _fsa_ctx else '空'}")
        llm_gen.receive_context(
            success_templates=retrieve_context.get("success_templates", []),
            forbidden_directions=_fd_ctx,
            fsa_forbidden=_fsa_ctx,
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
    summary["run_id"] = run_contract.run_id
    summary["experiment_id"] = run_contract.experiment_id
    summary["run_manifest"] = str(run_contract.manifest_path)
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
    # ── 2026-09-12 (W1): 产量旋钮 (默认值 = 原硬编码行为, 零回归) ──
    parser.add_argument(
        "--max-candidates", type=int,
        default=int(os.environ.get("FUXI_MAX_CANDIDATES", "10")),
        help="候选数上限 (默认 10 / env FUXI_MAX_CANDIDATES)。llm 路径为 prompt 软约束, "
             "forge 路径为去重后截断上界"
    )
    parser.add_argument(
        "--forge-generations", type=int,
        default=int(os.environ.get("FUXI_FORGE_GENERATIONS", "5")),
        help="forge 进化代数 (默认 5 / env FUXI_FORGE_GENERATIONS)"
    )
    parser.add_argument(
        "--forge-topk", type=int,
        default=int(os.environ.get("FUXI_FORGE_TOPK", "1")),
        help="forge 每代记录条数 (默认 1 = 原行为; >1 时每代保留 fitness 前 k 名, "
             "产量上界 = generations x k)"
    )
    parser.add_argument(
        "--forge-mine-topn", dest="forge_mine_top_n", type=int,
        default=int(os.environ.get("FUXI_FORGE_MINE_TOPN", "0")),
        help="额外并入 forge.final_population 前列条数 (默认 0 = 关; 该种群池此前"
             "从未被流水线消费, 是 forge 产量最大的未利用资产)"
    )
    parser.add_argument(
        "--run-id", type=str, default="",
        help="R0: 显式运行 ID；已完成 ID 不可复用，避免结果被覆盖"
    )
    parser.add_argument(
        "--experiment-id", type=str, default="",
        help="R0: 预注册实验 ID；缺省按公开参数稳定生成"
    )
    parser.add_argument(
        "--paradigm", type=str, default="",
        help="显式实验范式；缺省为空时仍由 MAB 选择"
    )
    parser.add_argument(
        "--resume-run", action="store_true",
        help="R0: 仅恢复未终结的同 ID 运行；完成/失败运行仍禁止覆盖"
    )

    args = parser.parse_args()
    from pathlib import Path as _FuxiPath
    import sys as _FuxiSys
    _FuxiSys.path.insert(0, str(_FuxiPath(__file__).resolve().parents[2]))
    from fuxi_v1.legacy_writers import legacy_cli_guard as _FuxiGuard
    _fuxi_guard = _FuxiGuard(__file__, read_only=not (args.ralph or args.diagnose or args.explore))


    csv_path = Path(args.csv) if args.csv else None
    pkl_path = Path(args.pkl) if args.pkl else None
    output_path = Path(args.output) if args.output else None

    if args.ralph:
        run_ralph(csv_path, pkl_path, generator=args.generator, evo_turns=args.evo_turns,
                  freq=args.freq,
                  max_candidates=args.max_candidates,
                  forge_generations=args.forge_generations,
                  forge_topk=args.forge_topk,
                  forge_mine_top_n=args.forge_mine_top_n,
                  run_id=args.run_id,
                  experiment_id=args.experiment_id,
                  paradigm=args.paradigm,
                  resume_run=args.resume_run)
    elif args.explore:
        run_explore(csv_path, pkl_path, output_path)
    elif args.diagnose:
        run_diagnose(csv_path, pkl_path, with_correlation=not args.no_corr)
    else:
        run_view()
