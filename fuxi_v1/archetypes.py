"""Archetype 归并（第 0 层）：把 731 个深扫 paradigm 归并到 24 个 archetype。

匹配规则（两阶段，确定性）：
  阶段1  paradigm 字段精确语义匹配（paradigm 是主语义标签，非空且非 __none__ 时优先）；
  阶段2  阶段1 未命中（含 __none__ 与无关键词命中的冷门 paradigm）时，用
         label + economic_rationale 关键词回填。

为什么两阶段：单阶段联合文本匹配会让 economic_rationale 里的次要词（如"行业轮动
通过资金流判断"里的"资金流"）污染 paradigm 主语义，把"行业轮动"误归到"资金流"。

本模块只读 data/paradigm_archetype_map.json，不写回深扫候选池源文件。
"""
import json
from pathlib import Path

_WORKSPACE = Path(__file__).resolve().parents[1]
_MAP_PATH = _WORKSPACE / "data" / "paradigm_archetype_map.json"


def load_archetypes():
    with open(_MAP_PATH, encoding="utf-8") as f:
        m = json.load(f)
    return m["archetypes"], m.get("unclassified_id", "unclassified")


def _match_in_text(text, archetypes):
    """在单段文本里按 archetype 顺序找第一个命中关键词者。"""
    low = text.lower()
    for a in archetypes:
        for kw in a["keywords"]:
            if kw and kw.lower() in low:
                return a["id"]
    return None


# 模板 mechanism → archetype（第 2 层 campaign diversity budget 用）。campaign 层
# 的候选只有 hypothesis.mechanism（英文模板名 / 变异继承名），没有 stage1 提案的
# paradigm 中文语义标签。此表把已知模板机制语义映射到 archetype，冷门/随机机制
# 回退关键词匹配（_match_in_text），再回退 unclassified。
# 注意 typed_random_null 是 random_ast 臂的类型化随机零假设总体，无单一语义方向，
# 故意不在本表——diversity 记账按 unclassified 落账，且 random_ast 臂本身豁免 HHI
# 集中度惩罚（它是多样性注入源，非重复产出源）。
_MECHANISM_ARCHETYPE = {
    # 平静溢价族（结构因子 lead）
    "close_cv5": "volatility", "narrow_range": "geometry", "low_turnover": "liquidity",
    # legacy 六机制
    "price_persistence": "momentum", "short_reversal": "reversal",
    "price_range": "geometry", "volatility": "volatility",
    "volume_surprise": "price_volume", "price_volume_disagreement": "price_volume",
    # expanded 字段空间
    "amount_surprise": "price_volume", "price_amount_disagreement": "price_volume",
    "turnover_surprise": "liquidity", "turnover_stability": "liquidity",
    "intraday_range": "geometry", "close_location": "geometry",
    "open_surprise": "price_volume", "size_proxy": "fundamental",
    "price_zscore": "volatility",
    # P-20261018 diversity expansion 新机制（跨族 / 补缺失族群）
    "ma_trend": "technical", "ma_deviation": "technical",
    "vol_zscore": "volatility", "vol_regime": "volatility",
    "momentum_accel": "momentum",
    "volume_delta": "price_volume", "volume_zscore": "price_volume",
    "amount_zscore": "price_volume",
    "turnover_delta": "liquidity", "turnover_zscore": "crowding",
    "range_zscore": "geometry",
    "near_high": "behavioral", "near_low": "behavioral",
    "size_zscore": "fundamental",
    # 组合/配对机制
    "cross_year_regime_complement": "regime",
    "combined_independent_hypotheses": "interaction",
}


def mechanism_archetype(mechanism, archetypes=None, unclassified=None):
    """把 campaign 层的 hypothesis.mechanism 映射到 archetype。

    优先级：显式模板映射 > 关键词匹配（英文）> unclassified。
    typed_random_null（random_ast 臂）落 unclassified（多样性注入源，无单一语义）。
    """
    if archetypes is None or unclassified is None:
        archetypes, unclassified = load_archetypes()
    m = (mechanism or "").strip()
    if not m:
        return unclassified
    if m == "typed_random_null":
        return unclassified  # 类型化随机零假设，无单一语义方向
    hit = _MECHANISM_ARCHETYPE.get(m)
    if hit:
        return hit
    hit = _match_in_text(m, archetypes)
    return hit if hit else unclassified


def archetype_of(proposal, archetypes=None, unclassified=None):
    if archetypes is None or unclassified is None:
        archetypes, unclassified = load_archetypes()
    paradigm = (proposal.get("paradigm") or "").strip()
    # 阶段1：paradigm 字段（主语义）精确匹配
    if paradigm and paradigm.lower() != "__none__":
        hit = _match_in_text(paradigm, archetypes)
        if hit:
            return hit
    # 阶段2：label + rationale 回填（__none__ 组与冷门 paradigm）
    text = " ".join([proposal.get("label") or "", proposal.get("economic_rationale") or ""])
    hit = _match_in_text(text, archetypes)
    return hit if hit else unclassified


# ---------------------------------------------------------------- 跨族配对（P-20261018）
# combine/refine 的跨族约束复用 archetype 作为"族"标签。interaction（组合产物）
# 与 unclassified（随机零假设）本身是跨族混合，视为通配，不阻塞任何跨族配对。
_FAMILY_WILDCARD = {"interaction", "unclassified"}


def parent_family(candidate):
    """candidate -> archetype 族标签（combine/refine 跨族配对用）。"""
    return mechanism_archetype((candidate.get("hypothesis") or {}).get("mechanism"))


def cross_family(a, b):
    """两 candidate 是否跨族。通配族（interaction/unclassified）恒视为跨族友好。"""
    fa, fb = parent_family(a), parent_family(b)
    if fa in _FAMILY_WILDCARD or fb in _FAMILY_WILDCARD:
        return True
    return fa != fb


_FIELD_FAMILY = {
    "open": "price", "high": "price", "low": "price", "close": "price",
    "volume": "volume", "amount": "amount",
    "turnover_rate": "liquidity",
    "total_shares": "size", "float_shares": "size", "free_shares": "size",
}


def field_family(name):
    """字段 -> 字段族（refine 跨字段族编辑用）。未知字段自身成族。"""
    return _FIELD_FAMILY.get(name, name)
