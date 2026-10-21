"""types_core.registry —— 叶子注册表（字段级 + 因子级）+ 名称模式推断。

四级解析（resolve 优先级，自上而下）：
1. 外部增量覆盖（load_overrides 注入的 JSON）—— 新因子的人工标注落点，不改源码；
2. 显式表（FIELD_REGISTRY / FACTOR_REGISTRY）—— 当前全集精确标注；
3. 名称模式推断（INFER_PATTERNS）—— 未来算法迭代产生的新叶子名自动归类；
4. fallback —— 技术型/未分类兜底（direction=0、family=technical）。

增量调整流程（2026-10-21 定稿）：新因子出现时**默认无需任何动作**——先走名称模式
推断自动归类；只有模式推断不准（语义歧义 / 特殊经济含义）才需要人工介入，此时把
标注写进外部 JSON（FUXI_TYPES_OVERRIDE 或 load_overrides(path)），**不修改本文件**。
稳定后可选把 override 固化回 FACTOR_REGISTRY 并 bump REGISTRY_VERSION。

设计动机（2026-10-21 用户裁决）：97 因子清单只是**一次算法运行（factor_alchemy）的
产物**，不是注册表真值源。注册表必须可扩展——后续 factor_alchemy 重跑或 combine 迭代
出新因子名时，绝大多数被模式规则自动覆盖，只有极少数需要补一条显式标注，避免每轮
迭代都重标整个表。

direction 语义：**先验方向提示（note-only）**，只对学术界高度一致的因子（价值 +1、
市值 -1、低波动 -1、动量 +1、反转 -1、非流动性 +1）给非零，其余 0。运行时方向真值 =
截面 IC 符号检测（factor_mining/data_loader.evaluate_composite），本字段不参与硬约束。
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .spec import TypeSpec, _digest

# 注册表版本身份：改任何枚举/标注/模式 = 改此版本 = 缓存身份自动失效。
# P0 仅定义（供 metadata 记录），P1 起才接线进 operator_catalog_hash（见评审 R-F）。
REGISTRY_VERSION = "fuxi.types.1"


def _f(dim=(0, 0), form="ratio", family="none", horizon="none", shape="signed", direction=0):
    """紧凑构造器：减少 97 行标注的样板。"""
    return TypeSpec(dim=dim, form=form, family=family, horizon=horizon,
                    shape=shape, direction=direction)


# ---------------------------------------------------------------------------
# 字段级注册表（fuxi_v1 侧，10 字段）
# dim 沿用 dsl.py UNITS；family 用字段数据属性（与 archetypes.field_family 对齐）
# ---------------------------------------------------------------------------

FIELD_REGISTRY = {
    "open":           _f((1, 0), "level", "none",      "none", "positive", 0),
    "high":           _f((1, 0), "level", "none",      "none", "positive", 0),
    "low":            _f((1, 0), "level", "none",      "none", "positive", 0),
    "close":          _f((1, 0), "level", "none",      "none", "positive", 0),
    "volume":         _f((0, 1), "flow",  "liquidity", "fast", "positive", 0),
    "amount":         _f((1, 1), "flow",  "liquidity", "fast", "positive", 0),
    "turnover_rate":  _f((0, 0), "ratio", "liquidity", "fast", "positive", 0),
    "float_shares":   _f((0, 1), "level", "size",      "none", "positive", 0),
    "free_shares":    _f((0, 1), "level", "size",      "none", "positive", 0),
    "total_shares":   _f((0, 1), "level", "size",      "none", "positive", 0),
}


# ---------------------------------------------------------------------------
# 因子级注册表（factor_mining 侧，97 叶子显式全覆盖，按 family 分组）
# ---------------------------------------------------------------------------

FACTOR_REGISTRY = {
    # ---- size（市值，取对数前量纲为货币 (1,0)）----
    "ln_mcap":            _f((1, 0), "level", "size", "slow", "positive", -1),
    "ln_circulating_mcap": _f((1, 0), "level", "size", "slow", "positive", -1),

    # ---- value（估值比率）----
    "ep":                 _f(form="ratio", family="value", horizon="slow", direction=+1),
    "bp":                 _f(form="ratio", family="value", horizon="slow", direction=+1),
    "sp":                 _f(form="ratio", family="value", horizon="slow", direction=+1),
    "cfp":                _f(form="ratio", family="value", horizon="slow", direction=+1),
    "dp":                 _f(form="ratio", family="value", horizon="slow", direction=+1),

    # ---- quality（质量 / 盈利）----
    "f_score":            _f(form="score", family="quality", horizon="slow", shape="bounded", direction=+1),
    "roe":                _f(form="ratio", family="quality", horizon="slow", direction=+1),
    "roa":                _f(form="ratio", family="quality", horizon="slow", direction=+1),
    "roic":               _f(form="ratio", family="quality", horizon="slow", direction=+1),
    "gross_margin":       _f(form="ratio", family="quality", horizon="slow", direction=+1),
    "net_margin":         _f(form="ratio", family="quality", horizon="slow", direction=+1),
    "accruals":           _f(form="ratio", family="quality", horizon="slow", direction=-1),
    "ocf_quality":        _f(form="ratio", family="quality", horizon="slow", direction=+1),
    "debt_coverage":      _f(form="ratio", family="quality", horizon="slow", direction=+1),
    "earnings_stability": _f(form="dispersion", family="quality", horizon="slow", shape="positive", direction=+1),
    "asset_turnover":     _f(form="ratio", family="quality", horizon="slow", direction=+1),
    # quality proxy（语义模糊类，direction 保守 0）
    "accrual_quality_proxy":      _f(form="ratio", family="quality", horizon="slow", direction=0),
    "earnings_quality_proxy":     _f(form="ratio", family="quality", horizon="slow", direction=0),
    "cashflow_matching_proxy":    _f(form="ratio", family="quality", horizon="slow", direction=0),
    "capital_efficiency_proxy":   _f(form="ratio", family="quality", horizon="slow", direction=0),
    "operational_efficiency_proxy": _f(form="ratio", family="quality", horizon="slow", direction=0),
    "bargaining_power_proxy":     _f(form="ratio", family="quality", horizon="slow", direction=0),
    "earnings_consistency_proxy": _f(form="ratio", family="quality", horizon="slow", direction=0),

    # ---- growth（成长）----
    "rev_growth_yoy":     _f(form="ratio", family="growth", horizon="slow", direction=+1),
    "earnings_growth_yoy": _f(form="ratio", family="growth", horizon="slow", direction=+1),
    "asset_growth":       _f(form="ratio", family="growth", horizon="slow", direction=+1),

    # ---- momentum（动量 / 收益）----
    "ret_1m":             _f(form="return", family="momentum", horizon="fast", direction=+1),
    "ret_3m":             _f(form="return", family="momentum", horizon="medium", direction=+1),
    "ret_6m":             _f(form="return", family="momentum", horizon="medium", direction=+1),
    "ret_12m":            _f(form="return", family="momentum", horizon="slow", direction=+1),
    "ret_1m_skip1m":      _f(form="return", family="momentum", horizon="medium", direction=+1),
    "max_ret_1m":         _f(form="return", family="momentum", horizon="fast", direction=0),
    "min_ret_1m":         _f(form="return", family="momentum", horizon="fast", direction=0),

    # ---- reversal（反转）----
    "short_rev_5d":       _f(form="return", family="reversal", horizon="fast", direction=-1),
    "intraday_reversal":  _f(form="return", family="reversal", horizon="fast", direction=-1),
    "overnight_5d":       _f(form="return", family="reversal", horizon="fast", direction=0),
    "gap_up":             _f(form="return", family="reversal", horizon="fast", direction=0),
    "opening_gap_momentum": _f(form="return", family="reversal", horizon="fast", direction=0),
    "volume_climax_reversal": _f(form="return", family="reversal", horizon="fast", direction=-1),
    "ideal_amplitude_w_cut_reversal_20d": _f(form="return", family="reversal", horizon="medium", direction=-1),
    "ret_open_2d_proxy":  _f(form="return", family="reversal", horizon="fast", direction=0),

    # ---- liquidity（流动性 / 换手 / 资金流）----
    "avg_turnover_1m":    _f(form="flow", family="liquidity", horizon="fast", shape="positive", direction=-1),
    "avg_turnover_3m":    _f(form="flow", family="liquidity", horizon="medium", shape="positive", direction=-1),
    "turnover_std":       _f(form="dispersion", family="liquidity", horizon="fast", shape="positive", direction=0),
    "turnover_change":    _f(form="flow", family="liquidity", horizon="fast", shape="signed", direction=0),
    "abnormal_turnover":  _f(form="flow", family="liquidity", horizon="fast", shape="signed", direction=0),
    "turnover_cv_20d":    _f(form="dispersion", family="liquidity", horizon="fast", shape="positive", direction=0),
    "turnover_to_vol":    _f(form="ratio", family="liquidity", horizon="fast", shape="positive", direction=0),
    "amihud_illiq":       _f(form="ratio", family="liquidity", horizon="fast", shape="positive", direction=+1),
    "dollar_vol_20d":     _f(form="flow", family="liquidity", horizon="fast", shape="positive", direction=0),
    "dollar_vol_stability": _f(form="dispersion", family="liquidity", horizon="fast", shape="positive", direction=0),
    "volume_ratio":       _f(form="ratio", family="liquidity", horizon="fast", shape="positive", direction=0),
    "volume_stability":   _f(form="dispersion", family="liquidity", horizon="fast", shape="positive", direction=0),
    "davol_20":           _f(form="flow", family="liquidity", horizon="fast", shape="positive", direction=0),
    "vpt":                _f(form="flow", family="liquidity", horizon="medium", shape="signed", direction=0),
    "money_flow_20":      _f(form="flow", family="liquidity", horizon="fast", shape="signed", direction=0),
    "vm_diff":            _f(form="flow", family="liquidity", horizon="fast", shape="signed", direction=0),

    # ---- volatility（波动 / 离散度）----
    "vol_1m":             _f(form="dispersion", family="volatility", horizon="fast", shape="positive", direction=-1),
    "vol_3m":             _f(form="dispersion", family="volatility", horizon="medium", shape="positive", direction=-1),
    "downside_vol":       _f(form="dispersion", family="volatility", horizon="fast", shape="positive", direction=-1),
    "atr_14":             _f(form="dispersion", family="volatility", horizon="fast", shape="positive", direction=-1),
    "volatility_of_volatility": _f(form="dispersion", family="volatility", horizon="fast", shape="positive", direction=0),
    "range_consistency":  _f(form="dispersion", family="volatility", horizon="fast", shape="positive", direction=0),
    "high_low_range":     _f(form="dispersion", family="volatility", horizon="fast", shape="positive", direction=0),
    "skewness_20":        _f(form="dispersion", family="volatility", horizon="medium", shape="signed", direction=0),
    "kurtosis_20":        _f(form="dispersion", family="volatility", horizon="medium", shape="positive", direction=0),
    "skew_1m":            _f(form="dispersion", family="volatility", horizon="fast", shape="signed", direction=0),

    # ---- sentiment（情绪）----
    "panic_selling":      _f(form="return", family="sentiment", horizon="fast", direction=-1),
    "attention_decay":    _f(form="flow", family="sentiment", horizon="fast", shape="positive", direction=0),

    # ---- technical（技术指标）----
    "streak":             _f(form="score", family="technical", horizon="fast", shape="signed", direction=0),
    "rsi_14":             _f(form="bounded", family="technical", horizon="fast", shape="bounded", direction=0),
    "cci_14":             _f(form="bounded", family="technical", horizon="fast", shape="bounded", direction=0),
    "bias_20":            _f(form="bounded", family="technical", horizon="medium", shape="signed", direction=0),
    "roc_20":             _f(form="bounded", family="technical", horizon="medium", shape="signed", direction=0),
    "plrc_12":            _f(form="score", family="technical", horizon="fast", shape="bounded", direction=0),
    "psy_12":             _f(form="score", family="technical", horizon="fast", shape="bounded", direction=0),
    "vr_26":              _f(form="bounded", family="technical", horizon="medium", shape="positive", direction=0),
    "vroc_12":            _f(form="bounded", family="technical", horizon="fast", shape="signed", direction=0),
    "boll_pct_b":         _f(form="bounded", family="technical", horizon="medium", shape="bounded", direction=0),
    "sharpe_20":          _f(form="ratio", family="technical", horizon="medium", shape="signed", direction=+1),
    "bull_power":         _f(form="level", family="technical", horizon="fast", shape="signed", direction=+1),
    "bear_power":         _f(form="level", family="technical", horizon="fast", shape="signed", direction=-1),
    "high_52w_rank":      _f(form="bounded", family="technical", horizon="slow", shape="bounded", direction=+1),
    "high_52w_dist":      _f(form="level", family="technical", horizon="slow", shape="signed", direction=-1),
    "trend_persistence_score": _f(form="score", family="technical", horizon="medium", shape="bounded", direction=0),
    "trend_smoothness":   _f(form="dispersion", family="technical", horizon="medium", shape="positive", direction=0),
    "max_drawdown_duration": _f(form="level", family="technical", horizon="slow", shape="positive", direction=-1),
    "tvma_20":            _f(form="level", family="technical", horizon="medium", shape="signed", direction=0),
    "price_1m":           _f(form="level", family="technical", horizon="fast", shape="positive", direction=0),
    "rank_1m":            _f(form="bounded", family="technical", horizon="fast", shape="bounded", direction=0),
    "relative_spread_proxy": _f(form="ratio", family="technical", horizon="fast", shape="positive", direction=0),

    # ---- event（事件驱动）----
    "post_earnings_stability": _f(form="dispersion", family="event", horizon="medium", shape="positive", direction=0),
    "earnings_volume_drift": _f(form="flow", family="event", horizon="medium", shape="signed", direction=0),
    "earnings_season_vol_div": _f(form="ratio", family="event", horizon="slow", shape="positive", direction=0),
    "post_list_external_flow_days": _f(form="flow", family="event", horizon="slow", shape="positive", direction=0),
}


# ---------------------------------------------------------------------------
# 名称模式推断（未来新叶子自动归类；当前 97 个已走显式表，此层是兜底）
# 每条 = (编译后的正则, TypeSpec 片段 dict)。按列表顺序首条命中即返回。
# ---------------------------------------------------------------------------

def _pat(pattern, **spec):
    """模式规则：正则 + TypeSpec 片段。"""
    return (re.compile(pattern), spec)


INFER_PATTERNS = [
    _pat(r"^ln_(circulating_)?mcap$", dim=(1, 0), form="level", family="size", horizon="slow", shape="positive", direction=-1),
    _pat(r"^(ep|bp|sp|cfp|dp)$", form="ratio", family="value", horizon="slow", direction=+1),
    _pat(r"^f_score$", form="score", family="quality", horizon="slow", shape="bounded", direction=+1),
    _pat(r"^(roe|roa|roic)$", form="ratio", family="quality", horizon="slow", direction=+1),
    _pat(r"(gross_margin|net_margin|accrual|ocf_quality|debt_coverage|earnings_stability|asset_turnover)", form="ratio", family="quality", horizon="slow"),
    _pat(r".*(rev|earnings|asset)_growth.*", form="ratio", family="growth", horizon="slow", direction=+1),
    _pat(r"^ret_", form="return", family="momentum", horizon="medium", direction=+1),
    _pat(r"(short_rev|overnight|gap_up|intraday_reversal|.*reversal.*|opening_gap)", form="return", family="reversal", horizon="fast"),
    _pat(r".*(_vol|volatility|atr_|high_low_range|skewness|kurtosis|skew_).*", form="dispersion", family="volatility", horizon="fast", shape="positive"),
    _pat(r"(turnover|amihud|dollar_vol|davol|money_flow|vpt|vm_diff|volume_ratio|volume_stability)", form="flow", family="liquidity", horizon="fast", shape="positive"),
    _pat(r"^(rsi_|cci_|bias_|roc_|vr_|vroc_|psy_|plrc_|boll_)", form="bounded", family="technical", horizon="fast", shape="bounded"),
    _pat(r".*_proxy$", form="ratio", family="quality", horizon="slow"),
    _pat(r".*_score$", form="score", family="technical", horizon="medium", shape="bounded"),
    _pat(r".*(panic|attention|sentiment).*", form="return", family="sentiment", horizon="fast"),
    _pat(r".*(earnings|post_|season).*", form="ratio", family="event", horizon="medium"),
]


# ---------------------------------------------------------------------------
# 外部增量覆盖（新因子的人工标注落点，不改源码）
# ---------------------------------------------------------------------------

_override_registry: dict = {}


def load_overrides(path=None) -> dict:
    """加载/重载外部增量覆盖 JSON。

    格式：``{"因子名": {TypeSpec dict 字段...}}``，可只给需要覆盖的字段（其余走默认）。
    path=None 时读环境变量 FUXI_TYPES_OVERRIDE；文件不存在则清空覆盖（空 dict）。

    典型用途：新因子名称模式推断不准时，把精确标注写进 JSON 增量追加，
    无需修改本文件。调用后 resolve() 立即生效，registry_digest() 随之变化。
    """
    global _override_registry
    if path is None:
        path = os.environ.get("FUXI_TYPES_OVERRIDE", "")
    if not path or not Path(path).exists():
        _override_registry = {}
        return _override_registry
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for name, spec in raw.items():
        out[name] = TypeSpec.from_dict(spec) if isinstance(spec, dict) else spec
    _override_registry = out
    return _override_registry


def overrides() -> dict:
    """当前生效的外部覆盖（只读视图）。"""
    return dict(_override_registry)


def resolve(name: str, kind: str = "factor") -> TypeSpec:
    """四级解析：外部覆盖 > 显式表 > 名称模式 > fallback。

    kind = "field" | "factor"，决定先查哪个显式表（两者可同时命中时以 kind 优先）。
    """
    if name in _override_registry:
        return _override_registry[name]
    table = FIELD_REGISTRY if kind == "field" else FACTOR_REGISTRY
    if name in table:
        return table[name]
    # 反向表兜底（如字段名被当因子查、或反之）
    other = FACTOR_REGISTRY if kind == "field" else FIELD_REGISTRY
    if name in other:
        return other[name]
    for rx, spec in INFER_PATTERNS:
        if rx.search(name):
            return TypeSpec(**spec)
    # fallback：技术型 / 未分类 / 方向未知
    return TypeSpec(dim=(0, 0), form="ratio", family="technical",
                    horizon="none", shape="signed", direction=0)


def factor_names() -> list:
    return sorted(FACTOR_REGISTRY.keys())


def field_names() -> list:
    return sorted(FIELD_REGISTRY.keys())


def validate() -> list:
    """schema 校验：全部显式注册项构造期已校验（TypeSpec.__post_init__）。
    返回空列表 = 通过；此处再做一次整体断言，暴露任何遗漏。"""
    errors = []
    for table, label in ((FIELD_REGISTRY, "field"), (FACTOR_REGISTRY, "factor")):
        for name, ts in table.items():
            if not isinstance(ts, TypeSpec):
                errors.append(f"{label}.{name}: not a TypeSpec")
    # 名称模式片段也必须能构造出合法 TypeSpec
    for i, (rx, spec) in enumerate(INFER_PATTERNS):
        try:
            TypeSpec(**spec)
        except AssertionError as exc:
            errors.append(f"pattern[{i}] ({rx.pattern}): {exc}")
    return errors


def registry_digest() -> str:
    """注册表内容摘要（改标注/模式/枚举/覆盖 = 摘要变化 = 缓存身份变化）。"""
    payload = {
        "version": REGISTRY_VERSION,
        "fields": {k: v.to_dict() for k, v in sorted(FIELD_REGISTRY.items())},
        "factors": {k: v.to_dict() for k, v in sorted(FACTOR_REGISTRY.items())},
        "patterns": [[rx.pattern, spec] for rx, spec in INFER_PATTERNS],
    }
    if _override_registry:
        payload["overrides"] = {k: v.to_dict() for k, v in sorted(_override_registry.items())}
    return _digest(payload)
