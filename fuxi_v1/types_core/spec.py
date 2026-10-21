"""types_core.spec —— TypeSpec 数据结构 + 类型格运算。

纯 stdlib（不依赖 fuxi_v1 / factor_mining 任何模块），是两侧共享的类型核心。
这里的"类型格"分两层：

- **量纲层 dim**：(price_exp, shares_exp) 指数对，沿用 fuxi_v1 dsl.py 的 UNITS 代数。
  字段级区分度良好（close=(1,0)、volume=(0,1)、amount=(1,1)），但对预计算的因子级
  叶子几乎全退化到 (0,0)。
- **语义层 form/family/horizon/shape/direction**：因子级判别的主轴。form 是测度形态
  （level/ratio/return/flow/dispersion/score/bounded/ranked），family 是经济角色
  （value/quality/momentum/...），horizon 是调仓频率语义，shape 是有界性/符号性
  （div 分母安全性），direction 是先验单调方向（note-only，运行时真值 = IC 符号检测）。

设计约束（与方案对齐）：
- 硬约束只管"任何经济叙事下都无意义的组合"（如 add(score, price)）；
- 软约束管"可疑"（跨 family 相加、horizon 跨档）；
- mul/div 跨 family 放开（交互项是因子工程的意义来源）。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

# ---------------------------------------------------------------------------
# 封闭枚举域（改枚举 = 改 REGISTRY_VERSION = 缓存身份自动失效，见 registry.py）
# ---------------------------------------------------------------------------

# 测度形态：因子级判别主轴
FORMS = (
    "level",       # 绝对水平 / 水平距离（ln_mcap, price_1m, high_52w_dist）
    "ratio",       # 比率 / 财务比率（ep, roe, turnover_to_vol）
    "return",      # 收益（ret_1m, short_rev_5d, overnight_5d）
    "flow",        # 流量（dollar_vol_20d, money_flow_20, vpt）
    "dispersion",  # 离散度（vol_1m, turnover_std, atr, skewness）
    "score",       # 序数分（f_score, psy_12, trend_persistence_score）
    "bounded",     # 有界指标（rsi_14, boll_pct_b, bias_20）
    "ranked",      # 已标准化（rank/zscore 的输出）
)

# 经济角色：软约束与诊断主轴
FAMILIES = (
    "value", "quality", "growth", "momentum", "reversal", "liquidity",
    "volatility", "size", "sentiment", "technical", "event",
    "mixed",        # 组合产物跨族（add(value, momentum) 等）
    "none",         # 未分类 / 字段级原材料
)

# 调仓频率语义
HORIZONS = ("fast", "medium", "slow", "none")

# 有界性/符号性：div 分母安全性 + add/sub 形状合并
SHAPES = ("signed", "positive", "bounded")

# 先验单调方向（note-only；运行时真值 = 截面 IC 符号检测）
DIRECTIONS = (-1, 0, 1)


# ---------------------------------------------------------------------------
# 量纲格单点真值（C2 治理）：unit 名 -> 量纲指数对。
# 旧位置 dsl.py:19 / data.py:56 / shared_catalog / full_market_input / datasets 的
# 硬编码 unit 枚举与 dim 映射一律从这里派生，改量纲格只改这一处。
# ---------------------------------------------------------------------------

UNIT_DIMENSIONS = {
    "price": (1, 0),     # 价格：1 单位价格、0 单位股数
    "shares": (0, 1),    # 股数：0 单位价格、1 单位股数
    "ratio": (0, 0),     # 比率/无量纲
    "currency": (1, 1),  # 金额：价格×股数
}
VALID_UNITS = frozenset(UNIT_DIMENSIONS)


@dataclass(frozen=True)
class TypeSpec:
    """一个叶子或子表达式的类型。所有字段进 schema 校验，改枚举即报错。"""

    dim: tuple = (0, 0)          # (price_exp, shares_exp) — 量纲指数对
    form: str = "ratio"          # 测度形态（FORMS）
    family: str = "none"         # 经济角色（FAMILIES）
    horizon: str = "none"        # 调仓频率（HORIZONS）
    shape: str = "signed"        # signed|positive|bounded（SHAPES）
    direction: int = 0           # +1|-1|0 先验方向（DIRECTIONS）

    def __post_init__(self):
        # schema 校验：任何非法枚举在构造期即抛，注册表加载时统一暴露。
        _validate(self)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TypeSpec":
        return cls(**d)

    def with_(self, **kw) -> "TypeSpec":
        """返回改指定字段的副本（frozen dataclass 的便捷替换）。"""
        return TypeSpec(**{**self.to_dict(), **kw})


def _validate(ts: TypeSpec):
    assert isinstance(ts.dim, tuple) and len(ts.dim) == 2 \
        and all(isinstance(x, int) for x in ts.dim), f"dim must be (int, int), got {ts.dim!r}"
    assert ts.form in FORMS, f"unknown form: {ts.form!r}"
    assert ts.family in FAMILIES, f"unknown family: {ts.family!r}"
    assert ts.horizon in HORIZONS, f"unknown horizon: {ts.horizon!r}"
    assert ts.shape in SHAPES, f"unknown shape: {ts.shape!r}"
    assert ts.direction in DIRECTIONS, f"unknown direction: {ts.direction!r}"


# ---------------------------------------------------------------------------
# 量纲代数（dim 层，纯函数）
# ---------------------------------------------------------------------------

def dim_add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def dim_sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def dim_mul(a, b):
    return (a[0] + b[0], a[1] + b[1])


def dim_div(a, b):
    return (a[0] - b[0], a[1] - b[1])


def dim_square(a):
    return (2 * a[0], 2 * a[1])


def dim_neg(a):
    return a


def is_dimensionless(dim):
    return dim == (0, 0)


# ---------------------------------------------------------------------------
# 语义层合并规则（form / family / shape）
# ---------------------------------------------------------------------------

def merge_shape(a: str, b: str) -> str:
    """add/sub 两侧 shape 合并。同 shape 保持；不同 shape 保守退 signed。

    positive + positive -> positive（正数相加仍正）
    bounded + bounded -> bounded（有界相加仍有界）
    其余混搭（positive+signed、bounded+signed、bounded+positive）-> signed（边界不可控）。
    """
    if a == b:
        return a
    return "signed"


def merge_family(a: str, b: str) -> str:
    """add/sub 输出 family。同 family 保持，跨 family 标 mixed。"""
    if a == b:
        return a
    if "none" in (a, b):
        # none 是"未分类"，与任何具体族相加退化为另一侧的族（不引入 mixed 噪音）。
        return a if b == "none" else b
    return "mixed"


def merge_horizon(a: str, b: str) -> str:
    """add/sub 输出 horizon。同档保持；跨档取更慢的一档（慢频率主导）。

    fast+medium -> medium；fast+slow -> slow；medium+slow -> slow；none 忽略。
    """
    order = {"fast": 0, "medium": 1, "slow": 2}
    if a == "none":
        return b
    if b == "none":
        return a
    return a if order[a] >= order[b] else b


def _digest(obj) -> str:
    """内容寻址摘要（registry 版本身份用）。纯 stdlib，稳定排序。"""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
