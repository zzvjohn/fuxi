"""
Factor Forge 算子原语
======================
定义 GP 表达式树中的所有原子操作。

算子分类:
  A. 算术: add, sub, mul, div, sqrt, abs, log, neg, inv, square, cube
  B. 时间序列(rolling): ts_sum, ts_mean, ts_std, ts_min, ts_max,
                          ts_delta, ts_pct, ts_rank, ts_zscore, ts_ema_decay
  C. 截面: rank, zscore, scale
  D. 输入变量: open, high, low, close, volume, vwap, ret, turnover(volume代理)

每个原语:
  - arity: 参数个数 (0=终端, 1=一元, 2=二元)
  - func: 计算函数
  - name: 显示名 (用于表达式字符串)
  - complexity: 复杂度惩罚 (表达式过于复杂会扣分)
"""

import numpy as np
from typing import List, Dict, Any, Callable


# ═══════════════════════════════════════════════════════
# 算子元数据
# ═══════════════════════════════════════════════════════

class Primitive:
    """一个GP原语操作"""
    def __init__(self, name: str, func: Callable, arity: int,
                 complexity: float = 1.0, is_input: bool = False):
        self.name = name
        self.func = func
        self.arity = arity
        self.complexity = complexity
        self.is_input = is_input

    def __repr__(self):
        return f"Primitive({self.name}, arity={self.arity})"


# ═══════════════════════════════════════════════════════
# A. 算术原语
# ═══════════════════════════════════════════════════════

def _safe_add(a, b): return a + b
def _safe_sub(a, b): return a - b
def _safe_mul(a, b): return a * b
def _safe_div(a, b):
    denom = np.where(np.abs(b) < 1e-8, np.nan, b)
    return a / denom
def _safe_sqrt(x): return np.sqrt(np.maximum(x, 0))
def _safe_abs(x): return np.abs(x)
def _safe_log(x): return np.log(np.maximum(x, 1e-8))
def _safe_neg(x): return -x
def _safe_inv(x):
    return np.where(np.abs(x) < 1e-8, np.nan, 1.0 / x)
def _safe_square(x): return x * x
def _safe_cube(x): return x * x * x
def _safe_sign(x): return np.sign(x)


ARITHMETIC_PRIMITIVES = [
    Primitive("add", _safe_add, 2, 0.5),
    Primitive("sub", _safe_sub, 2, 0.5),
    Primitive("mul", _safe_mul, 2, 1.0),
    Primitive("div", _safe_div, 2, 1.5),
    Primitive("sqrt", _safe_sqrt, 1, 0.5),
    Primitive("abs", _safe_abs, 1, 0.5),
    Primitive("log", _safe_log, 1, 1.0),
    Primitive("neg", _safe_neg, 1, 0.3),
    Primitive("inv", _safe_inv, 1, 1.0),
    Primitive("square", _safe_square, 1, 0.3),
    Primitive("sign", _safe_sign, 1, 0.3),
]

# ═══════════════════════════════════════════════════════
# B. 时间序列(rolling)原语
# ═══════════════════════════════════════════════════════

def _ts_window(x, window):
    """沿时间轴(轴0)滑窗得到 (T-w+1, N, w), 修复 ravel 跨股票污染 bug.
    x: (T, N). 返回滑窗数组与窗口大小 w."""
    w = int(np.clip(window, 2, 120))
    arr = np.asarray(x, dtype=float)
    sw = np.lib.stride_tricks.sliding_window_view(arr, w, axis=0)
    return sw, w, arr

def _ts_sum(x, window):
    """滚动求和 (逐股, 向量化)"""
    sw, w, arr = _ts_window(x, window)
    out = np.full(arr.shape, np.nan)
    out[w - 1:] = np.nansum(sw, axis=-1)
    return out

def _ts_mean(x, window):
    """滚动均值 (逐股, 向量化)"""
    sw, w, arr = _ts_window(x, window)
    out = np.full(arr.shape, np.nan)
    out[w - 1:] = np.nanmean(sw, axis=-1)
    return out

def _ts_std(x, window):
    """滚动标准差 (逐股, 向量化)"""
    sw, w, arr = _ts_window(x, window)
    out = np.full(arr.shape, np.nan)
    out[w - 1:] = np.nanstd(sw, axis=-1)
    return out

def _ts_min(x, window):
    """滚动最小值 (逐股, 向量化)"""
    sw, w, arr = _ts_window(x, window)
    out = np.full(arr.shape, np.nan)
    out[w - 1:] = np.nanmin(sw, axis=-1)
    return out

def _ts_max(x, window):
    """滚动最大值 (逐股, 向量化)"""
    sw, w, arr = _ts_window(x, window)
    out = np.full(arr.shape, np.nan)
    out[w - 1:] = np.nanmax(sw, axis=-1)
    return out

def _ts_delta(x, window):
    """滚动差值 (逐股, 向量化)"""
    w = int(np.clip(window, 2, 120))
    arr = np.asarray(x, dtype=float)
    out = np.full(arr.shape, np.nan)
    out[w:] = arr[w:] - arr[:-w]
    return out

def _ts_pct(x, window):
    """滚动变化率 (逐股, 向量化)"""
    w = int(np.clip(window, 2, 120))
    arr = np.asarray(x, dtype=float)
    out = np.full(arr.shape, np.nan)
    prev = arr[:-w]
    cur = arr[w:]
    out[w:] = np.where(np.abs(prev) < 1e-8, np.nan, (cur - prev) / np.abs(prev))
    return out

def _ts_rank(x, window):
    """滚动排名(窗口内当前值百分位), 向量化, 逐股.
    旧实现用 rolling().apply(python lambda) 极慢且受 ravel 污染; 现用 sliding_window_view."""
    w = int(np.clip(window, 2, 120))
    sw, _, arr = _ts_window(x, window)
    last = sw[..., -1:]                       # 当前值
    less = (sw < last).sum(axis=-1)          # 窗口内严格小于当前值的个数
    rank_pct = (less + 0.5) / w              # 当前值百分位 (0~1)
    out = np.full(arr.shape, np.nan)
    out[w - 1:] = rank_pct
    return out

def _ts_zscore(x, window):
    """滚动 z-score (逐股, 向量化)"""
    sw, w, arr = _ts_window(x, window)
    m = np.nanmean(sw, axis=-1)
    s = np.nanstd(sw, axis=-1)
    out = np.full(arr.shape, np.nan)
    out[w - 1:] = (arr[w - 1:] - m) / np.where(s == 0, np.nan, s)
    return out

def _ts_ema_decay(x, window):
    """EMA 与当前值的差异 (类似 MACD 信号, 逐股).
    用 scipy.signal.lfilter 向量化逐列 EMA(alpha=2/(w+1)), 避免 500 列 pandas 循环."""
    from scipy.signal import lfilter
    w = int(np.clip(window, 5, 60))
    alpha = 2.0 / (w + 1)
    arr = np.asarray(x, dtype=float)
    out = np.empty_like(arr)
    b = [alpha]
    a = [1.0, -(1.0 - alpha)]
    for s in range(arr.shape[1]):
        out[:, s] = lfilter(b, a, arr[:, s], axis=0)
    decay = arr - out
    decay[~np.isfinite(arr)] = np.nan
    return decay


TIMESERIES_PRIMITIVES = [
    Primitive("ts_sum", _ts_sum, 2, 1.5),
    Primitive("ts_mean", _ts_mean, 2, 1.0),
    Primitive("ts_std", _ts_std, 2, 1.0),
    Primitive("ts_min", _ts_min, 2, 1.0),
    Primitive("ts_max", _ts_max, 2, 1.0),
    Primitive("ts_delta", _ts_delta, 2, 1.0),
    Primitive("ts_pct", _ts_pct, 2, 1.5),
    Primitive("ts_rank", _ts_rank, 2, 2.0),
    Primitive("ts_zscore", _ts_zscore, 2, 1.5),
    Primitive("ts_ema_decay", _ts_ema_decay, 2, 1.5),
]

# ═══════════════════════════════════════════════════════
# C. 截面原语
# ═══════════════════════════════════════════════════════

def _cross_rank(x):
    """截面排名(百分位), 纯 numpy 向量化沿 axis=1 (股票轴).
    旧实现逐时间行 Python 循环 + 每行 pd.Series, 极慢; 现 argsort 向量化, NaN 安全."""
    x = np.asarray(x, dtype=float)
    valid = np.isfinite(x)
    xf = np.where(valid, x, np.inf)               # NaN -> +inf (排最后)
    order = np.argsort(xf, axis=1, kind='stable')  # 每行内排序索引
    ranks = np.empty(x.shape, dtype=float)
    rows = np.arange(x.shape[0])[:, None]
    ranks[rows, order] = np.arange(1, x.shape[1] + 1)   # 1..N (平局不取均值, 因子场景可接受)
    cnt = np.count_nonzero(valid, axis=1, keepdims=True)
    cnt = np.where(cnt == 0, 1, cnt)
    out = np.where(valid, ranks / cnt, np.nan)
    return out

def _cross_zscore(x):
    """截面 z-score, 向量化沿 axis=1"""
    x = np.asarray(x, dtype=float)
    m = np.nanmean(x, axis=1, keepdims=True)
    s = np.nanstd(x, axis=1, keepdims=True)
    s_safe = np.where(s == 0, np.nan, s)
    out = (x - m) / s_safe
    out = np.where(np.isfinite(x), out, np.nan)
    return out

def _cross_scale(x):
    """截面 min-max 归一化, 向量化沿 axis=1"""
    x = np.asarray(x, dtype=float)
    mn = np.nanmin(x, axis=1, keepdims=True)
    mx = np.nanmax(x, axis=1, keepdims=True)
    rng = mx - mn
    out = np.where(rng > 0, (x - mn) / rng, 0.0)
    out = np.where(np.isfinite(x), out, np.nan)
    return out


CROSS_SECTION_PRIMITIVES = [
    Primitive("rank", _cross_rank, 1, 2.0),
    Primitive("zscore", _cross_zscore, 1, 1.5),
    Primitive("scale", _cross_scale, 1, 1.0),
]

# ═══════════════════════════════════════════════════════
# D. 输入变量 (终端)
# ═══════════════════════════════════════════════════════

INPUT_PRIMITIVES = [
    Primitive("open", None, 0, 0, is_input=True),
    Primitive("high", None, 0, 0, is_input=True),
    Primitive("low", None, 0, 0, is_input=True),
    Primitive("close", None, 0, 0, is_input=True),
    Primitive("volume", None, 0, 0, is_input=True),
    Primitive("vwap", None, 0, 0, is_input=True),
    Primitive("returns", None, 0, 0, is_input=True),
]

# ═══════════════════════════════════════════════════════
# 汇总
# ═══════════════════════════════════════════════════════

ALL_PRIMITIVES = {
    "arithmetic": ARITHMETIC_PRIMITIVES,
    "timeseries": TIMESERIES_PRIMITIVES,
    "cross_section": CROSS_SECTION_PRIMITIVES,
    "input": INPUT_PRIMITIVES,
}

# 按名称索引
PRIMITIVE_BY_NAME = {}
for _cat_prims in ALL_PRIMITIVES.values():
    for _p in _cat_prims:
        PRIMITIVE_BY_NAME[_p.name] = _p

# 可用的非输入原语 (用于 GP 树的内部节点)
NON_INPUT_PRIMITIVES = (
    ARITHMETIC_PRIMITIVES + TIMESERIES_PRIMITIVES + CROSS_SECTION_PRIMITIVES
)

# 窗口参数候选值 (用于时间序列原语的第二个参数)
WINDOW_SIZES = [3, 5, 10, 12, 20, 26, 30, 40, 60]


def get_random_primitive(rng: np.random.RandomState = None,
                          include_ts: bool = True,
                          include_cs: bool = True) -> Primitive:
    """随机选取一个非输入原语"""
    if rng is None:
        rng = np.random.RandomState()
    candidates = list(ARITHMETIC_PRIMITIVES)
    if include_ts:
        candidates += TIMESERIES_PRIMITIVES
    if include_cs:
        candidates += CROSS_SECTION_PRIMITIVES
    return candidates[rng.randint(0, len(candidates))]


def get_random_terminal(rng: np.random.RandomState = None,
                         include_constants: bool = True) -> Primitive:
    """随机选取一个终端(输入变量或常量)"""
    if rng is None:
        rng = np.random.RandomState()
    if include_constants and rng.random() < 0.15:
        # 常量终端 (特殊处理)
        const_val = rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
        return Primitive(str(const_val), None, 0, 0, is_input=False)
    return INPUT_PRIMITIVES[rng.randint(0, len(INPUT_PRIMITIVES))]
