# -*- coding: utf-8 -*-
"""
通用 JQ 单因子回测代码生成器 (v0.9)
====================================
输入: 任意 S5 通过的 pandas 公式 (close_p/open_p/.rolling()/.shift()/.pct_change() 风格)
输出: research/factor_alchemy/jq_s5_pass_<name>_standalone.py (P-001 合规)

与 gen_jq_s5_standalone.py (手写注册表) 的区别:
  - 无需人工注册, 直接从公式字符串自动生成 per-stock 因子函数
  - JQ 老 pandas (0.25) 兼容重写:
      * Rolling.rank(pct=True) (pandas>=1.4 API) → numpy stride 等价实现
      * 遗留函数式调用 rolling_min(x, w) → (x).rolling(w).min()
  - 字段自动分析: 公式用到 open/high/low/amount → 自动 get_price 加载
    (amount 在 JQ 中字段名为 money)
  - lookback 自动推断: max(rolling/ewm/shift/pct_change 窗口)*2 + 40, 下限 60, 上限 400

用法:
  python scripts/gen_jq_generic.py --name xxx --formula "..." [--meaning ...]
  from gen_jq_generic import prepare_factor  # 编程接口
"""

import os
import re
import sys
import json
import ast
from datetime import datetime

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT_DIR = os.path.join(BASE, "research", "factor_alchemy")

# 复用 ralph_loop 的遗留函数规范化 (单一事实源)
sys.path.insert(0, os.path.join(BASE, "research", "factor_alchemy"))
try:
    from ralph_loop import _normalize_legacy_rolling  # noqa
except Exception:  # 独立运行时兜底
    _LEGACY = {"rolling_min": "min", "rolling_max": "max", "rolling_mean": "mean",
               "rolling_std": "std", "rolling_sum": "sum"}
    _LEGACY_RE = re.compile(
        r'\b(rolling_min|rolling_max|rolling_mean|rolling_std|rolling_sum)'
        r'\s*\(\s*([^,()]+)\s*,\s*(\d+)\s*\)'
    )

    def _normalize_legacy_rolling(formula):
        def _rep(m):
            return "(%s).rolling(%s).%s()" % (m.group(2).strip(), m.group(3), _LEGACY[m.group(1)])
        return _LEGACY_RE.sub(_rep, formula)


# ── JQ per-stock 终端 → 数据字段映射 ──
_TERMINAL_TO_FIELD = {
    "close": "close", "close_p": "close",
    "open": "open", "open_p": "open",
    "high": "high", "high_p": "high",
    "low": "low", "low_p": "low",
    "volume": "volume", "volume_p": "volume",
    "amount": "amount", "amount_p": "amount",
    "returns": "close", "returns_p": "close",
}
# JQ get_price 字段名 (amount 在 JQ 中为 money)
_FIELD_TO_JQ_FIELD = {"close": "close", "open": "open", "high": "high",
                      "low": "low", "volume": "volume", "amount": "money"}

# 公式中允许的函数名 (与 S5 eval 上下文一致)
_ALLOWED_FUNCS = {
    "np", "pd", "abs", "sqrt", "log", "log1p", "exp", "sign",
    "maximum", "minimum", "where", "clip",
    "range", "len", "int", "float", "list", "dict", "tuple", "str", "bool",
    "min", "max", "round", "sum", "zip", "sorted", "reversed",
    "enumerate", "map", "filter", "isinstance",
    "_rolling_rank_pct_vec", "_roll_rank_apply",
}


def rewrite_rolling_rank(formula):
    """JQ 老 pandas 兼容: 所有 X.rolling(W[, min_periods=M]).rank(pct=True)
    → _roll_rank_apply((X), W[, M])  (v0.6.2: Series/DataFrame 自适应,
      修复宽表 DataFrame 上 rolling rank 被按一维 Series 翻译的形状错位 bug)

    Returns: (新公式, 是否需要 helper)
    """
    needs = False
    start = 0
    guard = 0
    while True:
        guard += 1
        if guard > 50:
            break  # 防御: 意外循环保护
        idx = formula.find(".rolling(", start)
        if idx == -1:
            break
        # 1) rolling 参数闭合位置
        par_start = idx + len(".rolling(")
        depth, j = 1, par_start
        while j < len(formula) and depth > 0:
            if formula[j] == "(":
                depth += 1
            elif formula[j] == ")":
                depth -= 1
            j += 1
        if depth != 0:
            break  # 括号不平衡, 放弃
        rolling_args = formula[par_start:j - 1]
        # 2) 是否紧跟 .rank(...) 且含 pct=True
        m = re.match(r"\s*\.rank\s*\(", formula[j:])
        if not m:
            start = j  # 跳过当前 rolling, 从其后继续找
            continue
        rank_start = j + m.end()
        depth2, k = 1, rank_start
        while k < len(formula) and depth2 > 0:
            if formula[k] == "(":
                depth2 += 1
            elif formula[k] == ")":
                depth2 -= 1
            k += 1
        if depth2 != 0:
            break
        rank_args = formula[rank_start:k - 1]
        if "pct=True" not in rank_args.replace(" ", ""):
            start = k
            continue
        # 3) 接收者起点: 回退扫描 (括号配对 + 顶层边界)
        #    边界: 深度0的逗号 / 开括号; 两侧带空格的二元运算符 (本系统的
        #    中缀格式均为 " + "/" - "/" * "/" / " 带空格, 1e-6 不受影响)
        s, d3 = idx, 0
        while s > 0:
            c = formula[s - 1]
            if c == ")":
                d3 += 1
                s -= 1
            elif c == "(":
                if d3 > 0:
                    d3 -= 1
                    s -= 1
                else:
                    break
            elif c == "," and d3 == 0:
                break
            elif c in "+-*/" and d3 == 0 and s >= 2 and formula[s - 2] == " ":
                break
            else:
                s -= 1
        receiver = formula[s:idx].strip()
        if not receiver:
            break
        # 4) 窗口与 min_periods 参数
        w = rolling_args.strip().split(",")[0].strip()
        mp_m = re.search(r"min_periods\s*=\s*(\d+)", rolling_args)
        mp = mp_m.group(1) if mp_m else ""
        mp_arg = ", min_periods=%s" % mp if mp else ""
        new = ("_roll_rank_apply((%s), %s%s)") % (receiver, w, mp_arg)
        formula = formula[:s] + new + formula[k:]
        needs = True
        start = 0  # 结构已变, 从头重扫 (每次重写移除一个 .rolling(...).rank(...))
    return formula, needs


def analyze_terminals(formula):
    """公式 → (需要的字段列表, 未知标识符列表)

    Returns: (fields: List[str], unknown: List[str])
    """
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError:
        return [], ["<语法错误>"]
    import builtins
    allowed = _ALLOWED_FUNCS | set(dir(builtins))
    fields, unknown = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            n = node.id
            if n in _TERMINAL_TO_FIELD:
                fields.add(_TERMINAL_TO_FIELD[n])
            elif n not in allowed:
                unknown.add(n)
    return sorted(fields), sorted(unknown)


def infer_lookback(formula, default=100):
    """根据公式中的窗口参数推断 JQ get_price lookback。

    JQ 性能铁律: 每周期 get_price ≤5.5 万行 (v3 实测 4.5万行/周跑完全窗)。
    2026-08-31 修复: 旧版 max窗口*2+10 低估嵌套窗口链需求 —
      实证 weekly_backfill_calib_forge__36fe6513:
      shift(20)→rolling(12)→rolling(20).max() 串行叠加实际需末行 index≥50 (即≥51行),
      旧算法 max(20)*2+10=50 恰好差 1 行 → 末行全 NaN → valid=0 无组合收益。
    新版: 保守上界 = max(串行叠加和, 2×最大有效窗) + 15; 下限 30 上限 250。
    """
    roll_w = [int(x) for x in re.findall(r"rolling\((\d+)", formula)]
    shift_w = [int(x) for x in re.findall(r"shift\((\d+)", formula)]
    pc_w = [int(x) for x in re.findall(r"pct_change\((\d+)", formula)]
    ewm_w = [int(x) for x in re.findall(r"ewm\(span=(\d+)", formula)]
    wins_eff = roll_w + shift_w + pc_w + [x * 3 for x in ewm_w]
    if not wins_eff:
        return max(30, min(250, default))
    serial_sum = sum(roll_w) + sum(shift_w) + sum(pc_w) + sum(ewm_w)
    lb = max(serial_sum, 2 * max(wins_eff)) + 15
    return max(30, min(250, lb))


def infer_lookback_weekly(formula, slack_weeks=15, floor_weeks=12):
    """周频语义下的 lookback (**周数**) — P-20260917-026。

    🔴 为何不复用 infer_lookback: 它的 `2*max_window` 松弛是为**日频**
    「窗口并列/边界」设计的。周频宽表由策略内 daily→W-SUN resample 得到
    (每 5 个日频行才产出 1 周), 沿用同一松弛会把日频行数需求无谓放大 5 倍,
    导致本可生成的因子被上限拒绝 (实测 forge_gen4 链: infer_lookback=135 周
    ⇒ 700 日 > 上限)。

    此处改用**串行链真实需求** (所有窗口求和, 因为嵌套 rolling 是逐层消耗行数的)
    + 固定 slack(周):

        周数 = Σ(rolling + shift + pct_change + ewm) + slack_weeks

    下限 floor_weeks (默认 12 周 ≈ 1 季, resample 后至少要有 12 周才可能出非 NaN)。
    """
    roll_w = [int(x) for x in re.findall(r"rolling\((\d+)", formula)]
    shift_w = [int(x) for x in re.findall(r"shift\((\d+)", formula)]
    pc_w = [int(x) for x in re.findall(r"pct_change\((\d+)", formula)]
    ewm_w = [int(x) for x in re.findall(r"ewm\(span=(\d+)", formula)]
    serial = sum(roll_w) + sum(shift_w) + sum(pc_w) + sum(ewm_w)
    if serial <= 0:
        return int(floor_weeks)
    return max(int(floor_weeks), int(serial) + int(slack_weeks))


def infer_univ_cap(lookback, n_fields):
    """按 5.5 万行/周 铁律反推股票池上限 (close + 额外字段的 get_price 行数总和)。

    n_fields: get_price 调用次数 (1=仅 close, 2=close+volume 等)。
    下限 400 (保证截面排序有统计意义), 上限 1500。
    """
    cap = int(55000 / (lookback * max(1, n_fields)) // 50 * 50)
    return max(400, min(1500, cap))


def _build_generic_func(name, formula, meaning, minlen, req_fields,
                        weekly_semantics=False):
    """生成向量化因子函数源码 (镜像 S5 引擎的宽表 DataFrame 语义)。

    2026-08-17 事故修复: 旧版 per-stock eval 循环在 JQ 旧 pandas 环境
    (rank(pct=True)+NaN 行为差异) 返回 valid=0 → 组合无收益。
    改为全池一次 DataFrame 运算取最后一行横截面 — 与 v3_overreaction_reversal
    (JQ 实测 285 周正常) 同模式, 且与 S5 引擎 (wide pivots + eval) 数学完全等价。
    P-20260901-007 修复: _LOCALS 的 pick key 必须用 JQ 字段名
    (amount→money), 否则 _pick("amount") 取 None→nan_df→公式全 NaN→valid=0
    (chip_cost_zscore_volume_weighted / vwap_dev_ratio_5_60 无组合收益实证)。

    P-20260917-026 (周频语义): weekly_semantics=True 时先把日频宽表重采样为
    **周频宽表** (W-SUN 周期末, 聚合方式与 weekly_panel._RAW_AGGS 逐条一致:
    open=first/high=max/low=min/close=last/volume=sum/amount=sum) 再求值
    ⇒ 窗口保持**周语义** (rolling(60)=60 周), 与 weekly lane / Forge 周频建模
    同口径。修复前 rolling(60) 在日频宽表上退化为 60 日 = 5x 错位。
    """
    fname = "factor_" + re.sub(r"\W", "_", name)
    esc = formula.replace('"""', "'")
    def _pk(fld):
        jq_k = _FIELD_TO_JQ_FIELD.get(fld, fld)
        return '_pick("%s")' % jq_k
    if weekly_semantics:
        _agg_lit = ", ".join(
            '"%s": "%s"' % (k, v) for k, v in WEEKLY_AGG_MAP.items())
        wk_block = (
            ("    _WK_AGG = {%s}  # 与 weekly_panel._RAW_AGGS 一致"
             " (P-20260917-026)\n" % _agg_lit)
            + "    def _wk(df, how):\n"
            "        \"\"\"日频宽表 → 周频宽表 (W-SUN 周期末)。\n"
            "        P-20260917-026: weekly-native 因子必须在周线上求值, 否则\n"
            "        rolling(60) 从「60 周」退化为「60 日」= 5x 窗口语义错位。\"\"\"\n"
            "        if df is None or not isinstance(df, pd.DataFrame) or df.shape[0] == 0:\n"
            "            return df\n"
            "        try:\n"
            "            o = df.copy()\n"
            "            o.index = pd.to_datetime(o.index)\n"
            "            return getattr(o.resample(\"W\"), how)()\n"
            "        except Exception:\n"
            "            return df\n"
            "    close_df = _wk(close_df, \"last\")\n"
            "    nan_df = pd.DataFrame(np.nan, index=close_df.index,\n"
            "                          columns=close_df.columns)\n"
        )
        pick_ret = "        return _wk(df, _WK_AGG.get(key, \"last\"))\n"
        wsdoc = ("    周频语义 (P-20260917-026): 日频宽表已重采样为周频宽表 (W-SUN),\n"
                 "    窗口为周语义 (rolling(N) = N 周), 与 weekly lane 同口径\n")
    else:
        wk_block = ""
        pick_ret = "        return df\n"
        wsdoc = ""
    locals_src = (
        "    _LOCALS = {\n"
        '        "close": close_df, "open": %(open)s, "high": %(high)s,\n'
        '        "low": %(low)s, "volume": %(volume)s, "amount": %(amount)s,\n'
        '        "close_p": close_df, "open_p": %(open)s, "high_p": %(high)s,\n'
        '        "low_p": %(low)s, "volume_p": %(volume)s, "amount_p": %(amount)s,\n'
        '        "returns": close_df.pct_change(),\n'
        "    }\n"
    ) % {
        "open": _pk("open"), "high": _pk("high"), "low": _pk("low"),
        "volume": _pk("volume"), "amount": _pk("amount"),
    }
    body = (
        'def %(fname)s(stocks, close_df, extra_dfs=None):\n'
        '    """%(meaning)s\n'
        '    公式 (S5 通过口径): %(esc)s\n'
        '    向量化实现 (与 S5 引擎宽表语义一致): 全池一次 DataFrame 运算, 取最后一行横截面\n'
        '    HIGH = 做多 (与 S5JointFilter.backtest_factor ascending=False 一致)\n'
        '%(wsdoc)s'
        '    """\n'
        "    nan_df = pd.DataFrame(np.nan, index=close_df.index, columns=close_df.columns)\n"
        "    ex = extra_dfs or {}\n"
        "    _BASE_IDX = close_df.index\n"
        "%(wk_block)s"
        "    def _pick(key):\n"
        "        df = ex.get(key)\n"
        "        if df is None:\n"
        "            return nan_df\n"
        "        df = df.reindex(index=_BASE_IDX, columns=close_df.columns)\n"
        "%(pick_ret)s"
        "%(locals_src)s"
        "    val = eval(_EXPR, _GLOBALS, _LOCALS)\n"
        "    if isinstance(val, pd.DataFrame):\n"
        "        last = val.iloc[-1]\n"
        "    elif isinstance(val, pd.Series):\n"
        "        last = val\n"
        "    else:\n"
        "        last = pd.Series(np.full(len(stocks), float(val)), index=stocks)\n"
        "    arr = last.reindex(stocks).values.astype(float)\n"
        "    valid = np.isfinite(arr)\n"
        "    return arr, valid"
    ) % {
        "fname": fname, "meaning": meaning or "(自动生成)", "esc": esc,
        "wk_block": wk_block, "pick_ret": pick_ret, "wsdoc": wsdoc,
        "locals_src": locals_src,
    }
    return fname, body


_GLOBALS_BLOCK = '''_EXPR = "%s"
_GLOBALS = {
    "np": np, "pd": pd,
    "abs": np.abs, "sqrt": np.sqrt, "log": np.log, "log1p": np.log1p,
    "exp": np.exp, "sign": np.sign,
    "maximum": np.maximum, "minimum": np.minimum,
    "where": np.where, "clip": np.clip,
    "range": range, "len": len, "int": int, "float": float,
    "list": list, "dict": dict, "tuple": tuple,
    "_rolling_rank_pct_vec": _rolling_rank_pct_vec,
}
'''

_RANK_HELPER = '''def _rolling_rank_pct_vec(a, window, min_periods=None):
    """pandas>=1.4 Rolling.rank(pct=True) 的 numpy 等价实现 (JQ 老 pandas 兼容)
    - min_periods=None → 默认 = window (窗口含 NaN 则该位输出 NaN)
    - min_periods=N   → 窗口内非 NaN 计数 >= N 才计算, pct 分母 = 非 NaN 计数
    """
    from numpy.lib.stride_tricks import as_strided
    a = np.asarray(a, dtype=float)
    n = len(a)
    out = np.full(n, np.nan)
    if n < window:
        return out
    if min_periods is None:
        min_periods = window
    m = n - window + 1
    x = as_strided(a, shape=(m, window), strides=(a.strides[0], a.strides[0]))
    nonnan = (~np.isnan(x)).sum(axis=1)
    ok = nonnan >= min_periods
    last = x[:, -1]
    less = (x < last[:, None]).sum(axis=1)
    eq = (x == last[:, None]).sum(axis=1)
    rank = less + (eq + 1.0) / 2.0
    pct = rank / np.maximum(nonnan, 1)
    pct[~ok] = np.nan
    out[window - 1:] = pct
    return out


def _roll_rank_apply(x, window, min_periods=None):
    """v0.6.2: rolling(W).rank(pct=True) 的 Series/DataFrame 自适应等价实现。
    - DataFrame: 逐列沿时间轴滚动排名 (与 pandas df.rolling(W).rank(pct=True) 语义一致)
    - Series:   直接委托 _rolling_rank_pct_vec
    修复: 宽表 DataFrame 公式 (.rolling(30).rank(pct=True)) 曾被按一维翻译
    → 形状错位 → valid=0 事故 (2026-08-29 forge_gen1_square_square_tsmin)
    """
    if isinstance(x, pd.DataFrame):
        out = pd.DataFrame(np.nan, index=x.index, columns=x.columns)
        for c in x.columns:
            out[c] = _rolling_rank_pct_vec(
                x[c].values.astype(float), window, min_periods)
        return out
    return pd.Series(_rolling_rank_pct_vec(
        np.asarray(x, dtype=float), window, min_periods), index=x.index)


'''

# 2026-08-17: rank(pct=True) 的 JQ 兼容实现 (herding valid=0 事故修复)
# JQ 旧 pandas 的 Series/DataFrame.rank(pct=True)+NaN 行为与本地不一致,
# 统一用 scipy rankdata 实现: DataFrame → 逐列时间序列排名; Series → 直接排名。
# pct = rank/n_non_na (与本地 S5 引擎所用 pandas 3.x 语义一致; 旧 pandas 的
# (rank-1)/(n-1) 仅为其单调变换, 不影响横截面排序), NaN 保持 NaN。
_RANK_PCT_HELPER = '''def _rank_pct(obj):
    """pandas .rank(pct=True, na_option='keep') 的 JQ 兼容实现 (axis=0 逐列时序排名)。

    语义与本地 S5 引擎 (pandas 3.x): pct = rank / n_non_na, 单值列 -> 1.0, NaN 保持 NaN。
    """
    if isinstance(obj, pd.DataFrame):
        out = pd.DataFrame(np.nan, index=obj.index, columns=obj.columns)
        for c in obj.columns:
            out[c] = _rank_pct(obj[c])
        return out
    v = obj.values.astype(float)
    mask = np.isfinite(v)
    out = np.full(len(v), np.nan)
    k = int(mask.sum())
    if k >= 1:
        from scipy.stats import rankdata
        r = rankdata(v[mask])
        out[mask] = r / k
    return pd.Series(out, index=obj.index)
'''


# P-20260917-026: 日频宽表 → 周频宽表的聚合方式
# 🔴 单一事实源: 必须与 research/factor_alchemy/weekly_panel._RAW_AGGS 逐条一致
# (weekly lane / Forge 周频建模的权威聚合), 由 _verify_026_weekly 断言对拍。
WEEKLY_AGG_MAP = {
    "open": "first", "high": "max", "low": "min",
    "close": "last", "volume": "sum", "amount": "sum",
}

# P-20260917-026: 周频语义模式下 JQ get_price 可接受的日频行数上限。
# 周频窗口链 w 周 ⇒ 需 w*5 + 25 日频行 (5 交易日/周 + resample 边界 buffer)。
# 超出即**拒绝生成**, 而非截断 (截断会静默产出全 NaN 因子 = 无组合收益事故)。
# 520 → 600: 周频语义下「周窗口链 w 周」需 w*5+25 日频行; 实测
# forge_gen4 (Σ窗口=85 周) 需 85+15=100 周 ⇒ 525 日, 原 520 会误拒。
_JQ_WEEKLY_LB_CAP = int(os.environ.get("FUXI_JQ_WEEKLY_LB_CAP", "600"))


def detect_window_semantics(formula):
    """P-20260814-003: 检测公式中的长窗口 rolling/ewm/shift.

    背景: max_drawdown_duration 教训 — local 周频 rolling(120)=2.4年,
    JQ 日频 rolling(120)=120天, 窗口语义错位 → JQ -39.9% (hard_forbidden).

    Returns: [(kind, window, matched_text), ...] 仅返回窗口 >= 60 的项.
    """
    found = []
    for m in re.finditer(r"\.rolling\(\s*(\d+)\s*(?:,\s*min_periods\s*=\s*\d+\s*)?\)", formula):
        w = int(m.group(1))
        if w >= 60:
            found.append(("rolling", w, m.group(0)))
    for m in re.finditer(r"\.ewm\(\s*(?:span|halflife|com|alpha)\s*=\s*([\d.]+)\s*[,)]", formula):
        w = float(m.group(1))
        if w >= 60:
            found.append(("ewm", w, m.group(0)))
    for m in re.finditer(r"\.shift\(\s*(\d+)\s*\)", formula):
        w = int(m.group(1))
        if w >= 60:
            found.append(("shift", w, m.group(0)))
    return found


def prepare_factor(name, formula, meaning="", local_note="", lookback=None,
                   weekly_semantics=False):
    """公式 → 生成元数据 (或错误)。

    Parameters
    ----------
    weekly_semantics : bool
        P-20260917-026. True = 该因子原始验证频率为**周频**(natural_freq=weekly),
        生成的回测将在**周频宽表**上求值 ⇒:
          - 公式窗口保持周语义 (rolling(60)=60 周, 不做任何窗口改写)
          - lookback 需按「周数 × 5 交易日/周 + buffer」换算为日频行数
            (JQ get_price 只能按日频取数, 周频宽表由策略内 resample 得到)

    Returns:
        dict(ok=True, file/name/formula_jq/fields/lookback/minlen/func/func_name/needs_helper) 或
        dict(ok=False, reason=...)
    """
    f = (formula or "").strip()
    if not f:
        return {"ok": False, "reason": "空公式"}
    # 截面公式 (axis=1) 无法在 per-stock JQ 中复刻 → 拒绝
    if re.search(r"axis\s*=\s*1", f):
        return {"ok": False, "reason": "截面公式 (axis=1) 不支持自动生成, 需手工 JQ 实现"}
    f = _normalize_legacy_rolling(f)
    fields, unknown = analyze_terminals(f)
    if unknown:
        return {"ok": False, "reason": "未知标识符: %s" % ", ".join(unknown)}
    if not fields:
        return {"ok": False, "reason": "公式未引用价格终端 (close/open/high/low/volume/amount)"}
    # JQ 不支持 turnover (铁律1: turnover 用 volume 代理) → 明确拒绝
    jq_fields = []
    for fld in fields:
        jq_f = _FIELD_TO_JQ_FIELD.get(fld)
        if jq_f is None:
            return {"ok": False, "reason": "字段 %s 在 JQ 无对应数据" % fld}
        if jq_f not in jq_fields:
            jq_fields.append(jq_f)
    f_jq, needs_helper = rewrite_rolling_rank(f)
    # 2026-08-17: JQ 旧 pandas 的 Series.rank(pct=True)+NaN 行为与本地不一致
    # (herding 逐只 eval 事故: JQ valid=0)。rank 统一改写为 scipy rankdata 兼容实现
    # _rank_pct — DataFrame 逐列时间序列排名, pct=(rank-1)/(n_non_na-1), 与 S5 引擎语义一致。
    needs_rank_helper = ".rank(pct=True)" in f_jq
    if needs_rank_helper:
        f_jq = f_jq.replace(".rank(pct=True)", ".pipe(_rank_pct)")
    # v0.9: lookback/minlen 从重写前的公式推断 (rank 重写会吞掉 .rolling(W) 字样)
    lb = lookback or infer_lookback(f)
    wins = [int(x) for x in re.findall(r"rolling\((\d+)", f)] or [0]
    minlen = min(int(lb), max(30, max(wins) + 10))
    # ── P-20260917-026: 周频语义 → lookback 由「周数」换算为「日频行数」 ──
    #   原 lb 是**周数语义**窗口需求 (infer_lookback 只看字面窗口值);
    #   周频宽表由策略内 daily→W-SUN resample 得到 ⇒ 需 lb*5 日频行数。
    #   上限 520 (JQ get_price 行数/周 预算软约束, 见 infer_univ_cap),
    #   下限 60 (resample 后至少要求 12 周以上才可能出非 NaN)。
    lb_weekly = infer_lookback_weekly(f) if weekly_semantics else None
    if weekly_semantics:
        _need = int(lb_weekly) * 5 + 25
        if _need > _JQ_WEEKLY_LB_CAP:
            # 🔴 拒绝而非截断: 截断会让周频宽表周数不足 ⇒ 因子末行全 NaN
            #    ⇒ valid=0 ⇒ 组合零收益, 且全程无报错 (静默事故)。
            return {"ok": False, "reason": (
                "周频语义 lookback 超 JQ 预算: 需 %d 日频行 "
                "(周窗口链 %d 周 × 5 + 25) > 上限 %d ⇒ 拒绝生成。"
                "可选: 缩短窗口 / 显式传 lookback / "
                "FUXI_JQ_WEEKLY_SEMANTICS=off 并手工做窗口 x5 换算。"
                % (_need, lb_weekly, _JQ_WEEKLY_LB_CAP))}
        lb = int(min(_JQ_WEEKLY_LB_CAP, max(60, _need)))
        minlen = max(30, min(int(lb_weekly), max(wins) + 10))
    n_fields = 1 + len([x for x in jq_fields if x != "close"])
    # P-20260814-003: 长窗口语义检测
    window_warnings = detect_window_semantics(f)
    return {
        "ok": True,
        "name": name,
        "formula": f,
        "formula_jq": f_jq,
        "fields": jq_fields,
        "lookback": int(lb),
        "minlen": int(minlen),
        "needs_helper": needs_helper,
        "needs_rank_helper": needs_rank_helper,
        "univ_cap": infer_univ_cap(int(lb), n_fields),
        "window_warnings": window_warnings,
        # P-20260917-026: 周频语义审计字段
        "weekly_semantics": bool(weekly_semantics),
        "lookback_weekly": lb_weekly,
    }


def _smoke_last_row_valid(formula_jq, lookback):
    """本地随机数据 eval 公式, 检查最后一行是否有非 NaN 值。

    Returns: True=有效 / False=末行全 NaN (lookback 不足) / None=无法本地 eval
    (公式含 JQ helper 或本地缺依赖, 跳过检查不阻塞)。
    """
    try:
        import numpy as np      # 延迟导入: 本模块是纯模板生成器, 保持轻量
        import pandas as pd
        G = {
            "np": np, "pd": pd, "abs": np.abs, "sqrt": np.sqrt, "log": np.log,
            "log1p": np.log1p, "exp": np.exp, "sign": np.sign,
            "maximum": np.maximum, "minimum": np.minimum,
            "where": np.where, "clip": np.clip,
        }
        lb = int(lookback)
        idx = pd.date_range("2025-01-01", periods=lb, freq="B")
        close = pd.DataFrame(np.random.rand(lb, 4) + 10, index=idx,
                             columns=list("abcd"))
        volume = pd.DataFrame(np.random.rand(lb, 4) * 1e6, index=idx,
                              columns=list("abcd"))
        # P-20260905: 旧版 open/high/low 全部用 close 填充 → high==low==close
        # → hl_ratio 恒为 1 → rolling std=0 → 分母 clip 后仍爆炸 → cov 产 NaN
        # → smoke 假阳性 (gp_evo_002 实证: 真实分布下公式 300/300 finite)。
        # 改为与真实 OHLC 结构一致的随机数据 (high≥max(open,close), low≤min(open,close))。
        _rnd = np.random.rand
        open_df = close * (1 + _rnd(lb, 4) * 0.02 - 0.01)
        high_df = np.maximum(open_df, close) * (1 + _rnd(lb, 4) * 0.01)
        low_df = np.minimum(open_df, close) * (1 - _rnd(lb, 4) * 0.01)
        L = {
            "close_p": close, "volume_p": volume, "open_p": open_df,
            "high_p": high_df, "low_p": low_df, "amount_p": volume,
            "close": close, "open": open_df, "high": high_df, "low": low_df,
            "volume": volume, "amount": volume,
            "returns": close.pct_change(),
        }
        val = eval(formula_jq, G, L)
        if isinstance(val, pd.DataFrame):
            # JQ 端 valid = np.isfinite(arr): inf 也会被过滤 → 用 isfinite 对齐
            return bool(np.isfinite(val.iloc[-1].values).any())
        if isinstance(val, pd.Series):
            return bool(np.isfinite(val.values).any())
        return bool(np.isfinite(float(val)))
    except Exception:
        return None


def _smoke_fix_lookback(meta):
    """2026-08-31 事故防线: 生成前验证公式末行有效性, 不足则自动扩容 lookback。

    实证: 36fe6513 公式 shift(20)→rolling(12)→rolling(20).max() 串行叠加需
    末行 index>=50 (即>=51行), 旧 lookback=50 末行全 NaN → JQ valid=0 无组合收益。
    扩容 3 次仍全 NaN → 公式本身产 NaN (除以零/负值开方/字段缺失) → ok=False。
    """
    lb = int(meta.get("lookback") or 0)
    for attempt in range(3):
        sm = _smoke_last_row_valid(meta["formula_jq"], lb)
        if sm is not False:
            if lb != int(meta.get("lookback") or 0):
                meta["lookback"] = lb
                n_fields = 1 + len([x for x in meta.get("fields", [])
                                    if x != "close"])
                meta["univ_cap"] = infer_univ_cap(lb, n_fields)
            return meta
        new_lb = min(250, lb * 2 + 10)
        print("[LOOKBACK-SMOKE] %s: lookback=%d 末行全 NaN → 扩容至 %d (第%d次)"
              % (meta.get("name", "?"), lb, new_lb, attempt + 1))
        lb = new_lb
    meta["ok"] = False
    meta["reason"] = ("lookback 连扩 3 次后公式末行仍全 NaN, 公式本身产出 NaN "
                      "(疑似除以零/负值开方/字段缺失), 请人工检查公式")
    return meta


def generate_standalone(name, formula, meaning="", local_note="",
                        lookback=None, out_dir=None, window_freq="daily",
                        s5_passed=None, weekly_semantics=None):
    """生成 JQ 单因子回测文件. Returns (out_path, meta) 或 (None, {"reason": ...}).

    window_freq: 原始验证频率 ("daily"/"weekly"), P-20260814-003.
    凡公式含 rolling/ewm/shift 窗口 >= 60 的因子:
      - 打印 [WINDOW-SEMANTIC-WARNING] 到 stdout (ralph_loop 日志可见)
      - 在生成代码头部注释注明原始验证频率与窗口换算建议 (防止周频→日频语义错位)

    s5_passed (v0.7.23 P0-f): 候选的真实 S5 状态。
      - ``True``  → 文件名 ``jq_s5_pass_*`` (与历史一致)
      - ``False`` → 文件名 ``jq_s5_fail_*`` + 注释头显式标注 "⚠️ S5未通过"
      - ``None``  → 未知, 保持历史前缀 ``jq_s5_pass_*`` (向后兼容)
      🔴 背景: codegen 的入队条件是 ``eligible_for_jq`` (默认 4/5 阶段), 而
      ``FUXI_ELIGIBLE_REQUIRE_S5=off`` 时它**不等价于 S5 通过** ⇒ 实测出现
      ``jq_s5_pass_*.py`` 的注释头自写 ``S5: grade=B calmar=0.48`` 而 docstring
      写 "(S5 通过因子)" 的自相矛盾产物。本参数让"文件名/注释"与真实状态一致,
      不改变任何入队条件 (是否收紧见 FUXI_ELIGIBLE_REQUIRE_S5)。
    """
    _prefix = ("jq_s5_fail_" if s5_passed is False else "jq_s5_pass_")
    if s5_passed is False:
        _flag = "⚠️ S5未通过 (eligible_for_jq 触发, 未过联合过滤)"
        local_note = f"{_flag} | {local_note}" if local_note else _flag
        if not meaning:
            meaning = "(⚠️ S5未通过因子, eligible_for_jq 触发, 自动生成)"
    # ── P-20260917-026: 周频语义模式解析 ─────────────────────────────
    # 根因: window_freq 由调用方传入, 而 run_v4_pipeline 未把候选的
    #   natural_freq 接进来 ⇒ weekly-native 因子被按 "daily" 生成, 文件头
    #   「原始验证频率: daily」是**假话**, 且窗口语义实际错位 5x。
    # 现: ① 调用方可显式传 weekly_semantics; ② 未传则由 window_freq 推导;
    #     ③ 由 FUXI_JQ_WEEKLY_SEMANTICS 统一开关 (auto 默认 / off 回退日频)。
    _ws_env = (os.environ.get("FUXI_JQ_WEEKLY_SEMANTICS", "auto")
               or "auto").strip().lower()
    if weekly_semantics is None:
        weekly_semantics = (_ws_env != "off"
                            and str(window_freq or "").strip().lower() == "weekly")
    meta = prepare_factor(name, formula, meaning, local_note, lookback,
                          weekly_semantics=bool(weekly_semantics))
    if not meta["ok"]:
        return None, meta

    # 2026-08-31: 末行有效性 smoke 防线 (weekly_backfill_calib_forge__36fe6513
    # valid=0 无组合收益事故第二道保险; 第一道 = infer_lookback 保守上界)。
    # 生成前用本地随机数据实际 eval 公式, 末行全 NaN 则自动扩容 lookback,
    # 3 次扩容仍失败 → 公式本身产 NaN → SKIP。
    meta = _smoke_fix_lookback(meta)
    if not meta["ok"]:
        return None, meta

    # P-20260814-003: 长窗口语义校验输出 (周频→日频错位防护)
    ww = meta.get("window_warnings", [])
    window_note = ""
    if ww and meta.get("weekly_semantics"):
        # P-20260917-026: 语义已由「周频宽表求值」解决 ⇒ 不再提示 x5 换算
        # (旧文案会误导使用者以为还要手工改窗)
        meta["window_warning"] = None
        window_note = ""
    elif ww:
        detail = ", ".join("%s(%s)" % (k, int(w) if float(w) == int(w) else w)
                           for k, w, _ in ww)
        if window_freq == "weekly":
            warn = ("[WINDOW-SEMANTIC-WARNING] %s: 因子在周频验证, 含长窗口 %s. "
                    "JQ 日频需窗口x5换算 (原窗口->日频: %s), 否则语义错位!"
                    % (name, detail,
                       ", ".join("%s->%s" % (int(w) if float(w) == int(w) else w,
                                              int(w) * 5) for k, w, _ in ww)))
        else:
            warn = ("[WINDOW-SEMANTIC-WARNING] %s: 公式含长窗口 %s. "
                    "若该因子在周频验证, JQ 日频回测必须窗口x5换算, 否则语义错位!"
                    % (name, detail))
        print(warn)
        window_note = ("# [窗口语义警告 P-003] 原始验证频率: %s. 公式含长窗口: %s.\n"
                       "#   若验证频率为周频, 生成前请用 --window-freq weekly 重新生成,\n"
                       "#   或在日频回测中手动将窗口 x5 换算 (教训: max_drawdown_duration JQ -39.9%%).\n"
                       ) % (window_freq, detail)
        meta["window_warning"] = warn
    else:
        meta["window_warning"] = None

    # ── P-20260917-026: 周频语义模式 → 语义已解决, 覆盖「x5 换算」的劝改文案 ──
    if meta.get("weekly_semantics"):
        _lb_w = meta.get("lookback_weekly") or 0
        _lb_d = meta.get("lookback") or 0
        _lb_ref = infer_lookback(meta["formula"])
        _ratio = (_lb_d / float(_lb_ref)) if _lb_ref else 0.0
        window_note = (
            "# [周频语义 P-20260917-026] 原始验证频率: weekly ⇒ 本回测在**周频宽表**\n"
            "#   上求值 (策略内 daily→W-SUN resample, 聚合与 weekly_panel 一致),\n"
            "#   公式窗口保持周语义 (rolling(60)=60 周), 不做 x5 改写。\n"
            "#   lookback=%d 日频行 ≈ %d 周 (日频模式参考值 %d) ⇒ get_price 行数 ~%.1fx,\n"
            "#   单轮回测耗时上升属预期 (JQ 5.5万行/周 预算软约束)。\n"
            "#   回退日频语义: FUXI_JQ_WEEKLY_SEMANTICS=off 重新生成。\n"
            % (_lb_d, _lb_w, _lb_ref, _ratio)
        )
        print("[WEEKLY-SEMANTICS] %s: 周频宽表求值, lookback=%d 日 ≈ %d 周"
              % (name, _lb_d, _lb_w))
        meta["window_semantics"] = "weekly_panel"
    else:
        meta["window_semantics"] = "daily"

    # 复用 gen_jq_s5_standalone 的公共模板 (已在 JQ 验证)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gen_jq_s5_standalone import TEMPLATE

    # ── v0.7.23 P0-f: 模板标题/启动日志的 S5 状态如实化 ──────────────
    # 模板里这两处是**字面量**(非占位符), 若不改会与上面的 local_note/meaning
    # 自相矛盾 (实测: 文件头顶行写 "(S5 通过因子)" 而下一行写 "⚠️ S5未通过")。
    # 用字面量替换而非新增占位符 ⇒ 其余两个生产者 (gen_jq_s5_standalone /
    # pass_to_jq_bridge) 完全不受影响 (默认 = 历史文案)。
    code_prefix_swap = None
    if s5_passed is False:
        _T_OLD = "@FNAME@ 单因子独立回测 (S5 通过因子)"
        _T_NEW = "@FNAME@ 单因子独立回测 (⚠️ S5 未通过 — eligible_for_jq 触发)"
        _L_OLD = 'log.info("@FNAME@ 单因子 (S5通过, 滞后IC) 启动")'
        _L_NEW = 'log.info("@FNAME@ 单因子 (⚠️ S5未通过, eligible_for_jq 触发, 滞后IC) 启动")'
        code_prefix_swap = ((_T_OLD, _T_NEW), (_L_OLD, _L_NEW))

    func_name, func_body = _build_generic_func(
        name, meta["formula_jq"], meaning or "", meta["minlen"], meta["fields"],
        weekly_semantics=bool(meta.get("weekly_semantics")),
    )
    helper = _RANK_HELPER if meta["needs_helper"] else ""
    if meta["needs_rank_helper"]:
        helper += _RANK_PCT_HELPER
    rank_global = '    "_rolling_rank_pct_vec": _rolling_rank_pct_vec,\n' if meta["needs_helper"] else ""
    rank_global += '    "_roll_rank_apply": _roll_rank_apply,\n' if meta["needs_helper"] else ""
    rank_global += '    "_rank_pct": _rank_pct,\n' if meta["needs_rank_helper"] else ""
    globals_block = (_GLOBALS_BLOCK % meta["formula_jq"].replace('"', '\\"')).replace(
        '    "_rolling_rank_pct_vec": _rolling_rank_pct_vec,\n', rank_global
    )

    code = TEMPLATE
    if code_prefix_swap:
        for _old, _new in code_prefix_swap:
            code = code.replace(_old, _new)
    code = code.replace("@FNAME@", name)
    code = code.replace("@FNAME_SAFE@", func_name)
    code = code.replace("@MEANING@", meaning or "(S5 通过因子, 自动生成)")
    code = code.replace("@LOOKBACK@", str(meta["lookback"]))
    code = code.replace("@UNIV_CAP@", str(meta["univ_cap"]))
    code = code.replace("@LOCAL_NOTE@",
                        (window_note + (local_note or "自动生成 (v0.9 gen_jq_generic)")))
    code = code.replace("@GEN_TIME@", datetime.now().strftime("%Y-%m-%d %H:%M"))
    code = code.replace("@FUNC@", (helper + globals_block + func_body).strip("\n"))
    code = code.replace("@MINLEN@", str(meta["minlen"]))

    # 额外字段加载 (close 由模板无条件加载; volume/其他字段仅公式引用时加载 — v3 性能骨架)
    extra = ""
    for fld in meta["fields"]:
        if fld in ("close",):
            continue
        # P-20260901-001: prepare_factor 已把 amount→money 翻译, 此处 .get 幂等
        # (原直接下标 KeyError: 'money' — chip_cost_zscore_volume_weighted 生成崩溃)
        jq_f = _FIELD_TO_JQ_FIELD.get(fld, fld)
        extra += '''
    px_x = get_price(universe, count=lb, end_date=prev_date, frequency="daily",
                     fields="%s", skip_paused=False, fq="pre")
    if _P is not None and isinstance(px_x, _P):
        px_x = px_x["%s"] if "%s" in getattr(px_x, "items", []) else px_x.minor_xs("%s")
    if px_x is not None and px_x.shape[0] > 0 and px_x.shape[1] > 0:
        extra_dfs["%s"] = px_x.reindex(columns=universe)''' % (jq_f, jq_f, jq_f, jq_f, fld)
    code = code.replace("@EXTRA_LOAD@", extra)

    out_dir = out_dir or OUT_DIR
    # v0.7.23 P0-f: 前缀如实反映 S5 状态 (pass/fail), 不再一律写 s5_pass
    out_path = os.path.normpath(os.path.join(out_dir, f"{_prefix}{name}_standalone.py"))
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(code)
    # 语法校验
    compile(code, out_path, "exec")
    meta["file"] = out_path
    meta["func_name"] = func_name
    return out_path, meta


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="通用 JQ 单因子代码生成器 (v0.9)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--formula", required=True)
    ap.add_argument("--meaning", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--lookback", type=int, default=None)
    ap.add_argument("--window-freq", choices=["daily", "weekly"], default="daily",
                    help="原始验证频率 (P-003: 长窗口语义校验) "
                         "weekly ⇒ 默认启用周频宽表求值 (P-20260917-026)")
    ap.add_argument("--s5-passed", dest="s5_passed", default="unknown",
                    choices=["pass", "fail", "unknown"],
                    help="P-20260915 P0-f: 真实 S5 状态 → 决定文件名前缀 "
                         "(pass→jq_s5_pass_ / fail→jq_s5_fail_ / unknown→历史前缀)")
    args = ap.parse_args()
    _s5_flag = {"pass": True, "fail": False, "unknown": None}[args.s5_passed]
    out_path, meta = generate_standalone(
        args.name, args.formula, args.meaning, args.note, args.lookback,
        window_freq=args.window_freq, s5_passed=_s5_flag
    )
    if out_path:
        print("[OK] %s (lookback=%d fields=%s)" % (
            out_path, meta["lookback"], ",".join(meta["fields"])))
    else:
        print("[FAIL] %s" % meta.get("reason"))
        sys.exit(1)
