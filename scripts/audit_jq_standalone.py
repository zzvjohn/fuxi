# -*- coding: utf-8 -*-
"""audit_jq_standalone.py — 批量审计 research/factor_alchemy/ 下 JQ 单因子与组合文件的零收益反模式。

2026-09-01 六因子全零收益事故后固化。检测六类问题:
  A1. 同名函数重复定义 (后定义遮蔽前定义; 组合文件多成员不同名 def 属正常)
  A1b. 单因子文件 def factor_ 数 != 1
  A2. _EXPR = None 且存在 eval(_EXPR  → 每次 eval(None) TypeError → 永不交易
  A3. 死代码 _rank_pct (手写版不引用)
  B.  amount/money key 错位: 公式真正引用 amount 时,
      必须 (pick=amount 且 加载 extra_dfs["amount"]) 或 (pick=money 且 加载 extra_dfs["money"])
      或组合文件 amount = data["money"|"amount"]
  C.  语法错误 / FACTOR_FUNC_MAP 指向不存在的 def
  D.  _EXPR 引用 amount 但 rebalance 未加载 money/amount
  E.  缺差额微调保护 (P-20260901-008): order_target_value 前无 |delta|<100股 跳过,
      周频微调会刷屏级报"开仓/平仓数量不能小于 100"
  F.  np.argsort kind="stable" (P-20260901-009): JQ 旧 numpy 不支持,
      稳定排序必须用 mergesort
  G.  裸 sum() (P-20260901-012): JQ 沙箱 sum=numpy.sum, 一律 _sum_vals() 手动累加
  H.  JQ API 非法 kwargs (P-20260901-013): get_price 等 JQ API 无 panel 参数,
      误加非法参数 → 第一周调仓 TypeError → 从未建仓 → 面板 0 收益 (stub 因本地
      mock 函数有兜底参数而全通, 掩盖真实 JQ 签名不兼容)

用法: python scripts/audit_jq_standalone.py [--verbose]
退出码: 0=干净, 1=有问题
"""
import glob
import os
import re
import sys

FACTOR_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "research", "factor_alchemy")


def _expr_uses_amount(src):
    """公式是否真正引用 amount 终端 (排除 docstring/注释)。"""
    m = re.search(r'_EXPR\s*=\s*"([^"]*)"', src)
    if m:
        return bool(re.search(r'\bamount\b', m.group(1)))
    # 手写截面版: 只认代码里的 ex.get("amount") / amount_df (docstring 里的不算)
    if 'ex.get("amount")' in src or 'ex.get("amount_p")' in src:
        return True
    if re.search(r'^\s+amount(_df)?\s*=', src, re.M):
        return True
    return False


def _count_bare_sum(src):
    """统计裸 sum( 调用数 (tokenize 精确: 排除注释/字符串/np.sum/x.sum)。
    白名单: 含 SELFTEST 的探测行 (故意用裸 sum 探测 JQ 环境是否被 numpy 遮蔽)。"""
    import io as _io
    import tokenize as _tk
    try:
        toks = list(_tk.generate_tokens(_io.StringIO(src).readline))
    except Exception:
        return 0
    lines = src.splitlines()
    n = 0
    for i, t in enumerate(toks):
        if t.type != _tk.NAME or t.string != "sum":
            continue
        prev = toks[i - 1]
        nxt = toks[i + 1] if i + 1 < len(toks) else None
        if prev.type == _tk.OP and prev.string == ".":
            continue
        if not (nxt and nxt.type == _tk.OP and nxt.string == "("):
            continue
        r0 = t.start[0] - 1
        ctx = "\n".join(lines[max(0, r0 - 1):r0 + 2])   # 本行及上下各一行
        if "SELFTEST" in ctx:
            continue
        n += 1
    return n


def audit_file(path):
    src = open(path, encoding="utf-8").read()
    is_combo = "jq_combo_" in os.path.basename(path)
    issues = []

    # A1: 同名函数重复定义 (后定义遮蔽前定义 = 真遮蔽; 不同名多 def 对组合文件是正常结构)
    def_names = re.findall(r'^def\s+(\w+)\s*\(', src, re.M)
    dups = sorted({n for n in def_names if def_names.count(n) > 1})
    if dups:
        issues.append("A1 同名函数重复定义(遮蔽): %s" % ", ".join(dups))
    # A1b: S5 PASS 单因子文件 def factor_ 应恰为 1 (A/B 实验/gp_breed/lgbm 等
    #      多形态文件不适用此规则, 2026-09-01 扩宽扫描范围后修正)
    n_def = len(re.findall(r'^def factor_', src, re.M))
    if "jq_s5_pass_" in os.path.basename(path) and n_def != 1:
        issues.append("A1b 单因子文件 def factor_ 数=%d (应为1)" % n_def)
    if "_EXPR = None" in src and "eval(_EXPR" in src:
        issues.append("A2 _EXPR=None + eval(_EXPR) — eval(None) 必抛 TypeError")
    if "def _rank_pct(" in src:
        # 2026-09-08 修复盲区: _rank_pct 有三种合法引用形态 — ①直接调用 _rank_pct(
        # ②.pipe(_rank_pct) 函数对象传递 ③_GLOBALS dict 值 "_rank_pct": _rank_pct
        # (旧检查只识别 ①, 对 eval 公式文件的 ②③ 全部误报死代码)
        _body = src.replace("def _rank_pct(", "")
        _used = ("_rank_pct(" in _body) or (".pipe(_rank_pct)" in _body) or ('"_rank_pct":' in _body)
        if not _used:
            issues.append("A3 死代码 _rank_pct (无引用)")

    if _expr_uses_amount(src):
        pick_amount = '_pick("amount")' in src
        load_amount = 'extra_dfs["amount"]' in src
        load_money = 'extra_dfs["money"]' in src
        pick_money = '_pick("money")' in src
        hand_amount = 'ex.get("amount")' in src
        # 组合文件取数模式: amount = data["money"] / data["amount"] 且 rebalance 有对应 key 加载
        combo_money = bool(re.search(r'amount\s*=\s*data\["money"\]', src))
        combo_amount = bool(re.search(r'amount\s*=\s*data\["amount"\]', src))
        combo_ok = is_combo and (combo_money or combo_amount)
        if not ((pick_amount and load_amount) or (pick_money and load_money)
                or (hand_amount and load_amount) or combo_ok):
            issues.append("B amount key 错位: 公式引用 amount 但 pick/加载 key 不一致 "
                          "(pick_amount=%s load_amount=%s pick_money=%s load_money=%s hand=%s)"
                          % (pick_amount, load_amount, pick_money, load_money, hand_amount))
        elif not (load_amount or load_money or combo_ok):
            issues.append("D 公式引用 amount 但 rebalance 未加载 money/amount")

    try:
        import ast
        ast.parse(src)
    except SyntaxError as e:
        issues.append("C 语法错误 %s" % e.msg)

    # E: 差额微调保护 (P-20260901-008)
    if "order_target_value(s, w)" in src and "# P-20260901-008" not in src:
        issues.append("E 缺差额微调保护: order_target_value 前无 |delta|<100股 跳过 "
                      "(周频微调差额<100股会刷屏报'开仓/平仓数量不能小于 100')")

    # F: JQ 旧 numpy 兼容性 (P-20260901-009): kind="stable" 需 numpy>=1.15, 聚宽环境
    #     会抛 ValueError: stable is an unrecognized kind of sort → 稳定排序用 mergesort
    if re.search(r'np\.argsort\([^)]*kind="stable"', src):
        issues.append("F np.argsort kind=stable: JQ 旧 numpy 不支持 (ValueError: stable is "
                      "an unrecognized kind of sort), 稳定排序改用 kind=mergesort")

    # G: JQ 沙箱 sum 被 numpy 遮蔽 (P-20260901-012, 最高优先级静默错误)
    #    JQ 把 numpy 注入用户命名空间, 裸 sum() 实为 np.sum:
    #      np.sum(dict.values()) → 原样返回 dict_values (不报错! 下游 <= / float() 才炸)
    #      np.sum(generator)     → numpy>=1.25 直接 TypeError
    #    → 一律改用 _sum_vals() 手动累加。本机 numpy 2.4.6 已实测复现。
    _bare = _count_bare_sum(src)
    if _bare and "def _sum_vals(" not in src:
        issues.append("G 裸 sum() %d 处且无 _sum_vals helper: JQ 沙箱 sum=numpy.sum, "
                      "np.sum(dict.values()) 原样返回 dict_values / np.sum(generator) 抛 "
                      "TypeError → 改用 _sum_vals() 手动累加" % _bare)
    elif _bare:
        issues.append("G 残留裸 sum() %d 处 (已有 _sum_vals, 替换为 _sum_vals 即可)" % _bare)

    # H: JQ API 非法 kwargs (P-20260901-013) — 真实 JQ 签名严格, 非法参数第一周期
    #    即 TypeError → 从未建仓 → 面板 0 收益; stub 本地 mock 有兜底参数故漏检。
    import ast as _ast
    _LEGAL_KW = {
        "get_price": {"security", "start_date", "end_date", "frequency", "fields",
                      "skip_paused", "fq", "count"},
        "get_all_securities": {"types", "date"},
        "get_fundamentals": {"query_object", "date", "statDate"},
        "get_extras": {"info", "security_list", "start_date", "end_date", "df", "count"},
    }
    try:
        _tree = _ast.parse(src)
        for _node in _ast.walk(_tree):
            if (isinstance(_node, _ast.Call) and isinstance(_node.func, _ast.Name)
                    and _node.func.id in _LEGAL_KW):
                _legal = _LEGAL_KW[_node.func.id]
                for _kw in _node.keywords:
                    if _kw.arg and _kw.arg not in _legal:
                        issues.append(
                            "H %s() 非法参数 %s=: JQ 签名无此参数 → 回测全周期静默失败"
                            % (_node.func.id, _kw.arg))
    except SyntaxError:
        pass  # C 类已报

    # I: 内部自算净值 (P-20260901-014) — unitized nav 累计收益两次虚高实证
    #    (组合 v5 自报 +661% vs 面板 +43.79% = 15x; v6 A 自报 +3640% vs 面板 +12.98% = 280x),
    #    与自身 IC 证据矛盾, 误导裁决。铁律: JQ 面板是唯一真相源, 内部 nav 禁止用于裁决。
    #    检出 weekly_rets/cumprod/nav_prev 模式 → 警告 (不影响执行, 但禁止据此下结论)。
    _has_internal_nav = bool(
        re.search(r'weekly_rets|np\.cumprod\(1\s*\+|nav_prev|nav_cur', src))
    if _has_internal_nav:
        issues.append(
            "I 内部自算净值(unitized nav) — 两次虚高前科(15x/280x), 禁止用于裁决, "
            "面板为唯一真相源; 新文件建议移除该段")

    # J: system 日志抑制 (P-20260902-016) — order_target/order_target_value 对
    #    未持仓标的下单时, JQ 引擎内部查询 positions[security] → 每个标的刷一条
    #    WARNING "在 positions 中不存在, 为了保持兼容..." (首日建仓 ~80条/周)。
    #    JQ 官方抑制方法: log.set_level("system", "error")
    #    (joinquant.com/post/13861)。缺该行 → 日志被警告刷屏, 淹没自有 DIAG 日志。
    _has_order_line = ('log.set_level("order"' in src
                       or "log.set_level('order'" in src)
    _has_system_line = ('log.set_level("system"' in src
                        or "log.set_level('system'" in src)
    if ('order_target' in src) and _has_order_line and not _has_system_line:
        issues.append(
            "J 缺 system 日志抑制: order_target_value 对未持仓标的内部仓位查询会刷 "
            "WARNING (\"在 positions 中不存在\") → initialize 加 "
            "log.set_level(\"system\", \"error\")")

    for m in re.finditer(r'FACTOR_FUNC_MAP\s*=\s*\{(.*?)\}', src, re.S):
        for name_m in re.finditer(r'["\']([\w]+)["\']\s*:\s*factor_([\w]+)', m.group(1)):
            v = name_m.group(2)
            if "def factor_%s" % v not in src:
                issues.append("C MAP 指向缺失 def factor_%s" % v)

    return issues


def main():
    verbose = "--verbose" in sys.argv
    # 🔴 扫描范围 = FACTOR_DIR 下全部 jq_*_standalone.py (2026-09-01 扩宽:
    #    原 glob 只覆盖 jq_s5_pass_* / jq_combo_*, 漏掉 jq_smallcap_reversal_inpool_v6
    #    → panel=False 非法参数漏检 → 真实 JQ 零收益)
    files = sorted(glob.glob(os.path.join(FACTOR_DIR, "jq_*_standalone.py")))
    bad = []
    for f in files:
        issues = audit_file(f)
        if issues:
            bad.append((os.path.basename(f), issues))
        elif verbose:
            print("OK  %s" % os.path.basename(f))
    print("扫描 %d 个文件" % len(files))
    if not bad:
        print("CLEAN: 无任何零收益反模式残留")
        return 0
    print("%d 个文件有问题:" % len(bad))
    for name, issues in bad:
        print("  %s: %s" % (name, "; ".join(issues)))
    return 1


if __name__ == "__main__":
    sys.exit(main())
