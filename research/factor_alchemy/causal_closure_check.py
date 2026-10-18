# -*- coding: utf-8 -*-
"""P-20260910-001: 因子公式 DSL 静态因果闭包校验器 (shadow 默认, 零裁决)。

设计原则 (对齐 AQuA 'causality closed under composition'):
  - 时间序列算子只允许 trailing 窗口: rolling(N)/shift(k>=0)/ewm
  - 截面算子只允许当期单点 (rank 等需 axis=1, 属语义校验, 此处只做方向性检查)
  - 全样本累积算子 (裸 cummax/cummin/cumsum/cumprod/expanding) 禁止
  - 未来回填 (bfill/fillna(method='bfill')) 禁止

模式: FUXI_CAUSAL_CLOSURE = shadow(默认, 只记录不拦截) | enforce(拦截) | off
报告: data/causal_closure_report.json (shadow 期记录拦截数/误杀率, 30 天后评估再转 enforce)
"""
import ast
import json
import os
from datetime import datetime
from pathlib import Path

MODE = os.environ.get("FUXI_CAUSAL_CLOSURE", "shadow").lower()
assert MODE in ("shadow", "enforce", "off"), "FUXI_CAUSAL_CLOSURE must be shadow|enforce|off"

_REPORT_PATH = Path(__file__).resolve().parents[2] / "data" / "causal_closure_report.json"

# 全样本累积算子 (裸调用 = 读到序列起点 = 前视)
_FULL_SAMPLE_OPS = {"cummax", "cummin", "cumsum", "cumprod", "expanding"}
# 回看类算子 (需人工确认 axis 语义)
_LOOKBACK_OPS = {"idxmax", "idxmin", "argmax", "argmin"}


def _is_rolling_chain(node):
    """检查调用链上是否已挂 rolling/ewm 窗口 (窗口内累积 → 合法)。"""
    cur = node
    depth = 0
    while isinstance(cur, (ast.Call, ast.Attribute)) and depth < 8:
        depth += 1
        if isinstance(cur, ast.Call):
            f = cur.func
            if isinstance(f, ast.Attribute) and f.attr in ("rolling", "ewm"):
                return True
            cur = f
        else:
            cur = cur.value
    return False


def _num_value(a):
    """提取数值字面量 (含负数 UnaryOp(USub) 写法)。"""
    if isinstance(a, ast.Constant) and isinstance(a.value, (int, float)):
        return a.value
    if isinstance(a, ast.UnaryOp) and isinstance(a.op, ast.USub):
        v = _num_value(a.operand)
        return -v if v is not None else None
    return None


def check_causal_closure(formula: str) -> dict:
    """解析公式 AST, 返回 {verdict, violations, warnings, parse_ok}。
    verdict: ok | fatal | warn_only。shadow 模式下调用方不据此裁决。"""
    if not formula or not str(formula).strip():
        return {"verdict": "fatal", "violations": ["空公式"], "warnings": [], "parse_ok": False}

    try:
        tree = ast.parse(str(formula))
    except SyntaxError as e:
        return {"verdict": "warn_only", "violations": [],
                "warnings": [f"AST 解析失败: {e}"], "parse_ok": False}

    violations, warnings = [], []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        if not isinstance(f, ast.Attribute):
            continue
        attr = f.attr

        # 1) shift(k<0) / pct_change(k<0) → 前视
        if attr in ("shift", "pct_change") and n.args:
            v = _num_value(n.args[0])
            if v is not None and v < 0:
                violations.append(f"{attr}({v}): 读取未来 {int(-v)} 期 (前视)")

        # 2) 裸全样本累积算子
        if attr in _FULL_SAMPLE_OPS and not _is_rolling_chain(f):
            violations.append(f"{attr}(): 全样本累积 (隐式前视)")

        # 3) 未来回填
        if attr in ("bfill", "backfill"):
            violations.append(f"{attr}(): 用未来值回填 (前视)")
        if attr == "fillna":
            for kw in n.keywords:
                if kw.arg == "method" and isinstance(kw.value, ast.Constant) and kw.value.value == "bfill":
                    violations.append("fillna(method='bfill'): 用未来值回填 (前视)")

        # 4) 回看类算子 → 警告 (静态无法判定 axis, 需人工确认截面语义)
        if attr in _LOOKBACK_OPS:
            warnings.append(f"{attr}(): 需确认 axis=1 截面语义 (axis=0 为时间轴前视)")

    # 5) 文本级兜底: AST 覆盖不到的复杂写法 (字符串模板/嵌套调用)
    import re
    if not any(v.startswith("shift(-") for v in violations):
        for m in re.finditer(r"\.shift\(\s*-\s*(\d+)", str(formula)):
            violations.append(f"shift(-{m.group(1)}): 前视 (文本兜底)")
    if not any(v.startswith("pct_change(-") for v in violations):
        for m in re.finditer(r"\.pct_change\(\s*-\s*(\d+)", str(formula)):
            violations.append(f"pct_change(-{m.group(1)}): 前视 (文本兜底)")

    verdict = "fatal" if violations else ("warn_only" if warnings else "ok")
    return {"verdict": verdict, "violations": list(dict.fromkeys(violations)),
            "warnings": list(dict.fromkeys(warnings)), "parse_ok": True}


def record_batch(entries: list):
    """批量落盘 shadow 报告。entries: [{factor, formula_preview, verdict, violations, warnings, mode}]"""
    if MODE == "off":
        return
    try:
        rep = {}
        if _REPORT_PATH.exists():
            try:
                rep = json.loads(_REPORT_PATH.read_text(encoding="utf-8"))
            except Exception:
                rep = {}
        recs = rep.setdefault("records", [])
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for e in entries:
            recs.append({**e, "ts": now, "mode": MODE})
        # 防止无限膨胀: 只保留最近 500 条
        rep["records"] = recs[-500:]
        rep["summary"] = {
            "total_checked": rep.get("summary", {}).get("total_checked", 0) + len(entries),
            "total_fatal": sum(1 for e in entries if e.get("verdict") == "fatal"),
            "total_warn": sum(1 for e in entries if e.get("verdict") == "warn_only"),
            "last_run": now,
            "mode": MODE,
        }
        _REPORT_PATH.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"  [CausalClosure] 报告落盘失败 (不阻塞): {e}")


def should_block(verdict: str) -> bool:
    """enforce 模式下 verdict=fatal 才拦截; shadow 永不拦截。"""
    return MODE == "enforce" and verdict == "fatal"


if __name__ == "__main__":
    tests = [
        ("close.pct_change()", "ok"),
        ("close.shift(-5)", "fatal"),
        ("close.rolling(20).max() / close.shift(1)", "ok"),
        ("volume.cumsum()", "fatal"),
        ("close.expanding().mean()", "fatal"),
        ("df.fillna(method='bfill')", "fatal"),
        ("close.rolling(20).sum().shift(-1)", "fatal"),
        ("factor.idxmax(axis=1)", "warn_only"),
        ("close.diff(1) / close.shift(1)", "ok"),
        ("close.rolling(5).rank(pct=True)", "ok"),
    ]
    for formula, expect in tests:
        r = check_causal_closure(formula)
        mark = "OK " if r["verdict"] == expect else "MISMATCH"
        print(f"[{mark}] expect={expect:9s} got={r['verdict']:9s} | {formula[:48]}")
        if r["violations"]:
            print(f"       violations: {r['violations']}")
        if r["warnings"]:
            print(f"       warnings: {r['warnings']}")
