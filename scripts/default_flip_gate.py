# -*- coding: utf-8 -*-
"""翻默认四关检查单 —— 把文字铁律变成可执行入口 (2026-09-18 外部审计 S-6)。

背景:
    「翻默认四关」(getter 实测 / A/B 反事实 / 默认↔口径档不变量 / 端到端对账)
    一直是**文字铁律**, 散在 MEMORY 与 reports 里。而外部审计项目最值得学的一点是:
    它把研究者的心智模型固化成了**可执行断言**。

    本脚本把四关合并成**一个只读入口**, 输出 PASS/FAIL/SKIP 表 + 缺口清单。
    积木其实早就有了 (``data/default_vs_caliber_check_*.json``、
    ``data/default_flip_e2e_*.json``), 只是没合并。

🔴 本脚本**只读**:
    不自动跑高成本 A/B (那是小时级任务, 必须显式发起);
    不修改任何默认值、不写 env、不碰提案库。
    没有证据的关一律记 **SKIP(缺证据)**, 而不是默认 PASS —— 缺证据不等于通过。

四关映射:
    G1 getter 实测          : 源码默认位扫描 + (可选)真实 import 调用 getter
    G2 A/B 反事实           : 读外部 A/B 证据文件 (--ab-evidence)
    G3 默认↔口径档不变量    : data/default_vs_caliber_check_*.json
    G4 端到端对账           : data/default_flip_e2e_*.json

用法:
    python scripts/default_flip_gate.py --env FUXI_S5_IDX_ALIGN --new on --old off
    python scripts/default_flip_gate.py --env FUXI_S5_COST_BPS --new 15 --old 0 \
        --ab-evidence data/s5_fullpool_rerun_result_legacy.json
    python scripts/default_flip_gate.py --list-evidence
退出码: 0=全 PASS / 1=有 FAIL / 2=有 SKIP(证据不全)
"""
import argparse
import ast
import glob
import io
import json
import os
import sys
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
FA = BASE / "research" / "factor_alchemy"

# 扫描默认位的源码范围 (只看主实现, 显式排除历史快照与发布包 ⇒ 避免假命中)
_SCAN_DIRS = [FA, BASE / "scripts", BASE / "signals", BASE / "strategies"]
_EXCLUDE_PARTS = {"_versioning", "archive", "__pycache__", "output", "_ext",
                  "pandafactor_env", "third_party", "mcp-servers"}

DEFAULT_LITERAL_FUNCS = {"get", "setdefault"}


# ─────────────────────────── 通用小工具 ───────────────────────────

def _iter_src():
    for d in _SCAN_DIRS:
        if not d.exists():
            continue
        for root, dirs, files in os.walk(d):
            parts = set(Path(root).parts)
            if parts & _EXCLUDE_PARTS:
                dirs[:] = []
                continue
            for f in files:
                if f.endswith(".py"):
                    yield Path(root) / f


def _rel(p):
    try:
        return str(Path(p).relative_to(BASE)).replace("\\", "/")
    except Exception:
        return str(p)


def _read_text(p):
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _norm(v):
    """值归一成可比较字符串 (bool/int/float/str 统一)。"""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    return str(v).strip().lower()


def _find_default_sites(env):
    """AST 找 ``os.environ.get(ENV, <默认>)`` 的默认字面量。

    返回 [(file, lineno, default_str)]。用 AST 而非正则 ⇒ 不误抓注释与文档。
    """
    sites = []
    for py in _iter_src():
        text = _read_text(py)
        if env not in text:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if not isinstance(n, ast.Call):
                continue
            f = n.func
            fname = f.attr if isinstance(f, ast.Attribute) else (
                f.id if isinstance(f, ast.Name) else "")
            if fname not in DEFAULT_LITERAL_FUNCS or len(n.args) < 2:
                continue
            a0 = n.args[0]
            if not (isinstance(a0, ast.Constant) and a0.value == env):
                continue
            try:
                dv = ast.literal_eval(n.args[1])
            except Exception:
                dv = "<非字面量>"
            sites.append((_rel(py), n.lineno, _norm(dv)))
    return sites


def _find_getters(env):
    """AST 找引用该 env 且以 ``_`` 开头的模块级 getter 函数 → [(module, func)]。"""
    out = []
    for py in _iter_src():
        text = _read_text(py)
        if env not in text:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for n in tree.body:
            if not isinstance(n, ast.FunctionDef):
                continue
            seg = ast.get_source_segment(text, n) or ""
            if env in seg and "os.environ" in seg and n.name.startswith("_"):
                out.append((py, n.name))
    return out


def _latest(pattern):
    hits = sorted(glob.glob(str(DATA / pattern)), key=lambda p: os.path.getmtime(p))
    return Path(hits[-1]) if hits else None


def _load_json(p):
    try:
        return json.load(io.open(p, "r", encoding="utf-8"))
    except Exception:
        return None


# ─────────────────────────── 四关 ───────────────────────────

def gate1_getter(env, new, old, do_import=True):
    """G1: 源码默认位扫描 + 真实 getter 调用。"""
    ev, notes = [], []
    sites = _find_default_sites(env)
    if not sites:
        return "SKIP", "未找到 %s 的 os.environ.get 默认字面量" % env, ev

    vals = sorted({s[2] for s in sites})
    ev.append({"kind": "default_sites", "sites": ["%s:%d=%s" % s for s in sites],
               "distinct": vals})
    new_n, old_n = _norm(new), _norm(old)
    if vals == [new_n]:
        notes.append("全部默认位 = %s ✓ (%d 处)" % (new_n, len(sites)))
    elif old_n in vals:
        return ("FAIL",
                "仍有默认位停在旧值 %s: %s" % (
                    old_n, [s for s in sites if s[2] == old_n][:3]), ev)
    else:
        return "FAIL", "默认位既非新值(%s)也非旧值: %s" % (new_n, vals), ev

    if not do_import:
        return "PASS", "静态默认位通过 (未做 getter 实测: --no-import); " + "; ".join(notes), ev

    getters = _find_getters(env)
    if not getters:
        return "SKIP", "静态默认位通过, 但未找到 getter (无从实测)", ev

    sys.path.insert(0, str(FA))
    seen, results = set(), []
    for py, fn in getters:
        mod = py.stem
        if mod in seen:
            continue
        seen.add(mod)
        try:
            import importlib
            m = importlib.import_module(mod)
            g = getattr(m, fn)
        except Exception as e:
            results.append("%s.%s: import 失败 %s" % (mod, fn, type(e).__name__))
            continue
        # 缺省 ⇒ 应等于新默认
        saved = os.environ.pop(env, None)
        try:
            got = _norm(g())
        except Exception as e:
            results.append("%s.%s(缺省): 调用失败 %s" % (mod, fn, type(e).__name__))
            got = None
        if saved is not None:
            os.environ[env] = saved
        # 显式设旧值 ⇒ getter 应穿透
        os.environ[env] = old
        try:
            got_old = _norm(g())
        except Exception:
            got_old = None
        if saved is not None:
            os.environ[env] = saved
        else:
            os.environ.pop(env, None)

        ok = (got == new_n) and (got_old == old_n)
        results.append("%s.%s: 缺省=%s(期望 %s) 设旧值=%s(期望 %s) %s"
                       % (mod, fn, got, new_n, got_old, old_n, "✓" if ok else "✗"))
        ev.append({"kind": "getter_call", "module": mod, "func": fn,
                   "default": got, "expect_default": new_n,
                   "old": got_old, "expect_old": old_n, "ok": ok})

    bad = [r for r in results if "✗" in r or "失败" in r]
    if not results:
        return "SKIP", "无可用 getter", ev
    if bad:
        return "FAIL", "getter 实测不符: " + " | ".join(bad[:3]), ev
    return "PASS", "getter 实测通过: " + " | ".join(results[:2]), ev


def gate2_ab(env, ab_evidence):
    """G2: A/B 反事实 —— 只**读**外部证据, 不自动跑。"""
    ev = []
    switches = _read_text(BASE / "scripts" / "s5_fullpool_rerun.py")
    in_harness = ("\"%s\"" % env) in switches or ("'%s'" % env) in switches
    ev.append({"kind": "harness_membership", "in_s5_fullpool_rerun": in_harness})
    if ab_evidence:
        p = Path(ab_evidence)
        d = _load_json(p) if p.exists() else None
        if d is None:
            return "FAIL", "A/B 证据文件不可读: %s" % ab_evidence, ev
        keys = list(d.keys())[:14] if isinstance(d, dict) else []
        ev.append({"kind": "ab_evidence", "path": _rel(p), "top_keys": keys})
        return "PASS", "A/B 证据已提供: %s (顶层键 %s)" % (
            p.name, ",".join(map(str, keys[:6]))), ev
    if in_harness:
        return "SKIP", ("该 env 已是 s5_fullpool_rerun 口径/AB 台架的受控键, "
                        "但本次未提供 A/B 读数 (--ab-evidence)"), ev
    return "SKIP", "无 A/B 台架归属, 且未提供 A/B 证据", ev


def gate3_caliber(env):
    """G3: 默认↔口径档不变量 —— 读 default_vs_caliber_check_*.json。"""
    p = _latest("default_vs_caliber_check_*.json")
    if p is None:
        return "SKIP", "无 default_vs_caliber_check_*.json 证据", []
    d = _load_json(p) or {}
    checks = d.get("checks") or []
    mine = [c for c in checks if env in str(c.get("name", ""))]
    ev = [{"kind": "caliber_evidence", "path": _rel(p),
           "verdict": d.get("verdict"), "n_checks": len(checks),
           "matched": len(mine)}]
    if not mine:
        return "SKIP", ("证据 %s 中无 %s 相关条目 (该档未覆盖此 env) ⇒ 需先把它"
                        "纳入口径清单再验证" % (p.name, env)), ev
    bad = [c.get("name") for c in mine if not c.get("ok")]
    if bad:
        return "FAIL", "口径不变量失败: %s" % bad[:3], ev
    return "PASS", "口径不变量通过 (%d 条: %s)" % (
        len(mine), "; ".join(str(c.get("name"))[:48] for c in mine[:2])), ev


def gate4_e2e(env):
    """G4: 端到端对账 —— 读 default_flip_e2e_*.json。"""
    p = _latest("default_flip_e2e_*.json")
    if p is None:
        return "SKIP", "无 default_flip_e2e_*.json 证据", []
    d = _load_json(p) or {}
    scen = d.get("scenarios") or []
    mine = [s for s in scen if env in (s.get("env") or {})]
    ev = [{"kind": "e2e_evidence", "path": _rel(p), "verdict": d.get("verdict"),
           "n_scenarios": len(scen), "matched": len(mine),
           "switch_wired": d.get("switch_wired")}]
    if not mine:
        return "SKIP", ("证据 %s 的 %d 个场景均未设置 %s ⇒ 该 env 未被端到端覆盖"
                        % (p.name, len(scen), env)), ev
    bad = []
    for s in mine:
        for k, v in (s.get("checks") or {}).items():
            if not v:
                bad.append("%s/%s" % (s.get("tag"), k))
    if bad:
        return "FAIL", "端到端对账失败: %s" % bad[:3], ev
    return "PASS", "端到端对账通过 (%d 场景: %s)" % (
        len(mine), ", ".join(str(s.get("tag")) for s in mine[:3])), ev


# ─────────────────────────── 主流程 ───────────────────────────

def list_evidence():
    print("证据文件 (取最新):")
    for pat, gate in [("default_vs_caliber_check_*.json", "G3"),
                      ("default_flip_e2e_*.json", "G4")]:
        p = _latest(pat)
        print("  %s %-34s %s" % (gate, pat, _rel(p) if p else "(缺失)"))
    print("\n提示: G2 的 A/B 读数需用 --ab-evidence 显式指定 (本脚本不自动跑高成本 A/B)。")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", help="开关名, 如 FUXI_S5_IDX_ALIGN")
    ap.add_argument("--new", help="新默认值")
    ap.add_argument("--old", help="旧默认值")
    ap.add_argument("--ab-evidence", help="G2 的 A/B 读数文件")
    ap.add_argument("--no-import", action="store_true",
                    help="跳过 getter 真实 import 调用 (只做源码默认位扫描)")
    ap.add_argument("--list-evidence", action="store_true")
    ap.add_argument("--json", help="报告落盘路径")
    a = ap.parse_args()

    if a.list_evidence:
        return list_evidence()
    if not a.env:
        ap.error("需要 --env (或用 --list-evidence)")

    ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    print("=" * 76)
    print("翻默认四关检查单  @ %s" % ts)
    print("env=%s  new=%s  old=%s" % (a.env, a.new, a.old))
    print("=" * 76)

    gates = [("G1", "getter 实测", lambda: gate1_getter(
                  a.env, a.new or "", a.old or "", do_import=not a.no_import)),
             ("G2", "A/B 反事实", lambda: gate2_ab(a.env, a.ab_evidence)),
             ("G3", "默认↔口径不变量", lambda: gate3_caliber(a.env)),
             ("G4", "端到端对账", lambda: gate4_e2e(a.env))]

    results, evidence = [], []
    for gid, name, fn in gates:
        st, detail, ev = fn()
        results.append({"gate": gid, "name": name, "status": st, "detail": detail})
        evidence.append({"gate": gid, "evidence": ev})
        mark = {"PASS": "[PASS]", "FAIL": "[FAIL]", "SKIP": "[SKIP]"}[st]
        print("%s %s %-18s %s" % (mark, gid, name, detail))

    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    n_skip = sum(1 for r in results if r["status"] == "SKIP")
    if n_fail:
        verdict = "BLOCK: 有 %d 关 FAIL ⇒ **不可翻默认**" % n_fail
        rc = 1
    elif n_skip:
        verdict = "INCOMPLETE: %d 关缺证据 (SKIP) ⇒ 补齐后才可翻; 缺证据 ≠ 通过" % n_skip
        rc = 2
    else:
        verdict = "PASS: 四关全部通过"
        rc = 0
    print("-" * 76)
    print(verdict)
    if n_skip:
        print("缺证据的关: " + ", ".join("%s(%s)" % (r["gate"], r["name"])
                                        for r in results if r["status"] == "SKIP"))

    out = Path(a.json) if a.json else (DATA / ("default_flip_gate_%s_%s.json" % (
        a.env.replace("FUXI_", "").lower(), datetime.now().strftime("%Y%m%d"))))
    try:
        with io.open(out, "w", encoding="utf-8") as f:
            json.dump({"generated_at": ts, "env": a.env, "new": a.new, "old": a.old,
                       "gates": results, "evidence": evidence,
                       "n_fail": n_fail, "n_skip": n_skip, "verdict": verdict,
                       "rc": rc}, f, ensure_ascii=False, indent=2)
        print("[落盘] %s" % _rel(out))
    except Exception as e:
        print("[落盘失败] %s" % e)
    return rc


if __name__ == "__main__":
    sys.exit(main())
