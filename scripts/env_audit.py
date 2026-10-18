# -*- coding: utf-8 -*-
"""FUXI_* 环境变量三态审计 (活 / 影 / 死) —— 只读, 不改任何状态。

动机 (2026-09-18 外部审计 P0/S-2):
    仓库内 FUXI_* 环境变量已达三个数量级, 但「口径档」只覆盖其中很小一部分。
    仅凭「读点存在」无法判断一个开关是**正常可调参数**还是**无人写入的死开关**,
    必须靠机械审计区分, 不能靠感觉。

三态判据 (锚在**实现**, 不锚命名):
    活 = 存在**写入点** (代码里显式赋值该 env: ``os.environ[k] = v`` /
         ``os.environ.setdefault(k, v)`` / ``os.putenv``), 或落在口径/AB 清单内。
    影 = 无写入点, 但出现在 ``CHANGELOG.md`` 的提案引用里 (历史上有提案背书)。
    死 = 无写入点、不在任何清单、CHANGELOG 亦无引用 ⇒ 候选删除/归档。

🔴 为何用 AST 而非 import 取口径清单:
    ``import s5_fullpool_rerun`` 会连带 import 下游重模块 (s135_candidate_check),
    审计脚本必须**零副作用**。故全部清单就地用 AST 解析字面量。

🔴 为何排除 ``_versioning/`` 与 ``archive/``:
    这些目录是代码树内的历史快照 (含另一份 ralph_loop.py), 会让审计产生**假命中**
    (2026-09-18 外部审计实测: 统计 env 数量时被它们污染)。本脚本默认排除, 并在
    输出里单独报告「因排除而少见的命中数」, 使假命中可见而非静默消失。

用法:
    python scripts/env_audit.py                      # 全量三态审计
    python scripts/env_audit.py --only-dead          # 只看死开关
    python scripts/env_audit.py --json data/env_audit_report.json
"""
import argparse
import ast
import io
import json
import os
import re
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

# ── 扫描范围 ──
# 排除 = 非本项目源码 / 外部克隆 / 构建产物 / 依赖目录 / 缓存 / 历史快照
#        (历史快照单独统计假命中)
# 🔴 2026-09-18 实测: 不排除依赖目录会扫到 13968 个 .py (pandafactor_env 10247 +
#    mcp-servers 4877 + third_party 329), env 统计被 site-packages 污染。
_EXCLUDE_ALWAYS = {
    "_ext",                    # 外部克隆 (只读参考)
    "output",                  # 发布包与打包副本
    "pandafactor_env", "panda_factor", "third_party", "mcp-servers", "_ref_quantmind",
    "site-packages", "dist-packages", "node_modules",
    "__pycache__", ".git", ".workbuddy", ".idea", ".venv", ".tox", "build", "dist",
}
_EXCLUDE_HISTORY = {"_versioning", "archive"}      # 代码树内的历史快照 ⇒ 假命中源
_SCAN_SUFFIX = {".py"}

_ENV_RE = re.compile(r"\bFUXI_[A-Z0-9_]+\b")
# 读点: os.environ.get("X") / os.environ["X"] / os.getenv("X") / environ.get("X")
_READ_RE = re.compile(
    r"""(?:os\.)?environ(?:\.get)?\s*[\(\[]\s*['"](FUXI_[A-Z0-9_]+)['"]"""
    r"""|os\.getenv\s*\(\s*['"](FUXI_[A-Z0-9_]+)['"]"""
)
# 写点: os.environ["X"] = ...   (含 setdefault / putenv)
_WRITE_RE = re.compile(
    r"""(?:os\.)?environ\s*\[\s*['"](FUXI_[A-Z0-9_]+)['"]\s*\]\s*="""
    r"""|(?:os\.)?environ\.setdefault\s*\(\s*['"](FUXI_[A-Z0-9_]+)['"]"""
    r"""|os\.putenv\s*\(\s*['"](FUXI_[A-Z0-9_]+)['"]"""
)


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(BASE)).replace("\\", "/")
    except Exception:
        return str(p)


def _iter_py():
    """产出 (path, rel, is_history)。history=True 的命中只用于假命中统计。"""
    for root, dirs, files in os.walk(BASE):
        rp = Path(root)
        parts = set(rp.relative_to(BASE).parts) if rp != BASE else set()
        if parts & _EXCLUDE_ALWAYS:
            dirs[:] = []
            continue
        hist = bool(parts & _EXCLUDE_HISTORY)
        for f in files:
            if Path(f).suffix.lower() in _SCAN_SUFFIX:
                yield rp / f, _rel(rp / f), hist


def _find_groups(pattern, text):
    """返回 {env: [行号, ...]} (pattern 有多个可选捕获组)。"""
    out = {}
    for m in pattern.finditer(text):
        name = next((g for g in m.groups() if g), None)
        if not name:
            continue
        line = text.count("\n", 0, m.start()) + 1
        out.setdefault(name, []).append(line)
    return out


def collect_switch_lists():
    """AST 解析 s5_fullpool_rerun.py 的口径清单 (零 import)。"""
    path = BASE / "scripts" / "s5_fullpool_rerun.py"
    res = {"file": _rel(path), "exists": path.exists(),
           "ALL_SWITCH_ENVS": [], "CALIBERS": {}, "other_keys": {}}
    if not path.exists():
        return res
    src = path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(src)
    wanted = {"ALL_SWITCH_ENVS", "_WINDOW_KEYS", "_SETDEFAULT_KEYS", "_AB_KEYS", "CALIBERS"}

    def _lit(node):
        try:
            return ast.literal_eval(node)
        except Exception:
            return None

    for n in tree.body:
        if not isinstance(n, ast.Assign) or len(n.targets) != 1:
            continue
        t = n.targets[0]
        if not isinstance(t, ast.Name) or t.id not in wanted:
            continue
        v = _lit(n.value)
        if v is None:
            continue
        if t.id == "ALL_SWITCH_ENVS":
            res["ALL_SWITCH_ENVS"] = list(v)
        elif t.id == "CALIBERS":
            res["CALIBERS"] = {k: sorted(vv.keys()) for k, vv in v.items()}
        else:
            res["other_keys"][t.id] = sorted(v)
    return res


def collect_changelog():
    """影子态判据的两个来源: CHANGELOG 文本 + 提案库。

    🔴 只查 CHANGELOG 会**误判**: CHANGELOG 是渲染后的版本条目, 常常只写提案标题
    而不含 env 字面量 ⇒ 一个**有正式提案**的开关会被判成「死」(2026-09-18 首跑实测:
    FUXI_CAUSAL_CLOSURE / FUXI_S5_SPARSE_DIAG 均被误判)。提案库才是权威来源。
    """
    out = {}
    p = BASE / "CHANGELOG.md"
    out["changelog_path"] = _rel(p)
    out["changelog"] = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
    q = BASE / "data" / "stage1_optimization_proposals.json"
    out["proposal_path"] = _rel(q)
    out["proposals"] = q.read_text(encoding="utf-8", errors="replace") if q.exists() else ""
    return out


def _n_occur(hay, needle, limit=999):
    """计数 needle 在 hay 中出现的次数 (上限 limit)。"""
    n, i = 0, 0
    while True:
        i = hay.find(needle, i)
        if i < 0:
            return n
        n += 1
        i += len(needle)
        if n >= limit:
            return n


def audit():
    switches = collect_switch_lists()
    src = collect_changelog()
    chlog, props = src["changelog"], src["proposals"]

    # 口径档里显式被设置的全部键 (含 setdefault 族) —— 这些开不了口就是"影"
    caliber_keys = set()
    for _name, keys in (switches.get("CALIBERS") or {}).items():
        caliber_keys |= set(keys)
    for _k, v in (switches.get("other_keys") or {}).items():
        caliber_keys |= set(v)

    readers, writers = {}, {}
    history_reads = 0
    history_envs = set()
    scanned = 0
    for path, rel, is_hist in _iter_py():
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if is_hist:
            g = _find_groups(_READ_RE, text)
            history_reads += sum(len(l) for l in g.values())
            history_envs |= set(g)
            continue
        scanned += 1
        for name, lines in _find_groups(_READ_RE, text).items():
            readers.setdefault(name, []).append((rel, lines))
        for name, lines in _find_groups(_WRITE_RE, text).items():
            writers.setdefault(name, []).append((rel, lines))

    rows = []
    all_envs = sorted(set(readers) | set(writers))
    for name in all_envs:
        rd, wr = readers.get(name, []), writers.get(name, [])
        in_switch = name in set(switches.get("ALL_SWITCH_ENVS") or [])
        in_caliber = name in caliber_keys
        n_chlog = _n_occur(chlog, name)
        n_prop = _n_occur(props, name)
        if wr or in_switch or in_caliber:
            state = "live"
        elif n_chlog or n_prop:
            state = "shadow"
        else:
            state = "dead"
        rows.append({
            "env": name,
            "state": state,
            "n_readers": sum(len(l) for _f, l in rd),
            "n_writers": sum(len(l) for _f, l in wr),
            "reader_sites": ["%s:%d" % (f, ln) for f, ls in rd for ln in ls],
            "writer_sites": ["%s:%d" % (f, ln) for f, ls in wr for ln in ls],
            "in_ALL_SWITCH_ENVS": in_switch,
            "in_caliber_keys": in_caliber,
            "changelog_refs": n_chlog,
            "proposal_refs": n_prop,
            "single_site": (sum(len(l) for _f, l in rd) + sum(len(l) for _f, l in wr)) <= 1,
        })

    covered = set(switches.get("ALL_SWITCH_ENVS") or []) | caliber_keys
    rows.sort(key=lambda r: ({"dead": 0, "shadow": 1, "live": 2}[r["state"]], r["env"]))
    n = {s: sum(1 for r in rows if r["state"] == s) for s in ("live", "shadow", "dead")}
    return {
        "scanned_py": scanned,
        "excluded_dirs": sorted(_EXCLUDE_ALWAYS | _EXCLUDE_HISTORY),
        "switch_source": switches,
        "changelog_source": {k: v for k, v in src.items() if k in ("changelog_path", "proposal_path")},
        "caliber_covered_keys": sorted(covered),
        "n_caliber_covered": len(covered),
        "caliber_coverage_pct": round(100.0 * len(covered) / max(1, len(rows)), 2),
        "n_total": len(rows),
        "n_live": n["live"], "n_shadow": n["shadow"], "n_dead": n["dead"],
        "n_single_site": sum(1 for r in rows if r["single_site"]),
        "history_fake_hit_envs": len(history_envs),
        "history_fake_hit_sites": history_reads,
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(BASE / "data" / "env_audit_report.json"),
                    help="报告落盘路径")
    ap.add_argument("--only-dead", action="store_true", help="只打印死开关")
    ap.add_argument("--only-1site", action="store_true", help="只打印全文仅 1 处出现者")
    ap.add_argument("--top", type=int, default=0, help="最多打印 N 行 (0=不限)")
    a = ap.parse_args()

    rep = audit()
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    print("=" * 78)
    print("FUXI_* 环境变量三态审计  @ %s" % ts)
    print("=" * 78)
    print("扫描 .py 文件数      : %d" % rep["scanned_py"])
    print("排除目录             : %s" % ", ".join(rep["excluded_dirs"]))
    sw = rep["switch_source"]
    print("口径清单来源         : %s (存在=%s)" % (sw["file"], sw["exists"]))
    print("  ALL_SWITCH_ENVS    : %d 个" % len(sw["ALL_SWITCH_ENVS"]))
    for k, v in (sw.get("other_keys") or {}).items():
        print("  %-18s : %d 个 %s" % (k, len(v), v))
    print("-" * 78)
    print("env 总数             : %d" % rep["n_total"])
    print("  活 (live)          : %d" % rep["n_live"])
    print("  影 (shadow)        : %d" % rep["n_shadow"])
    print("  死 (dead)          : %d   <-- 候选删除/归档" % rep["n_dead"])
    print("全文仅 1 处出现      : %d" % rep["n_single_site"])
    print("口径档实际覆盖键     : %d / %d = %.1f%%   <-- 覆盖率越低, 零回归基线越可能"
          % (rep["n_caliber_covered"], rep["n_total"], rep["caliber_coverage_pct"]))
    print("历史快照带来的假命中 : %d 个 env / %d 处 (已排除, 仅统计)"
          % (rep["history_fake_hit_envs"], rep["history_fake_hit_sites"]))
    print("-" * 78)

    sel = rep["rows"]
    if a.only_dead:
        sel = [r for r in sel if r["state"] == "dead"]
    if a.only_1site:
        sel = [r for r in sel if r["single_site"]]
    if a.top > 0:
        sel = sel[:a.top]

    tag = {"live": "[活]", "shadow": "[影]", "dead": "[死]"}
    for r in sel:
        sites = r["reader_sites"][:2]
        print("%s %-40s 读%2d 写%2d 口径=%-2s chg=%2d prop=%2d%s"
              % (tag[r["state"]], r["env"], r["n_readers"], r["n_writers"],
                 "Y" if (r["in_ALL_SWITCH_ENVS"] or r["in_caliber_keys"]) else "N",
                 r["changelog_refs"], r["proposal_refs"],
                 "  首读点: " + ", ".join(sites) if sites else ""))
    if not sel:
        print("(无匹配行)")

    out = Path(a.json)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"generated_at": ts, **rep}, f, ensure_ascii=False, indent=2)
        print("\n[落盘] %s" % _rel(out))
    except Exception as e:
        print("\n[落盘失败] %s: %s" % (type(e).__name__, e))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
