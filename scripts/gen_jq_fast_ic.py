# -*- coding: utf-8 -*-
"""
JQ 单因子 IC 快验批量生成器 (v1.0, 2026-08-31)
==============================================

背景 (分离力分析 P-20260831):
  - 本地周频 pandas_icir 对 JQ 成败无分离力 (p=0.139); 仅 JQ 口径指标有 (jq_icir p=0.032)
  - 完整组合回测贵且慢 (每因子数分钟+人工看日志) → 锚点积累慢
  - 本工具: 一次 JQ 回测批量计算 N 个因子的周频 rank IC/ICIR (纯因子计算,
    无组合模拟/无订单/无成本) → [FAST_IC] 日志 → 本地 ingest 回写队列
  - 定位: 第二道门 (影子模式默认, FAST_IC_GATE_ENFORCE=False 只记录不否决)

用法:
  # 1. 生成快验脚本 (默认取队列中 natural_freq=weekly 且 pending_jq_run 的条目)
  python scripts/gen_jq_fast_ic.py --generate
  #    强制包含指定因子 (已回测的 backfill 也可用于校验快验与全回测口径一致性)
  python scripts/gen_jq_fast_ic.py --generate --include weekly_backfill_calib_forge__36fe6513
  # 2. 用户把生成的 jq_fast_ic_batch_*.py 上传聚宽跑回测, 复制日志到本地文件
  # 3. 回灌结果 (解析 [FAST_IC] 行 → 队列 fast_ic_result 字段)
  python scripts/gen_jq_fast_ic.py --ingest <日志文件>

依赖: 复用 scripts/gen_jq_generic.py 的公式翻译/字段分析/lookback 推断/smoke 防线
"""

import json
import os
import re
import sys
from datetime import datetime

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR = os.path.join(BASE, "research", "factor_alchemy")
QUEUE_PATH = os.path.join(DATA_DIR, "jq_codegen_queue.json")

# ── 门限配置 (影子模式默认) ──
FAST_IC_GATE_ENFORCE = False   # True = 不通过则状态置 fast_ic_blocked; False = 只记录
FAST_IC_ICIR_THRESHOLD = 0.20  # JQ 周频 ICIR 门槛 (取 abs, 负向因子取负即正)
FAST_IC_MIN_N = 20             # 最少有效周样本
FAST_IC_MIN_ABS_IC = 0.005     # |mean IC| 下限

# ── JQ 端快验参数 ──
UNIVERSE_IDX = ["000300.XSHG", "000905.XSHG"]   # 中证800 (池口径失真教训: 不用 200 小池)
END_DATE = "2026-08-28"
COUNT = 1300                                     # ~5 年日线, 周频网格约 260 周
WEEK = 5
MAX_FACTORS_PER_BATCH = 10

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_jq_generic import (  # noqa: E402
    prepare_factor, _smoke_fix_lookback, _RANK_HELPER, _RANK_PCT_HELPER,
)

_FNAME_RE = re.compile(r"\W")


def _sanitize(name):
    return "factor_" + _FNAME_RE.sub("_", name)


def _build_series_func(name, formula_jq, meaning):
    """生成全序列因子函数 (区别于 gen_jq_generic 的末行横截面版):
    eval 完整 DataFrame → 返回 dates×stocks 面板, 供周频网格 IC 计算。"""
    fname = _sanitize(name)
    esc = formula_jq.replace('"""', "'")
    body = (
        'def %(fname)s(close_df, extra_dfs=None):\n'
        '    """%(meaning)s\n'
        '    公式: %(esc)s\n'
        '    全序列版本 (fast-IC): 返回完整 dates x stocks DataFrame"""\n'
        "    nan_df = pd.DataFrame(np.nan, index=close_df.index, columns=close_df.columns)\n"
        "    ex = extra_dfs or {}\n"
        "    def _pick(key):\n"
        "        df = ex.get(key)\n"
        "        if df is None:\n"
        "            return nan_df\n"
        "        return df.reindex(index=close_df.index, columns=close_df.columns)\n"
        "    _LOCALS = {\n"
        '        "close": close_df, "open": _pick("open"), "high": _pick("high"),\n'
        '        "low": _pick("low"), "volume": _pick("volume"), "amount": _pick("amount"),\n'
        '        "close_p": close_df, "open_p": _pick("open"), "high_p": _pick("high"),\n'
        '        "low_p": _pick("low"), "volume_p": _pick("volume"), "amount_p": _pick("amount"),\n'
        '        "returns": close_df.pct_change(),\n'
        "    }\n"
        "    val = eval(_EXPR_%(fname)s, _GLOBALS_%(fname)s, _LOCALS)\n"
        "    if isinstance(val, pd.Series):\n"
        "        val = pd.DataFrame({%(fname)r: val}, index=close_df.index)\n"
        "    if not isinstance(val, pd.DataFrame):\n"
        "        val = pd.DataFrame(np.full(close_df.shape, float(val)),\n"
        "                            index=close_df.index, columns=close_df.columns)\n"
        "    return val.reindex(index=close_df.index, columns=close_df.columns)\n"
    ) % {"fname": fname, "meaning": meaning or "(fast-IC 批量快验)", "esc": esc}
    return fname, body


def _build_globals_block(fname, formula_jq):
    esc = formula_jq.replace('"', '\\"')
    return (
        '_EXPR_%(f)s = "%(expr)s"\n'
        '_GLOBALS_%(f)s = {\n'
        '    "np": np, "pd": pd,\n'
        '    "abs": np.abs, "sqrt": np.sqrt, "log": np.log, "log1p": np.log1p,\n'
        '    "exp": np.exp, "sign": np.sign,\n'
        '    "maximum": np.maximum, "minimum": np.minimum,\n'
        '    "where": np.where, "clip": np.clip,\n'
        '    "range": range, "len": len, "int": int, "float": float,\n'
        '    "list": list, "dict": dict, "tuple": tuple,\n'
        '    "_rolling_rank_pct_vec": _rolling_rank_pct_vec,\n'
        '    "_roll_rank_apply": _roll_rank_apply,\n'
        '    "_rank_pct": _rank_pct,\n'
        '}\n'
    ) % {"f": fname, "expr": esc}


_JQ_BATCH_TEMPLATE = '''# -*- coding: utf-8 -*-
"""
JQ 单因子 IC 快验 (批量, 无组合模拟) — 自动生成 {gen_time}
=========================================================
用途: 周频 rank IC/ICIR 第二道证据门 (分离力分析: 仅 JQ 口径指标有裁决力)
口径: 中证800 (000300+000905 并集), {count} 日线, 每 {week} 日网格周频 IC,
      因子用日线口径计算 (与完整回测 standalone 同一公式口径), fwd {week} 日收益为标签
输出: 日志 [FAST_IC] <name> ic=.. icir=.. n=.. hint=..  ← 本地 --ingest 解析回灌
成本: 无订单/无费用/无持仓 → 秒级~分钟级, 远快于组合回测
"""
from jqdata import *
import numpy as np
import pandas as pd
from scipy.stats import rankdata

{helpers}

{globals_blocks}

{series_funcs}

FACTOR_LIST = {factor_list}
FACTOR_FIELDS = {factor_fields}

def _spearman(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 20:
        return np.nan, int(m.sum())
    ra = rankdata(a[m])
    rb = rankdata(b[m])
    n = len(ra)
    ca = ra - ra.mean()
    cb = rb - rb.mean()
    va = float((ca * ca).sum())
    vb = float((cb * cb).sum())
    if va <= 0 or vb <= 0:
        return np.nan, n
    return float((ca * cb).sum()) / np.sqrt(va * vb), n

def initialize(context):
    log.info('[FAST-IC] start, building universe...')
    univ = []
    for idx in {universe_idx}:
        try:
            univ += list(get_index_stocks(idx))
        except Exception as e:
            log.warn('[FAST-IC] index fail %s: %s' % (idx, e))
    univ = sorted(set(univ))
    if len(univ) < 300:
        log.error('[FAST-IC] universe too small (%d), abort' % len(univ))
        return
    log.info('[FAST-IC] universe=%d, end=%s, count=%d' % (len(univ), '{end_date}', {count}))
    for name, fname in FACTOR_LIST:
        try:
            close = get_price(univ, end_date='{end_date}', count={count},
                              fields=['close'], fq='pre', skip_paused=False, panel=False)
            close = close.reindex(columns=univ)
            extras = {{}}
            for fld in FACTOR_FIELDS.get(name, []):
                if fld == 'close':
                    continue
                df = get_price(univ, end_date='{end_date}', count={count},
                               fields=[fld], fq='pre', skip_paused=False, panel=False)
                extras[fld] = df.reindex(index=close.index, columns=univ)
            val = globals()[fname](close, extras)
            fwd = close.shift(-{week}) / close - 1.0
            grid = np.arange(0, len(val), {week})
            ics = []
            for i in grid:
                ic, _n = _spearman(val.iloc[i].values, fwd.iloc[i].values)
                if np.isfinite(ic):
                    ics.append(ic)
            ics = np.asarray(ics, dtype=float)
            n_weeks = int(np.isfinite(ics).sum())
            if n_weeks >= 20:
                ic = float(np.nanmean(ics))
                sd = float(np.nanstd(ics, ddof=1))
                icir = ic / sd if sd > 0 else np.nan
            else:
                ic, icir = np.nan, np.nan
            hint = 'LONG' if ic > 0 else ('SHORT' if ic < 0 else 'FLAT')
            log.info('[FAST_IC] %s ic=%.4f icir=%.4f n=%d hint=%s'
                     % (name, ic, icir, n_weeks, hint))
        except Exception as e:
            log.info('[FAST_IC] %s ERROR %s' % (name, str(e)[:160]))

def handle_data(context, data):
    pass
'''


def _load_queue():
    if os.path.exists(QUEUE_PATH):
        try:
            return json.load(open(QUEUE_PATH, encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_queue(queue):
    json.dump(queue, open(QUEUE_PATH, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=str)


def select_entries(queue, include=None, all_weekly_pending=True):
    """挑选快验条目 → [(name, formula, meaning)]"""
    picked = []
    if include:
        for nm in include:
            e = queue.get(nm)
            if e and e.get("formula"):
                picked.append((nm, e["formula"], e.get("meaning", "")))
            else:
                print(f"[fast-ic] skip include {nm}: 队列无公式")
    if all_weekly_pending:
        for nm, e in queue.items():
            if not isinstance(e, dict) or not e.get("formula"):
                continue
            if nm in [p[0] for p in picked]:
                continue
            if e.get("natural_freq") == "weekly" and e.get("status") == "pending_jq_run":
                picked.append((nm, e["formula"], e.get("meaning", "")))
    return picked


def build_batch_script(entries):
    """entries: [(name, formula, meaning)] → (out_path, meta)"""
    ok_entries = []
    skipped = []
    for name, formula, meaning in entries:
        meta = prepare_factor(name, formula, meaning)
        if not meta["ok"]:
            skipped.append((name, meta.get("reason", "翻译失败")))
            continue
        meta = _smoke_fix_lookback(meta)
        if not meta["ok"]:
            skipped.append((name, meta.get("reason", "smoke 失败")))
            continue
        ok_entries.append((name, meta))
    if not ok_entries:
        return None, {"skipped": skipped}

    funcs, gblocks, names, fname_of = [], [], [], {}
    for name, meta in ok_entries:
        fname, body = _build_series_func(name, meta["formula_jq"],
                                         meta.get("meaning", ""))
        fname_of[name] = fname
        funcs.append(body)
        gblocks.append(_build_globals_block(fname, meta["formula_jq"]))
        names.append(name)
    needs_rank = any(m["needs_helper"] or m["needs_rank_helper"] for _, m in ok_entries)
    # 无条件生成 helper: _GLOBALS_* 块恒引用 _rolling_rank_pct_vec/_roll_rank_apply/_rank_pct,
    # 按需生成会在无 rank 公式的批次上触发 JQ NameError (2026-08-31 mock 实测捕获)。
    helpers = _RANK_HELPER + "\n\n" + _RANK_PCT_HELPER + "\n"

    code = _JQ_BATCH_TEMPLATE.format(
        gen_time=datetime.now().strftime("%Y-%m-%d %H:%M"),
        helpers=helpers,
        globals_blocks="\n\n".join(gblocks),
        series_funcs="\n\n".join(funcs),
        factor_list=json.dumps([[n, fname_of[n]] for n in names], ensure_ascii=False),
        factor_fields=json.dumps(
            {n: m["fields"] for n, m in ok_entries}, ensure_ascii=False),
        universe_idx=json.dumps(UNIVERSE_IDX),
        end_date=END_DATE, count=COUNT, week=WEEK,
    )
    stamp = datetime.now().strftime("%Y%m%d")
    out_path = os.path.join(OUT_DIR, f"jq_fast_ic_batch_{stamp}.py")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(code)
    return out_path, {"n_factors": len(ok_entries), "names": names, "skipped": skipped}


def build_batch_for_weekly_pending():
    """流水线钩子: 扫队列 weekly+pending_jq_run → 生成批量快验脚本。返回路径或 None"""
    queue = _load_queue()
    entries = select_entries(queue, include=None, all_weekly_pending=True)
    if not entries:
        return None
    out, meta = build_batch_script(entries)
    if out:
        for nm, formula, _m in entries:
            e = queue.get(nm)
            if isinstance(e, dict):
                e["fast_ic_batch_file"] = out
                e.setdefault("fast_ic_status", "fast_ic_queued")
        _save_queue(queue)
    return out


# ── 回灌解析 ────────────────────────────────────────────

_FAST_IC_RE = re.compile(
    r"^\[FAST_IC\]\s+(.+?)\s+ic=(-?\d+\.\d+|nan)\s+icir=(-?\d+\.\d+|nan)\s+"
    r"n=(\d+)\s+hint=(\S+)")


def _parse_log(text):
    out = []
    for line in text.splitlines():
        m = _FAST_IC_RE.match(line.strip())
        if m:
            name = m.group(1).strip()
            ic = float(m.group(2)) if m.group(2) != "nan" else None
            icir = float(m.group(3)) if m.group(3) != "nan" else None
            out.append({"name": name, "ic": ic, "icir": icir,
                        "n": int(m.group(4)), "hint": m.group(5)})
    return out


def _judge(rec):
    ic, icir, n = rec.get("ic"), rec.get("icir"), rec.get("n", 0)
    if ic is None or icir is None or n < FAST_IC_MIN_N:
        return False, "insufficient"
    if abs(icir) >= FAST_IC_ICIR_THRESHOLD and abs(ic) >= FAST_IC_MIN_ABS_IC:
        return True, "pass"
    return False, "below_threshold"


def ingest_fast_ic_results(log_text):
    """解析 [FAST_IC] 行 → 回写队列 fast_ic_result。返回 [(name, ok, detail)]"""
    recs = _parse_log(log_text)
    if not recs:
        return []
    queue = _load_queue()
    now = datetime.now().isoformat()
    report = []
    for rec in recs:
        name = rec["name"]
        entry = queue.get(name)
        if not isinstance(entry, dict):
            for k, v in queue.items():
                if k.startswith(name[:40]):
                    entry, name = v, k
                    break
        if not isinstance(entry, dict):
            report.append((rec["name"], "NOTFOUND", "队列无此因子"))
            continue
        passed, why = _judge(rec)
        entry["fast_ic_result"] = {
            "jq_ic": rec["ic"], "jq_icir": rec["icir"], "n": rec["n"],
            "hint": rec["hint"], "passed": passed, "reason": why,
            "gate_enforce": FAST_IC_GATE_ENFORCE, "ingested_at": now,
        }
        entry["fast_ic_status"] = "fast_ic_done"
        if FAST_IC_GATE_ENFORCE and not passed:
            entry["status"] = "fast_ic_blocked"
        # 与全回测结果对照 (若已有)
        jr = entry.get("jq_result") or {}
        cmp_note = ""
        if jr.get("jq_icir") is not None:
            cmp_note = (f" 全回测jq_icir={jr['jq_icir']} 快验jq_icir={rec['icir']} "
                        f"|Δ|={abs(jr['jq_icir'] - rec['icir']):.3f}")
        report.append((name, "PASS" if passed else "RECORD", why + cmp_note))
    _save_queue(queue)
    return report


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return
    cmd = args[0]
    if cmd == "--generate":
        include = []
        if "--include" in args:
            i = args.index("--include")
            include = [x.strip() for x in args[i + 1].split(",") if x.strip()]
        queue = _load_queue()
        entries = select_entries(queue, include=include,
                                 all_weekly_pending=("--no-pending-sweep" not in args))
        if not entries:
            print("[fast-ic] 无待快验条目 (weekly pending 或 --include 均空)")
            return
        out, meta = build_batch_script(entries)
        if not out:
            print("[fast-ic] 生成失败:", meta.get("skipped"))
            return
        for nm in [e[0] for e in entries]:
            e = queue.get(nm)
            if isinstance(e, dict):
                e["fast_ic_batch_file"] = out
                e.setdefault("fast_ic_status", "fast_ic_queued")
        _save_queue(queue)
        print(f"[fast-ic] 批量脚本: {out}")
        print(f"[fast-ic] 因子数: {meta['n_factors']} | 名单: {meta['names']}")
        if meta.get("skipped"):
            print("[fast-ic] 跳过:", meta["skipped"])
        print("[fast-ic] 队列状态已置 fast_ic_queued; "
              f"门限: ENFORCE={FAST_IC_GATE_ENFORCE} ICIR>={FAST_IC_ICIR_THRESHOLD} "
              f"N>={FAST_IC_MIN_N} (影子模式=只记录)")
    elif cmd == "--ingest":
        if len(args) < 2:
            print("用法: python gen_jq_fast_ic.py --ingest <logfile>")
            return
        log_path = args[1]
        with open(log_path, encoding="utf-8", errors="ignore") as f:
            text = f.read()
        report = ingest_fast_ic_results(text)
        print(f"[fast-ic] ingest 解析 {len(report)} 条 [FAST_IC]:")
        for name, ok, detail in report:
            print(f"  [{ok}] {name} — {detail}")
        print(f"[fast-ic] 门限: ENFORCE={FAST_IC_GATE_ENFORCE} "
              f"ICIR>={FAST_IC_ICIR_THRESHOLD} N>={FAST_IC_MIN_N}")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
