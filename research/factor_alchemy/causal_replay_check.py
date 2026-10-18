# -*- coding: utf-8 -*-
"""P-20260918-006: 运行时因果重放校验 (runtime causal replay)。

为什么需要它 (2026-09-18 外部审计 P0-2):
    现有 ``causal_closure_check.py`` 是**静态扫公式文本** (抓 shift(-N) /
    pct_change(-N) / 裸 cum* / expanding / bfill)。它抓不到**数据对齐层**的前视 ——
    例如「辅助通道同日对齐 (moneyflow T 日盘后) = 选股侧 1 日前视」
    (P-20260916-010): 那行公式本身**没有负向 shift**, 前视发生在 **join 的时点**上。

    **静态检查看不到 join 语义, 动态重算能看到。**

判据 (与外部项目 broad_market/frozen_strategy.py:73-75 同一机理):
    对任意求值函数 f, 取时点 t::

        f(panel[:t+1])[-1]   ==   f(panel[:T])[t]

    截断窗口重算的**末行**必须与全窗口在该时点的值逐位一致。不一致 ⇒ 该时点的
    取值依赖了 t 之后的信息 ⇒ **前视**。

模式 (FUXI_CAUSAL_REPLAY):
    shadow  (默认) : 只记录不拦截 —— 零裁决, 可安全挂进任何链路
    enforce        : verdict=mismatch 时返回 True 供调用方拦截
    off            : 完全跳过, 连报告都不写

报告: data/causal_replay_report.json

接线状态 (2026-09-18):
    本模块是**完整可用的校验框架** (含 CLI 与自检), 可对任意求值函数立即使用。
    把它自动挂进 ralph_loop QualityGate / S1 前置的接线**待提案审批**
    (ralph_loop.py 5000+ 行, 属五静默坑高发区, 须走翻默认四关)。

用法:
    python research/factor_alchemy/causal_replay_check.py --selftest
    python research/factor_alchemy/causal_replay_check.py --demo
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

MODE = os.environ.get("FUXI_CAUSAL_REPLAY", "shadow").lower()
assert MODE in ("shadow", "enforce", "off"), "FUXI_CAUSAL_REPLAY must be shadow|enforce|off"

_REPORT_PATH = Path(__file__).resolve().parents[2] / "data" / "causal_replay_report.json"
DEFAULT_SAMPLES = 20
DEFAULT_TOL = 1e-10


# ─────────────────────────── 核心 ───────────────────────────

def _at(x, i):
    """按**位置**取值 (pandas 2.x 的 ``series[i]`` 是标签索引 ⇒ 必须走 iloc)。"""
    try:
        return x.iloc[i]
    except AttributeError:
        return x[i]


def _val_of(x):
    """把 pandas/numpy/标量统一成 float; 无法转则返回 None。"""
    try:
        import math
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def replay_check(evaluate_fn, panel, n_samples=DEFAULT_SAMPLES, seed=0,
                 tol=DEFAULT_TOL, min_history=2):
    """截断重放校验。

    参数:
        evaluate_fn : callable(panel_slice) -> 与 panel 等长的序列 (末值 = 该时点值)
        panel       : 支持 len() 与切片 panel[:k] 的对象 (DataFrame/ndarray/list)
        n_samples   : 抽样时点数
        min_history : 采样下界 (避开历史不足导致两侧都 NaN 的假阳性)

    返回 dict:
        verdict      : ok | mismatch | insufficient
        n_compared   : **实际比较数量** (为 0 时 verdict=insufficient, 拒绝假 PASS)
        mismatches   : [{t, trunc, full, delta}]  (至多 20 条)
        n_verified   : 通过比较的时点数
    """
    T = len(panel)
    if T < max(3, min_history + 2):
        return {"verdict": "insufficient", "n_compared": 0, "n_verified": 0,
                "mismatches": [], "reason": "样本长度 %d 不足" % T}

    try:
        full = evaluate_fn(panel)
    except Exception as e:
        return {"verdict": "insufficient", "n_compared": 0, "n_verified": 0,
                "mismatches": [], "reason": "全窗口求值失败: %s: %s" % (type(e).__name__, e)}

    lo = min_history
    hi = T - 1                       # 须留出至少一个"未来"点才可能暴露前视
    if hi <= lo:
        return {"verdict": "insufficient", "n_compared": 0, "n_verified": 0,
                "mismatches": [], "reason": "可用时点区间为空"}

    # 确定性等距采样 + 随机补充 (固定 seed, 可复现)
    span = list(range(lo, hi + 1))
    if len(span) > n_samples:
        step = max(1, len(span) // n_samples)
        idxs = span[::step][:n_samples]
    else:
        idxs = span

    n_cmp, n_ok, mismatches = 0, 0, []
    for t in idxs:
        try:
            trunc = evaluate_fn(panel[:t + 1])
            a = _val_of(_at(trunc, -1))
        except Exception as e:
            mismatches.append({"t": int(t), "error": "%s: %s" % (type(e).__name__, e)})
            continue
        b = _val_of(_at(full, t))
        if a is None and b is None:
            continue                  # 两侧皆缺 ⇒ 无信息, 不计入比较数
        n_cmp += 1
        if a is None or b is None:
            mismatches.append({"t": int(t), "trunc": a, "full": b, "delta": None,
                               "reason": "单侧缺失 (截断=%s 全窗口=%s)" % (a, b)})
            continue
        d = abs(a - b)
        if d > tol:
            mismatches.append({"t": int(t), "trunc": a, "full": b, "delta": d})
        else:
            n_ok += 1

    # 🔴 断言实际比较数量 > 0 —— 否则"全部通过"是空集假 PASS
    if n_cmp == 0:
        return {"verdict": "insufficient", "n_compared": 0, "n_verified": 0,
                "mismatches": mismatches[:20], "reason": "实际比较数为 0 (样本全缺值)"}

    return {"verdict": "mismatch" if mismatches else "ok",
            "n_compared": n_cmp, "n_verified": n_ok,
            "mismatches": mismatches[:20],
            "max_delta": max([m["delta"] for m in mismatches if m.get("delta")] or [0.0])}


def should_block(verdict):
    """enforce 模式下 verdict=mismatch 才拦截; shadow 永不拦截。"""
    return MODE == "enforce" and verdict == "mismatch"


def record_report(entries):
    """批量落盘 shadow 报告。entries: [{factor, formula_preview, **replay_result}]"""
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
        recs = recs[-500:]
        rep["records"] = recs
        rep["summary"] = {
            "total_checked": rep.get("summary", {}).get("total_checked", 0) + len(entries),
            "total_mismatch": sum(1 for e in entries if e.get("verdict") == "mismatch"),
            "total_insufficient": sum(1 for e in entries if e.get("verdict") == "insufficient"),
            "last_run": now, "mode": MODE,
        }
        _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        _REPORT_PATH.write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                                encoding="utf-8")
    except Exception as e:
        print(f"  [CausalReplay] 报告落盘失败 (不阻塞): {e}")


# ─────────────────────────── 自检 / 演示 ───────────────────────────

def _make_panel(n=60):
    """合成价格面板 + 一条"盘后披露"的辅助通道。"""
    import pandas as pd
    close = pd.Series([10.0 * (1.01 ** i) for i in range(n)])
    aux = pd.Series([float(i % 7) for i in range(n)])
    return pd.DataFrame({"close": close, "aux": aux})


# ① 因果: 只用过去
def _fn_causal(df):
    return df["close"].shift(1).fillna(0.0)


# ② 前视: 显式读取未来 (静态检查能抓)
def _fn_future_shift(df):
    return df["close"].shift(-5)


# ③ 前视: join 时点 —— 公式内无任何负向 shift, 但用了**窗口末值**做填充
#    这正是 causal_closure_check.py 抓不到、而本模块能抓的类型
def _fn_join_lookahead(df):
    scale = float(df["aux"].iloc[-1])          # ← 用了"当前窗口最后一天"的盘后值
    return df["close"] * 0.0 + scale


# ④ 前视: 全样本 z-score (用整段均值/标准差归一)
def _fn_fullsample_z(df):
    return (df["close"] - df["close"].mean()) / (df["close"].std() or 1.0)


def selftest():
    """验证: 因果函数不误报, 三类前视函数**全部**能抓到。"""
    panel = _make_panel(60)
    ok = True
    cases = [("因果 shift(1)", _fn_causal, "ok"),
             ("前视 shift(-5)", _fn_future_shift, "mismatch"),
             ("前视 join 末值填充", _fn_join_lookahead, "mismatch"),
             ("前视 全样本 z-score", _fn_fullsample_z, "mismatch")]
    for name, fn, want in cases:
        r = replay_check(fn, panel, n_samples=20)
        good = (r["verdict"] == want) and (r["n_compared"] > 0)
        ok = ok and good
        print("  [%s] %-22s verdict=%-11s n_compared=%2d n_verified=%2d"
              % ("OK " if good else "FAIL", name, r["verdict"],
                 r["n_compared"], r["n_verified"]))
    # 空集保护: 全 NaN 必须判 insufficient, 不能判 ok
    import pandas as pd
    nan_fn = lambda df: pd.Series([float("nan")] * len(df))
    r2 = replay_check(nan_fn, panel, n_samples=5)
    empty_ok = (r2["verdict"] == "insufficient")
    ok = ok and empty_ok
    print("  [%s] %-22s verdict=%-11s (空集保护: 必须 insufficient)"
          % ("OK " if empty_ok else "FAIL", "全 NaN 求值", r2["verdict"]))
    print("\n[SELFTEST] %s" % ("全部通过" if ok else "**存在失败**"))
    return 0 if ok else 1


def demo():
    """打印一个 join 时点前视的判定细节 —— 静态检查**看不到**的那一类。"""
    panel = _make_panel(60)
    r = replay_check(_fn_join_lookahead, panel, n_samples=6, min_history=3)
    print("verdict=%s n_compared=%d n_verified=%d max_delta=%s"
          % (r["verdict"], r["n_compared"], r["n_verified"], r.get("max_delta")))
    for m in r["mismatches"][:4]:
        print("  t=%-3s trunc=%-8s full=%-8s delta=%s"
              % (m.get("t"), m.get("trunc"), m.get("full"), m.get("delta")))
    print("\n注: 该函数体内**没有任何 shift(-N)**, 静态扫描会判 ok。")
    print("    前视只存在于『用了窗口末值』这一 join 语义里。")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    a = ap.parse_args()
    print("[MODE] FUXI_CAUSAL_REPLAY=%s" % MODE)
    if a.selftest:
        return selftest()
    if a.demo:
        return demo()
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
