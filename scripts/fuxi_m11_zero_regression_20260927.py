# -*- coding: utf-8 -*-
"""M11 零回归验证 (2026-09-27): FUXI_S5_LEDGER 默认(proxy)下逐位不变。

方法 (A/B 反事实):
  进程 A = 当前 s5_joint_filter (含 M11 分流, 默认 proxy);
  进程 B = 副本, 分流分支替换为 ``if False`` (模拟改动前代码);
  同一因子 / 同一窗口分别跑 validate_factor, 对比:
    · _last_backtest_diag 全字段
    · 分年度指标 (excess/calmar/total_return/benchmark/max_dd/vol/n_periods)
    · portfolio_returns 逐位序列
  逐位一致 ⇒ 默认口径零回归 (新增分支不命中 + 新增字段 .get 默认 + error
  分支为新增 elif)。

用法 (两次进程, 内存隔离):
  python -X utf8 scripts/fuxi_m11_zero_regression_20260927.py --side A
  python -X utf8 scripts/fuxi_m11_zero_regression_20260927.py --side B
  python -X utf8 scripts/fuxi_m11_zero_regression_20260927.py --compare
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FA = ROOT / "research/factor_alchemy"
TMP = ROOT / "reports/_tmp"
SIDE_A = TMP / "m11_zero_reg_A.json"
SIDE_B = TMP / "m11_zero_reg_B.json"
PRE_COPY = FA / "_tmp_s5_pre_m11.py"

FACTOR_NAME = "m11_zero_regression_probe"
# docstring 示例公式 (3 个 pandas 操作, 通过 sanity; 有真实信号产出)
FORMULA = "(open_p/close_p.shift(1)-1).rolling(5).mean()"
TOP_N = 80

_JSON_DENY = {"portfolio_returns"}  # Series 单独序列化


def _build_pre_copy():
    """生成分流禁用副本 (模拟修改前); 只在 --side B 时调用。"""
    src = (FA / "s5_joint_filter.py").read_text(encoding="utf-8")
    marker = 'if _s5_ledger_mode() != "proxy":'
    assert src.count(marker) == 1, "dispatch marker not unique"
    patched = src.replace(
        marker, "if False:  # M11 zero-regression: pre-change simulation")
    PRE_COPY.write_text(patched, encoding="utf-8", newline="")


def _returns_list(result) -> list:
    r = getattr(result, "portfolio_returns", None)
    if r is None:
        return []
    if hasattr(r, "tolist"):
        return [float(v) for v in r.tolist()]
    return [float(v) for v in list(r)]


def _snapshot(result) -> dict:
    def _f(x):
        try:
            return float(x)
        except Exception:  # noqa: BLE001
            return None

    rets = _returns_list(result)
    return {
        "passed": bool(result.passed),
        "reason": str(result.reason),
        "error": result.error,
        "excess_2025": _f(result.excess_2025),
        "excess_2026": _f(result.excess_2026),
        "calmar_2025": _f(result.calmar_2025),
        "calmar_2026": _f(result.calmar_2026),
        "total_return_2025": _f(result.total_return_2025),
        "total_return_2026": _f(result.total_return_2026),
        "benchmark_return_2025": _f(result.benchmark_return_2025),
        "benchmark_return_2026": _f(result.benchmark_return_2026),
        "max_drawdown_2025": _f(result.max_drawdown_2025),
        "max_drawdown_2026": _f(result.max_drawdown_2026),
        "annual_vol_2025": _f(result.annual_vol_2025),
        "annual_vol_2026": _f(result.annual_vol_2026),
        "window_days_2025": result.window_days_2025,
        "window_days_2026": result.window_days_2026,
        "portfolio_returns_n": int(getattr(result, "portfolio_returns_n", 0) or 0),
        "returns": rets,
        "returns_sha": hashlib.sha256(
            json.dumps([round(v, 12) for v in rets]).encode("utf-8")).hexdigest(),
    }


def run_side(side: str):
    import os
    os.chdir(FA)
    sys.path.insert(0, str(FA))
    if side == "B":
        _build_pre_copy()
        import importlib
        mod = importlib.import_module("_tmp_s5_pre_m11")
    else:
        import importlib
        mod = importlib.import_module("s5_joint_filter")

    os.environ.setdefault("FUXI_S5_LEDGER", "proxy")
    s5 = mod.S5JointFilter(top_n=TOP_N)
    result = s5.validate_factor(FACTOR_NAME, FORMULA)
    snap = _snapshot(result)
    snap["side"] = side
    snap["ledger_mode_env"] = os.environ.get("FUXI_S5_LEDGER")
    snap["diag"] = {k: v for k, v in (s5._last_backtest_diag or {}).items()
                    if k not in _JSON_DENY}
    TMP.mkdir(parents=True, exist_ok=True)
    out = SIDE_A if side == "A" else SIDE_B
    out.write_text(json.dumps(snap, ensure_ascii=False, indent=1, default=str)
                   + "\n", encoding="utf-8", newline="\n")
    print(f"[{side}] passed={snap['passed']} excess25={snap['excess_2025']} "
          f"excess26={snap['excess_2026']} n_ret={snap['portfolio_returns_n']} "
          f"ret_sha={snap['returns_sha'][:16]} -> {out}")


def compare():
    a = json.loads(SIDE_A.read_text(encoding="utf-8"))
    b = json.loads(SIDE_B.read_text(encoding="utf-8"))
    fields = [k for k in a if k not in ("side", "diag")]
    diffs = []
    for k in fields:
        if a.get(k) != b.get(k):
            diffs.append((k, a.get(k), b.get(k)))
    da, db = a.get("diag") or {}, b.get("diag") or {}
    diag_only_a = sorted(set(da) - set(db))
    diag_only_b = sorted(set(db) - set(da))
    diag_diff = sorted(k for k in set(da) & set(db) if da[k] != db[k])
    payload = {
        "schema": "fuxi_m11_zero_regression/v1",
        "identical_fields": not diffs,
        "field_diffs": diffs,
        "returns_sha_equal": a["returns_sha"] == b["returns_sha"],
        "diag_only_in_A": diag_only_a,
        "diag_only_in_B": diag_only_b,
        "diag_value_diffs": diag_diff,
        "verdict": ("ZERO_REGRESSION_VERIFIED" if (not diffs and not diag_only_b
                    and not diag_diff) else "REGRESSION_DETECTED"),
    }
    if diag_only_a:
        payload["note_diag_only_in_A"] = (
            "允许: A 侧可能多出新披露字段 (M11 新增), 但值必须一致或仅新增键")
    print(json.dumps(payload, ensure_ascii=False, indent=1))
    return 0 if payload["verdict"] == "ZERO_REGRESSION_VERIFIED" else 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=["A", "B"])
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    if args.compare:
        raise SystemExit(compare())
    if not args.side:
        ap.error("need --side A|B or --compare")
    run_side(args.side)


if __name__ == "__main__":
    main()
