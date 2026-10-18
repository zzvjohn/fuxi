# -*- coding: utf-8 -*-
"""
v0.9 测试: 种子公式翻译修复 + S5 通过自动生成 JQ 代码
========================================================
覆盖:
  T1  21 个 champion 公式翻译覆盖率 (20/21, north_money 明确拒绝)
  T2  翻译产物在模拟 S5 per-stock 上下文中可执行
  T3  JQ 重写正确性 (rolling rank 链 / min_periods / 遗留函数)
  T4  numpy rank helper 与 pandas Rolling.rank 语义等价 (bit-exact)
  T5  端到端: generate_standalone → 编译 → 离线执行因子函数
  T6  _prepare_seed_recheck_candidates 翻译状态落盘 + 失败原因记录
  T7  record_seed_jq_code 状态回写

运行: cd e:/quant && python scripts/test_v09_jq_autogen.py
"""

import os
import sys
import json
import warnings
import tempfile
import importlib.util

warnings.filterwarnings("ignore")

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(BASE, "research", "factor_alchemy"))
sys.path.insert(0, os.path.join(BASE, "scripts"))

import numpy as np
import pandas as pd
from pathlib import Path

from ralph_loop import RalphLoop
from gen_jq_generic import prepare_factor, generate_standalone

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print("  [%s] %s %s" % ("PASS" if cond else "FAIL", name, detail if not cond else ""))


# ────────────────────────────────────────────────
# 公共测试数据
# ────────────────────────────────────────────────
rng = np.random.RandomState(7)
N = 400
SYN = {}
for s in ["000001.XSHE", "000002.XSHE", "000003.XSHE"]:
    close = np.maximum(10 + rng.randn(N).cumsum() * 0.5, 1.0)
    SYN[s] = {
        "close_p": close,
        "open_p": close * (1 + rng.randn(N) * 0.01),
        "high_p": close * (1 + np.abs(rng.randn(N)) * 0.02),
        "low_p": close * (1 - np.abs(rng.randn(N)) * 0.02),
        "volume_p": np.abs(rng.randn(N)) * 1e6 + 1e5,
        "amount_p": close * np.abs(rng.randn(N)) * 1e6,
    }


def s5_context_for(stock_series):
    """模拟 s5_joint_filter._build_series_context + base_context。"""
    ctx = dict(stock_series)
    alias = {"close": "close_p", "open": "open_p", "high": "high_p",
             "low": "low_p", "volume": "volume_p", "amount": "amount_p"}
    for short, full in alias.items():
        if full in ctx:
            ctx[short] = ctx[full]
    ctx.update({
        "np": np, "pd": pd,
        "abs": np.abs, "sqrt": np.sqrt, "log": np.log,
        "log1p": np.log1p, "exp": np.exp, "sign": np.sign,
        "maximum": np.maximum, "minimum": np.minimum,
        "where": np.where, "clip": np.clip,
        "range": range, "len": len, "int": int, "float": float,
        "list": list, "dict": dict, "tuple": tuple,
    })
    return ctx


def load_champions():
    df = pd.read_csv(os.path.join(BASE, "data", "unified_factor_pool.csv"))
    ch = df[(df.status == "champion_seed") | (df.source == "alpha_agent_v3_champion")]
    return [(r["name"], str(r["formula"])) for _, r in ch.iterrows()]


# ────────────────────────────────────────────────
# T1: 翻译覆盖率
# ────────────────────────────────────────────────
print("T1: champion 公式翻译覆盖率")
ralph = RalphLoop.__new__(RalphLoop)
champions = load_champions()
ok_list, fail_list = [], []
for name, f in champions:
    t, reason = RalphLoop._translate_seed_formula(ralph, f)
    if t:
        ok_list.append((name, t))
    else:
        fail_list.append((name, reason))
check("T1a 翻译成功数 >= 20", len(ok_list) >= 20, "成功 %d" % len(ok_list))
check("T1b north_money 显式拒绝", any("north_money" in r for _, r in fail_list),
      str(fail_list))
check("T1c 所有失败都有明确原因", all(r for _, r in fail_list))

# ────────────────────────────────────────────────
# T2: 翻译产物在模拟 S5 上下文中可执行
# ────────────────────────────────────────────────
print("T2: 翻译产物 S5 per-stock 可执行性")
exec_ok = 0
for name, t in ok_list:
    ctx = s5_context_for({k: pd.Series(v) for k, v in SYN["000001.XSHE"].items()})
    try:
        val = eval(t, {"__builtins__": {}, **ctx}, {})
        if isinstance(val, (pd.Series, np.ndarray)):
            vv = val.values if isinstance(val, pd.Series) else val
            vv = vv[~np.isnan(vv)] if vv.dtype.kind == "f" else vv
            if len(vv) > 0:
                exec_ok += 1
                continue
        elif isinstance(val, (int, float)) and not np.isnan(val):
            exec_ok += 1
            continue
    except Exception as e:
        print("    [%s] exec 失败: %s" % (name, e))
check("T2 全部翻译产物可执行", exec_ok == len(ok_list), "%d/%d" % (exec_ok, len(ok_list)))

# ────────────────────────────────────────────────
# T3: JQ 重写正确性
# ────────────────────────────────────────────────
print("T3: JQ 老 pandas 兼容重写")
m1 = prepare_factor("t_rank", "(3.0 + volume_p).rolling(30).rank(pct=True).rolling(26).rank(pct=True)")
check("T3a 双 rank 链重写且 helper=True",
      m1["ok"] and m1["needs_helper"] and "_rolling_rank_pct_vec" in m1["formula_jq"])
check("T3b 双 rank 链中不含 .rank(pct=True)",
      ".rank(pct=True)" not in m1["formula_jq"])
m2 = prepare_factor("t_minper", "-(close_p - low_p).rolling(252, min_periods=60).rank(pct=True)")
check("T3c min_periods 传递", m2["ok"] and "min_periods=60" in m2["formula_jq"])
m3 = prepare_factor("t_legacy", "-((close / rolling_min(close, 60)) - close)")
check("T3d 遗留 rolling_min → 方法链", m3["ok"] and "(close).rolling(60).min()" in m3["formula_jq"])
m4 = prepare_factor("t_turnover", "close_p.rolling(5).mean() / (turnover_p + 1e-6)")
check("T3e turnover 明确拒绝", not m4["ok"] and "turnover_p" in m4["reason"])
m5 = prepare_factor("t_axis", "close_p.rolling(20).mean(axis=1)")
check("T3f axis=1 截面拒绝", not m5["ok"])

# ────────────────────────────────────────────────
# T4: rank helper 语义等价 (bit-exact)
# ────────────────────────────────────────────────
print("T4: numpy rank helper 与 pandas Rolling.rank 等价")
c = pd.Series(SYN["000001.XSHE"]["close_p"])
v = pd.Series(SYN["000001.XSHE"]["volume_p"])
h = pd.Series(SYN["000001.XSHE"]["high_p"])
l = pd.Series(SYN["000001.XSHE"]["low_p"])
base4 = -((h - l) / (v.rolling(20).mean() + 1e-6 + 1e-6))
ref4 = base4.rolling(252, min_periods=60).rank(pct=True).iloc[-1]
tmp = tempfile.mkdtemp(prefix="v09t4_")
out4, meta4 = generate_standalone(
    "t4_rank", "-((high_p - low_p) / ((volume_p.rolling(20).mean() + 1e-6) + 1e-6)).rolling(252, min_periods=60).rank(pct=True)",
    out_dir=tmp)
spec = importlib.util.spec_from_file_location("m4", out4)
mod4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod4)
pdata4 = {"close": {"s": SYN["000001.XSHE"]["close_p"]},
          "volume": {"s": SYN["000001.XSHE"]["volume_p"]},
          "high": {"s": SYN["000001.XSHE"]["high_p"]},
          "low": {"s": SYN["000001.XSHE"]["low_p"]}}
arr4, valid4 = mod4.FACTOR_FUNC_MAP["t4_rank"](["s"], pdata4)
check("T4a 单 rank bit-exact", valid4.any() and abs(arr4[0] - ref4) < 1e-12,
      "diff=%.2e" % abs(arr4[0] - ref4))
r1 = (3.0 + v).rolling(30).rank(pct=True)
ref5 = r1.rolling(26).rank(pct=True).iloc[-1]
out5, meta5 = generate_standalone("t4_drank", "(3.0 + volume_p).rolling(30).rank(pct=True).rolling(26).rank(pct=True)", out_dir=tmp)
spec5 = importlib.util.spec_from_file_location("m5", out5)
mod5 = importlib.util.module_from_spec(spec5)
spec5.loader.exec_module(mod5)
arr5, valid5 = mod5.FACTOR_FUNC_MAP["t4_drank"](["s"], {"close": pdata4["close"], "volume": pdata4["volume"]})
check("T4b 双 rank bit-exact", valid5.any() and abs(arr5[0] - ref5) < 1e-12,
      "diff=%.2e" % abs(arr5[0] - ref5))

# ────────────────────────────────────────────────
# T5: 端到端 — 全部可翻译 champion → JQ 文件可生成可执行
# ────────────────────────────────────────────────
print("T5: 端到端 champion → JQ 文件 → 离线执行")
tmp5 = tempfile.mkdtemp(prefix="v09t5_")
gen_ok = 0
for name, t in ok_list:
    try:
        out, meta = generate_standalone("t5_" + name, t, out_dir=tmp5)
        spec = importlib.util.spec_from_file_location(name, out)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        pdata = {
            "close": {s: SYN[s]["close_p"] for s in SYN},
            "open": {s: SYN[s]["open_p"] for s in SYN},
            "volume": {s: SYN[s]["volume_p"] for s in SYN},
            "high": {s: SYN[s]["high_p"] for s in SYN},
            "low": {s: SYN[s]["low_p"] for s in SYN},
        }
        arr, valid = mod.FACTOR_FUNC_MAP["t5_" + name](list(SYN.keys()), pdata)
        if valid.sum() >= 1:
            gen_ok += 1
        else:
            print("    [t5_%s] valid=0 (lookback=%s minlen=%s)" % (name, meta["lookback"], meta["minlen"]))
    except Exception as e:
        print("    [t5_%s] 生成/执行失败: %s" % (name, e))
check("T5 全部可翻译 champion 均能生成并产出有效因子值",
      gen_ok == len(ok_list), "%d/%d" % (gen_ok, len(ok_list)))

# ────────────────────────────────────────────────
# T6: 种子重检准备 — 翻译状态落盘 + 失败原因记录
# ────────────────────────────────────────────────
print("T6: _prepare_seed_recheck_candidates 状态落盘")
tmp6 = tempfile.mkdtemp(prefix="v09t6_")
state_file = os.path.join(tmp6, "seed_recheck_state.json")
ralph6 = RalphLoop.__new__(RalphLoop)
ralph6._seed_recheck_state_path = lambda: Path(state_file)
ralph6._seed_recheck_cooloff_days = 30
ralph6._max_seed_recheck = 5
seeds = [{
    "factor_name": name, "formula": f,
    "ic": 0.1, "icir": 0.5, "status": "champion_seed", "paradigm": "测试",
} for name, f in champions]
picked = ralph6._prepare_seed_recheck_candidates(seeds, max_seeds=5)
check("T6a 选出 5 个种子 (上限)", len(picked) == 5, str(len(picked)))
check("T6b 选中种子公式已翻译 (非 Forge 前缀)",
      all(not (c["formula"].strip().startswith(("ts_", "rank(", "neg(", "div("))) for c in picked))
st = json.load(open(state_file, encoding="utf-8"))
ok_cnt = sum(1 for v in st.values() if v.get("translate", {}).get("ok"))
fail_cnt = sum(1 for v in st.values() if not v.get("translate", {}).get("ok"))
north = [k for k, v in st.items() if "north_money" in v.get("translate", {}).get("reason", "")]
check("T6c 翻译状态落盘: 20 ok / 1 fail", ok_cnt == 20 and fail_cnt == 1,
      "ok=%d fail=%d" % (ok_cnt, fail_cnt))
check("T6d north_money 原因记录", len(north) == 1, str(north))

# ────────────────────────────────────────────────
# T7: record_seed_jq_code 状态回写
# ────────────────────────────────────────────────
print("T7: record_seed_jq_code 回写")
ralph7 = RalphLoop.__new__(RalphLoop)
state_file7 = os.path.join(tmp6, "seed_recheck_state2.json")
ralph7._seed_recheck_state_path = lambda: Path(state_file7)
ralph7.record_seed_jq_code("demo_seed", "jq_s5_pass_demo_standalone.py", True)
ralph7.record_seed_jq_code("bad_seed", "", False, "未知终端: north_money")
st7 = json.load(open(state_file7, encoding="utf-8"))
check("T7a 成功回写", st7["demo_seed"]["jq_code"]["generated"] is True
      and st7["demo_seed"]["jq_code"]["file"].endswith(".py"))
check("T7b 失败回写含原因", st7["bad_seed"]["jq_code"]["generated"] is False
      and "north_money" in st7["bad_seed"]["jq_code"]["reason"])

# ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  结果: %d PASS / %d FAIL" % (len(PASS), len(FAIL)))
if FAIL:
    print("  失败项:", FAIL)
print("=" * 60)
sys.exit(0 if not FAIL else 1)
