# -*- coding: utf-8 -*-
"""Smoke the ralph_loop hook path: trade_gate_shadow.evaluate(candidates, results).

构造与 ralph_loop Phase E 传入形态一致的 candidates/results stub, 走真实 evaluate(),
确认 (a) 不崩, (b) 落盘 data/trade_gate_shadow.json, (c) summary 字段齐备。
"""
import json
import os
import sys
from types import SimpleNamespace

FA = r"e:\quant\research\factor_alchemy"
sys.path.insert(0, FA)
os.environ["FUXI_TRADE_GATE"] = "shadow"

import trade_gate_shadow as tg  # noqa: E402

candidates = [
    {"name": "cand_forge_gen4_r1", "paradigm": "流动性×微观结构",
     "formula": "-(np.log(np.maximum(volume_p.rolling(10).sum(), 1e-8)).rolling(5).std().rolling(60).mean().rolling(10).rank(pct=True))"},
    {"name": "cand_fast_mom_2", "paradigm": "动量",
     "formula": "(close / close.shift(2) - 1.0)"},
    {"name": "cand_slow_rev_60", "paradigm": "反转",
     "formula": "-(close / close.shift(60) - 1.0)"},
    {"name": "cand_no_formula", "paradigm": "空"},        # 应被 _pick_formulas 跳过
    {"name": "cand_s1_failed", "paradigm": "动量",
     "formula": "(close / close.shift(5) - 1.0)"},         # s1_passed=False → 跳过
]
results = [
    SimpleNamespace(s1_passed=True, eligible_for_jq=False),
    SimpleNamespace(s1_passed=True, eligible_for_jq=True),
    SimpleNamespace(s1_passed=True, eligible_for_jq=False),
    SimpleNamespace(s1_passed=True, eligible_for_jq=False),
    SimpleNamespace(s1_passed=False, eligible_for_jq=False),
]

print("=" * 78)
print("trade_gate_shadow.evaluate() —— ralph_loop hook 路径冒烟")
print("=" * 78)
rep = tg.evaluate(candidates, results, verbose=True)
print()
print("--- 落盘产物 ---")
p = r"e:\quant\data\trade_gate_shadow.json"
print("exists:", os.path.exists(p))
if os.path.exists(p):
    b = open(p, "rb").read()
    print("CRLF=%d bare_LF=%d" % (b.count(b"\r\n"), b.count(b"\n") - b.count(b"\r\n")))
    j = json.loads(b.decode("utf-8"))
    print("mode            =", j.get("mode"))
    print("n_checked       =", j.get("n_checked"))
    print("n_fragile       =", j.get("n_fragile"))
    print("n_steep         =", j.get("n_steep"))
    print("n_high_drag     =", j.get("n_high_drag"))
    print("n_unreliable    =", j.get("n_unreliable"))
    print("median_turnover =", j.get("median_turnover"))
    print("median_drag_30  =", j.get("median_drag_ratio_30"))
    print("forbidden_use 存在:", bool(j.get("forbidden_use")))
    print("caliber         =", json.dumps(j.get("caliber"), ensure_ascii=False)[:300])

print()
print("--- ralph_loop summary['trade_gate'] 将写入的字段 ---")
summary_line = {
    "mode": rep.get("mode"),
    "n_checked": rep.get("n_checked", 0),
    "n_fragile": rep.get("n_fragile", 0),
    "n_steep": rep.get("n_steep", 0),
    "n_high_drag": rep.get("n_high_drag", 0),
    "n_unreliable": rep.get("n_unreliable", 0),
    "median_turnover": rep.get("median_turnover", 0.0),
    "median_drag_ratio_30": rep.get("median_drag_ratio_30"),
}
print(json.dumps(summary_line, ensure_ascii=False, indent=2))

# off 模式必须完全惰性
print()
print("--- off 模式惰性验证 ---")
os.environ["FUXI_TRADE_GATE"] = "off"
rep_off = tg.evaluate(candidates, results, verbose=False)
print("mode=%s n_checked=%d rows=%d"
      % (rep_off.get("mode"), rep_off.get("n_checked"), len(rep_off.get("rows", []))))
assert rep_off.get("n_checked") == 0 and len(rep_off.get("rows", [])) == 0, "off 模式不惰性!"
print("✅ off 模式零计算")
