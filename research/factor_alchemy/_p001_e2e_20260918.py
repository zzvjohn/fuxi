# -*- coding: utf-8 -*-
"""P-20260918-001 端到端对账: 三路席位通道修复前后, 死消费者是否活过来。

死消费者 = NOVEL_FACTOR_LIBRARY 的 novel_234 / 235 / 236 (P-20260917-022 三路资金因子),
它们在 factor_ic_computer 路径下引用 lhb_yz_net_buy —— 修复前该列不存在 ⇒ NameError
⇒ eval 返回 None/空 ⇒ 静默丢候选。

本脚本同时检查 memory 铁律「接线修好 ⇒ 死消费者首跑即崩」(None→真值 ⇒ `if arr:` 二义)。
"""
import os
import sys

FA = r"e:\quant\research\factor_alchemy"
sys.path.insert(0, FA)
sys.path.insert(0, r"e:\quant\scripts")

MODE = sys.argv[1] if len(sys.argv) > 1 else "on"
os.environ["FUXI_LHB_TRI_CHANNEL"] = MODE
os.environ.setdefault("FUXI_TRADE_GATE", "off")   # 本测试不跑成本影子

print("=" * 78)
print("MODE =", MODE, "| FUXI_LHB_TRI_CHANNEL =", os.environ["FUXI_LHB_TRI_CHANNEL"])
print("=" * 78)

from factor_ic_computer import FactorICComputer  # noqa: E402

comp = FactorICComputer(lookback_days=500)
comp._load_data()

# ── 1) 面板是否真的有 lhb_yz_* 列 ──
cols = set(comp._price_df.columns) if hasattr(comp, "_price_df") else set()
trip = ["lhb_inst_net_buy", "lhb_north_net_buy", "lhb_yz_net_buy",
        "lhb_yz_buy", "lhb_yz_sell"]
print()
print("--- 面板列存在性 ---")
for c in trip:
    print("  %-22s %s" % (c, c in cols))
yz = comp._price_df.get("lhb_yz_net_buy")
if yz is not None:
    nz = int((yz.fillna(0) != 0).sum())
    print("  lhb_yz_net_buy 非零行 =", nz, "/", len(yz))

# ── 2) 死消费者端到端 eval ──
F = {
    "novel_234": """_n = lhb_north_net_buy.fillna(0.0)
_i = lhb_inst_net_buy.fillna(0.0)
_y = lhb_yz_net_buy.fillna(0.0)
_res = ((_n > 0) & (_i > 0) & (_y > 0)).astype(float)
factor = ((_n + _i + _y) * _res).rolling(5).sum().rank(pct=True, axis=1).fillna(0.5)""",
    "novel_235": """_both = ((lhb_north_net_buy.fillna(0.0) > 0) & (lhb_inst_net_buy.fillna(0.0) > 0) & (lhb_yz_net_buy.fillna(0.0) > 0)).astype(float)
factor = _both.rolling(20).sum().rank(pct=True, axis=1).fillna(0.5)""",
    "novel_236": """_n = lhb_north_net_buy.fillna(0.0)
_i = lhb_inst_net_buy.fillna(0.0)
_y = lhb_yz_net_buy.fillna(0.0)
_dir = ((_n > 0) & (_i > 0) & (_y > 0)).astype(float) - ((_n < 0) & (_i < 0) & (_y < 0)).astype(float)
factor = _dir.rolling(10).sum().rank(pct=True, axis=1).fillna(0.5)""",
    # 对照: 只用 inst+north (修复前就可求值) ⇒ 两模式都应有读数
    "ctrl_inst_north": """_n = lhb_north_net_buy.fillna(0.0)
_i = lhb_inst_net_buy.fillna(0.0)
factor = (_n + _i).rolling(5).sum().rank(pct=True, axis=1).fillna(0.5)""",
    # 对照: 只用 yz (修复前必失败)
    "yz_only": """factor = lhb_yz_net_buy.fillna(0.0).rolling(5).sum().rank(pct=True, axis=1).fillna(0.5)""",
}

print()
print("--- 端到端 eval 结果 (行数 / 因子值非空 / IC) ---")
res = {}
for nm, f in F.items():
    try:
        df = comp._eval_formula(f, nm)
    except Exception as e:
        df = None
        print("  %-18s EXCEPTION %s: %s" % (nm, type(e).__name__, e))
        res[nm] = {"ok": False, "err": f"{type(e).__name__}: {e}"}
        continue
    if df is None or len(df) == 0:
        print("  %-18s -> None/空 (静默丢弃)" % nm)
        res[nm] = {"ok": False, "rows": 0}
        continue
    fv = df["factor_value"] if "factor_value" in df.columns else None
    n_all = len(df)
    n_nn = int(fv.notna().sum()) if fv is not None else -1
    nuniq = int(fv.nunique()) if fv is not None else -1
    ic = icir = None
    try:
        st = comp.compute_ic(df, name=nm)
        if isinstance(st, dict):
            ic = st.get("ic_mean") or st.get("ic")
            icir = st.get("icir")
    except Exception:
        pass
    print("  %-18s rows=%-8d nonNaN=%-8d uniq=%-6d ic=%s icir=%s"
          % (nm, n_all, n_nn, nuniq,
             ("%.4f" % ic) if isinstance(ic, (int, float)) else "-",
             ("%.4f" % icir) if isinstance(icir, (int, float)) else "-"))
    res[nm] = {"ok": True, "rows": n_all, "nonnan": n_nn, "uniq": nuniq,
               "ic": ic, "icir": icir}

import json
print()
print("JSON_RESULT " + json.dumps(res, ensure_ascii=False, default=str))
