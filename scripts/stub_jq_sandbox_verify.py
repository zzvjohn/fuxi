# -*- coding: utf-8 -*-
r"""模拟 JQ 沙箱环境 (sum 被 np.sum 遮蔽) 跑完整调仓链路 — 通用版

验证 P-20260901-012 修复: _sum_vals() 手动累加免疫 np.sum(dict_values) 陷阱

用法:
  python scripts/stub_jq_sandbox_verify.py [策略文件路径] [周数]
默认: research/factor_alchemy/jq_combo_smallcap_duo_5050_standalone.py, 30 周

必须在带 numpy 的 venv 下运行 (如 envs/default, numpy 2.4.6):
  envs/default/Scripts/python.exe scripts/stub_jq_sandbox_verify.py <file> <weeks>
"""
import sys, types
import numpy as np
import pandas as pd

SRC = sys.argv[1] if len(sys.argv) > 1 else \
    "research/factor_alchemy/jq_combo_smallcap_duo_5050_standalone.py"
N_WEEKS = int(sys.argv[2]) if len(sys.argv) > 2 else 30

# ── JQ API stub ──
class G:
    pass
g = G()

class Log:
    def info(self, msg, *a):
        print("[LOG] " + (msg % a if a else msg))
    def set_level(self, *a, **k):
        pass
    def error(self, msg, *a):
        print("[ERR] " + (msg % a if a else msg))
    def warn(self, msg, *a):
        print("[WRN] " + (msg % a if a else msg))
log = Log()

STOCKS = ["00000%d.XSHE" % i for i in range(1, 10)] + \
         ["30000%d.XSHE" % i for i in range(10, 30)] + \
         ["6000%02d.XSHG" % i for i in range(30, 60)]
rng = np.random.default_rng(42)
_dates = pd.date_range("2021-01-04", periods=420, freq="B")
PRICES = {}
for s in STOCKS:
    PRICES[s] = 10.0 * np.exp(np.cumsum(rng.normal(0, 0.02, len(_dates))))


class CD:
    _i = 0
    def __getitem__(self, s):
        px = PRICES[s]
        i = min(CD._i, len(_dates) - 1)
        return types.SimpleNamespace(
            paused=False, is_st=False,
            high_limit=px[i] * 1.11, last_price=px[i], day_open=px[i])


class Pos:
    def __init__(self, amt, val):
        self.total_amount = amt
        self.value = val


class Portfolio:
    def __init__(self):
        self.positions = {}
        self.available_cash = 1e6
        self.total_value = 1e6


class Ctx:
    def __init__(self):
        self.portfolio = Portfolio()
        self.previous_date = None


CTX = Ctx()
ORDERS = {"n": 0, "fail": 0}


def _get_price(stocks, count=None, end_date=None, frequency=None,
               fields=None, skip_paused=None, fq=None, panel=None):
    idx = min(CD._i, len(_dates) - 1)
    start = max(0, idx - count + 1)
    dts = _dates[start:idx + 1]
    out = {}
    for f in fields:
        if f == "close":
            out[f] = pd.DataFrame({s: PRICES[s][start:idx + 1] for s in stocks}, index=dts)
        elif f == "volume":
            out[f] = pd.DataFrame(
                {s: np.abs(rng.normal(1e6, 1e5, len(dts))) for s in stocks}, index=dts)
        elif f == "money":
            out[f] = pd.DataFrame(
                {s: np.abs(rng.normal(1e7, 1e6, len(dts))) for s in stocks}, index=dts)
    return out


def _order_target_value(s, w):
    px = PRICES[s][min(CD._i, len(_dates) - 1)]
    amt = int(w / px / 100) * 100
    if amt <= 0:
        ORDERS["fail"] += 1
        return None
    CTX.portfolio.positions[s] = Pos(amt, w)
    CTX.portfolio.available_cash -= w
    ORDERS["n"] += 1
    return amt


def _order_target(s, n):
    if n == 0 and s in CTX.portfolio.positions:
        del CTX.portfolio.positions[s]
        ORDERS["n"] += 1


def _run_weekly(func, day, time=None, reference_security=None):
    RUN["weekly"] = func


RUN = {}
NS = {
    "g": g, "log": log, "np": np, "pd": pd,
    "set_benchmark": lambda *a, **k: None,
    "set_option": lambda *a, **k: None,
    "set_order_cost": lambda *a, **k: None,
    "set_slippage": lambda *a, **k: None,
    "run_weekly": _run_weekly,
    "get_all_securities": lambda t, d: pd.DataFrame(index=STOCKS),
    "get_current_data": lambda: CD(),
    "get_price": _get_price,
    "order_target_value": _order_target_value,
    "order_target": _order_target,
    "OrderCost": lambda **k: None,
    "FixedSlippage": lambda x: None,
    # 模拟 JQ 沙箱: 把内建 sum 遮蔽为 numpy.sum
    "sum": np.sum,
}

src = open(SRC, encoding="utf-8").read()
# 强制 scipy 走降级路径 (stub 环境无 scipy)
src = src.replace("from scipy.stats import", "from _no_such_module import")
code = compile(src, SRC, "exec")
exec(code, NS)

print("\n=== 环境自检 ===")
print("被测文件           :", SRC)
print("NS['sum'] is np.sum:", NS["sum"] is np.sum)
_sv = NS.get("_sum_vals")
print("_sum_vals 免疫     :", _sv({1: 0.5, 2: 0.5}.values()) if _sv else "N/A (该文件未用 sum)")

NS["initialize"](CTX)
rb = RUN["weekly"]

print("\n=== %d 周调仓模拟 ===" % N_WEEKS)
for wk in range(N_WEEKS):
    CD._i = 60 + wk * 5
    CTX.previous_date = _dates[CD._i]
    CTX.portfolio.total_value = 1e6
    CTX.portfolio.available_cash = CTX.portfolio.total_value * 0.6
    try:
        rb(CTX)
    except Exception as e:
        print("[FATAL] week %d 崩溃: %r" % (wk, e))
        import traceback
        traceback.print_exc()
        sys.exit(1)

print("\n=== 结果 ===")
print("累计下单次数 :", ORDERS["n"], " 拒单:", ORDERS["fail"])
print("当前持仓数   :", len(CTX.portfolio.positions))

_grp = getattr(g, "last_groups", None)
_w = getattr(g, "last_weights", None)
if _grp is not None:
    for k in ("A", "B"):
        wm = _grp.get(k) or {}
        print("last_groups[%s] : n=%d  sum=%.6f" % (k, len(wm), NS["_sum_vals"](wm.values())))
elif _w is not None:
    print("last_weights : n=%d  sum=%.6f" % (len(_w), NS["_sum_vals"](_w.values())))

_wr = getattr(g, "weekly_rets", None)
if isinstance(_wr, dict):
    for k in sorted(_wr):
        print("周收益[%s] 样本 : %d" % (k, len(_wr[k])))
else:
    print("周收益样本数 :", len(_wr or []))

_sk = getattr(g, "skip_stat", None)
if _sk:
    print("跳过原因统计 :", dict(sorted(_sk.items(), key=lambda kv: -kv[1])))

print("\nSTUB VERIFY PASSED (sum 被 np.sum 遮蔽环境下全链路无崩溃)")
