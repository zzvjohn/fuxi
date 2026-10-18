# -*- coding: utf-8 -*-
"""S5 第二套独立实现 (参考求值器) —— 纯标准库, **不 import 任何 S5 模块**。

动机 (2026-09-18 外部审计 P0-3):
    S5 只有**一套**实现。P-013 / P-016 / P-017 / P-021 一路踩的坑
    (基准不裁窗 / DATE_CEIL 语义 / 指纹漏窗口 / 同窗裁剪) **全都是静态检查抓不到
    的类型** —— 因为代码"看起来对", 错的是语义。
    唯一实用的手段是**第二套独立实现互算**: 同一订单序列、两条完全不同技术栈的
    求值路径, 任何口径偏差都会以数值差异暴露。

异构性设计 (对照价值的关键):
    S5 主实现  : pandas + numpy, 向量化, merge / groupby
    本参考实现 : **纯标准库** (csv + math), 逐日标量循环, 零第三方依赖
    ⇒ 两条路径不共享任何代码, 连浮点运算顺序都不同, 故能抓"实现型"偏差。

🔴 本实现锚定的是 S5 **实现** (s5_joint_filter.py::backtest_factor), 不是文档:
    ① 换手 = **双边腿比例** ∈[0,2]:
           (len(prev-sel) + len(sel-prev)) / top_n     [等权时 == Σ|Δw|]
       ⇒ 成本 = 双边腿比例 × bps/1e4。**不是**单边换手率 × bps。
       首期建仓 prev 为空 ⇒ 双边 = 买入腿 1.0 (不是 2.0)。
    ② 持有期日收益 = **当日存活标的的等权均值** (groupby(trade_date).mean())
       ⇒ 停牌/缺价标的的权重当日**重新分配**给存活标的, 不等价于"缺失记 0"。
       本文件用 --weight-mode 把这一差异**显式参数化**, 便于量化其影响。
    ③ 成本记在持有期**首个有效收益日**一次性扣除。
    ④ 收益按 close_p.pct_change() 语义 ⇒ 建仓日当日涨幅**不计入**。

用法:
    python scripts/s5_reference_eval.py --selftest
    python scripts/s5_reference_eval.py --orders o.csv --prices p.csv --out eq.csv
    python scripts/s5_reference_eval.py --orders o.csv --prices p.csv \
        --weight-mode fixed_w --compare data/s5_equity_golden.csv --tol 1e-10
退出码: 0=通过 / 1=比对超差或输入错误

── 接线状态 (2026-09-18) ──────────────────────────────────────────────
    本文件是**第二实现的完整实现**, 可独立自检、可独立跑任意 orders.csv。
    但 S5 侧目前**不导出 orders.csv** (backtest_factor 只返回
    (portfolio_returns, turnover_history), 不暴露持仓序列) ⇒ 让 S5 落订单
    需要改 s5_joint_filter.py 主体, 该接线**待提案审批**。
"""
import argparse
import csv
import io
import math
import sys

DEFAULT_BPS = 15.0          # 与 FUXI_S5_COST_BPS 默认一致
ANNUAL_DAYS = 252.0


# ─────────────────────────── 输入 ───────────────────────────

def read_orders(path):
    """orders.csv → [(rebalance_date, {code: weight}), ...], 按日期升序。

    同期权重和 > 1+1e-9 直接报错 (而不是静默归一) —— 静默归一正是"口径失真"。
    """
    periods, cur_date, cur = [], None, {}
    with io.open(path, "r", encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        need = {"rebalance_date", "ts_code", "weight"}
        if not need.issubset({(c or "").strip() for c in rd.fieldnames or []}):
            raise ValueError("orders.csv 表头须含 %s, 实际 %s" % (sorted(need), rd.fieldnames))
        for r in rd:
            d = (r["rebalance_date"] or "").strip()
            c = (r["ts_code"] or "").strip()
            w = float(r["weight"] or 0)
            if not d or not c:
                continue
            if cur_date is None:
                cur_date = d
            if d != cur_date:
                _check_sum(cur_date, cur)
                periods.append((cur_date, cur))
                cur_date, cur = d, {}
            cur[c] = cur.get(c, 0.0) + w
    if cur_date is not None:
        _check_sum(cur_date, cur)
        periods.append((cur_date, cur))
    periods.sort(key=lambda x: x[0])
    return periods


def _check_sum(date, wmap):
    s = sum(wmap.values())
    if s > 1.0 + 1e-9:
        raise ValueError("orders 权重和 %.10f > 1 @ %s ⇒ 疑似杠杆/口径错误 (拒绝静默归一)"
                         % (s, date))


def read_prices(path):
    """prices.csv → ({code: {date: close}}, [date, ...] 升序)。"""
    px, dates, seen = {}, [], set()
    with io.open(path, "r", encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            d = (r["trade_date"] or "").strip()
            c = (r["ts_code"] or "").strip()
            try:
                v = float(r["close"])
            except (TypeError, ValueError):
                continue
            if not d or not c or not (v > 0):
                continue
            px.setdefault(c, {})[d] = v
            if d not in seen:
                seen.add(d)
                dates.append(d)
    dates.sort()
    return px, dates


# ─────────────────────────── 口径原语 ───────────────────────────

def turnover_bilateral(w_new, w_old):
    """双边腿比例 ∈[0,2] —— 与 S5 的 (卖出腿+买入腿)/top_n 同义。

    满仓→空仓 = 1.0 ; 全部换股 = 2.0 ; 持仓不变 = 0.0 ; 首期建仓(空→满) = 1.0
    """
    s = 0.0
    for c, v in w_new.items():
        s += abs(v - w_old.get(c, 0.0))
    for c, v in w_old.items():
        if c not in w_new:
            s += abs(v)
    return s


def cost_linear(turn, bps):
    """线性近似: cost = 双边腿比例 × bps/1e4 —— S5 当前口径。"""
    return (bps / 1e4) * turn


def cost_fixedpoint(gross, weights, before, bps, iters=12):
    """自融资费用不动点: 费用按**实际成交额**收, 而成交额又取决于扣费后的净值。

        net = gross - fee · Σ|net·w - before|

    🔴 与线性口径的**方向**: 线性以 gross 为成交额基数; 不动点以**已扣费的 net**
    为基数。gross > net ⇒ 线性口径的税额是"用更大的基数算的"? 否 —— 关键在于
    线性直接忽略净值反馈, 实测 (T4) 线性 **低估**成本 ⇒ 净值偏高。
    大换手 + 满仓时偏差最大 (2026-09-18 外部审计 P1-5)。
    """
    fee = bps / 1e4
    net = gross
    for _ in range(iters):
        to = 0.0
        for c, w in weights.items():
            to += abs(net * w - before.get(c, 0.0))
        for c, v in before.items():
            if c not in weights:
                to += abs(v)
        net = gross - fee * to
    return net


def _period_cost(gross, w, before, bps, cost_mode):
    """按口径算本期成本。返回 (net_after_cost, 双边腿比例)。"""
    turn = turnover_bilateral(w, before)
    if bps <= 0:
        return gross, turn
    if cost_mode == "fixedpoint":
        return cost_fixedpoint(gross, w, before, bps), turn
    return gross - cost_linear(turn, bps), turn


# ─────────────────────────── 求值 ───────────────────────────

def simulate(orders, px, dates, bps=DEFAULT_BPS, cost_mode="linear",
             weight_mode="equal_alive"):
    """订单序列 + 价格 → (equity, daily_ret, turnovers)。

    weight_mode:
        equal_alive(S5 口径) : 当日存活标的等权均值 —— groupby(date).mean() 语义
        fixed_w              : 按订单权重, 缺价标的贡献 0 (等价现金)
    """
    if len(orders) < 2:
        return [], [], []
    idx = {d: i for i, d in enumerate(dates)}
    nav, out, rets, turns = 1.0, [], [], []
    prev_w = {}

    for k in range(len(orders) - 1):
        e_date, w = orders[k]
        x_date, _ = orders[k + 1]
        if e_date not in idx or x_date not in idx:
            raise ValueError("调仓日不在价格日历内: %s / %s" % (e_date, x_date))
        i0, i1 = idx[e_date], idx[x_date]
        cost_applied = False

        for j in range(i0 + 1, i1 + 1):
            d, dp = dates[j], dates[j - 1]
            pairs = []
            for c in w:
                p1, p0 = px.get(c, {}).get(d), px.get(c, {}).get(dp)
                if p1 is None or p0 is None or p0 <= 0:
                    continue                      # 缺价: 跳过, **不 fillna(0) 伪装**
                pairs.append((c, p1 / p0 - 1.0))
            if not pairs:
                out.append((d, nav))
                rets.append((d, 0.0))
                continue

            if weight_mode == "fixed_w":
                gross = 1.0 + sum(w[c] * r for c, r in pairs)
            else:                                  # equal_alive (S5 口径)
                gross = 1.0 + sum(r for _c, r in pairs) / len(pairs)

            if (not cost_applied) and bps > 0:
                gross, _t = _period_cost(gross, w, prev_w, bps, cost_mode)
                turns.append((d, _t))
                cost_applied = True

            nav *= gross
            rets.append((d, gross - 1.0))
            out.append((d, nav))
        prev_w = dict(w)
    return out, rets, turns


def metrics(rets):
    """年化 252 交易日; 无风险利率 = 0 (与 S5 一致, 显式声明)。"""
    if not rets:
        return {"days": 0, "total_return": 0.0, "annual_return": 0.0,
                "sharpe": 0.0, "max_drawdown": 0.0}
    r = [x for _d, x in rets]
    nav, peak, mdd = 1.0, 1.0, 0.0
    for x in r:
        nav *= (1.0 + x)
        peak = max(peak, nav)
        if peak > 0:
            mdd = max(mdd, 1.0 - nav / peak)
    n = len(r)
    ann = (nav ** (ANNUAL_DAYS / n) - 1.0) if nav > 0 else -1.0
    mu = sum(r) / n
    var = sum((x - mu) ** 2 for x in r) / (n - 1) if n > 1 else 0.0
    sd = math.sqrt(var)
    sharpe = (mu / sd) * math.sqrt(ANNUAL_DAYS) if sd > 0 else 0.0
    return {"days": n, "total_return": nav - 1.0, "annual_return": ann,
            "sharpe": sharpe, "max_drawdown": mdd}


# ─────────────────────────── 自检 ───────────────────────────

def selftest():
    """合成数据自检 —— 验证**记账口径本身**, 不依赖任何真实数据。

      T1 单标的满仓无成本 ⇒ 净值 == 价格比
      T2 建仓当日涨幅不计入 (时序正确性)
      T3 双边腿比例定义: 满仓→空仓=1.0 / 全换股=2.0 / 首期建仓=1.0
      T4 线性 vs 不动点: 大换手时线性**低估**成本 ⇒ 其净值偏高
      T5 equal_alive 与 fixed_w 在"有缺价标的"时**必须分叉** (差异可见)
    """
    ok = True

    def chk(name, got, want, tol=1e-12):
        nonlocal ok
        good = abs(got - want) <= tol
        ok = ok and good
        print("  [%s] %-44s got=%.12f want=%.12f" % ("OK " if good else "FAIL", name, got, want))

    # T1/T2: 3 天满仓单标的, 价格 10 → 11 → 12.1
    px = {"A": {"d0": 10.0, "d1": 11.0, "d2": 12.1}}
    dates = ["d0", "d1", "d2"]
    orders = [("d0", {"A": 1.0}), ("d1", {"A": 1.0}), ("d2", {"A": 1.0})]
    _e, rets, _t = simulate(orders, px, dates, bps=0.0)
    m = metrics(rets)
    chk("T1 满仓无成本 总收益 == 12.1/10-1", m["total_return"], 12.1 / 10.0 - 1.0)
    chk("T2 首日收益 == d1/d0-1 (建仓日不计)", rets[0][1], 11.0 / 10.0 - 1.0)

    # T3: 双边腿比例
    chk("T3 满仓→空仓", turnover_bilateral({}, {"A": 1.0}), 1.0)
    chk("T3 全换股", turnover_bilateral({"B": 1.0}, {"A": 1.0}), 2.0)
    chk("T3 首期建仓(空→满)", turnover_bilateral({"A": 1.0}, {}), 1.0)
    chk("T3 持仓不变", turnover_bilateral({"A": 1.0}, {"A": 1.0}), 0.0)

    # T4: 线性 vs 不动点 (全换股, gross=1.05, bps=15)
    fx = cost_fixedpoint(1.05, {"B": 1.0}, {"A": 1.0}, 15.0)
    lin = 1.05 - cost_linear(2.0, 15.0)
    print("  [CHK] T4 全换股 gross=1.05 bps=15: linear_nav=%.10f fixedpoint_nav=%.10f"
          % (lin, fx))
    if fx < lin - 1e-12:
        print("  [OK ] T4 不动点净值更低 ⇒ 线性低估成本 (方向与 P1-5 一致)")
    else:
        ok = False
        print("  [FAIL] T4 期望 fixedpoint_nav < linear_nav")
    chk("T4 零换手 ⇒ 成本为 0", cost_linear(0.0, 15.0), 0.0)

    # T5: 缺价标的 ⇒ equal_alive 与 fixed_w 必须分叉
    px2 = {"A": {"d0": 10.0, "d1": 11.0, "d2": 12.1},
           "B": {"d0": 10.0, "d2": 10.0}}            # B 在 d1 缺价
    o2 = [("d0", {"A": 0.5, "B": 0.5}), ("d2", {"A": 0.5, "B": 0.5})]
    _e1, r1, _ = simulate(o2, px2, dates, bps=0.0, weight_mode="equal_alive")
    _e2, r2, _ = simulate(o2, px2, dates, bps=0.0, weight_mode="fixed_w")
    d = abs(r1[0][1] - r2[0][1])
    print("  [CHK] T5 d1 收益: equal_alive=%.8f fixed_w=%.8f Δ=%.8f" % (r1[0][1], r2[0][1], d))
    if d > 1e-6:
        print("  [OK ] T5 缺价时两定权口径分叉可见 ⇒ 该差异不会被静默吞掉")
    else:
        ok = False
        print("  [FAIL] T5 期望两口径分叉")

    print("\n[SELFTEST] %s" % ("全部通过" if ok else "**存在失败**"))
    return 0 if ok else 1


# ─────────────────────────── 比对 ───────────────────────────

def compare(golden_path, nav):
    """与主实现导出的净值比对 —— 断言比较条数 > 0 (拒绝"无差异"的假 PASS)。"""
    g = {}
    with io.open(golden_path, "r", encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        cols = [c for c in ("nav", "equity", "value") if c in (rd.fieldnames or [])]
        if not cols:
            raise ValueError("golden 须含 nav/equity/value 列, 实际 %s" % rd.fieldnames)
        col = cols[0]
        dc = "trade_date" if "trade_date" in rd.fieldnames else rd.fieldnames[0]
        for r in rd:
            g[(r[dc] or "").strip()] = float(r[col])
    n, worst, worst_d = 0, 0.0, None
    for d, v in nav:
        if d not in g:
            continue
        n += 1
        dl = abs(v - g[d])
        if dl > worst:
            worst, worst_d = dl, d
    if n == 0:
        raise AssertionError("比对条数为 0 ⇒ 日期轴不匹配, 结论无效 (拒绝假 PASS)")
    return n, worst, worst_d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="合成数据自检 (不需真实数据)")
    ap.add_argument("--orders")
    ap.add_argument("--prices")
    ap.add_argument("--out", help="净值落盘 csv (trade_date,nav)")
    ap.add_argument("--bps", type=float, default=DEFAULT_BPS)
    ap.add_argument("--cost-mode", choices=["linear", "fixedpoint"], default="linear")
    ap.add_argument("--weight-mode", choices=["equal_alive", "fixed_w"],
                    default="equal_alive", help="equal_alive=S5 口径; fixed_w=按订单权重")
    ap.add_argument("--compare", help="与主实现净值 csv 比对")
    ap.add_argument("--tol", type=float, default=1e-10)
    a = ap.parse_args()

    if a.selftest:
        return selftest()
    if not (a.orders and a.prices):
        ap.error("需要 --orders 与 --prices (或用 --selftest)")

    orders = read_orders(a.orders)
    px, dates = read_prices(a.prices)
    print("[REF] 期数=%d 交易日=%d 标的=%d bps=%g cost=%s weight=%s"
          % (len(orders), len(dates), len(px), a.bps, a.cost_mode, a.weight_mode))
    nav, rets, turns = simulate(orders, px, dates, bps=a.bps,
                                cost_mode=a.cost_mode, weight_mode=a.weight_mode)
    m = metrics(rets)
    print("[REF] days=%d total=%.6f annual=%.6f sharpe=%.4f mdd=%.4f avg_turn_bilateral=%.6f"
          % (m["days"], m["total_return"], m["annual_return"], m["sharpe"],
             m["max_drawdown"], (sum(t for _d, t in turns) / len(turns)) if turns else 0.0))

    if a.out:
        with io.open(a.out, "w", encoding="utf-8", newline="") as f:
            f.write("trade_date,nav\r\n")
            for d, v in nav:
                f.write("%s,%.12f\r\n" % (d, v))
        print("[REF] 净值落盘 %s" % a.out)

    if a.compare:
        try:
            n, worst, wd = compare(a.compare, nav)
        except Exception as e:
            print("[COMPARE] 失败: %s: %s" % (type(e).__name__, e))
            return 1
        verdict = "PASS" if worst < a.tol else "FAIL"
        print("[COMPARE] 比对 %d 条 | max|Δ|=%.3e (tol=%.1e) @ %s ⇒ %s"
              % (n, worst, a.tol, wd, verdict))
        return 0 if worst < a.tol else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
