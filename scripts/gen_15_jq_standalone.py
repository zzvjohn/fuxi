# -*- coding: utf-8 -*-
"""为 15 个 E0~E4 全 pass 因子批量生成 standalone JQ 回测脚本。

每个脚本内嵌一个通用伏羲因子 AST 求值器 (eval_ast)，运行期解释 canonical_ast，
无需手工逐因子翻译 numpy。15 个文件共享同一骨架，只差 FACTOR_NAME / FACTOR_AST / 表达式。
"""
import json
import glob
import os

B = "e:/quant/data/fuxi_unified/runs/unified_779fe46ab39b01018533172f/research/campaign_779fe46ab39b01018533172f/bundles"
OUT = "e:/quant/output/jq_standalone_15"


def render(node):
    if not isinstance(node, dict):
        return str(node)
    if "field" in node:
        return node["field"]
    op = node.get("op")
    args = node.get("args") or []
    if op == "neg":
        return "-(" + render(args[0]) + ")"
    if op == "rank":
        return "rank(" + render(args[0]) + ")"
    if op in ("std", "mean", "sum", "min", "max", "delay", "delta", "return"):
        w = node.get("window")
        inner = render(args[0]) if args else ""
        return op + "(" + inner + (("," + str(w)) if w is not None else "") + ")"
    if op in ("add", "sub", "mul", "div", "pow"):
        return "(" + render(args[0]) + " " + {"add": "+", "sub": "-", "mul": "*", "div": "/", "pow": "^"}[op] + " " + render(args[1]) + ")"
    if op in ("log", "abs", "sign", "sqrt"):
        return op + "(" + render(args[0]) + ")"
    return op + "(" + ",".join(render(a) for a in args) + ")"


def has_volume(ast):
    s = json.dumps(ast)
    return "volume" in s


TEMPLATE = r'''# -*- coding: utf-8 -*-
"""
伏羲 v2.0 因子 standalone 组合收益回测 (聚宽) —— 自动生成
============================================================================
因子   : @FACTOR_NAME@
表达式 : @FACTOR_EXPR@
Rank IC: @IC@   ICIR: @ICIR@
来源   : S4 完整 campaign, E0~E4 全 valid+pass (research 级, 已获 JQ candidate 资格)
口径   : 全市场(0/3/6 开头, 剔 ST/停牌/涨停/次新) / top-30 等权 / 周频 /
         真实下单(佣金0.01%+印花税0.1%+滑点0.3%) —— 产出组合收益曲线
         科创板 688/689: 限价保护价下单(市价单会被平台拒单), 单笔最小 200 股
输出   : 1) 组合收益(真实下单口径) 2) 单因子 Rank IC/ICIR(P-001) 3) FINALJSON 行
防未来 : avoid_future_data=True + 因子只用 T 日收盘/成交量 + get_price(end_date=T)
"""
import json
import math
import numpy as np
import pandas as pd


# ---------------- 可调参数 ----------------
TOP_N = 30          # 持仓数 (全市场验证用 30 只等权)
LOOKBACK = 40       # 数据回看天数 (覆盖最大窗口 20 + return(20) 需 21 天)
UNIV_CAP = 1300     # 横截面池子规模 (5.5万行/周铁律反推: lookback=40 → 1300)
UNIV_SAMPLE = "spread"  # "spread"=全池等距抽样(覆盖 0/3/6); "head"=旧行为(按代码截断, 实际只剩 0 开头)
MINLEN = 21         # 单票最少有效收盘价数
RUN_END_DATE = ""   # 显式回测末日 YYYY-MM-DD; 留空则从 context.run_params.end_date 推断
SLIP_PCT = 0.003    # 显式滑点率 (与 set_slippage 一致), 科创板限价保护价按此计算
STAR_PREFIX = ("688", "689")   # 科创板: 市价单必须带保护限价, 否则整单被拒
STAR_MIN_LOT = 200  # 科创板单笔最小 200 股
BENCHMARK = "000905.XSHG"   # 中证500

FACTOR_NAME = "@FACTOR_NAME@"
FACTOR_AST = @FACTOR_AST@
NEED_VOLUME = @NEED_VOLUME@


def initialize(context):
    g.stock_num = TOP_N
    g.lookback = LOOKBACK
    g.trade_days = 0
    g.ic_records = []
    g.last_factor_stocks = []
    g.last_factor_values = None
    g.last_closes = {}
    g.exposure_records = []
    g.initial_cash = context.portfolio.total_value
    g.account_nav = []
    g.final_done = False

    set_benchmark(BENCHMARK)
    set_option("use_real_price", True)
    set_option("avoid_future_data", True)
    log.set_level("order", "error")
    log.set_level("system", "error")
    set_order_cost(OrderCost(open_tax=0, close_tax=0.001,
                             open_commission=0.0001, close_commission=0.0001),
                   type="stock")
    set_slippage(FixedSlippage(0.003))
    run_weekly(rebalance, 1, time="open", reference_security="000300.XSHG")
    run_daily(track_account, "14:55")
    run_daily(final_report, "15:05")
    log.info(FACTOR_NAME + " standalone 启动 (top-%d 周频)" % TOP_N)


# ═══ 通用 AST 求值器 ═══
def _rank_raw(x):
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan)
    mask = np.isfinite(x)
    if mask.sum() < 3:
        return out
    from scipy.stats import rankdata
    ranks = rankdata(x[mask])
    out[mask] = (ranks - 1) / (ranks.size - 1)
    return out


def eval_ast(ast, stocks, close_df, volume_df):
    """解释伏羲 canonical_ast → 截面因子数组(长度=len(stocks))。
    支持 field(close/volume) / std/mean/return(window) / rank / neg / add/sub/mul/div。
    ts=时间序列(DataFrame, 行=日期 列=股票); cs=截面标量(np.array)。
    """
    n = len(stocks)

    def ts_of(field):
        return close_df if field == "close" else volume_df

    def ev(node):
        if "field" in node:
            return ("ts", ts_of(node["field"]))
        op = node.get("op")
        if op in ("std", "mean"):
            k, v = ev(node["args"][0])
            w = node.get("window")
            if k != "ts":
                return ("cs", np.full(n, np.nan))
            return ("ts", v.rolling(w).std() if op == "std" else v.rolling(w).mean())
        if op == "return":
            k, v = ev(node["args"][0])
            w = node.get("window")
            if k != "ts":
                return ("cs", np.full(n, np.nan))
            return ("ts", v.pct_change(w))
        if op == "rank":
            k, v = ev(node["args"][0])
            if k == "ts":
                return ("cs", _rank_raw(v.iloc[-1].values))
            return ("cs", _rank_raw(v))
        if op == "neg":
            k, v = ev(node["args"][0])
            return (k, -v)
        if op in ("add", "sub", "mul", "div"):
            k1, v1 = ev(node["args"][0])
            k2, v2 = ev(node["args"][1])
            if op == "add":
                r = v1 + v2
            elif op == "sub":
                r = v1 - v2
            elif op == "mul":
                r = v1 * v2
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    r = v1 / v2
            return ("ts" if (k1 == "ts" or k2 == "ts") else "cs", r)
        raise ValueError("unsupported op %r" % op)

    k, v = ev(ast)
    if k == "ts":
        arr = v.iloc[-1].values.astype(float)
    else:
        arr = np.asarray(v, dtype=float)
    return arr, np.isfinite(arr)


def compute_factor(stocks, close_df, volume_df):
    return eval_ast(FACTOR_AST, stocks, close_df, volume_df)


# ═══ 辅助 ═══
def rank_pct(arr, valid):
    n = len(arr)
    result = np.full(n, 0.5)
    va = arr[valid]
    if len(va) < 3:
        return result
    from scipy.stats import rankdata
    ranks = rankdata(va)
    pct = (ranks - 1) / (len(va) - 1)
    result[valid] = pct
    result[~valid] = 0.5
    return result


# ═══ 科创板下单: 市价单必须带保护限价 ═══
def _is_star(s):
    return s.startswith(STAR_PREFIX)


def _protect_style(s, d, side):
    """科创板的限价保护价; 非科创板返回 None (走默认市价单, 行为与历史一致)。

    聚宽规则: 科创板不接受无保护价的市价单
      ("科创板市价单需要指定保护限价, 取值必须大于 0 且小于 1 万元")。
    买: ref*(1+SLIP_PCT) 向上取整到分, 不超过涨停价;
    卖: ref*(1-SLIP_PCT) 向下取整到分, 不低于跌停价。
    滑点在这里显式吃掉, 不再依赖 set_slippage 对限价单的处理。
    """
    if not _is_star(s):
        return None
    ref = d.day_open if (d.day_open and d.day_open > 0) else d.last_price
    if not ref or ref <= 0:
        return None
    if side == "buy":
        px = min(ref * (1.0 + SLIP_PCT), getattr(d, "high_limit", ref * 1.2) or ref * 1.2)
        px = math.ceil(px * 100.0 - 1e-9) / 100.0
    else:
        px = max(ref * (1.0 - SLIP_PCT), getattr(d, "low_limit", ref * 0.8) or ref * 0.8)
        px = math.floor(px * 100.0 + 1e-9) / 100.0
    if px <= 0:
        return None
    return LimitOrderStyle(price=px)


def _print_ic_report():
    if len(g.ic_records) == 0:
        log.info("[IC] %s: NO IC RECORDS" % FACTOR_NAME)
        return
    ic_arr = np.array(g.ic_records)
    ic_mean = float(np.mean(ic_arr))
    ic_std = float(np.std(ic_arr))
    icir = ic_mean / ic_std if ic_std > 0 else 0.0
    ic_pos = float(np.mean(ic_arr > 0))
    log.info("=" * 55)
    log.info("  %s  Per-Factor IC/ICIR Report (P-001)" % FACTOR_NAME)
    log.info("=" * 55)
    log.info("  Rank IC mean:       %+.4f" % ic_mean)
    log.info("  Rank IC std:         %.4f" % ic_std)
    log.info("  ICIR (mean/std):    %+.4f" % icir)
    log.info("  IC > 0 ratio:       %.1f%%" % (ic_pos * 100))
    log.info("  Total IC samples:   %d" % len(ic_arr))
    log.info("=" * 55)


def _print_exposure_report():
    if len(g.exposure_records) < 10:
        return
    held = np.array([e[1] for e in g.exposure_records], dtype=float)
    held = held[~np.isnan(held)]
    if len(held) < 10:
        return
    base_n = min(20, max(4, len(held) // 2))
    base = held[:base_n]
    recent = held[-4:]
    base_mean, recent_mean = float(np.mean(base)), float(np.mean(recent))
    ratio = recent_mean / base_mean if base_mean > 0 else float("nan")
    log.info("  [EXPOSURE] held std base=%.4f recent=%.4f ratio=%.2fx" %
             (base_mean, recent_mean, ratio))


# ═══ 周度调仓 ═══
def rebalance(context):
    import gc
    g.trade_days += 1
    if g.trade_days < 5:
        return

    prev_date = context.previous_date

    universe_all = get_all_securities(["stock"], prev_date).index.tolist()
    universe_all = [s for s in universe_all if s.startswith(("0", "3", "6"))]
    filtered = []
    try:
        cd = get_current_data()
        for s in universe_all:
            try:
                if cd[s].paused or cd[s].is_st:
                    continue
                if cd[s].high_limit <= cd[s].last_price:
                    continue
                filtered.append(s)
            except Exception:
                continue
    except Exception:
        filtered = list(universe_all)
    filtered = [s for s in filtered if s.startswith(("0", "3", "6"))]
    if UNIV_SAMPLE == "spread" and len(filtered) > UNIV_CAP:
        _step = len(filtered) / float(UNIV_CAP)
        universe = [filtered[int(i * _step)] for i in range(UNIV_CAP)]
    else:
        universe = filtered[:UNIV_CAP]
    if len(universe) < g.stock_num:
        return

    lb = g.lookback
    _P = getattr(pd, "Panel", None)

    px_c = get_price(universe, count=lb, end_date=prev_date, frequency="daily",
                     fields="close", skip_paused=False, fq="pre")
    if _P is not None and isinstance(px_c, _P):
        px_c = px_c["close"] if "close" in getattr(px_c, "items", []) else px_c.minor_xs("close")
    if px_c is None or px_c.shape[0] < 6 or px_c.shape[1] == 0:
        if not getattr(g, "diag_no_data", False):
            log.info("[DIAG] get_price 空/不足 at %s shape=%s" %
                     (prev_date, getattr(px_c, "shape", None)))
            g.diag_no_data = True
        return

    close_df = px_c.reindex(columns=universe)

    volume_df = None
    if NEED_VOLUME:
        px_v = get_price(universe, count=lb, end_date=prev_date, frequency="daily",
                         fields="volume", skip_paused=False, fq="pre")
        if _P is not None and isinstance(px_v, _P):
            px_v = px_v["volume"] if "volume" in getattr(px_v, "items", []) else px_v.minor_xs("volume")
        if px_v is not None and px_v.shape[1] > 0:
            volume_df = px_v.reindex(columns=universe)

    valid_u = [s for s in universe
               if s in close_df.columns and int((~close_df[s].isna()).sum()) >= MINLEN]
    if len(valid_u) < g.stock_num:
        return

    # 关键修复: 因子矩阵必须截到 valid_u 再算, 保证 返回数组[i] <-> valid_u[i] 同序同长。
    # 旧版把整个 universe 传给 eval_ast, rank 覆盖 universe 所有列, 返回长度 = len(universe),
    # 却仍用 valid_u 当下标取票 → 下标越界 IndexError / 未越界时张冠李戴(选错票 + IC 配对错位)。
    close_v = close_df[valid_u]
    volume_v = volume_df[valid_u] if volume_df is not None else None

    try:
        arr, v = compute_factor(valid_u, close_v, volume_v)
    except Exception as _fe:
        if not getattr(g, "diag_factor_err", False):
            log.info("[DIAG] 因子计算异常: %r at %s" % (_fe, prev_date))
            g.diag_factor_err = True
        return
    arr = np.asarray(arr, dtype=float).reshape(-1)
    v = np.asarray(v, dtype=bool).reshape(-1)
    if arr.shape[0] != len(valid_u) or v.shape[0] != len(valid_u):
        if not getattr(g, "diag_shape_mismatch", False):
            log.info("[DIAG] 因子数组失配 arr=%d valid=%d at %s — 跳过本轮" %
                     (arr.shape[0], len(valid_u), prev_date))
            g.diag_shape_mismatch = True
        return
    nv = int(np.sum(v))
    if nv < 10:
        if not getattr(g, "diag_low_valid", False):
            log.info("[DIAG] valid=%d (<10) at %s — 不交易" % (nv, prev_date))
            g.diag_low_valid = True
        return

    rp = rank_pct(arr, v)

    snap_closes = {}
    for s in px_c.columns:
        last_v = px_c[s].iloc[-1]
        if np.isfinite(last_v):
            snap_closes[s] = float(last_v)

    if len(g.last_factor_stocks) > 0 and g.last_factor_values is not None:
        fwd_rets, factor_vals = [], []
        for i, s in enumerate(g.last_factor_stocks):
            prev_close = g.last_closes.get(s)
            curr_close = snap_closes.get(s)
            fv = g.last_factor_values[i]
            if (prev_close and prev_close > 0 and curr_close and curr_close > 0
                    and not np.isnan(fv)):
                fwd_rets.append((curr_close / prev_close) - 1.0)
                factor_vals.append(fv)
        if len(fwd_rets) >= 30:
            try:
                from scipy.stats import spearmanr
                ic, _ = spearmanr(factor_vals, fwd_rets)
                if ic == ic:
                    g.ic_records.append(float(ic))
            except Exception:
                pass

    g.last_factor_stocks = valid_u
    g.last_factor_values = rp.copy()
    g.last_closes = {}
    for s in valid_u:
        if s in close_df.columns and s in snap_closes:
            g.last_closes[s] = snap_closes[s]

    # 选股 & 调仓 (HIGH=做多: 取最高 rp)
    comp = np.where(v, -rp, np.inf)
    topn = int(min(g.stock_num, int(np.sum(v))))
    order = np.argsort(comp, kind="mergesort")
    top_idx = [int(i) for i in order if comp[int(i)] < np.inf][:topn]
    top_stocks = [valid_u[i] for i in top_idx]

    try:
        held_vals = rp[top_idx][:len(top_stocks)]
        g.exposure_records.append((g.trade_days,
                                   float(np.std(held_vals)) if len(held_vals) > 1 else 0.0,
                                   float(np.std(rp[v])) if int(np.sum(v)) > 1 else 0.0))
        if len(g.exposure_records) > 500:
            g.exposure_records = g.exposure_records[-400:]
    except Exception:
        pass

    cd_order = get_current_data()

    for s in list(context.portfolio.positions.keys()):
        if s not in top_stocks:
            try:
                _dd = cd_order[s]
                if _dd.paused:
                    continue
                pos = context.portfolio.positions[s]
                if pos.total_amount < 100:
                    continue
                order_target(s, 0, style=_protect_style(s, _dd, "sell"))
            except Exception:
                pass

    if len(top_stocks) == 0:
        return

    w = context.portfolio.total_value / len(top_stocks)
    ordered = 0
    for s in top_stocks:
        try:
            if context.portfolio.available_cash < w * 0.5:
                break
            d = cd_order[s]
            if d.paused or d.is_st:
                continue
            if d.high_limit <= d.last_price:
                continue
            ref_px = d.day_open if (d.day_open and d.day_open > 0) else d.last_price
            if ref_px <= 0 or w / ref_px < 100:
                continue
            _w = w
            # 科创板单笔最小 200 股: 不足一手时补到 200 股, 现金不够则本轮跳过
            if _is_star(s) and int(_w / ref_px) < STAR_MIN_LOT:
                _w = STAR_MIN_LOT * ref_px
                if context.portfolio.available_cash < _w:
                    continue
            _pos = context.portfolio.positions.get(s)
            if _pos is not None and getattr(_pos, "total_amount", 0) > 0:
                _delta = int(_w / ref_px) - int(_pos.total_amount)
                if abs(_delta) < 100:
                    continue
            order_target_value(s, _w, style=_protect_style(s, d, "buy"))
            ordered += 1
        except Exception:
            continue

    latest_ic = g.ic_records[-1] if g.ic_records else 0.0
    log.info("%s | week=%d | ordered=%d/%d | univ=%d | IC=%.4f" %
             (FACTOR_NAME, g.trade_days, ordered, len(top_stocks), int(np.sum(v)), latest_ic))

    if len(g.ic_records) >= 20 and g.trade_days % 26 == 0:
        _print_ic_report()
        _print_exposure_report()

    gc.collect()


# ═══ 账户净值跟踪 + 期末汇总 ═══
def track_account(context):
    g.account_nav.append(
        (context.current_dt.date(), context.portfolio.total_value / g.initial_cash))


def final_report(context):
    if g.final_done:
        return
    # 旧版在取不到 run_params.end_date 时用 "当天" 兜底, 于是回测第一天就写下
    # total_ret=0.0 并锁死 final_done, 真正的期末汇总永远不会输出。改为判不出就跳过。
    last_day = None
    _cands = []
    if RUN_END_DATE:
        _cands.append(RUN_END_DATE)
    try:
        _cands.append(context.run_params.end_date)
    except Exception:
        pass
    for _c in _cands:
        if _c is None:
            continue
        try:
            last_day = get_trade_days(end_date=_c, count=1)[0]
            break
        except Exception:
            continue
    if last_day is None:
        return
    if context.current_dt.date() < last_day:
        return

    out = {}

    if len(g.account_nav) > 2:
        navs = [v for _, v in g.account_nav]
        dret = np.diff(navs) / navs[:-1]
        total = navs[-1] - 1.0
        n_days = len(navs)
        sr = float(dret.mean() / dret.std() * np.sqrt(252)) if dret.std() > 0 else 0.0
        dd = float(min(np.array(navs) / np.maximum.accumulate(navs) - 1.0))
        out["portfolio"] = {
            "total_ret_pct": round(total * 100, 2),
            "ann_pct": round(((1.0 + total) ** (252.0 / n_days) - 1.0) * 100, 2) if n_days > 0 else 0.0,
            "sr": round(sr, 3),
            "maxdd_pct": round(dd * 100, 2),
            "n_days": n_days,
        }
        out["jq_return"] = out["portfolio"]["total_ret_pct"]
    else:
        out["portfolio"] = {"total_ret_pct": 0.0, "n_days": 0}

    if g.ic_records:
        a = np.array(g.ic_records)
        out["factor_ic"] = {
            "factor": FACTOR_NAME,
            "ic_mean": round(float(a.mean()), 4),
            "icir": round(float(a.mean() / a.std()), 4) if a.std() > 0 else 0.0,
            "ic_pos_rate": round(float((a > 0).mean()), 3),
            "n": int(len(a)),
        }
    else:
        out["factor_ic"] = {"factor": FACTOR_NAME, "n": 0}

    try:
        if not g.account_nav:
            raise ValueError("no nav start")
        _run_start = g.account_nav[0][0]
        bm = get_price(BENCHMARK, start_date=_run_start,
                       end_date=last_day, fields=["close"], fq="pre",
                       panel=False, skip_paused=False)
        if len(bm) >= 2:
            out["benchmark_pct"] = round(
                (bm["close"].iloc[-1] / bm["close"].iloc[0] - 1.0) * 100, 2)
    except Exception as e:
        out["benchmark_error"] = str(e)

    log.info("FINALJSON|" + json.dumps(out, ensure_ascii=False))
    g.final_done = True
'''


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for ad in glob.glob(os.path.join(B, "run_*", "admission.json")):
        d = json.load(open(ad, encoding="utf-8"))
        if sorted(d["blockers"]) != ["independent_data_contract_receipt_missing"]:
            continue
        run = os.path.dirname(ad)
        cand = json.load(open(os.path.join(run, "candidate.json"), encoding="utf-8"))
        ev = json.load(open(os.path.join(run, "evaluation.json"), encoding="utf-8"))
        e1 = next(e for e in ev["evaluations"] if e["stage"] == "E1")
        sp = e1["metrics"]["spearman"]
        rows.append({
            "short": d["candidate_id"][:16],
            "cid": d["candidate_id"],
            "ast": cand["canonical_ast"],
            "expr": render(cand["canonical_ast"]),
            "ic": sp["mean"],
            "icir": sp["icir_nonannualized"],
        })
    rows.sort(key=lambda r: -r["ic"])

    index_lines = ["# 15 个 E0~E4 全 pass 因子 standalone JQ 回测脚本索引\n",
                   "# 按 Rank IC 降序 | 均为低波动反转族\n",
                   "| 文件名 | Rank IC | ICIR | 表达式 |\n| --- | --- | --- | --- |"]
    for i, r in enumerate(rows, 1):
        fname = "fuxi_%s_standalone.py" % r["short"]
        need_vol = has_volume(r["ast"])
        body = (TEMPLATE
                .replace("@FACTOR_NAME@", r["short"])
                .replace("@FACTOR_EXPR@", r["expr"])
                .replace("@FACTOR_AST@", repr(r["ast"]))
                .replace("@IC@", "%+.5f" % r["ic"])
                .replace("@ICIR@", "%+.3f" % r["icir"])
                .replace("@NEED_VOLUME@", "True" if need_vol else "False"))
        path = os.path.join(OUT, fname)
        with open(path, "w", encoding="utf-8") as f:
            f.write(body)
        index_lines.append("| %s | %+.5f | %+.3f | %s |" % (fname, r["ic"], r["icir"], r["expr"]))
        print("%2d. %-16s IC=%+.5f ICIR=%+.3f vol=%s -> %s" % (i, r["short"], r["ic"], r["icir"], need_vol, fname))

    with open(os.path.join(OUT, "INDEX.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(index_lines) + "\n")
    print("\n共生成 %d 个 standalone 文件 + INDEX.md 于 %s" % (len(rows), OUT))


if __name__ == "__main__":
    main()
