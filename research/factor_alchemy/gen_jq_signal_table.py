# -*- coding: utf-8 -*-
"""
M6 信号表导出 → JQ 执行器代码生成 (2026-08-23)
================================================
架构 = 参考策略(63846)同款: 本地求值 (窄池 LGBM fwd5 + EMA0.3 + 5日重平衡)
      → 信号表嵌入 → JQ 纯执行 (D+1 开盘, 无前视)。
配置依据: 18 配置扫描中净收益最优 + 日换手 10.1% 达标 (≤20% 约束)。
本地预期 (诚实 open-to-open): 毛 +61.59% / 净 -35.64% / SR净 -0.10。
JQ 裁决目的: 停牌/涨跌停/滑点等引擎细节的最终判定 (JQ 是唯一真相源)。
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
warnings.filterwarnings("ignore")

from p0_conditional_line import load_wide_light

DATA_DIR = BASE.parent / "data"
PRED_CSV = DATA_DIR / "narrow_pool_lgbm_fwd5_oos_preds.csv"
JQ_OUT = BASE / "jq_narrow_pool_lgbm_fwd5_reb5_signal_standalone.py"

TOP_K = 8
EMA_ALPHA = 0.3          # 与 narrow_pool_lgbm_stable_sim.py 一致
REBAL_CYCLE = 5
COST_NOTE = "OrderCost(0,0.001,0.0001,0.0001)+FixedSlippage(0.003)"


def build_signal_table():
    df = pd.read_csv(PRED_CSV)
    df["trade_date"] = pd.to_datetime(
        df["trade_date"].astype(str).str.replace(r"\.0$", "", regex=True),
        format="mixed", errors="coerce")
    df = df.drop_duplicates(subset=["trade_date", "ts_code"], keep="last")
    # 宽表 → EMA 平滑 (alpha=0.7 权重给当前值, 与 sim 完全一致)
    w = df.pivot(index="trade_date", columns="ts_code", values="pred").sort_index()
    w = w.ewm(alpha=1 - EMA_ALPHA, adjust=False, axis=0).mean()
    close_w, open_w, vol_w, lhb_w = load_wide_light()
    dates = close_w.index
    dev = close_w / close_w.rolling(20).mean() - 1.0

    first_d = w.index[0]
    table = {}
    for i, t in enumerate(w.index):
        if i % REBAL_CYCLE != 0:
            continue
        s = dev.loc[t]
        valid = np.isfinite(s.to_numpy())
        pool = s[valid].nsmallest(1000).index
        row = w.loc[t].reindex(pool).dropna()
        top = row.sort_values(ascending=False, kind="mergesort").nlargest(TOP_K)
        if len(top) < TOP_K:
            continue
        day = {c: round(float(v), 4) for c, v in top.items()}
        table[t.strftime("%Y-%m-%d")] = day
    print(f"  信号表: {len(table)} 个重平衡日 × {TOP_K} 只 = {len(table) * TOP_K} 条记录")
    print(f"  覆盖: {min(table)} ~ {max(table)}")
    return table


def _to_jq_code(ts_code):
    """Tushare ts_code -> 聚宽代码. 北交所(.BJ 920新代码段) 聚宽2024-10才启用, 2023年回测无法解析 -> 返回 None 剔除"""
    if ts_code.endswith('.SZ'):
        return ts_code[:-3] + '.XSHE'
    if ts_code.endswith('.SH'):
        return ts_code[:-3] + '.XSHG'
    return None


def gen_jq_code(table):
    lines = []
    lines.append("# -*- coding: utf-8 -*-")
    lines.append('"""')
    lines.append("窄池 LGBM 信号表执行器 — fwd5 + EMA0.3 + 5日重平衡 (2026-08-23)")
    lines.append("=" * 72)
    lines.append("架构 = 参考策略(聚宽63846)同款: 本地求值 → 信号表嵌入 → JQ 纯执行")
    lines.append("无前视保证: signal[D] 用 D 日收盘及以前数据在本地生成,")
    lines.append("  JQ 在 D+1 开盘执行 (context.previous_date == D, run_daily time='open')")
    lines.append("本地预期 (open-to-open 诚实口径, 2023-02~2026-05 全 OOS):")
    lines.append("  毛 +61.59% / 净 -35.64% / SR净 -0.10 / MaxDD -62.6% / 日换手 10.1%")
    lines.append("  分年净: 2023 -44.4% / 2024 +15.8% / 2025 +10.6% / 2026 -9.6%")
    lines.append("JQ 裁决目的: 停牌/涨跌停/滑点等引擎细节对净收益的最终判定")
    lines.append("建议回测区间: 2023-02-01 ~ 2026-05-31 (全 OOS 覆盖)")
    lines.append("注意: 勿只看 2024-2025 行情窗 — 参考策略的 181 天短窗幻觉教训")
    lines.append("P-001 合规: 回测结束输出信号分 Rank IC/ICIR")
    lines.append("代码格式: 已转换 .SZ->.XSHE / .SH->.XSHG; 北交所(.BJ)剔除(920段2024-10才在JQ可用)")
    lines.append('"""')
    lines.append("")
    lines.append("import numpy as np")
    lines.append("import pandas as pd")
    lines.append("")
    lines.append("# ========= 信号表 (重平衡日 -> {code: score}) =========")
    lines.append("SIGNAL_TABLE = {")
    for d in sorted(table):
        day = table[d]
        conv = []
        for c, v in sorted(day.items()):
            jc = _to_jq_code(c)
            if jc:
                conv.append('"%s": %s' % (jc, v))
        if not conv:
            continue   # 全北交所的极端日期: 跳过
        items = ", ".join(conv)
        lines.append('    "%s": {%s},' % (d, items))
    lines.append("}")
    lines.append("")
    lines.append("")
    lines.append("TOP_K = 8")
    lines.append("")
    lines.append("")
    lines.append("def initialize(context):")
    lines.append("    g.rebal_dates = sorted(SIGNAL_TABLE.keys())")
    lines.append("    g.trade_days = 0")
    lines.append("    g.daily_values = []       # 每日收盘总资产 (收益/MaxDD 统计)")
    lines.append("    g.ic_samples = []         # [(score, open_to_open_ret)] 信号分 IC 样本")
    lines.append("    g.pending_ic = []         # [(code, score, ref_open)] 待结算")
    lines.append("    g.turn_events = []        # 每次重平衡的 (卖出只数, 买入只数)")
    lines.append("    g.last_report_day = 0")
    lines.append("    set_benchmark('000852.XSHG')")
    lines.append("    set_option('use_real_price', True)")
    lines.append("    set_order_cost(OrderCost(open_tax=0, close_tax=0.001,")
    lines.append("                              open_commission=0.0001, close_commission=0.0001),")
    lines.append("                   type='stock')")
    lines.append("    set_slippage(FixedSlippage(0.003))")
    lines.append("    log.set_level('order', 'error')")
    lines.append("    log.set_level('system', 'error')  # P-20260902-016: 抑制 order 函数内部仓位查询的 WARNING")
    lines.append("    run_daily(rebalance, time='open')")
    lines.append("    run_daily(record_nav, time='close')")
    lines.append("    log.info('narrow_pool_lgbm signal executor started, rebal_days=%d, range=%s~%s'")
    lines.append("             % (len(g.rebal_dates), g.rebal_dates[0], g.rebal_dates[-1]))")
    lines.append("")
    lines.append("")
    lines.append("def _prev_close(stock):")
    lines.append("    try:")
    lines.append("        # count=2 取 [-2]: 盘中 count=1 的 close 是当天实时价, 不是昨收")
    lines.append("        h = attribute_history(stock, 2, '1d', ['close'], df=False)")
    lines.append("        return float(h['close'][-2])")
    lines.append("    except Exception:")
    lines.append("        return None")
    lines.append("")
    lines.append("")
    lines.append("def _today_open(stock):")
    lines.append("    try:")
    lines.append("        h = attribute_history(stock, 1, '1d', ['open'], df=False)")
    lines.append("        return float(h['open'][0])")
    lines.append("    except Exception:")
    lines.append("        return None")
    lines.append("")
    lines.append("")
    lines.append("def is_kechuang(stock):")
    lines.append("    return isinstance(stock, str) and stock.startswith('688') and stock.endswith('.XSHG')")
    lines.append("")
    lines.append("")
    lines.append("def record_nav(context):")
    lines.append("    g.daily_values.append(float(context.portfolio.total_value))")
    lines.append("")
    lines.append("")
    lines.append("def rebalance(context):")
    lines.append("    g.trade_days += 1")
    lines.append("    prev_date = context.previous_date.strftime('%Y-%m-%d')")
    lines.append("")
    lines.append("    # ---- 结算上批 IC 样本: score vs open-to-open 收益 ----")
    lines.append("    if g.pending_ic:")
    lines.append("        for code, score, ref_open in g.pending_ic:")
    lines.append("            o = _today_open(code)")
    lines.append("            if o and ref_open and ref_open > 0:")
    lines.append("                g.ic_samples.append((score, o / ref_open - 1.0))")
    lines.append("        g.pending_ic = []")
    lines.append("")
    lines.append("    if prev_date not in SIGNAL_TABLE:")
    lines.append("        return   # 非重平衡日: 持仓不动")
    lines.append("")
    lines.append("    target = SIGNAL_TABLE[prev_date]")
    lines.append("    cd = get_current_data()")
    lines.append("")
    lines.append("    # ---- 卖出: 不在目标名单的持仓清仓 (order_target 清仓语义含零股) ----")
    lines.append("    sell_cnt = 0")
    lines.append("    for s in list(context.portfolio.positions.keys()):")
    lines.append("        if s in target:")
    lines.append("            continue")
    lines.append("        try:")
    lines.append("            if cd[s].paused:")
    lines.append("                continue   # 停牌卖不出, 下个重平衡日再试")
    lines.append("            order_target(s, 0)")
    lines.append("            sell_cnt += 1")
    lines.append("        except Exception:")
    lines.append("            pass")
    lines.append("")
    lines.append("    # ---- 买入: 等权 8 只, 手数取整 order(s, lots*100), 跳过不足 100 股 ----")
    lines.append("    w = context.portfolio.total_value / float(len(target))")
    lines.append("    buy_cnt = 0")
    lines.append("    for s, score in target.items():")
    lines.append("        try:")
    lines.append("            if s in context.portfolio.positions:")
    lines.append("                continue")
    lines.append("            if context.portfolio.available_cash < w * 0.5:")
    lines.append("                break")
    lines.append("            d = cd[s]")
    lines.append("            if d.paused or d.is_st:")
    lines.append("                continue")
    lines.append("            px = d.day_open if (d.day_open and d.day_open > 0) else d.last_price")
    lines.append("            if px <= 0:")
    lines.append("                continue")
    lines.append("            if d.high_limit <= d.last_price:")
    lines.append("                continue   # 已涨停, 开盘买不进")
    lines.append("            lots = int(w / px / 100)")
    lines.append("            if lots < 1:")
    lines.append("                continue   # 不足 100 股")
    lines.append("            if is_kechuang(s):")
    lines.append("                pc = _prev_close(s)")
    lines.append("                if pc and pc > 0:")
    lines.append("                    order(s, lots * 100, LimitOrderStyle(min(pc * 1.2, 9999)))")
    lines.append("                else:")
    lines.append("                    continue")
    lines.append("            else:")
    lines.append("                order(s, lots * 100)")
    lines.append("            buy_cnt += 1")
    lines.append("            # IC 样本: 用当日开盘价作为买入参考价")
    lines.append("            o = _today_open(s)")
    lines.append("            g.pending_ic.append((s, score, o))")
    lines.append("        except Exception:")
    lines.append("            pass")
    lines.append("")
    lines.append("    g.turn_events.append((sell_cnt, buy_cnt))")
    lines.append("    if g.trade_days % 25 == 0:")
    lines.append("        log.info('reb | day=%d | prev_date=%s | sell=%d buy=%d | pos=%d | nav=%.0f'")
    lines.append("                 % (g.trade_days, prev_date, sell_cnt, buy_cnt,")
    lines.append("                    len(context.portfolio.positions), context.portfolio.total_value))")
    lines.append("")
    lines.append("")
    lines.append("def _final_report(context):")
    lines.append("    vals = np.array(g.daily_values)")
    lines.append("    total_ret = vals[-1] / vals[0] - 1.0 if len(vals) > 1 else 0.0")
    lines.append("    daily = np.diff(vals) / vals[:-1] if len(vals) > 1 else np.array([0.0])")
    lines.append("    sharpe = float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0")
    lines.append("    peak = np.maximum.accumulate(vals)")
    lines.append("    maxdd = float(np.min(vals / peak - 1.0)) if len(vals) else 0.0")
    lines.append("    # 换手 (平均每次重平衡换出的仓位占比)")
    lines.append("    tot_sell = sum(a for a, b in g.turn_events)")
    lines.append("    tot_buy = sum(b for a, b in g.turn_events)")
    lines.append("    n_reb = max(len(g.turn_events), 1)")
    lines.append("    avg_turn_pct = 100.0 * (tot_sell + tot_buy) / (2.0 * TOP_K * n_reb)")
    lines.append("    # 信号分 IC (score vs 买入次日 open-to-open 收益, 全体样本 spearman)")
    lines.append("    ic_mean, ic_pos = 0.0, 0.0")
    lines.append("    if len(g.ic_samples) >= 30:")
    lines.append("        try:")
    lines.append("            from scipy.stats import spearmanr")
    lines.append("            sc = np.array([a for a, b in g.ic_samples])")
    lines.append("            rt = np.array([b for a, b in g.ic_samples])")
    lines.append("            ic_mean, _ = spearmanr(sc, rt)")
    lines.append("            ic_pos = float(np.mean(rt > 0))")
    lines.append("        except Exception:")
    lines.append("            pass")
    lines.append("    log.info('=' * 60)")
    lines.append("    log.info('  [FINAL] narrow_pool_lgbm fwd5+EMA0.3+reb5 signal executor')")
    lines.append("    log.info('=' * 60)")
    lines.append("    log.info('  total_return_pct:  %.2f' % (total_ret * 100))")
    lines.append("    log.info('  sharpe:            %.4f' % sharpe)")
    lines.append("    log.info('  max_drawdown_pct:  %.2f' % (maxdd * 100))")
    lines.append("    log.info('  rebal_days:        %d | avg_turn_per_reb: %.1f%%' % (n_reb, avg_turn_pct))")
    lines.append("    log.info('  signal_rank_ic:    %.4f | ic>0_ratio: %.1f%% | n=%d'")
    lines.append("             % (ic_mean, ic_pos * 100, len(g.ic_samples)))")
    lines.append("    log.info('=' * 60)")
    lines.append("")
    lines.append("")
    lines.append("def on_strategy_end(context):")
    lines.append("    _final_report(context)")
    lines.append("")
    return "\n".join(lines)


def main():
    print("=" * 68)
    print("  窄池 LGBM 信号表导出 → JQ 执行器代码")
    print("=" * 68)
    table = build_signal_table()
    code = gen_jq_code(table)
    JQ_OUT.write_text(code, encoding="utf-8")
    print(f"  JQ 代码已生成: {JQ_OUT} ({len(code)} 字符)")


if __name__ == "__main__":
    main()
