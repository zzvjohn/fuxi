# -*- coding: utf-8 -*-
"""
JQ 数据 API 回测探针 — 验证非行情 API 在回测环境的真实行为
目的: 定位 lhb_freq_20d / margin_short_ratio / margin_balance_chg_21d / margin_mf_divergence 无收益的根因
用法: 回测区间设为 2018-01-02 ~ 2018-02-15 (或任意覆盖 2018-02-01 的短区间), 运行后查看日志输出
     将日志中所有 [PROBE] 行反馈给 AI
"""
import numpy as np
import pandas as pd

def initialize(context):
    set_benchmark("000300.XSHG")
    log.set_level("order", "error")
    log.set_level("system", "error")  # P-20260902-016: 抑制 order 函数内部仓位查询的 WARNING ("在 positions 中不存在" 刷屏)
    run_daily(probe, time="open")
    log.info("API 探针启动: 首次 >= 2018-01-25 的交易日执行探测")

def probe(context):
    if getattr(g, "done", False):
        return
    if context.current_dt.date() < pd.Timestamp("2018-01-25").date():
        return
    g.done = True
    prev = context.previous_date
    log.info("=" * 55)
    log.info("[PROBE] current=%s prev_date=%s" % (context.current_dt.date(), prev))

    # 1. get_mtss (两融)
    try:
        df = get_mtss(["000001.XSHE", "600000.XSHG", "300750.XSHE"],
                      start_date=pd.Timestamp(prev) - pd.Timedelta(days=30),
                      end_date=prev,
                      fields=["date", "sec_code", "fin_value", "fin_sec_value"])
        log.info("[PROBE] get_mtss OK rows=%d cols=%s" % (len(df), list(df.columns)))
        log.info("[PROBE] get_mtss sample: %s" % df.head(2).to_dict("records"))
    except Exception as e:
        log.info("[PROBE] get_mtss FAIL: %r" % (e,))

    # 2. get_billboard_list (龙虎榜)
    try:
        df = get_billboard_list(stock_list=None,
                                start_date=pd.Timestamp(prev) - pd.Timedelta(days=40),
                                end_date=prev, count=None)
        log.info("[PROBE] billboard OK rows=%d cols=%s" % (len(df), list(df.columns)))
        log.info("[PROBE] billboard code_sample=%s direction=%s" % (
            df["code"].head(3).tolist() if "code" in df.columns else "NO_code",
            df["direction"].unique().tolist() if "direction" in df.columns else "NO_dir"))
    except Exception as e:
        log.info("[PROBE] billboard FAIL: %r" % (e,))

    # 3. get_money_flow (资金流)
    try:
        df = get_money_flow(["000001.XSHE", "600000.XSHG"],
                            start_date=pd.Timestamp(prev) - pd.Timedelta(days=30),
                            end_date=prev,
                            fields=["date", "sec_code", "net_amount_main"])
        log.info("[PROBE] money_flow OK rows=%d cols=%s" % (len(df), list(df.columns)))
    except Exception as e:
        log.info("[PROBE] money_flow FAIL: %r" % (e,))

    # 4. get_money_flow_pro (资金流新版)
    try:
        df = get_money_flow_pro(["000001.XSHE", "600000.XSHG"],
                                start_date=pd.Timestamp(prev) - pd.Timedelta(days=30),
                                end_date=prev,
                                fields=["date", "sec_code", "net_amount_main"])
        log.info("[PROBE] money_flow_pro OK rows=%d cols=%s" % (len(df), list(df.columns)))
    except Exception as e:
        log.info("[PROBE] money_flow_pro FAIL: %r" % (e,))

    # 5. get_all_trade_days (回测交易日函数名确认)
    try:
        tds = get_all_trade_days()
        log.info("[PROBE] get_all_trade_days OK count=%d" % len(tds))
    except Exception as e:
        log.info("[PROBE] get_all_trade_days FAIL: %r" % (e,))

    # 6. get_mtss 大池一次查询 (1200只, 模拟正式策略的查询规模)
    try:
        all_stocks = get_all_securities(["stock"], prev).index.tolist()
        big = [s for s in all_stocks if s.startswith(("0", "3", "6"))][:1200]
        df = get_mtss(big, start_date=pd.Timestamp(prev) - pd.Timedelta(days=25),
                      end_date=prev,
                      fields=["date", "sec_code", "fin_value", "fin_sec_value"])
        log.info("[PROBE] get_mtss BIG(1200) OK rows=%d" % len(df))
    except Exception as e:
        log.info("[PROBE] get_mtss BIG(1200) FAIL: %r" % (e,))

    log.info("[PROBE] DONE")
    log.info("=" * 55)
