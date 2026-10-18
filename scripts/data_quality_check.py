# -*- coding: utf-8 -*-
"""
Phase 1: 数据基础设施质检
检查 daily_prices / daily_basic / stk_limit 的数据质量
生成详细质检报告
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime

DATA_DIR = r"e:\quant\data\raw"
OUTPUT_DIR = r"e:\quant\reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

REPORT_LINES = []
CHECK_PASS = 0
CHECK_FAIL = 0
CHECK_WARN = 0


def log(msg, level="INFO"):
    prefix = {"INFO": "  ", "PASS": "✅", "FAIL": "❌", "WARN": "⚠️ ", "H1": "\n## ", "H2": "### "}
    line = f"{prefix.get(level, '')}{msg}"
    REPORT_LINES.append(line)
    print(line)


def tally(ok):
    global CHECK_PASS, CHECK_FAIL
    if ok:
        CHECK_PASS += 1
        log("通过", "PASS")
    else:
        CHECK_FAIL += 1


def warn(msg):
    global CHECK_WARN
    CHECK_WARN += 1
    log(msg, "WARN")


# ============================================================
# 1. daily_prices.csv 核心检验
# ============================================================
log("1. daily_prices.csv 核心价格数据", "H1")

df = pd.read_csv(os.path.join(DATA_DIR, "daily_prices.csv"), encoding="utf-8-sig",
                 dtype={"ts_code": str})
df["trade_date"] = pd.to_datetime(df["trade_date"], format="mixed")
log(f"总行数: {len(df):,}")

# 1.1 基本信息
log("1.1 基本统计", "H2")
log(f"列: {list(df.columns)}")
log(f"日期范围: {df['trade_date'].min().date()} ~ {df['trade_date'].max().date()}")
log(f"股票数量: {df['ts_code'].nunique()}")
log(f"日均股票数: {df.groupby('trade_date').size().mean():.0f}")

# 1.2 缺失值
log("1.2 缺失值检查", "H2")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
for col in df.columns:
    if missing[col] > 0:
        tally(False)
        log(f"{col}: {missing[col]:,} 缺失 ({missing_pct[col]}%)")
    else:
        tally(True)

# 1.3 重复项
log("1.3 重复项检查", "H2")
dupes = df.duplicated(subset=["ts_code", "trade_date"], keep=False)
n_dupes = dupes.sum()
if n_dupes > 0:
    tally(False)
    log(f"重复条数: {n_dupes} (共{df[dupes]['ts_code'].nunique()}只股票)")
    dupes_sample = df[dupes].head(6)
    log(f"样本:\n{dupes_sample.to_string(index=False)}")
else:
    tally(True)

# 1.4 价格合理性
log("1.4 价格合理性", "H2")

# 价格为0或负
bad_price = (df["open"] <= 0) | (df["high"] <= 0) | (df["low"] <= 0) | (df["close"] <= 0)
n_bad = bad_price.sum()
if n_bad > 0:
    tally(False)
    log(f"价格<=0的条数: {n_bad}")
else:
    tally(True)

# close不在[low, high]范围
bad_range = (df["close"] < df["low"]) | (df["close"] > df["high"])
n_range = bad_range.sum()
if n_range > 0:
    tally(False)
    log(f"close不在[low, high]范围: {n_range} 条")
else:
    tally(True)

# pct_chg准确性: (close - pre_close) / pre_close vs pct_chg
df_check = df[df["pre_close"].notna() & (df["pre_close"] > 0)].copy()
calc_pct = ((df_check["close"] - df_check["pre_close"]) / df_check["pre_close"] * 100).round(4)
actual_pct = df_check["pct_chg"].round(4)
pct_diff = (calc_pct - actual_pct).abs()
bad_pct = pct_diff > 0.01
n_bad_pct = bad_pct.sum()
if n_bad_pct > 0:
    tally(False)
    log(f"pct_chg计算不一致: {n_bad_pct} 条 (差异>0.01%)")
    worst = pct_diff.nlargest(5)
    log(f"最大差异样本: {worst.to_dict()}")
else:
    tally(True)

# 1.5 日期连续性
log("1.5 日期连续性", "H2")
all_dates = sorted(df["trade_date"].unique())
n_dates = len(all_dates)
log(f"总交易日数: {n_dates}")

# 读取交易日历
cal_path = os.path.join(DATA_DIR, "trade_calendar.csv")
if os.path.exists(cal_path):
    cal = pd.read_csv(cal_path, encoding="utf-8-sig")
    cal["cal_date"] = pd.to_datetime(cal["cal_date"], format="mixed")
    cal_dates = set(cal["cal_date"].dt.date)
    price_dates = set(d.date() for d in all_dates)
    missing_dates = cal_dates - price_dates
    if missing_dates:
        tally(False)
        log(f"交易日历有但数据缺失的天数: {len(missing_dates)}")
        log(f"最早5天: {sorted(missing_dates)[:5]}")
        log(f"最晚5天: {sorted(missing_dates)[-5:]}")
    else:
        tally(True)
else:
    warn("交易日历文件不存在，跳过日期连续性检查")

# 检查是否包含周末数据
df["weekday"] = df["trade_date"].dt.dayofweek
weekend_data = df[df["weekday"] >= 5]
if len(weekend_data) > 0:
    warn(f"周末数据: {len(weekend_data)} 条 ({weekend_data['trade_date'].nunique()}天)")

# 1.6 停牌检测（当日无成交）
log("1.6 停牌/无成交检测", "H2")
no_volume = df["vol"] == 0
n_no_vol = no_volume.sum()
zero_amount = df["amount"] == 0
n_zero_amt = zero_amount.sum()
log(f"成交量为0: {n_no_vol} 条")
log(f"成交额为0: {n_zero_amt} 条")
if n_no_vol > 0:
    warn(f"存在{n_no_vol}条零成交量数据（可能是停牌日）")

# 1.7 涨跌停标记
log("1.7 涨跌停检测", "H2")
limit_path = os.path.join(DATA_DIR, "stk_limit.csv")
if os.path.exists(limit_path):
    limits = pd.read_csv(limit_path, encoding="utf-8-sig", dtype={"ts_code": str})
    limits["trade_date"] = pd.to_datetime(limits["trade_date"], format="mixed")
    # 合并检查涨停/跌停
    df_l = df.merge(limits[["ts_code", "trade_date", "up_limit", "down_limit"]],
                    on=["ts_code", "trade_date"], how="left")
    at_up = (df_l["close"] >= df_l["up_limit"]) & df_l["up_limit"].notna()
    at_down = (df_l["close"] <= df_l["down_limit"]) & df_l["down_limit"].notna()
    n_up = at_up.sum()
    n_down = at_down.sum()
    log(f"涨停收盘: {n_up} 条 ({n_up/len(df_l)*100:.2f}%)")
    log(f"跌停收盘: {n_down} 条 ({n_down/len(df_l)*100:.2f}%)")
    if n_up > 0:
        warn(f"涨停日不可买入，回测需过滤（否则虚假收益来源）")
    tally(True)
else:
    warn("stk_limit.csv 不存在，跳过涨跌停检测")

# 1.8 复权因子
log("1.8 复权因子", "H2")
adj_path = os.path.join(DATA_DIR, "adj_factor.csv")
if os.path.exists(adj_path):
    adj = pd.read_csv(adj_path, encoding="utf-8-sig", dtype={"ts_code": str})
    adj["trade_date"] = pd.to_datetime(adj["trade_date"], format="mixed")
    log(f"复权因子行数: {len(adj):,}, 股票数: {adj['ts_code'].nunique()}")
    # 检查覆盖范围
    price_stocks = set(df["ts_code"].unique())
    adj_stocks = set(adj["ts_code"].unique())
    missing_adj = price_stocks - adj_stocks
    if missing_adj:
        warn(f"{len(missing_adj)}只股票有价格但无复权因子")
    else:
        tally(True)
else:
    warn("adj_factor.csv 不存在")

# ============================================================
# 2. daily_basic.csv 日频估值数据检验
# ============================================================
log("2. daily_basic.csv 日频估值数据", "H1")

if os.path.exists(os.path.join(DATA_DIR, "daily_basic.csv")):
    db = pd.read_csv(os.path.join(DATA_DIR, "daily_basic.csv"), encoding="utf-8-sig",
                     dtype={"ts_code": str})
    db["trade_date"] = pd.to_datetime(db["trade_date"], format="mixed")
    log(f"总行数: {len(db):,}, 股票数: {db['ts_code'].nunique()}")

    # 2.1 与价格数据覆盖率
    log("2.1 与 daily_prices 的匹配率", "H2")
    price_keys = set(zip(df["ts_code"], df["trade_date"].dt.date))
    basic_keys = set(zip(db["ts_code"], db["trade_date"].dt.date))
    matched = len(price_keys & basic_keys)
    coverage = matched / len(price_keys) * 100
    log(f"匹配率: {coverage:.1f}% ({matched:,}/{len(price_keys):,})")
    if coverage >= 95:
        tally(True)
    else:
        tally(False)

    # 2.2 PE/PB 合理性
    log("2.2 PE/PB 数值合理性", "H2")
    for col, label, lo, hi in [("pe_ttm", "PE_TTM", 0, 1000),
                                 ("pb", "PB", 0, 50),
                                 ("ps_ttm", "PS_TTM", 0, 100)]:
        if col in db.columns:
            valid = db[col].notna()
            neg = (db[col] <= 0) & valid
            extreme = (db[col] > hi) & valid
            log(f"{label}: 缺失{db[col].isna().sum():,}条, 负值{neg.sum():,}条, 极端>{hi}: {extreme.sum():,}条")
            if neg.sum() > len(db) * 0.05:
                warn(f"{label}负值占比过高: {neg.sum()/len(db)*100:.1f}%")

    # 2.3 市值合理性
    log("2.3 市值合理性", "H2")
    if "total_mv" in db.columns:
        valid_mv = db["total_mv"].notna() & (db["total_mv"] > 0)
        log(f"总市值(万元): 中位数{db.loc[valid_mv, 'total_mv'].median()/1e4:.0f}亿, "
            f"最小{db.loc[valid_mv, 'total_mv'].min()/1e4:.1f}亿, "
            f"最大{db.loc[valid_mv, 'total_mv'].max()/1e4:.0f}亿")
        tally(True)
else:
    warn("daily_basic.csv 不存在")

# ============================================================
# 3. 股票列表检查
# ============================================================
log("3. 股票列表", "H1")
stock_path = os.path.join(DATA_DIR, "stock_list.csv")
if os.path.exists(stock_path):
    stocks = pd.read_csv(stock_path, encoding="utf-8-sig", dtype=str)
    log(f"股票总数: {len(stocks)}")
    log(f"列: {list(stocks.columns)}")

    # 检查是否有退市股在价格数据中
    if "list_status" in stocks.columns:
        active = stocks[stocks["list_status"] == "L"]
        delisted = stocks[stocks["list_status"] == "D"]
        log(f"上市: {len(active)}, 退市: {len(delisted)}")

        # 退市股在daily_prices中的存在情况
        if "symbol" in stocks.columns:
            price_codes = set(df["ts_code"].unique())
            list_codes_active = set(active["symbol"].str.replace("XSHG", ".XSHG").str.replace("XSHE", ".XSHE"))
            # 简单匹配
            log(f"价格数据中的股票数: {len(price_codes)}")
            log(f"上市股票数: {len(active)}")
    tally(True)
else:
    warn("stock_list.csv 不存在")

# ============================================================
# 4. 申万行业数据
# ============================================================
log("4. 申万行业数据 (sw_daily.csv)", "H1")
sw_path = os.path.join(DATA_DIR, "sw_daily.csv")
if os.path.exists(sw_path):
    sw = pd.read_csv(sw_path, encoding="utf-8-sig", dtype={"ts_code": str})
    sw["trade_date"] = pd.to_datetime(sw["trade_date"], format="mixed")
    log(f"行数: {len(sw):,}, 股票数: {sw['ts_code'].nunique()}")
    log(f"日期范围: {sw['trade_date'].min().date()} ~ {sw['trade_date'].max().date()}")
    tally(True)
else:
    warn("sw_daily.csv 不存在")

# ============================================================
# 5. 总结
# ============================================================
log("\n" + "=" * 60, "H1")
log("质检总结", "H1")
log(f"通过: {CHECK_PASS} 项", "PASS")
log(f"失败: {CHECK_FAIL} 项", "FAIL" if CHECK_FAIL > 0 else "INFO")
log(f"警告: {CHECK_WARN} 项", "WARN" if CHECK_WARN > 0 else "INFO")
log("=" * 60)

# 保存报告
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_path = os.path.join(OUTPUT_DIR, f"data_quality_report_{timestamp}.txt")
with open(report_path, "w", encoding="utf-8-sig") as f:
    f.write("\n".join(REPORT_LINES))
log(f"\n报告已保存: {report_path}", "INFO")
