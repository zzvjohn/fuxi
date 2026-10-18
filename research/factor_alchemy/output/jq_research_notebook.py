"""
JQ 研究笔记本: Regime-Lasso 因子选择
======================================
在 JQ 研究平台运行此文件:
  1. 全量因子计算 (price + fundamentals)
  2. Regime 检测 (large_dominant / small_bull)
  3. ElasticNetCV per regime 因子选择
  4. 输出因子权重 → 复制到本地, 用 JQGenerator 生成策略

注意: 需要 JQData 付费版 (2021-2026 全量数据)
"""

# Cell 1: Imports & Setup
import numpy as np
import pandas as pd
try:
    from jqdata import *
except ImportError:
    from jqdatasdk import *
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# 配置
START_DATE = "2021-01-01"
END_DATE = "2026-06-30"
TOP_N = 30
WEEKDAY = 5  # 周五调仓
LOOKBACK_DAYS = 260  # 因子计算回看天数

# Regime 检测阈值
STYLE_SPREAD_THRESHOLD = 0.02   # CSI500-CSI300 > 2% = small_bull
TREND_THRESHOLD = -0.01         # CSI_ALL > -1% = 上涨

print("✓ Setup complete")

# Cell 2: 获取股票池
def get_stock_pool():
    """获取全A股股票池 (剔除ST/退市/新股)"""
    all_securities = get_all_securities(["stock"], date=END_DATE)
    pool = []
    for code, row in all_securities.iterrows():
        # 剔除ST/退市 (名称含ST或退)
        name = str(row.get("display_name", ""))
        if "ST" in name or "退" in name or "*" in name[:3]:
            continue
        # 剔除上市不足180天
        days_listed = (pd.Timestamp(END_DATE) - pd.Timestamp(row["start_date"])).days
        if days_listed > 180:
            pool.append(code)
    return pool

ALL_STOCKS = get_stock_pool()
print(f"Stock pool: {len(ALL_STOCKS)} stocks")
print("Sample:", ALL_STOCKS[:5])

# Cell 3: 加载价格数据
print("Loading price data...")

# 批量获取日线
price_fields = ["open", "close", "high", "low", "volume", "money"]
raw_price = get_price(
    ALL_STOCKS,
    start_date=START_DATE,
    end_date=END_DATE,
    frequency="daily",
    fields=price_fields,
    fq="post",  # 后复权
    skip_paused=False,
    panel=False
)
print(f"Raw price: {raw_price.shape}")

# 转成周频 (周五收盘)
raw_price['week'] = raw_price.index.to_period('W')
price_weekly = raw_price.groupby(["code", "week"]).last().reset_index()
price_weekly = price_weekly.set_index("week").sort_index()
weeks = sorted(price_weekly.index.unique())
print(f"Weekly: {len(weeks)} weeks, {weeks[0]} ~ {weeks[-1]}")

# 加载指数数据 (regime 检测用)
INDEX_CODES = {"CSI300": "000300.XSHG", "CSI500": "000905.XSHG", "CSI_ALL": "000985.XSHG"}
index_data = {}
for name, code in INDEX_CODES.items():
    px = get_price(code, start_date=START_DATE, end_date=END_DATE, frequency="daily", fields=["close"], fq="post")
    index_data[name] = px
    print(f"  {name}: {len(px)} days")

# Cell 4: 加载基本面数据
print("Loading fundamentals...")

q = query(
    valuation.code, valuation.market_cap, valuation.circulating_market_cap,
    valuation.pb_ratio, valuation.ps_ratio, valuation.pe_ratio,
    indicator.roe, indicator.roa, indicator.gross_profit_margin,
    indicator.net_profit_margin, indicator.inc_revenue_year_on_year,
    indicator.inc_net_profit_year_on_year,
    balance.total_assets, balance.total_equity,
    cash_flow.net_operate_cash_flow, cash_flow.net_profit,
    income.total_operating_revenue, income.operating_profit
)

# 按季度获取
fundamentals = {}
for year in range(2021, 2027):
    for qtr in range(1, 5):
        date_str = f"{year}-{qtr*3:02d}-{28 if qtr!=4 else 31}"
        if date_str > END_DATE:
            break
        try:
            df = get_fundamentals(q, statDate=date_str)
            if df is not None and len(df) > 0:
                fundamentals[date_str] = df.set_index("code")
        except:
            pass
print(f"Fundamental snapshots: {len(fundamentals)}")

# 合并为因子可用格式
def get_fundamental_for_stocks(stocks, date_str):
    """获取某一期所有股票的基本面数据"""
    # 找到最近的 snapshot
    avail = sorted(fundamentals.keys())
    target = None
    for d in avail:
        if d <= date_str:
            target = d
    if target is None:
        return {}
    df = fundamentals[target]
    result = {}
    for s in stocks:
        if s in df.index:
            row = df.loc[s]
            result[s] = {
                "market_cap": row.get("market_cap"),
                "circ_mcap": row.get("circulating_market_cap"),
                "pb": row.get("pb_ratio"),
                "ps": row.get("ps_ratio"),
                "roe": row.get("roe"),
                "roa": row.get("roa"),
                "gross_margin": row.get("gross_profit_margin"),
                "net_margin": row.get("net_profit_margin"),
                "rev_growth_yoy": row.get("inc_revenue_year_on_year"),
                "earnings_growth_yoy": row.get("inc_net_profit_year_on_year"),
                "total_assets": row.get("total_assets"),
                "total_equity": row.get("total_equity"),
                "ocf": row.get("net_operate_cash_flow"),
                "net_income": row.get("net_profit"),
                "revenue": row.get("total_operating_revenue"),
                "operating_profit": row.get("operating_profit"),
            }
    return result

# Cell 5: 因子计算函数

# 提取单周截面价格数据
def build_price_data_slice(week_str, stock_list):
    """为指定周构建 price_data dict (包含回看期数据)"""
    week_idx = weeks.index(week_str)
    lookback_start = max(0, week_idx - 52)  # 回看52周
    hist_weeks = weeks[lookback_start:week_idx+1]
    
    mask = (price_weekly["week"].isin(hist_weeks)) & (price_weekly["code"].isin(stock_list))
    hist = price_weekly[mask]
    
    result = {"close": {}, "open": {}, "high": {}, "low": {}, "volume": {}, "amount": {}}
    for code in stock_list:
        s_data = hist[hist["code"] == code]
        if len(s_data) >= 10:
            result["close"][code] = s_data["close"].values
            result["open"][code] = s_data["open"].values
            result["high"][code] = s_data["high"].values
            result["volume"][code] = s_data["volume"].values
            result["amount"][code] = s_data["money"].values
    return result

def compute_abnormal_turnover(stocks, price_data):
    volume = price_data.get('volume', {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        v = volume.get(s, np.array([]))
        if len(v) >= 20:
            arr[i] = (v[-1] - np.nanmean(v[-20:])) / (np.nanstd(v[-20:]) + 1e-10)
    valid = np.isfinite(arr)

def compute_abnormal_turnover_neut(stocks, price_data):
    arr, valid = compute_abnormal_turnover(stocks, price_data)
    return arr, valid  # TODO: add sector neutralization

def compute_accrual_quality_proxy(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    close_d = price_data.get('close', {})
    for i, s in enumerate(stocks):
        if s not in close_d:
            continue
        c = np.array(close_d[s], dtype=float)
        c_arr = c[np.isfinite(c)]
        if len(c_arr) < 8:
            continue
        ret = np.diff(c_arr) / c_arr[:-1]
        if len(ret) < 5:
            continue
        ret_std = np.nanstd(ret[-20:]) if len(ret) >= 20 else np.nanstd(ret)
        ret_abs_mean = np.nanmean(np.abs(ret[-20:])) if len(ret) >= 20 else np.nanmean(np.abs(ret))
        if ret_abs_mean is None or np.isnan(ret_abs_mean):
            continue
        quality = -(ret_std / (ret_abs_mean + 0.001))
        arr[i] = quality
    valid = np.isfinite(arr)
    return arr, valid

def compute_accruals(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    if hasattr(price_data, 'fundamentals') and price_data.fundamentals is not None:
        for i, s in enumerate(stocks):
            f = price_data.fundamentals.get(s, {})
            ni = f.get('net_income', np.nan)
            ocf = f.get('ocf', np.nan)
            ta = f.get('total_assets', np.nan)
            if np.isfinite(ni) and np.isfinite(ocf) and np.isfinite(ta) and ta > 0:
                arr[i] = (ni - ocf) / ta
    valid = np.isfinite(arr)

def compute_amihud_illiq(stocks, price_data):
    amount = price_data.get('amount', {})
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        amt = amount.get(s, np.array([]))
        c = close[s]
        if len(amt) >= 20 and len(c) >= 20:
            rets = np.abs(c[-20:] / np.roll(c[-20:], 1) - 1); rets[0] = 0
            ratio = rets / (amt[-20:] + 1)
            arr[i] = np.nanmean(ratio) * 1e8
    valid = np.isfinite(arr)

def compute_asset_growth(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    if hasattr(price_data, 'fundamentals') and price_data.fundamentals is not None:
        for i, s in enumerate(stocks):
            f = price_data.fundamentals.get(s, {})
            ta = f.get('total_assets', np.nan)
            ta_q = f.get('total_assets_q4', np.nan)
            if np.isfinite(ta) and np.isfinite(ta_q) and ta_q > 0:
                arr[i] = ta / ta_q - 1
    valid = np.isfinite(arr)

def compute_asset_turnover(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    if hasattr(price_data, 'fundamentals') and price_data.fundamentals is not None:
        for i, s in enumerate(stocks):
            f = price_data.fundamentals.get(s, {})
            rev = f.get('revenue', np.nan)
            ta = f.get('total_assets', np.nan)
            if np.isfinite(rev) and np.isfinite(ta) and ta > 0:
                arr[i] = rev / ta
    valid = np.isfinite(arr)

def compute_atr_14(stocks, price_data):
    """ATR(14) normalized by close, negated. Low ATR = positive signal."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    hd = price_data.get('high', {})
    ld = price_data.get('low', {})
    cd = price_data.get('close', {})
    for i, s in enumerate(stocks):
        if s not in hd or s not in ld or s not in cd:
            continue
        h = np.array(hd[s], dtype=float)
        l = np.array(ld[s], dtype=float)
        c = np.array(cd[s], dtype=float)
        if len(h) < 16 or len(l) < 16 or len(c) < 16:
            continue
        tr = np.zeros(len(c) - 1)
        for t in range(1, len(c)):
            tr[t - 1] = max(
                h[t] - l[t],
                abs(h[t] - c[t - 1]),
                abs(l[t] - c[t - 1]))
        if len(tr) < 14:
            continue
        alpha = 2.0 / 15.0
        ema = tr[0]
        for t in range(1, len(tr)):
            ema = alpha * tr[t] + (1 - alpha) * ema
        c_now = c[-1]
        if c_now == 0 or np.isnan(c_now) or np.isnan(ema):
            continue
        arr[i] = -ema / c_now
    valid = np.isfinite(arr)
    return arr, valid

def compute_atr_14_neut(stocks, price_data):
    arr, valid = compute_atr_14(stocks, price_data)
    return arr, valid  # TODO: add sector neutralization

def compute_attention_decay(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    return arr, np.full(n, False)  # Not implemented on JQ

def compute_avg_turnover_1m(stocks, price_data):
    volume = price_data.get('volume', {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        v = volume.get(s, np.array([]))
        if len(v) >= 21:
            arr[i] = np.nanmean(v[-21:])
    valid = np.isfinite(arr)

def compute_avg_turnover_1m_neut(stocks, price_data):
    arr, valid = compute_avg_turnover_1m(stocks, price_data)
    return arr, valid  # TODO: add sector neutralization

def compute_avg_turnover_3m(stocks, price_data):
    volume = price_data.get('volume', {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        v = volume.get(s, np.array([]))
        if len(v) >= 63:
            arr[i] = np.nanmean(v[-63:])
    valid = np.isfinite(arr)

def compute_bargaining_power_proxy(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    return arr, np.full(n, False)  # Not implemented on JQ

def compute_bear_power(stocks, price_data):
    low = price_data.get('low', {})
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        l_ = low.get(s, np.array([]))
        c = close[s]
        if len(l_) >= 13 and len(c) >= 13:
            ema = c[-1]
            alpha = 2/14
            for p in c[-14:-1][::-1]:
                ema = alpha * p + (1-alpha) * ema
            arr[i] = ema - l_[-1]  # ema高于最低价=多头
    valid = np.isfinite(arr)

def compute_beta(stocks, price_data):
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 252 and hasattr(price_data, 'market_ret') and price_data.market_ret is not None and len(price_data.market_ret) >= 252:
            stock_rets = c[-252:] / np.roll(c[-252:], 1) - 1
            stock_rets[0] = 0
            mkt = price_data.market_ret[-252:]
            valid_mask = np.isfinite(stock_rets) & np.isfinite(mkt)
            if np.sum(valid_mask) >= 60:
                cov = np.cov(stock_rets[valid_mask], mkt[valid_mask])[0,1]
                var = np.var(mkt[valid_mask])
                if var > 0:
                    arr[i] = cov / var
    valid = np.isfinite(arr)

def compute_bias_20(stocks, price_data):
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 20:
            ma = np.nanmean(c[-20:])
            if ma > 0:
                arr[i] = (c[-1] - ma) / ma
    valid = np.isfinite(arr)

def compute_boll_pct_b(stocks, price_data):
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 20:
            ma = np.nanmean(c[-20:])
            std = np.nanstd(c[-20:])
            if std > 0:
                arr[i] = (c[-1] - (ma - 2*std)) / (4*std)
    valid = np.isfinite(arr)

def compute_bp(stocks, context):
    """Book-to-Price = 1/pb_ratio. Higher BP = value = positive signal."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    try:
        q = query(valuation.code, valuation.pb_ratio).filter(
            valuation.code.in_(stocks))
        df = get_fundamentals(q, date=context.previous_date)
    except:
        return arr, np.zeros(n, dtype=bool)
    if df is None or len(df) == 0:
        return arr, np.zeros(n, dtype=bool)
    stock_idx = {s: i for i, s in enumerate(stocks)}
    for _, row in df.iterrows():
        code = row['code']
        pb = row.get('pb_ratio')
        if code in stock_idx and pb is not None and not np.isnan(pb) and pb > 0:
            arr[stock_idx[code]] = 1.0 / pb
    valid = np.isfinite(arr)
    return arr, valid

def compute_bp_neut(stocks, price_data):
    arr, valid = compute_bp(stocks, price_data)
    return arr, valid  # TODO: add sector neutralization

def compute_bull_power(stocks, price_data):
    high = price_data.get('high', {})
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        h = high.get(s, np.array([]))
        c = close[s]
        if len(h) >= 13 and len(c) >= 13:
            ema = c[-1]
            alpha = 2/14
            for p in c[-14:-1][::-1]:
                ema = alpha * p + (1-alpha) * ema
            arr[i] = h[-1] - ema
    valid = np.isfinite(arr)

def compute_capital_efficiency_proxy(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    return arr, np.full(n, False)  # Not implemented on JQ

def compute_cashflow_matching_proxy(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    return arr, np.full(n, False)  # Not implemented on JQ

def compute_cci_14(stocks, price_data):
    high, low, close = price_data.get('high',{}), price_data.get('low',{}), price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]; h = high.get(s, c); l_ = low.get(s, c)
        if len(c) >= 14 and len(h) >= 14 and len(l_) >= 14:
            tp = (h[-14:] + l_[-14:] + c[-14:]) / 3
            ma = np.nanmean(tp)
            md = np.nanmean(np.abs(tp - ma))
            if md > 0:
                arr[i] = (tp[-1] - ma) / (0.015 * md)
    valid = np.isfinite(arr)

def compute_cfp(stocks, price_data):
    # 经营性现金流/市值
    n = len(stocks)
    arr = np.full(n, np.nan)
    if hasattr(price_data, 'fundamentals') and price_data.fundamentals is not None:
        for i, s in enumerate(stocks):
            f = price_data.fundamentals.get(s, {})
            ocf = f.get('ocf', np.nan)
            mcap = f.get('market_cap', np.nan)
            if np.isfinite(ocf) and np.isfinite(mcap) and mcap > 0:
                arr[i] = ocf / mcap
    valid = np.isfinite(arr)

def compute_davol_20(stocks, price_data):
    volume = price_data.get('volume', {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        v = volume.get(s, np.array([]))
        if len(v) >= 20:
            arr[i] = v[-1] / np.nanmean(v[-20:]) - 1
    valid = np.isfinite(arr)

def compute_davol_20_neut(stocks, price_data):
    arr, valid = compute_davol_20(stocks, price_data)
    return arr, valid  # TODO: add sector neutralization

def compute_debt_coverage(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    if hasattr(price_data, 'fundamentals') and price_data.fundamentals is not None:
        for i, s in enumerate(stocks):
            f = price_data.fundamentals.get(s, {})
            ebitda = f.get('ebitda', np.nan)
            interest = f.get('interest_expense', np.nan)
            if np.isfinite(ebitda) and np.isfinite(interest) and interest > 0:
                arr[i] = ebitda / interest
    valid = np.isfinite(arr)

def compute_dollar_vol_20d(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    close_d = price_data.get('close', {})
    vol_d = price_data.get('volume', {})
    for i, s in enumerate(stocks):
        if s not in close_d or s not in vol_d:
            continue
        c = np.array(close_d[s], dtype=float)
        v = np.array(vol_d[s], dtype=float)
        min_len = min(len(c), len(v))
        if min_len < 10:
            continue
        dollar_vol = c[-min_len:] * v[-min_len:]
        avg_dv = np.nanmean(
            dollar_vol[-20:]) if len(dollar_vol) >= 20 else np.nanmean(dollar_vol)
        if np.isnan(avg_dv) or avg_dv <= 0:
            continue
        arr[i] = -np.log(avg_dv)
    valid = np.isfinite(arr)
    return arr, valid

def compute_dollar_vol_20d_neut(stocks, price_data):
    arr, valid = compute_dollar_vol_20d(stocks, price_data)
    return arr, valid  # TODO: add sector neutralization

def compute_dollar_vol_stability(stocks, price_data):
    volume = price_data.get('volume', {})
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        v = volume.get(s, np.array([]))
        c = close[s]
        if len(v) >= 20 and len(c) >= 20:
            dv = np.array([v[j] * c[j] for j in range(-20, 0)])
            arr[i] = np.nanstd(dv) / (np.nanmean(dv) + 1e-10)
    valid = np.isfinite(arr)

def compute_downside_vol(stocks, price_data):
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 63:
            rets = c[-63:] / np.roll(c[-63:], 1) - 1
            rets[0] = 0
            neg_rets = rets[rets < 0]
            if len(neg_rets) > 0:
                arr[i] = np.nanstd(neg_rets) * np.sqrt(252)
    valid = np.isfinite(arr)

def compute_downside_vol_neut(stocks, price_data):
    arr, valid = compute_downside_vol(stocks, price_data)
    return arr, valid  # TODO: add sector neutralization

def compute_dp(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    if hasattr(price_data, 'fundamentals') and price_data.fundamentals is not None:
        for i, s in enumerate(stocks):
            f = price_data.fundamentals.get(s, {})
            dps = f.get('dividend_per_share', np.nan)
            price = f.get('price', np.nan)
            if np.isfinite(dps) and np.isfinite(price) and price > 0:
                arr[i] = dps / price
    valid = np.isfinite(arr)

def compute_earnings_consistency_proxy(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    return arr, np.full(n, False)  # Not implemented on JQ

def compute_earnings_growth_yoy(stocks, context):
    n = len(stocks)
    arr = np.full(n, np.nan)
    try:
        q = query(indicator.code, indicator.inc_net_profit_year_on_year)
        q = q.filter(indicator.code.in_(stocks))
        df = get_fundamentals(q, date=context.previous_date)
    except:
        log.error('[eg] FAILED: %s' % str(e))
        return arr, np.zeros(n, dtype=bool)
    if df is None or len(df) == 0:
        return arr, np.zeros(n, dtype=bool)
    stock_idx = {s: i for i, s in enumerate(stocks)}
    for _, row in df.iterrows():
        code = row['code']
        val = row.get('inc_net_profit_year_on_year')
        if code in stock_idx and val is not None and not np.isnan(val):
            arr[stock_idx[code]] = val
    valid = np.isfinite(arr)
    return arr, valid

def compute_earnings_quality_proxy(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    return arr, np.full(n, False)  # Not implemented on JQ

def compute_earnings_season_vol_div(stocks, price_data):
    """中报窗口量价背离度: 价变与量变秩相关取负(背离→正向), 5日平滑.

    对齐本地 EarningsSeasonVolDiv:
      pct = close.pct_change(); vol = volume.pct_change()
      5日窗口内 (pct, vol) 的 Spearman秩相关取负 → 价涨量缩(背离)为正
      取最近5个窗口的均值作为截面信号
    """
    n = len(stocks)
    arr = np.full(n, np.nan)
    close_d = price_data.get('close', {})
    vol_d = price_data.get('volume', {})
    win = 5
    for i, s in enumerate(stocks):
        if s not in close_d or s not in vol_d:
            continue
        c = np.asarray(close_d[s], dtype=float)
        v = np.asarray(vol_d[s], dtype=float)
        if len(c) < 15 or len(v) < 15:
            continue
        c = c[-20:]; v = v[-20:]
        pct = np.diff(c) / c[:-1]
        denom = np.where(v[:-1] != 0, v[:-1], np.nan)
        vc = np.diff(v) / denom
        pct = np.nan_to_num(pct)
        vc = np.nan_to_num(vc)
        L = len(pct)
        rho = np.full(L, np.nan)
        for t in range(win - 1, L):
            x = pct[t - win + 1:t + 1]
            y = vc[t - win + 1:t + 1]
            rx = x.argsort().argsort().astype(float)
            ry = y.argsort().argsort().astype(float)
            mr = rx.mean(); my = ry.mean()
            num = ((rx - mr) * (ry - my)).sum()
            den = np.sqrt(((rx - mr) ** 2).sum() * ((ry - my) ** 2).sum())
            rho[t] = (-num / den) if (den > 0 and np.isfinite(den)) else 0.0
        last5 = rho[-win:]
        val = np.nanmean(last5)
        if np.isfinite(val):
            arr[i] = val
    valid = np.isfinite(arr)
    return arr, valid

def compute_earnings_stability(stocks, context):
    """盈利稳定性 = -std(inc_net_profit_year_on_year 历史). 波动小=稳定=正向.
    对齐本地 EarningsStability(-rolling(8q)std(netprofit_yoy)).
    预热: 首次调用用过去~10个季度末快照填充历史, 保证第1个交易日即有效."""
    import calendar
    n = len(stocks)
    arr = np.full(n, np.nan)
    if not hasattr(g, 'earnings_yoy_hist') or g.earnings_yoy_hist is None:
        g.earnings_yoy_hist = {}

    def _snap(dt):
        snap = {}
        try:
            end = dt.strftime('%Y-%m-%d') if hasattr(dt, 'strftime') else dt
            q = query(indicator.code, indicator.inc_net_profit_year_on_year).filter(
                indicator.code.in_(stocks))
            df = get_fundamentals(q, date=end)
        except Exception as e:
            if g.debug:
                log.error('[earnings_stability] get_fundamentals FAILED: %s' % str(e))
            return snap
        if df is None or len(df) == 0:
            return snap
        for _, row in df.iterrows():
            code = row['code']
            val = row.get('inc_net_profit_year_on_year')
            if val is not None and np.isfinite(val):
                snap[code] = float(val)
        return snap

    # 首次预热: 过去~10个季度末快照
    if not getattr(g, 'earnings_yoy_warmed', False):
        end = context.previous_date
        y, m = end.year, end.month
        q = (m - 1) // 3
        qe_month = q * 3 + 3
        qe_y = y
        if qe_month > 12:
            qe_month -= 12; qe_y += 1
        ends = [datetime.date(qe_y, qe_month, calendar.monthrange(qe_y, qe_month)[1])]
        for _ in range(9):
            qe_month -= 3
            if qe_month < 1:
                qe_month += 12; qe_y -= 1
            ends.append(datetime.date(qe_y, qe_month, calendar.monthrange(qe_y, qe_month)[1]))
        for d in ends:
            s0 = _snap(d)
            for s in stocks:
                if s in s0:
                    h = g.earnings_yoy_hist.setdefault(s, [])
                    h.append(s0[s])
                    if len(h) > 40:
                        h[:] = h[-40:]
        g.earnings_yoy_warmed = True

    # 当前快照
    snap = _snap(context.previous_date)
    for s in stocks:
        if s in snap:
            h = g.earnings_yoy_hist.setdefault(s, [])
            h.append(snap[s])
            if len(h) > 40:
                h[:] = h[-40:]
    stock_idx = {s: i for i, s in enumerate(stocks)}
    valid = np.zeros(n, dtype=bool)
    for s in snap:
        if s in stock_idx:
            h = g.earnings_yoy_hist.get(s, [])
            hv = [x for x in h if x is not None and np.isfinite(x)]
            if len(hv) >= 4:
                arr[stock_idx[s]] = -np.std(hv, ddof=1)
                valid[stock_idx[s]] = True
    return arr, valid

def compute_earnings_volume_drift(stocks, price_data):
    """业绩窗量价漂移: 缩量上涨=筹码锁定→正向. 仅保留上涨段.

    对齐本地 EarningsVolumeDrift:
      ret_5d   = close/close.shift(5) - 1                (取最新截面)
      vol_chg  = MA5(volume)/MA5(volume).shift(5) - 1
      signal   = ret_5d.clip(-0.3,0.3) * (-vol_chg).clip(-1,2)
      signal   = signal.where(ret_5d > 0, 0)
    """
    n = len(stocks)
    arr = np.full(n, np.nan)
    close_d = price_data.get('close', {})
    vol_d = price_data.get('volume', {})
    for i, s in enumerate(stocks):
        if s not in close_d or s not in vol_d:
            continue
        c = np.asarray(close_d[s], dtype=float)
        v = np.asarray(vol_d[s], dtype=float)
        if len(c) < 12 or len(v) < 12:
            continue
        c = c[-12:]; v = v[-12:]
        c0 = c[-6]
        if not np.isfinite(c0) or c0 == 0:
            continue
        r = c[-1] / c0 - 1.0
        if not np.isfinite(r):
            continue
        m5 = np.mean(v[-5:])
        m5p = np.mean(v[-11:-6]) if len(v) >= 11 else np.nan
        if not np.isfinite(m5p) or m5p == 0:
            vc = 0.0
        else:
            vc = m5 / m5p - 1.0
        sig = np.clip(r, -0.3, 0.3) * np.clip(-vc, -1.0, 2.0)
        if r <= 0:
            sig = 0.0
        arr[i] = sig
    valid = np.isfinite(arr)
    return arr, valid

def compute_ep(stocks, context):
    """Earnings-to-Price = 1/pe_ratio. Higher EP = value = positive signal."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    try:
        q = query(valuation.code, valuation.pe_ratio).filter(
            valuation.code.in_(stocks))
        df = get_fundamentals(q, date=context.previous_date)
    except:
        return arr, np.zeros(n, dtype=bool)
    if df is None or len(df) == 0:
        return arr, np.zeros(n, dtype=bool)
    stock_idx = {s: i for i, s in enumerate(stocks)}
    for _, row in df.iterrows():
        code = row['code']
        pe = row.get('pe_ratio')
        if code in stock_idx and pe is not None and not np.isnan(pe) and pe > 0:
            arr[stock_idx[code]] = 1.0 / pe
    valid = np.isfinite(arr)
    return arr, valid

def compute_f_score(stocks, context):
    n = len(stocks)
    arr = np.full(n, np.nan)
    try:
        q = query(
            indicator.code,
            indicator.roa, indicator.roe, indicator.gross_profit_margin,
            indicator.inc_net_profit_year_on_year
        ).filter(indicator.code.in_(stocks))
        df1 = get_fundamentals(q, date=context.previous_date)
    except:
        df1 = None
    try:
        q = query(
            balance.code, balance.current_ratio,
            balance.total_assets, balance.total_liability
        ).filter(balance.code.in_(stocks))
        df2 = get_fundamentals(q, date=context.previous_date)
    except:
        df2 = None
    try:
        q = query(cash_flow.code, cash_flow.net_operate_cash_flow).filter(cash_flow.code.in_(stocks))
        df3 = get_fundamentals(q, date=context.previous_date)
    except:
        df3 = None
    if df1 is None and df2 is None:
        return arr, np.zeros(n, dtype=bool)
    stock_idx = {s: i for i, s in enumerate(stocks)}
    for s in stocks:
        idx = stock_idx[s]
        score = 0
        roa_v = None; gm_v = None; roe_v = None
        cr_v = None; lev_v = None; cfo_v = None
        if df1 is not None:
            row1 = df1[df1['code'] == s]
            if len(row1) > 0:
                roa_v = row1.iloc[0].get('roa')
                gm_v = row1.iloc[0].get('gross_profit_margin')
                roe_v = row1.iloc[0].get('roe')
        if df2 is not None:
            row2 = df2[df2['code'] == s]
            if len(row2) > 0:
                cr_v = row2.iloc[0].get('current_ratio')
                ta = row2.iloc[0].get('total_assets') or 0
                tl = row2.iloc[0].get('total_liability') or 0
                if ta > 0:
                    lev_v = tl / ta
        if df3 is not None:
            row3 = df3[df3['code'] == s]
            if len(row3) > 0:
                cfo_v = row3.iloc[0].get('net_operate_cash_flow')
        if roa_v is not None and roa_v > 0:
            score += 1
        if cfo_v is not None and cfo_v > 0:
            score += 1
        if roe_v is not None and roe_v > 0:
            score += 1
        if gm_v is not None and gm_v > 0:
            score += 1
        if cr_v is not None and cr_v > 1:
            score += 1
        if lev_v is not None and lev_v < 0.7:
            score += 1
        arr[idx] = score
    valid = np.isfinite(arr)
    return arr, valid

def compute_gap_up(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    open_d = price_data.get('open', {})
    close_d = price_data.get('close', {})
    for i, s in enumerate(stocks):
        if s not in open_d or s not in close_d:
            continue
        o = np.array(open_d[s], dtype=float)
        c = np.array(close_d[s], dtype=float)
        if len(o) < 2 or len(c) < 2:
            continue
        o_today = o[-1]
        c_prev = c[-2]
        if np.isnan(o_today) or np.isnan(c_prev) or c_prev == 0:
            continue
        arr[i] = o_today / c_prev - 1
    valid = np.isfinite(arr)
    return arr, valid

def compute_gross_margin(stocks, context):
    n = len(stocks)
    arr = np.full(n, np.nan)
    try:
        q = query(indicator.code, indicator.gross_profit_margin)
        q = q.filter(indicator.code.in_(stocks))
        df = get_fundamentals(q, date=context.previous_date)
    except:
        log.error('[gm] FAILED: %s' % str(e))
        return arr, np.zeros(n, dtype=bool)
    if df is None or len(df) == 0:
        return arr, np.zeros(n, dtype=bool)
    stock_idx = {s: i for i, s in enumerate(stocks)}
    for _, row in df.iterrows():
        code = row['code']
        gm = row.get('gross_profit_margin')
        if code in stock_idx and gm is not None and not np.isnan(gm):
            arr[stock_idx[code]] = gm
    valid = np.isfinite(arr)
    return arr, valid

def compute_high_52w_dist(stocks, price_data):
    high = price_data.get('high', {})
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        h = high.get(s, np.array([]))
        c = close[s]
        if len(h) >= 252 and len(c) >= 1:
            h52 = np.nanmax(h[-252:])
            if h52 > 0:
                arr[i] = (c[-1] - h52) / h52
    valid = np.isfinite(arr)

def compute_high_52w_rank(stocks, price_data):
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 252:
            h52 = np.nanmax(c[-252:])
            l52 = np.nanmin(c[-252:])
            if h52 > l52:
                arr[i] = (c[-1] - l52) / (h52 - l52)
    valid = np.isfinite(arr)

def compute_high_low_range(stocks, price_data):
    """20d avg (H-L)/close, negated. Narrow range = positive signal."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    hd = price_data.get('high', {})
    ld = price_data.get('low', {})
    cd = price_data.get('close', {})
    for i, s in enumerate(stocks):
        if s not in hd or s not in ld or s not in cd:
            continue
        h = np.array(hd[s], dtype=float)
        l = np.array(ld[s], dtype=float)
        c = np.array(cd[s], dtype=float)
        min_len = min(len(h), len(l), len(c))
        if min_len < 10:
            continue
        h_arr = h[-min_len:]
        l_arr = l[-min_len:]
        c_arr = c[-min_len:]
        daily_range = (h_arr - l_arr) / c_arr
        valid_range = daily_range[np.isfinite(daily_range) & (c_arr > 0)]
        if len(valid_range) < 10:
            continue
        roll = valid_range[-20:] if len(valid_range) >= 20 else valid_range
        avg_range = np.nanmean(roll)
        if np.isnan(avg_range):
            continue
        arr[i] = -avg_range
    valid = np.isfinite(arr)
    return arr, valid

def compute_idio_vol(stocks, price_data):
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 63:
            rets = c[-63:] / np.roll(c[-63:], 1) - 1
            rets[0] = 0
            if hasattr(price_data, 'market_ret') and price_data.market_ret is not None:
                mkt = price_data.market_ret[-63:]
                resid = rets - mkt
            else:
                resid = rets - np.nanmean(rets)
            arr[i] = np.nanstd(resid) * np.sqrt(252)
    valid = np.isfinite(arr)

def compute_intraday_reversal(stocks, price_data):
    open_p = price_data.get('open', {})
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        o = open_p.get(s, np.array([]))
        c = close[s]
        if len(o) >= 1 and len(c) >= 1:
            arr[i] = -(c[-1] - o[-1]) / (o[-1] + 0.001)  # 负值=日内跌→明天反转涨
    valid = np.isfinite(arr)

def compute_kurtosis_20(stocks, price_data):
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 20:
            rets = c[-20:] / np.roll(c[-20:], 1) - 1
            rets[0] = 0
            std = np.nanstd(rets)
            if std > 0:
                arr[i] = np.nanmean((rets - np.nanmean(rets))**4) / (std**4) - 3
    valid = np.isfinite(arr)

def compute_ln_circulating_mcap(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    if hasattr(price_data, 'circ_mcap') and price_data.circ_mcap is not None:
        for i, s in enumerate(stocks):
            if s in price_data.circ_mcap:
                arr[i] = np.log(price_data.circ_mcap[s] + 1)
    valid = np.isfinite(arr)

def compute_ln_mcap(stocks, price_data):
    # 需要市值数据, 在 notebook 中从 get_fundamentals 获取
    n = len(stocks)
    arr = np.full(n, np.nan)
    if hasattr(price_data, 'mcap') and price_data.mcap is not None:
        for i, s in enumerate(stocks):
            if s in price_data.mcap:
                arr[i] = np.log(price_data.mcap[s] + 1)
    valid = np.isfinite(arr)

def compute_max_drawdown_duration(stocks, price_data):
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 60:
            peak = c[-60]; trough = c[-60]; dd_days = 0; max_dd_days = 0
            for p in c[-59:]:
                if p >= peak:
                    peak = p; trough = p; dd_days = 0
                else:
                    if p < trough:
                        trough = p
                    dd_days += 1
                    max_dd_days = max(max_dd_days, dd_days)
            arr[i] = -max_dd_days / 60  # 负值=回撤短=更高分
    valid = np.isfinite(arr)

def compute_max_ret_1m(stocks, price_data):
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 21:
            daily_ret = c[-21:] / np.roll(c[-21:], 1) - 1
            daily_ret[0] = 0
            arr[i] = np.nanmax(daily_ret)
    valid = np.isfinite(arr)

def compute_max_ret_1m_neut(stocks, price_data):
    arr, valid = compute_max_ret_1m(stocks, price_data)
    return arr, valid  # TODO: add sector neutralization

def compute_money_flow_20(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    high_d = price_data.get('high', {})
    low_d = price_data.get('low', {})
    close_d = price_data.get('close', {})
    vol_d = price_data.get('volume', {})
    for i, s in enumerate(stocks):
        if s not in high_d or s not in low_d or s not in close_d or s not in vol_d:
            continue
        h = np.array(high_d[s], dtype=float)
        l = np.array(low_d[s], dtype=float)
        c = np.array(close_d[s], dtype=float)
        v = np.array(vol_d[s], dtype=float)
        min_len = min(len(h), len(l), len(c), len(v))
        if min_len < 21:
            continue
        h = h[-min_len:]; l = l[-min_len:]; c = c[-min_len:]; v = v[-min_len:]
        tp = (h + l + c) / 3.0
        mf = tp * v
        mf_ma = np.mean(mf[-20:])
        if mf_ma == 0:
            continue
        arr[i] = -(mf[-1] / mf_ma - 1.0)
    valid = np.isfinite(arr)
    return arr, valid

def compute_net_margin(stocks, context):
    n = len(stocks)
    arr = np.full(n, np.nan)
    try:
        q = query(indicator.code, indicator.net_profit_margin)
        q = q.filter(indicator.code.in_(stocks))
        df = get_fundamentals(q, date=context.previous_date)
    except:
        return arr, np.zeros(n, dtype=bool)
    if df is None or len(df) == 0:
        return arr, np.zeros(n, dtype=bool)
    stock_idx = {s: i for i, s in enumerate(stocks)}
    for _, row in df.iterrows():
        code = row['code']
        val = row.get('net_profit_margin')
        if code in stock_idx and val is not None and not np.isnan(val):
            arr[stock_idx[code]] = val
    valid = np.isfinite(arr)
    return arr, valid

def compute_ocf_quality(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    if hasattr(price_data, 'fundamentals') and price_data.fundamentals is not None:
        for i, s in enumerate(stocks):
            f = price_data.fundamentals.get(s, {})
            ocf = f.get('ocf', np.nan)
            ni = f.get('net_income', np.nan)
            if np.isfinite(ocf) and np.isfinite(ni) and abs(ni) > 1e6:
                arr[i] = ocf / ni
    valid = np.isfinite(arr)

def compute_opening_gap_momentum(stocks, price_data):
    open_p = price_data.get('open', {})
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        o = open_p.get(s, np.array([]))
        c = close[s]
        if len(o) >= 2 and len(c) >= 2:
            arr[i] = o[-1] / c[-2] - 1  # 今日开盘相对昨收的缺口
    valid = np.isfinite(arr)

def compute_operational_efficiency_proxy(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    return arr, np.full(n, False)  # Not implemented on JQ

def compute_panic_selling(stocks, price_data):
    volume = price_data.get('volume', {})
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        v = volume.get(s, np.array([]))
        c = close[s]
        if len(v) >= 20 and len(c) >= 20:
            ret = c[-1] / c[-2] - 1
            avg_v = np.nanmean(v[-20:])
            if avg_v > 0 and ret < -0.02:
                arr[i] = -(v[-1] / avg_v)  # 放量下跌=负因子(恐慌), 取负后高分=非恐慌
    valid = np.isfinite(arr)

def compute_plrc_12(stocks, price_data):
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 12:
            up = np.sum(np.maximum(np.diff(c[-13:]), 0))
            down = np.sum(np.abs(np.minimum(np.diff(c[-13:]), 0)))
            total = up + down
            if total > 0:
                arr[i] = up / total
    valid = np.isfinite(arr)

def compute_post_earnings_stability(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    return arr, np.full(n, False)  # Not implemented on JQ

def compute_price_1m(stocks, price_data):
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 21:
            arr[i] = c[-1] / c[-21] - 1
    valid = np.isfinite(arr)

def compute_psy_12(stocks, price_data):
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 12:
            up_days = np.sum(np.diff(c[-13:]) > 0)
            arr[i] = up_days / 12 * 100
    valid = np.isfinite(arr)

def compute_range_consistency(stocks, price_data):
    """Range CV over 20d, negated. Consistent range = positive signal."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    hd = price_data.get('high', {})
    ld = price_data.get('low', {})
    for i, s in enumerate(stocks):
        if s not in hd or s not in ld:
            continue
        h = np.array(hd[s], dtype=float)
        l = np.array(ld[s], dtype=float)
        min_len = min(len(h), len(l))
        if min_len < 10:
            continue
        rng = np.abs(h[-min_len:] - l[-min_len:])
        roll = rng[-20:] if len(rng) >= 20 else rng
        std_r = np.nanstd(roll)
        mean_r = np.nanmean(roll)
        if mean_r == 0:
            continue
        arr[i] = -std_r / mean_r
    valid = np.isfinite(arr)
    return arr, valid

def compute_rank_1m(stocks, price_data):
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    rets = []
    idxs = []
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 21:
            rets.append(c[-1] / c[-21] - 1)
            idxs.append(i)
    if rets:
        from scipy.stats import rankdata
        ranks = rankdata(rets)
        for j, idx in enumerate(idxs):
            arr[idx] = ranks[j] / len(rets)
    valid = np.isfinite(arr)

def compute_relative_spread_proxy(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    high_d = price_data.get('high', {})
    low_d = price_data.get('low', {})
    close_d = price_data.get('close', {})
    for i, s in enumerate(stocks):
        if s not in high_d or s not in low_d or s not in close_d:
            continue
        h = np.array(high_d[s], dtype=float)
        l = np.array(low_d[s], dtype=float)
        c = np.array(close_d[s], dtype=float)
        if len(h) < 1 or len(l) < 1 or len(c) < 1:
            continue
        hv = h[-1]; lv = l[-1]; cv = c[-1]
        if np.isnan(hv) or np.isnan(lv) or np.isnan(cv) or cv == 0:
            continue
        arr[i] = -(hv - lv) / (cv + 0.001)
    valid = np.isfinite(arr)
    return arr, valid

def compute_ret_12m(stocks, price_data):
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 252:
            arr[i] = c[-1] / c[-252] - 1
    valid = np.isfinite(arr)

def compute_ret_1m(stocks, price_data):
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 21:
            arr[i] = c[-1] / c[-21] - 1
    valid = np.isfinite(arr)

def compute_ret_1m_skip1m(stocks, price_data):
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 42:
            arr[i] = c[-22] / c[-42] - 1
    valid = np.isfinite(arr)

def compute_ret_3m(stocks, price_data):
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 63:
            arr[i] = c[-1] / c[-63] - 1
    valid = np.isfinite(arr)

def compute_ret_3m_neut(stocks, price_data):
    arr, valid = compute_ret_3m(stocks, price_data)
    return arr, valid  # TODO: add sector neutralization

def compute_ret_6m(stocks, price_data):
    """6-month return reversal. Negative past return = positive signal."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    cd = price_data.get('close', {})
    for i, s in enumerate(stocks):
        if s not in cd:
            continue
        c = np.array(cd[s], dtype=float)
        c_clean = c[np.isfinite(c) & (c > 0)]
        if len(c_clean) < 131:
            continue
        ret = c_clean[-1] / c_clean[-126] - 1.0
        arr[i] = -ret
    valid = np.isfinite(arr)
    return arr, valid

def compute_ret_open_2d_proxy(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    open_d = price_data.get('open', {})
    close_d = price_data.get('close', {})
    for i, s in enumerate(stocks):
        if s not in open_d or s not in close_d:
            continue
        o = np.array(open_d[s], dtype=float)
        c = np.array(close_d[s], dtype=float)
        if len(o) < 3 or len(c) < 3:
            continue
        vals = []
        for t in range(-2, 0):
            ot = o[t]; cp = c[t - 1]
            if np.isnan(ot) or np.isnan(cp) or cp == 0:
                continue
            vals.append(ot / cp - 1)
        if len(vals) == 0:
            continue
        arr[i] = np.mean(vals)
    valid = np.isfinite(arr)
    return arr, valid

def compute_rev_growth_yoy(stocks, context):
    """营收同比增长率 (or_yoy). 取最新财报快照, 值越高=成长越强=正向."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    try:
        q = query(indicator.code, indicator.inc_revenue_year_on_year).filter(
            indicator.code.in_(stocks))
        df = get_fundamentals(q, date=context.previous_date)
    except:
        return arr, np.zeros(n, dtype=bool)
    if df is None or len(df) == 0:
        return arr, np.zeros(n, dtype=bool)
    stock_idx = {s: i for i, s in enumerate(stocks)}
    for _, row in df.iterrows():
        code = row['code']
        val = row.get('inc_revenue_year_on_year')
        if code in stock_idx and val is not None and np.isfinite(val):
            arr[stock_idx[code]] = val
    valid = np.isfinite(arr)
    return arr, valid

def compute_roa(stocks, context):
    n = len(stocks)
    arr = np.full(n, np.nan)
    try:
        q = query(indicator.code, indicator.roa)
        q = q.filter(indicator.code.in_(stocks))
        df = get_fundamentals(q, date=context.previous_date)
    except:
        return arr, np.zeros(n, dtype=bool)
    if df is None or len(df) == 0:
        return arr, np.zeros(n, dtype=bool)
    stock_idx = {s: i for i, s in enumerate(stocks)}
    for _, row in df.iterrows():
        code = row['code']
        val = row.get('roa')
        if code in stock_idx and val is not None and not np.isnan(val):
            arr[stock_idx[code]] = val
    valid = np.isfinite(arr)
    return arr, valid

def compute_roc_20(stocks, price_data):
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 20:
            arr[i] = (c[-1] - c[-20]) / c[-20] * 100
    valid = np.isfinite(arr)

def compute_roe(stocks, context):
    n = len(stocks)
    arr = np.full(n, np.nan)
    try:
        q = query(indicator.code, indicator.roe)
        q = q.filter(indicator.code.in_(stocks))
        df = get_fundamentals(q, date=context.previous_date)
    except:
        log.error('[roe] FAILED: %s' % str(e))
        return arr, np.zeros(n, dtype=bool)
    if df is None or len(df) == 0:
        return arr, np.zeros(n, dtype=bool)
    stock_idx = {s: i for i, s in enumerate(stocks)}
    for _, row in df.iterrows():
        code = row['code']
        val = row.get('roe')
        if code in stock_idx and val is not None and not np.isnan(val):
            arr[stock_idx[code]] = val
    valid = np.isfinite(arr)
    return arr, valid

def compute_roic(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    if hasattr(price_data, 'fundamentals') and price_data.fundamentals is not None:
        for i, s in enumerate(stocks):
            f = price_data.fundamentals.get(s, {})
            nopat = f.get('nopat', np.nan)
            ic = f.get('invested_capital', np.nan)
            if np.isfinite(nopat) and np.isfinite(ic) and ic > 0:
                arr[i] = nopat / ic
    valid = np.isfinite(arr)

def compute_rsi_14(stocks, price_data):
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 15:
            delta = np.diff(c[-15:])
            gain = np.nanmean(delta[delta > 0]) if np.any(delta > 0) else 0
            loss = -np.nanmean(delta[delta < 0]) if np.any(delta < 0) else 1e-10
            rs = gain / loss if loss > 0 else 100
            arr[i] = 100 - 100 / (1 + rs)
    valid = np.isfinite(arr)

def compute_scc_network_centrality(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    return arr, np.full(n, False)  # Not implemented on JQ

def compute_sharpe_20(stocks, price_data):
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 20:
            rets = c[-20:] / np.roll(c[-20:], 1) - 1
            rets[0] = 0
            std = np.nanstd(rets)
            if std > 0:
                arr[i] = np.nanmean(rets) / std
    valid = np.isfinite(arr)

def compute_short_rev_5d(stocks, price_data):
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 5:
            arr[i] = -(c[-1] / c[-5] - 1)  # 负收益=正因子 (反转)
    valid = np.isfinite(arr)

def compute_skew_1m(stocks, price_data):
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 21:
            rets = c[-21:] / np.roll(c[-21:], 1) - 1
            rets[0] = 0
            std = np.nanstd(rets)
            if std > 0:
                arr[i] = np.nanmean((rets - np.nanmean(rets))**3) / (std**3)
    valid = np.isfinite(arr)

def compute_skewness_20(stocks, price_data):
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 20:
            rets = c[-20:] / np.roll(c[-20:], 1) - 1
            rets[0] = 0
            std = np.nanstd(rets)
            if std > 0:
                arr[i] = np.nanmean((rets - np.nanmean(rets))**3) / (std**3)
    valid = np.isfinite(arr)

def compute_sp(stocks, context):
    """S/P = 1/ps_ratio. 销售/价格比, 越高越便宜=价值正向信号."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    try:
        q = query(valuation.code, valuation.ps_ratio).filter(
            valuation.code.in_(stocks))
        df = get_fundamentals(q, date=context.previous_date)
    except:
        return arr, np.zeros(n, dtype=bool)
    if df is None or len(df) == 0:
        return arr, np.zeros(n, dtype=bool)
    stock_idx = {s: i for i, s in enumerate(stocks)}
    for _, row in df.iterrows():
        code = row['code']
        ps = row.get('ps_ratio')
        if code in stock_idx and ps is not None and np.isfinite(ps) and ps > 0:
            arr[stock_idx[code]] = 1.0 / ps
    valid = np.isfinite(arr)
    return arr, valid

def compute_streak(stocks, price_data):
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 2:
            if c[-1] > c[-2]:
                streak = 1
                for j in range(-3, -1):
                    if j >= -len(c) and c[j] > c[j-1]:
                        streak += 1
                    else:
                        break
                arr[i] = streak
            elif c[-1] < c[-2]:
                streak = -1
                for j in range(-3, -1):
                    if j >= -len(c) and c[j] < c[j-1]:
                        streak -= 1
                    else:
                        break
                arr[i] = streak
            else:
                arr[i] = 0
    valid = np.isfinite(arr)

def compute_trend_persistence_score(stocks, price_data):
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 60:
            up = np.sum(np.diff(c[-60:]) > 0)
            down = np.sum(np.diff(c[-60:]) < 0)
            total = up + down
            if total > 0:
                consec = 0; max_consec = 0
                prev_up = None
                for j in range(-60+1, 0):
                    curr_up = c[j] > c[j-1]
                    if prev_up is not None and curr_up == prev_up:
                        consec += 1
                    else:
                        consec = 1
                    max_consec = max(max_consec, consec)
                    prev_up = curr_up
                arr[i] = (up - down) / total * max_consec / 10
    valid = np.isfinite(arr)

def compute_trend_smoothness(stocks, price_data):
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 60:
            rets = c[-60:] / np.roll(c[-60:], 1) - 1; rets[0] = 0
            sign_changes = np.sum(np.diff(np.sign(rets[10:])) != 0)
            arr[i] = -sign_changes / 50  # 负值=更平滑=更高分
    valid = np.isfinite(arr)

def compute_turnover_change(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    vol_d = price_data.get('volume', {})
    for i, s in enumerate(stocks):
        if s not in vol_d:
            continue
        v = np.array(vol_d[s], dtype=float)
        v = v[np.isfinite(v) & (v > 0)]
        if len(v) < 10:
            continue
        avg5 = np.mean(v[-5:])
        avg20 = np.mean(v[-20:])
        if avg20 == 0 or np.isnan(avg20):
            continue
        change = avg5 / avg20 - 1
        arr[i] = -change
    valid = np.isfinite(arr)
    return arr, valid

def compute_turnover_cv_20d(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    vol_d = price_data.get('volume', {})
    for i, s in enumerate(stocks):
        if s not in vol_d:
            continue
        v = np.array(vol_d[s], dtype=float)
        v_arr = v[np.isfinite(v) & (v > 0)]
        if len(v_arr) < 10:
            continue
        roll = v_arr[-20:] if len(v_arr) >= 20 else v_arr
        std_v = np.nanstd(roll)
        mean_v = np.nanmean(roll)
        if mean_v == 0:
            continue
        cv = std_v / mean_v
        arr[i] = -cv
    valid = np.isfinite(arr)
    return arr, valid

def compute_turnover_std(stocks, context):
    n = len(stocks)
    arr = np.full(n, np.nan)
    # 1. 取当日 turnover_rate 快照 (与 gross_margin 同款已验证 get_fundamentals 模式)
    try:
        q = query(valuation.code, valuation.turnover_rate)
        q = q.filter(valuation.code.in_(stocks))
        df = get_fundamentals(q, date=context.previous_date)
    except:
        log.error('[tstd] FAILED: %s' % str(e))
        return arr, np.zeros(n, dtype=bool)
    if df is None or len(df) == 0:
        return arr, np.zeros(n, dtype=bool)
    snap = {}
    for _, row in df.iterrows():
        code = row['code']
        tr = row.get('turnover_rate')
        if tr is not None and not (isinstance(tr, float) and np.isnan(tr)):
            snap[code] = float(tr)
    # 2. 维护滚动历史 (每周一个快照)
    if not hasattr(g, 'turnover_rate_hist') or g.turnover_rate_hist is None:
        g.turnover_rate_hist = {}
    for s in stocks:
        hist = g.turnover_rate_hist.setdefault(s, [])
        if s in snap:
            hist.append(snap[s])
        if len(hist) > 20:
            hist[:] = hist[-20:]
    # 3. 计算近20周 std 取负
    stock_idx = {s: i for i, s in enumerate(stocks)}
    valid = np.zeros(n, dtype=bool)
    for s, v in snap.items():
        if s in stock_idx:
            hist = g.turnover_rate_hist.get(s, [])
            hv = [x for x in hist if x is not None and not np.isnan(x)]
            if len(hv) >= 10:
                arr[stock_idx[s]] = -np.std(hv, ddof=1)
                valid[stock_idx[s]] = True
    return arr, valid

def compute_turnover_to_vol(stocks, price_data):
    volume = price_data.get('volume', {})
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        v = volume.get(s, np.array([]))
        c = close[s]
        if len(v) >= 21 and len(c) >= 21:
            tv = np.nanmean(v[-21:])
            rets = c[-21:] / np.roll(c[-21:], 1) - 1; rets[0] = 0
            vol = np.nanstd(rets)
            if vol > 0:
                arr[i] = tv / vol
    valid = np.isfinite(arr)

def compute_tvma_20(stocks, price_data):
    volume = price_data.get('volume', {})
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        v = volume.get(s, np.array([]))
        c = close[s]
        if len(v) >= 20 and len(c) >= 20:
            total_v = np.nansum(v[-20:])
            if total_v > 0:
                w_price = np.nansum(v[-20:] * c[-20:]) / total_v
                arr[i] = c[-1] / w_price - 1
    valid = np.isfinite(arr)

def compute_vm_diff(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    vol_d = price_data.get('volume', {})
    for i, s in enumerate(stocks):
        if s not in vol_d:
            continue
        v = np.array(vol_d[s], dtype=float)
        v_arr = v[np.isfinite(v)]
        if len(v_arr) < 26:
            continue
        span12 = 12; span26 = 26
        alpha12 = 2.0 / (span12 + 1); alpha26 = 2.0 / (span26 + 1)
        ema12 = v_arr[0]; ema26 = v_arr[0]
        for t in range(1, len(v_arr)):
            ema12 = alpha12 * v_arr[t] + (1 - alpha12) * ema12
            ema26 = alpha26 * v_arr[t] + (1 - alpha26) * ema26
        if ema26 == 0:
            continue
        diff_norm = ema12 / ema26 - 1
        arr[i] = -np.clip(diff_norm, -5, 5)
    valid = np.isfinite(arr)
    return arr, valid

def compute_vol_1m(stocks, price_data):
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 21:
            rets = c[-21:] / np.roll(c[-21:], 1) - 1
            rets[0] = 0
            arr[i] = np.nanstd(rets) * np.sqrt(252)
    valid = np.isfinite(arr)

def compute_vol_1m_neut(stocks, price_data):
    arr, valid = compute_vol_1m(stocks, price_data)
    return arr, valid  # TODO: add sector neutralization

def compute_vol_3m(stocks, price_data):
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 63:
            rets = c[-63:] / np.roll(c[-63:], 1) - 1
            rets[0] = 0
            arr[i] = np.nanstd(rets) * np.sqrt(252)
    valid = np.isfinite(arr)

def compute_vol_3m_neut(stocks, price_data):
    arr, valid = compute_vol_3m(stocks, price_data)
    return arr, valid  # TODO: add sector neutralization

def compute_volatility_of_volatility(stocks, price_data):
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 63:
            rets = c[-63:] / np.roll(c[-63:], 1) - 1
            rets[0] = 0
            rolling_vol = np.array([np.nanstd(rets[max(0,j-20):j+1]) for j in range(20, len(rets))])
            if len(rolling_vol) > 0:
                arr[i] = np.nanstd(rolling_vol)
    valid = np.isfinite(arr)

def compute_volume_climax_reversal(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    vol_d = price_data.get('volume', {})
    for i, s in enumerate(stocks):
        if s not in vol_d:
            continue
        v = np.array(vol_d[s], dtype=float)
        v_arr = v[np.isfinite(v)]
        if len(v_arr) < 21:
            continue
        vol_ma = np.mean(v_arr[-21:-1])
        if vol_ma == 0:
            continue
        vol_ratio = v_arr[-2] / vol_ma - 1.0
        if vol_ratio > 0:
            arr[i] = -vol_ratio
        else:
            arr[i] = 0.0
    valid = np.isfinite(arr)
    return arr, valid

def compute_volume_ratio(stocks, price_data):
    volume = price_data.get('volume', {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        v = volume.get(s, np.array([]))
        if len(v) >= 6:
            ma5 = np.nanmean(v[-6:-1])
            if ma5 > 0:
                arr[i] = v[-1] / ma5
    valid = np.isfinite(arr)

def compute_volume_stability(stocks, price_data):
    """Volume CV over 20d, negated. Stable volume = positive signal."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    vd = price_data.get('volume', {})
    for i, s in enumerate(stocks):
        if s not in vd:
            continue
        v = np.array(vd[s], dtype=float)
        v_clean = v[np.isfinite(v) & (v > 0)]
        if len(v_clean) < 10:
            continue
        roll = v_clean[-20:] if len(v_clean) >= 20 else v_clean
        std_v = np.nanstd(roll)
        mean_v = np.nanmean(roll)
        if mean_v == 0:
            continue
        arr[i] = -std_v / mean_v
    valid = np.isfinite(arr)
    return arr, valid

def compute_vpt(stocks, price_data):
    volume = price_data.get('volume', {})
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        v = volume.get(s, np.array([]))
        c = close[s]
        if len(v) >= 26 and len(c) >= 27:
            chg = c[-26:] / np.roll(c[-26:], 1) - 1; chg[0] = 0
            arr[i] = np.nansum(v[-26:] * chg)
    valid = np.isfinite(arr)

def compute_vr_26(stocks, price_data):
    volume = price_data.get('volume', {})
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        v = volume.get(s, np.array([]))
        c = close[s]
        if len(v) >= 26 and len(c) >= 26:
            up_vol = np.sum(v[-26:][np.diff(c[-27:-1]) > 0]) if len(c) >= 27 else 0
            down_vol = np.sum(v[-26:][np.diff(c[-27:-1]) < 0]) if len(c) >= 27 else 1e-10
            if down_vol > 0:
                arr[i] = up_vol / down_vol
    valid = np.isfinite(arr)

def compute_vroc_12(stocks, price_data):
    volume = price_data.get('volume', {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        v = volume.get(s, np.array([]))
        if len(v) >= 12 and v[-12] > 0:
            arr[i] = (v[-1] - v[-12]) / v[-12]
    valid = np.isfinite(arr)

# 所有已实现的因子: 107
ALL_FACTOR_NAMES = ['abnormal_turnover', 'abnormal_turnover_neut', 'accrual_quality_proxy', 'accruals', 'amihud_illiq', 'asset_growth', 'asset_turnover', 'atr_14', 'atr_14_neut', 'attention_decay', 'avg_turnover_1m', 'avg_turnover_1m_neut', 'avg_turnover_3m', 'bargaining_power_proxy', 'bear_power', 'beta', 'bias_20', 'boll_pct_b', 'bp', 'bp_neut', 'bull_power', 'capital_efficiency_proxy', 'cashflow_matching_proxy', 'cci_14', 'cfp', 'davol_20', 'davol_20_neut', 'debt_coverage', 'dollar_vol_20d', 'dollar_vol_20d_neut', 'dollar_vol_stability', 'downside_vol', 'downside_vol_neut', 'dp', 'earnings_consistency_proxy', 'earnings_growth_yoy', 'earnings_quality_proxy', 'earnings_season_vol_div', 'earnings_stability', 'earnings_volume_drift', 'ep', 'f_score', 'gap_up', 'gross_margin', 'high_52w_dist', 'high_52w_rank', 'high_low_range', 'idio_vol', 'intraday_reversal', 'kurtosis_20', 'ln_circulating_mcap', 'ln_mcap', 'max_drawdown_duration', 'max_ret_1m', 'max_ret_1m_neut', 'money_flow_20', 'net_margin', 'ocf_quality', 'opening_gap_momentum', 'operational_efficiency_proxy', 'panic_selling', 'plrc_12', 'post_earnings_stability', 'price_1m', 'psy_12', 'range_consistency', 'rank_1m', 'relative_spread_proxy', 'ret_12m', 'ret_1m', 'ret_1m_skip1m', 'ret_3m', 'ret_3m_neut', 'ret_6m', 'ret_open_2d_proxy', 'rev_growth_yoy', 'roa', 'roc_20', 'roe', 'roic', 'rsi_14', 'scc_network_centrality', 'sharpe_20', 'short_rev_5d', 'skew_1m', 'skewness_20', 'sp', 'streak', 'trend_persistence_score', 'trend_smoothness', 'turnover_change', 'turnover_cv_20d', 'turnover_std', 'turnover_to_vol', 'tvma_20', 'vm_diff', 'vol_1m', 'vol_1m_neut', 'vol_3m', 'vol_3m_neut', 'volatility_of_volatility', 'volume_climax_reversal', 'volume_ratio', 'volume_stability', 'vpt', 'vr_26', 'vroc_12']
FACTOR_DIRECTIONS = {'relative_spread_proxy': 'positive', 'gap_up': 'positive', 'turnover_change': 'positive', 'vm_diff': 'positive', 'ret_open_2d_proxy': 'positive', 'accrual_quality_proxy': 'positive', 'volume_climax_reversal': 'positive', 'money_flow_20': 'positive', 'dollar_vol_20d': 'positive', 'turnover_cv_20d': 'positive', 'turnover_std': 'positive', 'gross_margin': 'positive', 'earnings_growth_yoy': 'positive', 'roe': 'positive', 'roa': 'positive', 'net_margin': 'positive', 'f_score': 'positive', 'abnormal_turnover': 'positive', 'abnormal_turnover_neut': 'positive', 'accruals': 'positive', 'amihud_illiq': 'positive', 'asset_growth': 'positive', 'asset_turnover': 'positive', 'atr_14': 'positive', 'atr_14_neut': 'positive', 'attention_decay': 'positive', 'avg_turnover_1m': 'positive', 'avg_turnover_1m_neut': 'positive', 'avg_turnover_3m': 'positive', 'bargaining_power_proxy': 'positive', 'bear_power': 'positive', 'beta': 'positive', 'bias_20': 'positive', 'boll_pct_b': 'positive', 'bp': 'positive', 'bp_neut': 'positive', 'bull_power': 'positive', 'capital_efficiency_proxy': 'positive', 'cashflow_matching_proxy': 'positive', 'cci_14': 'positive', 'cfp': 'positive', 'davol_20': 'positive', 'davol_20_neut': 'positive', 'debt_coverage': 'positive', 'dollar_vol_20d_neut': 'positive', 'dollar_vol_stability': 'positive', 'downside_vol': 'positive', 'downside_vol_neut': 'positive', 'dp': 'positive', 'earnings_consistency_proxy': 'positive', 'earnings_quality_proxy': 'positive', 'earnings_season_vol_div': 'positive', 'earnings_stability': 'positive', 'earnings_volume_drift': 'positive', 'ep': 'positive', 'high_52w_dist': 'positive', 'high_52w_rank': 'positive', 'high_low_range': 'positive', 'idio_vol': 'positive', 'intraday_reversal': 'positive', 'kurtosis_20': 'positive', 'ln_circulating_mcap': 'positive', 'ln_mcap': 'positive', 'max_drawdown_duration': 'positive', 'max_ret_1m': 'positive', 'max_ret_1m_neut': 'positive', 'ocf_quality': 'positive', 'opening_gap_momentum': 'positive', 'operational_efficiency_proxy': 'positive', 'panic_selling': 'positive', 'plrc_12': 'positive', 'post_earnings_stability': 'positive', 'price_1m': 'positive', 'psy_12': 'positive', 'range_consistency': 'positive', 'rank_1m': 'positive', 'ret_12m': 'positive', 'ret_1m': 'positive', 'ret_1m_skip1m': 'positive', 'ret_3m': 'positive', 'ret_3m_neut': 'positive', 'ret_6m': 'positive', 'rev_growth_yoy': 'positive', 'roc_20': 'positive', 'roic': 'positive', 'rsi_14': 'positive', 'scc_network_centrality': 'positive', 'sharpe_20': 'positive', 'short_rev_5d': 'positive', 'skew_1m': 'positive', 'skewness_20': 'positive', 'sp': 'positive', 'streak': 'positive', 'trend_persistence_score': 'positive', 'trend_smoothness': 'positive', 'turnover_to_vol': 'positive', 'tvma_20': 'positive', 'vol_1m': 'positive', 'vol_1m_neut': 'positive', 'vol_3m': 'positive', 'vol_3m_neut': 'positive', 'volatility_of_volatility': 'positive', 'volume_ratio': 'positive', 'volume_stability': 'positive', 'vpt': 'positive', 'vr_26': 'positive', 'vroc_12': 'positive'}

# Cell 6: Regime 检测
def detect_regime(week_str):
    """CSI500 vs CSI300 风格 + 全A趋势 → regime"""
    for name in INDEX_CODES:
        pass
    
    csi300_rets = []
    csi500_rets = []
    csi_all_rets = []
    
    for name, data in index_data.items():
        w_ret = data["close"].resample("W").last().pct_change(4).dropna()
        idx = w_ret.index.strftime("%Y-%m-%d")
        if name == "CSI300":
            csi300_rets = w_ret
        elif name == "CSI500":
            csi500_rets = w_ret
        else:
            csi_all_rets = w_ret
    
    # 找到当前周对应的指数收益
    week_date = pd.Timestamp(week_str)
    
    if len(csi500_rets) == 0 or len(csi300_rets) == 0:
        return "large_dominant"
    
    style_spread = None; trend = None
    for i, dt in enumerate(csi500_rets.index):
        if dt >= week_date:
            if i >= 4:
                style_spread = csi500_rets.iloc[i] - csi300_rets.iloc[i]
            if i >= 4 and len(csi_all_rets) > i:
                trend = csi_all_rets.iloc[i]
            break
    
    if style_spread is None:
        return "large_dominant"
    
    if style_spread > STYLE_SPREAD_THRESHOLD and (trend is None or trend > TREND_THRESHOLD):
        return "small_bull"
    return "large_dominant"

# Cell 7: ElasticNetCV per Regime
print("Computing factors and running ElasticNetCV...")

MIN_WEEKS_FOR_REGIME = 20
# 跳过前52周 (无足够回看数据)
valid_weeks = weeks[52:]
print(f"Valid weeks: {len(valid_weeks)}")

# 按 regime 分组
regime_weeks = {"large_dominant": [], "small_bull": []}
for w in valid_weeks:
    r = detect_regime(str(w))
    regime_weeks[r].append(w)
print(f"large_dominant: {len(regime_weeks['large_dominant'])} weeks")
print(f"small_bull: {len(regime_weeks['small_bull'])} weeks")

# 逐周计算因子 + forward return → 构建训练集
def build_training_data(regime_name):
    """为指定 regime 构建 X(factors) 和 y(forward return)"""
    r_weeks = regime_weeks[regime_name]
    if len(r_weeks) < MIN_WEEKS_FOR_REGIME:
        return None, None
    
    X_list, y_list = [], []
    for w in r_weeks[:-1]:  # 最后一周没有 forward return
        next_w_idx = weeks.index(w) + 1
        if next_w_idx >= len(weeks):
            continue
        next_w = weeks[next_w_idx]
        
        # 获取本周的活跃股票
        mask = price_weekly["week"] == w
        current_stocks = list(price_weekly[mask]["code"].unique())
        if len(current_stocks) < 50:
            continue
        
        # 构建 price_data
        pd_slice = build_price_data_slice(str(w), current_stocks[:500])  # 限500只控制内存
        
        # 计算所有因子
        factor_values = np.full((len(current_stocks[:500]), len(ALL_FACTOR_NAMES)), np.nan)
        for j, fname in enumerate(ALL_FACTOR_NAMES):
            try:
                func = globals().get(f"compute_{fname}")
                if func:
                    arr, _ = func(current_stocks[:500], pd_slice)
                    factor_values[:, j] = arr
            except:
                pass
        
        # Forward return
        mask_next = price_weekly["week"] == next_w
        next_prices = price_weekly[mask_next].set_index("code")["close"]
        curr_prices = price_weekly[mask].set_index("code")["close"]
        
        fwd_ret = np.full(len(current_stocks[:500]), np.nan)
        for i, s in enumerate(current_stocks[:500]):
            if s in curr_prices.index and s in next_prices.index:
                if curr_prices[s] > 0:
                    fwd_ret[i] = next_prices[s] / curr_prices[s] - 1
        
        X_list.append(factor_values)
        y_list.append(fwd_ret)
        
        if len(X_list) % 10 == 0:
            print(f"  {regime_name}: {len(X_list)}/{len(r_weeks)} weeks processed")
    
    if len(X_list) == 0:
        return None, None
    
    X = np.vstack(X_list)
    y = np.hstack(y_list)
    valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    return X[valid], y[valid]

# 训练
for regime in ["large_dominant", "small_bull"]:
    print(f"\nTraining ElasticNetCV for {regime}...")
    X, y = build_training_data(regime)
    if X is None or len(X) < 100:
        print(f"  Insufficient data: {len(X) if X is not None else 0} samples")
        continue
    
    print(f"  X: {X.shape}, y: {y.shape}")
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # ElasticNetCV
    enet = ElasticNetCV(
        l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
        cv=5,
        max_iter=5000,
        n_jobs=-1,
        random_state=42
    )
    enet.fit(X_scaled, y)
    
    print(f"  alpha={enet.alpha_:.6f}, l1_ratio={enet.l1_ratio_:.3f}")
    print(f"  R2 (CV): {enet.score(X_scaled, y):.4f}")
    
    # 输出权重
    coef = enet.coef_
    nonzero = np.abs(coef) > 1e-6
    print(f"  Non-zero factors: {nonzero.sum()}/{len(coef)}")
    
    weights = {}
    for j in range(len(coef)):
        if nonzero[j]:
            fname = ALL_FACTOR_NAMES[j]
            w = coef[j] * FACTOR_DIRECTIONS.get(fname, 1)
            weights[fname] = float(w)
            print(f"    {fname}: {w:+.6f}")
    
    print(f"\n  WEIGHTS_{regime.upper()} = {weights}")

print("\n✓ Complete! Copy the WEIGHTS dicts above to local JQGenerator.")