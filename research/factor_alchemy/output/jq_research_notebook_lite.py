"""
JQ 研究笔记本 (精简版): Regime-ElasticNet 因子选择
====================================================
适配 JQ 研究平台 1GB 免费内存。

策略: 500只大市值股票 + 3年窗口(2023-2026) + 30个核心因子
预估内存: ~200MB, 远低于1GB限制

工作流:
  1. 加载500只大市值股票日线
  2. 计算30核心因子
  3. Regime检测
  4. ElasticNetCV per regime → 输出权重
"""

# ============================================================
# Cell 1: Imports & 配置
# ============================================================
import numpy as np
import pandas as pd
try:
    from jqdata import *
except ImportError:
    from jqdatasdk import *
from scipy.stats import spearmanr, zscore
from sklearn.linear_model import ElasticNetCV
import warnings, gc
warnings.filterwarnings("ignore")

# ★ 精简配置 ★
START_DATE   = "2023-01-01"   # 3年窗口 (省内存)
END_DATE     = "2026-06-30"
N_STOCKS     = 500            # Top500市值 (非5000+)
N_FACTORS    = 30             # 核心因子 (非107)
TOP_N        = 30
MAX_TURNOVER = 0.60

STYLE_SPREAD_THRESHOLD = 0.02
TREND_THRESHOLD = -0.01

print(f"✓ Config: {N_STOCKS} stocks, {START_DATE}~{END_DATE}")

# ============================================================
# Cell 2: 获取股票池 — Top500流通市值
# ============================================================
# 先拿全列表，然后按市值排序取Top500
all_secs = get_all_securities(["stock"], date=END_DATE)

pool_candidates = []
for code, row in all_secs.iterrows():
    name = str(row.get("display_name", ""))
    if "ST" in name or "退" in name or "*" in name[:3]:
        continue
    days_listed = (pd.Timestamp(END_DATE) - pd.Timestamp(row["start_date"])).days
    if days_listed < 180:
        continue
    pool_candidates.append(code)

print(f"  Candidates after filter: {len(pool_candidates)}")

# 拿最新一期流通市值排序，取Top500
try:
    q_mcap = query(
        valuation.code, valuation.circulating_market_cap
    ).filter(
        valuation.code.in_(pool_candidates)
    )
    mcap_df = get_fundamentals(q_mcap, statDate="2026-03-31")
    if mcap_df is not None and len(mcap_df) > 0:
        mcap_df = mcap_df.set_index("code")["circulating_market_cap"].sort_values(ascending=False)
        ALL_STOCKS = list(mcap_df.head(N_STOCKS).index)
    else:
        ALL_STOCKS = pool_candidates[:N_STOCKS]
except Exception:
    ALL_STOCKS = pool_candidates[:N_STOCKS]

print(f"✓ Stock pool: {len(ALL_STOCKS)} stocks (Top{N_STOCKS} by market cap)")
print(f"  Sample: {ALL_STOCKS[:5]}")

# ============================================================
# Cell 3: 加载价格数据 (日线 → 周频)
# ============================================================
print("\n[Cell 3] Loading price data...")

price_fields = ["open", "close", "high", "low", "volume", "money"]

# 分批加载 (每批100只, 防OOM)
batch_size = 100
all_price_frames = []
for i in range(0, len(ALL_STOCKS), batch_size):
    batch = ALL_STOCKS[i:i+batch_size]
    print(f"  Batch {i//batch_size+1}/{ (len(ALL_STOCKS)+batch_size-1)//batch_size }: {len(batch)} stocks...", end=" ")
    try:
        df = get_price(
            batch, start_date=START_DATE, end_date=END_DATE,
            frequency="daily", fields=price_fields, fq="post",
            skip_paused=True, panel=False
        )
        if df is not None and len(df) > 0:
            all_price_frames.append(df)
            print(f"{len(df)} rows")
        else:
            print("no data")
    except Exception as e:
        print(f"error: {e}")
    gc.collect()  # ★ 主动GC防内存堆积

raw_price = pd.concat(all_price_frames, ignore_index=True) if all_price_frames else pd.DataFrame()
del all_price_frames; gc.collect()

print(f"✓ Raw: {raw_price.shape[0]} daily rows, {raw_price.code.nunique()} stocks")

# 转周频: JQ get_price(panel=False) 的 time 是列不是索引
raw_price['time'] = pd.to_datetime(raw_price['time'])
raw_price['week'] = raw_price['time'].dt.to_period('W')
price_weekly = raw_price.groupby(["code", "week"]).last().reset_index()
price_weekly = price_weekly.set_index("week").sort_index()
del raw_price; gc.collect()

weeks = sorted(price_weekly.index.unique())
print(f"✓ Weekly: {len(weeks)} weeks, {weeks[0]}~{weeks[-1]}")

# 加载指数
INDEX_CODES = {"CSI300": "000300.XSHG", "CSI500": "000905.XSHG", "CSI_ALL": "000985.XSHG"}
index_data = {}
for name, code in INDEX_CODES.items():
    ix = get_price(code, start_date=START_DATE, end_date=END_DATE, frequency="daily",
                    fields=["close"], fq="post")
    index_data[name] = ix
    print(f"  {name}: {len(ix)} days")

# ============================================================
# Cell 4: 加载基本面 (分批查询, JQ免费版单次限量)
# ============================================================
print("\n[Cell 4] Loading fundamentals...")

# 精简到最核心6个字段
fund_cols = [
    valuation.code,
    valuation.pb_ratio,
    valuation.pe_ratio,
    indicator.roe,
    indicator.roa,
    indicator.gross_profit_margin,
]

fund_snapshots = {}
FUND_BATCH = 100  # 每批100只(低于JQ免费版限制)

for year in range(2022, 2026):
    date_str = f"{year}-12-31"
    all_dfs = []
    for b in range(0, len(ALL_STOCKS), FUND_BATCH):
        batch = ALL_STOCKS[b:b+FUND_BATCH]
        try:
            q = query(*fund_cols).filter(valuation.code.in_(batch))
            df = get_fundamentals(q, statDate=date_str)
            if df is not None and len(df) > 0:
                all_dfs.append(df)
        except Exception:
            pass
    if all_dfs:
        merged = pd.concat(all_dfs)
        fund_snapshots[date_str] = merged.set_index("code")
        print(f"  {date_str}: {len(merged)} stocks")
    gc.collect()

print(f"✓ {len(fund_snapshots)} annual snapshots")

def get_fund_for_stock(stock, date_str):
    """取最近一期基本面, 返回标准字段名"""
    for snap_date in sorted(fund_snapshots.keys(), reverse=True):
        if snap_date <= date_str:
            df = fund_snapshots[snap_date]
            if stock in df.index:
                row = df.loc[stock]
                return {
                    "roe": row.get("indicator.roe", np.nan),
                    "roa": row.get("indicator.roa", np.nan),
                    "gross_margin": row.get("indicator.gross_profit_margin", np.nan),
                    "pb": row.get("valuation.pb_ratio", np.nan),
                    "pe": row.get("valuation.pe_ratio", np.nan),
                }
    return {}

# ============================================================
# Cell 5: 因子计算函数 (30核心因子)
# ============================================================
print("\n[Cell 5] Computing factors...")

def build_price_slice(week_str, stock_list):
    """构建回看52周的价格slice"""
    week_idx = weeks.index(week_str)
    lb = max(0, week_idx - 52)
    hist = weeks[lb:week_idx+1]
    mask = price_weekly.index.isin(hist)
    sub = price_weekly[mask]
    
    result = {"close": {}, "open": {}, "high": {}, "low": {}, "volume": {}, "amount": {}}
    for s in stock_list:
        sd = sub[sub["code"] == s]
        if len(sd) >= 10:
            result["close"][s]   = sd["close"].values
            result["open"][s]    = sd["open"].values
            result["high"][s]    = sd["high"].values
            result["low"][s]     = sd["low"].values
            result["volume"][s]  = sd["volume"].values
            result["amount"][s]  = sd["money"].values
    return result

# --- 动量/反转类 ---
def compute_ret_1m(stocks, pd_):
    """最近1月收益"""
    close = pd_.get("close", {})
    arr = np.full(len(stocks), np.nan)
    for i, s in enumerate(stocks):
        c = close.get(s, np.array([]))
        if len(c) >= 5: arr[i] = c[-1] / c[-min(4,len(c))] - 1
    return arr, ~np.isnan(arr)

def compute_ret_3m(stocks, pd_):
    close = pd_.get("close", {})
    arr = np.full(len(stocks), np.nan)
    for i, s in enumerate(stocks):
        c = close.get(s, np.array([]))
        if len(c) >= 13: arr[i] = c[-1] / c[-min(13,len(c))] - 1
    return arr, ~np.isnan(arr)

def compute_ret_6m(stocks, pd_):
    close = pd_.get("close", {})
    arr = np.full(len(stocks), np.nan)
    for i, s in enumerate(stocks):
        c = close.get(s, np.array([]))
        if len(c) >= 26: arr[i] = c[-1] / c[-min(26,len(c))] - 1
    return arr, ~np.isnan(arr)

def compute_short_rev_5d(stocks, pd_):
    """5日反转"""
    close = pd_.get("close", {})
    arr = np.full(len(stocks), np.nan)
    for i, s in enumerate(stocks):
        c = close.get(s, np.array([]))
        if len(c) >= 2: arr[i] = -(c[-1] / c[-2] - 1)
    return arr, ~np.isnan(arr)

def compute_max_ret_1m(stocks, pd_):
    """1月最大单日收益"""
    close = pd_.get("close", {})
    arr = np.full(len(stocks), np.nan)
    for i, s in enumerate(stocks):
        c = close.get(s, np.array([]))
        if len(c) >= 5: arr[i] = np.max(np.diff(c[-5:]) / c[-5:-1])
    return arr, ~np.isnan(arr)

# --- 波动率类 ---
def compute_vol_1m(stocks, pd_):
    close = pd_.get("close", {})
    arr = np.full(len(stocks), np.nan)
    for i, s in enumerate(stocks):
        c = close.get(s, np.array([]))
        if len(c) >= 6:
            rets = np.diff(c[-6:]) / c[-6:-1]
            arr[i] = np.std(rets) * np.sqrt(5)
    return arr, ~np.isnan(arr)

def compute_vol_3m(stocks, pd_):
    close = pd_.get("close", {})
    arr = np.full(len(stocks), np.nan)
    for i, s in enumerate(stocks):
        c = close.get(s, np.array([]))
        if len(c) >= 14:
            rets = np.diff(c[-14:]) / c[-14:-1]
            arr[i] = np.std(rets) * np.sqrt(5)
    return arr, ~np.isnan(arr)

def compute_downside_vol(stocks, pd_):
    close = pd_.get("close", {})
    arr = np.full(len(stocks), np.nan)
    for i, s in enumerate(stocks):
        c = close.get(s, np.array([]))
        if len(c) >= 21:
            rets = np.diff(c[-21:]) / c[-21:-1]
            neg = rets[rets < 0]
            arr[i] = np.std(neg) * np.sqrt(5) if len(neg) > 0 else 0
    return arr, ~np.isnan(arr)

def compute_atr_14(stocks, pd_):
    close = pd_.get("close", {}); high = pd_.get("high", {}); low = pd_.get("low", {})
    arr = np.full(len(stocks), np.nan)
    for i, s in enumerate(stocks):
        c = close.get(s, np.array([])); h = high.get(s, np.array([])); l = low.get(s, np.array([]))
        if len(c) >= 15 and len(h) >= 15 and len(l) >= 15:
            tr = np.maximum(h[-14:]-l[-14:], np.abs(h[-14:]-np.roll(c[-14:],1)))
            tr[0] = h[-14] - l[-14]
            arr[i] = np.mean(tr) / c[-1] if c[-1] > 0 else np.nan
    return arr, ~np.isnan(arr)

# --- 量能类 ---
def compute_turnover_cv_20d(stocks, pd_):
    volume = pd_.get("volume", {})
    arr = np.full(len(stocks), np.nan)
    for i, s in enumerate(stocks):
        v = volume.get(s, np.array([]))
        if len(v) >= 20:
            arr[i] = np.std(v[-20:]) / (np.mean(v[-20:]) + 1e-8)
    return arr, ~np.isnan(arr)

def compute_money_flow_20(stocks, pd_):
    close = pd_.get("close", {}); high = pd_.get("high", {}); low = pd_.get("low", {})
    amount = pd_.get("amount", {})
    arr = np.full(len(stocks), np.nan)
    for i, s in enumerate(stocks):
        c = close.get(s, np.array([])); h = high.get(s, np.array([]))
        l = low.get(s, np.array([])); a = amount.get(s, np.array([]))
        n = min(len(c), len(h), len(l), len(a), 20)
        if n >= 5:
            arr[i] = np.sum(a[-n:] * (2*c[-n:]-l[-n:]-h[-n:]) / ((h[-n:]-l[-n:])+1e-8))
    return arr, ~np.isnan(arr)

def compute_dollar_vol_20d(stocks, pd_):
    amount = pd_.get("amount", {})
    arr = np.full(len(stocks), np.nan)
    for i, s in enumerate(stocks):
        a = amount.get(s, np.array([]))
        if len(a) >= 20: arr[i] = np.log(np.mean(a[-20:]) + 1)
    return arr, ~np.isnan(arr)

def compute_avg_turnover_1m(stocks, pd_):
    volume = pd_.get("volume", {})
    arr = np.full(len(stocks), np.nan)
    for i, s in enumerate(stocks):
        v = volume.get(s, np.array([]))
        if len(v) >= 5: arr[i] = np.mean(v[-5:])
    return arr, ~np.isnan(arr)

def compute_turnover_std(stocks, pd_):
    volume = pd_.get("volume", {})
    arr = np.full(len(stocks), np.nan)
    for i, s in enumerate(stocks):
        v = volume.get(s, np.array([]))
        if len(v) >= 20: arr[i] = np.std(v[-20:]) / (np.mean(v[-20:]) + 1e-8)
    return arr, ~np.isnan(arr)

def compute_turnover_change(stocks, pd_):
    volume = pd_.get("volume", {})
    arr = np.full(len(stocks), np.nan)
    for i, s in enumerate(stocks):
        v = volume.get(s, np.array([]))
        if len(v) >= 10:
            arr[i] = np.mean(v[-5:]) / (np.mean(v[-10:-5]) + 1e-8) - 1
    return arr, ~np.isnan(arr)

# --- 技术指标 ---
def compute_rsi_14(stocks, pd_):
    close = pd_.get("close", {})
    arr = np.full(len(stocks), np.nan)
    for i, s in enumerate(stocks):
        c = close.get(s, np.array([]))
        if len(c) >= 16:
            diffs = np.diff(c[-16:])
            gains = np.sum(np.maximum(diffs, 0))
            losses = np.sum(np.abs(np.minimum(diffs, 0)))
            if losses > 0: arr[i] = 100 - 100/(1 + gains/losses)
            else: arr[i] = 100
    return arr, ~np.isnan(arr)

def compute_beta(stocks, pd_):
    """个股Beta vs 沪深300"""
    close = pd_.get("close", {})
    ix_close = index_data["CSI300"].set_index("time") if "time" in index_data["CSI300"].columns else index_data["CSI300"]
    if hasattr(ix_close.index, 'tz'): ix_close = ix_close.resample('W').last()
    ix_ret = ix_close["close"].pct_change(5).dropna().values
    
    arr = np.full(len(stocks), np.nan)
    for i, s in enumerate(stocks):
        c = close.get(s, np.array([]))
        if len(c) >= 14:
            s_ret = np.diff(c[-14:]) / c[-14:-1]
            n = min(len(s_ret), len(ix_ret))
            if n >= 5:
                cov = np.cov(s_ret[-n:], ix_ret[-n:])
                arr[i] = cov[0,1] / (cov[1,1] + 1e-8)
    return arr, ~np.isnan(arr)

def compute_gap_up(stocks, pd_):
    """跳空高开"""
    close = pd_.get("close", {}); open_ = pd_.get("open", {})
    arr = np.full(len(stocks), np.nan)
    for i, s in enumerate(stocks):
        c = close.get(s, np.array([])); o = open_.get(s, np.array([]))
        if len(c) >= 3 and len(o) >= 2:
            arr[i] = (o[-1] / c[-2] - 1) * 100
    return arr, ~np.isnan(arr)

def compute_relative_spread_proxy(stocks, pd_):
    """开盘-前收盘价差"""
    open_ = pd_.get("open", {}); close = pd_.get("close", {})
    arr = np.full(len(stocks), np.nan)
    for i, s in enumerate(stocks):
        o = open_.get(s, np.array([])); c = close.get(s, np.array([]))
        if len(o) >= 5 and len(c) >= 5:
            spreads = (o[-5:] - np.roll(c[-5:], 1)[-5:]) / (np.roll(c[-5:], 1)[-5:] + 1e-8)
            arr[i] = np.mean(spreads)
    return arr, ~np.isnan(arr)

def compute_ret_open_2d_proxy(stocks, pd_):
    """开盘价2日动量"""
    open_ = pd_.get("open", {})
    arr = np.full(len(stocks), np.nan)
    for i, s in enumerate(stocks):
        o = open_.get(s, np.array([]))
        if len(o) >= 3: arr[i] = o[-1] / o[-3] - 1
    return arr, ~np.isnan(arr)

def compute_vm_diff(stocks, pd_):
    """成交量-金额背离"""
    volume = pd_.get("volume", {}); amount = pd_.get("amount", {})
    arr = np.full(len(stocks), np.nan)
    for i, s in enumerate(stocks):
        v = volume.get(s, np.array([])); a = amount.get(s, np.array([]))
        n = min(len(v), len(a), 20)
        if n >= 10:
            v_chg = np.mean(v[-5:]) / (np.mean(v[-10:-5]) + 1e-8) - 1
            a_chg = np.mean(a[-5:]) / (np.mean(a[-10:-5]) + 1e-8) - 1
            arr[i] = v_chg - a_chg
    return arr, ~np.isnan(arr)

def compute_volume_climax_reversal(stocks, pd_):
    """量能高潮反转"""
    volume = pd_.get("volume", {}); close = pd_.get("close", {})
    arr = np.full(len(stocks), np.nan)
    for i, s in enumerate(stocks):
        v = volume.get(s, np.array([])); c = close.get(s, np.array([]))
        if len(v) >= 10 and len(c) >= 10:
            vol_peak = v[-1] > 2 * np.mean(v[-10:-1])
            price_down = c[-1] < c[-5] if len(c) >= 6 else False
            arr[i] = -1.0 if (vol_peak and price_down) else 0.0
    return arr, ~np.isnan(arr)

# --- 基本面因子 ---
def compute_roe(stocks, pd_, week_str=""):
    arr = np.full(len(stocks), np.nan)
    for i, s in enumerate(stocks):
        f = get_fund_for_stock(s, str(week_str))
        arr[i] = f.get("roe", np.nan)
    return arr, ~np.isnan(arr)

def compute_roa(stocks, pd_, week_str=""):
    arr = np.full(len(stocks), np.nan)
    for i, s in enumerate(stocks):
        f = get_fund_for_stock(s, str(week_str))
        arr[i] = f.get("roa", np.nan)
    return arr, ~np.isnan(arr)

def compute_gross_margin(stocks, pd_, week_str=""):
    arr = np.full(len(stocks), np.nan)
    for i, s in enumerate(stocks):
        f = get_fund_for_stock(s, str(week_str))
        arr[i] = f.get("gross_margin", np.nan)
    return arr, ~np.isnan(arr)

# 注意: net_margin / earnings_growth 因 JQ 免费版字段受限暂不可用

def compute_sp(stocks, pd_, week_str=""):
    """Sales/Price (估值)"""
    arr = np.full(len(stocks), np.nan)
    for i, s in enumerate(stocks):
        f = get_fund_for_stock(s, str(week_str))
        ps = f.get("pe", np.nan)  # pe proxy for valuation
        arr[i] = 1/ps if (ps and ps > 0) else np.nan
    return arr, ~np.isnan(arr)

def compute_bp(stocks, pd_, week_str=""):
    """Book/Price"""
    arr = np.full(len(stocks), np.nan)
    for i, s in enumerate(stocks):
        f = get_fund_for_stock(s, str(week_str))
        pb = f.get("pb", np.nan)
        arr[i] = 1/pb if (pb and pb > 0) else np.nan
    return arr, ~np.isnan(arr)

# 因子注册表 (独立于registry，精简版)
FACTOR_REGISTRY = {
    # 动量/反转
    "ret_1m":       {"func": compute_ret_1m,       "direction": -1},
    "ret_3m":       {"func": compute_ret_3m,       "direction": -1},
    "ret_6m":       {"func": compute_ret_6m,       "direction":  1},
    "short_rev_5d": {"func": compute_short_rev_5d, "direction":  1},
    "max_ret_1m":   {"func": compute_max_ret_1m,   "direction": -1},
    # 波动率
    "vol_1m":       {"func": compute_vol_1m,        "direction": -1},
    "vol_3m":       {"func": compute_vol_3m,        "direction": -1},
    "downside_vol": {"func": compute_downside_vol,  "direction": -1},
    "atr_14":       {"func": compute_atr_14,        "direction": -1},
    # 量能
    "turnover_cv_20d":     {"func": compute_turnover_cv_20d,      "direction":  1},
    "money_flow_20":       {"func": compute_money_flow_20,        "direction":  1},
    "dollar_vol_20d":      {"func": compute_dollar_vol_20d,       "direction":  1},
    "avg_turnover_1m":     {"func": compute_avg_turnover_1m,      "direction": -1},
    "turnover_std":        {"func": compute_turnover_std,         "direction": -1},
    "turnover_change":     {"func": compute_turnover_change,      "direction":  1},
    "volume_climax_rev":   {"func": compute_volume_climax_reversal, "direction": 1},
    # 技术指标
    "rsi_14":              {"func": compute_rsi_14,               "direction": -1},
    "beta":                {"func": compute_beta,                 "direction": -1},
    "gap_up":              {"func": compute_gap_up,               "direction":  1},
    "relative_spread_proxy":{"func": compute_relative_spread_proxy,"direction": 1},
    "ret_open_2d_proxy":   {"func": compute_ret_open_2d_proxy,   "direction":  1},
    "vm_diff":             {"func": compute_vm_diff,              "direction":  1},
    # 基本面
    "roe":                 {"func": compute_roe,                  "direction":  1},
    "roa":                 {"func": compute_roa,                  "direction":  1},
    "gross_margin":        {"func": compute_gross_margin,         "direction":  1},
    "sp":                  {"func": compute_sp,                   "direction":  1},  # 估值
    "bp":                  {"func": compute_bp,                   "direction":  1},  # 估值
}

print(f"✓ {len(FACTOR_REGISTRY)} factors registered")
for name in FACTOR_REGISTRY:
    print(f"  {name}")

# ============================================================
# Cell 6: 构建因子矩阵 (周频)
# ============================================================
print("\n[Cell 6] Building factor matrix...")

# 取可用的周末 (前52周热身 + 每4周取一次以减少计算量)
WARMUP = 52
VALID_WEEKS = weeks[WARMUP:]  # 前52周是回看期
STRIDE = 4                     # 每4周取一个截面 → ~40周 vs ~200周
SAMPLE_WEEKS = VALID_WEEKS[::STRIDE]

print(f"  Total weeks: {len(VALID_WEEKS)}, sampling every {STRIDE} → {len(SAMPLE_WEEKS)} weeks")

# 存储: factor_data[factor_name] = DataFrame(weeks × stocks)
# forward_returns = DataFrame(weeks × stocks)
factor_data = {name: pd.DataFrame(index=[str(w) for w in SAMPLE_WEEKS], columns=ALL_STOCKS, dtype=float)
               for name in FACTOR_REGISTRY}
forward_returns = pd.DataFrame(index=[str(w) for w in SAMPLE_WEEKS], columns=ALL_STOCKS, dtype=float)

for widx, week_str in enumerate(SAMPLE_WEEKS):
    ws = str(week_str)
    if widx % 10 == 0:
        print(f"  [{widx+1}/{len(SAMPLE_WEEKS)}] {ws}...")
    
    # 本周活跃股票
    sub = price_weekly[price_weekly.index == week_str]
    active = set(sub["code"]) & set(ALL_STOCKS)
    active_list = sorted(active)
    
    if len(active_list) < 30:
        if widx == 0: print(f"    [DEBUG] Week {ws}: only {len(active_list)} active, skipping")
        continue
    
    # 构建price_data
    pd_slice = build_price_slice(week_str, active_list)
    
    if widx == 0:
        n_close = sum(1 for v in pd_slice.get('close',{}).values() if len(v)>0)
        first_stock = active_list[0] if active_list else 'N/A'
        c_vals = pd_slice.get('close',{}).get(first_stock, [])
        print(f"    [DEBUG] Week {ws}: {len(active_list)} stocks, {n_close} w/ close data")
        print(f"    [DEBUG] {first_stock} close[0:5] = {c_vals[:5] if len(c_vals)>0 else 'EMPTY'}")
    
    # 计算所有因子 (去掉 try/except, 直接看错误)
    for fname, fmeta in FACTOR_REGISTRY.items():
        if "week_str" in fmeta["func"].__code__.co_varnames:
            arr, mask = fmeta["func"](active_list, pd_slice, week_str=ws)
        else:
            arr, mask = fmeta["func"](active_list, pd_slice)
        # 标准化 (zscore, clip[-3,3])
        valid_arr = arr[mask]
        if len(valid_arr) > 5:
            med = np.median(valid_arr)
            mad_val = np.median(np.abs(valid_arr - med)) * 1.4826
            if mad_val > 0:
                z = (valid_arr - med) / mad_val
                z = np.clip(z, -3, 3)
                arr[mask] = z
        # 存入
        for j, s in enumerate(active_list):
            if mask[j]:
                factor_data[fname].loc[ws, s] = arr[j] * fmeta["direction"]  # ×方向统一
    
    # 计算forward return (下一周)
    next_week_idx = weeks.index(week_str) + 1
    if next_week_idx < len(weeks):
        next_week = weeks[next_week_idx]
        sub_nxt = price_weekly[price_weekly.index == next_week]
        sub_cur = price_weekly[price_weekly.index == week_str]
        if widx == 0:
            print(f"    [DEBUG] fwd: cur({len(sub_cur)} rows, {len(sub_cur.code.unique())} stocks), nxt({len(sub_nxt)} rows, {len(sub_nxt.code.unique())} stocks)")
        fwd_count = 0
        for s in active_list:
            if s in sub_nxt["code"].values and s in sub_cur["code"].values:
                p_nxt = sub_nxt[sub_nxt["code"]==s]["close"].values[0]
                p_cur = sub_cur[sub_cur["code"]==s]["close"].values[0]
                if p_cur > 0:
                    forward_returns.loc[ws, s] = p_nxt / p_cur - 1
                    fwd_count += 1
        if widx == 0:
            print(f"    [DEBUG] fwd values computed: {fwd_count}")
    
    gc.collect()

print(f"✓ Factor matrix built: {len(SAMPLE_WEEKS)} weeks × {len(ALL_STOCKS)} stocks × {len(FACTOR_REGISTRY)} factors")

# ★ 诊断: 检查因子值和 forward returns
fw_nn = forward_returns.notna().sum().sum()
fd_nn = sum(factor_data[fn].notna().sum().sum() for fn in FACTOR_REGISTRY)
print(f"  [DIAG] forward_returns non-NaN cells: {fw_nn}")
print(f"  [DIAG] factor_data total non-NaN cells: {fd_nn}")
for fn in list(FACTOR_REGISTRY.keys())[:5]:
    nn = factor_data[fn].notna().sum().sum()
    print(f"    {fn}: {nn} non-NaN")

# 保存中间结果 (可选)
# pd.to_pickle(dict(factor_data=factor_data, forward_returns=forward_returns), "factor_matrix.pkl")

# ============================================================
# Cell 7: Regime 检测
# ============================================================
print("\n[Cell 7] Regime detection...")

CSI300_ret = index_data["CSI300"]["close"].pct_change(4).dropna()  # 周收益
CSI500_ret = index_data["CSI500"]["close"].pct_change(4).dropna()
CSI_ALL_ret = index_data["CSI_ALL"]["close"].pct_change(4).dropna()

regime_labels = {}
for week_str in SAMPLE_WEEKS:
    ws = str(week_str)
    try:
        style_spread = CSI500_ret.loc[ws] - CSI300_ret.loc[ws]
        trend = CSI_ALL_ret.loc[ws]
        if style_spread > STYLE_SPREAD_THRESHOLD and trend > TREND_THRESHOLD:
            regime_labels[ws] = "small_bull"
        else:
            regime_labels[ws] = "large_dominant"
    except KeyError:
        regime_labels[ws] = "large_dominant"

counts = pd.Series(regime_labels).value_counts()
print(f"  Regimes: {counts.to_dict()}")

# ============================================================
# Cell 8: ElasticNetCV per regime
# ============================================================
print("\n[Cell 8] ElasticNetCV per regime...")

# 合并因子为X, forward return为y
def build_xy_for_regime(target_regime):
    factor_names = list(FACTOR_REGISTRY.keys())
    
    # Phase 1: 收集每周的原始X (允许NaN列)
    raw_weeks = []  # [(stock_list, y, X_unfiltered)]
    for week_str in SAMPLE_WEEKS:
        ws = str(week_str)
        if regime_labels.get(ws) != target_regime:
            continue
        if ws not in forward_returns.index:
            continue
        fwd = forward_returns.loc[ws].dropna()
        if len(fwd) < 30:
            continue
        stock_list = list(fwd.index)
        X = np.column_stack([
            factor_data[fn].loc[ws, stock_list].values for fn in factor_names
        ])
        raw_weeks.append((stock_list, fwd.values, X))
    
    if not raw_weeks:
        print(f"    No weeks with fwd returns!")
        return None, None, None
    
    # Phase 2: 堆叠所有X, 找NaN>50%的列并剔除
    all_X_stack = np.vstack([w[2] for w in raw_weeks])
    col_nan_ratio = np.isnan(all_X_stack).mean(axis=0)
    good_cols = col_nan_ratio < 0.5
    
    if good_cols.sum() == 0:
        print(f"    All {len(factor_names)} factor cols >50% NaN, abort")
        return None, None, None
    
    dropped = [fn for i, fn in enumerate(factor_names) if not good_cols[i]]
    kept_names = [fn for i, fn in enumerate(factor_names) if good_cols[i]]
    if dropped:
        print(f"    Dropped {len(dropped)} NaN-heavy factors: {dropped}")
        print(f"    Keeping {len(kept_names)}: {kept_names}")
    
    # Phase 3: 用good_cols重做X, 然后逐行去NaN
    X_list, y_list = [], []
    for stock_list, y, X in raw_weeks:
        X_good = X[:, good_cols]
        valid_rows = ~np.isnan(X_good).any(axis=1)
        if valid_rows.sum() > 30:
            X_list.append(X_good[valid_rows])
            y_list.append(y[valid_rows])
    
    if not X_list:
        print(f"    After NaN filtering: 0 valid rows")
        return None, None, None
    
    X_all = np.vstack(X_list)
    y_all = np.hstack(y_list)
    print(f"    Total samples: {len(y_all)} × {len(kept_names)} factors")
    return X_all, y_all, kept_names

results = {}
for regime in ["large_dominant", "small_bull"]:
    print(f"\n  === {regime} ===")
    outcome = build_xy_for_regime(regime)
    if outcome is None or outcome[0] is None:
        print(f"    No data!")
        results[regime] = {"status": "no_data"}
        continue
    
    X, y, factor_names = outcome
    print(f"    Samples: {len(y)} (stocks × weeks)")
    
    # ElasticNetCV (自动选alpha+l1_ratio)
    model = ElasticNetCV(
        l1_ratio=[.1, .5, .7, .9, .95, 1],
        cv=5, max_iter=2000, n_jobs=1, random_state=42
    )
    
    # 标准化
    X_mean = np.nanmean(X, axis=0)
    X_std = np.nanstd(X, axis=0) + 1e-8
    X_scaled = (X - X_mean) / X_std
    
    model.fit(X_scaled, y)
    
    print(f"    Best alpha={model.alpha_:.6f}, l1_ratio={model.l1_ratio_:.3f}")
    
    # 提取非零权重
    coef = model.coef_ / X_std  # 反标准化
    
    weights = {}
    for i, fn in enumerate(factor_names):
        w = coef[i]
        if abs(w) > 1e-6:
            weights[fn] = round(float(w), 6)
    
    # 归一化
    total = sum(abs(v) for v in weights.values())
    if total > 0:
        weights = {k: round(v/total, 4) for k, v in weights.items()}
    
    # IS拟合: 计算信号IC
    signal = np.dot(X, coef)
    ic, _ = spearmanr(signal, y)
    
    print(f"    Non-zero factors: {len(weights)}")
    print(f"    IS IC: {ic:.4f}")
    print(f"    Weights: {weights}")
    
    results[regime] = {
        "status": "ok",
        "alpha": float(model.alpha_),
        "l1_ratio": float(model.l1_ratio_),
        "ic": round(float(ic), 4),
        "n_factors": len(weights),
        "weights": weights,
    }

# ============================================================
# Cell 9: 输出最终结果
# ============================================================
print("\n" + "="*60)
print("弹性网络因子选择结果")
print("="*60)

for regime, res in results.items():
    print(f"\n● {regime}:")
    if res.get("status") == "ok":
        print(f"  IS IC: {res['ic']:.4f} | α={res['alpha']:.6f} | L1={res['l1_ratio']:.3f}")
        print(f"  选中 {res['n_factors']}/{len(FACTOR_REGISTRY)} 个因子:")
        sorted_w = sorted(res["weights"].items(), key=lambda x: abs(x[1]), reverse=True)
        for fname, weight in sorted_w:
            sign = "+" if weight > 0 else ""
            print(f"    {fname:>20s}: {sign}{weight:.4f}")
    else:
        print(f"  NO DATA")

print("\n" + "="*60)
print("完成! 将上述权重复制到本地, 用 JQGenerator 生成策略")
print("  large_dominant 权重 → JQGenerator(weights_large_dominant)")
print("  small_bull 权重      → JQGenerator(weights_small_bull)")
print("="*60)

import json
print("\n# JSON 格式 (复制到本地):")
print(json.dumps(results, indent=2, ensure_ascii=False))
