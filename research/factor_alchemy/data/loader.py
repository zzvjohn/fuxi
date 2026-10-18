"""
数据加载器
==========
从已有 Tushare 数据文件加载:
- daily_prices.csv (不复权日线)
- daily_basic.csv (日频估值 + 换手率)
- fina_indicator_rich.csv (季度财务指标, 38列)
- adj_factor.csv (复权因子)

输出后复权(hfq)价格 + 估值 + 财务数据
"""
import numpy as np
import pandas as pd
from pathlib import Path

# 工作区绝对路径 — 优先从 __file__ 推导, 回退到 e:/quant
try:
    _WORKSPACE = Path(__file__).resolve().parent.parent.parent.parent
except:
    _WORKSPACE = Path('e:/quant')
_DATA_RAW = _WORKSPACE / 'data' / 'raw'
_DATA_TUSHARE = _WORKSPACE / 'data' / 'tushare'

# 如果推导失败, 直接用绝对路径
if not _DATA_RAW.exists():
    _WORKSPACE = Path('e:/quant')
    _DATA_RAW = _WORKSPACE / 'data' / 'raw'
    _DATA_TUSHARE = _WORKSPACE / 'data' / 'tushare'


def load_all_data(start_date='2021-01-01', end_date='2026-04-30'):
    """
    加载全部数据
    
    Returns
    -------
    dict
        {
            'price_data': dict {field: pd.DataFrame},
            'financial_data': pd.DataFrame,
            'valuation_data': pd.DataFrame,
            'industry_map': dict,
        }
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    
    # 1. 加载日线价格 (不复权) — 加 open/high/low 支持量价因子
    print("  加载 daily_prices.csv ...")
    price_raw = _load_csv_chunked(_DATA_RAW / 'daily_prices.csv', 
                                   usecols=['ts_code', 'trade_date', 'open', 'high', 'low',
                                            'close', 'vol', 'amount'],
                                   start_date=start, end_date=end)
    # _load_csv_chunked 已处理 trade_date 类型转换，无需重复调用 pd.to_datetime
    
    # 2. 加载复权因子
    print("  加载 adj_factor.csv ...")
    adj_raw = _load_csv_chunked(_DATA_RAW / 'adj_factor.csv',
                                 usecols=['ts_code', 'trade_date', 'adj_factor'],
                                 start_date=start, end_date=end)
    # 去重 adj_factor (Stage 0 重建可能产生重复)
    adj_dup = adj_raw.duplicated(subset=['ts_code', 'trade_date']).sum()
    if adj_dup > 0:
        adj_raw = adj_raw.drop_duplicates(subset=['ts_code', 'trade_date'])
    
    # 计算后复权价格: hfq = close * adj_factor
    print("  计算后复权价格...")
    price_adj = price_raw.merge(adj_raw, on=['ts_code', 'trade_date'], how='left')
    
    # ★ 去重: Stage 0 数据重建可能导致 (ts_code, trade_date) 重复
    dup_count = price_adj.duplicated(subset=['ts_code', 'trade_date']).sum()
    if dup_count > 0:
        print(f"    去重: {dup_count} 行 ({price_adj.shape[0]} → ", end='')
        price_adj = price_adj.drop_duplicates(subset=['ts_code', 'trade_date'])
        print(f"{price_adj.shape[0]})")
    # 修复 adj_factor NaN: 数据可能从较晚日期开始 (如2021-03-30)
    # 对每只股票, 用其最早的可用 adj_factor 回填缺失值
    if price_adj['adj_factor'].isna().any():
        nan_count = price_adj['adj_factor'].isna().sum()
        price_adj = price_adj.sort_values(['ts_code', 'trade_date'])
        price_adj['adj_factor'] = price_adj.groupby('ts_code')['adj_factor'].transform('bfill')
        price_adj['adj_factor'] = price_adj['adj_factor'].fillna(1.0)
        print(f"    修复 adj_factor NaN: {nan_count} → {price_adj['adj_factor'].isna().sum()}")
    
    price_adj['close_hfq'] = price_adj['close'] * price_adj['adj_factor']
    # ★ 后复权 open/high/low
    price_adj['open_hfq'] = price_adj['open'] * price_adj['adj_factor']
    price_adj['high_hfq'] = price_adj['high'] * price_adj['adj_factor']
    price_adj['low_hfq'] = price_adj['low'] * price_adj['adj_factor']
    
    # 分离出几类数据
    # 收益率用 close_hfq 计算
    # 换手率从 daily_basic 获取
    # 成交额 = amount (元)
    
    # 3. 加载估值数据
    print("  加载 daily_basic.csv ...")
    val_raw = _load_csv_chunked(_DATA_RAW / 'daily_basic.csv',
                                 usecols=['ts_code', 'trade_date', 'turnover_rate',
                                          'pe_ttm', 'pb', 'ps_ttm',
                                          'dv_ttm', 'total_mv', 'circ_mv'],
                                 start_date=start, end_date=end)
    # _load_csv_chunked 已处理 trade_date 类型转换
    dup_count = val_raw.duplicated(subset=['ts_code', 'trade_date']).sum()
    if dup_count > 0:
        print(f"    daily_basic 去重: {dup_count} 行 ({val_raw.shape[0]} → ", end='')
        val_raw = val_raw.drop_duplicates(subset=['ts_code', 'trade_date'])
        print(f"{val_raw.shape[0]})")

    # 4. 加载财务数据 (优先用 rich 版本, 38列)
    rich_path = _DATA_TUSHARE / 'fina_indicator_rich.csv'
    if rich_path.exists():
        print("  加载 fina_indicator_rich.csv (38列) ...")
        fin_raw = pd.read_csv(rich_path, encoding='utf-8-sig')
        # 日期是 float YYYYMMDD 格式, 需要特殊处理
        if 'ann_date' in fin_raw.columns:
            fin_raw['ann_date'] = pd.to_datetime(
                fin_raw['ann_date'].fillna(0).astype(int).astype(str), format='%Y%m%d',
                errors='coerce'
            )
        if 'end_date' in fin_raw.columns:
            fin_raw['end_date'] = pd.to_datetime(
                fin_raw['end_date'].fillna(0).astype(int).astype(str), format='%Y%m%d',
                errors='coerce'
            )
    else:
        print("  加载 fina_indicator_full.csv ...")
        fin_raw = pd.read_csv(_DATA_RAW / 'fina_indicator_full.csv', encoding='utf-8-sig')
        fin_raw['ann_date'] = pd.to_datetime(fin_raw['ann_date'])
        fin_raw['end_date'] = pd.to_datetime(fin_raw['end_date'])
    
    # 过滤: 只保留 ann_date 在范围内的
    fin_raw = fin_raw[(fin_raw['ann_date'] >= start) & (fin_raw['ann_date'] <= end)]
    
    # ★ 去重: 同一股票同一 end_date 取最新 ann_date
    fin_raw = fin_raw.sort_values('ann_date').drop_duplicates(
        subset=['ts_code', 'end_date'], keep='last'
    )
    
    # 构建 price_data dict
    close_df = price_adj.pivot(index='trade_date', columns='ts_code', values='close_hfq')
    open_df = price_adj.pivot(index='trade_date', columns='ts_code', values='open_hfq')
    high_df = price_adj.pivot(index='trade_date', columns='ts_code', values='high_hfq')
    low_df = price_adj.pivot(index='trade_date', columns='ts_code', values='low_hfq')
    vol_df = price_adj.pivot(index='trade_date', columns='ts_code', values='vol')
    amount_df = price_adj.pivot(index='trade_date', columns='ts_code', values='amount')

    # ★ 订单流字段 (2026-09-16, Stage2 PASS 因子注册)
    #   数据源 data/raw/moneyflow_daily.csv (Tushare moneyflow)。
    #   用途: crowded_reversal_5d_large_order_flow 等订单流因子消费 buy_lg_vol/sell_lg_vol。
    #   失败不阻塞: 缺文件/读失败时 orderflow 为空, 消费因子自行返回 None 降级。
    #   注: 该表实测仅覆盖 2025-08-08 起 (约 1 年), 早期截面为 NaN。
    _ORDERFLOW_COLS = ['buy_sm_vol', 'sell_sm_vol', 'buy_lg_vol', 'sell_lg_vol',
                       'buy_elg_vol', 'sell_elg_vol', 'net_mf_amount']
    orderflow = {}
    mf_path = _DATA_RAW / 'moneyflow_daily.csv'
    if mf_path.exists():
        try:
            mf_raw = _load_csv_chunked(
                mf_path,
                usecols=['ts_code', 'trade_date'] + _ORDERFLOW_COLS,
                start_date=start, end_date=end,
            )
            if len(mf_raw) > 0:
                mf_raw = mf_raw.drop_duplicates(subset=['ts_code', 'trade_date'])
                for _c in _ORDERFLOW_COLS:
                    if _c in mf_raw.columns:
                        orderflow[_c] = mf_raw.pivot(
                            index='trade_date', columns='ts_code', values=_c
                        ).reindex(index=close_df.index)
                print(f"  加载 moneyflow_daily.csv 订单流: {mf_raw.shape[0]} 行, "
                      f"{len(orderflow)} 字段, 覆盖 "
                      f"{mf_raw['trade_date'].min().date()}~{mf_raw['trade_date'].max().date()}")
            else:
                print("  [WARN] moneyflow_daily.csv 在区间内无数据, 订单流因子不可用")
        except Exception as _e:
            print(f"  [WARN] moneyflow_daily.csv 加载失败, 订单流因子不可用: {_e}")
    else:
        print("  [WARN] moneyflow_daily.csv 不存在, 订单流因子不可用")

    # 从估值数据中取换手率
    if 'turnover_rate' in val_raw.columns:
        turnover_df = val_raw.pivot(index='trade_date', columns='ts_code', values='turnover_rate')
    else:
        turnover_df = pd.DataFrame()
    
    # ★ 加载沪深300指数价格 (用于Beta/IdioVol)
    index_close = None
    idx_path = _DATA_RAW / 'index_000300_daily.csv'
    if idx_path.exists():
        idx_df = pd.read_csv(idx_path, encoding='utf-8-sig')
        if 'trade_date' in idx_df.columns:
            # ★ 兼容 int YYYYMMDD 和 字符串格式
            if idx_df['trade_date'].dtype in ('int64', 'int32', 'float64', 'float32'):
                idx_df['trade_date'] = pd.to_datetime(
                    idx_df['trade_date'].fillna(0).astype(int).astype(str), format='%Y%m%d',
                    errors='coerce'
                )
            else:
                idx_df['trade_date'] = pd.to_datetime(idx_df['trade_date'], format='mixed', errors='coerce')
            idx_df = idx_df.sort_values('trade_date')
            index_close = idx_df.set_index('trade_date')['close']  # ★ FIX: 变量名 index_close
            index_close.name = '000300'
    else:
        print("  ⚠ 未找到 index_000300_daily.csv, Beta/IdioVol 将不可用")
    
    price_data = {
        'close': close_df,
        'open': open_df,          # ★ 新增
        'high': high_df,          # ★ 新增
        'low': low_df,            # ★ 新增
        'volume': vol_df,
        'amount': amount_df,
        'turnover': turnover_df,
        'index_close': index_close,  # ★ 现在不会为None了
    }
    # ★ 订单流字段合并进 price_data (2026-09-16, 仅在上文加载成功时非空)
    price_data.update(orderflow)
    
    # 估值数据 (保留原始格式, 不pivot)
    val_out = val_raw.copy()
    # ★ 统一列名: ts_code → code (因子模块统一使用 code)
    if 'ts_code' in val_out.columns:
        val_out = val_out.rename(columns={'ts_code': 'code'})
    col_map = {
        'pe_ttm': 'pe_ttm',
        'pb': 'pb',
        'ps_ttm': 'ps_ttm',
        'dv_ttm': 'dv_ttm',
        'total_mv': 'market_cap',
        'circ_mv': 'circ_mv',
        'turnover_rate': 'turnover',
    }
    val_out = val_out.rename(columns={k: v for k, v in col_map.items() if k in val_out.columns})
    
    print(f"  完成! 价格={close_df.shape}, 估值={val_out.shape}, 财务={fin_raw.shape}")
    
    # ★ 加载申万一级行业分类 (stock→industry_label dict)
    industry_map = _load_industry_map()

    return {
        'price_data': price_data,
        'financial_data': fin_raw,
        'valuation_data': val_out,
        'industry_map': industry_map,
    }


def _load_industry_map():
    """
    加载申万一级行业分类 → dict: stock_code → industry_name.
    
    数据源: data/sw_industry_map.csv (从 Tushare index_member 拉取)
    """
    from pathlib import Path
    csv_path = Path(__file__).parent.parent.parent.parent / 'data' / 'sw_industry_map.csv'
    
    if not csv_path.exists():
        print("  [WARN] sw_industry_map.csv 不存在, 行业中性化将不可用")
        return {}
    
    df = pd.read_csv(csv_path, dtype={'code': str})
    # 取最新(is_new='Y' 或 out_date 为空)的映射
    industry_map = dict(zip(df['code'], df['industry']))
    print(f"  [INDUSTRY] 加载申万一级行业映射: {len(industry_map)} 只股票")
    return industry_map


def _load_csv_chunked(filepath, usecols=None, start_date=None, end_date=None, chunksize=500000):
    """
    分块读取大 CSV, 按日期过滤
    
    Returns
    -------
    pd.DataFrame
    """
    import sys
    
    chunks = []
    for chunk in pd.read_csv(filepath, encoding='utf-8-sig',
                              usecols=usecols, chunksize=chunksize,
                              low_memory=False):
        if 'trade_date' in chunk.columns:
            # ★ 兼容两种格式: 字符串 '2021-01-04' 或 int YYYYMMDD
            if chunk['trade_date'].dtype in ('int64', 'int32', 'float64', 'float32'):
                chunk['trade_date'] = pd.to_datetime(
                    chunk['trade_date'].fillna(0).astype(int).astype(str), format='%Y%m%d',
                    errors='coerce'
                )
            else:
                chunk['trade_date'] = pd.to_datetime(chunk['trade_date'], format='mixed', errors='coerce')
            if start_date:
                chunk = chunk[chunk['trade_date'] >= start_date]
            if end_date:
                chunk = chunk[chunk['trade_date'] <= end_date]
        
        if len(chunk) > 0:
            chunks.append(chunk)
    
    if not chunks:
        return pd.DataFrame()
    
    return pd.concat(chunks, ignore_index=True)
