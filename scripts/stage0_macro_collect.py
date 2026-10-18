# -*- coding: utf-8 -*-
"""P-20260908-003: 宏观月度数据采集 (PMI/CPI/PPI/M1/M2) — 多源 + 防前视

数据源优先级: Tushare(cn_pmi/cn_cpi/cn_ppi/cn_m) → akshare(macro_china_pmi_yearly)
→ 手工 CSV 补充 (data/raw/macro_manual.csv, 列: month,pmi_mfg,cpi_yoy,ppi_yoy,m1_yoy,m2_yoy)

🔴 防前视设计 (用户钦定):
  - 所有月度指标加入 release_lag: 官方发布滞后 (PMI 次月1日, CPI/PPI 次月9-15日,
    M1/M2 次月10-16日, 用电量次月14-18日)
  - 输出列 avail_month = 该值最早可用的月份 (发布月)
  - 因子构造端必须只用 avail_month <= 当前回测月的滞后值 (shift 语义由消费端实现)

输出: data/raw/macro_monthly.csv
用法: python scripts/stage0_macro_collect.py
"""
import sys
import re
from pathlib import Path
from datetime import datetime

import pandas as pd

sys.path.insert(0, 'e:/quant')
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / 'data' / 'raw' / 'macro_monthly.csv'
MANUAL = PROJECT_ROOT / 'data' / 'raw' / 'macro_manual.csv'

# 各指标发布滞后 (月): 官方惯例
RELEASE_LAG = {
    'pmi_mfg': 1,   # 制造业 PMI: 次月 1 日发布
    'cpi_yoy': 1,   # CPI: 次月 9-15 日
    'ppi_yoy': 1,   # PPI: 次月 9-15 日
    'm1_yoy': 1,    # 货币供应: 次月 10-16 日
    'm2_yoy': 1,
}


def _month_shift(month_str: str, n: int) -> str:
    """YYYYMM 月偏移 n 个月"""
    y, m = int(month_str[:4]), int(month_str[4:6])
    idx = y * 12 + (m - 1) + n
    return f'{idx // 12}{idx % 12 + 1:02d}'


def collect_tushare():
    """Tushare 宏观接口 (cn_pmi/cn_cpi/cn_ppi/cn_m)"""
    try:
        from credentials import get_tushare_api
        pro = get_tushare_api()
        frames = {}
        # PMI: pmi010 = 制造业PMI
        try:
            df = pro.cn_pmi(start_m='200501', end_m='209912',
                            fields='month,pmi010')
            if len(df) and df['pmi010'].notna().any():
                frames['pmi_mfg'] = df.set_index('month')['pmi010']
        except Exception as e:
            print(f'  [tushare] cn_pmi FAIL: {str(e)[:80]}')
        try:
            df = pro.cn_cpi(start_m='200501', end_m='209912',
                            fields='month,nt_yoy')
            if len(df) and df['nt_yoy'].notna().any():
                frames['cpi_yoy'] = df.set_index('month')['nt_yoy']
        except Exception as e:
            print(f'  [tushare] cn_cpi FAIL: {str(e)[:80]}')
        try:
            df = pro.cn_ppi(start_m='200501', end_m='209912',
                            fields='month,ppi_yoy')
            if len(df) and df['ppi_yoy'].notna().any():
                frames['ppi_yoy'] = df.set_index('month')['ppi_yoy']
        except Exception as e:
            print(f'  [tushare] cn_ppi FAIL: {str(e)[:80]}')
        try:
            df = pro.cn_m(start_m='200501', end_m='209912',
                          fields='month,m1_yoy,m2_yoy')
            if len(df):
                if df['m1_yoy'].notna().any():
                    frames['m1_yoy'] = df.set_index('month')['m1_yoy']
                if df['m2_yoy'].notna().any():
                    frames['m2_yoy'] = df.set_index('month')['m2_yoy']
        except Exception as e:
            print(f'  [tushare] cn_m FAIL: {str(e)[:80]}')
        if frames:
            out = pd.DataFrame(frames)
            out.index.name = 'month'
            out = out.reset_index()
            out['source'] = 'tushare'
            return out
    except Exception as e:
        print(f'  [tushare] 整体失败: {str(e)[:100]}')
    return None


def collect_akshare():
    """akshare 兜底: 官方制造业 PMI (含今值/前值)"""
    try:
        import akshare as ak
        df = ak.macro_china_pmi_yearly()
        # 列: 商品/日期/今值/预测值/前值 → 取制造业 PMI
        mfg = df[df['商品'].str.contains('制造业', na=False)].copy()
        mfg['month'] = pd.to_datetime(mfg['日期']).dt.strftime('%Y%m')
        mfg = mfg.sort_values('month').drop_duplicates('month', keep='last')
        out = pd.DataFrame({
            'month': mfg['month'],
            'pmi_mfg': pd.to_numeric(mfg['今值'], errors='coerce'),
        })
        out['source'] = 'akshare'
        return out
    except Exception as e:
        print(f'  [akshare] FAIL: {str(e)[:100]}')
        return None


def load_manual():
    if MANUAL.exists():
        df = pd.read_csv(MANUAL, dtype={'month': str})
        df['source'] = 'manual'
        return df
    return None


def main():
    print('[宏观采集] 多源拉取...')
    parts = []
    for fn in [collect_tushare, collect_akshare, load_manual]:
        try:
            r = fn()
            if r is not None and len(r):
                parts.append(r)
                src = r['source'].iloc[0]
                print(f'  [{src}] {len(r)} 行, 最新 {r["month"].max()}')
        except Exception as e:
            print(f'  FAIL: {str(e)[:80]}')
    if not parts:
        print('[宏观采集] ❌ 全部数据源失败, 不落盘')
        return

    merged = pd.concat(parts, ignore_index=True)
    # 🔴 按列合并 (非按行): 各源覆盖的列不同 (tushare 有 CPI/PPI/M1/M2 但 PMI 字段不可用;
    #    akshare 有 PMI 但滞后) → 每列独立选非空值, 优先级 tushare > akshare > manual
    prio = {'tushare': 0, 'akshare': 1, 'manual': 2}
    merged['_p'] = merged['source'].map(prio).fillna(9)
    merged = merged.sort_values(['month', '_p'])
    val_cols = [c for c in ['pmi_mfg', 'cpi_yoy', 'ppi_yoy', 'm1_yoy', 'm2_yoy']
                if c in merged.columns]
    for c in val_cols:
        merged[c] = pd.to_numeric(merged[c], errors='coerce')
    # 同月多源: 逐列优先取非空
    wide = merged.pivot_table(index='month', columns='source', values=val_cols,
                              aggfunc='first')
    out_cols = {}
    for c in val_cols:
        col_series = None
        for src in ['tushare', 'akshare', 'manual']:
            if (c, src) in wide.columns:
                s = wide[(c, src)]
                col_series = s if col_series is None else col_series.fillna(s)
        out_cols[c] = col_series
    merged = pd.DataFrame(out_cols)
    merged.index.name = 'month'
    merged = merged.reset_index()
    merged['source'] = 'tushare+akshare+manual'

    # 防前视: avail_month = 指标最早可用月 (发布月 = 数据月 + 发布滞后)
    for col, lag in RELEASE_LAG.items():
        merged[f'{col}_avail'] = merged['month'].apply(lambda m: _month_shift(m, lag))

    merged = merged[['month', 'pmi_mfg', 'cpi_yoy', 'ppi_yoy', 'm1_yoy', 'm2_yoy',
                     'pmi_mfg_avail', 'cpi_yoy_avail', 'ppi_yoy_avail',
                     'm1_yoy_avail', 'm2_yoy_avail', 'source']]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT, index=False)
    print(f'[宏观采集] 落盘 {OUT}: {len(merged)} 月, 最新数据月 {merged["month"].max()}')
    # 数据新鲜度检查
    now = datetime.now().strftime('%Y%m')
    latest = merged['month'].max()
    lag_months = (int(now[:4]) - int(latest[:4])) * 12 + (int(now[4:]) - int(latest[4:]))
    if lag_months >= 2:
        print(f'  ⚠️ 数据滞后 {lag_months} 个月 (最新 {latest} vs 当前 {now})')
        print('  → 因子提案保持 pending_data, 待数据源更新/手工 CSV 补充后消费')
    else:
        print(f'  ✅ 数据新鲜 ({lag_months} 个月滞后, 正常发布节奏内)')


if __name__ == '__main__':
    from pathlib import Path as _FuxiPath
    import sys as _FuxiSys
    _FuxiSys.path.insert(0, str(_FuxiPath(__file__).resolve().parents[1]))
    from fuxi_v1.legacy_writers import legacy_cli_guard as _FuxiGuard
    _fuxi_guard = _FuxiGuard(__file__)
    main()
