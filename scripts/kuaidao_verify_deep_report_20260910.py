# -*- coding: utf-8 -*-
"""
kuaidao_verify_deep_report_20260910.py
快刀 18 期深度对比报告【推】项 Tushare 复核 + P0 新规则数据验证

复核对象（报告模拟打分中标注【推】的约 25 处）：
  A. 门控预估净利润 (五粮液/迈瑞/京东方/华电辽能/伊利/药明/云南锗业) + 发帖日门控重算
  B. s9 人均持股 (白药/京东方/招行/伊利/牧原/紫金/中信)
  C. s5 分红与支付率 (五粮液/紫金)
  D. s4c 资本开支率 (白药)
  E. s8 增减持 12 月窗口 (白药/紫金/西矿/宁德)
  F. s7 三率同行对比 (京东方 vs 面板, 伊利 vs 乳品)
  G. s10 估值同行分位 (京东方/伊利/牧原/中信)
  H. s6 行业增速 (盐津/白药/京东方, 用同行营收增速近似)
  I. 周期股 PB/PS 历史分位 (P0-1 分流后门控重算)
  J. P0 排雷规则: R1 更名史 (神奇/华电辽能/云南锗业), R3 超额分红套现 (神奇), R2 洗澡同行对比 (神奇 2020)

口径与 skill 实现 (scripts/kuaidao_test_v1.py) 保持一致:
  - income: total_revenue, n_income_attr_p (万元)
  - balancesheet: money_cap, trad_asset, st_borr, lt_borr, bond_payable, non_cur_liab_due_1y, total_share(股)
  - cashflow: n_cashflow_act, c_pay_acq_const_fiolta
  - dividend: cash_div (元/股); 同年多条取 max
  - s9: (total_share - sum(top10 hold_amount)) * close / holder_num / 1e4 (万)
  - 预估净利 = 2025年报净利 x (1 + 2026Q1同比) ; 发帖日市值用 post_date 最近交易日 daily_basic

用法: python kuaidao_verify_deep_report_20260910.py
输出: data/kuaidao_test/verify_deep/verify_result.json + verify_report.md
"""
import sys, os, json, time, traceback
from pathlib import Path
sys.path.insert(0, 'e:/quant/scripts')
sys.path.insert(0, 'e:/quant')
os.chdir('e:/quant')

import pandas as pd
import numpy as np
from credentials import get_tushare_api

OUT_DIR = Path('e:/quant/data/kuaidao_test/verify_deep')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (ts_code, 名称, 发帖日 YYYYMMDD, 周期股, 金融股)
STOCKS = [
    ('002847.SZ', '盐津铺子',  '20260518', False, False),
    ('000858.SZ', '五粮液',    '20260529', False, False),
    ('300750.SZ', '宁德时代',  '20260531', False, False),
    ('000538.SZ', '云南白药',  '20260605', False, False),
    ('000725.SZ', '京东方A',   '20260605', True,  False),
    ('600396.SH', '华电辽能',  '20260607', False, False),
    ('600036.SH', '招商银行',  '20260611', False, True),
    ('300760.SZ', '迈瑞医疗',  '20260616', False, False),
    ('600887.SH', '伊利股份',  '20260706', False, False),
    ('603259.SH', '药明康德',  '20260711', False, False),
    ('002714.SZ', '牧原股份',  '20260725', True,  False),
    ('002428.SZ', '云南锗业',  '20260808', True,  False),
    ('601899.SH', '紫金矿业',  '20260809', True,  False),
    ('600613.SH', '神奇制药',  '20260822', False, False),
    ('600030.SH', '中信证券',  '20260830', True,  True),
    ('601168.SH', '西部矿业',  '20260905', True,  False),
]

# 同行池 (s6/s7/s10)
PEERS = {
    '面板':   ['000100.SZ', '000050.SZ', '002387.SZ', '600707.SH', '688055.SH'],
    '乳品':   ['600597.SH', '002946.SZ', '600429.SH', '600882.SH', '600419.SH'],
    '生猪':   ['300498.SZ', '000876.SZ', '002157.SZ', '002567.SZ', '603477.SH'],
    '券商':   ['601688.SH', '601211.SH', '600999.SH', '000776.SZ', '600958.SH', '300059.SZ'],
    '中药':   ['600436.SH', '000423.SZ', '600085.SH', '600332.SH', '000999.SZ'],
    '休闲食品': ['002582.SZ', '300783.SZ', '603719.SH', '603517.SH', '002956.SZ', '002991.SZ'],
}
STOCK_PEER = {
    '002847.SZ': ('休闲食品', None),          # s6
    '000538.SZ': ('中药', None),              # s6
    '000725.SZ': ('面板', '000100.SZ'),       # s6/s7/s10
    '600887.SH': ('乳品', '600597.SH'),       # s7/s10
    '002714.SZ': ('生猪', '300498.SZ'),       # s10
    '600030.SH': ('券商', '601688.SH'),       # s10
    '600613.SH': ('中药', None),              # R2 洗澡同行
}

# 报告【推】期望值 (用于对比)
EXPECT = {
    '预估净利_亿': {
        '000858.SZ': 140.0, '300760.SZ': 74.5, '000725.SZ': 102.0, '600396.SH': 2.2,
        '600887.SH': 139.0, '603259.SH': 243.0, '002428.SZ': 0.3,
    },
    's9人均_万': {
        '000538.SZ': 29.8, '000725.SZ': 13.5, '600036.SH': 153.0, '600887.SH': 29.8,
        '002714.SZ': 36.7, '601899.SH': 116.0, '600030.SH': 43.0,
    },
    's4c资本开支率': {'000538.SZ': 0.11},
}

pro = get_tushare_api()
CAL = pro.trade_cal(exchange='SSE', start_date='20260101', end_date='20260910',
                    fields='cal_date,is_open')


def safe_call(fn, *args, retries=2, **kwargs):
    for i in range(retries + 1):
        try:
            df = fn(*args, **kwargs)
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            if i == retries:
                print(f'  [WARN] {fn.__name__} {args[:2]} failed: {e}')
                return pd.DataFrame()
            time.sleep(0.8)


def last_trade_date(post_date):
    """post_date 当天或之前的最近交易日"""
    cal = CAL[CAL['cal_date'].astype(str) <= post_date]
    if len(cal) == 0:
        return post_date
    return str(cal['cal_date'].max())


def load_frame(ts_code, kind):
    """带磁盘缓存的单股接口数据"""
    fp = OUT_DIR / f'{kind}_{ts_code.replace(".", "_")}.csv'
    if fp.exists():
        return pd.read_csv(fp)
    df = pd.DataFrame()
    if kind == 'income':
        df = safe_call(pro.income, ts_code=ts_code, start_date='20190101', end_date='20260910',
                       fields='ts_code,ann_date,end_date,report_type,total_revenue,n_income_attr_p')
        if len(df) > 0:
            df = df[df['report_type'] == '1']
    elif kind == 'fina':
        df = safe_call(pro.fina_indicator, ts_code=ts_code, start_date='20190101', end_date='20260910',
                       fields='ts_code,ann_date,end_date,roe,roe_dt,grossprofit_margin,netprofit_margin,or_yoy,netprofit_yoy')
    elif kind == 'bal':
        df = safe_call(pro.balancesheet, ts_code=ts_code, start_date='20190101', end_date='20260910',
                       fields='ts_code,ann_date,end_date,report_type,money_cap,trad_asset,st_borr,lt_borr,'
                              'bond_payable,non_cur_liab_due_1y,accounts_receiv,contract_liab,adv_receipts,'
                              'goodwill,total_assets,fix_assets,cip,intang_assets,total_share')
        if len(df) > 0:
            df = df[df['report_type'] == '1']
    elif kind == 'cf':
        df = safe_call(pro.cashflow, ts_code=ts_code, start_date='20190101', end_date='20260910',
                       fields='ts_code,ann_date,end_date,report_type,c_fr_sale_sg,n_cashflow_act,c_pay_acq_const_fiolta')
        if len(df) > 0:
            df = df[df['report_type'] == '1']
    elif kind == 'div':
        df = safe_call(pro.dividend, ts_code=ts_code, fields='ts_code,end_date,ann_date,cash_div,div_proc')
    elif kind == 'daily':
        df = safe_call(pro.daily_basic, ts_code=ts_code, start_date='20230101', end_date='20260910',
                       fields='ts_code,trade_date,close,pe,pe_ttm,pb,ps,ps_ttm,total_mv,circ_mv,total_share')
    elif kind == 'top10':
        df = safe_call(pro.top10_holders, ts_code=ts_code, period='20260630',
                       fields='ts_code,ann_date,end_date,holder_name,hold_ratio,hold_amount,holder_type')
    elif kind == 'holder':
        df = safe_call(pro.stk_holdernumber, ts_code=ts_code, start_date='20250901', end_date='20260910',
                       fields='ts_code,ann_date,end_date,holder_num')
    elif kind == 'trade':
        df = safe_call(pro.stk_holdertrade, ts_code=ts_code, start_date='20250901', end_date='20260910',
                       fields='ts_code,ann_date,holder_name,in_de,change_vol,change_ratio')
    elif kind == 'name':
        df = safe_call(pro.namechange, ts_code=ts_code, fields='ts_code,name,start_date,end_date,change_reason')
    if len(df) > 0:
        df.to_csv(fp, index=False)
    return df


def annual(df, yyyy):
    """取某年度 1231 记录"""
    if len(df) == 0:
        return None
    d = df[df['end_date'].astype(str) == f'{yyyy}1231']
    if len(d) == 0:
        return None
    return d.sort_values('ann_date', na_position='first').iloc[-1]


def num(x):
    try:
        v = float(x)
        return v if pd.notna(v) else None
    except Exception:
        return None


def pct(x):
    return f'{x*100:.2f}%' if x is not None else 'N/A'


def main():
    results = {}
    daily_all = {}   # ts_code -> 发帖日 daily_basic row
    frames = {}      # ts_code -> {kind: df}
    all_codes = [s[0] for s in STOCKS] + [p for ps in PEERS.values() for p in ps]

    print('=== Step 1: 拉取核心数据 (目标 16 只 + 同行 %d 只) ===' % len(all_codes))
    for i, code in enumerate(dict.fromkeys(all_codes)):
        print(f'[{i+1}/{len(dict.fromkeys(all_codes))}] {code}')
        f = {}
        for kind in ('income', 'fina', 'bal', 'cf', 'div', 'daily', 'name'):
            f[kind] = load_frame(code, kind)
        # top10/holder/trade 只对目标股 (同行不需要)
        f['top10'] = load_frame(code, 'top10') if code in [s[0] for s in STOCKS] else pd.DataFrame()
        f['holder'] = load_frame(code, 'holder') if code in [s[0] for s in STOCKS] else pd.DataFrame()
        f['trade'] = load_frame(code, 'trade') if code in [s[0] for s in STOCKS] else pd.DataFrame()
        frames[code] = f
        time.sleep(0.12)

    print('=== Step 2: 逐项计算 ===')
    stock_info = {s[0]: {'name': s[1], 'post_date': s[2], 'cycle': s[3], 'fin': s[4]} for s in STOCKS}

    for code, info in stock_info.items():
        r = {'name': info['name'], 'post_date': info['post_date']}
        f = frames[code]
        td = last_trade_date(info['post_date'])
        r['trade_date'] = td
        dly = f['daily']
        row = None
        if len(dly) > 0:
            dd = dly[dly['trade_date'].astype(str) <= td]
            if len(dd) > 0:
                row = dd.sort_values('trade_date').iloc[-1]
        daily_all[code] = row
        if row is not None:
            r['close'] = num(row['close'])
            r['total_mv_亿'] = num(row['total_mv']) / 1e4 if num(row['total_mv']) else None
            r['pe_ttm'] = num(row['pe_ttm'])
            r['pb'] = num(row['pb'])
            r['ps_ttm'] = num(row['ps_ttm'])
        # 年报
        inc25, inc24, inc23 = annual(f['income'], 2025), annual(f['income'], 2024), annual(f['income'], 2023)
        bal25 = annual(f['bal'], 2025)
        fina25 = annual(f['fina'], 2025)
        cf25 = annual(f['cf'], 2025)
        r['np_2025_亿'] = num(inc25['n_income_attr_p']) / 1e8 if inc25 is not None and num(inc25.get('n_income_attr_p')) is not None else None
        r['np_2024_亿'] = num(inc24['n_income_attr_p']) / 1e8 if inc24 is not None else None
        r['np_2023_亿'] = num(inc23['n_income_attr_p']) / 1e8 if inc23 is not None else None
        r['rev_2025_亿'] = num(inc25['total_revenue']) / 1e8 if inc25 is not None else None
        r['roe_2025'] = num(fina25['roe']) if fina25 is not None else None
        # Q1 yoy
        q1_26 = f['income'][f['income']['end_date'].astype(str) == '20260331'] if len(f['income']) else None
        q1_25 = f['income'][f['income']['end_date'].astype(str) == '20250331'] if len(f['income']) else None
        q1yoy = None
        if q1_26 is not None and len(q1_26) > 0 and q1_25 is not None and len(q1_25) > 0:
            a = num(q1_26.sort_values('ann_date').iloc[-1].get('n_income_attr_p'))
            b = num(q1_25.sort_values('ann_date').iloc[-1].get('n_income_attr_p'))
            if a and b and b != 0:
                q1yoy = a / b - 1
        r['q1_yoy'] = q1yoy
        if r['np_2025_亿'] is not None and q1yoy is not None:
            r['est_np_亿'] = r['np_2025_亿'] * (1 + q1yoy)
        # 净现金 + 调整市值
        if bal25 is not None:
            money = num(bal25.get('money_cap')) or 0
            trad = num(bal25.get('trad_asset')) or 0
            debt = (num(bal25.get('st_borr')) or 0) + (num(bal25.get('lt_borr')) or 0) + \
                   (num(bal25.get('bond_payable')) or 0) + (num(bal25.get('non_cur_liab_due_1y')) or 0)
            r['net_cash_亿'] = (money + trad - debt) / 1e8 if not info['fin'] else 0.0
            r['fix_ratio'] = (num(bal25.get('fix_assets')) / num(bal25.get('total_assets'))) if num(bal25.get('total_assets')) else None
            heavy = (num(bal25.get('fix_assets')) or 0) + (num(bal25.get('cip')) or 0) + (num(bal25.get('intang_assets')) or 0)
            r['heavy_ratio'] = (heavy / num(bal25.get('total_assets'))) if num(bal25.get('total_assets')) else None  # P0-4 重资产口径 2026-09-10 扩展: 固资+在建+无形
            r['total_share_亿股'] = num(bal25.get('total_share')) / 1e8 if num(bal25.get('total_share')) else None
        if r.get('total_mv_亿') is not None and r.get('net_cash_亿') is not None:
            r['adj_mv_亿'] = r['total_mv_亿'] - r['net_cash_亿']
        # PR (非金融非周期)
        if not info['fin'] and not info['cycle']:
            if r.get('adj_mv_亿') and r.get('est_np_亿') and r.get('roe_2025'):
                r['PR'] = r['adj_mv_亿'] / r['est_np_亿'] / r['roe_2025']
        # PS/PB 历史分位 (周期股; 3.5年与5年双窗口)
        if len(dly) > 0 and row is not None:
            hist = dly[dly['trade_date'].astype(str) <= td].sort_values('trade_date')
            hist5 = hist[hist['trade_date'].astype(str) >= '20210101']
            if len(hist) >= 30:
                for key, col in (('pb', 'pb'), ('ps_ttm', 'ps_ttm')):
                    vals = hist[col].dropna()
                    cur = num(row[col])
                    if cur is not None and len(vals) > 0:
                        r[f'{key}_hist_pct'] = float((vals <= cur).mean())
                    if len(hist5) >= 30:
                        vals5 = hist5[col].dropna()
                        if cur is not None and len(vals5) > 0:
                            r[f'{key}_hist5_pct'] = float((vals5 <= cur).mean())
        # s4c
        if cf25 is not None:
            capex = num(cf25.get('c_pay_acq_const_fiolta'))
            ocf = num(cf25.get('n_cashflow_act'))
            if capex and ocf and ocf != 0:
                r['s4c_capex_ratio'] = capex / ocf
            r['s4b_ocf_np'] = (ocf / (r['np_2025_亿'] * 1e8)) if (ocf and r.get('np_2025_亿')) else None
        # s5 分红
        div = f['div']
        if len(div) > 0 and r.get('total_share_亿股') and inc25 is not None:
            pay_years = {}
            for y in ('2022', '2023', '2024', '2025'):
                yd = div[div['end_date'].astype(str).str.startswith(y)]
                if len(yd) > 0 and yd['cash_div'].notna().any() and (yd['cash_div'] > 0).any():
                    max_div = yd['cash_div'].max()
                    np_y = None
                    ia = annual(f['income'], int(y))
                    if ia is not None:
                        np_y = num(ia.get('n_income_attr_p'))
                    if np_y and np_y > 0:
                        payout = max_div * r['total_share_亿股'] * 1e8 / np_y
                        pay_years[y] = {'payout': payout}
            if pay_years:
                r['s5_payout_by_year'] = {y: round(v['payout'], 3) for y, v in pay_years.items()}
        # s9
        top10 = f['top10']
        holder = f['holder']
        if len(top10) > 0 and len(holder) > 0 and r.get('total_share_亿股') and row is not None:
            top10_amt = top10['hold_amount'].dropna().sum()
            hn = holder.dropna(subset=['holder_num']).sort_values('end_date')
            hn = hn[hn['end_date'].astype(str) <= td]
            if len(hn) > 0:
                holder_num = num(hn.iloc[-1]['holder_num'])
                ts = r['total_share_亿股'] * 1e8
                if top10_amt > 0 and top10_amt < ts:
                    tradable = ts - top10_amt
                else:
                    tradable = ts * (1 - top10['hold_ratio'].dropna().sum() / 100.0)
                if holder_num:
                    r['s9_per_capita_万'] = tradable * r['close'] / holder_num / 1e4
        # s8
        tr = f['trade']
        if len(tr) > 0:
            r['s8_in_n'] = int((tr['in_de'] == 'IN').sum())
            r['s8_de_n'] = int((tr['in_de'] == 'DE').sum())
            r['s8_holders'] = tr['holder_name'].dropna().unique().tolist()[:8]
        # R1 更名史 (修正窗口: 曾ST不限年限 + 近20年变更>=3次)
        nm = f['name']
        if len(nm) > 0:
            full = nm.copy()
            r['R1_ever_st_any'] = bool((full['name'].astype(str).str.contains('ST', case=False, na=False)).any())
            recent = nm[(nm['end_date'].fillna('99991231').astype(str) >= '20060101') |
                        (nm['start_date'].astype(str) >= '20060101')]
            r['R1_changes_20y'] = len(recent)
            r['R1_names'] = nm['name'].tolist()
        results[code] = r

    # ---- 同行对比项 (s6/s7/s10) ----
    print('=== Step 3: 同行对比 ===')
    for code, (peer_key, _) in STOCK_PEER.items():
        peer_codes = PEERS[peer_key]
        r = results[code]
        r['peer_key'] = peer_key
        # s6 行业增速: 同行 2025 vs 2024 营收合计 yoy (近似)
        revs = {}
        for pc in peer_codes + [code]:
            inc25 = annual(frames[pc]['income'], 2025)
            inc24 = annual(frames[pc]['income'], 2024)
            if inc25 is not None and inc24 is not None:
                v25, v24 = num(inc25.get('total_revenue')), num(inc24.get('total_revenue'))
                if v25 and v24:
                    revs[pc] = (v25, v24)
        if revs:
            s25 = sum(v[0] for v in revs.values())
            s24 = sum(v[1] for v in revs.values())
            if s24:
                r['s6_peer_rev_yoy'] = s25 / s24 - 1
        # s7: 三率对比 (目标 vs 同行, 2025 年报)
        r['s7_metrics'] = {}
        for pc in peer_codes + [code]:
            fa = annual(frames[pc]['fina'], 2025)
            if fa is not None:
                r['s7_metrics'][pc] = {
                    'gross': num(fa.get('grossprofit_margin')),
                    'net': num(fa.get('netprofit_margin')),
                    'roe': num(fa.get('roe')),
                }
        # s10: 发帖日估值同行分位 (稳定盈利=PR口径 pe_ttm/roe; 金融/周期用PB近似并注明)
        tgt = daily_all.get(code)
        if tgt is not None and num(tgt.get('pb')) is not None:
            def _s10_val(c):
                row_p = daily_all.get(c)
                if row_p is None and c in frames:
                    dly_p = frames[c]['daily']
                    if len(dly_p) > 0:
                        dd = dly_p[dly_p['trade_date'].astype(str) <= results[code]['trade_date']]
                        if len(dd) > 0:
                            row_p = dd.sort_values('trade_date').iloc[-1]
                            daily_all[c] = row_p
                if row_p is None:
                    return None
                fa = annual(frames[c]['fina'], 2025)
                roe = num(fa.get('roe')) if fa is not None else None
                pe = num(row_p.get('pe_ttm'))
                is_fin_c = c in [s[0] for s in STOCKS if s[4]] or c in ('601688.SH', '601211.SH', '600999.SH', '000776.SZ', '600958.SH', '300059.SZ')
                if not is_fin_c and pe and roe and roe > 0:
                    return pe / roe  # PR (ROE为百分数)
                return num(row_p.get('pb'))
            vals = []
            for c in peer_codes + [code]:
                v = _s10_val(c)
                if v is not None:
                    vals.append(v)
            if len(vals) >= 3:
                tgt_v = vals[-1]
                r['s10_peer_pct'] = float(pd.Series(vals).rank(pct=True)[len(vals) - 1])
                r['s10_target_val'] = round(tgt_v, 2)
                r['s10_metric'] = 'PR' if not info['fin'] else 'PB'

    # ---- 期望值对比 ----
    print('=== Step 4: 输出 ===')
    comparisons = []
    for code, exp in EXPECT['预估净利_亿'].items():
        got = results[code].get('est_np_亿')
        ok = bool(abs(got - exp) / exp < 0.25) if got else False
        comparisons.append({'项': '预估净利', '股票': stock_info[code]['name'],
                            '报告推断': exp, 'Tushare': round(got, 1) if got else None,
                            '一致': ok})
    for code, exp in EXPECT['s9人均_万'].items():
        got = results[code].get('s9_per_capita_万')
        ok = bool(abs(got - exp) / exp < 0.3) if got else False
        comparisons.append({'项': 's9人均持股', '股票': stock_info[code]['name'],
                            '报告推断': exp, 'Tushare': round(got, 1) if got else None,
                            '一致': ok})
    for code, exp in EXPECT['s4c资本开支率'].items():
        got = results[code].get('s4c_capex_ratio')
        ok = bool(abs(got - exp) / exp < 0.4) if got else False
        comparisons.append({'项': 's4c资本开支率', '股票': stock_info[code]['name'],
                            '报告推断': exp, 'Tushare': round(got, 3) if got else None,
                            '一致': ok})

    out_json = {'results': results, 'comparisons': comparisons}
    with open(OUT_DIR / 'verify_result.json', 'w', encoding='utf-8') as fp:
        json.dump(out_json, fp, ensure_ascii=False, indent=2, default=str)

    # Markdown 报告
    lines = ['# 快刀深度报告【推】项 Tushare 复核结果', '',
             f'> 生成时间: {time.strftime("%Y-%m-%d %H:%M")}。口径与 skill 实现一致；发帖日估值用最近交易日 daily_basic。', '']
    lines.append('## 一、预估净利润复核（门控分母）')
    lines.append('| 股票 | 2025净利(亿) | Q1同比 | 预估净利(亿) | 报告推断(亿) | 一致 |')
    lines.append('|---|---|---|---|---|---|')
    for code, exp in EXPECT['预估净利_亿'].items():
        r = results[code]
        got = r.get('est_np_亿')
        ok = abs(got - exp) / exp < 0.25 if got else False
        lines.append(f"| {r['name']} | {r.get('np_2025_亿')} | {pct(r.get('q1_yoy'))} | {round(got,1) if got else 'N/A'} | {exp} | {'✅' if ok else '❌'} |")
    lines.append('')
    lines.append('## 二、发帖日门控重算（含 P0-1 周期/金融分流）')
    lines.append('| 股票 | 发帖日市值(亿) | 净现金(亿) | 调整市值(亿) | ROE% | PR | PB历史分位(3.5y/5y) | PS历史分位 | 分流 |')
    lines.append('|---|---|---|---|---|---|---|---|---|')
    for code in [s[0] for s in STOCKS]:
        r = results[code]
        info = stock_info[code]
        lane = '周期(PS/PB)' if info['cycle'] else ('金融(PR豁免净现金)' if info['fin'] else 'PR')
        pb5 = f"{pct(r['pb_hist_pct']) if r.get('pb_hist_pct') is not None else 'N/A'}/{pct(r['pb_hist5_pct']) if r.get('pb_hist5_pct') is not None else 'N/A'}"
        lines.append(f"| {r['name']} | {r.get('total_mv_亿')} | {r.get('net_cash_亿') and round(r['net_cash_亿'],1)} | {r.get('adj_mv_亿') and round(r['adj_mv_亿'],1)} | {r.get('roe_2025')} | {r.get('PR') and round(r['PR'],2)} | {pb5} | {r.get('ps_ttm_hist_pct') and pct(r['ps_ttm_hist_pct'])} | {lane} |")
    lines.append('')
    lines.append('## 三、s9 人均持股复核')
    lines.append('| 股票 | 人均(万) | 报告推断(万) | 一致 |')
    lines.append('|---|---|---|---|')
    for code, exp in EXPECT['s9人均_万'].items():
        r = results[code]
        got = r.get('s9_per_capita_万')
        ok = abs(got - exp) / exp < 0.3 if got else False
        lines.append(f"| {r['name']} | {round(got,1) if got else 'N/A'} | {exp} | {'✅' if ok else '❌'} |")
    lines.append('')
    lines.append('## 四、s4c / s5 / s8 复核')
    for code in ('000538.SZ', '000858.SZ', '601899.SH', '601168.SH', '300750.SZ'):
        r = results[code]
        lines.append(f"**{r['name']}**: s4c资本开支率={r.get('s4c_capex_ratio') and pct(r['s4c_capex_ratio'])}, "
                     f"s4b净现比={r.get('s4b_ocf_np') and round(r['s4b_ocf_np'],2)}, "
                     f"s5支付率={r.get('s5_payout_by_year')}, s8增减持={r.get('s8_in_n',0)}IN/{r.get('s8_de_n',0)}DE {r.get('s8_holders','')}")
    lines.append('')
    lines.append('## 五、s7 三率同行对比（2025 年报）')
    for code in ('000725.SZ', '600887.SH'):
        r = results[code]
        lines.append(f"**{r['name']}** ({r.get('peer_key')}):")
        lines.append('| 代码 | 毛利率% | 净利率% | ROE% |')
        lines.append('|---|---|---|---|')
        for pc, m in r.get('s7_metrics', {}).items():
            nm = stock_info.get(pc, {}).get('name', '') if pc in stock_info else ''
            lines.append(f"| {pc} {nm} | {m['gross']} | {m['net']} | {m['roe']} |")
        lines.append('')
    lines.append('## 六、s10 估值同行分位 + s6 行业增速（同行营收近似）')
    lines.append('| 股票 | 同行 | s10口径 | 目标值 | 同行分位 | 同行营收增速(25 vs 24) |')
    lines.append('|---|---|---|---|---|---|')
    for code in ('000725.SZ', '600887.SH', '002714.SZ', '600030.SH', '002847.SZ', '000538.SZ'):
        r = results[code]
        lines.append(f"| {r['name']} | {r.get('peer_key')} | {r.get('s10_metric', 'N/A')} | {r.get('s10_target_val', 'N/A')} | {r.get('s10_peer_pct') and pct(r['s10_peer_pct'])} | {r.get('s6_peer_rev_yoy') and pct(r['s6_peer_rev_yoy'])} |")
    lines.append('')
    lines.append('## 七、P0 排雷规则数据验证')
    lines.append('### R1 更名史（曾ST[不限年限] + 近20年变更次数）')
    lines.append('| 股票 | 曾ST(全历史) | 近20年变更次数 | 名称轨迹 | R1判定 |')
    lines.append('|---|---|---|---|---|')
    for code in ('600613.SH', '600396.SH', '002428.SZ', '000858.SZ', '002847.SZ'):
        r = results[code]
        hit = r.get('R1_ever_st_any') and r.get('R1_changes_20y', 0) >= 3
        lines.append(f"| {r['name']} | {r.get('R1_ever_st_any')} | {r.get('R1_changes_20y', 0)} | {r.get('R1_names', [])} | {'✅命中' if hit else '不命中'} |")
    lines.append('')
    lines.append('### R3 超额分红套现（神奇制药 2022-2025 支付率 + 近12月减持）')
    r = results['600613.SH']
    lines.append(f"- 近4年支付率: {r.get('s5_payout_by_year')}")
    lines.append(f"- 近12月增减持: {r.get('s8_in_n',0)} IN / {r.get('s8_de_n',0)} DE, 名单: {r.get('s8_holders','')}")
    pay = r.get('s5_payout_by_year', {})
    over100 = [y for y, v in pay.items() if v > 1.0]
    hit3 = len(over100) >= 2 and r.get('s8_de_n', 0) > 0
    lines.append(f"- R3 判定: 支付率>100% 年份={over100}, 减持DE={r.get('s8_de_n',0)} → {'✅命中' if hit3 else '不命中'}")
    lines.append('')
    lines.append('### R2 财务洗澡（神奇制药 2020 亏损 vs 中药同行 2020 净利）')
    r = results['600613.SH']
    np2020 = None
    inc2020 = annual(frames['600613.SH']['income'], 2020)
    if inc2020 is not None:
        np2020 = num(inc2020.get('n_income_attr_p')) / 1e8
    peers_np = []
    for pc in PEERS['中药']:
        i20 = annual(frames[pc]['income'], 2020)
        if i20 is not None:
            v = num(i20.get('n_income_attr_p'))
            if v is not None:
                peers_np.append(v / 1e8)
    lines.append(f"- 神奇制药 2020 净利: {np2020} 亿")
    lines.append(f"- 中药同行 2020 净利: {peers_np}, 中位数: {np.median(peers_np) if peers_np else 'N/A'} 亿")
    lines.append(f"- R2 判定: {'命中(自己亏/同行中位赚)' if (np2020 is not None and np2020 < 0 and peers_np and np.median(peers_np) > 0) else '不命中或数据不足'}")
    lines.append('')

    with open(OUT_DIR / 'verify_report.md', 'w', encoding='utf-8') as fp:
        fp.write('\n'.join(lines))
    print('DONE. json + md written to', OUT_DIR)
    print(json.dumps(comparisons, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
