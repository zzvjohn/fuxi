# -*- coding: utf-8 -*-
"""快刀斩股 skill 全市场 3 期时点回测引擎 v1 (2026-09-10)
对齐修改后 skill 全流程: 排雷门控(R1/R2/R3) -> 价格门控(周期A方案/洗澡例外) -> 十项打分 -> 裁决
3 个时点: 2024-04-30(2023年报) / 2025-04-30(2024年报) / 2026-04-30(2025年报)
每期选池后等权持有至下期, 拼接净值曲线 vs 沪深300。无前视: 全部数据 ann_date<=run_date。
"""
import sys, os, json, glob, math
import numpy as np
import pandas as pd

sys.path.insert(0, 'e:/quant/scripts')
os.chdir('e:/quant')
from kuaidao_test_v1 import (INDUSTRY_META, CYCLE_POSITION, GDP_GROWTH,
                             CYCLICAL_INDUSTRIES, FINANCIAL_INDS)

FM = 'data/kuaidao_test/full_market'
OUT = 'data/kuaidao_test/full_backtest'
os.makedirs(OUT, exist_ok=True)

# 价格门控 PR 上限 (2026-09-11 用户钦定 = 1.5)。曾试 2.0 并重跑全量:
#   PR<=1.5 +39.14%/0.693/-26.69%  vs  PR<=2 +32.58%/0.601/-25.98%
# 实测 1.5 更优 (t≈0.94 不显著, 记为倾向性证据) → 回退到 1.5。
# 改此值会换池成员 ⇒ 必须同步核对 full_market/pool_ohlc 覆盖, 否则静默低估。
PR_GATE = float(os.environ.get('KUAIDAO_PR_GATE', 1.5))

PERIODS = [
    dict(tag='P2023', report_end='20231231', run_date='20240430', bt_start='20240501', bt_end='20250430'),
    dict(tag='P2024', report_end='20241231', run_date='20250430', bt_start='20250501', bt_end='20260430'),
    dict(tag='P2025', report_end='20251231', run_date='20260430', bt_start='20260501', bt_end='20260910'),
]

# 重资产豁免"无形不计入"行业清单 (2026-09-10 用户钦定): 医药/软件/传媒类无形主体=并购溢价/批文/版权
HEAVY_INTAN_EXCLUDE = {
    # 东财行业名
    '化学制药', '生物制药', '中成药', '医药商业', '医疗服务', '医疗器械', '中药生产',
    '软件开发', 'IT服务', '互联网服务', '游戏', '影视院线', '文化传媒', '出版', '通信服务',
    # 申万一/二级名
    '医药生物', '计算机', '传媒',
}
# 周期行业关键词 (stock_basic.industry 东财分类)
CYCLE_KW = ['煤炭', '有色', '钢铁', '光伏', '养殖', '农牧饲渔', '面板', '证券', '航运', '港口',
            '石油', '化纤', '农药', '化肥', '小金属', '工业金属', '贵金属', '能源金属', '采掘',
            '焦炭', '水泥', '玻璃', '造纸', '普钢', '特钢', '化学原料', '化学制品', '燃气', '电力']
FIN_KW = ['银行', '保险', '证券', '券商', '信托', '多元金融']


def load_universe():
    sb = pd.read_csv(f'{FM}/full_stock_basic.csv', dtype={'ts_code': str, 'list_date': str})
    sw = pd.read_csv(f'{FM}/full_sw.csv', dtype=str)
    sw_map = sw.drop_duplicates('ts_code').set_index('ts_code')
    universe = {}
    for _, r in sb.iterrows():
        c = r['ts_code']
        ind = str(r.get('industry', ''))
        l1 = str(sw_map.loc[c, 'l1_name']) if c in sw_map.index else ''
        l2 = str(sw_map.loc[c, 'l2_name']) if c in sw_map.index else ''
        universe[c] = dict(
            name=str(r['name']), industry=ind, list_date=str(r['list_date']),
            l1=l1, l2=l2,
            is_fin=any(k in ind for k in FIN_KW) or any(k in l1 for k in FIN_KW),
            is_cyc=any(k in ind for k in CYCLE_KW),
            intan_exclude=(ind in HEAVY_INTAN_EXCLUDE) or (l1 in HEAVY_INTAN_EXCLUDE) or (l2 in HEAVY_INTAN_EXCLUDE),
        )
    return universe


def load_tables():
    """逐只读缓存 -> {code: {kind: DataFrame}}"""
    tables = {c: {} for c in pd.read_csv(f'{FM}/full_stock_basic.csv', dtype={'ts_code': str})['ts_code']}
    for kind, sub in [('income', 'per_stock_income'), ('fina', 'per_stock_fina'), ('bal', 'per_stock_bal')]:
        for f in glob.glob(f'{FM}/{sub}/*.csv'):
            code = os.path.basename(f).split(f'{kind}_')[-1].replace('.csv', '').replace('_', '.')
            try:
                d = pd.read_csv(f, dtype={'ts_code': str, 'end_date': str, 'ann_date': str})
                if code in tables:
                    tables[code][kind] = d
            except Exception:
                pass
    return tables


def load_market_tables():
    mt = {}
    mt['nm'] = pd.read_csv(f'{FM}/full_nm.csv', dtype=str) if os.path.exists(f'{FM}/full_nm.csv') else pd.DataFrame()
    mt['div'] = pd.read_csv(f'{FM}/full_div.csv', dtype={'ts_code': str, 'end_date': str, 'ann_date': str}) if os.path.exists(f'{FM}/full_div.csv') else pd.DataFrame()
    mt['trades'] = pd.read_csv(f'{FM}/full_trades.csv', dtype={'ts_code': str, 'ann_date': str}) if os.path.exists(f'{FM}/full_trades.csv') else pd.DataFrame()
    mt['rep'] = pd.read_csv(f'{FM}/full_rep.csv', dtype={'ts_code': str, 'ann_date': str}) if os.path.exists(f'{FM}/full_rep.csv') else pd.DataFrame()
    mt['top10'] = pd.read_csv(f'{FM}/full_top10.csv', dtype={'ts_code': str, 'end_date': str, 'ann_date': str})
    mt['holder'] = pd.read_csv(f'{FM}/full_holder.csv', dtype={'ts_code': str, 'end_date': str, 'ann_date': str})
    mt['db'] = pd.read_csv(f'{FM}/full_db_daily.csv', dtype={'ts_code': str, 'trade_date': str}) if os.path.exists(f'{FM}/full_db_daily.csv') else pd.DataFrame()
    mt['bench'] = pd.read_csv('data/raw/index_000300_daily.csv', dtype={'trade_date': str}) if os.path.exists('data/raw/index_000300_daily.csv') else pd.DataFrame()
    return mt


def _num(x):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        return float(x)
    except Exception:
        return None


def stock_snapshot(code, tab, mt, uni, P, r2_med, fina_cross):
    """时点快照: 全部数据 ann_date<=run_date (无前视)"""
    rd, re_ = P['run_date'], P['report_end']
    inc = tab.get('income')
    fin = tab.get('fina')
    bal = tab.get('bal')
    s = dict(code=code, **{k: uni[code].get(k) for k in ('name', 'industry', 'l1', 'l2', 'is_fin', 'is_cyc', 'intan_exclude', 'list_date')})
    if inc is None or len(inc) == 0:
        return None
    inc = inc[(inc['ann_date'].astype(str) <= rd) & (inc['end_date'].astype(str) <= re_)].copy()
    annual = inc[inc['end_date'].astype(str).str.endswith('1231')].sort_values('end_date', ascending=False).drop_duplicates('end_date')
    if len(annual) < 3:
        return None
    s['annual'] = annual
    s['np3'] = [_num(x) for x in annual['n_income_attr_p'].head(3)]          # 近3年年报归母净利 [最新,...]
    s['rev3'] = [_num(x) for x in annual['revenue'].head(3)]
    s['np5'] = [_num(x) for x in annual['n_income_attr_p'].head(5)]
    s['imp'] = _num(annual.iloc[0].get('assets_impair_loss'))               # 当期减值
    # 洗澡重述信号: 当年年报营收 < 当年前三季度累计营收 (数学不可能 -> 收入重述)
    y = re_[:4]
    q3 = inc[inc['end_date'].astype(str) == y + '0930']
    s['restated'] = False
    if len(q3) and _num(q3.iloc[0]['revenue']) and _num(annual.iloc[0]['revenue']):
        s['restated'] = _num(annual.iloc[0]['revenue']) < _num(q3.iloc[0]['revenue']) * 0.98
    # fina 当期
    if fin is not None and len(fin):
        f = fin[(fin['ann_date'].astype(str) <= rd) & (fin['end_date'].astype(str) <= re_)]
        fa = f[f['end_date'].astype(str).str.endswith('1231')].sort_values('end_date', ascending=False).drop_duplicates('end_date')
        s['fina_ann'] = fa
        s['roe3'] = [_num(x) for x in fa['roe'].head(3)]
        s['gm'] = _num(fa.iloc[0].get('grossprofit_margin')) if len(fa) else None
        s['nm'] = _num(fa.iloc[0].get('netprofit_margin')) if len(fa) else None
        s['dedt'] = _num(fa.iloc[0].get('dedt_profit')) if len(fa) else None
    else:
        s['roe3'], s['gm'], s['nm'], s['dedt'] = None, None, None, None
    # 当期资产负债表 + 上一年年报 (应收同比用)
    if bal is not None and len(bal):
        b = bal[(bal['ann_date'].astype(str) <= rd) & (bal['end_date'].astype(str) <= re_)]
        b = b.sort_values('end_date', ascending=False).drop_duplicates('end_date')
        if len(b):
            s['bal'] = b.iloc[0]
            if len(b) >= 2:
                s['bal_prev'] = b.iloc[1]
    if 'bal' not in s:
        return None
    b = s['bal']
    s['total_share'] = _num(b.get('total_share'))
    s['fix'] = _num(b.get('fix_assets')) or 0
    s['cip'] = _num(b.get('cip')) or 0
    s['intan'] = _num(b.get('intan_assets')) or 0
    s['ta'] = _num(b.get('total_assets'))
    s['goodwill'] = _num(b.get('goodwill')) or 0
    s['ar'] = _num(b.get('accounts_receiv')) or 0
    debt = (_num(b.get('st_borr')) or 0) + (_num(b.get('lt_borr')) or 0) + (_num(b.get('bond_payable')) or 0) + (_num(b.get('non_cur_liab_due_1y')) or 0)
    cash = (_num(b.get('money_cap')) or 0) + (_num(b.get('trad_asset')) or 0)
    s['net_cash'] = cash - debt
    s['total_cash'] = cash
    # 估值 (run_date 前最近一天)
    db = mt['db']
    if len(db):
        d = db[db['ts_code'] == code]
        d = d[d['trade_date'].astype(str) <= rd].sort_values('trade_date')
        if len(d):
            s['db'] = d.iloc[-1]
            s['close'] = _num(d.iloc[-1]['close'])
            s['total_mv'] = _num(d.iloc[-1]['total_mv'])      # 万元
            s['pe_ttm'] = _num(d.iloc[-1]['pe_ttm'])
            s['ps_ttm'] = _num(d.iloc[-1]['ps_ttm'])
            # PS/PB 时点前130自然日历史分位
            h = d[d['trade_date'].astype(str) >= str(int(rd) - 130000)]
            ps_h = h['ps_ttm'].dropna()
            if len(ps_h) > 10 and s.get('ps_ttm'):
                s['ps_pct'] = float((ps_h <= s['ps_ttm']).mean())
            pb_h = h['pb'].dropna() if 'pb' in h else pd.Series(dtype=float)
            if len(pb_h) > 10 and _num(d.iloc[-1].get('pb')):
                s['pb_pct'] = float((pb_h <= _num(d.iloc[-1]['pb'])).mean())
    if not s.get('total_mv') or not s.get('np3') or s['np3'][0] in (None, 0) and not (s['np3'][0] is not None):
        pass
    return s


def gate_mine_sweep(s, mt, uni, r2_med=None, P=None):
    """排雷门控 R1/R2/R3 -> (passed, detail)"""
    code = s['code']
    # R1 更名史: 曾ST(不限年限) + 核心词断裂 (相邻简称不共享>=2字)
    nm = mt['nm']
    hits = []
    if len(nm):
        nmc = nm[nm['ts_code'] == code].sort_values('start_date')
        names = [str(x) for x in nmc['name'].tolist()]
        ever_st = any('ST' in n.upper() for n in names)
        if ever_st and len(names) >= 3:
            # 核心词断裂: 最后一次更名前后核心词不共享
            cur = s['name'].replace('ST', '').replace('*', '').replace('退', '')
            prev = names[-1].replace('ST', '').replace('*', '').replace('退', '')
            if len(set(cur) & set(prev)) < 2:
                hits.append(f'R1:曾ST+换马甲({names[-3:]})')
    # R2 财务洗澡: 近5年任一年净利<0 且该年同L2中位数>0
    l2 = s.get('l2', '')
    for i, npv in enumerate(s['np5']):
        if npv is not None and npv < 0:
            yr = int(str(s['annual'].iloc[i]['end_date'])[:4])
            med = r2_med.get((l2, yr))
            if med is not None and med > 0:
                hits.append(f'R2:{yr}亏{npv/1e8:.1f}亿但同L2中位{med/1e8:.1f}亿>0')
                break
    # R3 超额分红套现: 近5年支付率>100%>=2次 + 近12月大股东减持
    div = mt['div']
    np_by_year = {int(str(r['end_date'])[:4]): _num(r['n_income_attr_p']) for _, r in s['annual'].head(5).iterrows()}
    over = 0
    if s.get('total_share') and len(div):
        dc = div[div['ts_code'] == code]
        for yr, npv in np_by_year.items():
            if npv and npv > 0:
                dd = dc[dc['end_date'].astype(str).str.startswith(str(yr))]
                if len(dd) and dd['cash_div'].notna().any():
                    mx = dd['cash_div'].max()
                    if mx and mx * s['total_share'] / npv > 1.0:
                        over += 1
    if over >= 2:
        tr = mt['trades']
        trc = tr[(tr['ts_code'] == code) & (tr['in_de'] == 'DE') & (tr['ann_date'].astype(str) >= str(int(P['run_date']) - 10000))] if len(tr) else pd.DataFrame()
        big_de = 0
        if len(trc) and 'change_ratio' in trc.columns:
            big_de = (trc['change_ratio'].fillna(0) > 0.1).sum()  # 单笔减持>0.1%股本
        if big_de > 0:
            hits.append(f'R3:支付率>100%{over}次+大股东减持{big_de}笔')
    return (len(hits) == 0), hits


def gate_price(s, mt, P):
    """价格门控 -> (passed, lane, detail)"""
    np0 = s['np3'][0] if s['np3'] else None
    if np0 is None or not s.get('total_mv'):
        return None, 'N/A', '数据不足'
    stable = all(v is not None and v > 0 for v in s['np3'][:3])
    tmv = s['total_mv'] * 1e4          # 万元->元
    mv_adj = tmv - (s['net_cash'] if not s['is_fin'] else 0)
    npv = np0
    imp_ok = s.get('imp') is not None and s['np3'][1] and s['imp'] >= s['np3'][1] * 0.5
    bath = imp_ok or s.get('restated', False)
    lane = None
    if s['is_cyc']:
        # A 方案: 周期位置三态 + 绝对 PE 上限; 位置未标注的行业按"上行中段"PE 兜底
        key = None
        for k in CYCLE_POSITION:
            if k in s['industry'] or k in (s.get('l1') or ''):
                key = k
                break
        pos = CYCLE_POSITION.get(key, ('上行中段', '位置未标注->PE兜底'))[0]
        pe = s.get('pe_ttm')
        if pos == '顶部':
            return False, '周期顶部', '顶部一律淘汰(防低PE陷阱)'
        if pos == '底部' and (pe is None or pe <= 0 or pe > 40):
            return True, '周期底部', '底部亏损豁免PE(排雷兜底)'
        if pe is None or pe <= 0:
            return None, '周期N/A', 'PE缺失'
        if pe > 40:
            return False, '周期PE', f'上行中段PE={pe:.0f}>40'
        return True, '周期', f'PE={pe:.0f}<=40'
    if s['is_fin']:
        # 金融股: PB 上限 3 + 股息率兜底 (无 PR 口径)
        db = s.get('db')
        pb = _num(db.get('pb')) if db is not None else None
        if pb is not None and pb > 3:
            return False, '金融PB', f'PB={pb:.2f}>3'
        return True, '金融', f'PB={pb and round(pb,2)}'
    if stable:
        pr = mv_adj / npv if npv > 0 else None
        if pr is None:
            return None, 'N/A', 'PR不可算'
        s['PR'] = pr
        if bath:
            return True, '洗澡例外', f'PR={pr:.1f}但减值/重述出清 -> 走PS'
        if pr > PR_GATE:
            return False, 'PR', f'PR={pr:.2f}>{PR_GATE:g}'
        return True, 'PR', f'PR={pr:.2f}'
    # 非稳定盈利: PS 自身历史分位
    psp = s.get('ps_pct')
    if psp is None:
        return True, 'PS-N/A', 'PS历史不足, 通过并注明'
    if psp > 0.70:
        return False, 'PS', f'PS分位{psp:.0%}>70%'
    return True, 'PS', f'PS分位{psp:.0%}'


def heavy_ratio(s):
    """重资产比例 (行业感知): 无形不计清单行业 -> 固资+在建; 否则固资+在建+无形"""
    ta = s.get('ta')
    if not ta:
        return None
    base = s['fix'] + s['cip']
    if not s['intan_exclude']:
        base += s['intan']
    return base / ta


def score_ten(s, mt, P, s7_rank, roe_top10_pct, cashflow_fn):
    """十项打分 -> dict {item: (score, evidence)}"""
    sc = {}
    annual = s['annual']
    # s1 营收净利 (扣非 P1-7)
    if s['is_cyc']:
        key = next((k for k in CYCLE_POSITION if k in s['industry'] or k in (s.get('l1') or '')), None)
        if key:
            pos = CYCLE_POSITION[key][0]
            sc['s1'] = (1 if pos == '底部' else -1 if pos == '顶部' else 0, f'周期{s["industry"]}位置{pos}')
        else:
            sc['s1'] = (0, '周期位置未标注')
    else:
        rev3, np3 = s['rev3'][:3], s['np3'][:3]
        grew = all(rev3[i] and rev3[i + 1] and rev3[i] > rev3[i + 1] and np3[i] and np3[i + 1] and np3[i] > np3[i + 1] for i in range(2))
        crash = any(np3[i] and np3[i + 1] and np3[i + 1] > 0 and np3[i] / np3[i + 1] < 0.5 for i in range(2))
        if s.get('dedt') is not None and np3[0] and abs(s['dedt']) > abs(np3[0]) * 0.5:
            sc['s1'] = (0, f'扣非{ s["dedt"]/1e8:.1f}亿占净利>50% 疑利润操纵')
        elif grew:
            sc['s1'] = (1, '近3年营收净利逐年增长')
        elif crash:
            sc['s1'] = (-1, '近3年净利腰斩')
        else:
            sc['s1'] = (0, '增长不连续')
    # s2 ROE 趋势
    roe3 = s.get('roe3')
    if s['is_cyc']:
        sc['s2'] = (0, '周期股不判')
    elif not roe3 or any(v is None for v in roe3):
        sc['s2'] = (None, 'ROE缺失')
    elif all(v >= 15 for v in roe3):
        sc['s2'] = (1, f'ROE3年均>=15 {roe3}')
    elif len(roe3) >= 3 and roe3[0] and roe3[2] and roe3[0] < roe3[2] * 0.7:
        sc['s2'] = (-1, f'ROE趋势下滑 {roe3}')
    else:
        sc['s2'] = (0, f'ROE不极端 {roe3}')
    # s3 资产负债
    if s['is_fin']:
        sc['s3'] = (None, '金融股三指标无Tushare接口 N/A')
    else:
        rev0 = s['rev3'][0] or 0
        a = (s['ar'] > 0 and rev0 > 0 and s['ar'] / rev0 <= 0.2)
        tc = s['total_cash']
        b = tc > 0 and s['net_cash'] / tc > 0.8
        c = s['goodwill'] == 0 or (s['ta'] and s['goodwill'] / s['ta'] <= 0.05)
        hr = heavy_ratio(s)
        nr = cashflow_fn(s, 'ncr') if cashflow_fn else None
        h_ok = hr is not None and hr > 0.4 and nr is not None and nr >= 1
        if h_ok:
            b = True
        if a and b and c:
            sc['s3'] = (1, f'三过(hr={hr and round(hr,2)},ncr={nr and round(nr,2)})')
        elif sum([not a, not b, not c]) >= 2:
            sc['s3'] = (-1, f'a={a},b={b},c={c}')
        else:
            sc['s3'] = (0, '部分过')
    # s4 现金流
    if s['is_fin']:
        sc['s4'] = (None, '金融股 N/A')
    else:
        cf = cashflow_fn(s, 'all') if cashflow_fn else None
        if cf is None:
            sc['s4'] = (None, '现金流数据缺失')
        else:
            a, bb, c_, d_, ncr, ar_yoy = cf
            hr = heavy_ratio(s)
            grow = s['rev3'][0] and s['rev3'][2] and s['rev3'][0] / s['rev3'][2] > 1.52
            c_ok = c_ or (grow and ncr and ncr >= 1) or (hr and hr > 0.4 and ncr and ncr >= 1)
            if a and bb and c_ok and d_:
                sc['s4'] = (1, '四过')
            else:
                if (not bb) and ar_yoy is not None and ar_yoy > 0.5:
                    sc['s4'] = (-1, f'净现比<1且应收同比{ar_yoy:.0%}>50% 联动-1')
                elif sum([not a, not bb, not c_ok, not d_]) >= 2:
                    sc['s4'] = (-1, f'a={a},b={bb},c={c_ok},d={d_}')
                else:
                    sc['s4'] = (0, '部分过')
    # s5 分红
    div = mt['div']
    dc = div[div['ts_code'] == s['code']] if len(div) else pd.DataFrame()
    if s['is_cyc'] and s['np3'][0] and s['np3'][0] < 0:
        sc['s5'] = (0, '周期亏损年豁免')
    elif not len(dc):
        sc['s5'] = (None, '分红数据缺失')
    else:
        pays = {}
        for _, r in s['annual'].head(3).iterrows():
            yr = str(r['end_date'])[:4]
            npv = _num(r['n_income_attr_p'])
            dd = dc[dc['end_date'].astype(str).str.startswith(yr)]
            if npv and npv > 0 and len(dd) and dd['cash_div'].notna().any():
                pays[yr] = dd['cash_div'].max() * s['total_share'] / npv if s.get('total_share') else None
        if len(pays) < 3 or not [v for v in pays.values() if v]:
            sc['s5'] = (0, f'分红年数{len([v for v in pays.values() if v])}/3')
        else:
            vs = [v for v in pays.values() if v]
            over = sum(1 for v in vs if v > 1.0)
            drop = len(vs) >= 2 and vs[-2] > 0 and vs[-1] < vs[-2] * 0.25
            if over:
                sc['s5'] = (0, f'支付率>100%{over}次')
            elif drop:
                sc['s5'] = (-1, f'分红骤降 {round(vs[-2],2)}->{round(vs[-1],2)}')
            else:
                sc['s5'] = (1, f'近3年正常分红 {[(k, round(v,2)) for k, v in pays.items()]}')
    # s6 行业 (周期股联动 s1)
    if s['is_cyc']:
        sc['s6'] = (0, '周期股与s1联动不判增速')
    else:
        ikey = next((k for k in INDUSTRY_META if k in s['industry'] or k in (s.get('l1') or '')), None)
        if ikey is None:
            sc['s6'] = (None, '行业空间数据不可得 N/A')
        else:
            meta = INDUSTRY_META[ikey]
            if meta.get('market_size_yi', 0) < 1000:
                sc['s6'] = (0, f'{ikey}规模<千亿')
            else:
                g = meta.get('g_by_year', {})
                yr = int(P['report_end'][:4])
                ok = [yv for yy, yv in g.items() if yv is not None and (yv >= GDP_GROWTH.get(yy, 5) + 2 or yv >= 1.5 * GDP_GROWTH.get(yy, 5))]
                sc['s6'] = (1 if len(ok) >= max(1, len(g) // 2) else 0, f'{ikey} 增速达标{len(ok)}/{len(g)}')
    # s7 龙头 (全A截面 TOP3%)
    rank = s7_rank.get(s['code'])
    if rank is None:
        sc['s7'] = (None, '截面数据缺失')
    else:
        n_members, r_gm, r_nm, r_roe = rank
        if s['is_fin']:
            top = sum(1 for r in (r_nm, r_roe) if r and r <= max(3, math.ceil(n_members * 0.03)) or r and n_members < 10 and r <= 3)
            sc['s7'] = (1 if top >= 1 else 0, f'金融两率TOP{top} n={n_members}')
        else:
            line = max(3, math.ceil(n_members * 0.03)) if n_members >= 10 else 3
            top = sum(1 for r in (r_gm, r_nm, r_roe) if r is not None and r <= line)
            sc['s7'] = (1 if top >= 2 else 0, f'三率中{top}项<=线{line} n={n_members}')
    # s8 股权变动
    rd = P['run_date']
    tr = mt['trades']
    rep = mt['rep']
    trc = tr[(tr['ts_code'] == s['code']) & (tr['ann_date'].astype(str) >= str(int(rd) - 10000)) & (tr['ann_date'].astype(str) <= rd)] if len(tr) else pd.DataFrame()
    rpc = rep[(rep['ts_code'] == s['code']) & (rep['ann_date'].astype(str) >= str(int(rd) - 10000)) & (rep['ann_date'].astype(str) <= rd)] if len(rep) else pd.DataFrame()
    n_in = int((trc['in_de'] == 'IN').sum()) if len(trc) else 0
    n_de = int((trc['in_de'] == 'DE').sum()) if len(trc) else 0
    buyback = 0
    avg_px = None
    if len(rpc):
        done = rpc[rpc['proc'].astype(str).str.contains('实施|完成', na=False)]
        if len(done):
            v, a_ = _num(done.iloc[0].get('vol')), _num(done.iloc[0].get('amount'))
            if v and a_:
                avg_px = a_ / v
        amt = max([_num(x) for x in rpc['amount'].fillna(0)], default=0)
        if s.get('total_mv') and amt:
            buyback = amt / (s['total_mv'] * 1e4)   # amount 元 / (市值万元*1e4)
    pos_pts = 0
    pos_ev = []
    if buyback >= 0.01:
        pos_pts += 1
        pos_ev.append(f'回购{buyback:.1%}市值(注销档近似)')
    elif buyback > 0 and avg_px and s.get('close') and s['close'] < avg_px * 0.9:
        pos_pts += 1
        pos_ev.append(f'回购均价{avg_px:.1f} 现价折10%')
    if n_in > 0:
        pos_pts += 1
        pos_ev.append(f'增持{n_in}笔')
    neg_ev = f'减持{n_de}笔' if n_de > 0 else None
    if n_de > 0:
        ev = (f'{",".join(pos_ev)} | ' if pos_pts else '') + f'{neg_ev} -> -1 (增/回购{pos_pts}分不对冲不抵减持)'
        sc['s8'] = (-1, ev)
    elif pos_pts > 0:
        sc['s8'] = (1, ','.join(pos_ev))
    else:
        sc['s8'] = (0, '近12月无增减持回购')
    # s9 人均持股市值 (散户化陷阱)
    t10 = mt['top10']
    hd = mt['holder']
    t10c = t10[(t10['ts_code'] == s['code']) & (t10['end_date'].astype(str) <= P['report_end'])] if len(t10) else pd.DataFrame()
    hdc = hd[(hd['ts_code'] == s['code']) & (hd['end_date'].astype(str) <= P['report_end'])] if len(hd) else pd.DataFrame()
    if len(t10c) and len(hdc) and s.get('total_share') and s.get('close'):
        t10c = t10c.sort_values('end_date')
        last_ed = t10c['end_date'].max()
        amt = t10c[t10c['end_date'] == last_ed]['hold_amount'].fillna(0).sum()
        hnn = hdc.sort_values('end_date')
        hn = _num(hnn.iloc[-1]['holder_num'])
        tradable = max(s['total_share'] - amt, 0)
        if hn and tradable > 0:
            pc = tradable * s['close'] / hn / 1e4
            chg60 = None
            db = s.get('db')
            if db is not None:
                dd = mt['db'][(mt['db']['ts_code'] == s['code']) & (mt['db']['trade_date'].astype(str) <= P['run_date'])].sort_values('trade_date')
                if len(dd) >= 61:
                    chg60 = dd.iloc[-1]['close'] / dd.iloc[-61]['close'] - 1
            trap = pc < 10 and chg60 is not None and chg60 > 0.5
            if pc >= 30:
                sc['s9'] = (1, f'人均{pc:.1f}万')
            elif trap:
                sc['s9'] = (-1, f'人均{pc:.1f}万<10万且60日涨{chg60:.0%} 散户化陷阱(负分计2)')
            elif pc < 10:
                sc['s9'] = (-1, f'人均{pc:.1f}万<10万')
            else:
                sc['s9'] = (0, f'人均{pc:.1f}万')
        else:
            sc['s9'] = (None, '股东数/股本异常')
    else:
        sc['s9'] = (None, 'top10/股东数据缺失')
    # s10 估值
    if s['is_cyc']:
        pe = s.get('pe_ttm')
        peer = s10_peer_pe.get(s['code'])
        if peer is None or pe is None or pe <= 0:
            sc['s10'] = (None, '周期同行PE样本不足/亏损 N/A')
        else:
            pct = peer
            sc['s10'] = (1 if pct <= 0.3 else -1 if pct >= 0.7 else 0, f'同行PE分位{pct:.0%}')
    elif s['is_fin']:
        pct = s10_div_pct.get(s['code'])
        if pct is None:
            sc['s10'] = (None, '股息率分位缺失')
        else:
            sc['s10'] = (1 if pct <= 0.3 else -1 if pct >= 0.7 else 0, f'金融股息率分位{pct:.0%}')
    else:
        stable = all(v is not None and v > 0 for v in s['np3'][:3])
        my = s.get('PR')
        if not stable:
            my = s.get('ps_ttm')
        dist = s10_dist_pr if (stable and s10_dist_pr) else s10_dist_ps
        if my is None or len(dist) < 100:
            sc['s10'] = (None, '全A截面不足')
        else:
            arr = np.array(dist)
            pct = float((arr < my).mean() + 0.5 * (arr == my).mean())
            roe0 = s['roe3'][0] if s.get('roe3') else None
            premium = roe0 is not None and roe_top10_pct is not None and roe0 >= roe_top10_pct
            hi = 0.8 if premium else 0.7
            lo = 0.4 if premium else 0.3
            sc['s10'] = (1 if pct <= lo else -1 if pct >= hi else 0, f'全A{"PR" if stable else "PS"}分位{pct:.0%}{" 质地溢价" if premium else ""}')
    return sc


def build_s7_cross(P, tab_all, uni):
    """全A 截面: 申万一级内 毛利率/净利率/ROE 排名 (当期年报, ann_date<=run_date)"""
    rows = []
    for code, tab in tab_all.items():
        fin = tab.get('fina')
        if fin is None or len(fin) == 0:
            continue
        f = fin[(fin['ann_date'].astype(str) <= P['run_date']) & (fin['end_date'].astype(str) <= P['report_end'])]
        fa = f[f['end_date'].astype(str).str.endswith('1231')].sort_values('end_date', ascending=False).drop_duplicates('end_date')
        if not len(fa):
            continue
        r = fa.iloc[0]
        rows.append(dict(code=code, l1=uni[code]['l1'], gm=_num(r.get('grossprofit_margin')),
                         nm=_num(r.get('netprofit_margin')), roe=_num(r.get('roe'))))
    cr = pd.DataFrame(rows)
    cr = cr[cr['l1'] != '']
    rank = {}
    for l1, grp in cr.groupby('l1'):
        n = len(grp)
        line = max(3, math.ceil(n * 0.03)) if n >= 10 else 3
        for col, key in (('gm', 'r_gm'), ('nm', 'r_nm'), ('roe', 'r_roe')):
            g2 = grp[grp[col].notna()].sort_values(col, ascending=False)
            for i, (_, rr) in enumerate(g2.iterrows(), 1):
                rank.setdefault(rr['code'], [n, None, None, None])
                rank[rr['code']][{'r_gm': 1, 'r_nm': 2, 'r_roe': 3}[key]] = i if i <= line * 5 else None
    return rank, cr


def build_r2_median(P, tab_all, uni):
    """R2: 申万二级 x 年度 净利润中位数 (剔除次新/ST)"""
    rows = []
    for code, tab in tab_all.items():
        inc = tab.get('income')
        if inc is None or len(inc) == 0 or uni[code]['l2'] == '':
            continue
        if uni[code]['list_date'] > str(int(P['report_end']) - 10000):
            continue
        if 'ST' in uni[code]['name'].upper():
            continue
        f = inc[(inc['ann_date'].astype(str) <= P['run_date']) & (inc['end_date'].astype(str) <= P['report_end'])]
        fa = f[f['end_date'].astype(str).str.endswith('1231')]
        for _, r in fa.drop_duplicates('end_date').iterrows():
            rows.append((uni[code]['l2'], int(str(r['end_date'])[:4]), _num(r['n_income_attr_p'])))
    df = pd.DataFrame(rows, columns=['l2', 'yr', 'np'])
    df = df.dropna()
    med = df.groupby(['l2', 'yr'])['np'].median().to_dict()
    return med


def build_s10_cross(P, tab_all, uni, db_day):
    """s10 全A PR/PS 分位截面 + 周期同行PE分位 + 金融股息率(近似用 pb 倒数替代注释)"""
    prs, pss, pes_by_l1 = [], [], {}
    for code, tab in tab_all.items():
        inc = tab.get('income')
        if inc is None or len(inc) == 0:
            continue
        f = inc[(inc['ann_date'].astype(str) <= P['run_date']) & (inc['end_date'].astype(str) <= P['report_end'])]
        fa = f[f['end_date'].astype(str).str.endswith('1231')].sort_values('end_date', ascending=False).drop_duplicates('end_date')
        if not len(fa):
            continue
        npv = _num(fa.iloc[0]['n_income_attr_p'])
        row = db_day.get(code)
        if row is None:
            continue
        tmv = _num(row.get('total_mv'))
        if not tmv or not npv or npv <= 0:
            continue
        b = tab.get('bal')
        net_cash = 0
        if b is not None and len(b):
            bb = b[(b['ann_date'].astype(str) <= P['run_date'])].sort_values('end_date', ascending=False).drop_duplicates('end_date')
            if len(bb):
                r0 = bb.iloc[0]
                debt = (_num(r0.get('st_borr')) or 0) + (_num(r0.get('lt_borr')) or 0) + (_num(r0.get('bond_payable')) or 0) + (_num(r0.get('non_cur_liab_due_1y')) or 0)
                cash = (_num(r0.get('money_cap')) or 0) + (_num(r0.get('trad_asset')) or 0)
                if not uni[code]['is_fin']:
                    net_cash = cash - debt
        prs.append((tmv * 1e4 - net_cash) / npv)
        if _num(row.get('ps_ttm')):
            pss.append(_num(row.get('ps_ttm')))
        if uni[code]['is_cyc'] and _num(row.get('pe_ttm')) and _num(row.get('pe_ttm')) > 0:
            pes_by_l1.setdefault(uni[code]['l1'], []).append((_num(row.get('pe_ttm')), code))
    return prs, pss, pes_by_l1


def cashflow_lazy_loader(codes, mt_dates):
    """门控通过者增量拉取现金流 -> {code: {end_date: (ocf, capex, sales_recv)}}"""
    sys.path.insert(0, 'e:/quant')
    from credentials import get_tushare_api
    import time as _t
    pro = get_tushare_api()
    out = {}
    for i, c in enumerate(codes):
        try:
            df = pro.cashflow(ts_code=c, start_date='20200101', end_date=mt_dates,
                              fields='ts_code,end_date,ann_date,n_cashflow_act,c_pay_acq_const_fiolta,c_recp_sellg_or_renders')
            out[c] = df
        except Exception:
            pass
        _t.sleep(0.13)
        if (i + 1) % 200 == 0:
            print(f'  [cashflow] {i+1}/{len(codes)}', flush=True)
    return out


s10_dist_pr, s10_dist_ps, s10_peer_pe, s10_div_pct = [], [], {}, {}


def run_period(P, tab_all, mt, uni):
    global s10_dist_pr, s10_dist_ps, s10_peer_pe, s10_div_pct
    print(f"\n{'='*70}\n[{P['tag']}] report={P['report_end']} run_date={P['run_date']}\n{'='*70}", flush=True)
    # 全市场截面预计算
    r2_med = build_r2_median(P, tab_all, uni)
    s7_rank, cr = build_s7_cross(P, tab_all, uni)
    db = mt['db']
    db_day = {}
    if len(db):
        db = db.drop_duplicates(subset=['ts_code', 'trade_date'])
        mt['db'] = db
        dd = db[db['trade_date'].astype(str) <= P['run_date']]
        last_d = dd['trade_date'].max()
        day_df = dd[dd['trade_date'] == last_d].drop_duplicates(subset=['ts_code'])
        db_day = day_df.set_index('ts_code').to_dict('index')
        print(f'[cross] est date={last_d}, R2 median {len(r2_med)} groups, s7 {len(s7_rank)} stocks', flush=True)
    # 快照
    snaps = {}
    for code, tab in tab_all.items():
        s = stock_snapshot(code, tab, mt, uni, P, r2_med, None)
        if s is not None:
            snaps[code] = s
    print(f'[snapshot] {len(snaps)} stocks with 3yr annuals + bal', flush=True)
    # 门控
    survivors = {}
    stats = dict(total=len(snaps), r1=0, r2=0, r3=0, gp=0, stable=0, ps=0, cyc=0, fin=0, elim_pr=0, elim_ps=0, elim_pe=0, elim_top=0, elim_pb=0, bath=0)
    for code, s in snaps.items():
        if 'ST' in s['name'].upper() or s['list_date'] > str(int(P['report_end']) - 400):
            continue
        ok, hits = gate_mine_sweep(s, mt, uni, r2_med, P)
        if not ok:
            for h in hits:
                stats['r1' if h.startswith('R1') else 'r2' if h.startswith('R2') else 'r3'] += 1
            continue
        gp, lane, det = gate_price(s, mt, P)
        if gp is None:
            continue
        if not gp:
            stats['elim_pr' if lane.startswith('PR') else 'elim_ps' if lane.startswith('PS') else 'elim_pe' if lane.startswith('周期PE') else 'elim_top' if lane.startswith('周期顶') else 'elim_pb'] += 1
            continue
        if lane == '洗澡例外':
            stats['bath'] += 1
        stats['gp'] += 1
        survivors[code] = s
    print(f'[gate] survivors={len(survivors)} | R1={stats["r1"]} R2={stats["r2"]} R3={stats["r3"]} | '
          f'elim: PR={stats["elim_pr"]} PS={stats["elim_ps"]} 周期PE={stats["elim_pe"]} 顶部={stats["elim_top"]} 金融PB={stats["elim_pb"]}', flush=True)
    # s10 截面 (全A)
    s10_dist_pr, s10_dist_ps, pes_by_l1 = build_s10_cross(P, tab_all, uni, db_day)
    s10_peer_pe = {}
    for l1, vals in pes_by_l1.items():
        arr = sorted(v for v, _ in vals)
        for v, c in vals:
            s10_peer_pe[c] = (arr.index(v) + 0.5) / len(arr)
    # 金融股息率: 无股息接口批量 -> 用近12月分红/市值近似 (full_div)
    div, dbd = mt['div'], db_day
    div_pct = {}
    fin_codes = [c for c, s in survivors.items() if s['is_fin']]
    for c in fin_codes:
        dv = div[(div['ts_code'] == c) & (div['ann_date'].astype(str) >= str(int(P['run_date']) - 10000))] if len(div) else pd.DataFrame()
        tmv = dbd.get(c, {}).get('total_mv')
        if len(dv) and tmv:
            total_d = dv['cash_div'].fillna(0).max()
            # 分红对应股本 vs 总市值: 近似用现金分红总额(万)/市值(万) 需乘股本比, 此处用 per-share 口径
            s = survivors[c]
            if s.get('total_share'):
                dy = total_d * s['total_share'] / (tmv * 1e4)
                div_pct[c] = dy
    arr = sorted(div_pct.values())
    s10_div_pct = {c: (arr.index(v) + 0.5) / len(arr) for c, v in div_pct.items()} if arr else {}
    # ROE 前10% 分位线
    roes = [s['roe3'][0] for s in snaps.values() if s.get('roe3') and s['roe3'][0] is not None]
    roe_top10_pct = float(np.percentile(roes, 90)) if len(roes) > 50 else None
    # 现金流增量拉取 (门控通过者)
    cache_fn = f'{OUT}/cashflow_{P["tag"]}_full.pkl'
    if os.path.exists(cache_fn):
        cf_map = pd.read_pickle(cache_fn)
    else:
        print(f'[cashflow] fetching {len(survivors)} stocks ...', flush=True)
        cf_map = cashflow_lazy_loader(list(survivors.keys()), P['run_date'])
        pd.to_pickle(cf_map, cache_fn)

    def cf_fn(s, mode):
        df = cf_map.get(s['code'])
        if df is None or not len(df):
            return None if mode == 'ncr' else None
        d = df[(df['ann_date'].astype(str) <= P['run_date']) & (df['end_date'].astype(str) <= P['report_end'])]
        fa = d[d['end_date'].astype(str).str.endswith('1231')].sort_values('end_date', ascending=False).drop_duplicates('end_date')
        if not len(fa):
            return None
        r0 = fa.iloc[0]
        ocf, npv = _num(r0.get('n_cashflow_act')), s['np3'][0]
        ncr = ocf / npv if ocf is not None and npv and npv > 0 else None
        if mode == 'ncr':
            return ncr
        capex = _num(r0.get('c_pay_acq_const_fiolta')) or 0
        recv = _num(r0.get('c_recp_sellg_or_renders'))
        rev0 = s['rev3'][0]
        a = recv is not None and rev0 and recv / rev0 > 1
        bb = ncr is not None and ncr > 1
        c_ = ocf and capex / ocf <= 0.10 if ocf else False
        d_ = False
        if len(fa) >= 3:
            fcf_ok = True
            for i in range(3):
                o = _num(fa.iloc[i].get('n_cashflow_act'))
                cx = _num(fa.iloc[i].get('c_pay_acq_const_fiolta')) or 0
                if o is None or o - cx <= 0:
                    fcf_ok = False
                    break
            d_ = fcf_ok
        ar_yoy = None
        bprev = s.get('bal_prev')
        if bprev is not None:
            ar_prev = _num(bprev.get('accounts_receiv'))
            if ar_prev and ar_prev > 0 and s['ar'] > 0:
                ar_yoy = s['ar'] / ar_prev - 1
        return (bool(a), bool(bb), bool(c_), bool(d_), ncr, ar_yoy)
    # 打分+裁决
    results = []
    for code, s in survivors.items():
        sc = score_ten(s, mt, P, s7_rank, roe_top10_pct, cf_fn)
        pos = sum(1 for v in sc.values() if v[0] == 1)
        neg = sum(1 for v in sc.values() if v[0] == -1)
        neg_w = sum(2 if '散户化陷阱' in (v[1] or '') else 1 for v in sc.values() if v[0] == -1)
        na = sum(1 for v in sc.values() if v[0] is None)
        total = sum(v[0] for v in sc.values() if v[0] is not None)
        if neg_w >= 5:
            dec = '剔除'
        elif pos >= 7:
            dec = '直通'
        else:
            dec = 'TOP15候选'
        results.append(dict(code=code, name=s['name'], industry=s['industry'], lane=s.get('PR'), total=total, pos=pos, neg=neg, na=na, decision=dec, scores={k: v[0] for k, v in sc.items()}, ev={k: v[1] for k, v in sc.items()}))
    rdf = pd.DataFrame(results)
    pool = rdf[rdf['decision'] != '剔除'].sort_values('total', ascending=False)
    top15 = pool[pool['decision'] == 'TOP15候选'].head(15)
    final_pool = pd.concat([pool[pool['decision'] == '直通'], top15])
    print(f'[score] scored={len(rdf)} direct={(rdf["decision"]=="直通").sum()} pool={len(final_pool)}', flush=True)
    rdf.to_csv(f'{OUT}/scores_{P["tag"]}.csv', index=False)
    final_pool.to_csv(f'{OUT}/pool_{P["tag"]}.csv', index=False)
    return final_pool


def backtest(pools_by_tag, mt):
    """分 3 段等权回测: 段内逐日收益=池子各股 close 日收益等权均值, nav 累乘"""
    db = mt['db']
    bench = mt['bench']
    nav = 100.0
    nav_rows = []
    seg_stats = []
    for P in PERIODS:
        pool = pools_by_tag[P['tag']]
        codes = [c for c in pool['code'].tolist() if c]
        seg = db[(db['trade_date'].astype(str) >= P['bt_start']) & (db['trade_date'].astype(str) <= P['bt_end']) & (db['ts_code'].isin(codes))]
        dates = sorted(seg['trade_date'].unique())
        if not dates or not codes:
            seg_stats.append(dict(tag=P['tag'], ret=0.0, n=len(codes)))
            continue
        px = seg.pivot_table(index='trade_date', columns='ts_code', values='close', aggfunc='last').ffill()
        # 期初价: bt_start 前最近可得 close
        pre = db[(db['trade_date'].astype(str) < P['bt_start']) & (db['ts_code'].isin(codes))].sort_values('trade_date')
        base = pre.groupby('ts_code')['close'].last().to_dict()
        # 缺期初价的用段内首个价, 并从首个有效价日起计入
        codes_ok = [c for c in codes if base.get(c) or (c in px.columns and px[c].notna().any())]
        seg_nav = [1.0]
        for i in range(1, len(dates)):
            rets = []
            for c in codes_ok:
                col = px[c] if c in px.columns else None
                if col is None:
                    continue
                p_prev, p_cur = col.get(dates[i - 1]), col.get(dates[i])
                if p_cur and p_cur == p_cur and p_prev and p_prev == p_prev and p_prev > 0:
                    rets.append(p_cur / p_prev - 1)
            r = float(np.mean(rets)) if rets else 0.0
            seg_nav.append(seg_nav[-1] * (1 + r))
        seg_ret = seg_nav[-1] - 1
        seg_stats.append(dict(tag=P['tag'], ret=seg_ret, n=len(codes_ok)))
        for d, v in zip(dates, seg_nav):
            nav_rows.append((d, nav * v, P['tag']))
        nav *= (1 + seg_ret)
    nav_df = pd.DataFrame(nav_rows, columns=['date', 'nav', 'seg'])
    b = bench.copy()
    b = b[(b['trade_date'].astype(str) >= PERIODS[0]['bt_start']) & (b['trade_date'].astype(str) <= PERIODS[-1]['bt_end'])].sort_values('trade_date')
    if len(b):
        b['bench_nav'] = b['close'] / b['close'].iloc[0] * 100
    return nav_df, b, seg_stats


def main():
    print('[load] universe + tables ...', flush=True)
    uni = load_universe()
    tab_all = load_tables()
    mt = load_market_tables()
    print(f'[load] universe={len(uni)}, tables={len(tab_all)}, db rows={len(mt["db"])}', flush=True)
    pools = {}
    for P in PERIODS:
        pools[P['tag']] = run_period(P, tab_all, mt, uni)
    nav_df, bench_df, seg_stats = backtest(pools, mt)
    nav_df.to_csv(f'{OUT}/nav_curve.csv', index=False)
    pd.to_pickle(dict(pools={t: p.to_dict('records') for t, p in pools.items()}, seg_stats=seg_stats), f'{OUT}/backtest_state.pkl')
    print('[seg_stats]', json.dumps(seg_stats, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
