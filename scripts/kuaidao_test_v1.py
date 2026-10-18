"""
kuaidao_test_v1.py  (字段名修正版)
快刀斩股选股策略 Skill 端到端测试
"""
import sys, os, json, time, traceback
from pathlib import Path
sys.path.insert(0, 'e:/quant')
os.chdir('e:/quant')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from credentials import get_tushare_api

OUT_DIR = Path('e:/quant/data/kuaidao_test')
OUT_DIR.mkdir(parents=True, exist_ok=True)
RUN_TS = datetime.now().strftime('%Y%m%d_%H%M%S')
TODAY = datetime.now().strftime('%Y%m%d')

TEST_UNIVERSE = [
    ('600519.SH', '贵州茅台', '龙头稳定+高ROE', '白酒'),
    ('000858.SZ', '五粮液',   '龙头稳定',       '白酒'),
    ('600941.SH', '中国移动', '寡头+高现金',     '通信'),
    ('600036.SH', '招商银行', '银行',           '银行'),
    ('000001.SZ', '平安银行', '银行',           '银行'),
    ('600276.SH', '恒瑞医药', '创新药',         '医药'),
    ('000333.SZ', '美的集团', '家电龙头',       '家电'),
    ('002594.SZ', '比亚迪',   '高增长',         '汽车'),
    ('601012.SH', '隆基绿能', '周期光伏底部',   '光伏'),
    ('002714.SZ', '牧原股份', '周期养殖底部',   '养殖'),
    ('601225.SH', '陕西煤业', '周期煤炭',       '煤炭'),
    ('601088.SH', '中国神华', '周期煤炭',       '煤炭'),
    ('300750.SZ', '宁德时代', '高增长',         '电池'),
    ('300760.SZ', '迈瑞医疗', '稳定',           '医疗器械'),
    ('600887.SH', '伊利股份', '稳定消费',       '乳制品'),
    ('002475.SZ', '立讯精密', '高增长',         '电子'),
    ('300059.SZ', '东方财富', '券商',           '券商'),
    ('688981.SH', '中芯国际', '亏损科技',       '半导体'),
    ('600460.SH', '士兰微',   '半导体',         '半导体'),
    ('002415.SZ', '海康威视', '稳定',           '安防'),
    ('601318.SH', '中国平安', '综合金融',       '保险'),
    ('000725.SZ', '京东方A',  '面板周期',       '面板'),
    ('688041.SH', '复星医药', '医药',           '医药'),
    ('002230.SZ', '科大讯飞', '边缘',           'AI软件'),
    ('002821.SZ', '凯莱英',   'CXO利润下滑',    'CXO'),
]

# ============================================================
# 行业字典 (B1/B7 修复): 优先加载外置 data/kuaidao_test/industry_meta.json,
# 覆盖申万一级 31 行业 + 20 细分行业; 文件缺失时退回内置精简版。source 字段可维护。
# ============================================================
def _load_industry_meta():
    meta_path = OUT_DIR / 'industry_meta.json'
    if meta_path.exists():
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            meta.pop('_note', None)
            if meta:
                return meta
        except Exception as e:
            print(f'  [warn] industry_meta.json 加载失败({e}), 退回内置版')
    return INDUSTRY_META_FALLBACK

INDUSTRY_META_FALLBACK = {
    '白酒': {'market_size_yi': 10000, 'g_by_year': {2021: 18, 2022: 14, 2023: 15, 2024: 10, 2025: 6}, 'source': '内置回退', 'cycle': False},
    '银行': {'market_size_yi': 400000, 'g_by_year': {2021: 8, 2022: 5, 2023: 4, 2024: 3, 2025: 3}, 'source': '内置回退', 'cycle': False},
    '医药': {'market_size_yi': 9000, 'g_by_year': {2021: 18, 2022: 8, 2023: 5, 2024: 4, 2025: 5}, 'source': '内置回退', 'cycle': False},
    '家电': {'market_size_yi': 15000, 'g_by_year': {2021: 15, 2022: 3, 2023: 4, 2024: 5, 2025: 5}, 'source': '内置回退', 'cycle': False},
    '光伏': {'market_size_yi': 12000, 'g_by_year': {2021: 30, 2022: 60, 2023: 25, 2024: -10, 2025: -5}, 'source': '内置回退', 'cycle': True},
    '养殖': {'market_size_yi': 15000, 'g_by_year': {2021: -25, 2022: 0, 2023: -10, 2024: 5, 2025: 8}, 'source': '内置回退', 'cycle': True},
    '煤炭': {'market_size_yi': 25000, 'g_by_year': {2021: 50, 2022: 30, 2023: -10, 2024: -15, 2025: -8}, 'source': '内置回退', 'cycle': True},
    '半导体': {'market_size_yi': 15000, 'g_by_year': {2021: 25, 2022: 18, 2023: 5, 2024: 18, 2025: 20}, 'source': '内置回退', 'cycle': False},
}

INDUSTRY_META = _load_industry_meta()

GDP_GROWTH = {2021: 8.1, 2022: 3.0, 2023: 5.2, 2024: 5.0, 2025: 5.0, 2026: 4.8}

# 周期行业集合 (与 industry_meta.json 的 cycle 标记双轨, 此处兜底)
CYCLICAL_INDUSTRIES = {k for k, v in INDUSTRY_META.items() if v.get('cycle')} | {
    '钢铁', '化工', '航运', '有色', '地产链'}

# (B4 修复 + A 方案 2026-09-10) 周期位置集中管理: 人工判断口径 (来源: 商品价格/产能周期研报, 需定期复核)
#   三态: 底部 -> 门控底部豁免PE + s1 +1; 上行中段 -> PE>40 淘汰 + s1 0; 顶部 -> 门控一律淘汰 + s1 -1
#   与 gating.md 周期专节 A 方案、scoring.md s1/s6 联动同一口径
CYCLE_POSITION = {
    '光伏': ('底部', 'CPIA 2025 硅料价格触底, 产能出清中 (人工判断)'),
    '养殖': ('底部', '2025-2026 猪价周期底部, 能繁去化 (人工判断)'),
    '煤炭': ('顶部', '2023 起煤价高位回落, 需求承压 (人工判断)'),
    '面板': ('上行中段', '面板价格 2025 触底回升初期, 未到历史峰值 (人工判断)'),
    '有色': ('上行中段', '铜金价格上行中段, 紫金/西矿 PE 10-20 倍绝对估值不贵 (人工判断)'),
}

# 新常量 (B3/B8/B5 修复)
S8_WINDOW_DAYS = 365     # s8 增减持查询窗口: 近 12 个月 (原 120 天数据稀疏 2/25)
PS_HIST_DAYS = 130       # G3 PS 历史分位: 拉近 ~130 自然日 ps_ttm 序列
PS_PCT_HIGH = 0.70       # 自身历史 PS 分位 > 70% 高估淘汰
PS_PCT_LOW = 0.30        # < 30% 低估
# A 方案 (2026-09-10 用户钦定): 周期股绝对 PE 上限
CYCLE_PE_CAP = 40.0      # 上行中段 TTM PE > 40 -> 淘汰 (兜底上限, 通过不代表便宜)

# ============================================================
def get_pro():
    return get_tushare_api()

def safe_call(fn, *args, retries=2, **kwargs):
    for i in range(retries+1):
        try:
            df = fn(*args, **kwargs)
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            if i == retries: return pd.DataFrame()
            time.sleep(0.5)

def cache_path(name):
    return OUT_DIR / f'{name}.csv'

def load_or_fetch(pro, name, fetch_fn):
    cp = cache_path(name)
    if cp.exists():
        return pd.read_csv(cp)
    df = fetch_fn(pro)
    if len(df) > 0:
        df.to_csv(cp, index=False)
    return df

def collect_one(pro, ts_code, name, expected):
    data = {'ts_code': ts_code, 'name': name, 'expected': expected}
    # 1) 基础
    df = safe_call(pro.stock_basic, ts_code=ts_code, fields='ts_code,name,industry,list_date,is_st')
    if len(df) > 0:
        r = df.iloc[0]
        data['industry_l1'] = r['industry']
        data['list_date'] = r['list_date']
        data['is_st'] = r.get('is_st', 'N')
    # 2) 最新日线
    df = safe_call(pro.daily_basic, ts_code=ts_code, trade_date=TODAY,
                   fields='ts_code,trade_date,close,total_mv,pe,pe_ttm,pb,ps,ps_ttm,total_share,circ_mv,h_float_share')
    if len(df) == 0:
        for back in range(1, 15):
            d_back = (datetime.strptime(TODAY, '%Y%m%d') - timedelta(days=back)).strftime('%Y%m%d')
            df2 = safe_call(pro.daily_basic, trade_date=d_back, fields='ts_code,trade_date,close,total_mv,pe,pe_ttm,pb,ps,ps_ttm,total_share,circ_mv,h_float_share')
            if len(df2) > 0:
                df2 = df2[df2['ts_code'] == ts_code]
                if len(df2) > 0:
                    df = df2
                    break
    if len(df) > 0:
        r = df.iloc[0]
        data['price_date'] = r['trade_date']
        data['close'] = float(r['close']) if pd.notna(r['close']) else None
        data['total_mv_yi'] = float(r['total_mv']) / 1e4 if pd.notna(r['total_mv']) else None  # 万元→亿元
        data['pe'] = float(r['pe']) if pd.notna(r['pe']) else None
        data['pe_ttm'] = float(r['pe_ttm']) if pd.notna(r['pe_ttm']) else None
        data['pb'] = float(r['pb']) if pd.notna(r['pb']) else None
        data['ps_ttm'] = float(r['ps_ttm']) if pd.notna(r['ps_ttm']) else None
        data['total_share'] = float(r['total_share']) * 1e4 if pd.notna(r['total_share']) else None  # 万股→股
        data['h_float_share'] = float(r['h_float_share']) * 1e4 if 'h_float_share' in r and pd.notna(r['h_float_share']) else 0.0  # 万股→股 (A+H s9 修正 P1-8)
    # 2b) 近 61 日日线 (P1-8 散户化陷阱: 近60日涨幅)
    df = load_or_fetch(pro, f'd60_{ts_code}',
        lambda p: safe_call(p.daily, ts_code=ts_code, start_date=(datetime.now() - timedelta(days=90)).strftime('%Y%m%d'),
                            end_date=TODAY, fields='ts_code,trade_date,close'))
    if len(df) > 0:
        df = df.sort_values('trade_date')
        if len(df) >= 61:
            data['close_60d_ago'] = float(df.iloc[-61]['close'])   # 升序后 -61 = 60 个交易日前
    # 3) 财务指标 (近 40 期) —— 只取 fina 类字段
    df = load_or_fetch(pro, f'fina_{ts_code}',
        lambda p: safe_call(p.fina_indicator, ts_code=ts_code, limit=40,
                            fields='ts_code,end_date,ann_date,roe,grossprofit_margin,netprofit_margin,tr_yoy,netprofit_yoy,bps'))
    if len(df) > 0:
        df = df.sort_values('end_date', ascending=False)
        df = df.drop_duplicates('end_date')   # 银行等有会计重述/双披露重复记录
        annual = df[df['end_date'].astype(str).str.endswith('1231')].head(3)
        data['fina_annual'] = []
        for _, r in annual.iterrows():
            data['fina_annual'].append({
                'end_date': r['end_date'],
                'roe': float(r['roe']) if pd.notna(r['roe']) else None,
                'gross_margin': float(r['grossprofit_margin']) if pd.notna(r['grossprofit_margin']) else None,
                'net_margin': float(r['netprofit_margin']) if pd.notna(r['netprofit_margin']) else None,
                'rev_yoy': float(r['tr_yoy']) if pd.notna(r['tr_yoy']) else None,
                'profit_yoy': float(r['netprofit_yoy']) if pd.notna(r['netprofit_yoy']) else None,
            })
    # 4) 利润表 (近 24 期, report_type=1 合并报表; comp_type 是公司类型过滤会误杀银行/保险)
    # inc2: 2026-09-10 起加 dedt_profit (P1-7 扣非口径), 新缓存名避免旧缓存字段缺失
    df = load_or_fetch(pro, f'inc2_{ts_code}',
        lambda p: safe_call(p.income, ts_code=ts_code, limit=24, report_type='1',
                            fields='ts_code,end_date,total_revenue,n_income,n_income_attr_p,dedt_profit'))
    if len(df) > 0:
        df = df.sort_values('end_date', ascending=False)
        df = df.drop_duplicates('end_date')   # 防合并/母公司双记录 (comp_type 兜底)
        annual = df[df['end_date'].astype(str).str.endswith('1231')].head(3)
        data['income_annual'] = []
        for _, r in annual.iterrows():
            data['income_annual'].append({
                'end_date': r['end_date'],
                'revenue': float(r['total_revenue']) if pd.notna(r['total_revenue']) else None,
                'net_profit': float(r['n_income']) if pd.notna(r['n_income']) else None,
                'dedt_profit': float(r['dedt_profit']) if 'dedt_profit' in r and pd.notna(r['dedt_profit']) else None,
            })
        # 全量季度 (TTM 营收计算用)
        data['income_quarters'] = [{'end_date': str(r['end_date']),
                                    'revenue': float(r['total_revenue']) if pd.notna(r['total_revenue']) else None}
                                   for _, r in df.iterrows()]
    # 5) 资产负债表 (report_type=1 合并报表)
    # bal2: 2026-09-10 起加 cip/infang_assets (P0-4 重资产新口径), 新缓存名避免旧缓存字段缺失
    df = load_or_fetch(pro, f'bal2_{ts_code}',
        lambda p: safe_call(p.balancesheet, ts_code=ts_code, limit=12, report_type='1',
                            fields='ts_code,end_date,money_cap,trad_asset,st_borr,lt_borr,bond_payable,non_cur_liab_due_1y,accounts_receiv,contract_liab,adv_receipts,goodwill,total_assets,total_liab,total_share,fix_assets,cip,intang_assets'))
    if len(df) > 0:
        df = df.sort_values('end_date', ascending=False)
        df = df.drop_duplicates('end_date')
        r = df.iloc[0]
        data['balance_latest'] = {
            'end_date': r['end_date'],
            'monetary': float(r['money_cap']) if pd.notna(r['money_cap']) else None,
            'trad_asset': float(r['trad_asset']) if pd.notna(r['trad_asset']) else None,
            'st_loan': float(r['st_borr']) if pd.notna(r['st_borr']) else None,
            'lt_loan': float(r['lt_borr']) if pd.notna(r['lt_borr']) else None,
            'bond_pay': float(r['bond_payable']) if pd.notna(r['bond_payable']) else None,
            'ltb_1y': float(r['non_cur_liab_due_1y']) if pd.notna(r['non_cur_liab_due_1y']) else None,
            'ar': float(r['accounts_receiv']) if pd.notna(r['accounts_receiv']) else None,
            'contract_liab': float(r['contract_liab']) if pd.notna(r['contract_liab']) else None,
            'advance_receipts': float(r['adv_receipts']) if pd.notna(r['adv_receipts']) else None,
            'goodwill': float(r['goodwill']) if pd.notna(r['goodwill']) else None,
            'total_assets': float(r['total_assets']) if pd.notna(r['total_assets']) else None,
            'total_liab': float(r['total_liab']) if pd.notna(r['total_liab']) else None,
            'total_share': float(r['total_share']) if pd.notna(r['total_share']) else None,  # balancesheet.total_share 已是股
            'fix_assets': float(r['fix_assets']) if 'fix_assets' in r and pd.notna(r['fix_assets']) else None,
            'cip': float(r['cip']) if 'cip' in r and pd.notna(r['cip']) else None,   # 在建工程
            'intang_assets': float(r['intang_assets']) if 'intang_assets' in r and pd.notna(r['intang_assets']) else None,
        }
        # 第二近年度 (P1-9 应收同比联动 + P0-6 重资产)
        if len(df) >= 2:
            r2 = df.iloc[1]
            data['balance_prev'] = {
                'end_date': r2['end_date'],
                'ar': float(r2['accounts_receiv']) if pd.notna(r2['accounts_receiv']) else None,
            }
    # 6) 现金流量表 (report_type=1 合并报表)
    df = load_or_fetch(pro, f'cf_{ts_code}',
        lambda p: safe_call(p.cashflow, ts_code=ts_code, limit=16, report_type='1',
                            fields='ts_code,end_date,c_fr_sale_sg,n_cashflow_act,c_pay_acq_const_fiolta,free_cashflow,net_profit'))
    if len(df) > 0:
        df = df.sort_values('end_date', ascending=False)
        df = df.drop_duplicates('end_date')
        data['cf_annual'] = []
        for _, r in df.head(4).iterrows():
            data['cf_annual'].append({
                'end_date': r['end_date'],
                'cf_from_sales': float(r['c_fr_sale_sg']) if pd.notna(r['c_fr_sale_sg']) else None,
                'ocf': float(r['n_cashflow_act']) if pd.notna(r['n_cashflow_act']) else None,
                'capex': float(r['c_pay_acq_const_fiolta']) if pd.notna(r['c_pay_acq_const_fiolta']) else None,
                'fcf': float(r['free_cashflow']) if pd.notna(r['free_cashflow']) else None,
            })
    # 7) 分红
    df = load_or_fetch(pro, f'div_{ts_code}',
        lambda p: safe_call(p.dividend, ts_code=ts_code, limit=50,
                            fields='ts_code,end_date,ann_date,cash_div,cash_div_tax,div_proc'))
    if len(df) > 0:
        data['dividends'] = []
        for _, r in df.iterrows():
            data['dividends'].append({
                'end_date': r['end_date'],
                'cash_div': float(r['cash_div']) if pd.notna(r['cash_div']) else None,  # 元/股（税前），茅台2023年度30.876元实证
            })
    # 8) 股东人数
    df = load_or_fetch(pro, f'hol_{ts_code}',
        lambda p: safe_call(p.stk_holdernumber, ts_code=ts_code,
                            fields='ts_code,ann_date,end_date,holder_num,holder_total_change'))
    if len(df) > 0:
        df = df.sort_values('end_date', ascending=False)
        r = df.iloc[0]
        data['holder_latest'] = {'end_date': r['end_date'], 'holder_num': int(r['holder_num']) if pd.notna(r['holder_num']) else None}
    # 8b) 前10大股东
    df = load_or_fetch(pro, f'top10_{ts_code}',
        lambda p: safe_call(p.top10_holders, ts_code=ts_code,
                            fields='ts_code,ann_date,end_date,holder_name,hold_ratio,hold_amount,holder_type'))
    if len(df) > 0:
        # 取最新一期的前10大股东 (Tushare 返回多期历史, 必须先按 end_date 过滤)
        latest_end = df['end_date'].max()
        df = df[df['end_date'] == latest_end]
        df = df.sort_values('hold_ratio', ascending=False).head(10)
        data['top10_holders'] = [{'name': r['holder_name'], 'hold_ratio': float(r['hold_ratio']) if pd.notna(r['hold_ratio']) else 0, 'hold_amount': float(r['hold_amount']) if pd.notna(r['hold_amount']) else 0, 'holder_type': r.get('holder_type'), 'end_date': str(r.get('end_date',''))} for _, r in df.iterrows()]
    # 9) 增减持 (B3 修复: 窗口从近3月扩到近12月, 原 120 天数据稀疏 2/25)
    try:
        start_d = (datetime.now() - timedelta(days=S8_WINDOW_DAYS)).strftime('%Y%m%d')
        df = safe_call(pro.stk_holdertrade, ts_code=ts_code, start_date=start_d, end_date=TODAY,
                       fields='ts_code,ann_date,holder_name,holder_type,in_de,change_vol,change_ratio')
        data['trades_3m'] = len(df) if len(df) > 0 else 0
        data['trades_sample'] = df.head(5).to_dict('records') if len(df) > 0 else []
        if len(df) > 0:
            in_de_s = df['in_de'].astype(str).str.upper()
            data['trades_in_n'] = int((in_de_s == 'IN').sum())
            data['trades_de_n'] = int((in_de_s == 'DE').sum())
        else:
            data['trades_in_n'] = 0
            data['trades_de_n'] = 0
    except Exception as e:
        data['trades_3m'] = -1
        data['trades_error'] = str(e)[:100]
    # 10) 证券简称变更历史 (P0-2 R1 更名史)
    df = load_or_fetch(pro, f'nm_{ts_code}',
        lambda p: safe_call(p.namechange, ts_code=ts_code, fields='ts_code,name,start_date,end_date,change_reason'))
    if len(df) > 0:
        data['name_changes'] = [{'name': r['name'], 'start_date': str(r['start_date']), 'end_date': str(r.get('end_date', '')) if pd.notna(r.get('end_date')) else ''} for _, r in df.iterrows()]
    # 11) 回购 (P1-1 s8 回购两档: 注销 vs 激励)
    df = load_or_fetch(pro, f'rep_{ts_code}',
        lambda p: safe_call(p.repurchase, ts_code=ts_code, fields='ts_code,proc,vol,amount'))
    if len(df) > 0:
        data['repurchase'] = [{'proc': r['proc'], 'vol': float(r['vol']) if pd.notna(r['vol']) else 0, 'amount': float(r['amount']) if pd.notna(r['amount']) else 0} for _, r in df.iterrows()]
    return data

# ============================================================
# Skill 逻辑 (与 references/*.md 1:1 对应)
# ============================================================
def industry_key(data):
    """优先用 name→key 显式映射(精确); 退化用 Tushare industry(申万一级)"""
    name = data['name']
    m = {
        '贵州茅台':'白酒','五粮液':'白酒','中国移动':'通信',
        '招商银行':'银行','平安银行':'银行','工商银行':'银行',
        '恒瑞医药':'医药','复星医药':'医药','美的集团':'家电',
        '比亚迪':'汽车','隆基绿能':'光伏','牧原股份':'养殖',
        '陕西煤业':'煤炭','中国神华':'煤炭','宁德时代':'电池',
        '迈瑞医疗':'医疗器械','伊利股份':'乳制品','立讯精密':'电子',
        '士兰微':'电子','东方财富':'券商','中芯国际':'半导体',
        '海康威视':'安防','中国平安':'保险','京东方A':'面板',
        '科大讯飞':'AI软件','凯莱英':'CXO',
    }
    explicit = m.get(name)
    if explicit: return explicit
    # 退化: Tushare 申万一级 → INDUSTRY_META
    sw = data.get('industry_l1', '')
    sw_map = {
        '白酒':'白酒','银行':'银行','电信运营':'通信','通信':'通信',
        '化学制药':'医药','医疗保健':'医药','医疗器械':'医疗器械',
        '家用电器':'家电','汽车整车':'汽车','电气设备':'光伏',
        '农业综合':'养殖','煤炭开采':'煤炭','乳制品':'乳制品',
        '元器件':'电子','半导体':'半导体','证券':'券商',
        'IT设备':'安防','保险':'保险','软件服务':'AI软件',
        '生物制药':'CXO','全国地产':'银行','区域地产':'银行',
    }
    return sw_map.get(sw, sw)

def is_cyclical(data):
    return industry_key(data) in CYCLICAL_INDUSTRIES

def is_stable_profit(data):
    """B9 用户钦定口径: 稳定盈利 = 近3年每年净利均 > 0。
    返回三态: True=稳定(走PR) / False=非稳定(任一年净利≤0, 走PS) / None=数据不足(年份或净利缺失)"""
    ia = data.get('income_annual', [])
    if len(ia) < 3:
        return None
    vals = [x.get('net_profit') for x in ia[:3]]
    if any(v is None for v in vals):
        return None
    return all(v > 0 for v in vals)


FINANCIAL_INDS = {'银行', '保险', '券商'}   # 金融股不适用净现金调整 (报表科目差异)

# 重资产豁免"无形不计入"行业清单 (2026-09-10 用户钦定): 医药/软件/传媒类无形主体=并购溢价/批文/版权
HEAVY_INTAN_EXCLUDE = {
    '化学制药', '生物制药', '中成药', '医药商业', '医疗服务', '医疗器械', '中药生产',
    '软件开发', 'IT服务', '互联网服务', '游戏', '影视院线', '文化传媒', '出版', '通信服务',
    '医药生物', '计算机', '传媒',
}

def heavy_ratio_ind(data, bal):
    """重资产比例 (行业感知): (固资+在建) 无行业限制; +无形 仅清单外行业; 行业缺失不计无形 (豁免从严)"""
    ta = bal.get('total_assets') or 0
    if ta <= 0: return None
    base = (bal.get('fix_assets') or 0) + (bal.get('cip') or 0)
    ind = data.get('industry_l1') or data.get('industry') or ''
    if ind and ind not in HEAVY_INTAN_EXCLUDE:
        base += bal.get('intang_assets') or 0
    return base / ta


def calc_ttm_revenue(data):
    """TTM 营收 = 最新期累计 + 去年年报 − 去年同期累计; 数据缺失回退最新年报营收"""
    rows = sorted([r for r in data.get('income_quarters', []) if r.get('revenue')],
                  key=lambda r: r['end_date'])
    ia = data.get('income_annual', [])
    if len(rows) < 2:
        return ia[0].get('revenue') if ia else None
    latest = rows[-1]
    ly = int(latest['end_date'][:4])
    fy_key = str(ly - 1) + '1231'
    same_last = str(ly - 1) + latest['end_date'][4:]
    by_date = {r['end_date']: r['revenue'] for r in rows}
    fy = by_date.get(fy_key)
    sl = by_date.get(same_last)
    if fy and sl:
        return latest['revenue'] + fy - sl
    return ia[0].get('revenue') if ia else None


def fetch_ps_history(pro, ts_code, data):
    """B5: 自身历史 PS 近似序列 = 日线收盘价 × 当前总股本 ÷ TTM营收 (~130 自然日, 1 次调用)。
    ps_now 同口径 (total_mv/ttm_rev), 序列可比; 误差来自股本变动与 TTM 用当前值回推, 已注明。
    """
    cp = cache_path(f'pshist_{ts_code}')
    if cp.exists():
        try:
            df = pd.read_csv(cp)
            vals = [float(v) for v in df['ps'].tolist() if pd.notna(v) and v > 0]
            if len(vals) >= 30:
                return vals
        except Exception:
            pass
    ttm = calc_ttm_revenue(data)
    close_now = data.get('close')
    mv_yi = data.get('total_mv_yi')
    if not ttm or ttm <= 0 or not close_now or not mv_yi:
        return None
    share = mv_yi * 1e8 / close_now          # 当前总股本(股)
    start_d = (datetime.now() - timedelta(days=PS_HIST_DAYS)).strftime('%Y%m%d')
    df = safe_call(pro.daily, ts_code=ts_code, start_date=start_d, end_date=TODAY,
                   fields='trade_date,close')
    if len(df) == 0:
        return None
    df = df.sort_values('trade_date')
    df['ps'] = df['close'] * share / ttm
    df.to_csv(cp, index=False)
    vals = [float(v) for v in df['ps'].tolist() if pd.notna(v) and v > 0]
    return vals if len(vals) >= 30 else None

# ---- 门控 ----
def _share_core(a, b, k=2):
    """两简称是否共享 >=k 个连续字符的核心词 (R1 壳股换马甲判定)。
    清洗 *ST/G 等非核心词缀: 伊利股份<->*ST伊利 共享'伊利'=经营型戴帽; 永生数据->神奇制药 断裂=壳股。"""
    def clean(s):
        return (s or '').replace('*', '').replace('ST', '').replace('st', '').replace('G', '')
    a, b = clean(a), clean(b)
    la, lb = len(a), len(b)
    best = 0
    for i in range(la):
        for j in range(lb):
            l = 0
            while i + l < la and j + l < lb and a[i + l] == b[j + l]:
                l += 1
            best = max(best, l)
    return best >= k

def gate_mine_sweep(data, peer_cache=None):
    """P0-2 排雷门控 (2026-09-10): R1 更名史 / R2 财务洗澡 / R3 超额分红套现。命中任一 -> 剔除。
    快刀斩股最核心能力: 先证伪、再估值。神奇制药/华电辽能/云南锗业三期实证。"""
    hits = []
    # R1 更名史 (核心词断裂修正 2026-09-10): 曾 ST 且更名序列存在核心词断裂 (壳股换马甲)
    #   伊利 2009 *ST(三聚氰胺)/京东方 2007 *ST(面板周期) 核心词贯穿 -> 经营型戴帽, 不剔除
    ncs = data.get('name_changes') or []
    ever_st = any(('ST' in (c['name'] or '').upper()) for c in ncs)
    if ever_st and len(ncs) >= 2:
        ncs_sorted = sorted(ncs, key=lambda c: str(c.get('start_date', '')))
        broken = any(not _share_core(ncs_sorted[i]['name'], ncs_sorted[i + 1]['name'])
                     for i in range(len(ncs_sorted) - 1))
        if broken:
            hits.append({'rule': 'R1', 'detail': f'曾ST + 更名核心词断裂(壳股换马甲): {[c["name"] for c in ncs_sorted]}'})
    # R2 财务洗澡: 近5年(可得近3年)任一年净利<0 且同行净利中位数>0
    ia = data.get('income_annual', [])
    loss_year = next((x for x in ia if x.get('net_profit') is not None and x['net_profit'] < 0), None)
    if loss_year is not None:
        peer_med = None
        if peer_cache:
            ikey = industry_key(data)
            peer_nps = [(d.get('income_annual') or [{}])[0].get('net_profit') for d in peer_cache.values()
                        if d.get('ts_code') != data['ts_code'] and industry_key(d) == ikey
                        and (d.get('income_annual') or [{}]) and (d.get('income_annual') or [{}])[0].get('net_profit') is not None]
            if len(peer_nps) >= 3:
                peer_med = pd.Series(peer_nps).median()
        if peer_med is not None and peer_med > 0:
            hits.append({'rule': 'R2', 'detail': f'亏损年 {loss_year["end_date"]} 净利{loss_year["net_profit"]/1e8:.1f}亿 但同行中位数 {peer_med/1e8:.1f}亿>0 (样本内近似)'})
    # R3 超额分红套现: 近5年(可得近3年)支付率>100% >=2 次 且 近12月减持
    divs = data.get('dividends') or []
    ia_by_year = {str(x.get('end_date', ''))[:4]: x for x in ia}
    total_share = data.get('total_share') or (data.get('balance_latest') or {}).get('total_share')
    over100_years = set()
    if total_share and total_share > 0:
        for d in divs:
            y = str(d.get('end_date') or '')[:4]
            if not y or not d.get('cash_div'): continue
            ia_y = ia_by_year.get(y)
            if ia_y and ia_y.get('net_profit') and ia_y['net_profit'] > 0:
                if d['cash_div'] * total_share / ia_y['net_profit'] > 1.0:
                    over100_years.add(y)
    de_n = data.get('trades_de_n') or 0
    if len(over100_years) >= 2 and de_n > 0:
        hits.append({'rule': 'R3', 'detail': f'支付率>100% 年份 {sorted(over100_years)} + 近12月减持 {de_n} 条'})
    if hits:
        return {'pass': False, 'hits': hits, 'detail': '; '.join(h['detail'] for h in hits)}
    return {'pass': True, 'hits': [], 'detail': '排雷通过'}

def gate_industry(data):
    ikey = industry_key(data)
    if ikey not in INDUSTRY_META:
        return {'pass': False, 'verdict': 'INDUSTRY_META_MISSING', 'na': True, 'detail': f'未在 INDUSTRY_META 中配置 {ikey}'}
    meta = INDUSTRY_META[ikey]
    size_ok = meta['market_size_yi'] >= 1000
    g = meta['g_by_year']
    fast_yrs = [y for y, r in g.items() if r is not None and (r >= GDP_GROWTH.get(y, 5) + 2 or r >= 1.5 * GDP_GROWTH.get(y, 5))]
    stable_yrs = [y for y, r in g.items() if r is not None and abs(r - GDP_GROWTH.get(y, 5)) <= 2 and r > 0]
    if not size_ok:
        return {'pass': False, 'verdict': '空间不足', 'detail': f'规模 {meta["market_size_yi"]}亿<千亿'}
    if len(fast_yrs) >= max(1, len(g)//2):
        return {'pass': True, 'verdict': '金饭碗', 'detail': f'规模 {meta["market_size_yi"]}亿, 显著高于GDP年份 {fast_yrs}', 'source': meta['source']}
    if len(stable_yrs) >= max(1, len(g)//2):
        return {'pass': True, 'verdict': '不掉血优质赛道', 'detail': f'规模 {meta["market_size_yi"]}亿, 稳定年份 {stable_yrs}', 'source': meta['source']}
    return {'pass': False, 'verdict': '增速不足', 'detail': f'fast_yrs={len(fast_yrs)}, stable_yrs={len(stable_yrs)}, total={len(g)}'}

def gate_company(data, peer_cache, market_snap=None, stock_ind_map=None, market_fina=None):
    ikey = industry_key(data)
    is_fin = ikey in FINANCIAL_INDS
    my = (data.get('fina_annual') or [{}])[0] if data.get('fina_annual') else None
    # 金融股无"毛利率"科目: 用 {净利率, ROE} 两项, ≥1 项行业TOP3 (口径折中, 文档已注明)
    metrics = ['net_margin', 'roe'] if is_fin else ['gross_margin', 'net_margin', 'roe']
    if not my or any(my.get(m) is None for m in metrics):
        return {'pass': None, 'na': True, 'detail': '财务指标缺失'}
    ranks, peer_n = {}, None
    for m in metrics:
        r, n = None, 0
        if market_fina and stock_ind_map and data.get('industry_l1'):
            r, n = _peer_rank_metric(data['ts_code'], data['industry_l1'],
                                     stock_ind_map, market_fina, m)
        if r is None:
            # 样本内回退
            peers = [(t, ((d.get('fina_annual') or [{}])[0].get(m) or 0)) for t, d in peer_cache.items()
                     if industry_key(d) == ikey and d.get('fina_annual')
                     and (d.get('fina_annual') or [{}])[0].get(m) is not None]
            peers.sort(key=lambda x: x[1], reverse=True)
            for rr, (t, v) in enumerate(peers, 1):
                if t == data['ts_code']:
                    r = rr
                    break
            n = len(peers)
        ranks[m] = r
        if peer_n is None:
            peer_n = n
    top3 = sum(1 for m in metrics if ranks[m] is not None and ranks[m] <= 3)
    need = 1 if is_fin else 2
    scope = '全A' if market_fina and stock_ind_map and data.get('industry_l1') else '样本内'
    return {'pass': top3 >= need, 'top3_count': top3, 'need': need,
            'ranks': ranks, 'peer_count': peer_n,
            'metric_scope': ('金融股两项口径' if is_fin else '标准三项') + '/' + scope, 'na': False}

def _detect_wash(data):
    """P0-1a 财务洗澡识别: 近3年任一年净利>0 但扣非<0 (非经常性损益兜底由盈转亏), 五粮液实证"""
    ia = data.get('income_annual', [])
    for x in ia[:2]:
        np_ = x.get('net_profit'); dedt = x.get('dedt_profit')
        if np_ is not None and dedt is not None and np_ > 0 and dedt < 0:
            return {'year': str(x.get('end_date'))[:4], 'np': np_, 'dedt': dedt}
    return None

def _has_buyback_support(data):
    """P0-1a 洗澡例外佐证: 大额回购(≥1%市值) 或 大股东/高管增持"""
    reps = data.get('repurchase') or []
    mv = data.get('total_mv_yi')
    if mv and mv > 0:
        amt = sum(r.get('amount') or 0 for r in reps)
        if amt >= mv * 1e8 * 0.01:
            return True
    return (data.get('trades_in_n') or 0) > 0

def gate_price_cyclical(data):
    """P0-1b 周期股 A 方案门控 (2026-09-10 用户钦定): 周期位置三态 + 绝对 PE 上限。
    前版 PB/PS 自身历史分位被 18 期数据证伪 (6 只正确率 3/6), A 方案 6/6 全对。"""
    ikey = industry_key(data)
    pos = CYCLE_POSITION.get(ikey)
    if pos is None:
        return {'pass': None, 'method': 'CYCLE', 'na': True,
                'detail': f'周期股 {ikey} 周期位置未配置 (A 方案需人工标注+研报佐证)'}
    state, source = pos
    pe = data.get('pe_ttm')
    if state == '顶部':
        return {'pass': False, 'method': 'CYCLE', 'verdict': '周期顶部淘汰', 'pe': pe,
                'detail': f'{ikey} 处周期顶部 ({source}): 低PE陷阱, 盈利顶峰的低PE是卖点不是买点'}
    if state == '底部':
        if pe is None or pe <= 0 or pe > CYCLE_PE_CAP:
            return {'pass': True, 'method': 'CYCLE', 'verdict': '底部豁免PE', 'pe': pe,
                    'detail': f'{ikey} 周期底部 ({source}): 亏损/爆表PE豁免 (反转前夜特征, 排雷门控已兜底)'}
        return {'pass': True, 'method': 'CYCLE', 'verdict': '通过', 'pe': pe,
                'detail': f'{ikey} 周期底部 ({source}): PE={pe:.1f}≤{CYCLE_PE_CAP}'}
    # 上行中段
    if pe is None:
        return {'pass': None, 'method': 'CYCLE', 'na': True, 'detail': f'{ikey} 上行中段 TTM PE 缺失'}
    if pe > CYCLE_PE_CAP:
        return {'pass': False, 'method': 'CYCLE', 'verdict': 'PE>40淘汰', 'pe': pe,
                'detail': f'{ikey} 上行中段 TTM PE={pe:.1f}>{CYCLE_PE_CAP} (云南锗业 679 倍实证)'}
    return {'pass': True, 'method': 'CYCLE', 'verdict': '通过', 'pe': pe,
            'detail': f'{ikey} 上行中段 ({source}): TTM PE={pe:.1f}≤{CYCLE_PE_CAP} (绝对判断)'}

def gate_price(data, pro=None):
    # P0-1b 周期股 A 方案分支
    if is_cyclical(data):
        return gate_price_cyclical(data)
    # P0-1a 洗澡例外: 净利>0 但扣非<0 + 大额回购/增持佐证 -> 改走 PS 路径 (五粮液实证)
    wash = _detect_wash(data)
    if wash and _has_buyback_support(data):
        if pro is None:
            return {'pass': None, 'method': 'PS(洗澡例外)', 'na': True, 'detail': '无 pro 句柄'}
        ttm = calc_ttm_revenue(data)
        mv_yi = data.get('total_mv_yi')
        if not ttm or ttm <= 0 or not mv_yi:
            return {'pass': None, 'method': 'PS(洗澡例外)', 'na': True, 'detail': 'TTM营收/市值缺失'}
        ps_now = mv_yi * 1e8 / ttm
        vals = fetch_ps_history(pro, data['ts_code'], data)
        if not vals:
            return {'pass': None, 'method': 'PS(洗澡例外)', 'na': True, 'detail': '历史 PS 序列不足'}
        pct = float((pd.Series(vals) < ps_now).mean() + 0.5 * (pd.Series(vals) == ps_now).mean())
        if pct > PS_PCT_HIGH:
            return {'pass': False, 'method': 'PS(洗澡例外)', 'na': False, 'verdict': '高估',
                    'ps_pct': round(pct, 3), 'detail': f'洗澡例外({wash["year"]} 扣非为负)+回购增持佐证, 自身历史PS分位{pct:.0%}>70%'}
        return {'pass': True, 'method': 'PS(洗澡例外)', 'na': False,
                'verdict': '低估' if pct < PS_PCT_LOW else '合理', 'ps_pct': round(pct, 3),
                'detail': f'洗澡例外({wash["year"]} 扣非为负)+回购增持佐证, 自身历史PS分位{pct:.0%}'}
    stable = is_stable_profit(data)
    if stable is None: return {'pass': None, 'method': 'N/A', 'na': True, 'detail': '近3年净利润数据不足'}
    if stable:
        # ---- PR 路径: 需要市值/PE/净现金/去年净利/ROE ----
        if data.get('total_mv_yi') is None or data.get('pe_ttm') is None:
            return {'pass': None, 'method': 'PR', 'na': True, 'detail': '市值/PE 缺失'}
        if industry_key(data) in FINANCIAL_INDS:
            # 银行/保险/券商报表无"货币资金"科目, 净现金调整不适用 -> 用总市值
            net_cash = 0.0
            adj_mv = data['total_mv_yi'] * 1e8
        else:
            bal = data.get('balance_latest')
            if not bal or bal.get('monetary') is None:
                return {'pass': None, 'method': 'PR', 'na': True, 'detail': '净现金数据缺失'}
            monetary = bal.get('monetary') or 0
            trad = bal.get('trad_asset') or 0
            debt = (bal.get('st_loan') or 0) + (bal.get('lt_loan') or 0) + (bal.get('bond_pay') or 0) + (bal.get('ltb_1y') or 0)
            net_cash = monetary + trad - debt
            adj_mv = data['total_mv_yi'] * 1e8 - net_cash
        ia = data.get('income_annual', [])
        if len(ia) < 1 or ia[0].get('net_profit') is None:
            return {'pass': None, 'method': 'PR', 'na': True, 'detail': '去年净利润缺失'}
        yoy = (data.get('fina_annual') or [{}])[0].get('profit_yoy')  # 用 fina 的 YoY
        est_np = ia[0]['net_profit'] * (1 + (yoy or 0)/100)
        if est_np <= 0: return {'pass': None, 'method': 'PR', 'na': True, 'detail': '预估净利≤0'}
        if adj_mv <= 0: return {'pass': None, 'method': 'PR', 'na': True, 'detail': '调整后市值为负'}
        growth_pe = adj_mv / est_np
        roe = (data.get('fina_annual') or [{}])[0].get('roe')
        if not roe or roe <= 0: return {'pass': None, 'method': 'PR', 'na': True, 'detail': 'ROE 缺失或非正'}
        pr = growth_pe / roe
        if pr < 1:   return {'pass': True, 'method': 'PR', 'pr': round(pr, 3), 'verdict': '低估', 'adj_mv_yi': round(adj_mv/1e8, 1), 'net_cash_yi': round(net_cash/1e8, 1)}
        if pr <= 1.5:return {'pass': True, 'method': 'PR', 'pr': round(pr, 3), 'verdict': '合理', 'adj_mv_yi': round(adj_mv/1e8, 1), 'net_cash_yi': round(net_cash/1e8, 1)}
        return {'pass': False, 'method': 'PR', 'pr': round(pr, 3), 'verdict': '高估', 'adj_mv_yi': round(adj_mv/1e8, 1), 'net_cash_yi': round(net_cash/1e8, 1), 'na': False}
    else:
        # ---- (B5 修复) PS 路径: 自身历史分位, ps_now 与序列同口径 (总市值÷TTM营收) ----
        if pro is None:
            return {'pass': None, 'method': 'PS', 'na': True, 'detail': '无 pro 句柄, 未拉取历史 PS'}
        ttm = calc_ttm_revenue(data)
        mv_yi = data.get('total_mv_yi')
        if not ttm or ttm <= 0 or not mv_yi:
            return {'pass': None, 'method': 'PS', 'na': True, 'detail': 'TTM营收/市值缺失'}
        ps_now = mv_yi * 1e8 / ttm
        vals = fetch_ps_history(pro, data['ts_code'], data)
        if not vals:
            return {'pass': None, 'method': 'PS', 'na': True, 'detail': '历史 PS 序列不足(上市不满或接口失败)'}
        pct = float((pd.Series(vals) < ps_now).mean() + 0.5 * (pd.Series(vals) == ps_now).mean())
        n_days = len(vals)
        if pct > PS_PCT_HIGH:
            return {'pass': False, 'method': 'PS', 'na': False, 'verdict': '高估',
                    'ps_pct': round(pct, 3), 'ps_now': round(ps_now, 2),
                    'detail': f'自身历史PS分位{pct:.0%}>70% (ps_ttm序列{n_days}天, 近似口径: 未扣净现金)'}
        verdict = '低估' if pct < PS_PCT_LOW else '合理'
        return {'pass': True, 'method': 'PS', 'na': False, 'verdict': verdict,
                'ps_pct': round(pct, 3), 'ps_now': round(ps_now, 2),
                'detail': f'自身历史PS分位{pct:.0%} (ps_ttm序列{n_days}天, 近似口径: 未扣净现金)'}

# ---- 打分 ----
def s1_revenue_profit(data):
    if is_cyclical(data):
        ikey = industry_key(data)
        pos = CYCLE_POSITION.get(ikey)
        if pos:
            state, source = pos
            if state == '底部': return 1, f'周期股 {ikey} 处于底部 ({source})'
            if state == '顶部': return -1, f'周期股 {ikey} 处于顶部 ({source})'
            return 0, f'周期股 {ikey} 位置 {state} ({source})'
        return 0, f'周期股 {ikey} 周期位置未配置 (B4: 需商品价格等客观信号, 暂记 0)'
    ia = data.get('income_annual', [])
    if len(ia) < 3: return None, '近3年净利润数据不足'
    # P1-7 扣非口径: 近3年累计非经常性损益绝对值 > 累计净利绝对值 50% -> 改用扣非
    use_dedt = False
    nps = [x.get('net_profit') for x in ia[:3]]
    ddts = [x.get('dedt_profit') for x in ia[:3]]
    if all(v is not None for v in nps) and all(v is not None for v in ddts):
        if abs(sum(nps)) > 0 and abs(sum(nps) - sum(ddts)) > abs(sum(nps)) * 0.5:
            use_dedt = True
    prof = lambda x: (x.get('dedt_profit') if use_dedt else x.get('net_profit'))
    # 严格逐年增长: ia[0] 最新 (2024), ia[1] 2023, ia[2] 2022; 要求 0>1>2
    grew = True
    for i in range(len(ia)-1):
        if not (ia[i].get('revenue') and ia[i+1].get('revenue')) or ia[i]['revenue'] <= ia[i+1]['revenue']:
            grew = False; break
        pi, pi1 = prof(ia[i]), prof(ia[i+1])
        if not (pi and pi1) or pi <= pi1:
            grew = False; break
    # 暴跌检测
    yoys = [data['fina_annual'][i].get('profit_yoy') for i in range(min(3, len(data.get('fina_annual', []))))]
    crash = any(y is not None and y < -50 for y in yoys)
    dedt_note = ' (扣非口径 P1-7)' if use_dedt else ''
    if grew: return 1, f'近3年营收与净利润均逐年增长{dedt_note}'
    if crash: return -1, f'近3年出现净利润大幅下跌 yoy< -50%: {yoys}{dedt_note}'
    return 0, f'未满足逐年增长且无暴跌 yoys={yoys}{dedt_note}'

def s2_roe(data):
    if is_cyclical(data): return 0, '周期股不判断'
    fa = data.get('fina_annual', [])
    if len(fa) < 3: return None, '近3年ROE数据不足'
    roes = [fa[i].get('roe') for i in range(3)]
    if any(r is None for r in roes): return None, 'ROE 含空值'
    # P0-5 趋势降档: 均≥15% 但逐年下降且累计降幅>30% -> 0 (迈瑞 38.2% 实证)
    if all(r >= 15 for r in roes):
        if roes[0] < roes[1] < roes[2] and (roes[2] - roes[0]) / roes[2] > 0.30:
            return 0, f'近3年ROE均≥15%但趋势性恶化 (P0-5): {roes}, 累计降幅{(roes[2]-roes[0])/roes[2]:.0%}>30%'
        return 1, f'近3年ROE均≥15%无趋势恶化: {roes}'
    if all(r < 10 for r in roes): return -1, f'近3年ROE均<10%: {roes}'
    return 0, f'近3年ROE不极端: {roes}'

def s3_balance(data):
    ikey = industry_key(data)
    # P1-4 金融股分支: 三指标(不良率/拨备覆盖率/资本充足率) Tushare 无标准接口 -> N/A
    if ikey in FINANCIAL_INDS:
        return None, '金融股第3项改用不良率/拨备覆盖率/资本充足率三指标 (P1-4), Tushare 无标准接口, 记 N/A'
    bal = data.get('balance_latest')
    ia = data.get('income_annual', [])
    if not bal: return None, '资产负债表缺失'
    if not ia or len(ia) < 1: return None, '营收数据缺失'
    rev = ia[0].get('revenue') or 0
    ar = bal.get('ar') or 0
    contract = (bal.get('contract_liab') or 0) + (bal.get('advance_receipts') or 0)
    monetary = bal.get('monetary') or 0
    trad = bal.get('trad_asset') or 0
    debt = (bal.get('st_loan') or 0) + (bal.get('lt_loan') or 0) + (bal.get('bond_pay') or 0) + (bal.get('ltb_1y') or 0)
    net_cash = monetary + trad - debt
    total_cash = monetary + trad
    goodwill = bal.get('goodwill') or 0
    total_assets = bal.get('total_assets') or 0
    a = (ar > 0 and rev > 0 and (ar/rev) <= 0.20) or (contract > 0 and rev > 0 and (contract/rev) >= 0.20)
    b = total_cash > 0 and (net_cash / total_cash) > 0.80
    c = goodwill == 0 or (total_assets > 0 and (goodwill/total_assets) <= 0.05)
    # P0-4 重资产豁免 (2026-09-10 口径分层: 固资+在建无限制; +无形需行业不在医药/软件/传媒清单), 前提净现比>=1
    hr = heavy_ratio_ind(data, bal)
    heavy_ok = hr is not None and hr > 0.40
    heavy_scope = '含无形' if (data.get('industry_l1') or data.get('industry') or '') and (data.get('industry_l1') or data.get('industry') or '') not in HEAVY_INTAN_EXCLUDE else '纯固资+在建'
    if heavy_ok:
        cf = data.get('cf_annual', [])
        ia0 = ia[0] if ia else {}
        ocf = cf[0].get('ocf') if cf else None
        np0 = ia0.get('net_profit')
        npr = (ocf / np0) if (ocf and np0) else None
        if npr is not None and npr >= 1.0:
            b = True   # "矿山重资产、负债普遍, 只要现金流没问题就不扣分"
    ar_pct = ar/rev if rev>0 else 0
    nc_pct = net_cash/total_cash if total_cash>0 else 0
    gw_pct = goodwill/total_assets if total_assets>0 else 0
    heavy_note = f' [重资产豁免({heavy_scope}){hr*100:.0f}%>40%]' if heavy_ok and b else ''
    if a and b and c: return 1, f'三指标全过 (AR={ar_pct:.1%}, NC={nc_pct:.1%}, GW={gw_pct:.1%}){heavy_note}'
    if sum([not a, not b, not c]) >= 2: return -1, f'≥2项不满足 (a={a},b={b},c={c}); AR={ar_pct:.1%}, NC/TC={nc_pct:.1%}, GW/TA={gw_pct:.1%}'
    return 0, f'部分通过 (a={a},b={b},c={c})'

def s4_cashflow(data):
    ikey = industry_key(data)
    # P1-4 金融股: 现金流科目与实体企业不可比 -> 整体 N/A
    if ikey in FINANCIAL_INDS:
        return None, '金融股第4项整体 N/A (P1-4: 金融企业现金流科目与实体企业不可比)'
    cf = data.get('cf_annual', [])
    ia = data.get('income_annual', [])
    if len(cf) < 1: return None, '现金流数据不足'
    r0 = cf[0]
    rev0 = ia[0].get('revenue') if ia else None
    np0 = ia[0].get('net_profit') if ia else None
    a = r0.get('cf_from_sales') and rev0 and r0['cf_from_sales']/rev0 > 1
    b = r0.get('ocf') and np0 and r0['ocf']/np0 > 1
    # P1-9 现金流溯源联动: b 不满足 且 应收同比>50% -> 直接 -1
    if not b:
        ar_now = (data.get('balance_latest') or {}).get('ar')
        ar_prev = (data.get('balance_prev') or {}).get('ar')
        if ar_now and ar_prev and ar_prev > 0 and (ar_now - ar_prev) / ar_prev > 0.50:
            return -1, f'净现比<1 且应收同比+{(ar_now-ar_prev)/ar_prev:.0%}>50% (P1-9 现金流恶化系应收膨胀, 直接-1)'
    capex = r0.get('capex') or 0
    c = r0.get('ocf') and (capex/r0['ocf']) <= 0.10
    # P0-4 成长期/重资产豁免 c (口径分层: 固资+在建无限制; +无形需行业不在清单), 前提净现比>=1
    bal = data.get('balance_latest') or {}
    hr = heavy_ratio_ind(data, bal)
    ta = bal.get('total_assets') or 0
    heavy_ok = hr is not None and hr > 0.40
    # 成长期: 近3年营收复合增速>15%
    revs = [x.get('revenue') for x in ia[:3] if x.get('revenue')]
    growth_ok = len(revs) >= 3 and revs[0] > 0 and revs[-1] > 0 and (revs[0]/revs[-1])**(1/(len(revs)-1)) - 1 > 0.15
    exempt_c = (heavy_ok or growth_ok) and b
    if exempt_c:
        c = True   # "资本开支 18.5 亿 > 净利, 但快速增长期例外" (盐津实证); 前提净现比达标
    d = len(cf) >= 3 and all((x.get('fcf') or 0) > 0 for x in cf[:3])
    exempt_note = ' [豁免c:重资产]' if exempt_c and heavy_ok else (' [豁免c:成长期]' if exempt_c else '')
    if a and b and c and d: return 1, f'四指标全过{exempt_note}'
    failed = sum([not a, not b, not c, not d])
    if failed >= 2: return -1, f'≥2项不满足 (a={a},b={b},c={c},d={d}){exempt_note}'
    return 0, f'部分通过 (a={a},b={b},c={c},d={d}){exempt_note}'

def s5_dividend(data):
    divs = data.get('dividends', [])
    if not divs: return None, '分红数据缺失'
    def _year(d):
        ed = str(d.get('end_date') or '')
        return ed[:4] if len(ed) >= 4 else ''
    # 按年聚合最大 cash_div
    year_max = {}
    for d in divs:
        y = _year(d)
        if y and d.get('cash_div'):
            year_max[y] = max(year_max.get(y, 0), d['cash_div'])
    paid_years = {y for y, v in year_max.items() if v > 0}
    ia = data.get('income_annual', [])
    ia_by_year = {str(x.get('end_date', ''))[:4]: x for x in ia}
    # P0-6 周期豁免: 周期底部亏损年不分红视为通过 (牧原 2023 实证)
    if is_cyclical(data):
        ikey = industry_key(data)
        pos = CYCLE_POSITION.get(ikey)
        if pos and pos[0] == '底部':
            for y, x in ia_by_year.items():
                if x.get('net_profit') is not None and x['net_profit'] < 0 and y not in paid_years:
                    paid_years.add(y)   # 亏损年不分红 = 财务纪律, 视为通过
    if not paid_years: return -1, '所有记录均无分红'
    all_years = sorted(paid_years | {_year(d) for d in divs if _year(d)}, reverse=True)
    recent_years = sorted(paid_years, reverse=True)[:3]
    if len(recent_years) < 3:
        return 0, f'近3年有分红年数 {len(recent_years)}/3'
    total_share = data.get('total_share') or (data.get('balance_latest') or {}).get('total_share')
    # 逐年支付率 (P0-6 骤降检测用)
    payouts = {}
    if total_share and total_share > 0:
        for y in recent_years:
            ia_y = ia_by_year.get(y)
            if ia_y and ia_y.get('net_profit') and ia_y['net_profit'] > 0:
                payouts[y] = year_max[y] * total_share / ia_y['net_profit']
    # P0-6 分红骤降: 当年支付率同比降幅>50% 且当年<30%; 连续两年 -> -1 (西矿 81.3%->3.9% 实证)
    abnormal = any(v > 1.0 for v in payouts.values())
    if len(payouts) >= 2:
        ps = [payouts[y] for y in sorted(payouts, reverse=True)]
        drops = [ps[i] < ps[i+1] * 0.5 and ps[i] < 0.30 for i in range(len(ps)-1)]
        if any(drops):
            if len(drops) >= 2 and drops[0] and drops[1]:
                return -1, f'连续两年分红骤降 (P0-6): {payouts}'
            return 0, f'分红骤降 (P0-6): {payouts}'
    if abnormal: return 0, f'近3年每年分红但存在支付率>100%'
    return 1, f'近3年每年正常分红 (分红年数 {recent_years}, 支付率 {payouts})'

def s6_industry(data):
    ikey = industry_key(data)
    # P0-3 周期股分支: 不判行业增速(全行业亏损恰是反转前夜), 与 s1 周期位置联动
    if is_cyclical(data):
        pos = CYCLE_POSITION.get(ikey)
        if pos:
            state, source = pos
            if state == '底部': return 1, f'周期股 {ikey} 底部 (P0-3 联动 s1): 全行业亏损是反转前夜 ({source})'
            if state == '顶部': return -1, f'周期股 {ikey} 顶部 (P0-3 联动 s1) ({source})'
            return 0, f'周期股 {ikey} 位置 {state} (P0-3 联动 s1)'
        return 0, f'周期股 {ikey} 位置未配置'
    if ikey not in INDUSTRY_META: return None, f'行业 {ikey} 未配置'
    meta = INDUSTRY_META[ikey]
    if meta['market_size_yi'] < 1000: return 0, f'行业规模<千亿'
    g = meta['g_by_year']
    sig = [y for y, r in g.items() if r is not None and (r >= GDP_GROWTH.get(y, 5) + 2 or r >= 1.5 * GDP_GROWTH.get(y, 5))]
    if len(sig) >= max(1, len(g)//2): return 1, f'行业 {ikey} 近3年每年显著高于GDP ({sig})'
    return 0, f'行业 {ikey} 增速未显著高于GDP'

def s7_leader(data, peer_cache, market_snap=None, stock_ind_map=None, market_fina=None):
    ikey = industry_key(data)
    is_fin = ikey in FINANCIAL_INDS
    my = (data.get('fina_annual') or [{}])[0] if data.get('fina_annual') else None
    # 金融股无"毛利率"科目: 用 {净利率, ROE} 两项, ≥1 项行业 TOP3% (口径折中)
    metrics = ['net_margin', 'roe'] if is_fin else ['gross_margin', 'net_margin', 'roe']
    if not my or any(my.get(m) is None for m in metrics):
        return None, '财务指标缺失'
    ranks, peer_n = {}, None
    for m in metrics:
        r, n = None, 0
        if market_fina and stock_ind_map and data.get('industry_l1'):
            r, n = _peer_rank_metric(data['ts_code'], data['industry_l1'],
                                     stock_ind_map, market_fina, m)
        if r is None:
            # 样本内回退
            peers = [(t, ((d.get('fina_annual') or [{}])[0].get(m) or 0)) for t, d in peer_cache.items()
                     if industry_key(d) == ikey and d.get('fina_annual')
                     and (d.get('fina_annual') or [{}])[0].get(m) is not None]
            peers.sort(key=lambda x: x[1], reverse=True)
            for rr, (t, v) in enumerate(peers, 1):
                if t == data['ts_code']:
                    r = rr
                    break
            n = len(peers)
        ranks[m] = r
        if peer_n is None:
            peer_n = n
    # P1-6 用户钦定 (2026-09-10): 行业 <10 家 -> 排名线固定 top3; >=10 家 -> max(3, ceil(n*3%))
    cut_line = 3 if (peer_n and peer_n < 10) else max(3, int(__import__('math').ceil((peer_n or 0) * 0.03)))
    in_top = sum(1 for m in metrics if ranks[m] is not None and ranks[m] <= cut_line)
    need = 1 if is_fin else 2
    labels = '/'.join(f'{k}={v}' for k, v in ranks.items())
    if in_top >= need: return 1, f'{(("两项" if is_fin else "三指标"))}≥{need}个进入TOP{cut_line} (行业{peer_n}家, P1-6) ({labels})'
    return 0, f'TOP{cut_line}达标数={in_top}/{need} (行业{peer_n}家) ({labels})'

def s8_equity_change(data):
    """P1-1/P1-3 (2026-09-10): 拆三子项, 增持与减持不对冲、分别计分。
    返回 (a_score, b_score, c_score, evidence)。
    a: 回购两档(注销≥1%市值不看价格 / 激励看价格) 或 大额增持 -> +1/0
    b: 战略投资者/控制权变更 -> 数据不可得记 0 (需公告 NLP)
    c: 非风投机构大额减持 -> -1/0 (与 a 不抵消, P1-3 迈瑞实证)
    """
    n = data.get('trades_3m', 0)
    if n < 0:
        return None, None, None, f'增减持接口异常: {data.get("trades_error","")}'
    in_n = data.get('trades_in_n') or 0
    de_n = data.get('trades_de_n') or 0
    a_score, a_evs = 0, []
    # P1-1 回购两档: 注销且≥1%市值 -> +1 不看价格 (宁德实证)
    reps = data.get('repurchase') or []
    mv = data.get('total_mv_yi')
    if reps and mv and mv > 0:
        amt_total = sum(r.get('amount') or 0 for r in reps)
        if amt_total >= mv * 1e8 * 0.01:
            a_score = 1
            a_evs.append(f'回购≥1%市值({amt_total/1e8:.1f}亿, 注销档不看价格)')
    if a_score != 1 and in_n > 0:
        a_score = 1
        a_evs.append(f'近12月增持 {in_n} 条 (战略入股需人工确认)')
    # P1-1 激励档: 现价 < 回购均价*0.9 -> +1 (回购均价 = 实施记录 amount/vol, 2026-09-10 补)
    if a_score == 0 and reps:
        done = [r for r in reps if ('实施' in str(r.get('proc','')) or '完成' in str(r.get('proc',''))) and r.get('vol') and r.get('amount')]
        if done and data.get('close'):
            avg_px = done[0]['amount'] / done[0]['vol']
            if data['close'] < avg_px * 0.9:
                a_score = 1
                a_evs.append(f'激励回购档: 现价{data["close"]:.1f}<均价{avg_px:.1f}九折')
            else:
                a_evs.append(f'回购<1%市值且现价未折破均价九折({data["close"]:.1f} vs {avg_px:.1f}) 记0')
        else:
            a_evs.append('有回购记录但<1%市值或均价数据不足 记0')
    if not a_evs:
        a_evs.append('无回购/无增持')
    # b: 战略投资者/控制权变更 (无法自动化)
    b_score, b_ev = 0, '战略投资者/控制权变更需公告 NLP, 数据不可得记 0 (B8)'
    # c: 减持
    c_score, c_ev = 0, ''
    if de_n > 0:
        c_score = -1
        c_ev = f'近12月减持 {de_n} 条 (P1-3 与增持不抵消, 财务性减持需人工复核主体)'
    else:
        c_ev = '无减持'
    evidence = f'a(+1回购增持): {" / ".join(a_evs)} | b(战投): {b_ev} | c(减持): {c_ev}'
    return a_score, b_score, c_score, evidence

def s9_per_capita(data):
    """P1-8 (2026-09-10): 人均持股市值口径(用户钦定确认) + A+H 股本扣 H 股 + 散户化陷阱。
    返回 (score, evidence, trap_flag): trap=True 时负分项计数×2。
    """
    bal = data.get('balance_latest')
    if not bal or bal.get('total_share') is None: return None, '总股本缺失', False
    if not data.get('top10_holders'): return None, 'stk_top10_holders 未接入', False
    # 用 hold_amount (持股数/股) 求和, 比 hold_ratio 累加更稳 (避免跨期/重复计算)
    top10_amount = sum((h.get('hold_amount') or 0) for h in data['top10_holders'])
    top10_ratio = sum((h.get('hold_ratio') or 0) for h in data['top10_holders'])
    total_share = bal['total_share']
    # A+H 修正 (P1-8): 总股本含 H 股 -> 扣掉, 避免把 H 股流通盘算进 A 股人均高估
    h_share = data.get('h_float_share') or 0.0
    if h_share > 0:
        total_share = total_share - h_share
    # 优先用股本差; 若异常(>总股本)回退用比例
    if top10_amount > 0 and top10_amount < total_share:
        tradable_share = total_share - top10_amount
    else:
        tradable_share = total_share * (1 - top10_ratio / 100.0)
    if tradable_share <= 0: return None, f'流通股本≤0 (top10_ratio={top10_ratio:.1f}%, 异常)', False
    price = data.get('close') or 0
    if price <= 0: return None, '股价缺失', False
    mkt_cap_tradable = tradable_share * price
    holder_num = (data.get('holder_latest') or {}).get('holder_num') or 0
    if holder_num == 0: return None, '股东人数缺失', False
    per_capita = mkt_cap_tradable / holder_num
    per_capita_wan = per_capita / 1e4
    # P1-8 散户化陷阱: 人均<10万 且 近60日涨幅>50% -> 负分项计数×2 (华电辽能 6.4万+163.9% 实证)
    trap = False
    c60 = data.get('close_60d_ago')
    if per_capita_wan < 10 and c60 and c60 > 0 and price / c60 - 1 > 0.50:
        trap = True
    h_note = f', 扣H股{h_share/1e8:.1f}亿股' if h_share > 0 else ''
    trap_note = ' [散户化陷阱: 近60日涨{:.0%}, 负分计2项]'.format(price/c60-1) if trap else ''
    if per_capita_wan >= 30: return 1, f'人均 {per_capita_wan:.1f}万 (流通{tradable_share/1e8:.2f}亿股{h_note}, {holder_num}人)', False
    if per_capita_wan < 10: return -1, f'人均 {per_capita_wan:.1f}万 (流通{tradable_share/1e8:.2f}亿股{h_note}, {holder_num}人){trap_note}', trap
    return 0, f'人均 {per_capita_wan:.1f}万{h_note}', False

def fetch_market_snapshot(pro, trade_date=None):
    """B2/B6 修复: 拉全 A 当日 daily_basic 截面 (分页 5000), 推导 PR/PS 全 A 分布。
    - ROE_TTM ≈ pb / pe_ttm  (E/B = (P/B)/(P/E), 常用近似)
    - PR ≈ pe_ttm / ROE_TTM = pe_ttm² / pb   (稳定盈利股估值)
    - PS = ps_ttm (非稳定盈利股估值)
    返回 {'pr': [..], 'ps': [..], 'rows': {ts_code: {pe_ttm, pb, ps_ttm, roe_ttm, pr}}}, 带当天缓存。
    """
    td = trade_date or TODAY
    cp = cache_path('market_snapshot')
    if cp.exists():
        try:
            df = pd.read_csv(cp)
            if len(df) > 1000 and str(df['trade_date'].iloc[0]).startswith(td[:8]):
                return _snapshot_from_df(df)
        except Exception:
            pass
    frames = []
    for off in (0, 5000, 10000):
        df = safe_call(pro.daily_basic, trade_date=td, limit=5000, offset=off,
                       fields='ts_code,trade_date,close,pe_ttm,ps_ttm,pb,total_mv,circ_mv')
        if len(df) == 0:
            break
        frames.append(df)
        if len(df) < 5000:
            break
    if not frames:
        # 回退 1-14 天
        for back in range(1, 15):
            d_back = (datetime.strptime(td[:8], '%Y%m%d') - timedelta(days=back)).strftime('%Y%m%d')
            frames = []
            for off in (0, 5000, 10000):
                df = safe_call(pro.daily_basic, trade_date=d_back, limit=5000, offset=off,
                               fields='ts_code,trade_date,close,pe_ttm,ps_ttm,pb,total_mv,circ_mv')
                if len(df) == 0:
                    break
                frames.append(df)
                if len(df) < 5000:
                    break
            if frames:
                break
    if not frames:
        return None
    full = pd.concat(frames, ignore_index=True).drop_duplicates('ts_code')
    full.to_csv(cp, index=False)
    return _snapshot_from_df(full)


def _snapshot_from_df(df):
    rows = {}
    prs, pss = [], []
    for _, r in df.iterrows():
        try:
            pe = float(r['pe_ttm']) if pd.notna(r['pe_ttm']) else None
            pb = float(r['pb']) if pd.notna(r['pb']) else None
            ps = float(r['ps_ttm']) if pd.notna(r['ps_ttm']) else None
            roe = (pb / pe) if (pe and pb and pe > 0 and pb > 0) else None
            pr = (pe * pe / pb) if (pe and pb and pe > 0 and pb > 0) else None
            rows[str(r['ts_code'])] = {'pe_ttm': pe, 'pb': pb, 'ps_ttm': ps,
                                       'roe_ttm': roe, 'pr': pr}
            # 口径: pr = pe_ttm/ROE_TTM(小数) = pe_ttm²/pb, 典型值 20~500;
            # 阈值 1000 只剔除 pe 极高/pb 极低的异常值, 保留完整分布基线
            if pr and 0 < pr < 1000:
                prs.append(pr)
            if ps and 0 < ps < 500:
                pss.append(ps)
        except Exception:
            continue
    return {'pr': prs, 'ps': pss, 'rows': rows}


def fetch_stock_industry_map(pro):
    """B2: 全 A 上市股票 -> 行业(申万二级, stock_basic.industry) 映射, 带缓存。"""
    cp = cache_path('stock_industry_map')
    if cp.exists():
        try:
            df = pd.read_csv(cp)
            if len(df) > 1000:
                return dict(zip(df['ts_code'].astype(str), df['industry'].astype(str)))
        except Exception:
            pass
    frames = []
    for off in (0, 5000, 10000):
        df = safe_call(pro.stock_basic, list_status='L', fields='ts_code,industry',
                       limit=5000, offset=off)
        if len(df) == 0:
            break
        frames.append(df)
        if len(df) < 5000:
            break
    if not frames:
        return {}
    full = pd.concat(frames, ignore_index=True).drop_duplicates('ts_code')
    full.to_csv(cp, index=False)
    return dict(zip(full['ts_code'].astype(str), full['industry'].astype(str)))


def fetch_full_market_fina(pro, stock_codes=None, resume=True):
    """B2 完整闭环: 全 A 最新财务指标 (fina_indicator 逐只, 限速 0.15s/只)。
    Tushare fina_indicator 无全市场批量接口 (ts_code 必填), 只能循环。
    resume=True 断点续拉 (缓存已拉部分跳过); 返回 {ts_code: {'roe','gross_margin','net_margin'}}。
    """
    cp = cache_path('market_fina')
    done = {}
    if resume and cp.exists():
        try:
            df = pd.read_csv(cp)
            for _, r in df.iterrows():
                done[str(r['ts_code'])] = {
                    'roe': float(r['roe']) if pd.notna(r['roe']) else None,
                    'gross_margin': float(r['grossprofit_margin']) if pd.notna(r['grossprofit_margin']) else None,
                    'net_margin': float(r['netprofit_margin']) if pd.notna(r['netprofit_margin']) else None,
                }
        except Exception:
            done = {}
    if stock_codes is None:
        stock_codes = list(done.keys())          # 仅返回已有缓存
        return done
    codes = [c for c in stock_codes if c not in done]
    new_rows = []
    t0 = time.time()
    for i, code in enumerate(codes, 1):
        df = safe_call(pro.fina_indicator, ts_code=code, limit=1,
                       fields='ts_code,end_date,roe,grossprofit_margin,netprofit_margin')
        if len(df) > 0:
            r = df.iloc[0]
            new_rows.append({'ts_code': code, 'end_date': r['end_date'],
                             'roe': r['roe'], 'grossprofit_margin': r['grossprofit_margin'],
                             'netprofit_margin': r['netprofit_margin']})
        if i % 200 == 0 or i == len(codes):
            if new_rows:
                pd.DataFrame(new_rows).to_csv(cp, mode='a', header=not cp.exists(), index=False)
                new_rows = []
            if i % 2000 == 0 or i == len(codes):
                print(f'    [fina] {i}/{len(codes)} 已拉取 ({time.time()-t0:.0f}s)', flush=True)
        time.sleep(0.15)
    # 重读合并
    if cp.exists():
        df = pd.read_csv(cp)
        out = {}
        for _, r in df.iterrows():
            out[str(r['ts_code'])] = {
                'roe': float(r['roe']) if pd.notna(r['roe']) else None,
                'gross_margin': float(r['grossprofit_margin']) if pd.notna(r['grossprofit_margin']) else None,
                'net_margin': float(r['netprofit_margin']) if pd.notna(r['netprofit_margin']) else None,
            }
        return out
    return done


def _peer_rank_metric(ts_code, ind_l1, stock_ind_map, market_fina, metric):
    """B2: 全 A 同行业(申万二级) metric 排名。返回 (rank, peer_count); 无数据返回 (None, 0)。"""
    if not market_fina:
        return None, 0
    peers = [(c, market_fina[c].get(metric)) for c, ind in stock_ind_map.items()
             if ind == ind_l1 and c in market_fina and market_fina[c].get(metric) is not None]
    if len(peers) < 3:
        return None, len(peers)
    peers.sort(key=lambda x: x[1], reverse=True)
    for r, (c, v) in enumerate(peers, 1):
        if c == ts_code:
            return r, len(peers)
    return None, len(peers)


def s10_valuation(data, market_snap=None, peer_cache=None, stock_ind_map=None):
    """B6 修复: 全 A 估值分位。稳定盈利 -> PR 分布分位; 非稳定 -> PS 分布分位; 周期股 -> 同行 TTM PE 分位;
    金融股 -> 股息率行业分位 (P1-5)。质地溢价: ROE 全 A 前 10% 门槛放宽 10pct (P1-5)。
    """
    if market_snap is None:
        return None, '全 A 估值截面未获取'
    code = data['ts_code']
    rows = market_snap.get('rows', {})
    my = rows.get(code)
    if my is None:
        return None, '该股不在全 A 截面中'
    ikey = industry_key(data)
    # 周期股分支 (2026-09-10 与门控 A 方案同步): 同行 TTM PE 分位
    if is_cyclical(data):
        pe = data.get('pe_ttm')
        if pe is None or pe <= 0:
            return None, '周期股亏损(PE无意义)记 N/A'
        peers = [(c, rows[c].get('pe_ttm')) for c, r in rows.items()
                 if r.get('pe_ttm') is not None and r['pe_ttm'] > 0
                 and stock_ind_map and stock_ind_map.get(c) == data.get('industry_l1')]
        if len(peers) < 3:
            return None, '同行 PE 样本不足'
        vals = [v for _, v in peers]
        pct = float((pd.Series(vals) < pe).mean() + 0.5 * (pd.Series(vals) == pe).mean())
        if pct <= 0.30: return 1, f'周期股同行PE分位{pct:.0%} (PE={pe:.1f}, n={len(vals)})'
        if pct >= 0.70: return -1, f'周期股同行PE分位{pct:.0%} (PE={pe:.1f}, n={len(vals)})'
        return 0, f'周期股同行PE分位{pct:.0%} (PE={pe:.1f}, n={len(vals)})'
    # 金融股分支 (P1-5): 股息率行业分位 (样本内近似)
    if ikey in FINANCIAL_INDS and peer_cache:
        def _div_yield(d):
            divs = d.get('dividends') or []
            best = 0
            for x in divs:
                if x.get('cash_div'):
                    best = max(best, x['cash_div'])
            close = d.get('close') or 0
            return best / close if close > 0 else None
        my_y = _div_yield(data)
        peer_y = [(t, _div_yield(p)) for t, p in peer_cache.items()
                  if industry_key(p) == ikey and p.get('close') and (p.get('dividends'))]
        peer_y = [(t, v) for t, v in peer_y if v is not None]
        if my_y is None or len(peer_y) < 3:
            return None, '股息率数据不足'
        vals = [v for _, v in peer_y]
        pct = float((pd.Series(vals) < my_y).mean() + 0.5 * (pd.Series(vals) == my_y).mean())
        if pct >= 0.70: return 1, f'金融股股息率{my_y:.1%}处行业头部30% (分位{pct:.0%}, 越高越便宜, P1-5)'
        if pct <= 0.30: return -1, f'金融股股息率{my_y:.1%}处行业尾部30% (分位{pct:.0%}, P1-5)'
        return 0, f'金融股股息率{my_y:.1%}行业中性 (分位{pct:.0%}, P1-5)'
    stable = is_stable_profit(data)
    if stable:
        my_val = my.get('pr')
        dist = market_snap.get('pr') or []
        label = 'PR'
    else:
        my_val = my.get('ps_ttm')
        dist = market_snap.get('ps') or []
        label = 'PS'
    if my_val is None or len(dist) < 100:
        return None, f'全 A {label} 分布不足 (n={len(dist)})'
    s = pd.Series(dist)
    pct = float((s < my_val).mean() + 0.5 * (s == my_val).mean())
    # P1-5 质地溢价: ROE 全 A 前 10% -> 门槛放宽 10pct (-1线70%->80%, +1线30%->40%)
    hi_neg, hi_pos = 0.70, 0.30
    roe_mine = my.get('roe_ttm')
    if roe_mine is not None:
        roes = [r.get('roe_ttm') for r in rows.values() if r.get('roe_ttm') is not None]
        if len(roes) >= 100:
            roe_pct = float((pd.Series(roes) < roe_mine).mean())
            if roe_pct >= 0.90:
                hi_neg, hi_pos = 0.80, 0.40
    q_note = ' [质地溢价放宽10pct]' if hi_neg == 0.80 else ''
    if pct <= hi_pos:
        return 1, f'全A {label}={my_val:.2f} 处于便宜区间 (分位{pct:.0%}, n={len(dist)}){q_note}'
    if pct >= hi_neg:
        return -1, f'全A {label}={my_val:.2f} 处于昂贵区间 (分位{pct:.0%}, n={len(dist)}){q_note}'
    return 0, f'全A {label}={my_val:.2f} 估值中性 (分位{pct:.0%}, n={len(dist)}){q_note}'

def decide(pos, neg, total):
    if neg >= 5: return '剔除(负分>=5)'
    if pos >= 7: return '直通候选池'
    return 'TOP15候选'

def main():
    print('=== 快刀斩股选股策略 Skill 端到端测试 v2 (P0/P1 批次同步版 2026-09-10) ===')
    pro = get_pro()
    print(f'\n[1/5] 拉全 A 截面 (B2/B6: 估值分布 + 行业映射 + 财务指标)...')
    market_snap = fetch_market_snapshot(pro)
    stock_ind_map = fetch_stock_industry_map(pro)
    market_fina = fetch_full_market_fina(pro, stock_codes=list(stock_ind_map.keys()))
    if market_snap:
        print(f'  全A截面: {len(market_snap["rows"])} 只 | PR分布 n={len(market_snap["pr"])} | PS分布 n={len(market_snap["ps"])} | 行业映射 {len(stock_ind_map)} 只 | 财务指标 {len(market_fina)} 只')
    else:
        print('  [warn] 全 A 截面获取失败, s10/全A排名退回 N/A')
    print(f'\n[2/5] 收集 {len(TEST_UNIVERSE)} 只样本...')
    all_data = {}
    for code, name, expected, hint in TEST_UNIVERSE:
        try:
            d = collect_one(pro, code, name, expected)
            d['hint'] = hint
            all_data[code] = d
            ok = 'OK' if d.get('fina_annual') and d.get('income_annual') else 'PART'
            print(f'  {code} {name}: {ok} 行业={d.get("industry_l1","?")} PE={d.get("pe_ttm","-")} ROE[0]={d.get("fina_annual",[{}])[0].get("roe","-")}')
        except Exception as e:
            print(f'  {code} {name}: ERROR {e}')
            all_data[code] = {'ts_code': code, 'name': name, 'expected': expected, 'error': str(e)}
    print(f'\n[3/5] 门控 (排雷 -> 价格, 2026-09-10 门控一二已删除)...')
    gate_results = {}
    for code, d in all_data.items():
        if 'error' in d:
            gate_results[code] = {'error': True}
            continue
        gm = gate_mine_sweep(d, all_data)
        if gm.get('pass') is False:
            gp = {'pass': False, 'method': 'SKIP', 'verdict': '排雷淘汰', 'detail': '排雷命中跳过价格门控'}
        else:
            gp = gate_price(d, pro)
        gate_results[code] = {'mine_sweep': gm, 'price': gp}
        passed = 1 if (gm.get('pass') is True and gp.get('pass') is True) else 0
        print(f'  {code} {d["name"]}: pass={passed}/2 (排雷:{gm.get("detail","")[:24]}... 价格:{gp.get("verdict", gp.get("na"))})')
    print(f'\n[4/5] 十项打分...')
    scorecards = []
    for code, d in all_data.items():
        if 'error' in d or any(gate_results[code].get(k, {}).get('pass') is False for k in ('mine_sweep', 'price')):
            scorecards.append({'code': code, 'name': d.get('name',''), 'gating': gate_results[code], 'scores': {}, 'summary': {'decision': '门控淘汰'}})
            continue
        sc = {
            's1_revenue_profit': s1_revenue_profit(d),
            's2_roe': s2_roe(d),
            's3_balance': s3_balance(d),
            's4_cashflow': s4_cashflow(d),
            's5_dividend': s5_dividend(d),
            's6_industry': s6_industry(d),
            's7_leader': s7_leader(d, all_data, market_snap, stock_ind_map, market_fina),
            's9_per_capita': s9_per_capita(d),
            's10_valuation': s10_valuation(d, market_snap, all_data, stock_ind_map),
        }
        # s8 三子项 (P1-1/P1-3): a 回购增持 / b 战略 / c 减持, 分别计分
        s8a, s8b, s8c, s8ev = s8_equity_change(d)
        # 正负项计数: s9 散户化陷阱负分计 2 (P1-8)
        pos = sum(1 for v in sc.values() if v and v[0] == 1)
        neg = sum(1 for v in sc.values() if v and v[0] == -1)
        if s8a == 1: pos += 1
        if s8b == 1: pos += 1
        if s8c == -1: neg += 1
        s9 = sc.get('s9_per_capita')
        s9_trap = (len(s9) > 2 and s9[2]) if s9 else False
        if s9_trap: neg += 1   # 散户化陷阱负分计 2 项
        na = [k for k, v in sc.items() if v is None]
        if s8a is None and s8b is None and s8c is None:
            na.append('s8_equity_change')
        total = sum(v[0] for v in sc.values() if v and v[0] is not None)
        total += sum(x for x in (s8a, s8b, s8c) if x is not None)
        decision = decide(pos, neg, total)
        scorecards.append({
            'code': code, 'name': d.get('name',''),
            'industry_l1': d.get('industry_l1',''),
            'industry_meta_key': industry_key(d),
            'is_cyclical': is_cyclical(d),
            'is_stable_profit': is_stable_profit(d),
            'gating': gate_results[code],
            'scores': {k: {'score': v[0] if v else None, 'evidence': v[1] if v else None} for k, v in sc.items()},
            'scores_s8': {'a': s8a, 'b': s8b, 'c': s8c, 'evidence': s8ev},
            's9_trap': s9_trap,
            'summary': {'positive': pos, 'negative': neg, 'na_items': na, 'total': total, 'decision': decision}
        })
        print(f'  {code} {d["name"]}: total={total} +{pos} -{neg} NA={len(na)} → {decision}')
    print(f'\n[5/5] 写报告...')
    fields_needed = ['industry_l1','total_mv_yi','pe_ttm','fina_annual','income_annual','balance_latest','cf_annual','dividends','holder_latest','trades_3m']
    coverage = {f: sum(1 for d in all_data.values() if d.get(f) not in (None, [], 0)) for f in fields_needed}
    direct = [s for s in scorecards if s['summary'].get('decision') == '直通候选池']
    passed = [s for s in scorecards if s['summary'].get('decision') == 'TOP15候选']
    passed.sort(key=lambda x: x['summary']['total'], reverse=True)
    out = {
        'run_ts': RUN_TS, 'sample_size': len(TEST_UNIVERSE),
        'data_coverage': coverage,
        'market_snapshot': {
            'n_stocks': len(market_snap['rows']) if market_snap else 0,
            'pr_n': len(market_snap['pr']) if market_snap else 0,
            'ps_n': len(market_snap['ps']) if market_snap else 0,
        },
        'gating_results': gate_results,
        'scorecards': scorecards,
        'final_pool': {
            'direct': [{'code': s['code'], 'name': s['name'], 'total': s['summary']['total']} for s in direct],
            'top15': [{'code': s['code'], 'name': s['name'], 'total': s['summary']['total']} for s in passed[:15]]
        },
        'data_gaps': [
            'S8 b 子项 战略投资者/控制权变更: 无 Tushare 标准接口, 需公告 NLP 解析 (B8)',
            'S7 全 A 排名: 依赖逐股拉 fina_indicator 构建基线 (FULL 模式路径), 样本内回退仅作展示',
            '周期股定位: CYCLE_POSITION 为人工判断集中管理 (底部/上行中段/顶部三态), 客观化需商品价格数据 (B4/A方案)',
            '行业 metadata: 31 申万一级+20 细分为人工估算, 需定期复核 (B7 已外置 JSON 可维护)',
            'G3 PS 路径: 用 ps_ttm 近似, 未扣净现金 (skill 原式为调整后市值÷营收) (B5 已实装, 口径注明)',
            'P1-4 金融股 s3 三指标(不良率/拨备覆盖率/资本充足率): Tushare 无标准接口, 记 N/A',
            'P0-2 R2 财务洗澡: 同行净利中位数为样本内近似, 全市场执行需行业×年度中位数缓存',
        ]
    }
    out_path = OUT_DIR / f'scorecard_{RUN_TS}.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f'\n✅ {out_path}')
    print(f'数据覆盖率: {coverage}')
    print(f'直通: {len(direct)} | TOP15候选: {len(passed)} | 淘汰: {sum(1 for s in scorecards if s["summary"].get("decision")=="门控淘汰")}')

if __name__ == '__main__':
    main()
