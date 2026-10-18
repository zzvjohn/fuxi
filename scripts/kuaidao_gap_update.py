# -*- coding: utf-8 -*-
"""快刀缺口表日常增量更新 (P-20260911-kd, 2026-09-11)

目的: 把 data/raw/kuaidao_gaps/ 的 8 张表纳入流水线常规维护, 避免一次性采集后
      数据陈旧 (2026-09-09 一次性采集完成, 此前无任何增量机制)。

策略 (省 Tushare 额度):
  - 三表 (income / cashflow / balancesheet): **逐股**调用（Tushare 该组接口强制
    要求 ts_code, period/ann_date 批量均报"必填参数, ts_code"）, 故只在**财报季**
    (1/4/8/10 月) 执行; 以全局 last_ann_date 为起点做公告日区间增量。
  - 事件/状态表 (dividend / holdertrade / holdernumber / repurchase / top10):
    接口支持日期区间批量 → **每次运行都做**, 调用数少 (十几~几十次)。
  - 历史行情 (daily_prices_2018_2020 / daily_basic_2018_2020): 静态历史, 不更新。

用法:
  python scripts/kuaidao_gap_update.py            # auto: 财报季走全量, 否则只事件表
  python scripts/kuaidao_gap_update.py --force-fin # 强制三表增量
  python scripts/kuaidao_gap_update.py --event-only

断点/水位: data/raw/kuaidao_gaps/_update_state.json
"""
import sys, os, io, time, json, datetime
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', write_through=True)
sys.path.insert(0, 'e:/quant')
from credentials import get_tushare_api

OUT = 'e:/quant/data/raw/kuaidao_gaps'
STATE = os.path.join(OUT, '_update_state.json')
os.makedirs(OUT, exist_ok=True)
pro = get_tushare_api()

CF_FIELDS = ('ts_code,ann_date,f_ann_date,end_date,report_type,'
             'n_cashflow_act,c_pay_acq_const_fiolta,c_fr_sale_sg')
IN_FIELDS = ('ts_code,ann_date,f_ann_date,end_date,report_type,revenue,operate_profit,'
             'total_profit,income_tax,n_income,n_income_attr_p')
BS_FIELDS = ('ts_code,ann_date,f_ann_date,end_date,report_type,total_assets,money_cap,'
             'trad_asset,notes_receiv,accounts_receiv,goodwill,st_borr,lt_borr,bond_payable,'
             'ncl_1y,adv_receipts,contract_liab')

FIN_TABLES = [('income.csv', 'income', IN_FIELDS, ['ts_code', 'end_date', 'ann_date']),
              ('cashflow.csv', 'cashflow', CF_FIELDS, ['ts_code', 'end_date', 'ann_date']),
              ('balancesheet.csv', 'balancesheet', BS_FIELDS, ['ts_code', 'end_date', 'ann_date'])]


def load_state():
    return json.load(open(STATE, encoding='utf-8')) if os.path.exists(STATE) else {}


def save_state(st):
    json.dump(st, open(STATE, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


def _retry(fn, desc, tries=4):
    for i in range(tries):
        try:
            df = fn()
            return None if df is None or len(df) == 0 else df
        except Exception as e:
            msg = str(e)[:120]
            if any(k in msg for k in ('每分钟', '抱歉', '2000次', '访问频率', '频率')):
                w = 15 + 10 * i
                print('  [限速] %s 第%d次 -> sleep %ds' % (desc, i + 1, w)); time.sleep(w)
            else:
                print('  [重试] %s 第%d次: %s' % (desc, i + 1, msg)); time.sleep(2.0)
    return None


_UNI = None


def universe():
    global _UNI
    if _UNI is None:
        c = pd.read_csv('e:/quant/data/raw/daily_prices.csv', usecols=['ts_code'],
                        dtype=str)['ts_code'].dropna().unique().tolist()
        # 🔴 因子数据源必须覆盖**全市场**, 不得沿用 CF 策略股票池的排除口径。
        #    2026-09-11 实证: 三表曾因排除 688/689/8/4 而缺 618 只科创板, 使全部
        #    kd_* 因子在这 11% 股票上恒为 NaN。已用 _collect_kuaidao_star.py 补齐,
        #    此处同步放开, 否则增量又会把科创板漏掉。
        _UNI = sorted(set(x for x in c if x.split('.')[-1] in ('SH', 'SZ', 'BJ')))
    return _UNI


def _detect_newline(path):
    """探测既有文件行尾 (P2-2b, 2026-09-28): 不得把 CRLF 表静默改写成 LF。

    kuaidao_gaps 下主表实测均为 CRLF; pd.to_csv 默认 LF 会导致
    「增量一次 = 行尾全量改写」⇒ 字节级对账与行尾铁律双违规。
    """
    try:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                head = f.read(65536)
            if b'\r\n' in head:
                return '\r\n'
    except Exception:
        pass
    return '\n'


def _append(path, frames):
    if frames:
        df = pd.concat(frames, ignore_index=True)
        header = not os.path.exists(path)
        nl = _detect_newline(path)
        with io.open(path, 'a', encoding='utf-8', newline='') as f:
            df.to_csv(f, header=header, index=False, lineterminator=nl)


def _dedupe(path, keys, report1=False):
    if not os.path.exists(path):
        return 0
    d = pd.read_csv(path, dtype=str, keep_default_na=False)
    if len(d) == 0:
        return 0
    before = len(d)
    if report1 and 'report_type' in d.columns:
        d['_r'] = np.where(d['report_type'] == '1', 0, 1)
        d = d.sort_values(keys + ['_r']).drop_duplicates(subset=keys, keep='first').drop(columns=['_r'])
    else:
        d = d.drop_duplicates(subset=keys, keep='first')
    nl = _detect_newline(path)
    with io.open(path, 'w', encoding='utf-8', newline='') as f:
        d.to_csv(f, index=False, lineterminator=nl)
    return before - len(d)


def _month_range(mo):
    y, m = int(mo[:4]), int(mo[4:6])
    nxt = datetime.date(y + 1, 1, 1) if m == 12 else datetime.date(y, m + 1, 1)
    return mo + '01', (nxt - datetime.timedelta(days=1)).strftime('%Y%m%d')


def _recent_months(n=3):
    out, d = [], datetime.date.today()
    y, m = d.year, d.month
    for _ in range(n):
        out.append('%04d%02d' % (y, m))
        m -= 1
        if m == 0:
            y -= 1; m = 12
    return sorted(out)


# ── 三表: 逐股公告日区间增量 (仅财报季) ─────────────────────────────────────
def update_financials(st, start_date):
    print('=' * 62)
    print('[财报三表] 逐股公告日区间增量  start=%s end=%s' % (start_date, datetime.date.today().strftime('%Y%m%d')))
    print('=' * 62)
    today = datetime.date.today().strftime('%Y%m%d')
    total_add = 0
    for fname, api, fields, keys in FIN_TABLES:
        path = os.path.join(OUT, fname)
        codes = universe()
        done = set(st.get('fin_done_' + api, []))
        todo = [c for c in codes if c not in done]
        if not todo:
            print('%s: 本轮已完成' % fname)
            continue
        buf, n, t0 = [], 0, time.time()
        for i, c in enumerate(todo):
            df = _retry(lambda c=c: getattr(pro, api)(ts_code=c, start_date=start_date,
                                                      end_date=today, fields=fields),
                        '%s %s' % (api, c))
            if df is not None:
                buf.append(df); n += len(df)
            done.add(c)
            if len(buf) >= 200:
                _append(path, buf); buf = []
                st['fin_done_' + api] = sorted(done); save_state(st)
            if (i + 1) % 500 == 0 or i == len(todo) - 1:
                el = time.time() - t0
                print('  %s %d/%d 新增 %d 行 (%.0fs)' % (fname, i + 1, len(todo), n, el))
            time.sleep(0.12)
        _append(path, buf)
        st['fin_done_' + api] = []            # 本轮完成 → 清空, 下次重新按新水位
        save_state(st)
        dup = _dedupe(path, keys, report1=True)
        print('  %s 完成: 新增 %d 行, 去重 %d 行' % (fname, n, dup))
        total_add += n
    st['fin_last_ann'] = today
    save_state(st)
    return total_add


# ── 事件/状态表: 日期区间批量 (每次都跑) ────────────────────────────────────
def update_events(st):
    print('=' * 62)
    print('[事件/状态表] 月区间批量增量')
    print('=' * 62)
    months = _recent_months(3)
    done = set(st.get('ev_done_months', []))
    todo = [m for m in months if m not in done]

    # 1) 股东户数
    if todo:
        buf, n = [], 0
        for mo in todo:
            s, e = _month_range(mo)
            df = _retry(lambda s=s, e=e: pro.stk_holdernumber(
                start_date=s, end_date=e, fields='ts_code,ann_date,end_date,holder_num'), 'holdernum %s' % mo)
            if df is not None:
                buf.append(df); n += len(df)
            done.add(mo); st['ev_done_months'] = sorted(done); save_state(st)
            time.sleep(0.3)
        _append(os.path.join(OUT, 'stk_holdernumber.csv'), buf)
        print('  stk_holdernumber 新增 %d 行' % n)
        _dedupe(os.path.join(OUT, 'stk_holdernumber.csv'), ['ts_code', 'ann_date', 'end_date'])

        # 2) 大股东增减持
        buf, n = [], 0
        for mo in todo:
            s, e = _month_range(mo)
            df = _retry(lambda s=s, e=e: pro.stk_holdertrade(
                start_date=s, end_date=e,
                fields='ts_code,ann_date,holder_name,holder_type,in_de,change_vol,change_ratio,'
                       'after_share,after_ratio'), 'holdertrade %s' % mo)
            if df is not None:
                buf.append(df); n += len(df)
            time.sleep(0.3)
        _append(os.path.join(OUT, 'stk_holdertrade.csv'), buf)
        print('  stk_holdertrade 新增 %d 行' % n)
        _dedupe(os.path.join(OUT, 'stk_holdertrade.csv'), ['ts_code', 'ann_date', 'holder_name', 'in_de'])

    # 3) 回购 (月区间)
    buf, n = [], 0
    for mo in todo:
        s, e = _month_range(mo)
        df = _retry(lambda s=s, e=e: pro.repurchase(start_date=s, end_date=e), 'repurchase %s' % mo)
        if df is not None:
            buf.append(df); n += len(df)
        time.sleep(0.35)
    _append(os.path.join(OUT, 'repurchase.csv'), buf)
    print('  repurchase 新增 %d 行' % n)
    _dedupe(os.path.join(OUT, 'repurchase.csv'), ['ts_code', 'ann_date', 'end_date', 'proc'])

    # 4) 分红 (季末分期全周期; P2-2b 2026-09-28 修复)
    #    旧版两缺陷 (2026-09-28 审计实证):
    #      ① 只采 end_date=YYYY1231 ⇒ 中期(0331/0630/0930)分红永不入库
    #         (旧主表 end_date 100% 为 1231, 仅 2018~2025; 全周期补采后 0630 有 11 万行);
    #      ② 三键去重 ⇒ 同 (ts_code,end_date,ann_date) 的「预案/股东大会通过/实施」
    #         多状态行被压成一行 (正确键为四键 +div_proc)。
    #    新版: 最近 2 年 × 季末 4 期, end_date 分期 + offset 分页 (2000/页);
    #    四键去重保留多状态行; 不做 done 跳过 (分红状态随公告演进, 需重复刷新)。
    yrs = sorted({datetime.date.today().year, datetime.date.today().year - 1})
    periods = ['%d%s' % (y, md) for y in yrs for md in ('0331', '0630', '0930', '1231')]
    buf, n = [], 0
    div_rows = {}
    for p in periods:
        off, pn = 0, 0
        while True:
            df = _retry(lambda p=p, off=off: pro.dividend(
                end_date=p, offset=off, limit=2000,
                fields='ts_code,end_date,ann_date,div_proc,cash_div,'
                       'cash_div_tax,ex_date,record_date'),
                'dividend %s off=%d' % (p, off))
            if df is None:          # 空页 (正常结束) 或失败 (重试耗尽)
                break
            buf.append(df); n += len(df); pn += len(df)
            if len(df) < 2000:
                break
            off += 2000
            time.sleep(0.35)
        div_rows[p] = pn
        time.sleep(0.3)
    _append(os.path.join(OUT, 'dividend.csv'), buf)
    print('  dividend 新增 %d 行 | 分期读数 %s' % (n, div_rows))
    st['div_period_rows'] = div_rows    # 审计记账: 行数骤降可事后发现
    # 全行去重 (keys=None): 同 (ts_code,end_date,ann_date,div_proc) 仍可能有
    # 合法多行 (如 cash_div=0.0 vs 空 的修订轨迹) —— P2-2b 并入实测 358 组;
    # 用四键/三键会误删, 故保持与并入同口径的全行去重。
    _dedupe(os.path.join(OUT, 'dividend.csv'), None)

    st['ev_last_run'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    save_state(st)


if __name__ == '__main__':
    st = load_state()
    t0 = time.time()
    today = datetime.date.today()
    is_fin_season = today.month in (1, 4, 8, 10)
    force_fin = '--force-fin' in sys.argv
    event_only = '--event-only' in sys.argv

    print('快刀缺口表增量更新 | 今日 %s | 财报季=%s' % (today, is_fin_season))
    if not event_only and (is_fin_season or force_fin):
        start = st.get('fin_last_ann') or '20260901'
        update_financials(st, start)
    else:
        print('非财报季 → 跳过三表增量 (省 Tushare 额度; --force-fin 可强制)')
    update_events(st)
    print('\n完成, 用时 %.0fs' % (time.time() - t0))
