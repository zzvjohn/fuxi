"""
扩展数据加载模块 — 支持基本面(ROE/EPS) + 订单流(moneyflow) 数据
用于 daily_factor_hypothesis.py 的因子评估
"""
import pandas as pd
import numpy as np
from pathlib import Path
import glob
import os

# P-20260902-001: 原为相对路径 Path('data'), 而流水线从 research/factor_alchemy 运行,
# 导致 report_rc.csv / sw_index_member_all.csv 恒为"不存在" → 分析师分歧字段
# (eps_disp/tp_disp/rating_disp/n_cover) 与行业 peer 收益 (industry_ret_peer) 全链路为空。
# 改为基于 __file__ 的绝对路径, 与 factor_ic_computer.RAW_DIR 口径对齐。
DATA_DIR = Path(__file__).resolve().parent.parent / 'data'


def _asof_coverage_check(field, w, label=''):
    """P-20260911-008 防复发自检: asof 合并宽表的覆盖率体检。

    核心教训: `reindex(daily_index)` 是**取交集**, 会丢弃窗口外全部公告日 → ffill
    无源可填 → 窗口前半段近乎全空。此时「末行非空率」看着正常 (91%), 因为窗口内
    最近一次公告总能填上末行; 真正暴露问题的是 **first_valid 的中位位置**。

    判据 (任意命中即告警, 不抛异常以免打断流水线):
      ① first_valid 中位数落在面板后 50% 且 前半段非空率 < 5% 而末日 > 30% ⇒ 疑似 reindex 取交集 bug
      ② last 与 mid 非空率**同时为 0** ⇒ 必为代码 bug, 而非数据缺失
    """
    try:
        n = len(w)
        if n < 10 or w.shape[1] == 0:
            return
        nn = w.notna()
        arr = nn.values
        pos = [int(np.argmax(arr[:, j])) for j in range(arr.shape[1]) if arr[:, j].any()]
        if not pos:
            print(f'  [自检⚠️] {label or field}: 全表无有效值 —— 请检查数据源文件是否存在')
            return
        med_pos = float(np.median(pos))
        first_half = float(nn.iloc[: n // 2].values.mean())
        last_rate = float(nn.iloc[-1].values.mean())
        mid_rate = float(nn.iloc[n // 2].values.mean())
        if med_pos > 0.5 * n and first_half < 0.05 and last_rate > 0.30:
            print(f'  [自检🔴BUG] {label or field}: first_valid 中位数落在面板第 {int(med_pos)}/{n} 日, '
                  f'前半段非空率仅 {first_half:.1%} 而末日 {last_rate:.1%} '
                  f'⇒ 疑似 reindex 取交集 (ffill 无源)。检查 asof 合并是否用 index.union — 见 P-20260911-008')
        elif last_rate == 0 and mid_rate == 0:
            print(f'  [自检⚠️] {label or field}: last 与 mid 非空率同时为 0 '
                  f'⇒ 必为代码 bug 而非数据缺失 — 见 P-20260911-008')
    except Exception as e:  # 自检自身不得影响主流程
        print(f'  [自检] {field} 覆盖率检查异常: {e}')


def load_fundamental_wide(daily_index, daily_columns):
    """
    加载 ROE/EPS 基本面数据，返回 wide-format DataFrame。
    
    参数:
        daily_index: 日频数据的日期索引 (DatetimeIndex)
        daily_columns: 日频数据的股票列 (stock codes)
    
    返回:
        roe_wide: DataFrame (dates × stocks) — ROE 填值到日频
        eps_wide: DataFrame (dates × stocks) — EPS 填值到日频
    """
    # 加载所有 fina_batch 文件
    batch_files = sorted(glob.glob(str(DATA_DIR / 'tushare' / 'fina_batch_*.csv')))
    if not batch_files:
        print('  [基本面] 无 fina_batch 文件，返回空')
        return None, None
    
    dfs = []
    for f in batch_files:
        df = pd.read_csv(f, usecols=['ts_code', 'end_date', 'roe', 'eps'],
                         dtype={'ts_code': str, 'end_date': str})
        dfs.append(df)
    
    fina = pd.concat(dfs, ignore_index=True)
    fina['end_date'] = pd.to_datetime(fina['end_date'])
    
    # 过滤在 daily 股票池中的股票
    fina = fina[fina['ts_code'].isin(daily_columns)]
    
    # Pivot: stocks × dates → 只取需要的股票
    roe_pivot = fina.pivot_table(values='roe', index='end_date', columns='ts_code', aggfunc='last')
    eps_pivot = fina.pivot_table(values='eps', index='end_date', columns='ts_code', aggfunc='last')
    
    # 重新索引到日频日期, forward fill (季频数据在报告期之间不变)
    # 🔴 P-20260911-008: 原为 `reindex(daily_index)` = 取交集 → 窗口起始日之前的
    #    报告期 (如 2025-09-30) 被丢弃 → ffill 无源 → 面板前段 NaN。必须 union。
    #    注: 本表仅有 end_date 无 ann_date, 报告期口径本身含公告滞后 (既有语义, 未改)。
    roe_daily = roe_pivot.reindex(roe_pivot.index.union(daily_index)).ffill().reindex(index=daily_index)
    eps_daily = eps_pivot.reindex(eps_pivot.index.union(daily_index)).ffill().reindex(index=daily_index)
    
    # 确保包含所有 daily 列 (缺失的填 NaN)
    for col in daily_columns:
        if col not in roe_daily.columns:
            roe_daily[col] = np.nan
        if col not in eps_daily.columns:
            eps_daily[col] = np.nan
    roe_daily = roe_daily[daily_columns]
    eps_daily = eps_daily[daily_columns]
    _asof_coverage_check('roe', roe_daily, 'fundamental_roe')
    _asof_coverage_check('eps', eps_daily, 'fundamental_eps')
    
    n_valid_roe = roe_daily.iloc[-1].notna().sum()
    n_valid_eps = eps_daily.iloc[-1].notna().sum()
    print(f'  [基本面] ROE: {n_valid_roe} 只有效, EPS: {n_valid_eps} 只有效')
    
    return roe_daily, eps_daily


def load_moneyflow_wide(daily_index, daily_columns):
    """
    加载资金流向数据 (小单/大单买卖量)，返回 wide-format DataFrames。
    
    返回:
        buy_sm_vol: 小单买入量 (散户)
        sell_sm_vol: 小单卖出量
        buy_lg_vol: 大单买入量 (机构)
        sell_lg_vol: 大单卖出量
        buy_elg_vol: 超大单买入量 (大机构)
        sell_elg_vol: 超大单卖出量
        net_mf_amount: 主力净流入额 (P-20260814-002 三路资金分歧数据)
    """
    mf_path = DATA_DIR / 'raw' / 'moneyflow_daily.csv'
    if not mf_path.exists():
        print('  [订单流] moneyflow_daily.csv 不存在，返回空')
        return [None] * 7
    
    mf = pd.read_csv(mf_path, dtype={'ts_code': str, 'trade_date': str})
    mf['trade_date'] = pd.to_datetime(mf['trade_date'])
    
    # 过滤股票池
    mf = mf[mf['ts_code'].isin(daily_columns)]
    
    fields = ['buy_sm_vol', 'sell_sm_vol', 'buy_lg_vol', 'sell_lg_vol',
              'buy_elg_vol', 'sell_elg_vol', 'net_mf_amount']

    # ── P-20260916-003 缺失口径开关 ──────────────────────────────────────────
    # 实测 (2026-09-16): moneyflow_daily.csv 覆盖期内有价但无订单流的 (股,日) 对
    #   补采前 46.01% → 补采后 5.46%; 残留缺口的 90.6% 是**北交所** ——
    #   Tushare moneyflow 对 .BJ 仅 10,832 行 (vs daily_prices 287,648 行, 覆盖 3.8%),
    #   属数据源能力边界, 补不回来。
    # 🔴 原实现 `reindex(...).fillna(0)` 把「无订单流」伪造成「零净买入」:
    #   lg_imbalance=0 ⇒ lg_rank≈0.5 常量 ⇒ crowd=0.5×amount_rank ⇒ 门控退化为
    #   「按成交额份额排序的反转」, 凭空造出信号。实证同因子 fillna(0) ICIR 0.342
    #   vs 保留 NaN 0.277 (补采前) / 0.436 vs 0.370 (补采后)。
    # 🔴 v0.7.29 翻默认 nan (用户授权, 与「跨截面样本量门槛」同批) ─────────────
    #   翻默认判据 (「会不会凭空造出 PASS」): 本项属**单向收紧 + 去伪造** ——
    #   缺失样本从「伪造成零净买入」改为「退出截面」, 只会让因子读数变差或不变,
    #   不会凭空造出 PASS, 故可单独翻。
    #   ⚠️ 但它会**同时**让稀疏因子的有效截面数变少, 故必须与
    #   daily_factor_hypothesis.py 的 `_ic_dates_gate` (v0.7.29) **同批**上线 ——
    #   否则会放进 `large_order_flow_zscore_20d` 式「仅 37 截面却 ICIR 0.401」的假 PASS。
    #   回退: 显式设 FUXI_MF_MISSING=zero 即恢复修正前行为 (零回归基线)。
    _policy = os.environ.get('FUXI_MF_MISSING', 'nan').strip().lower()
    _fill = 0.0 if _policy != 'nan' else None

    results = []
    for field in fields:
        pivot = mf.pivot_table(values=field, index='trade_date', columns='ts_code', aggfunc='sum')
        pivot = pivot.reindex(daily_index)
        if _fill is None:
            pivot = pivot.reindex(columns=daily_columns)          # 保留 NaN: 缺失即缺失
        else:
            pivot = pivot.fillna(_fill)
            for col in daily_columns:
                if col not in pivot.columns:
                    pivot[col] = _fill
            pivot = pivot[daily_columns]
        results.append(pivot)

    n_valid = (~results[0].iloc[-1].isna()).sum() if results[0] is not None else 0
    print(f'  [订单流] {n_valid} 只有效数据 (+net_mf_amount) [缺失口径={_policy}]')

    return tuple(results)


def load_margin_wide(daily_index, daily_columns):
    """
    加载两融数据 (margin_detail.csv)，返回 wide-format DataFrames。
    P-20260814-001: 两融信号范式 (未覆盖范式17) 数据接入。

    注意: margin_detail.csv 最新日期为 2025-12-31 (滞后待补采),
    因子研发用 2015-2025 十年样本, JQ 回测用平台实时两融数据。

    返回:
        rzye: 融资余额
        rqye: 融券余额
        rzmre: 融资买入额
        rqyl: 融券余量
        rqchl: 融券偿还量
        rzche: 融资偿还额      (P-20260911-margin, 2026-08 起)
        rqmcl: 融券卖出量      (同上)
        rzrqye: 融资融券余额   (同上)
    """
    margin_path = DATA_DIR / 'raw' / 'margin_detail.csv'
    if not margin_path.exists():
        print('  [两融] margin_detail.csv 不存在，返回空')
        return [None] * 8

    mg = pd.read_csv(margin_path, dtype={'ts_code': str, 'trade_date': str})
    mg['trade_date'] = pd.to_datetime(mg['trade_date'], format='mixed')
    mg = mg[mg['ts_code'].isin(daily_columns)]
    # 只取 daily_index 覆盖范围 (daily_index 一般为近 270 日)
    mg = mg[(mg['trade_date'] >= daily_index.min() - pd.Timedelta(days=500))
            & (mg['trade_date'] <= daily_index.max())]

    fields = ['rzye', 'rqye', 'rzmre', 'rqyl', 'rqchl',
              # P-20260911-margin (2026-09-11): Tushare 2026-08 起新增 3 列
              'rzche', 'rqmcl', 'rzrqye']
    results = []
    for field in fields:
        if field not in mg.columns:
            # 新增列在存量行 (2026-08 前) 为 NaN → 仍产出宽表以对齐字段名
            pivot = pd.DataFrame(np.nan, index=daily_index, columns=daily_columns)
        else:
            pivot = mg.pivot_table(values=field, index='trade_date', columns='ts_code', aggfunc='last')
            pivot = pivot.reindex(daily_index).ffill()  # 两融余额存量 → 前向填充
        for col in daily_columns:
            if col not in pivot.columns:
                pivot[col] = np.nan
        pivot = pivot[daily_columns]
        results.append(pivot)

    n_valid = int(results[0].iloc[-1].notna().sum()) if results[0] is not None else 0
    print(f'  [两融] {" ".join(fields)} 已接入, {n_valid} 只有效 (最新 {mg["trade_date"].max().date() if len(mg) else "N/A"})')

    return tuple(results)


def load_analyst_forecast_wide(daily_index, daily_columns):
    """P-20260819-002: 从 report_rc.csv 构建分析师预测分歧宽表 (MFD 分析师版数据层)。

    分歧定义 (Machine Forecast Disagreement 分析师版):
      - eps_disp:    近 90 日同股票 eps 预测的变异系数 std/|mean| (数值分歧)
      - tp_disp:     近 90 日目标价预测变异系数 (分歧越大→预期越不确定)
      - rating_disp: 近 90 日评级数值化后的标准差 (买入=2/增持=1.5/中性=1/减持=0.5/卖出=0)
      - n_cover:     近 90 日覆盖研报家数 (分歧的置信度)

    宽表对齐 daily_index 日频, 发布后前向填充 (limit=60 防陈旧)。
    返回 4 个宽表 (dates × stocks), 数据缺失返回 [None]*4 不阻塞。
    """
    rc_path = DATA_DIR / 'raw' / 'report_rc.csv'
    if not rc_path.exists():
        print('  [分析师] report_rc.csv 不存在，返回空')
        return [None] * 4

    RATING_MAP = {'买入': 2.0, '增持': 1.5, '强烈推荐': 2.0, '推荐': 1.5,
                  '审慎推荐': 1.25, '谨慎推荐': 1.25, '优于大市': 1.5,
                  '中性': 1.0, '持有': 1.0, '回避': 0.5, '减持': 0.5, '卖出': 0.0}

    rc = pd.read_csv(rc_path, dtype={'ts_code': str, 'report_date': str,
                                     'ann_date': str, 'rating': str, 'eps': str,
                                     'tp': str, 'org_name': str})
    date_col = 'report_date' if 'report_date' in rc.columns else 'ann_date'
    rc[date_col] = pd.to_datetime(rc[date_col], format='mixed', errors='coerce')
    rc = rc.dropna(subset=[date_col])
    rc['eps'] = pd.to_numeric(rc['eps'], errors='coerce')
    rc['tp'] = pd.to_numeric(rc['tp'], errors='coerce')
    rc['rating_num'] = rc['rating'].map(RATING_MAP)
    rc = rc[rc['ts_code'].isin(daily_columns)]

    if len(rc) == 0:
        print('  [分析师] 无池内研报数据，返回空')
        return [None] * 4

    rc = rc.sort_values(date_col)

    def _roll_field(col, agg='std', window='90D'):
        """按股票滚动 90 天聚合, 返回 (ts_code, date, value) 长表"""
        sub = rc.dropna(subset=[col])[[date_col, 'ts_code', col]]
        if len(sub) == 0:
            return None
        g = sub.set_index(date_col).groupby('ts_code')[col]
        if agg == 'std':
            s = g.apply(lambda x: x.rolling(window, min_periods=1).std(), include_groups=False)
        elif agg == 'mean':
            s = g.apply(lambda x: x.rolling(window, min_periods=1).mean(), include_groups=False)
        elif agg == 'count':
            s = g.apply(lambda x: x.rolling(window, min_periods=1).count(), include_groups=False)
        s = s.reset_index().rename(columns={date_col: 'd', col: 'v'})
        return s

    def _to_wide(long_df, fill_limit=60):
        if long_df is None or len(long_df) == 0:
            return None
        p = long_df.pivot_table(index='d', columns='ts_code', values='v', aggfunc='last')
        p = p.reindex(index=daily_index, columns=daily_columns).ffill(limit=fill_limit)
        return p

    # eps 分歧 = std / |mean| (变异系数, 跨股票可比)
    eps_std = _roll_field('eps', 'std')
    eps_mean = _roll_field('eps', 'mean')
    eps_std_w, eps_mean_w = _to_wide(eps_std), _to_wide(eps_mean)
    eps_disp = None
    if eps_std_w is not None and eps_mean_w is not None:
        eps_disp = eps_std_w / (eps_mean_w.abs() + 1e-9)

    tp_std, tp_mean = _roll_field('tp', 'std'), _roll_field('tp', 'mean')
    tp_std_w, tp_mean_w = _to_wide(tp_std), _to_wide(tp_mean)
    tp_disp = None
    if tp_std_w is not None and tp_mean_w is not None:
        tp_disp = tp_std_w / (tp_mean_w.abs() + 1e-9)

    rating_disp = _to_wide(_roll_field('rating_num', 'std'))
    n_cover = _to_wide(_roll_field('rating_num', 'count'))

    n_valid = int(eps_disp.iloc[-1].notna().sum()) if eps_disp is not None else 0
    latest = rc[date_col].max().date() if len(rc) else 'N/A'
    print(f'  [分析师] eps/tp/rating 分歧 + 覆盖家数已接入, 最新 {latest}, '
          f'eps_disp 近端 {n_valid} 只有效')
    return eps_disp, tp_disp, rating_disp, n_cover


def load_industry_wide(daily_index, daily_columns):
    """P-20260819-003: 申万 L1 行业 peer 收益宽表 (Salience DS 数据层)。

    数据源: data/raw/sw_index_member_all.csv (index_member_all 接口, 2026-08-19 采集)
    方法: 个股日收益 -> 按 l1_code 等权聚合行业收益 -> 每股票列填充其所属行业收益。
    局限: 成分映射为当前时点快照 (静态近似, 有成分调整偏差, 审计已记录)。

    返回: industry_ret_peer (dates × stocks, 该股所属 L1 行业当日等权收益)
    """
    mem_path = DATA_DIR / 'raw' / 'sw_index_member_all.csv'
    if not mem_path.exists():
        print('  [行业] sw_index_member_all.csv 不存在，返回空')
        return None
    mem = pd.read_csv(mem_path, dtype={'ts_code': str, 'l1_code': str})
    code2l1 = dict(zip(mem['ts_code'], mem['l1_code']))
    n_mapped = len(set(daily_columns) & set(code2l1))
    if n_mapped == 0:
        print('  [行业] 映射覆盖 0 只，返回空')
        return None

    dp_path = DATA_DIR / 'raw' / 'daily_prices.csv'
    daily = pd.read_csv(dp_path, dtype={'ts_code': str, 'trade_date': str},
                        usecols=['ts_code', 'trade_date', 'close'])
    daily['trade_date'] = pd.to_datetime(daily['trade_date'], format='mixed')
    daily = daily[daily['ts_code'].isin(daily_columns)].sort_values(['ts_code', 'trade_date'])
    daily['ret'] = daily.groupby('ts_code')['close'].pct_change()
    daily['l1'] = daily['ts_code'].map(code2l1)

    ind_ret = daily.dropna(subset=['l1', 'ret']) \
                   .groupby(['trade_date', 'l1'])['ret'].mean().reset_index()
    ind_wide = ind_ret.pivot(index='trade_date', columns='l1', values='ret') \
                      .reindex(daily_index)

    peer = pd.DataFrame(index=daily_index, columns=daily_columns, dtype=float)
    for code in daily_columns:
        l1 = code2l1.get(code)
        if l1 is not None and l1 in ind_wide.columns:
            peer[code] = ind_wide[l1]

    # 无行业归属 -> 全市场等权收益 (保守代理)
    mkt = daily.dropna(subset=['ret']).groupby('trade_date')['ret'].mean()
    peer = peer.apply(lambda col: col.fillna(mkt))

    print(f'  [行业] L1 映射 {n_mapped}/{len(daily_columns)} 只, peer 收益宽表已接入')
    return peer


def load_top_list_wide(daily_index, daily_columns):
    """
    加载龙虎榜数据 (top_list.csv + top_inst.csv)，返回 wide-format DataFrames。
    P-20260812-026: 龙虎榜因子 (游资席位/机构席位) 数据接入。

    设计: 龙虎榜为事件稀疏数据 (每日 40-80 只上榜), 未上榜日填 0。
    因子公式应在时间序列上做 rolling 聚合 (如近 20 日上榜次数)。

    返回:
        lhb_flag:         当日上榜次数 (同日多原因上榜累计)
        lhb_net_amount:   当日龙虎榜净买入额合计 (万元)
        lhb_net_rate:     当日净买率合计 (%)
        lhb_amount:       当日龙虎榜成交额合计
        lhb_inst_net_buy: 当日机构专用席位净买入合计 (万元)
        lhb_inst_buy:     当日机构专用席位买入合计
        lhb_inst_sell:    当日机构专用席位卖出合计
        lhb_north_net_buy: 当日北向专用席位(深股通/沪股通)净买入合计 (P-20260823-001)
        lhb_north_buy:    当日北向专用席位买入合计
        lhb_north_sell:   当日北向专用席位卖出合计
        lhb_yz_net_buy:   当日游资营业部净买入合计 (P-20260904-004: 排除机构专用/北向专用后的营业部席位)
        lhb_yz_buy:       当日游资营业部买入合计
        lhb_yz_sell:      当日游资营业部卖出合计
    """
    tl_path = DATA_DIR / 'raw' / 'top_list.csv'
    ti_path = DATA_DIR / 'raw' / 'top_inst.csv'

    if not tl_path.exists():
        print('  [龙虎榜] top_list.csv 不存在，返回空')
        return [None] * 13

    tl = pd.read_csv(tl_path, dtype={'ts_code': str, 'trade_date': str})
    tl['trade_date'] = pd.to_datetime(tl['trade_date'], format='%Y%m%d')
    tl = tl[tl['ts_code'].isin(daily_columns)]
    tl = tl[(tl['trade_date'] >= daily_index.min() - pd.Timedelta(days=500))
            & (tl['trade_date'] <= daily_index.max())]
    for c in ['net_amount', 'net_rate', 'amount']:
        tl[c] = pd.to_numeric(tl[c], errors='coerce')

    # top_list 聚合
    # fix(2026-08-27): values 与 columns 不能同为 ts_code (pandas 2.3.3 报
    # "Grouper for 'ts_code' not 1-dimensional"); 用 net_amount 做 values,
    # aggfunc='size' 只计数行数, values 内容不参与。
    lhb_flag = tl.pivot_table(values='net_amount', index='trade_date',
                              columns='ts_code', aggfunc='size')
    lhb_net_amount = tl.pivot_table(values='net_amount', index='trade_date',
                                    columns='ts_code', aggfunc='sum')
    lhb_net_rate = tl.pivot_table(values='net_rate', index='trade_date',
                                  columns='ts_code', aggfunc='sum')
    lhb_amount = tl.pivot_table(values='amount', index='trade_date',
                                columns='ts_code', aggfunc='sum')

    # top_inst 机构专用席位聚合 (可选, 文件不存在则置 None)
    lhb_inst_net_buy = lhb_inst_buy = lhb_inst_sell = None
    if ti_path.exists():
        ti = pd.read_csv(ti_path, dtype={'ts_code': str, 'trade_date': str})
        ti['trade_date'] = pd.to_datetime(ti['trade_date'], format='%Y%m%d')
        inst = ti[ti['exalter'].str.contains('机构专用', na=False)]
        inst = inst[inst['ts_code'].isin(daily_columns)]
        inst = inst[(inst['trade_date'] >= daily_index.min() - pd.Timedelta(days=500))
                    & (inst['trade_date'] <= daily_index.max())]
        for c in ['buy', 'sell', 'net_buy']:
            inst[c] = pd.to_numeric(inst[c], errors='coerce')
        lhb_inst_net_buy = inst.pivot_table(values='net_buy', index='trade_date',
                                            columns='ts_code', aggfunc='sum')
        lhb_inst_buy = inst.pivot_table(values='buy', index='trade_date',
                                        columns='ts_code', aggfunc='sum')
        lhb_inst_sell = inst.pivot_table(values='sell', index='trade_date',
                                         columns='ts_code', aggfunc='sum')

    # top_inst 北向专用席位聚合 (深股通/沪股通专用, P-20260823-001)
    lhb_north_net_buy = lhb_north_buy = lhb_north_sell = None
    if ti_path.exists():
        ti = pd.read_csv(ti_path, dtype={'ts_code': str, 'trade_date': str})
        ti['trade_date'] = pd.to_datetime(ti['trade_date'], format='%Y%m%d')
        north = ti[ti['exalter'].str.contains('深股通专用|沪股通专用', na=False, regex=True)]
        north = north[north['ts_code'].isin(daily_columns)]
        north = north[(north['trade_date'] >= daily_index.min() - pd.Timedelta(days=500))
                      & (north['trade_date'] <= daily_index.max())]
        for c in ['buy', 'sell', 'net_buy']:
            north[c] = pd.to_numeric(north[c], errors='coerce')
        lhb_north_net_buy = north.pivot_table(values='net_buy', index='trade_date',
                                              columns='ts_code', aggfunc='sum')
        lhb_north_buy = north.pivot_table(values='buy', index='trade_date',
                                          columns='ts_code', aggfunc='sum')
        lhb_north_sell = north.pivot_table(values='sell', index='trade_date',
                                           columns='ts_code', aggfunc='sum')

    # top_inst 游资营业部聚合 (P-20260904-004: 排除机构专用/北向专用后的营业部席位)
    lhb_yz_net_buy = lhb_yz_buy = lhb_yz_sell = None
    if ti_path.exists():
        ti = pd.read_csv(ti_path, dtype={'ts_code': str, 'trade_date': str})
        ti['trade_date'] = pd.to_datetime(ti['trade_date'], format='%Y%m%d')
        yz = ti[~ti['exalter'].str.contains('机构专用|深股通专用|沪股通专用', na=False, regex=True)]
        yz = yz[yz['exalter'].str.contains('营业部', na=False)]
        yz = yz[yz['ts_code'].isin(daily_columns)]
        yz = yz[(yz['trade_date'] >= daily_index.min() - pd.Timedelta(days=500))
                & (yz['trade_date'] <= daily_index.max())]
        for c in ['buy', 'sell', 'net_buy']:
            yz[c] = pd.to_numeric(yz[c], errors='coerce')
        lhb_yz_net_buy = yz.pivot_table(values='net_buy', index='trade_date',
                                        columns='ts_code', aggfunc='sum')
        lhb_yz_buy = yz.pivot_table(values='buy', index='trade_date',
                                    columns='ts_code', aggfunc='sum')
        lhb_yz_sell = yz.pivot_table(values='sell', index='trade_date',
                                     columns='ts_code', aggfunc='sum')

    results = []
    for pivot in [lhb_flag, lhb_net_amount, lhb_net_rate, lhb_amount,
                  lhb_inst_net_buy, lhb_inst_buy, lhb_inst_sell,
                  lhb_north_net_buy, lhb_north_buy, lhb_north_sell,
                  lhb_yz_net_buy, lhb_yz_buy, lhb_yz_sell]:
        if pivot is None or len(pivot) == 0:
            results.append(None)
            continue
        pivot = pivot.reindex(daily_index).fillna(0)
        for col in daily_columns:
            if col not in pivot.columns:
                pivot[col] = 0.0
        pivot = pivot[daily_columns]
        results.append(pivot)

    n_days = int((results[0] > 0).sum().sum()) if results[0] is not None else 0
    print(f'  [龙虎榜] 13 字段已接入, 上榜 stock-days={n_days} '
          f'(top_list 最新 {tl["trade_date"].max().date() if len(tl) else "N/A"})')

    return tuple(results)


def load_hk_hold_wide(daily_index, daily_columns):
    """
    加载北向个股持股数据 (hk_hold.csv, 仅 SH+SZ)，返回 wide-format DataFrames。
    P-20260814-002: 三路资金分歧 (2018-2024 历史验证) 数据接入。

    重要: 北向明细自 2024-08-19 起停更 (仅零星 20240930/20241231),
    2025+ 无数据 → pivot 只覆盖 2018-2024 历史窗口, 近端为 NaN。
    因子公式用 2018-2024 窗口做历史 IC 验证。

    返回:
        north_vol:   北向持股量 (股)
        north_ratio: 北向持股占流通股本比 (%)
    """
    hh_path = DATA_DIR / 'raw' / 'hk_hold.csv'
    if not hh_path.exists():
        print('  [北向] hk_hold.csv 不存在，返回空')
        return [None] * 2

    hh = pd.read_csv(hh_path, dtype={'ts_code': str, 'trade_date': str})
    hh = hh[hh['exchange'].isin(['SH', 'SZ'])]
    hh['trade_date'] = pd.to_datetime(hh['trade_date'], format='%Y%m%d')
    hh = hh[hh['ts_code'].isin(daily_columns)]
    hh = hh[(hh['trade_date'] >= daily_index.min() - pd.Timedelta(days=500))
            & (hh['trade_date'] <= daily_index.max())]
    for c in ['vol', 'ratio']:
        hh[c] = pd.to_numeric(hh[c], errors='coerce')

    north_vol = hh.pivot_table(values='vol', index='trade_date',
                               columns='ts_code', aggfunc='sum')
    north_ratio = hh.pivot_table(values='ratio', index='trade_date',
                                 columns='ts_code', aggfunc='sum')

    results = []
    for pivot in [north_vol, north_ratio]:
        pivot = pivot.reindex(daily_index)  # 不 fillna: 缺失=停更/未持有
        for col in daily_columns:
            if col not in pivot.columns:
                pivot[col] = np.nan
        pivot = pivot[daily_columns]
        results.append(pivot)

    n_valid = int(results[0].iloc[-1].notna().sum()) if results[0] is not None else 0
    print(f'  [北向] north_vol/north_ratio 已接入 (2018-2024 历史窗口, '
          f'最新 {hh["trade_date"].max().date() if len(hh) else "N/A"}, 近端 {n_valid} 只有效)')

    return tuple(results)


def load_fina_growth_wide(daily_index, daily_columns):
    """P-20260827-002: 财务成长质量宽表 (netprofit_yoy/roe_dt/tr_yoy)。

    数据源: data/raw/fina_indicator.csv
    口径: ann_date (公告日) asof 合并 → reindex 价格日历 → ffill
          = 公告日之前不可见 (无前视), 与 FactorICComputer 同口径。
    返回: (netprofit_yoy, roe_dt, tr_yoy) 三个 (dates × stocks) 宽表, 缺数据返回 (None,)*3
    """
    fp = DATA_DIR / 'raw' / 'fina_indicator.csv'
    if not fp.exists():
        print('  [财务成长] fina_indicator.csv 不存在，返回空')
        return None, None, None
    fi = pd.read_csv(fp, usecols=['ts_code', 'ann_date', 'end_date', 'netprofit_yoy', 'roe_dt', 'tr_yoy'],
                     dtype={'ts_code': str, 'ann_date': str})
    fi['ann_date'] = pd.to_datetime(fi['ann_date'].astype(str).str.replace(r'\.0$', '', regex=True),
                                    format='mixed', errors='coerce')
    fi = fi.dropna(subset=['ann_date'])
    fi = fi[fi['ts_code'].isin(daily_columns)]
    # 🔴 P2-2c (2026-09-28): 与 factor_ic_computer 同步。单键 (ts_code,ann_date) 去重会丢弃
    # 「同一公告日的多期财报」(年报 1231 与一季报 0331 同日披露 => 年报被丢; 实测 120 股
    # 主表 1231 记录 -29.3%)。改三键去重 + end_date 参与排序 => 下方 pivot(aggfunc='last')
    # 在同公告日取【最新报告期】(end_date 最大); 对旧主表(二键已唯一)为等价变换。
    fi = fi.sort_values(['ts_code', 'ann_date', 'end_date'], kind='stable')
    fi = fi.drop_duplicates(['ts_code', 'end_date', 'ann_date'], keep='last')
    outs = []
    for col in ['netprofit_yoy', 'roe_dt', 'tr_yoy']:
        w = fi.pivot_table(index='ann_date', columns='ts_code', values=col, aggfunc='last')
        # 🔴 必须 union: reindex(daily_index) 是取交集, 会丢弃窗口外的全部公告日,
        #    ffill 无源可填 → 窗口内字段大面积 NaN (2026-09-11 发现, 与 _kd_asof 同源 bug)
        w = w.reindex(w.index.union(daily_index)).ffill()
        w = w.reindex(index=daily_index).reindex(columns=daily_columns)
        _asof_coverage_check(col, w, 'fina_growth')
        outs.append(w)
    n_valid = int(outs[0].iloc[-1].notna().sum())
    print(f'  [财务成长] netprofit_yoy/roe_dt/tr_yoy 已接入 (ann_date asof, 最新 {n_valid} 只有效)')
    return tuple(outs)


def load_valuation_wide(daily_index, daily_columns):
    """P-20260827-004 + P-20260911-kd: 估值宽表 (日频)。

    数据源: data/raw/daily_basic.csv
    字段: circ_mv / total_mv (万元) + pb / ps_ttm / pe_ttm / dv_ratio / dv_ttm
    口径变更 (P-20260911-kd): 原为**月末快照 ffill**, 会让同一月内所有交易日的
      因子值完全相同 → IC 序列自相关被高估、日频 fwd5 检验失真。改**日频直出**
      + 仅对 reindex 后的缺口 ffill。
    🔴 合并前必须 drop_duplicates(['ts_code','trade_date']): daily_basic.csv 含
      1.90x 重复行 (13.77M → 7.03M 唯一), 不去重会让后续 merge+shift 错位。
    🔴 PS 自身 250 日分位在**加载器内**用完整历史算 (kd_ps_pct_own):
      流水线 load_data 只取最近 n_days=270 天面板, 因子层 .rolling(250) 仅覆盖
      最后 ~20 天 → 基本无值。长窗口统计一律在此层预先算好。
    返回: (circ_mv, total_mv, pb, ps_ttm, pe_ttm, dv_ratio, dv_ttm, kd_ps_pct_own),
          缺数据返回 (None,)*8
    """
    fp = DATA_DIR / 'raw' / 'daily_basic.csv'
    if not fp.exists():
        print('  [估值] daily_basic.csv 不存在，返回空')
        return (None,) * 8
    usecols = ['ts_code', 'trade_date', 'circ_mv', 'total_mv', 'pb', 'ps_ttm',
               'pe_ttm', 'dv_ratio', 'dv_ttm']
    fields = usecols[2:]
    ba = pd.read_csv(fp, usecols=usecols, dtype={'ts_code': str, 'trade_date': str})
    ba['trade_date'] = pd.to_datetime(ba['trade_date'], format='mixed', errors='coerce')
    ba = ba.dropna(subset=['trade_date'])
    ba = ba[ba['ts_code'].isin(daily_columns)]
    ba = ba.drop_duplicates(['ts_code', 'trade_date'], keep='last')
    # 只保留 [daily_index.min()-600d, ...]: 600 自然日 ≈ 275 交易日, 够 250 日分位
    lo = pd.Timestamp(daily_index.min()) - pd.Timedelta(days=600)
    ba = ba[ba['trade_date'] >= lo]
    outs = []
    for c in fields:
        w = ba.pivot(index='trade_date', columns='ts_code', values=c)
        w = w.reindex(daily_index).ffill().reindex(columns=daily_columns)
        outs.append(w)
    # PS 自身 250 日分位 (长窗口, 因子层无法承担)
    ps_w = ba.pivot(index='trade_date', columns='ts_code', values='ps_ttm')
    ps_pct = ps_w.rolling(250, min_periods=120).rank(pct=True)
    ps_pct = ps_pct.reindex(daily_index).ffill().reindex(columns=daily_columns)
    outs.append(ps_pct)
    n_valid = int(outs[0].iloc[-1].notna().sum())
    print(f'  [估值] circ_mv/total_mv/pb/ps_ttm/pe_ttm/dv_ratio/dv_ttm/ps_pct_own '
          f'已接入 (日频直出, 最新 {n_valid} 只有效)')
    return tuple(outs)


def load_industry_l1_wide(daily_index, daily_columns):
    """P-20260827-004: 申万 L1 行业代码静态映射宽表 (industry_l1, str 分类)。

    数据源: data/raw/sw_index_member_all.csv (静态成分快照, 无时变)
    返回: (industry_l1 宽表,), 缺数据返回 (None,)
    """
    fp = DATA_DIR / 'raw' / 'sw_index_member_all.csv'
    if not fp.exists():
        print('  [行业] sw_index_member_all.csv 不存在，返回空')
        return (None,)
    mem = pd.read_csv(fp, dtype={'ts_code': str, 'l1_code': str}, usecols=['ts_code', 'l1_code'])
    code2l1 = dict(zip(mem['ts_code'], mem['l1_code']))
    w = pd.DataFrame(index=daily_index, columns=daily_columns, dtype=object)
    for c in daily_columns:
        w[c] = code2l1.get(c, 'NA')
    n_mapped = sum(1 for c in daily_columns if code2l1.get(c) not in (None, 'NA'))
    print(f'  [行业] industry_l1 静态映射 {n_mapped}/{len(daily_columns)} 只')
    return (w,)


def load_feature_peer_wide(daily_index, daily_columns):
    """P-20260906-001: 任意量价特征行业均值宽表 (双重同伴效应分解数据层)。

    广发金工《根据同伴股票因子暴露带来的全新Alpha》(2026-09): 将因子分解为
    同伴均值 (peer_avg) 与个股相对偏差 (peer_dev), peer_avg 项 T 统计量
    远高于原因子。本函数提供特征行业均值宽表, peer_dev 由公式侧
    (feature - ind_avg_feature) 计算。

    特征: amount(成交额) / turnover(vol/close 量价代理) / amplitude(振幅) / returns
    局限: 成分映射为当前时点快照 (与 load_industry_wide 同口径, 审计已记录)。

    返回: (ind_avg_amount, ind_avg_turnover, ind_avg_amplitude, ind_avg_returns)
    """
    mem_path = DATA_DIR / 'raw' / 'sw_index_member_all.csv'
    dp_path = DATA_DIR / 'raw' / 'daily_prices.csv'
    if not mem_path.exists() or not dp_path.exists():
        print('  [行业] sw_index_member_all.csv 或 daily_prices.csv 不存在，返回空')
        return (None,) * 4
    mem = pd.read_csv(mem_path, dtype={'ts_code': str, 'l1_code': str},
                      usecols=['ts_code', 'l1_code'])
    code2l1 = dict(zip(mem['ts_code'], mem['l1_code']))
    n_mapped = len(set(daily_columns) & set(code2l1))
    if n_mapped == 0:
        print('  [行业] 映射覆盖 0 只，返回空')
        return (None,) * 4

    daily = pd.read_csv(
        dp_path,
        dtype={'ts_code': str, 'trade_date': str},
        usecols=['ts_code', 'trade_date', 'open', 'high', 'low', 'close',
                 'pre_close', 'vol', 'amount'],
    )
    daily['trade_date'] = pd.to_datetime(daily['trade_date'], format='mixed')
    daily = daily[daily['ts_code'].isin(daily_columns)].sort_values(
        ['ts_code', 'trade_date'])
    daily = daily.drop_duplicates(subset=['ts_code', 'trade_date'], keep='last')

    daily['returns'] = daily.groupby('ts_code')['close'].pct_change()
    daily['turnover'] = daily['vol'] / daily['close'].clip(lower=0.01)
    daily['amplitude'] = (daily['high'] - daily['low']) / daily['pre_close'].clip(lower=0.01)
    daily['l1'] = daily['ts_code'].map(code2l1)
    daily = daily.dropna(subset=['l1'])

    feats = {'ind_avg_amount': 'amount', 'ind_avg_turnover': 'turnover',
             'ind_avg_amplitude': 'amplitude', 'ind_avg_returns': 'returns'}
    out = []
    for out_name, feat in feats.items():
        ind_mean = (daily.dropna(subset=[feat])
                    .groupby(['trade_date', 'l1'])[feat].mean().reset_index())
        ind_wide = ind_mean.pivot(index='trade_date', columns='l1', values=feat) \
                          .reindex(daily_index)
        peer = pd.DataFrame(index=daily_index, columns=daily_columns, dtype=float)
        for code in daily_columns:
            l1 = code2l1.get(code)
            if l1 is not None and l1 in ind_wide.columns:
                peer[code] = ind_wide[l1]
        out.append(peer)

    print(f'  [行业] 特征 peer 宽表接入 (L1 映射 {n_mapped}/{len(daily_columns)} 只): '
          f'{" / ".join(feats.keys())}')
    return tuple(out)


def load_limit_up_wide(daily_index, daily_columns):
    """P-20260906-003: 涨停生态市场级指标宽表 (截面广播, 涨停家数/开板率/溢价)。

    数据源: daily_prices.csv 的 pct_chg/pre_close/high/close。
    涨停判定: pct_chg >= 9.8 (含 10% 主板板与 20% 创业板/科创板板; ST 5% 板
    天然不满足)。开板: 触及涨停价 (high >= pre_close*1.095 或 1.195) 但未封住。
    涨停溢价: 昨日涨停集合今日等权平均收益。

    返回: (limit_up_count, limit_up_open_rate, limit_up_premium) 三张截面广播宽表
    """
    dp_path = DATA_DIR / 'raw' / 'daily_prices.csv'
    if not dp_path.exists():
        print('  [涨停生态] daily_prices.csv 不存在，返回空')
        return (None,) * 3
    # 比窗口多读 10 天, 保证首日的"昨日涨停溢价"有名单
    start_dt = pd.Timestamp(daily_index.min()) - pd.Timedelta(days=10)
    daily = pd.read_csv(
        dp_path,
        dtype={'ts_code': str, 'trade_date': str},
        usecols=['ts_code', 'trade_date', 'high', 'close', 'pre_close', 'pct_chg'],
    )
    daily['trade_date'] = pd.to_datetime(daily['trade_date'], format='mixed')
    daily = daily[daily['trade_date'] >= start_dt]
    daily = daily.drop_duplicates(subset=['ts_code', 'trade_date'], keep='last')
    daily = daily.dropna(subset=['pct_chg'])
    daily = daily[daily['ts_code'].isin(daily_columns)]

    daily['is_limit_up'] = daily['pct_chg'] >= 9.8
    # 触及涨停价: 主板 10% 板 1.095 / 创业板(30)科创(68) 20% 板 1.195
    board20 = daily['ts_code'].str[:3].isin(['300', '301', '688', '689'])
    limit_ratio = np.where(board20, 1.195, 1.095)
    daily['touched'] = daily['high'] >= daily['pre_close'] * limit_ratio
    daily['open_board'] = daily['touched'] & ~daily['is_limit_up']

    daily['ret'] = daily.groupby('ts_code')['close'].pct_change()

    mkt = daily.groupby('trade_date').agg(
        n_limit=('is_limit_up', 'sum'),
        n_touched=('touched', 'sum'),
        n_open=('open_board', 'sum'),
    )
    mkt['open_rate'] = mkt['n_open'] / mkt['n_touched'].replace(0, np.nan)

    # 涨停溢价: 昨日涨停集合今日等权收益
    daily_prev = daily[['ts_code', 'trade_date', 'is_limit_up']].copy()
    daily_prev['trade_date'] = daily_prev['trade_date'] + pd.Timedelta(days=1)
    premium = (daily_prev[daily_prev['is_limit_up']]
               .merge(daily[['ts_code', 'trade_date', 'ret']],
                      on=['ts_code', 'trade_date'], how='inner')
               .groupby('trade_date')['ret'].mean())

    mkt = mkt.reindex(daily_index)
    premium = premium.reindex(daily_index)

    # pandas 3.x: Series.rolling.corr(other) 窗口内含 any NaN → 该窗口 NaN (pairwise
    # 行为已移除)。市场序列 NaN (无昨日涨停股/分母为0) 填 0: 语义=无赚钱效应样本,
    # 保证下游 rolling corr 类因子 (lu_*_beta) 可计算。
    mkt['n_limit'] = mkt['n_limit'].fillna(0.0)
    mkt['open_rate'] = mkt['open_rate'].fillna(0.0)
    premium = premium.fillna(0.0)

    out = []
    for series, name in [(mkt['n_limit'], 'limit_up_count'),
                         (mkt['open_rate'], 'limit_up_open_rate'),
                         (premium, 'limit_up_premium')]:
        w = pd.DataFrame(index=daily_index, columns=daily_columns, dtype=float)
        for c in daily_columns:
            w[c] = series
        out.append(w)

    n_days = mkt['n_limit'].notna().sum()
    print(f'  [涨停生态] 覆盖 {n_days} 交易日, 日均涨停 '
          f'{mkt["n_limit"].mean():.0f} 只 / 开板率 {mkt["open_rate"].mean():.1%}')
    return tuple(out)


# ═══════════════════════════════════════════════════════════════════════════
# 狮子锚点通道 (P-20260915-lion, 2026-09-15)
# ---------------------------------------------------------------------------
# 来源: 聚宽社区《变种小狮子 带涨停基因的股池轮动 V3.0》的 bias 构造。
#       审计见 reports/lion_v3_strategy_audit_20260915.md。
# 语义: 找**最近一次涨停日 L** → 取 L 及之前最后一根阴线的 low 作 anchor
#       → bias = 后复权收盘 / 后复权 anchor_low。衡量"现价距最近一次主力发力
#       起点(启动点)的涨幅", 本质是**结构锚点型反转**, 而非固定窗口反转。
#
# 🔴 后复权口径铁律 (本通道的核心难点):
#   1. daily_prices.csv 是**未复权**价; adj_factor.csv 只有 46 个**季频快照**
#      (每股 ~24 个, 2021Q1~2026Q3), 直接 asof ffill 会漏掉快照之间的除权。
#   2. 改用 pre_close(t)/close(t-1) 自算**日频**复权调整比 → cumprod 得 adj_t。
#      已与官方因子交叉校验: 600519 自算 1/1.1679 = 0.8562 vs 官方 0.8563 (误差 1e-4)。
#      识别到的除权日密度 0.345% ≈ 1 次/股/年, 与 A 股分红频率吻合。
#   3. 🔴 adj_t = cumprod(pre_close/close_prev) 是**递减**的 ⇒ 它是后复权因子 A
#      的**倒数**。后复权价 = close / adj_t = close × A_t。若误写成 close × adj_t,
#      有除权的票 bias 会**反向**偏移 (实测 7.5% 样本偏差 >5%, p99 达 44.7%)。
#      自检判据: 后复权 bias 应恒 >= 未复权 bias (分红压低未复权价)。
#
# 🔴 涨停判定铁律:
#   - 优先 stk_limit.up_limit **精确法** (close ≈ up_limit), 覆盖 97.0%。
#   - 缺价日 (2026-08-04 之后 + 早期零星, 共 517 个交易日) 用 pct_chg >= 9.5 兜底。
#   - ❌ 不可用 pct_chg >= 9.8 单独判定: 实测在**有价日**漏掉 19306 个 10% 板真涨停
#     (除权日 pct_chg 失真), 同时多认 33007 个"涨了 10% 但未封板"的创业板/科创票。
#   - ❌ 兜底阈值不能用 9.8, 否则漏掉全部 20% 板 (300/301/688/689) 涨停。
#
# 🔴 长窗口: 锚点回看 760 交易日 >> load_data(n_days=270) 的 ~185 交易日面板
#   ⇒ 必须在本层用完整文件算完再 reindex (同 _kd_* 通道铁律)。因子层只引用字段名。
# ═══════════════════════════════════════════════════════════════════════════

_LION_LOOKBACK = 760          # 锚点/涨停统计回看窗口 (交易日), 原策略 3*250


def load_lion_anchor_wide(daily_index, daily_columns, lookback=_LION_LOOKBACK):
    """狮子锚点宽表: (lion_bias, lion_lu_days, lion_lu_cnt)。

    lion_bias     后复权 close / 后复权 anchor_low (最近涨停日之前最后一根阴线的 low)。
                  越小 = 越贴近启动点。窗口内无涨停 → NaN。
    lion_lu_days  距最近涨停的交易日数; 窗口内无涨停 → 截断为 lookback (保留
                  "常年无涨停"这一信息, 不置 NaN)。
    lion_lu_cnt   lookback 窗口内涨停次数。

    实测 (2025-12~2026-09, 188 交易日, 全市场日频 fwd5):
      lion_bias    RankIC -0.0250  日ICIR -0.178  → 过 S1 闸门
      lion_lu_days RankIC +0.0439  日ICIR +0.410  → 过 S1 闸门 (本族最强)
      lion_lu_cnt  RankIC -0.0195  日ICIR -0.236  → |IC| 略低于 0.02 门槛
    """
    dp_path = DATA_DIR / 'raw' / 'daily_prices.csv'
    if not dp_path.exists():
        print('  [狮子锚点] daily_prices.csv 不存在, 返回空')
        return (None,) * 3

    # 回看 lookback 交易日 + 缓冲 → 日历天 (1.6x 覆盖 760 交易日 + 提前量)
    lo = pd.Timestamp(daily_index.min()) - pd.Timedelta(days=int(lookback * 1.6))
    hi = pd.Timestamp(daily_index.max())
    dp = pd.read_csv(
        dp_path, dtype={'ts_code': str, 'trade_date': str},
        usecols=['ts_code', 'trade_date', 'open', 'high', 'low', 'close',
                 'pre_close', 'pct_chg'])
    dp['trade_date'] = pd.to_datetime(dp['trade_date'], format='mixed')
    dp = dp[(dp['trade_date'] >= lo) & (dp['trade_date'] <= hi)]
    dp = dp[dp['ts_code'].isin(daily_columns)]
    # 🔴 去重必须先做: 重复 (ts_code, trade_date) 会让 groupby().shift(1) 错位
    # (同型坑: daily_basic.csv 1.90x 重复; daily_prices 实测无重复, 但防御性保留)
    dp = dp.drop_duplicates(['ts_code', 'trade_date'], keep='last')
    dp = dp.sort_values(['ts_code', 'trade_date'], kind='mergesort').reset_index(drop=True)
    if len(dp) == 0:
        print('  [狮子锚点] 窗口内无日线数据, 返回空')
        return (None,) * 3

    # ---- 日频后复权因子 (自算, 见文件头铁律 2/3) ----
    g = dp.groupby('ts_code', sort=False)
    r = dp['pre_close'] / g['close'].shift(1)
    dp['_adj_t'] = np.where(r.isna() | ~r.between(0.2, 5.0), 1.0, r)
    dp['_A'] = 1.0 / dp.groupby('ts_code', sort=False)['_adj_t'].cumprod()
    hc = dp['close'] * dp['_A']
    dp['_h_open'] = dp['open'] * dp['_A']
    dp['_h_low'] = dp['low'] * dp['_A']
    del r, g

    # ---- 涨停判定 ----
    sl_path = DATA_DIR / 'raw' / 'stk_limit.csv'
    if sl_path.exists():
        sl = pd.read_csv(sl_path, dtype={'ts_code': str, 'trade_date': str},
                         usecols=['ts_code', 'trade_date', 'up_limit'])
        sl['trade_date'] = pd.to_datetime(sl['trade_date'], format='mixed')
        sl = sl.drop_duplicates(['ts_code', 'trade_date'], keep='last')
        dp = dp.merge(sl, on=['ts_code', 'trade_date'], how='left')
        del sl
    else:
        dp['up_limit'] = np.nan
        print('  [狮子锚点] stk_limit.csv 缺失, 涨停判定全部回退 pct_chg 兜底')
    exact = np.isclose(dp['close'], dp['up_limit'], atol=0.011)
    fallback = dp['up_limit'].isna() & (dp['pct_chg'] >= 9.5)
    dp['_lu'] = exact | fallback
    n_exact, n_fb = int(exact.sum()), int(fallback.sum())
    del exact, fallback

    # ---- 锚点: 最近涨停日之前最后一根阴线的 low ----
    # 涨停日 close == high_limit == 当日最高 ⇒ close >= open, 不可能是阴线,
    # 故"L 及之前最后一根阴线"等价于"L 之前最后一根阴线" (与原策略语义一致)。
    dp['_bear_low'] = np.where(hc < dp['_h_open'], dp['_h_low'], np.nan)
    ff_bear = dp.groupby('ts_code', sort=False)['_bear_low'].ffill()
    dp['_anc_lu'] = np.where(dp['_lu'], ff_bear, np.nan)     # 只在涨停日取值
    anc = dp.groupby('ts_code', sort=False)['_anc_lu'].ffill()  # 广播给其后每一行
    del ff_bear, dp['_bear_low'], dp['_anc_lu']

    dp['_pos'] = dp.groupby('ts_code', sort=False).cumcount()
    dp['_lu_pos'] = np.where(dp['_lu'], dp['_pos'], np.nan)
    last_lu = dp.groupby('ts_code', sort=False)['_lu_pos'].ffill()
    has = last_lu.notna() & ((dp['_pos'] - last_lu) <= lookback)

    bias_w = np.where(has, hc / anc, np.nan)
    days_w = np.where(has, dp['_pos'] - last_lu,
                      np.where(dp['_pos'] >= lookback, float(lookback), np.nan))
    del anc, last_lu, has, hc
    cnt_w = (dp.groupby('ts_code', sort=False)['_lu']
             .transform(lambda s: s.rolling(lookback, min_periods=1).sum()))

    dp['lion_bias'] = bias_w
    dp['lion_lu_days'] = days_w
    dp['lion_lu_cnt'] = cnt_w
    del bias_w, days_w, cnt_w

    outs = []
    for c in ['lion_bias', 'lion_lu_days', 'lion_lu_cnt']:
        w = dp.pivot(index='trade_date', columns='ts_code', values=c)
        w = w.reindex(index=daily_index, columns=daily_columns)
        _asof_coverage_check(c, w, label=f'狮子锚点.{c}')
        outs.append(w)

    last_row = int(outs[0].iloc[-1].notna().sum())
    mid_row = int(outs[0].iloc[len(outs[0]) // 2].notna().sum())
    n_lu = int(dp['_lu'].sum())
    print(f'  [狮子锚点] lion_bias/lion_lu_days/lion_lu_cnt 已接入 '
          f'(涨停 {n_lu} 次: 精确 {n_exact} + 兜底 {n_fb}; '
          f'末日非空 {last_row} / 中位 {mid_row})')
    return tuple(outs)


# ═══════════════════════════════════════════════════════════════════════════
# 快刀斩股基本面通道 (P-20260911-kd, 2026-09-11)
# ---------------------------------------------------------------------------
# 数据源: data/raw/kuaidao_gaps/ (Tushare 逐股/月区间采集)
# 口径铁律:
#   1. 季频表 (balance/income/cashflow/fina) 是**年初至今累计值** → 必须先在
#      **季度层**做「单季化 → trailing 4Q 求和」得 TTM, 再 asof 到日频。
#      绝不可先 asof 到日频再 rolling(250) —— 后者是"250 个交易日", 语义错误。
#   2. 一律 ann_date (公告日) asof, 且 lag=1 交易日生效, 杜绝公告日当天使用
#      (财报多盘后披露, 当日建仓=1 日前视; 与 backtest-water-audit 铁律一致)。
#   3. 状态量 (资产负债/股东户数/前十大) 用 ffill 是对的; 事件量 (分红/增减持/
#      回购) 绝不 ffill —— ffill 会让同一次事件被窗口重复计入。
# ═══════════════════════════════════════════════════════════════════════════

_KD_GAP_DIR = DATA_DIR / 'raw' / 'kuaidao_gaps'


def _kd_load(fname, usecols):
    """读 kuaidao_gaps 下的表; fname 为绝对路径时直接读 (如 fina_indicator.csv)。"""
    fp = Path(fname)
    if not fp.is_absolute():
        fp = _KD_GAP_DIR / fname
    if not fp.exists():
        print(f'  [快刀基本面] {fp.name} 不存在, 跳过')
        return None
    try:
        return pd.read_csv(fp, usecols=lambda c: c in set(usecols), low_memory=False,
                           dtype={'ts_code': str, 'ann_date': str, 'end_date': str})
    except Exception as e:
        print(f'  [快刀基本面] 读取 {fp.name} 失败: {e}')
        return None


def _kd_num(d, cols):
    for c in cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors='coerce')
    return d


def _kd_dates(d):
    for c in ('ann_date', 'end_date'):
        if c in d.columns:
            d[c] = pd.to_datetime(d[c].astype(str).str.replace(r'\.0$', '', regex=True),
                                  format='mixed', errors='coerce')
    return d


def _kd_quarterly(fname, value_cols, ttm=True):
    """季频表 → 每 (ts_code, end_date) 取最新 ann_date → 单季化 → TTM(4Q)。

    返回 DataFrame[ts_code, ann_date, end_date, *value_cols, *{col}_ttm]；
    缺表/缺列返回 None。
    """
    use = ['ts_code', 'ann_date', 'end_date', 'report_type'] + list(value_cols)
    d = _kd_load(fname, use)
    if d is None:
        return None
    if any(c not in d.columns for c in ('ts_code', 'ann_date', 'end_date')):
        print(f'  [快刀基本面] {Path(fname).name} 缺主键列, 跳过')
        return None
    # report_type: 同报告期多口径时只取合并报表(1), 对齐 JQ/快刀口径
    if 'report_type' in d.columns:
        rt = d['report_type'].astype(str)
        if (rt == '1').any():
            d = d[rt == '1']
        d = d.drop(columns=['report_type'])
    miss = [c for c in value_cols if c not in d.columns]
    if miss:
        print(f'  [快刀基本面] {Path(fname).name} 缺列 {miss}, 跳过')
        return None
    d = _kd_dates(d).dropna(subset=['ann_date', 'end_date'])
    d = _kd_num(d, value_cols)
    # 同一报告期多次公告 (修正/快报) → 取最新 ann_date
    d = d.sort_values(['ts_code', 'end_date', 'ann_date']).drop_duplicates(
        ['ts_code', 'end_date'], keep='last').reset_index(drop=True)
    if ttm:
        yr = d['end_date'].dt.year.values
        for c in value_cols:
            cum = d[c]
            prev = cum.groupby([d['ts_code'].values, yr]).shift(1)
            sq = (cum - prev).where(prev.notna(), cum)          # Q1 无前值 → 用累计
            frame = pd.DataFrame({'g': d['ts_code'].values, 'v': sq.values})
            ttm_v = frame.groupby('g')['v'].transform(
                lambda s: s.rolling(4, min_periods=4).sum())
            d[c + '_ttm'] = ttm_v.values
    return d


def _kd_ord(d):
    """按「公告日 → 报告期」稳定排序。

    同一 (ts_code, ann_date) 可能出现**两条不同报告期**的记录 (实测 920220.BJ
    2024-12-23 同日披露 holder_num=8 与 15)。pivot_table(aggfunc='last') 取的是
    **文件顺序**的最后一行, 语义上不可控。显式排序后 'last' = 最新报告期, 正确。
    kind='mergesort' 保证并列时顺序可复现 (quicksort 不稳定, 会让 seed 截断与
    不截断结果分叉 —— 2026-09-11 实测 6.2 万格数值差异)。
    """
    cols = [c for c in ('ann_date', 'end_date') if c in d.columns]
    return d.sort_values(cols, kind='mergesort') if cols else d


def _kd_event_trim(d, daily_index, window):
    """事件量前置截断: 保留 `window` 个索引位所覆盖的日历跨度即可。

    ✅ 实测与不截断**完全等价** (事件窗口类字段 diff=0): 截断点之前的事件无论
    如何都落在窗口之外, 贡献恒为 0。
    上界取 window*1.7+30 天: union 索引 = 交易日 ∪ 公告日, 250 位最多跨约 365
    自然日, 425 天足够覆盖。
    """
    if d is None or len(daily_index) == 0 or 'ann_date' not in d.columns:
        return d
    lo = daily_index.min() - pd.Timedelta(days=int(window * 1.7) + 30)
    return d[d['ann_date'] >= lo]


def _kd_asof(d, value_cols, daily_index, daily_columns, lag=1):
    """状态量: ann_date asof → 日频宽表 (ffill, lag 交易日后生效)。

    🔴🔴 必须 `union` 而不是 `reindex(daily_index)`: reindex 是**取交集**, 会把
    daily_index 窗口之外的所有 ann_date 行直接丢弃, ffill 便无源可填 →
    字段在窗口内近乎全空 (2026-09-11 实测: balancesheet 2297 个公告日,
    reindex 后仅剩 10 个非空单元格, 全表 0 只有效)。正确顺序:
        union(公告日 ∪ 日频) → ffill → 切回 daily_index → shift(1 交易日)

    ⚠ 不做「每股 seed」前置截断: ffill 会跳过 NaN, 即「取最近一个**有值**的报告」
    而非「最近一期报告」。要复刻这一语义就必须保留在窗口之前的最近非空行
    (逐列不同), 否则长停牌股/不披露该科目的公司 (如银行的 adv_receipts) 会
    由有值变全空 —— 实测 adv_receipts 覆盖率 357773 → 145416。既有
    load_fina_growth_wide 同样是「全历史 ffill」口径, 保持一致。
    """
    if d is None:
        return [None] * len(value_cols)
    d = _kd_ord(d)          # 并列 ann_date 时 'last' = 最新报告期
    outs = []
    for c in value_cols:
        if c not in d.columns:
            outs.append(None)
            continue
        w = d.pivot_table(index='ann_date', columns='ts_code', values=c, aggfunc='last')
        w = w.reindex(w.index.union(daily_index)).ffill()
        w = w.reindex(index=daily_index)
        if lag:
            w = w.shift(lag)          # 在日频网格上 shift = 精确 1 个交易日
        w = w.reindex(columns=daily_columns)
        _asof_coverage_check(c, w, 'kuaidao_asof')
        outs.append(w)
    return outs


def _kd_event_window(d, value_col, daily_index, daily_columns, window, how='sum', lag=1):
    """事件量: 稀疏日频 (不 ffill) → rolling(window).agg。无事件→0 (不是 NaN)。

    绝不可 ffill: 同一次事件会被窗口重复计入。
    🔴 同样用 union: 否则窗口起始段会漏掉窗口外的历史事件, 造成前 window 行低估。
    """
    if d is None or value_col not in d.columns:
        return None
    d = _kd_event_trim(d, daily_index, window)
    dd = d.dropna(subset=['ann_date']).copy()
    piv = dd.pivot_table(index='ann_date', columns='ts_code', values=value_col, aggfunc='sum')
    w = piv.reindex(piv.index.union(daily_index)).fillna(0.0)   # 不 ffill → 仅事件日有值
    r = w.rolling(window, min_periods=1)
    agg = r.sum() if how == 'sum' else (r.max() if how == 'max' else r.mean())
    agg = agg.reindex(index=daily_index)
    if lag:
        agg = agg.shift(lag)
    return agg.reindex(columns=daily_columns).fillna(0.0)


def load_kuaidao_fund_wide(daily_index, daily_columns, lag=1, verbose=True):
    """快刀斩股基本面通道 → dict{字段名: 日频宽表}。

    产出字段 (缺失的键不出现):
      资产负债表 (asof 最新一期, 单位: 元)
        total_assets / kd_money_cap / kd_trad_asset / kd_accounts_receiv /
        kd_notes_receiv / kd_goodwill / kd_st_borr / kd_lt_borr /
        kd_bond_payable / kd_adv_receipts / kd_contract_liab
      利润表 (TTM 与年报口径)
        kd_revenue_ttm / kd_n_income_ttm / kd_revenue_ann / kd_n_income_ann
      现金流量表 (TTM, 收现比/净现比/资本开支比/FCF 用)
        kd_ncf_act_ttm / kd_c_fr_sale_sg_ttm / kd_capex_ttm
      扣非净利润 (TTM, 来自 fina_indicator)
        kd_profit_dedt_ttm
      状态量
        kd_holder_num (股东户数)
        kd_top10_hold_sum (前十大股东合计持股数) / kd_top10_hold_ratio (合计比例%)
      事件窗口累计
        kd_cash_div_12m (近 250 交易日每股分红合计, 元/股)
        kd_holder_net_ratio_3m (近 60 交易日大股东增减持净比例 %)
        kd_holder_in_ratio_3m / kd_holder_de_ratio_3m
        kd_repurchase_flag_3m (近 60 日是否有回购) / kd_repurchase_price_3m (回购均价)
    """
    if not _KD_GAP_DIR.exists():
        if verbose:
            print('  [快刀基本面] kuaidao_gaps 目录不存在, 整通道跳过')
        return {}
    extra = {}

    # ── 1) 资产负债表 (asof) ────────────────────────────────────────────
    bal_cols = ['total_assets', 'money_cap', 'trad_asset', 'accounts_receiv',
                'notes_receiv', 'goodwill', 'st_borr', 'lt_borr', 'bond_payable',
                'adv_receipts', 'contract_liab']
    bal = _kd_quarterly('balancesheet.csv', bal_cols, ttm=False)
    if bal is not None:
        names = ['kd_total_assets', 'kd_money_cap', 'kd_trad_asset', 'kd_accounts_receiv',
                 'kd_notes_receiv', 'kd_goodwill', 'kd_st_borr', 'kd_lt_borr',
                 'kd_bond_payable', 'kd_adv_receipts', 'kd_contract_liab']
        for nm, w in zip(names, _kd_asof(bal, bal_cols, daily_index, daily_columns, lag)):
            if w is not None:
                extra[nm] = w
        if verbose:
            print(f'  [快刀基本面] 资产负债表 {len(bal)} 期, 最新有效 '
                  f'{int(extra.get("kd_total_assets", pd.DataFrame()).iloc[-1].notna().sum()) if "kd_total_assets" in extra else 0} 只')

    # ── 2) 利润表 TTM + 年报 ────────────────────────────────────────────
    inc_cols = ['revenue', 'n_income']
    inc = _kd_quarterly('income.csv', inc_cols, ttm=True)
    if inc is not None:
        for nm, w in zip(['kd_revenue_ttm', 'kd_n_income_ttm'],
                         _kd_asof(inc, ['revenue_ttm', 'n_income_ttm'], daily_index, daily_columns, lag)):
            if w is not None:
                extra[nm] = w
        ann = inc[inc['end_date'].dt.month == 12]
        for nm, w in zip(['kd_revenue_ann', 'kd_n_income_ann'],
                         _kd_asof(ann, inc_cols, daily_index, daily_columns, lag)):
            if w is not None:
                extra[nm] = w
        if verbose:
            v = extra.get('kd_revenue_ttm')
            print(f'  [快刀基本面] 利润表 TTM 已接入 (最新有效 {int(v.iloc[-1].notna().sum()) if v is not None else 0} 只)')

    # ── 3) 现金流量表 TTM ───────────────────────────────────────────────
    cf_cols = ['n_cashflow_act', 'c_pay_acq_const_fiolta', 'c_fr_sale_sg']
    cf = _kd_quarterly('cashflow.csv', cf_cols, ttm=True)
    if cf is not None:
        for nm, w in zip(['kd_ncf_act_ttm', 'kd_capex_ttm', 'kd_c_fr_sale_sg_ttm'],
                         _kd_asof(cf, ['n_cashflow_act_ttm', 'c_pay_acq_const_fiolta_ttm',
                                       'c_fr_sale_sg_ttm'], daily_index, daily_columns, lag)):
            if w is not None:
                extra[nm] = w
        if verbose:
            v = extra.get('kd_c_fr_sale_sg_ttm')
            print(f'  [快刀基本面] 现金流 TTM 已接入 (收现字段有效 '
                  f'{int(v.iloc[-1].notna().sum()) if v is not None else 0} 只)')

    # ── 4) 扣非净利润 TTM (fina_indicator) ──────────────────────────────
    fi_path = DATA_DIR / 'raw' / 'fina_indicator.csv'
    fi = _kd_quarterly(fi_path, ['profit_dedt'], ttm=True) if fi_path.exists() else None
    if fi is not None:
        w = _kd_asof(fi, ['profit_dedt_ttm'], daily_index, daily_columns, lag)[0]
        if w is not None:
            extra['kd_profit_dedt_ttm'] = w
            if verbose:
                print(f'  [快刀基本面] 扣非净利 TTM 已接入 (最新有效 {int(w.iloc[-1].notna().sum())} 只)')

    # ── 5) 股东户数 (状态量) ────────────────────────────────────────────
    hn = _kd_load('stk_holdernumber.csv', ['ts_code', 'ann_date', 'end_date', 'holder_num'])
    if hn is not None:
        hn = _kd_num(_kd_dates(hn).dropna(subset=['ann_date']), ['holder_num'])
        hn = hn.sort_values(['ts_code', 'end_date', 'ann_date']).drop_duplicates(
            ['ts_code', 'end_date'], keep='last').reset_index(drop=True)
        # 同比 (4 期前, 季频层算 —— 日频 rolling(250) 在 n_days=270 面板下几乎无值)
        hn = hn.sort_values(['ts_code', 'end_date'])
        prev4 = hn.groupby('ts_code')['holder_num'].shift(4)
        hn['_yoy'] = (hn['holder_num'] / prev4 - 1.0).where(prev4 > 0)
        for nm, w in zip(['kd_holder_num', 'kd_holder_num_yoy'],
                         _kd_asof(hn, ['holder_num', '_yoy'], daily_index, daily_columns, lag)):
            if w is not None:
                extra[nm] = w
        if verbose:
            v = extra.get('kd_holder_num')
            print(f'  [快刀基本面] 股东户数(+同比)已接入 (最新有效 {int(v.iloc[-1].notna().sum()) if v is not None else 0} 只)')

    # ── 6) 前十大股东合计 (状态量) ──────────────────────────────────────
    t10 = _kd_load('top10_periods.csv', ['ts_code', 'ann_date', 'end_date', 'hold_amount', 'hold_ratio'])
    if t10 is not None:
        t10 = _kd_num(_kd_dates(t10).dropna(subset=['ann_date']), ['hold_amount', 'hold_ratio'])
        agg = t10.groupby(['ts_code', 'end_date', 'ann_date'], as_index=False).agg(
            {'hold_amount': 'sum', 'hold_ratio': 'sum'})
        agg = agg.sort_values(['ts_code', 'end_date', 'ann_date']).drop_duplicates(
            ['ts_code', 'end_date'], keep='last')
        for nm, w in zip(['kd_top10_hold_sum', 'kd_top10_hold_ratio'],
                         _kd_asof(agg, ['hold_amount', 'hold_ratio'], daily_index, daily_columns, lag)):
            if w is not None:
                extra[nm] = w
        if verbose:
            v = extra.get('kd_top10_hold_sum')
            print(f'  [快刀基本面] 前十大合计已接入 (最新有效 {int(v.iloc[-1].notna().sum()) if v is not None else 0} 只, '
                  f'{agg["end_date"].nunique()} 期)')

    # ── 7) 分红 (事件型, 近 250 交易日合计) ─────────────────────────────
    dv = _kd_load('dividend.csv', ['ts_code', 'ann_date', 'end_date', 'div_proc', 'cash_div'])
    if dv is not None:
        dv = _kd_num(_kd_dates(dv).dropna(subset=['ann_date']), ['cash_div'])
        if 'div_proc' in dv.columns:
            keep = dv['div_proc'].astype(str).str.contains('实施', na=False)
            if keep.sum() > 0:
                dv = dv[keep]
        w = _kd_event_window(dv, 'cash_div', daily_index, daily_columns, 250, 'sum', lag)
        if w is not None:
            extra['kd_cash_div_12m'] = w
            if verbose:
                print(f'  [快刀基本面] 分红(近250日)已接入 (最新非零 {int((w.iloc[-1] > 0).sum())} 只)')

    # ── 8) 大股东增减持 (事件型, 近 60 交易日) ──────────────────────────
    ht = _kd_load('stk_holdertrade.csv', ['ts_code', 'ann_date', 'end_date', 'in_de', 'change_ratio'])
    if ht is not None and 'change_ratio' in ht.columns:
        ht = _kd_num(_kd_dates(ht).dropna(subset=['ann_date']), ['change_ratio'])
        ind = ht['in_de'].astype(str).str.upper() if 'in_de' in ht.columns else None
        net = ht.assign(_v=ht['change_ratio'] * (np.where(ind.str.startswith('IN'), 1.0, -1.0) if ind is not None else 1.0))
        for nm, src, how in [('kd_holder_net_ratio_3m', net, 'sum')]:
            w = _kd_event_window(src.rename(columns={'_v': 'v'}), 'v', daily_index, daily_columns, 60, how, lag)
            if w is not None:
                extra[nm] = w
        if ind is not None:
            for nm, sel, sign in [('kd_holder_in_ratio_3m', ind.str.startswith('IN'), 1.0),
                                  ('kd_holder_de_ratio_3m', ind.str.startswith('DE'), 1.0)]:
                sub = ht[sel].copy()
                if len(sub):
                    sub['v'] = sub['change_ratio'] * sign
                    w = _kd_event_window(sub, 'v', daily_index, daily_columns, 60, 'sum', lag)
                    if w is not None:
                        extra[nm] = w
        if verbose:
            v = extra.get('kd_holder_net_ratio_3m')
            print(f'  [快刀基本面] 增减持(近60日)已接入 (最新非零 {int((v.iloc[-1] != 0).sum()) if v is not None else 0} 只)')

    # ── 9) 回购 (事件型) ────────────────────────────────────────────────
    rp = _kd_load('repurchase.csv', ['ts_code', 'ann_date', 'end_date', 'proc', 'vol', 'amount'])
    if rp is not None:
        rp = _kd_num(_kd_dates(rp).dropna(subset=['ann_date']), ['vol', 'amount'])
        rp['_flag'] = 1.0
        w = _kd_event_window(rp, '_flag', daily_index, daily_columns, 60, 'max', lag)
        if w is not None:
            extra['kd_repurchase_flag_3m'] = w
        rp2 = rp.copy()
        rp2['_px'] = rp2['amount'] / rp2['vol'].where(rp2['vol'] > 0)
        rp2 = rp2.dropna(subset=['_px'])
        if len(rp2):
            w = _kd_event_window(rp2, '_px', daily_index, daily_columns, 60, 'max', lag)
            if w is not None:
                extra['kd_repurchase_price_3m'] = w
        # 近 250 交易日最大累计回购金额 (amount 是方案累计值 → 用 max 不重复累加)
        w = _kd_event_window(rp, 'amount', daily_index, daily_columns, 250, 'max', lag)
        if w is not None:
            extra['kd_repurchase_amt_12m'] = w
        if verbose:
            v = extra.get('kd_repurchase_flag_3m')
            print(f'  [快刀基本面] 回购(近60日)已接入 (最新有回购 {int((v.iloc[-1] > 0).sum()) if v is not None else 0} 只)')

    # ── 自检: 状态量字段「末日全空」几乎一定是 asof/reindex 类代码 bug, 不是数据缺失
    #    (数据缺失的特征是部分股票有值; 全 0 只有效 = ffill 无源)。
    #    2026-09-11 实证: reindex 取交集 bug 曾让全部资产负债/利润字段静默全 NaN。
    _empty = []
    for _k, _w in extra.items():
        if isinstance(_w, pd.DataFrame) and len(_w) and int(_w.iloc[-1].notna().sum()) == 0:
            _empty.append(_k)
    if _empty:
        print(f'  [快刀基本面] ⚠️ 以下字段末日全空, 疑为 asof/ffill 逻辑故障 (非数据缺失): {_empty}')
    if verbose:
        print(f'  [快刀基本面] 通道完成: 共 {len(extra)} 个字段')
    return extra
