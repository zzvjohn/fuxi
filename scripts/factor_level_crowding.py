# -*- coding: utf-8 -*-
"""P-20260821-003: 因子级多空波动率拥挤度监控 (纯监控 — 只输出读数, 默认不干预管线)
+ P-20260826-002: comomentum 拥挤度量 (Lou & Polk 2013 简化: 多头池 leave-one-out
  横截面 beta, 影子指标与多空波动率比并列输出, 不干预)

对库内 status ∈ {candidate, jq_done} 因子逐一计算「多头组波动率 / 空头组波动率」比
(华泰口径代理: 每日因子截面 Top-8 / Bottom-8 等权次日收益 → 20日滚动波动率之比),
按 paradigm 族聚合 (族内取最大比), 阈值 1.5 (2026-07 踩踏前达 2.8)。

输出: data/factor_level_crowding.json + stdout 读数。
软降权: SOFT_DOWNGRADE_ENABLED=False 默认关闭; 开启后仅在 JSON 写 soft_downgrade_flags
         (供因子池注入层参考), 不写回 library_orthogonality_state.json。

用法: python scripts/factor_level_crowding.py [--days 120] [--with-moneyflow]
"""
import sys
import json
import re
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
OUT_FILE = DATA_DIR / 'factor_level_crowding.json'
DAILY_PRICES = DATA_DIR / 'raw' / 'daily_prices.csv'
LIB_STATE = DATA_DIR / 'library_orthogonality_state.json'

TOP_K = 8           # 多空组合个股数
VOL_WINDOW = 20     # 波动率滚动窗口
TH_LS_RATIO = 1.5   # 多空波动率比告警阈值
STATUSES = ('candidate', 'jq_done')
SOFT_DOWNGRADE_ENABLED = False   # P-20260821-003 rollback: 默认纯监控零干预

_ROLL_RANK = re.compile(r'\.rolling\((\d+)\)\.rank\(pct\s*=\s*True\)')
_PLAIN_RANK = re.compile(r'\.rank\(pct\s*=\s*True\)')


def fix_rank_axis(formula: str) -> str:
    """v0.9.5 三段式 rank 修正: 保护 rolling(N).rank, 裸 rank → axis=1"""
    holders = []
    s = _ROLL_RANK.sub(
        lambda m: holders.append(m.group(0)) or f'.rolling({m.group(1)}).__RRK{len(holders)-1}__',
        formula)
    s = _PLAIN_RANK.sub('.rank(pct=True, axis=1)', s)
    for i, _ in enumerate(holders):
        s = s.replace(f'__RRK{i}__', 'rank(pct=True)')
    return s


def load_price_panels(days: int):
    """近 N 自然日日线 → 五变量 + amount 宽表"""
    df = pd.read_csv(
        DAILY_PRICES,
        dtype={'ts_code': str, 'trade_date': str},
        usecols=['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount'],
    )
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')
    end = df['trade_date'].max()
    df = df[(df['trade_date'] >= end - timedelta(days=days)) & (df['trade_date'] <= end)]
    df = df.drop_duplicates(subset=['ts_code', 'trade_date'])
    cols = {'close': 'close_p', 'open': 'open_p', 'high': 'high_p',
            'low': 'low_p', 'vol': 'volume_p', 'amount': 'amount_p'}
    panels = {}
    for src, dst in cols.items():
        panels[dst] = df.pivot(index='trade_date', columns='ts_code', values=src)
    return panels


def load_moneyflow(panels: dict):
    """资金流字段 (可选): buy_lg_vol/sell_lg_vol/buy_sm_vol/sell_sm_vol/buy_elg_vol/sell_elg_vol"""
    try:
        sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
        from data_loader_ext import load_moneyflow_wide
        daily_index = panels['close_p'].index
        daily_cols = list(panels['close_p'].columns)
        mf = load_moneyflow_wide(daily_index, daily_cols)
        if mf[0] is None:
            return {}
        return {
            'buy_sm_vol': mf[0], 'sell_sm_vol': mf[1],
            'buy_lg_vol': mf[2], 'sell_lg_vol': mf[3],
            'buy_elg_vol': mf[4], 'sell_elg_vol': mf[5],
            'net_mf_vol': mf[6],
        }
    except Exception as e:
        print(f'  [资金流] 加载失败: {e}')
        return {}


def eval_expression(expr: str, panels: dict, extra: dict):
    """求值库表达式 (支持裸名/_p 后缀/多行 import 块)"""
    local_ns = {
        'close_p': panels['close_p'], 'open_p': panels['open_p'], 'high_p': panels['high_p'],
        'low_p': panels['low_p'], 'volume_p': panels['volume_p'], 'amount_p': panels.get('amount_p'),
        'close': panels['close_p'], 'open': panels['open_p'], 'open_': panels['open_p'],
        'high': panels['high_p'], 'low': panels['low_p'], 'volume': panels['volume_p'],
        'amount': panels.get('amount_p'),
        'np': np, 'pd': pd,
        **{k: v for k, v in extra.items() if v is not None},
    }
    clean = fix_rank_axis(expr)
    try:
        out = eval(compile(clean, '<factor>', 'eval'), local_ns)
    except SyntaxError:
        exec(compile(clean, '<factor>', 'exec'), local_ns)
        out = local_ns.get('factor') or local_ns.get('result')
        if out is None:
            raise ValueError('公式未定义 factor/result 变量')
    if not isinstance(out, pd.DataFrame):
        raise ValueError(f'求值结果非 DataFrame: {type(out).__name__}')
    return out


def ls_vol_ratio(factor_df: pd.DataFrame, ret_df: pd.DataFrame):
    """每日截面 Top-8/Bottom-8 → 次日等权收益 → 20日滚动波动率比 (返回最后一日读数)"""
    series, n = ls_vol_ratio_series(factor_df, ret_df)
    if series is None:
        return np.nan, n
    return float(series.iloc[-1]), n


def _daily_ls_returns(factor_df: pd.DataFrame, ret_df: pd.DataFrame):
    """P-20260827-003: 共享的多空组合日度收益序列生成器

    每日截面 Top-8/Bottom-8 → 次日等权收益 → (long_ret, short_ret, dates)。
    数据不足返回 (None, None, n_obs)。供 ls_vol_ratio_series / ls_longrun_return 复用
    (避免每指标重复循环, 原 ls_vol_ratio_series 行为不变)。
    """
    long_ret, short_ret, dates = [], [], []
    for i in range(len(factor_df) - 1):
        fv = factor_df.iloc[i].dropna()
        if len(fv) < 40:
            continue
        top = fv.nlargest(TOP_K).index
        bot = fv.nsmallest(TOP_K).index
        r_next = ret_df.iloc[i + 1]
        lr = r_next.reindex(top).mean()
        sr = r_next.reindex(bot).mean()
        if np.isnan(lr) or np.isnan(sr):
            continue
        long_ret.append(lr)
        short_ret.append(sr)
        dates.append(r_next.name)
    return long_ret, short_ret, dates


def ls_vol_ratio_series(factor_df: pd.DataFrame, ret_df: pd.DataFrame):
    """P-20260826-004: 日度多空波动率比序列 (事件回放用)

    返回 (ratio_series, n_obs); 数据不足返回 (None, n_obs)。
    ratio_series 索引 = ret_df 交易日 (从第 1 个有效截面日起)。
    """
    long_ret, short_ret, dates = _daily_ls_returns(factor_df, ret_df)
    if len(long_ret) < VOL_WINDOW:
        return None, len(long_ret)
    lv = pd.Series(long_ret, index=dates).rolling(VOL_WINDOW).std()
    sv = pd.Series(short_ret, index=dates).rolling(VOL_WINDOW).std()
    ratio = lv / sv.replace(0, np.nan)
    ratio = ratio[ratio.notna()]
    if ratio.empty:
        return None, len(long_ret)
    return ratio, len(long_ret)


LONGRUN_WINDOW = 250      # P-20260827-003: 长期收益反转回看窗口 (约12个月)
LONGRUN_MINP = 60         # 最少有效观测 (不足则退回全期累计)


def ls_longrun_return(factor_df: pd.DataFrame, ret_df: pd.DataFrame):
    """P-20260827-003: 长期收益反转维度 — 多空组合过去 12 个月累计收益

    国泰海通拥挤度四指标之「长期收益反转」落地:
      因子多空组合 (Top-8/Bottom-8) 日度收益 → rolling(250) 累计 → 最后读数。
    涨太久的因子族 (过去一年累计正收益过高) = 资金追涨聚集 = 拥挤候选,
    与 ls_vol_ratio (波动) / comomentum (同动) 构成三维族级拥挤分。

    返回 (longrun_12m, n_obs); 数据不足返回 (nan, n_obs)。
    """
    long_ret, short_ret, dates = _daily_ls_returns(factor_df, ret_df)
    if len(long_ret) < LONGRUN_MINP:
        return np.nan, len(long_ret)
    ls = pd.Series(np.asarray(long_ret) - np.asarray(short_ret), index=dates)
    cum = ls.rolling(LONGRUN_WINDOW, min_periods=LONGRUN_MINP).sum()
    if cum.notna().any():
        return float(cum.iloc[-1]) if not np.isnan(cum.iloc[-1]) else float(cum.dropna().iloc[-1]), len(long_ret)
    return np.nan, len(long_ret)


def ls_comomentum(factor_df: pd.DataFrame, ret_df: pd.DataFrame):
    """P-20260826-002: 池成员对池收益的时间序列 beta (Lou-Polk comomentum 简化)

    方法 (2026-08-26 修正, 横截面 leave-one-out 单日回归恒为 -(K-1) 无信息量):
      pool_ret_t = 当日 Top-8 池的同期等权平均收益 (时间序列);
      每只股票 i 全期收益对 pool_ret 回归 → beta_i (对共同冲击的暴露);
      每日 comomentum_t = 当日池成员的 beta 均值 → 近 20 日均值/离散度。
    池成员 beta 越高 = 个股收益越被池共同驱动力主导 = 同涨同跌 = 套利资金聚集 = 拥挤。
    """
    pool_dates, pool_vals = [], []
    for i in range(len(factor_df) - 1):
        fv = factor_df.iloc[i].dropna()
        if len(fv) < 40:
            continue
        top = fv.nlargest(TOP_K).index
        r_same = ret_df.iloc[i].reindex(top)   # 同期收益 (拥挤诊断, 非预测信号)
        if r_same.notna().sum() < 4:
            continue
        pool_dates.append(ret_df.index[i])
        pool_vals.append(r_same.mean())
    if len(pool_vals) < VOL_WINDOW:
        return np.nan, np.nan, len(pool_vals)
    pool = pd.Series(pool_vals, index=pool_dates)
    R = ret_df.loc[pool.index]                  # T'×N 个股收益
    p = pool.values.astype(float)
    p_dm = p - p.mean()
    denom = float((p_dm ** 2).sum())
    if denom < 1e-12:
        return np.nan, np.nan, len(pool_vals)
    Rv = R.values.astype(float)
    valid_cols = np.isfinite(Rv).any(axis=0)
    r_mean = np.zeros(Rv.shape[1])
    cov = np.full(Rv.shape[1], np.nan)
    if valid_cols.any():
        with np.errstate(invalid='ignore'):
            r_mean[valid_cols] = np.nanmean(Rv[:, valid_cols], axis=0)
            cov[valid_cols] = np.nanmean(
                (Rv[:, valid_cols] - r_mean[None, valid_cols]) * p_dm[:, None], axis=0)
    betas = cov / denom
    beta_series = pd.Series(betas, index=ret_df.columns)
    # 每日池成员的 beta 均值 → comomentum 序列
    cm_vals = []
    for i in range(len(factor_df) - 1):
        fv = factor_df.iloc[i].dropna()
        if len(fv) < 40:
            continue
        top = fv.nlargest(TOP_K).index
        bm = beta_series.reindex(top).dropna()
        if len(bm) >= 4:
            cm_vals.append(float(bm.mean()))
    if len(cm_vals) < VOL_WINDOW:
        return np.nan, np.nan, len(cm_vals)
    cm = pd.Series(cm_vals)
    return (float(cm.rolling(VOL_WINDOW).mean().iloc[-1]),
            float(cm.rolling(VOL_WINDOW).std().iloc[-1]), len(cm))


def main():
    ap = argparse.ArgumentParser(description='因子级拥挤度监控 (纯监控)')
    ap.add_argument('--days', type=int, default=120, help='回溯自然日数')
    ap.add_argument('--with-moneyflow', action='store_true', help='加载资金流字段 (慢)')
    args = ap.parse_args()

    if not (DAILY_PRICES.exists() and LIB_STATE.exists()):
        print('[因子拥挤] 数据文件缺失, 跳过')
        sys.exit(0)

    with open(LIB_STATE, 'r', encoding='utf-8') as f:
        lib = json.load(f)
    sels = [f for f in lib.get('factors', []) if f.get('status') in STATUSES]
    if not sels:
        print('[因子拥挤] 无 candidate/jq_done 因子, 跳过')
        sys.exit(0)

    panels = load_price_panels(args.days)
    last_date = panels['close_p'].index.max().strftime('%Y-%m-%d')
    print(f'[因子拥挤监控] {len(sels)} 个候选因子, 数据截止 {last_date}')

    extra = load_moneyflow(panels) if args.with_moneyflow else {}
    ret_df = panels['close_p'].pct_change(fill_method=None)

    rows = []
    for f in sels:
        name = f.get('name', '?')
        expr = str(f.get('expression', ''))
        try:
            fv = eval_expression(expr, panels, extra)
            ratio, n = ls_vol_ratio(fv, ret_df)
            b_mean, b_disp, n_cm = ls_comomentum(fv, ret_df)
            lr12, n_lr = ls_longrun_return(fv, ret_df)  # P-20260827-003
            rows.append({
                'name': name,
                'paradigm': f.get('paradigm', ''),
                'status': f.get('status', ''),
                'ls_vol_ratio': None if np.isnan(ratio) else round(ratio, 3),
                'comomentum_beta': None if np.isnan(b_mean) else round(b_mean, 3),
                'comomentum_disp': None if np.isnan(b_disp) else round(b_disp, 3),
                'longrun_return_12m': None if np.isnan(lr12) else round(lr12, 4),
                'n_obs': n,
            })
        except Exception as e:
            rows.append({
                'name': name, 'paradigm': f.get('paradigm', ''),
                'status': f.get('status', ''), 'ls_vol_ratio': None,
                'comomentum_beta': None, 'comomentum_disp': None,
                'longrun_return_12m': None,
                'error': str(e)[:120],
            })

    # 族级聚合 (族内取最大比)
    fam = {}
    for r in rows:
        if r['ls_vol_ratio'] is None:
            continue
        p = r['paradigm'] or '未标注'
        fam[p] = max(fam.get(p, 0.0), r['ls_vol_ratio'])

    # P-20260826-002: 族级 comomentum (族内取最大 beta; 影子指标不干预)
    fam_cm = {}
    for r in rows:
        if r['comomentum_beta'] is None:
            continue
        p = r['paradigm'] or '未标注'
        fam_cm[p] = max(fam_cm.get(p, 0.0), r['comomentum_beta'])

    # P-20260827-003: 长期收益反转维度 — 族内最大 12m 累计多空收益 + 全库横截面分位
    fam_lr = {}
    lr_vals = [r['longrun_return_12m'] for r in rows
               if r['longrun_return_12m'] is not None]
    if lr_vals:
        from scipy.stats import rankdata
        _lr_sorted = sorted(lr_vals)
        _rank_of = {v: rk for v, rk in
                    zip(_lr_sorted, rankdata(_lr_sorted, method='average') / len(_lr_sorted))}
        fam_lr_pct_acc = {}
        fam_lr_cnt = {}
        for r in rows:
            v = r['longrun_return_12m']
            if v is None:
                continue
            p = r['paradigm'] or '未标注'
            fam_lr[p] = max(fam_lr.get(p, float('-inf')), v)
            fam_lr_pct_acc[p] = fam_lr_pct_acc.get(p, 0.0) + _rank_of[v]
            fam_lr_cnt[p] = fam_lr_cnt.get(p, 0) + 1
        fam_lr_pct = {p: round(acc / fam_lr_cnt[p], 3)
                      for p, acc in fam_lr_pct_acc.items()}
    else:
        fam_lr, fam_lr_pct = {}, {}

    alerts = []
    for p, ratio in sorted(fam.items(), key=lambda x: -x[1]):
        flag = '  ⚠️ 拥挤' if ratio > TH_LS_RATIO else ''
        cm = fam_cm.get(p)
        cm_s = f' | comomentum={cm:.2f}' if cm is not None else ' | comomentum=NA'
        lr = fam_lr.get(p)
        lr_s = f' | 12m累计={lr*100:+.1f}%' if lr is not None and lr != float('-inf') else ' | 12m累计=NA'
        print(f'  族[{p}]: 多空波动率比 = {ratio:.2f} (n因子≥1){cm_s}{lr_s}{flag}')
        if ratio > TH_LS_RATIO:
            alerts.append(p)

    # P-20260827-003: 长期收益反转观察读数 (仅打印, 不触发告警)
    if fam_lr:
        lr_top = sorted(((p, v, fam_lr_pct.get(p, 0.0))
                         for p, v in fam_lr.items() if v != float('-inf')),
                        key=lambda x: -x[1])[:5]
        print('  [longrun] 12月累计多空收益 Top-5 (观察, 不干预): ' +
              ', '.join(f'{p}={v*100:+.1f}%(分位{pct:.2f})'
                        for p, v, pct in lr_top))
        high_lr = [p for p, v, pct in lr_top if pct >= 0.9]
        if high_lr:
            print('  [longrun] ⚠️ 高历史收益族 (全库分位≥0.9): ' +
                  ' / '.join(high_lr) + ' — 观察是否拥挤踩踏候选')

    # P-20260826-002: comomentum 影子读数 (仅打印, 不触发告警)
    cm_flagged = {p: b for p, b in fam_cm.items() if b > 1.25}
    if cm_flagged:
        top_cm = sorted(cm_flagged.items(), key=lambda x: -x[1])[:3]
        print(f'  [comomentum] 高同涨同跌族 (beta>1.25, 影子观察): ' +
              ', '.join(f'{p}={b:.2f}' for p, b in top_cm))

    # P-20260831-004: 两把尺子分歧对比 (多空波动率比 vs comomentum, 影子)
    # 双高=拥挤证据相互印证; 分歧=单一指标可能失真 (风格切换/度量口径漂移)
    CM_TH = 1.25
    n_agree = n_diverge = 0
    diverge_pairs = []
    for p, ratio in fam.items():
        cm = fam_cm.get(p)
        if cm is None:
            continue
        ls_hot = ratio > TH_LS_RATIO
        cm_hot = cm > CM_TH
        if ls_hot == cm_hot:
            n_agree += 1
        else:
            n_diverge += 1
            diverge_pairs.append((p, ratio, cm))
    if diverge_pairs:
        diverge_pairs.sort(key=lambda x: -abs(x[1] - x[2]))
        ds = ', '.join(f'{p}(LS={r:.2f}/CM={c:.2f})' for p, r, c in diverge_pairs[:5])
        print(f'  [两把尺子] 一致={n_agree}族 / 分歧={n_diverge}族: {ds}')
        print(f'  [两把尺子] ⚠️ 分歧族提示: 单指标拥挤可能失真, 观察连续出现次数')
    else:
        print(f'  [两把尺子] 一致={n_agree}族 / 分歧=0族 (两指标同向, 拥挤读数可信度较高)')

    soft_flags = alerts if SOFT_DOWNGRADE_ENABLED else []
    out = {
        'date': last_date,
        'generated_at': datetime.now().isoformat(),
        'factors': rows,
        'family_max_ratio': fam,
        'family_comomentum': fam_cm,
        'comomentum_note': 'P-20260826-002: Lou-Polk leave-one-out 横截面 beta 均值(20日), 影子指标不干预',
        'family_longrun_return': fam_lr,
        'family_longrun_pct': fam_lr_pct,
        'longrun_note': 'P-20260827-003: 族内因子多空组合过去250日累计收益(族内max)及全库横截面分位, 观察指标不干预',
        'threshold': TH_LS_RATIO,
        'alerts': alerts,
        'soft_downgrade_flags': soft_flags,
        'soft_downgrade_applied': SOFT_DOWNGRADE_ENABLED,
        'metric_divergence': {
            'note': 'P-20260831-004: 多空波动率比(>1.5) vs comomentum beta(>1.25) 两把尺子分歧统计',
            'n_agree': n_agree,
            'n_diverge': n_diverge,
            'diverge_families': [{'paradigm': p, 'ls_vol_ratio': r, 'comomentum_beta': c}
                                 for p, r, c in diverge_pairs],
        },
        'note': '纯监控指标, 软降权默认关闭 (P-20260821-003)',
    }
    OUT_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'  → {OUT_FILE}')
    if alerts:
        print(f'  🔥 拥挤族: {" / ".join(alerts)} — 仅提醒, 软降权未启用')


if __name__ == '__main__':
    main()
