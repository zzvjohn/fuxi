# -*- coding: utf-8 -*-
"""P-20260914-001: Spread-Compression 拥挤诊断 (纯影子 — 只输出读数与标签, 零裁决)

方法论 (QuanterLab FM103 Crowdedness 口径):
  对每个因子每期计算 top-vs-bottom 分位组合的远期收益价差 St,
  对时间做 OLS 回归 St = α + β·t:
    β < 0 且 R² >= 0.20  → 'crowding'  (价差被系统性侵蚀 = 拥挤)
    β < 0 且 R² >= 0.50  → 'fast_decay' (快速衰减子类)
  该口径与 factor_level_crowding.py (corr 同质化口径) 互补:
  corr 度量同质化, spread compression 度量拥挤对多空价差的侵蚀。
  独立佐证: McLean & Pontiff (2016, JF) 97 因子发表后截面收益衰减 ~58%。

范围: data/library_orthogonality_state.json 中 status ∈ {candidate, jq_done}
      且 expression 可在 OHLCV 面板上求值的头部因子 (默认按 |icir| 取 Top-20;
      引用 moneyflow/kuaidao 等扩展字段的公式自动跳过计 eval_fail)。

输出: data/factor_spread_compression.json + stdout 读数行
      [SpreadComp] 评估 N → crowding X / fast_decay Y / healthy Z / eval_fail W

用法:
  python scripts/factor_spread_compression.py [--days 250] [--top 20]
                                              [--fwd 5] [--r2-thresh 0.20]
                                              [--names f1,f2,...]
回滚: 纯影子脚本, 删除本文件即完全移除; 不写回任何库状态。
"""
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
DAILY_PRICES = DATA_DIR / 'raw' / 'daily_prices.csv'
LIB_STATE = DATA_DIR / 'library_orthogonality_state.json'
OUT_FILE = DATA_DIR / 'factor_spread_compression.json'

STATUSES = ('candidate', 'jq_done')


def load_price_panels(days: int, fwd: int):
    """近 (days + fwd 缓冲) 自然日日线 → OHLCV 宽表 (公式求值 + 远期收益)"""
    df = pd.read_csv(
        DAILY_PRICES,
        dtype={'ts_code': str, 'trade_date': str},
        usecols=['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol'],
    )
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')
    end = df['trade_date'].max()
    df = df[(df['trade_date'] >= end - pd.Timedelta(days=days + fwd * 4)) & (df['trade_date'] <= end)]
    df = df.drop_duplicates(subset=['ts_code', 'trade_date'])
    close = df.pivot(index='trade_date', columns='ts_code', values='close').sort_index()
    panels = {
        'close_p': close,
        'open_p': df.pivot(index='trade_date', columns='ts_code', values='open').sort_index(),
        'high_p': df.pivot(index='trade_date', columns='ts_code', values='high').sort_index(),
        'low_p': df.pivot(index='trade_date', columns='ts_code', values='low').sort_index(),
        'volume_p': df.pivot(index='trade_date', columns='ts_code', values='vol').sort_index(),
    }
    # 远期 fwd 日收益 (诊断用途, 允许 forward — 不参与任何裁决)
    fwd_ret = close.shift(-fwd) / close - 1
    return panels, fwd_ret


def eval_formula(expression: str, panels: dict):
    """在 OHLCV 宽表命名空间求值因子公式 (对齐 daily_factor_hypothesis.evaluate_factor 语义)"""
    local_ns = {
        'close_p': panels['close_p'], 'open_p': panels['open_p'],
        'high_p': panels['high_p'], 'low_p': panels['low_p'],
        'volume_p': panels['volume_p'], 'np': np, 'pd': pd,
        'close': panels['close_p'], 'open': panels['open_p'], 'open_': panels['open_p'],
        'high': panels['high_p'], 'low': panels['low_p'], 'volume': panels['volume_p'],
    }
    try:
        return eval(compile(expression, '<factor>', 'eval'), dict(local_ns))  # noqa: S307
    except SyntaxError:
        exec(compile(expression, '<factor>', 'exec'), dict(local_ns))  # noqa: S102
        out = local_ns.get('factor', local_ns.get('result'))
        if out is None:
            raise ValueError("公式未定义 factor 或 result 变量")
        return out


def spread_series(factor: pd.DataFrame, fwd_ret: pd.DataFrame,
                  rebalance: int = 5, top_q: float = 0.2):
    """每 rebalance 日: top/bottom 分位等权远期收益价差序列"""
    dates = factor.index
    out = []
    for i in range(0, len(dates) - 1, rebalance):
        t = dates[i]
        row = factor.loc[t].dropna()
        if len(row) < 50:
            continue
        n_g = max(int(len(row) * top_q), 5)
        ranked = row.rank(ascending=False)          # 因子值大 → rank 1
        top_ids = ranked[ranked <= n_g].index
        bot_ids = ranked[ranked > len(row) - n_g].index
        fr = fwd_ret.loc[t]
        s_top = fr.reindex(top_ids).mean()
        s_bot = fr.reindex(bot_ids).mean()
        if pd.notna(s_top) and pd.notna(s_bot):
            out.append((t, s_top - s_bot))
    return pd.Series([v for _, v in out], index=[t for t, _ in out])


def ols_trend(spread: pd.Series):
    """价差对时间 OLS: 返回 (beta, r2, n_obs); n<10 不足以判别"""
    s = spread.dropna()
    n = len(s)
    if n < 10:
        return None, None, n
    x = np.arange(n, dtype=float)
    y = s.values.astype(float)
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    if sxx == 0:
        return None, None, n
    beta = ((x - xm) * (y - ym)).sum() / sxx
    y_hat = ym + beta * (x - xm)
    ss_res = ((y - y_hat) ** 2).sum()
    ss_tot = ((y - ym) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(beta), float(r2), n


def main():
    ap = argparse.ArgumentParser(description='P-20260914-001 Spread-Compression 拥挤诊断 (影子)')
    ap.add_argument('--days', type=int, default=250, help='回归窗口(自然日)')
    ap.add_argument('--top', type=int, default=20, help='按 |icir| 取头部因子数')
    ap.add_argument('--fwd', type=int, default=5, help='远期收益天数')
    ap.add_argument('--top-q', dest='top_q', type=float, default=0.2, help='多空分位')
    ap.add_argument('--r2-thresh', dest='r2_thresh', type=float, default=0.20, help='拥挤 R² 门槛')
    ap.add_argument('--names', type=str, default='', help='逗号分隔显式因子名 (优先于 --top)')
    args = ap.parse_args()

    if not LIB_STATE.exists():
        print(f"[SpreadComp] 库状态不存在: {LIB_STATE} → 跳过")
        return 1
    state = json.loads(LIB_STATE.read_text(encoding='utf-8'))
    factors = [f for f in state.get('factors', [])
               if f.get('status') in STATUSES and f.get('expression')]
    if args.names:
        want = [n.strip() for n in args.names.split(',') if n.strip()]
        factors = [f for f in factors if f['name'] in want]
    else:
        factors.sort(key=lambda f: abs(float(f.get('icir') or 0)), reverse=True)
        factors = factors[:args.top]

    if not factors:
        print("[SpreadComp] 无候选因子 → 跳过")
        return 0

    panels, fwd_ret = load_price_panels(args.days, args.fwd)
    results, n_fail = [], 0
    for f in factors:
        rec = {'name': f['name'], 'paradigm': f.get('paradigm', ''),
               'icir': f.get('icir'), 'status': f.get('status')}
        try:
            factor = eval_formula(f['expression'], panels)
            if not isinstance(factor, pd.DataFrame):
                raise ValueError("公式结果非 DataFrame")
            spread = spread_series(factor, fwd_ret, top_q=args.top_q)
            beta, r2, n_obs = ols_trend(spread)
            if beta is None:
                rec.update(label='insufficient', n_obs=n_obs)
            else:
                if beta < 0 and r2 >= 0.50:
                    label = 'fast_decay'
                elif beta < 0 and r2 >= args.r2_thresh:
                    label = 'crowding'
                else:
                    label = 'healthy'
                rec.update(label=label, beta=round(beta, 5), r2=round(r2, 4), n_obs=n_obs)
        except Exception as e:  # noqa: BLE001 — 扩展字段公式求值失败属预期
            n_fail += 1
            rec.update(label='eval_fail', reason=str(e)[:120])
        results.append(rec)

    summary = {}
    for rec in results:
        summary[rec['label']] = summary.get(rec['label'], 0) + 1
    out = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'proposal': 'P-20260914-001',
        'mode': 'shadow',
        'params': {'days': args.days, 'top': args.top, 'fwd': args.fwd,
                   'top_q': args.top_q, 'r2_thresh': args.r2_thresh},
        'summary': summary,
        'results': results,
    }
    OUT_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    n = len(results)
    print(f"[SpreadComp] 评估 {n} 因子 → crowding {summary.get('crowding', 0)} / "
          f"fast_decay {summary.get('fast_decay', 0)} / healthy {summary.get('healthy', 0)} / "
          f"insufficient {summary.get('insufficient', 0)} / eval_fail {summary.get('eval_fail', 0)}"
          f" (输出: {OUT_FILE.name})")
    for rec in results:
        if rec['label'] in ('crowding', 'fast_decay'):
            print(f"  ⚠️ {rec['label']:10s} {rec['name'][:40]:42s} β={rec.get('beta')} R²={rec.get('r2')}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
