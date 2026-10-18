# -*- coding: utf-8 -*-
"""P-20260906-004: 发现系统滚动重执行评估 (AEAP arXiv:2609.00731 移植, 影子)。

评估发现系统(Stage2 假设检验裁决层)本身的可靠性, 而非单次产出:
同一组固定候选因子在 3 个滚动数据窗口上重跑裁决, 比较通过因子集的
稳定性 (Jaccard 相似度) 与方向一致性。发现过程可靠 ⇒ 通过集在相邻
窗口高度重叠; 重叠低 ⇒ 裁决对数据区间敏感 (疑似过拟合到特定窗口)。

零干预: 只输出 data/discovery_reliability.json, 不触碰流水线状态。
"""
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = PROJECT_ROOT / 'data' / 'discovery_reliability.json'

N_FACTORS = 15
WIN_LEN = 180        # 每窗口交易日数
WIN_SHIFT = 60       # 窗口起点错开交易日数
N_WINDOWS = 3
FWD = 5
IC_MIN = 0.02
ICIR_MIN = 0.10

# 公式中引用这些 extra 字段的因子在纯 OHLCV 滚动评估中不可执行, 过滤掉
_EXTRA_KEYS = [
    'roe', 'eps', 'eps_qoq', 'buy_sm_vol', 'sell_sm_vol', 'buy_lg_vol',
    'sell_lg_vol', 'buy_elg_vol', 'sell_elg_vol', 'net_mf_amount', 'rzye',
    'rqye', 'rzmre', 'rqyl', 'rqchl', 'lhb_', 'eps_disp', 'tp_disp',
    'rating_disp', 'n_cover', 'netprofit_yoy', 'roe_dt', 'tr_yoy', 'circ_mv',
    'industry_l1', 'industry_ret_peer', 'ind_avg_', 'limit_up_',
    'moneyflow', 'margin', 'north',
]


def pick_candidates():
    pool = pd.read_csv(PROJECT_ROOT / 'data' / 'unified_factor_pool.csv')
    pool = pool[pool['formula'].notna() & (pool['formula'].astype(str).str.len() > 5)]
    pool = pool[pool['daily_icir'].notna()]
    pool = pool.sort_values('daily_icir', ascending=False)
    cands = []
    for _, row in pool.iterrows():
        formula = str(row['formula'])
        if any(k in formula for k in _EXTRA_KEYS):
            continue
        cands.append({'name': row['name'], 'formula': formula,
                      'daily_icir': row['daily_icir']})
        if len(cands) >= N_FACTORS:
            break
    return cands


def load_wide():
    daily = pd.read_csv(
        PROJECT_ROOT / 'data' / 'raw' / 'daily_prices.csv',
        dtype={'ts_code': str, 'trade_date': str},
        usecols=['ts_code', 'trade_date', 'open', 'high', 'low', 'close',
                 'vol', 'amount'],
    )
    daily['trade_date'] = pd.to_datetime(daily['trade_date'], format='mixed')
    daily = daily.drop_duplicates(subset=['ts_code', 'trade_date'], keep='last')
    daily = daily.sort_values(['ts_code', 'trade_date'])
    pivots = {}
    for col, name in [('open', 'open'), ('high', 'high'), ('low', 'low'),
                      ('close', 'close'), ('vol', 'volume'), ('amount', 'amount')]:
        pivots[name] = daily.pivot(index='trade_date', columns='ts_code',
                                   values=col)
    return pivots


def fix_rank_semantics(formula):
    """与流水线一致的 rank 语义修正: 裸 rank(pct=True) → axis=1, rolling 保持。"""
    roll_re = re.compile(r'(\.rolling\([^)]*\))\.rank\(pct\s*=\s*True\)')
    plain_re = re.compile(r'\.rank\(pct\s*=\s*True\)')
    holders = []
    f = roll_re.sub(lambda m: holders.append(m.group(0)) or
                    f'{m.group(1)}.__RRK{len(holders)-1}__', formula)
    f = plain_re.sub('.rank(pct=True, axis=1)', f)
    for i, h in enumerate(holders):
        f = f.replace(f'__RRK{i}__', 'rank(pct=True)')
    return f


def evaluate_on_window(formula, wide, window):
    """在窗口宽表切片上 eval 公式并计算 fwd5 截面 IC/ICIR。"""
    sub = {k: v.iloc[window[0]:window[1]] for k, v in wide.items()}
    close = sub['close']
    fwd5 = close.shift(-FWD) / close - 1

    local_ns = {
        'close_p': close, 'open_p': sub['open'], 'high_p': sub['high'],
        'low_p': sub['low'], 'volume_p': sub['volume'],
        'close': close, 'open_': sub['open'], 'open': sub['open'],
        'high': sub['high'], 'low': sub['low'], 'volume': sub['volume'],
        'amount': sub['amount'], 'np': np, 'pd': pd,
    }
    fixed = fix_rank_semantics(formula)
    try:
        factor_raw = eval(compile(fixed, '<factor>', 'eval'), local_ns)
    except SyntaxError:
        exec(compile(fixed, '<factor>', 'exec'), local_ns)
        factor_raw = local_ns.get('factor')
    except Exception:
        return None
    if factor_raw is None:
        return None
    if isinstance(factor_raw, pd.Series):
        factor_raw = factor_raw.to_frame().T.reindex(index=close.index)
    if not isinstance(factor_raw, pd.DataFrame):
        return None
    factor_raw = factor_raw.reindex(index=close.index, columns=close.columns)

    ics = []
    for date in close.index:
        fv = factor_raw.loc[date]
        fw = fwd5.loc[date]
        common = fv.dropna().index.intersection(fw.dropna().index)
        if len(common) < 30:
            continue
        ic = fv[common].rank().corr(fw[common].rank())
        if not np.isnan(ic):
            ics.append(ic)
    if len(ics) < 30:
        return None
    ic_mean = float(np.mean(ics))
    icir = float(ic_mean / np.std(ics)) if np.std(ics) > 1e-12 else 0.0
    return {
        'ic': round(ic_mean, 5),
        'icir': round(icir, 4),
        'n_days': len(ics),
        'verdict': 'PASS' if (abs(ic_mean) >= IC_MIN and icir >= ICIR_MIN)
                   else 'FAIL',
        'sign': 'pos' if ic_mean > 0 else 'neg',
    }


def jaccard(s1, s2):
    if not s1 and not s2:
        return 1.0
    return len(s1 & s2) / len(s1 | s2)


def main():
    print('=' * 70)
    print('  发现系统滚动重执行评估 (P-20260906-004, 影子)')
    print('=' * 70)
    cands = pick_candidates()
    print(f'  固定候选集 {len(cands)} 个 (纯 OHLCV 公式):')
    for c in cands:
        print(f"    - {c['name']} (pool ICIR={c['daily_icir']})")

    wide = load_wide()
    n_rows = len(wide['close'])
    print(f'  全量面板 {n_rows} 交易日')
    windows = [(WIN_SHIFT * w, WIN_SHIFT * w + WIN_LEN) for w in range(N_WINDOWS)]
    for i, (s, e) in enumerate(windows):
        d0, d1 = wide['close'].index[s].date(), wide['close'].index[e - 1].date()
        print(f'  W{i+1}: [{d0} ~ {d1}] ({e - s} 交易日)')

    results = []
    for ci, c in enumerate(cands):
        print(f"  [{ci+1}/{len(cands)}] {c['name']}")
        row = {'factor': c['name'], 'windows': {}}
        for wi, (s, e) in enumerate(windows):
            r = evaluate_on_window(c['formula'], wide, (s, e))
            row['windows'][f'W{wi+1}'] = r
            if r:
                print(f"    W{wi+1}: IC={r['ic']:+0.4f} ICIR={r['icir']:0.3f} "
                      f"{r['verdict']}")
        results.append(row)

    pass_sets = []
    for wi in range(N_WINDOWS):
        s = {r['factor'] for r in results
             if r['windows'].get(f'W{wi+1}') and
             r['windows'][f'W{wi+1}']['verdict'] == 'PASS'}
        pass_sets.append(s)
    for wi, s in enumerate(pass_sets):
        print(f'  W{wi+1} 通过 {len(s)} 个: {sorted(s) or "-"}')

    jac = [round(jaccard(pass_sets[i], pass_sets[i + 1]), 3)
           for i in range(len(pass_sets) - 1)]

    # 方向一致性: 三窗口 ic 符号完全一致的因子占比
    sign_consistent = 0
    n_valid = 0
    for r in results:
        valid_windows = sum(1 for i in range(N_WINDOWS)
                            if r['windows'].get(f'W{i+1}'))
        signs = {r['windows'][f'W{i+1}']['sign']
                 for i in range(N_WINDOWS)
                 if r['windows'].get(f'W{i+1}')}
        if valid_windows >= 2:
            n_valid += 1
            if len(signs) == 1:
                sign_consistent += 1

    out = {
        'generated_at': pd.Timestamp.now().isoformat(),
        'method': f'固定候选集在 {N_WINDOWS} 个滚动窗口(各 {WIN_LEN} 交易日, '
                  f'错开 {WIN_SHIFT})重跑 fwd{FWD} 截面 IC 裁决 '
                  f'(ic>={IC_MIN} & icir>={ICIR_MIN}), 比较通过集 Jaccard',
        'candidates': [c['name'] for c in cands],
        'results': results,
        'pass_sets': {f'W{i+1}': sorted(s) for i, s in enumerate(pass_sets)},
        'jaccard_adjacent': {f'J(W{i+1},W{i+2})': jac[i]
                             for i in range(len(jac))},
        'sign_consistency': round(sign_consistent / n_valid, 3) if n_valid else None,
        'reliability_reading': (
            'HIGH' if all(j >= 0.5 for j in jac)
            else 'MEDIUM' if any(j >= 0.3 for j in jac) else 'LOW'),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT_PATH, 'w', encoding='utf-8'), ensure_ascii=False,
              indent=2, default=str)
    print(f'\n  完成: {OUT_PATH.name}')
    print(f'  Jaccard(相邻窗口) = {jac}')
    print(f'  可靠性读数: {out["reliability_reading"]} '
          f'(方向一致性 {out["sign_consistency"]})')


if __name__ == '__main__':
    main()
