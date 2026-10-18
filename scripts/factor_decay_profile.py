# -*- coding: utf-8 -*-
"""P-20260906-005: IC 半衰期档案升级 (影子, 不裁决)。

将 P-20260902-002 的瞬时半衰期影子升级为可检索档案:
  1. 对主力因子批量计算日频半衰期 (FactorICComputer.compute_decay)
  2. 周频通道半衰期 (WeeklyLaneJudge 环境, IC by lag 1/2/4/8 周指数拟合)
  3. 分段标签: slow(>15日, 优) / mid(5-15日) / fast(<5日, 快衰)
  4. 落盘 data/factor_decay_profile.json

依据: 高频因子衰减实证 (半衰期>15日年化超额 18.7% vs <5日仅 4.2%);
衰减速度比 IC 本身更重要。仍只标记不裁决。
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'research' / 'factor_alchemy'))

OUT_PATH = PROJECT_ROOT / 'data' / 'factor_decay_profile.json'
N_FACTORS = 15

WEEKLY_LAGS = (1, 2, 4, 8)   # 周


def half_life_report():
    """🔴 P-20260918-003 (影子, 零裁决): 半衰期阈值复核报告。

    不做任何阈值改动, 只把「按候选阈值重分类会怎样」算清楚 + 暴露度量的
    **可识别性**问题。依据: 投顾猫 09-14「A股单一因子半衰期从 2-3 年压缩至
    6-12 个月」— 但注意该「半衰期」指的是**因子有效生命周期**(年尺度),
    与本档 `half_life_days`(**IC 在 lag 1-20 交易日上的自衰减半衰期**, 日尺度)
    **不是同一个量**。直接拿月数去改日线阈值 = 量纲错配。本报告把两者并列,
    并把「重分类后果」摊开, 供人工裁决 —— 不自动改任何阈值。
    """
    print('=' * 78)
    print('  P-20260918-003 半衰期阈值复核 (影子报告, 不裁决)')
    print('=' * 78)
    if not OUT_PATH.exists():
        print(f'  ⚠️ 档案不存在: {OUT_PATH} (先跑主流程生成)')
        return 2
    d = json.load(open(OUT_PATH, encoding='utf-8', newline=''))
    factors = d.get('factors', []) or []
    print(f"  档案 generated_at = {d.get('generated_at')}")
    print(f"  样本 n_factors    = {len(factors)}")
    print(f"  现行分段          = {d.get('segments')}")
    print()

    rows = []
    for e in factors:
        dd = e.get('daily') or {}
        hl = dd.get('half_life_days')
        rows.append({
            'factor': e.get('factor'),
            'icir': e.get('daily_icir'),
            'hl': hl if isinstance(hl, (int, float)) else None,
            'r2': dd.get('fit_r2'),
            'cls': e.get('class_daily'),
        })

    # ── 1) 可识别性: inf / 缺失 / r2<0 ──
    inf_n = sum(1 for r in rows if r['hl'] is not None and not np.isfinite(r['hl']))
    fin = [r for r in rows if r['hl'] is not None and np.isfinite(r['hl'])]
    none_n = sum(1 for r in rows if r['hl'] is None)
    negr2 = [r for r in rows if isinstance(r['r2'], (int, float)) and r['r2'] < 0]
    print('  ── 可识别性 (度量能否区分因子) ──')
    print(f'    half_life = inf (λ<=1e-6, IC 在 lag 1-20 内不衰减) : {inf_n}/{len(rows)}')
    print(f'    half_life 有限                                     : {len(fin)}/{len(rows)}')
    print(f'    half_life 缺失 (compute_decay 失败)                : {none_n}/{len(rows)}')
    print(f'    fit_r2 < 0 (拟合无效)                              : {len(negr2)}/{len(rows)}')
    if len(fin):
        a = np.array([r['hl'] for r in fin], dtype=float)
        print(f'    有限读数范围: min={a.min():.1f}  中位={np.median(a):.1f}  max={a.max():.1f} (日)')
    print()

    # ── 2) 阈值扫描: 候选 fast 线 -> 重分类比例 ──
    print('  ── 阈值扫描 (若把 fast 判定线改为 < X 日, 多少因子被标 fast) ──')
    print('    %-12s %-10s %-10s %s' % ('候选 fast 线', '被标 fast', '占比', '相对现行增量'))
    cur_fast = sum(1 for r in rows if r['cls'] == 'fast')
    cands = [5, 15, 30, 60, 89, 90, 120, 180, 250, 365]
    for th in cands:
        n_f = sum(1 for r in rows
                  if r['hl'] is not None and np.isfinite(r['hl']) and r['hl'] < th)
        # inf 读数不会被任何有限阈值判为 fast (方向: 只在有限读数上收紧)
        pct = 100.0 * n_f / len(rows) if rows else 0.0
        delta = n_f - cur_fast
        tag = '(现行)' if th == 5 else ''
        print('    %-12s %-10s %-10s %+d %s' % (f'< {th} 日', f'{n_f}/{len(rows)}',
                                               f'{pct:.0f}%', delta, tag))
    print()
    print(f'    🔴 关键: {inf_n}/{len(rows)} 读数为 inf ⇒ 有限阈值无论取 5 还是 120, '
          '对这批因子**全部不触发**。')
    print('       阈值改动的作用域只落在极少数「有限且偏短」的因子上 ⇒ 改它是在')
    print('       **加权一个不可识别的度量**, 增加任意性, 而不是去掉任意性。')
    print()

    # ── 3) 量纲对照: 两个不同概念 ──
    print('  ── 量纲对照 (提案证据与本地度量的口径差异) ──')
    print('    提案证据 (投顾猫 09-14): 因子半衰期 2-3 年 -> 6-12 个月')
    print('      = **因子有效生命周期** (alpha 在年尺度上被套利掉的时间), 年尺度')
    print('    本地度量 half_life_days: IC 在 lag 1-20 **交易日** 上的自衰减')
    print('      = **信号平滑度/微观持续性**, 日尺度')
    print('    ⇒ 两者不是同一个量, 不存在「把 15 日线改成 120 日线」的映射关系。')
    print('    ⇒ 若要跟进行业「半衰期压缩」, 应新建**年尺度**生命周期度量')
    print('      (如按年度分组的 IC 趋势 / 滚动 12 个月 IC 斜率), 而非改本档阈值。')
    print()

    # ── 4) 落盘 ──
    rep = {
        'generated_at': pd.Timestamp.now().isoformat(),
        'proposal': 'P-20260918-003',
        'mode': 'shadow (零裁决, 不改任何阈值)',
        'source_profile': str(OUT_PATH.name),
        'source_generated_at': d.get('generated_at'),
        'n_factors': len(rows),
        'current_segments': d.get('segments'),
        'identifiability': {
            'inf': int(inf_n), 'finite': int(len(fin)), 'missing': int(none_n),
            'negative_r2': int(len(negr2)),
        },
        'threshold_sweep': {
            f'lt_{th}': sum(1 for r in rows
                            if r['hl'] is not None and np.isfinite(r['hl']) and r['hl'] < th)
            for th in cands
        },
        'verdict_note': (
            '量纲错配: 提案的 6-12 个月 = 因子有效生命周期(年尺度), 本档 '
            'half_life_days = IC 在 lag 1-20 交易日的自衰减(日尺度)。'
            '且 12/14 读数为 inf(不可识别) ⇒ 改阈值只在极少数有限读数上生效, '
            '属加权不可识别度量。建议不直接改本档阈值, 另建年尺度生命周期度量。'),
        'rows': rows,
    }
    out = PROJECT_ROOT / 'data' / 'factor_half_life_report.json'
    # 🔴 data/ 下 JSON 约定 CRLF: 显式 newline='\r\n' (用 newline='' 会落裸 LF)
    with open(out, 'w', encoding='utf-8', newline='\r\n') as f:
        json.dump(rep, f, ensure_ascii=False, indent=2, default=str)
    print(f'  报告 -> {out.name}')
    return 0



def pick_main_factors():
    pool = pd.read_csv(PROJECT_ROOT / 'data' / 'unified_factor_pool.csv')
    pool = pool[pool['formula'].notna() & (pool['formula'].astype(str).str.len() > 5)]
    pool = pool[pool['daily_icir'].notna()]  # 剔除 icir 缺失的 v3_xxx 占位行
    jq = pool[pool['jq_verified'].notna()].sort_values('daily_icir', ascending=False)
    rest = pool[~pool.index.isin(jq.index)].sort_values('daily_icir', ascending=False)
    sel = pd.concat([jq, rest]).head(N_FACTORS)
    return sel[['name', 'formula', 'daily_icir', 'jq_verified']].to_dict('records')


def classify(half_life_days):
    if half_life_days is None or np.isnan(half_life_days):
        return 'unknown'
    if half_life_days > 15:
        return 'slow'
    if half_life_days >= 5:
        return 'mid'
    return 'fast'


def fit_half_life(mean_ic_by_lag, lags):
    """指数拟合半衰期: log(|ic_k|/|ic_1|) = -λ(k-1), t½ = ln2/λ。返回 (t½, r2)。"""
    base = mean_ic_by_lag.get(lags[0], 0.0)
    if abs(base) < 0.005:
        return None, None
    xs, ys = [], []
    for k in lags[1:]:
        v = mean_ic_by_lag.get(k, 0.0)
        if abs(base) < 1e-9 or abs(v) < 1e-9:
            continue
        xs.append(k - lags[0])
        ys.append(np.log(abs(v) / abs(base)))
    if len(xs) < 2:
        return None, None
    lam = -np.polyfit(xs, ys, 1)[0]
    if lam <= 1e-6:
        return float('inf'), 1.0
    return float(np.log(2) / lam), float(np.corrcoef(xs, ys)[0, 1] ** 2
                                         if len(xs) > 2 else 1.0)


def daily_half_life(icc, formula, name):
    r = icc.compute_decay(formula, name)
    return {
        'half_life_days': r.get('decay_half_life_days'),
        'lambda': r.get('decay_lambda'),
        'fit_r2': r.get('fit_r2'),
        'ic_by_lag': r.get('ic_by_lag', {}),
        'n_days': r.get('n_days', 0),
        'ok': bool(r.get('decay_eval_ok')),
    }


def weekly_half_life(formula, name, judge):
    """周频通道: 复用 WeeklyLaneJudge 的宽表环境, IC by lag 拟合。"""
    try:
        clean = judge._clean_formula(formula)
    except Exception as e:
        return {'ok': False, 'reason': f'clean: {str(e)[:80]}'}
    safe_builtins = {
        'range': range, 'len': len, 'int': int, 'float': float,
        'list': list, 'dict': dict, 'tuple': tuple, 'str': str, 'bool': bool,
        'True': True, 'False': False, 'None': None,
        'abs': abs, 'min': min, 'max': max, 'round': round,
        'sum': sum, 'zip': zip, 'sorted': sorted,
        'print': print, 'isinstance': isinstance,
        'Exception': Exception, 'ValueError': ValueError,
        'TypeError': TypeError,
    }
    base_context = {
        "np": np, "pd": pd,
        "abs": np.abs, "sqrt": np.sqrt, "log": np.log,
        "log1p": np.log1p, "exp": np.exp, "sign": np.sign,
        "maximum": np.maximum, "minimum": np.minimum,
        "where": np.where, "clip": np.clip,
        "range": range, "len": len, "int": int, "float": float,
        "list": list, "dict": dict, "tuple": tuple,
    }
    full_globals = {"__builtins__": safe_builtins, **base_context, **judge._wide_env}
    try:
        result = eval(clean, full_globals, {})
    except Exception as e:
        return {'ok': False, 'reason': f'eval: {type(e).__name__}: {str(e)[:80]}'}
    if not isinstance(result, pd.DataFrame):
        return {'ok': False, 'reason': f'结果类型 {type(result).__name__}'}
    factor_wide = result
    close = judge._wide_env['close']

    ic_by_lag = {}
    n_weeks = 0
    for k in WEEKLY_LAGS:
        fwd = close.shift(-k) / close - 1
        ics = []
        for date in factor_wide.index:
            fv = factor_wide.loc[date]
            fw = fwd.loc[date] if date in fwd.index else None
            if fw is None or fv.notna().sum() < 30 or fw.notna().sum() < 30:
                continue
            common = fv.dropna().index.intersection(fw.dropna().index)
            if len(common) < 30:
                continue
            ic = fv[common].rank().corr(fw[common].rank())
            if not np.isnan(ic):
                ics.append(ic)
        ic_by_lag[k] = float(np.nanmean(ics)) if ics else 0.0
        if k == WEEKLY_LAGS[0]:
            n_weeks = len(ics)
    if n_weeks == 0:
        return {'ok': False, 'reason': '无有效周频截面'}

    hl_weeks, r2 = fit_half_life(ic_by_lag, list(WEEKLY_LAGS))
    hl_days = hl_weeks * 5 if hl_weeks is not None else None
    return {
        'ok': True,
        'half_life_weeks': hl_weeks,
        'half_life_days': hl_days,
        'fit_r2': r2,
        'ic_by_lag': {f'w{k}': round(v, 5) for k, v in ic_by_lag.items()},
        'n_weeks': n_weeks,
    }


def main():
    print('=' * 70)
    print('  IC 半衰期档案升级 (P-20260906-005, 影子不裁决)')
    print('=' * 70)
    factors = pick_main_factors()
    print(f'  主力因子 {len(factors)} 个:')
    for f in factors:
        print(f"    - {f['name']} (ICIR={f['daily_icir']}, jq={f['jq_verified']})")

    # 周频裁决器 (仅用其数据环境, 不做裁决)
    from weekly_lane import WeeklyLaneJudge
    judge = WeeklyLaneJudge()
    weekly_loaded = judge._load()
    print(f'  周频数据: {"可用" if weekly_loaded else "不可用"}')

    # 日频 IC 计算机单实例复用 (避免每因子全量重载 ~7M 行)
    from factor_ic_computer import FactorICComputer
    icc = FactorICComputer(lookback_days=360)

    entries = []
    for i, f in enumerate(factors):
        print(f"  [{i+1}/{len(factors)}] {f['name']} ...")
        d = daily_half_life(icc, f['formula'], f['name'])
        entry = {
            'factor': f['name'],
            'daily_icir': f['daily_icir'],
            'jq_verified': f['jq_verified'],
            'daily': d,
            'weekly': None,
            'class_daily': classify(d['half_life_days']),
            'class_weekly': 'unknown',
        }
        if weekly_loaded:
            w = weekly_half_life(f['formula'], f['name'], judge)
            entry['weekly'] = w
            if w.get('ok'):
                entry['class_weekly'] = classify(w.get('half_life_days'))
        entries.append(entry)

    dist = {}
    for e in entries:
        dist[e['class_daily']] = dist.get(e['class_daily'], 0) + 1
    out = {
        'generated_at': pd.Timestamp.now().isoformat(),
        'method': '日频: FactorICComputer.compute_decay (lag 1-20日指数拟合); '
                  '周频: WeeklyLaneJudge 环境 IC by lag 1/2/4/8周指数拟合(×5折天)',
        'segments': 'slow>15日(优) / mid 5-15日 / fast<5日(快衰)',
        'n_factors': len(entries),
        'distribution_daily': dist,
        'factors': entries,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT_PATH, 'w', encoding='utf-8'), ensure_ascii=False,
              indent=2, default=str)
    print(f'\n  完成: {OUT_PATH.name}')
    print(f'  日频分段分布: {dist}')
    n_w_ok = sum(1 for e in entries if e['weekly'] and e['weekly'].get('ok'))
    print(f'  周频通道成功 {n_w_ok}/{len(entries)}')


if __name__ == '__main__':
    if '--half-life-report' in sys.argv:
        sys.exit(half_life_report())
    main()
