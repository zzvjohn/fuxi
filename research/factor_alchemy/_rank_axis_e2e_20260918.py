# -*- coding: utf-8 -*-
"""axis=1 族因子端到端双路径验证 — 2026-09-18

命题 (待证伪): 「31 条含 axis=1 的因子在 S5 层必抛 ValueError 被静默丢弃」。
上次结论来自推理, 未实跑。本脚本把 31 条真送进两条求值路径:
  - _evaluate_cross_section : 路由目标 (needs_cross_section=True)
  - _evaluate_per_stock     : 反事实 (强制走 Series 上下文)
并额外做参数写法反事实: 把 `.rank(axis=1, pct=True)` 改写成 `.rank(pct=True,axis=1)`
(逗号后无空格) 看路由是否漏判 ⇒ 证明隐患是**字面白名单脆弱性**而非当下崩。

用合成面板 (12 股 × 200 日), 字段名与真实面板一致。只读, 不改任何数据。
"""
import sys, re, json, traceback

sys.path.insert(0, r'e:/quant/research/factor_alchemy')
sys.path.insert(0, r'e:/quant/scripts')
sys.path.insert(0, r'e:/quant')

import numpy as np
import pandas as pd

AX1 = re.compile(r'\baxis\s*=\s*1\b')

FIELDS = [
    'open_p', 'high_p', 'low_p', 'close_p', 'volume_p', 'amount_p',
    'overnight_p', 'amplitude_p', 'returns_p', 'turnover_p',
    'buy_lg_vol', 'sell_lg_vol', 'buy_sm_vol', 'sell_sm_vol',
    'buy_md_vol', 'sell_md_vol', 'buy_elg_vol', 'sell_elg_vol',
    'net_mf_vol', 'net_mf_amount', 'total_assets', 'intan_assets', 'goodwill',
    'pe_ttm', 'pb', 'dv_ratio', 'roe_dt', 'tr_yoy', 'netprofit_yoy',
    'lhb_flag', 'lhb_net_amount', 'lhb_amount', 'lhb_net_rate',
    'lhb_inst_net_buy', 'lhb_inst_buy', 'lhb_inst_sell',
    'lhb_north_net_buy', 'lhb_north_buy', 'lhb_north_sell',
    'lhb_yz_net_buy', 'lhb_yz_buy', 'lhb_yz_sell',
    'rzye', 'rqye', 'rzmre', 'rqyl', 'rqchl',
]


def build_panel(n_stock=12, n_day=200, seed=7):
    rng = np.random.default_rng(seed)
    codes = ['%06d.SZ' % (1 + i) for i in range(n_stock)]
    dates = pd.bdate_range('2025-01-02', periods=n_day)
    rows = []
    base = {}
    for c in codes:
        base[c] = 10.0 + rng.random() * 30
    for d in dates:
        for c in codes:
            base[c] *= float(np.exp(rng.normal(0, 0.02)))
            px = base[c]
            vol = float(np.exp(rng.normal(14, 0.6)))
            row = {
                'trade_date': d.strftime('%Y%m%d'), 'ts_code': c,
                'open_p': px * (1 + rng.normal(0, 0.004)),
                'high_p': px * (1 + abs(rng.normal(0, 0.01))),
                'low_p': px * (1 - abs(rng.normal(0, 0.01))),
                'close_p': px,
                'volume_p': vol, 'amount_p': vol * px,
                'overnight_p': rng.normal(0, 0.01), 'amplitude_p': abs(rng.normal(0, 0.02)),
                'returns_p': rng.normal(0, 0.02), 'turnover_p': abs(rng.normal(2, 0.7)),
                'buy_lg_vol': vol * 0.3, 'sell_lg_vol': vol * 0.28,
                'buy_sm_vol': vol * 0.2, 'sell_sm_vol': vol * 0.22,
                'buy_md_vol': vol * 0.25, 'sell_md_vol': vol * 0.25,
                'buy_elg_vol': vol * 0.1, 'sell_elg_vol': vol * 0.1,
                'net_mf_vol': vol * rng.normal(0, 0.05), 'net_mf_amount': vol * px * rng.normal(0, 0.05),
                'total_assets': 1e10, 'intan_assets': 1e8, 'goodwill': 1e7,
                'pe_ttm': abs(rng.normal(25, 8)), 'pb': abs(rng.normal(2.5, 1)),
                'dv_ratio': abs(rng.normal(1.5, 0.6)), 'roe_dt': rng.normal(8, 4),
                'tr_yoy': rng.normal(10, 15), 'netprofit_yoy': rng.normal(8, 20),
                # 龙虎榜: 稀疏事件 (约 8% 命中)
                'lhb_flag': 1.0 if rng.random() < 0.08 else 0.0,
                'lhb_net_amount': rng.normal(0, 1e7), 'lhb_amount': abs(rng.normal(5e7, 2e7)),
                'lhb_net_rate': rng.normal(0, 3),
                'lhb_inst_net_buy': rng.normal(0, 2e6) if rng.random() < 0.05 else 0.0,
                'lhb_inst_buy': 0.0, 'lhb_inst_sell': 0.0,
                'lhb_north_net_buy': rng.normal(0, 2e6) if rng.random() < 0.05 else 0.0,
                'lhb_north_buy': 0.0, 'lhb_north_sell': 0.0,
                'lhb_yz_net_buy': rng.normal(0, 2e6) if rng.random() < 0.05 else 0.0,
                'lhb_yz_buy': 0.0, 'lhb_yz_sell': 0.0,
                'rzye': abs(rng.normal(1e9, 1e8)), 'rqye': abs(rng.normal(1e7, 1e6)),
                'rzmre': abs(rng.normal(1e8, 1e7)), 'rqyl': abs(rng.normal(1e6, 1e5)),
                'rqchl': abs(rng.normal(1e5, 1e4)),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def _targets():
    import daily_factor_hypothesis as dh
    out = []
    for e in dh.NOVEL_FACTOR_LIBRARY:
        f = str(e.get('formula', ''))
        if AX1.search(f):
            out.append((str(e.get('id')), str(e.get('name')), f))
    return out


def main():
    import s5_joint_filter as S
    f = S.S5JointFilter(sample_stocks=None)
    df = build_panel()
    print('合成面板: %d 行, %d 股, 字段 %d 个' % (len(df), df['ts_code'].nunique(), len(FIELDS)))

    # 采样确认 _evaluate_cross_section 实际 pivot 了哪些字段
    avail_extra = [c for c in FIELDS if c not in (
        'open_p', 'high_p', 'low_p', 'close_p', 'volume_p', 'amount_p',
        'overnight_p', 'amplitude_p', 'returns_p', 'turnover_p')]
    print('非基础字段 (pivot 清单候选): %d 个' % len(avail_extra))

    rows = []
    n_cs_ok = n_cs_fail = n_series_ok = n_series_fail = 0
    for fid, name, formula in _targets():
        det = bool(f._needs_cross_section(formula))
        rec = {'id': fid, 'name': name, 'routed_cs': det}
        # A. 路由目标路径
        try:
            r = f._evaluate_cross_section(df, formula, fid)
            rec['cs_rows'] = 0 if r is None else len(r)
            rec['cs_ok'] = bool(r is not None and len(r) > 0)
        except Exception as e:
            rec['cs_ok'] = False
            rec['cs_err'] = '%s: %s' % (type(e).__name__, str(e)[:110])
        # B. 反事实: 强制逐股 Series 路径
        try:
            r2 = f._evaluate_per_stock(df, formula, fid)
            rec['series_ok'] = bool(r2 is not None and len(r2) > 0)
            rec['series_rows'] = 0 if r2 is None else len(r2)
        except Exception as e:
            rec['series_ok'] = False
            rec['series_err'] = '%s: %s' % (type(e).__name__, str(e)[:110])
        n_cs_ok += rec.get('cs_ok', False)
        n_cs_fail += not rec.get('cs_ok', False)
        n_series_ok += rec.get('series_ok', False)
        n_series_fail += not rec.get('series_ok', False)
        rows.append(rec)

    print()
    print('%-10s %-30s %-6s %-9s %-9s' % ('id', 'name', 'route', 'cross', 'series'))
    for r in rows:
        print('%-10s %-30s %-6s %-9s %-9s %s' % (
            r['id'], str(r['name'])[:30], 'CS' if r['routed_cs'] else 'SER',
            'OK(%s)' % r.get('cs_rows') if r.get('cs_ok') else 'FAIL',
            'OK(%s)' % r.get('series_rows') if r.get('series_ok') else 'FAIL',
            (r.get('cs_err') or r.get('series_err') or '')[:60]))
    print()
    print('汇总: cross-section 成功 %d / 失败 %d ; series 成功 %d / 失败 %d' % (
        n_cs_ok, n_cs_fail, n_series_ok, n_series_fail))

    # C. 写法脆弱性反事实: 去掉逗号后的空格
    print()
    print('=' * 78)
    print('C. 路由字面白名单脆弱性 (改写法 → 漏判?)')
    print('=' * 78)
    variants = []
    for fid, name, formula in _targets():
        for newf, label in [
            (formula.replace('rank(axis=1, pct=True)', 'rank(pct=True,axis=1)'), 'rank(pct=True,axis=1) 无空格'),
            (formula.replace('rank(pct=True, axis=1)', 'rank(pct=True,ascending=False,axis=1)'), '插入 ascending='),
            (formula.replace('rank(pct=True, axis=1)', 'rank(pct = True, axis = 1)'), '等号两侧加空格'),
        ]:
            if newf != formula:
                variants.append((fid, label, bool(f._needs_cross_section(newf)), newf))
                break
    n_miss = 0
    for fid, label, det, newf in variants:
        if not det:
            n_miss += 1
        print('   %-10s %-26s 路由判定=%-5s %s' % (fid, label, det, '<== 漏判 → 落 Series' if not det else ''))
    print('   样本 %d 条, 漏判 %d 条' % (len(variants), n_miss))

    # D. 漏判后的实际后果
    print()
    print('=' * 78)
    print('D. 漏判后落 Series 路径的实际后果')
    print('=' * 78)
    bad = [v for v in variants if not v[2]]
    if bad:
        fid, label, _, newf = bad[0]
        try:
            r = f._evaluate_per_stock(df, newf, fid)
            print('   %s 改写后 series 路径: 返回 %s' % (fid, 'None/空' if r is None or len(r) == 0 else len(r)))
        except Exception as e:
            print('   %s 改写后 series 路径抛错: %s: %s' % (fid, type(e).__name__, str(e)[:90]))
    else:
        print('   无漏判样本 — 现有写法均在白名单内')

    with open(r'e:/quant/data/_rank_axis_e2e_20260918.json', 'w',
              encoding='utf-8', newline='\r\n') as fh:
        json.dump({'n_targets': len(rows), 'cross_ok': n_cs_ok, 'cross_fail': n_cs_fail,
                   'series_ok': n_series_ok, 'series_fail': n_series_fail,
                   'rows': rows,
                   'variant_miss': n_miss, 'variants': [
                       {'id': a, 'label': b, 'routed': c} for a, b, c, _ in variants]},
                  fh, ensure_ascii=False, indent=2)
    print()
    print('written: e:/quant/data/_rank_axis_e2e_20260918.json')


if __name__ == '__main__':
    main()
