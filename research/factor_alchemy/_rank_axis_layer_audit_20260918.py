# -*- coding: utf-8 -*-
"""axis=1 跨层轴语义缺陷 — 全库量化审计 (2026-09-18)

命题: 公式里写 `axis=1` 表示「横截面」(需要 DataFrame pivot 上下文),
      但 s5_joint_filter._needs_cross_section() 用的是**字面子串白名单**,
      写法变体 (逗号后无空格 / 参数顺序 / 插入 ascending=) 会漏判
      ⇒ 落进 _evaluate_per_stock (逐股 Series) ⇒ Series.rank(axis=1) ValueError
      ⇒ L1594 的 except Exception 打印一行后 return None = **静默丢候选**。

本脚本只测量, 不修改任何东西。
判据锚在**实现**(正则扫真实公式文本), 不锚命名。
"""
import sys, os, json, re

sys.path.insert(0, r'e:/quant/research/factor_alchemy')
sys.path.insert(0, r'e:/quant/scripts')
sys.path.insert(0, r'e:/quant')

import pandas as pd  # noqa: E402

NOVEL_OUT = r'e:/quant/data/_rank_axis_layer_audit_novel_20260918.json'
LIB_OUT = r'e:/quant/data/_rank_axis_layer_audit_lib_20260918.json'

# 任一 pandas 归约/排名调用带 axis=1 (容忍任意空白与参数顺序)
AX1_CALL = re.compile(r'\.\s*\w+\s*\([^()]*\baxis\s*=\s*1\b')
AX1_ANY = re.compile(r'\baxis\s*=\s*1\b')
# rank 族带 axis=1 (真正的 Series 崩溃源)
AX1_RANK = re.compile(r'\.\s*rank\s*\([^()]*\baxis\s*=\s*1\b')


def _load_lib():
    p = r'e:/quant/data/library_orthogonality_state.json'
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f).get('factors', [])


def _novel_lib():
    """NOVEL_FACTOR_LIBRARY 是 list[dict] (id/name/formula/...), 归一为 (key, dict) 序列。"""
    import daily_factor_hypothesis as dh
    raw = getattr(dh, 'NOVEL_FACTOR_LIBRARY', [])
    if isinstance(raw, dict):
        return list(raw.items())
    out = []
    for i, ent in enumerate(raw):
        if isinstance(ent, dict):
            out.append((str(ent.get('id') or ent.get('name') or i), ent))
        else:
            out.append((str(i), {'formula': str(ent)}))
    return out


def main():
    import s5_joint_filter as S
    needs = S.S5JointFilter._needs_cross_section
    dummy = object()  # 该方法只读 formula 参数, 无需真实实例

    print('=' * 78)
    print('A. Series.rank(axis=1) 是否真的抛错 (最小复现)')
    print('=' * 78)
    s = pd.Series([1.0, 2.0, 3.0])
    for expr in ['s.rank(pct=True, axis=1)', 's.rank(axis=1)']:
        try:
            eval(expr, {'s': s})
            print('   %-28s -> OK (未抛错)' % expr)
        except Exception as e:
            print('   %-28s -> %s: %s' % (expr, type(e).__name__, str(e)[:70]))

    print()
    print('=' * 78)
    print('B. NOVEL_FACTOR_LIBRARY (270 条)')
    print('=' * 78)
    nl = _novel_lib()
    n_total = n_ax1 = n_missed = n_rank_ax1 = 0
    missed = []
    for key, ent in nl:
        formula = str(ent.get('formula') or ent.get('expression') or '')
        if not formula:
            continue
        n_total += 1
        has_ax1 = bool(AX1_ANY.search(formula))
        has_rank_ax1 = bool(AX1_RANK.search(formula))
        if has_ax1:
            n_ax1 += 1
        if has_rank_ax1:
            n_rank_ax1 += 1
        try:
            det = bool(needs(dummy, formula))
        except Exception:
            det = None
        if has_ax1 and det is False:
            n_missed += 1
            missed.append({
                'key': key,
                'name': (ent.get('name') if isinstance(ent, dict) else '') or '',
                'axis1_calls': sorted(set(AX1_CALL.findall(formula))),
                'has_rank_ax1': has_rank_ax1,
                'formula': formula[:400],
            })
    print('  总条目(有公式)      : %d' % n_total)
    print('  含 axis=1          : %d  (%.1f%%)' % (n_ax1, 100.0 * n_ax1 / max(n_total, 1)))
    print('  其中 rank(axis=1)  : %d' % n_rank_ax1)
    print('  被 _needs_cross_section 漏判: %d  (%.1f%%)' % (
        n_missed, 100.0 * n_missed / max(n_total, 1)))
    for r in missed:
        print('    - %-34s ax1=%s rank=%s' % (str(r['key'])[:34], r['axis1_calls'][:2], r['has_rank_ax1']))
        print('        %s' % r['formula'][:150].replace('\n', ' '))

    print()
    print('=' * 78)
    print('C. library_orthogonality_state.json (全库)')
    print('=' * 78)
    lib = _load_lib()
    L_total = L_ax1 = L_missed = 0
    L_missed_rank = []
    for f in lib:
        expr = str(f.get('expression', '') or '')
        if not expr:
            continue
        L_total += 1
        has_ax1 = bool(AX1_ANY.search(expr))
        if has_ax1:
            L_ax1 += 1
        try:
            det = bool(needs(dummy, expr))
        except Exception:
            det = None
        if has_ax1 and det is False:
            L_missed += 1
            status = f.get('status')
            if status in ('candidate', 's5_pending', 's5_pass') or AX1_RANK.search(expr):
                L_missed_rank.append({
                    'name': f.get('name'), 'status': status, 'icir': f.get('icir'),
                    'expression': expr[:300],
                })
    print('  总条目            : %d' % L_total)
    print('  含 axis=1         : %d' % L_ax1)
    print('  被漏判            : %d' % L_missed)
    print('  漏判且 status 活跃/含 rank(axis=1) 的 : %d' % len(L_missed_rank))
    for r in L_missed_rank[:25]:
        print('    - %-46s status=%-10s icir=%s' % (
            str(r['name'])[:46], r['status'], r['icir']))

    with open(NOVEL_OUT, 'w', encoding='utf-8', newline='\r\n') as f:
        json.dump({'n_total': n_total, 'n_ax1': n_ax1, 'n_rank_ax1': n_rank_ax1,
                   'n_missed': n_missed, 'missed': missed},
                  f, ensure_ascii=False, indent=2)
    with open(LIB_OUT, 'w', encoding='utf-8', newline='\r\n') as f:
        json.dump({'n_total': L_total, 'n_ax1': L_ax1, 'n_missed': L_missed,
                   'missed_active': L_missed_rank},
                  f, ensure_ascii=False, indent=2)
    print()
    print('written:', NOVEL_OUT)
    print('written:', LIB_OUT)


if __name__ == '__main__':
    main()
