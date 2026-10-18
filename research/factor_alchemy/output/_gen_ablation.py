#!/usr/bin/env python
"""Generate JQ ablation scripts from the master Ralph v2 code."""
import re

master_path = 'fa_ralph_jq_20260806.py'
with open(master_path, 'r') as f:
    master = f.read()

subsets = {
    'A': {
        'name': 'Ablation A: 3因子 最小域重叠',
        'factors': ['f1', 'f3', 'f6'],
        'min_factors': 2,
        'top_n': 50,
        'desc': 'close(max_drawdown) + volume(crowding) + close+vol(earnings_drift) -- 3个数据域',
    },
    'B': {
        'name': 'Ablation B: 4因子 ICIR Top-4',
        'factors': ['f1', 'f2', 'f3', 'f7'],
        'min_factors': 3,
        'top_n': 60,
        'desc': 'ICIR 1.665+1.401+1.245+0.741 -- 纯信号强度优先',
    },
    'C': {
        'name': 'Ablation C: 5因子 范式覆盖',
        'factors': ['f1', 'f2', 'f3', 'f5', 'f9'],
        'min_factors': 3,
        'top_n': 70,
        'desc': 'close + vol-regime + vol-crowding + lottery + liquidity -- 5范式',
    },
}

icir_rank = {
    'f1': 1.665, 'f2': 1.401, 'f3': 1.245, 'f4': 1.102,
    'f5': 1.003, 'f6': 0.856, 'f7': 0.741, 'f8': 0.602, 'f9': 0.301,
}

factor_domain = {
    'f1': 'close_only', 'f2': 'close+vol', 'f3': 'volume', 'f4': 'volume+close',
    'f5': 'close+vol', 'f6': 'close+vol', 'f7': 'close_only', 'f8': 'close_only',
    'f9': 'close+vol',
}

# Report
print('=' * 60)
print('  因子精选: Domain-Based Ablation')
print('=' * 60)

print('\n--- Factor Domain Groups ---')
domains = {}
for fk, dom in factor_domain.items():
    domains.setdefault(dom, []).append(fk)
for dom, facts in sorted(domains.items()):
    items = ' + '.join('{}({:.3f})'.format(f, icir_rank[f]) for f in facts)
    print('  [{:15s}] {}'.format(dom, items))

print('\n--- Selected Subsets ---')
for sid, cfg in subsets.items():
    fnames = cfg['factors']
    doms = [factor_domain[f] for f in fnames]
    unique_doms = len(set(doms))
    print('\n  Subset {}: {}'.format(sid, cfg['name']))
    items = ' + '.join('{}({:.2f})'.format(f, icir_rank[f]) for f in fnames)
    print('    Factors: {}'.format(items))
    print('    Domains: {} unique / {} total ({})'.format(unique_doms, len(fnames), set(doms)))
    print('    Config:  Top-N={}, min_valid={}'.format(cfg['top_n'], cfg['min_factors']))
    print('    Desc:    {}'.format(cfg['desc']))

# Generate scripts
print('\n--- Generating JQ Scripts ---')
for sid, cfg in subsets.items():
    code = master

    # Strategy name (try v3 first, then v2 fallback)
    old_name = 'Ralph Loop v0.2 v3 优化版'
    if old_name not in code:
        old_name = 'Ralph Loop v0.2 v2 修复版'
    code = code.replace(
        old_name,
        'Ralph Ablation {}'.format(sid)
    )

    # TOP_N
    code = re.sub(r'TOP_N\s*=\s*\d+', 'TOP_N = {}'.format(cfg['top_n']), code)

    # Factor list
    old_f = '''    all_factors = [f1, f2, f3, f4, f5, f6, f7, f8, f9]
    all_valid   = [v1, v2, v3, v4, v5, v6, v7, v8, v9]'''
    f_list = ', '.join(cfg['factors'])
    v_list = ', '.join(f.replace('f', 'v') for f in cfg['factors'])
    new_f = '    all_factors = [{}]\n    all_valid   = [{}]'.format(f_list, v_list)
    code = code.replace(old_f, new_f)

    # Loop range
    code = code.replace('for j in range(9):', 'for j in range({}):'.format(len(cfg['factors'])))

    # min_factors
    code = code.replace('min_factors = 5', 'min_factors = {}'.format(cfg['min_factors']))

    # Comment
    old_cmt = '# 放宽: 至少5个因子有效 (不是全部9个)'
    new_cmt = '# Ablation {}: 至少{}个因子有效 ({}因子子集)'.format(
        sid, cfg['min_factors'], len(cfg['factors']))
    code = code.replace(old_cmt, new_cmt)

    # Log message: factor count
    code = code.replace(
        '因子: 9个 (全 Grade A), Top-',
        '因子: {}({}因子子集), Top-'.format(len(cfg['factors']), sid),
    )

    fname = 'fa_ralph_ablation_{}_20260807.py'.format(sid.lower())
    with open(fname, 'w') as f:
        f.write(code)
    print('  Generated: {} ({} chars)'.format(fname, len(code)))

print('\nDone!')
