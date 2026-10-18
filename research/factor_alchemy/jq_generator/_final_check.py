"""Final sanity check on both JQ strategy files before JoinQuant deployment."""
import ast, re, os

files = [
    'research/strategies/drafts/fa_v79_lasso_large_dominant.py',
    'research/strategies/drafts/fa_v79_lasso_small_bull.py',
]

all_good = True
for fp in files:
    fp_full = os.path.join('e:/quant', fp)
    print(f'=== {fp.split("/")[-1]} ===')
    with open(fp_full, 'rb') as f:
        raw = f.read()
    with open(fp_full, 'r', encoding='utf-8') as f:
        code = f.read()

    # 1. AST
    try:
        ast.parse(code)
        print('  AST: OK')
    except SyntaxError as e:
        print(f'  AST: FAIL - {e}')
        all_good = False

    # 2. Line endings
    crlf = raw.count(b'\r\n')
    tabs = raw.count(b'\t')
    print(f'  CRLF={crlf} Tabs={tabs}')
    if crlf > 0 or tabs > 0:
        print('  WARNING: bad line endings/tabs!')
        all_good = False

    # 3. Try/except
    try_stack = []; issues = 0
    for i, line in enumerate(code.split('\n'), 1):
        content = line.strip()
        if not content or content.startswith('#'):
            continue
        leading = len(line) - len(line.lstrip())
        if re.match(r'^try\s*:', content):
            try_stack.append(leading)
        elif re.match(r'^except\b', content):
            if not try_stack or leading != try_stack[-1]:
                print(f'  TRY/EXCEPT MISMATCH L{i}: except={leading} try={try_stack[-1] if try_stack else "N/A"}')
                issues += 1
            if try_stack:
                try_stack.pop()
    if issues == 0:
        print('  Try/Except: OK')
    else:
        all_good = False
    if try_stack:
        print(f'  WARNING: {len(try_stack)} unclosed try blocks')

    # 4. Factor functions - check real vs placeholder
    funcs = re.findall(r'def compute_(\w+)\((.*?)\):', code)
    for fname, args in funcs:
        if fname == 'factor':
            continue
        m = re.search(rf'def compute_{fname}\(.*?\):(.*?)(?=\n    def |\n# ===|\Z)', code, re.DOTALL)
        if m:
            body = m.group(1)
            # Check if it's a one-liner that just returns nan array
            lines_body = [l.strip() for l in body.strip().split('\n') if l.strip()]
            is_trivial = len(lines_body) <= 3 and any('np.full(n, np.nan)' in l for l in lines_body)
            is_placeholder = is_trivial and any('return arr, valid' in l for l in lines_body[:3])
            status = 'REAL' if not is_placeholder else 'PLACEHOLDER'
            print(f'  compute_{fname}({args}): {status} ({len(body)} chars)')
            if is_placeholder:
                print(f'    WARNING: still placeholder!')
                all_good = False

    # 5. Strategy name & weights
    name_m = re.search(r"STRATEGY_NAME\s*=\s*'([^']+)'", code)
    mcap_m = re.search(r'MCAP_MIN\s*=\s*(\d+)', code)
    weights_m = re.search(r'FACTOR_WEIGHTS\s*=\s*(\{[^}]+\})', code)
    print(f'  Strategy: {name_m.group(1) if name_m else "NOT FOUND"}')
    print(f'  MCAP_MIN: {mcap_m.group(1) if mcap_m else "NOT FOUND"}')
    if weights_m:
        import json
        w_str = weights_m.group(1)
        try:
            wd = eval(w_str)
            print(f'  Weights: {wd}')
        except:
            print(f'  Weights: {w_str[:100]}')

    # 6. PRICE_FIELDS
    pf_m = re.search(r'PRICE_FIELDS\s*=\s*(\[[^\]]+\])', code)
    if pf_m:
        fields = eval(pf_m.group(1))
        print(f'  PRICE_FIELDS: {fields}')

    # 7. Check bp/ep use context dispatch
    if 'compute_bp(stocks, context)' in code:
        print('  bp dispatch: context (correct)')
    if 'compute_ep(stocks, context)' in code:
        print('  ep dispatch: context (correct)')

    # 8. Line count
    print(f'  Lines: {code.count(chr(10))}')
    print()

print(f'OVERALL: {"ALL PASS" if all_good else "ISSUES FOUND"}')
