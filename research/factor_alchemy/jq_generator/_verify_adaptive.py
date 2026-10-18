"""Verify the regime-adaptive merged strategy file."""
import ast, re, os

fp = os.path.join('e:/quant', 'research/strategies/drafts/fa_v79_regime_adaptive.py')

# Fix line endings
with open(fp, 'r', newline='') as f:
    text = f.read()
lines = text.split('\n')
cleaned = '\n'.join(line.rstrip() for line in lines)
if not cleaned.endswith('\n'):
    cleaned += '\n'
with open(fp, 'w', newline='', encoding='utf-8') as f:
    f.write(cleaned)

# Stats
with open(fp, 'rb') as f:
    raw = f.read()
crlf = raw.count(b'\r\n')
tabs = raw.count(b'\t')
lc = raw.count(b'\n')

with open(fp, 'r') as f:
    code = f.read()

print(f'File: {os.path.basename(fp)}')
print(f'Lines: {lc}, CRLF={crlf}, Tabs={tabs}')

# AST
try:
    ast.parse(code)
    print('AST: OK')
except SyntaxError as e:
    print(f'AST FAIL: {e}')
    exit(1)

# Try/Except
try_stack = []
issues = 0
for i, line in enumerate(code.split('\n'), 1):
    content = line.strip()
    if not content or content.startswith('#'):
        continue
    leading = len(line) - len(line.lstrip())
    if re.match(r'^try\s*:', content):
        try_stack.append(leading)
    elif re.match(r'^except\b', content):
        if not try_stack or leading != try_stack[-1]:
            print(f'  TRY/EXCEPT MISMATCH L{i}')
            issues += 1
        if try_stack:
            try_stack.pop()
if issues == 0:
    print(f'Try/Except: OK ({len(re.findall(r"^\\s+except\\b", code, re.MULTILINE))} except blocks)')
else:
    print(f'Try/Except: {issues} ISSUES')
    exit(1)

# Factor functions audit
funcs = re.findall(r'def compute_(\w+)\(', code)
for fname in funcs:
    if fname == 'factor':
        continue
    m = re.search(rf'def compute_{fname}\(.*?\):(.*?)(?=\n\ndef |\n# ===|\Z)', code, re.DOTALL)
    if m:
        body = m.group(1)
        lines_body = [l for l in body.strip().split('\n') if l.strip()]
        is_trivial = len(lines_body) <= 3 and any('np.full(n, np.nan)' in l for l in lines_body)
        if is_trivial:
            print(f'  compute_{fname}: PLACEHOLDER - BAD!')
        else:
            print(f'  compute_{fname}: REAL ({len(body)} chars)')

# Feature checks
checks = {
    'detect_regime function': 'def detect_regime' in code,
    'REGIME_LOOKBACK_DAYS': 'REGIME_LOOKBACK_DAYS = 35' in code,
    'Dual weight sets': 'LARGE_DOMINANT_WEIGHTS' in code and 'SMALL_BULL_WEIGHTS' in code,
    'Weight switching logic': 'active_weights = g.small_weights' in code,
    'Regime logging': 'REGIME=' in code,
    'bp dispatch (context)': 'compute_bp(stocks, context)' in code,
    'ep dispatch (context)': 'compute_ep(stocks, context)' in code,
    'Index data fetch (CSI300)': "000300.XSHG" in code,
    'Index data fetch (CSI500)': "000905.XSHG" in code,
    'Index data fetch (CSI_ALL)': "000985.XSHG" in code,
    'Get all factors': 'g.all_factors' in code,
    'Active factors only': 'active_factors' in code,
}

all_ok = True
for label, result in checks.items():
    status = 'OK' if result else 'MISSING'
    if not result:
        all_ok = False
    print(f'  [{status}] {label}')

print()
if all_ok:
    print('ALL CHECKS PASSED')
else:
    print('SOME CHECKS FAILED')
    exit(1)
