import re

with open('e:/quant/scripts/gen_rp_ding_cn.js', 'r', encoding='utf-8') as f:
    content = f.read()

# Find all ASCII double quotes inside Chinese text contexts
# Strategy: find lines that are P("...") calls and fix inner quotes
lines = content.split('\n')
fixed = []
for line in lines:
    # Check if this is a P("...") line with inner quotes problem
    stripped = line.strip()
    if stripped.startswith('P("') and stripped.endswith('"),'):
        inner = stripped[3:-3]  # text between P(" and "),
        # Count ASCII quotes in inner
        quote_count = inner.count('"')
        if quote_count > 0:
            # Replace pairs of quotes with angle brackets
            # First find all quote positions
            parts = []
            i = 0
            in_quote = False
            current = []
            for ch in inner:
                if ch == '"':
                    if in_quote:
                        current.append('\u300d')  # right bracket
                        in_quote = False
                    else:
                        current.append('\u300c')  # left bracket
                        in_quote = True
                else:
                    current.append(ch)
            new_inner = ''.join(current)
            line = line.replace(inner, new_inner)
    fixed.append(line)

result = '\n'.join(fixed)
with open('e:/quant/scripts/gen_rp_ding_cn.js', 'w', encoding='utf-8') as f:
    f.write(result)
print('Fixed', sum(1 for l in fixed if '\u300c' in l), 'lines with Chinese quotes')
