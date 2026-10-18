"""Fix jq_compute_function indentation by operating on the Python object directly,
then rewriting the entire registry.py file."""
import re

def fix_func_body(body):
    """Fix indentation of a function body string (without def line).
    body: multi-line string with try/except that may have wrong indentation.
    Returns: fixed string with correct 4/8-space indentation.
    """
    lines = body.split('\n')
    fixed = []
    state = 'func'  # func, try_body, except_body, post_try
    
    for line in lines:
        stripped = line.lstrip()
        
        if not stripped:
            fixed.append('')
            continue
        
        if stripped == 'try:':
            state = 'try_body'
            fixed.append('    try:')
            continue
        
        if stripped.startswith('except'):
            state = 'except_body'
            fixed.append('    except:')
            continue
        
        # Detect transition from except to post-try
        if state == 'except_body':
            # Lines like 'if ...' or 'stock_idx' or 'for ...' are post-try
            if (stripped.startswith('if ') or stripped.startswith('stock_idx')
                or stripped.startswith('field_key') or stripped.startswith('for ')
                or stripped.startswith('valid ') or stripped.startswith('arr[')):
                state = 'post_try'
        
        if state == 'try_body':
            fixed.append('        ' + stripped)
        elif state == 'except_body':
            fixed.append('        ' + stripped)
        elif state == 'post_try':
            fixed.append('    ' + stripped)
        else:  # func state
            fixed.append('    ' + stripped)
    
    return '\n'.join(fixed)

# Read the registry file
with open('registry.py', 'r', encoding='utf-8') as f:
    content = f.read()

# First, make sure we can reload the module after the pending fixes
# Strategy: find each 'jq_compute_function': section, extract body, fix it, replace

fixed_count = 0
pos = 0
result_parts = []
marker = "'jq_compute_function':"

while True:
    idx = content.find(marker, pos)
    if idx == -1:
        result_parts.append(content[pos:])
        break
    
    # Find opening and closing triple-quotes
    quote_start = content.find('"""', idx + len(marker))
    if quote_start == -1:
        result_parts.append(content[pos:])
        break
    
    body_start = quote_start + 3  # after opening """
    
    # Find closing """ - need to skip any \"\"\" inside the body
    quote_end = content.find('"""', body_start)
    if quote_end == -1:
        result_parts.append(content[pos:])
        break
    
    body = content[body_start:quote_end]
    rest_start = quote_end + 3
    
    # Check if this block needs fixing (has try:)
    if 'try:' in body:
        # Check if indentation is wrong (12-space lines after try)
        needs_fix = False
        for body_line in body.split('\n'):
            indent = len(body_line) - len(body_line.lstrip())
            if indent > 8 and body_line.strip():
                needs_fix = True
                break
        
        if needs_fix:
            fixed_body = fix_func_indent(body)
            result_parts.append(content[pos:body_start])
            result_parts.append(fixed_body)
            result_parts.append(content[quote_end:rest_start])
            fixed_count += 1
            print(f"Fixed: {content[idx+len(marker):idx+len(marker)+60].strip()}")
        else:
            result_parts.append(content[pos:rest_start])
    else:
        result_parts.append(content[pos:rest_start])
    
    pos = rest_start

result = ''.join(result_parts)

print(f"\nTotal fixed: {fixed_count} blocks")

# Verify syntax
import py_compile
try:
    py_compile.compile('registry.py', doraise=True)  # dummy compile of result
except py_compile.PyCompileError as e:
    print(f"Syntax ERROR before write: {e}")
    exit(1)

with open('registry.py', 'w', encoding='utf-8') as f:
    f.write(result)

# Final compile check
try:
    py_compile.compile('registry.py', doraise=True)
    print("Final syntax: OK")
except py_compile.PyCompileError as e:
    print(f"Final syntax ERROR: {e}")
