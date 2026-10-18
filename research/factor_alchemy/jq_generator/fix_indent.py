"""Fix indentation of jq_compute_function blocks in registry.py."""
import re

# Read registry file
with open('registry.py', 'r', encoding='utf-8') as f:
    content = f.read()

def fix_body_indent(body_text):
    """Fix indentation of a function body with try/except blocks."""
    lines = body_text.split('\n')
    fixed = []
    
    state = 'func'  # func, try_body, except_body, post_try
    for line in lines:
        stripped = line.lstrip()
        
        if not stripped:
            fixed.append('')
            continue
        
        # Detect state transitions
        if stripped == 'try:':
            state = 'try_body'
            fixed.append('    try:')
            continue
        
        if stripped.startswith('except'):
            state = 'except_body'
            fixed.append('    except:')
            continue
        
        # After except, detect transition back to function-level code
        if state == 'except_body':
            if not stripped.startswith('return ') and not stripped.startswith('pass'):
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

# Match jq_compute_function with """\n...\n""" multi-line format
pattern1 = re.compile(
    r"('jq_compute_function':\s*\"\"\"\n)(.*?)(\n\"\"\")",
    re.DOTALL
)

# Also handle same-line """...""" or content without leading newline
pattern2 = re.compile(
    r"('jq_compute_function':\s*\"\"\")",
    re.DOTALL
)

def fix_all(content):
    n_fixed = 0
    
    # Method: find each 'jq_compute_function': section, extract content,
    # fix it, replace.
    # We'll use a manual parser since regex with """ is fragile.
    
    marker = "'jq_compute_function':"
    pos = 0
    result_parts = []
    
    while True:
        idx = content.find(marker, pos)
        if idx == -1:
            result_parts.append(content[pos:])
            break
        
        # Copy everything up to and including the marker and opening quotes
        quote_start = content.find('"""', idx + len(marker))
        if quote_start == -1:
            result_parts.append(content[pos:])
            break
        
        # Find closing """
        quote_end = content.find('"""', quote_start + 3)
        if quote_end == -1:
            result_parts.append(content[pos:])
            break
        
        # Extract: everything before marker, then marker+prefix, then content, then closing
        prefix = content[idx:quote_start + 3]  # 'jq_compute_function': """
        body = content[quote_start + 3:quote_end]
        suffix = '"""'
        rest_start = quote_end + 3
        
        result_parts.append(content[pos:idx])
        
        if 'try:' in body:
            fixed_body = fix_body_indent(body)
            # Ensure proper wrapping
            if fixed_body.startswith('\n'):
                result_parts.append(prefix + fixed_body + '\n' + suffix)
            else:
                result_parts.append(prefix + '\n' + fixed_body + '\n' + suffix)
            n_fixed += 1
        else:
            result_parts.append(content[idx:rest_start])
        
        pos = rest_start
    
    result = ''.join(result_parts)
    print(f"Fixed {n_fixed} jq_compute_function blocks")
    return result

result = fix_all(content)

with open('registry.py', 'w', encoding='utf-8') as f:
    f.write(result)

# Verify
import py_compile
try:
    py_compile.compile('registry.py', doraise=True)
    print("Syntax OK")
except py_compile.PyCompileError as e:
    print(f"Syntax ERROR: {e}")
