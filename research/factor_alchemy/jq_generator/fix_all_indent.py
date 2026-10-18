"""Fix ALL jq_compute_function indentation issues in both registry.py and registry_generated.py.

Approach: parse each function body with Python's tokenizer, rebuild with correct indentation.
"""
import tokenize, io, sys
from pathlib import Path

def fix_python_code(code_text):
    """Fix indentation of a Python function body using tokenizer.
    Returns the fixed code or original if it's already valid.
    """
    # First, check if code is already valid by wrapping in a dummy function
    try:
        compile("def _dummy():\n" + code_text, "<test>", "exec")
        return code_text  # Already valid
    except (SyntaxError, IndentationError):
        pass
    
    # Normalize: strip all indentation, re-add based on structure
    lines = code_text.split('\n')
    
    # Remove leading whitespace from all lines
    stripped = [ln.lstrip() for ln in lines]
    
    # Use tokenize to determine indentation levels
    # We'll rebuild by tracking INDENT/DEDENT tokens
    test_code = "def _dummy():\n" + '\n'.join(' ' + ln for ln in stripped if ln)
    
    indent_stack = [0]
    result_lines = []
    
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(test_code).readline))
        current_indent = 0
        line_tokens = {}
        
        for tok in tokens:
            line_no = tok.start[0]
            if line_no not in line_tokens:
                line_tokens[line_no] = []
            line_tokens[line_no].append(tok)
        
        # Process each line
        for line_no in sorted(line_tokens.keys()):
            if line_no == 1:  # Skip def _dummy
                continue
            
            toks = line_tokens[line_no]
            
            # Get the line text
            line_text = test_code.split('\n')[line_no - 1] if line_no - 1 < len(test_code.split('\n')) else ''
            
            # Handle INDENT/DEDENT
            has_indent = any(t.type == tokenize.INDENT for t in toks)
            has_dedent = any(t.type == tokenize.DEDENT for t in toks)
            
            if has_dedent:
                if indent_stack:
                    indent_stack.pop()
                if indent_stack:
                    current_indent = indent_stack[-1]
                
            if has_indent:
                indent_stack.append(current_indent + 4)
                current_indent = indent_stack[-1]
            
            # Get the stripped content
            content = line_text.strip()
            if not content:
                result_lines.append('')
            else:
                result_lines.append(' ' * current_indent + content)
    
    except tokenize.TokenError:
        # Fallback: use simpler heuristic
        return fix_with_heuristic(code_text)
    
    return '\n'.join(result_lines)


def fix_with_heuristic(code_text):
    """Simple indentation fix based on colon-based block tracking."""
    lines = code_text.split('\n')
    stripped = [ln.lstrip() for ln in lines]
    
    result = []
    indent = 4  # Function body starts at 4 spaces
    stack = []  # Stack of indent levels
    
    for i, line in enumerate(stripped):
        if not line:
            result.append('')
            continue
        
        # Check if previous line opened a block (ends with :)
        prev_opened = False
        if i > 0:
            prev_clean = stripped[i-1].rstrip()
            prev_opened = prev_clean.endswith(':') and not prev_clean.startswith('#')
        
        # Dedent: if current line is a keyword that typically ends a block
        # or if we detect we're at a lower nesting level
        is_return = line.startswith('return ') or line == 'return'
        is_continue = line == 'continue' or line.startswith('continue')
        is_break = line == 'break' or line.startswith('break')
        is_pass = line == 'pass' or line.startswith('pass')
        is_except = line.startswith('except')
        is_elif = line.startswith('elif ')
        is_else = line.startswith('else:')
        
        # Dedent handling
        if is_except or is_elif or is_else:
            if stack:
                # These go back to the try/if level
                while len(stack) > 1:
                    stack.pop()
                indent = stack[-1] if stack else 4
        
        # Add line with current indent
        if is_except or is_elif or is_else:
            # These are at parent level
            result.append(' ' * indent + line)
        elif is_return or is_continue or is_break or is_pass:
            # These need to be at current binding level
            current = stack[-1] if stack else indent
            result.append(' ' * current + line)
        else:
            result.append(' ' * indent + line)
        
        # Push new block if this line ends with :
        if line.rstrip().endswith(':') and not line.startswith('#'):
            # Don't count slice [:] as block starter
            if not line.rstrip().endswith('[:]'):
                stack.append(indent)
                indent += 4
    
    return '\n'.join(result)


def fix_file(filepath):
    """Fix all jq_compute_function blocks in a file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Find '''...''' blocks after 'jq_compute_function':
    pos = 0
    fixed = 0
    marker_3q = "'jq_compute_function': '''"
    result_parts = []
    
    while True:
        idx = text.find(marker_3q, pos)
        if idx == -1:
            break
        
        body_start = idx + len(marker_3q)
        if text[body_start:body_start+1] == '\n':
            body_start += 1
        
        qe = text.find("'''", body_start)
        if qe == -1:
            result_parts.append(text[pos:])
            pos = len(text)
            break
        
        body_raw = text[body_start:qe]
        
        if body_raw.strip():
            fixed_body = fix_python_code(body_raw)
            if fixed_body != body_raw:
                result_parts.append(text[pos:body_start])
                result_parts.append(fixed_body)
                result_parts.append("'''")
                fixed += 1
            else:
                result_parts.append(text[pos:qe+3])
        else:
            result_parts.append(text[pos:qe+3])
        
        pos = qe + 3
    
    if pos < len(text):
        result_parts.append(text[pos:])
    
    result = ''.join(result_parts)
    
    # Verify
    import py_compile
    try:
        py_compile.compile(filepath, doraise=True)
    except py_compile.PyCompileError as e:
        print(f"  WARN: File syntax error after fix: {e}")
        return result, fixed
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(result)
    
    print(f"  Fixed {fixed} blocks")
    return result, fixed


if __name__ == '__main__':
    # Fix both files
    for fname in ['registry.py', 'registry_generated.py']:
        fp = Path(__file__).parent / fname
        if fp.exists():
            print(f"\n=== {fname} ===")
            fix_file(str(fp))
    
    print("\nDone!")
