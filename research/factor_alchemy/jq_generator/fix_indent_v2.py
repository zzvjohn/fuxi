"""Fix jq_compute_function indentation using Python's tokenizer.
The previous fix_all_indent.py normalized everything to uniform 4-space indent,
losing nested block structure. This version uses the tokenizer's INDENT/DEDENT
tokens to properly reconstruct nested indentation.
"""
import tokenize, io, sys
from pathlib import Path

def fix_function(code_text):
    """Re-indent a Python function body using tokenizer INDENT/DEDENT.
    Assumes code_text starts with a function definition and is the ONLY
    top-level construct.
    """
    # Clean up: remove leading blank lines
    code_text = code_text.lstrip('\n')
    if not code_text.strip():
        return code_text
    
    try:
        # Try to tokenize the code as-is
        tokens = list(tokenize.generate_tokens(io.StringIO(code_text).readline))
    except (tokenize.TokenError, IndentationError):
        # If tokenization fails, try with uniform dedent
        import textwrap
        dedented = textwrap.dedent(code_text)
        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(dedented).readline))
        except (tokenize.TokenError, IndentationError):
            return code_text  # Can't fix
    
    if not tokens:
        return code_text
    
    # Build line → (indent_level, content) mapping
    # Track INDENT/DEDENT to determine correct nesting
    INDENT = tokenize.INDENT
    DEDENT = tokenize.DEDENT
    NL = tokenize.NL
    NEWLINE = tokenize.NEWLINE
    ENDMARKER = tokenize.ENDMARKER
    COMMENT = tokenize.COMMENT
    
    indent_stack = [0]  # Stack of indent levels
    result_parts = []
    last_line = -1
    line_content = {}
    line_newline_type = {}  # NEWLINE vs NL vs COMMENT
    
    for tok in tokens:
        line = tok.start[0]
        if line != last_line:
            if line not in line_content:
                line_content[line] = []
            last_line = line
        
        # Skip INDENT/DEDENT/NEWLINE/NL for content building
        if tok.type not in (INDENT, DEDENT, NEWLINE, NL, ENDMARKER):
            line_content.setdefault(line, []).append(tok)
        
        if tok.type == NEWLINE or tok.type == NL:
            line_newline_type[line] = tok.type
    
    # Now rebuild: process lines in order, tracking indent level
    # Start with the first meaningful line
    for tok in tokens:
        line_no = tok.start[0]
        col = tok.start[1]
        
        if tok.type == INDENT:
            indent_stack.append(indent_stack[-1] + 4)
            continue
        
        if tok.type == DEDENT:
            if len(indent_stack) > 1:
                indent_stack.pop()
            continue
    
    # Actually, let's use a simpler approach:
    # 1. Parse the code with ast, get the proper structure
    # 2. Use unparse to get correctly-indented code
    
    import ast
    try:
        # Try to parse the code directly
        tree = ast.parse(code_text)
        import ast as ast_module
        fixed = ast_module.unparse(tree)
        return fixed
    except (SyntaxError, IndentationError):
        pass
    
    # Fallback: token-based reconstruction
    try:
        return _fix_via_tokens(code_text)
    except Exception:
        return code_text


def _fix_via_tokens(code_text):
    """Reconstruct code with proper indentation using token stream."""
    import textwrap
    
    # Remove all indentation first
    lines = code_text.split('\n')
    stripped = [ln.lstrip() for ln in lines]
    
    # Build pure-indentation-free code for tokenization
    flat_code = '\n'.join(stripped)
    
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(flat_code).readline))
    except (tokenize.TokenError, IndentationError):
        return code_text
    
    INDENT = tokenize.INDENT
    DEDENT = tokenize.DEDENT
    NEWLINE = tokenize.NEWLINE
    NL = tokenize.NL
    ENDMARKER = tokenize.ENDMARKER
    
    # Collect lines with their indent levels
    result = []
    indent = 0
    last_line = -1
    line_texts = {}
    
    # First pass: collect all line content
    for tok in tokens:
        line = tok.start[0]
        if tok.type not in (INDENT, DEDENT, NEWLINE, NL, ENDMARKER):
            if line not in line_texts:
                line_texts[line] = ''
            # Reconstruct the line
            if tok.type == tokenize.STRING:
                line_texts[line] += tok.string
            else:
                line_texts[line] += tok.string
    
    # Second pass: assign indent levels
    indent_stack = [0]
    line_indents = {}
    
    for tok in tokens:
        line = tok.start[0]
        if tok.type == INDENT:
            indent_stack.append(indent_stack[-1] + 4)
        elif tok.type == DEDENT:
            if len(indent_stack) > 1:
                indent_stack.pop()
        elif tok.type not in (NEWLINE, NL, ENDMARKER):
            if line not in line_indents:
                line_indents[line] = indent_stack[-1]
    
    # Build result
    max_line = max(line_texts.keys()) if line_texts else 0
    for line_no in range(1, max_line + 1):
        text = line_texts.get(line_no, '')
        indent = line_indents.get(line_no, 0)
        if text.strip():
            result.append(' ' * indent + text)
        else:
            result.append('')
    
    return '\n'.join(result)


def fix_file(filepath):
    """Fix all jq_compute_function blocks in a Python file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Find all '''...''' blocks after 'jq_compute_function':
    marker = "'jq_compute_function': '''"
    pos = 0
    fixed = 0
    result_parts = []
    
    while True:
        idx = text.find(marker, pos)
        if idx == -1:
            result_parts.append(text[pos:])
            break
        
        body_start = idx + len(marker)
        if text[body_start:body_start+1] == '\n':
            body_start += 1
        
        qe = text.find("'''", body_start)
        if qe == -1:
            result_parts.append(text[pos:])
            break
        
        body_raw = text[body_start:qe]
        
        if body_raw.strip():
            fixed_body = fix_function(body_raw)
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
    
    result = ''.join(result_parts)
    return result, fixed


if __name__ == '__main__':
    for fname in ['registry.py', 'registry_generated.py']:
        fp = Path(__file__).parent / fname
        if not fp.exists():
            continue
        
        print(f"=== {fname} ===")
        result, fixed = fix_file(str(fp))
        
        with open(fp, 'w', encoding='utf-8') as f:
            f.write(result)
        
        # Verify syntax
        import py_compile
        try:
            py_compile.compile(str(fp), doraise=True)
            print(f"  Syntax OK, fixed {fixed} blocks")
        except py_compile.PyCompileError as e:
            print(f"  Syntax ERROR: {e}, fixed {fixed} blocks")
    
    print("\nDone!")
