"""Fix jq_compute_function indentation using autopep8.

autopep8 properly handles Python indentation, including nested blocks,
try/except, for/if loops, etc. We run each function body through autopep8
and verify the result compiles.
"""
import autopep8, sys
from pathlib import Path

def fix_function(code_text):
    """Fix indentation using autopep8. Returns fixed code."""
    if not code_text.strip():
        return code_text
    
    # Try fixing with autopep8 (aggressive mode to fix all issues)
    try:
        fixed = autopep8.fix_code(
            code_text,
            options={
                'aggressive': 3,  # Most aggressive fixing
                'indent_size': 4,
            }
        )
    except Exception:
        return code_text
    
    # Verify the fixed code compiles
    try:
        compile(fixed, '<test>', 'exec')
    except (SyntaxError, IndentationError):
        # Try with slightly different options
        try:
            fixed = autopep8.fix_code(
                code_text,
                options={
                    'aggressive': 2,
                    'indent_size': 4,
                    'max_line_length': 200,
                }
            )
            compile(fixed, '<test>', 'exec')
        except Exception:
            return code_text  # Can't fix, return original
    
    return fixed


def fix_file(filepath):
    """Fix all jq_compute_function '''...''' blocks in a file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    marker = "'jq_compute_function': '''"
    pos = 0
    fixed = 0
    skipped = 0
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
                skipped += 1
        else:
            result_parts.append(text[pos:qe+3])
        
        pos = qe + 3
    
    result = ''.join(result_parts)
    return result, fixed, skipped


if __name__ == '__main__':
    for fname in ['registry.py', 'registry_generated.py']:
        fp = Path(__file__).parent / fname
        if not fp.exists():
            continue
        
        print(f"\n=== {fname} ===")
        result, fixed, skipped = fix_file(str(fp))
        
        with open(fp, 'w', encoding='utf-8') as f:
            f.write(result)
        
        import py_compile
        try:
            py_compile.compile(str(fp), doraise=True)
            print(f"  Syntax OK — Fixed: {fixed}, Skipped: {skipped}")
        except py_compile.PyCompileError as e:
            print(f"  Syntax ERROR: {e}")
    
    print("\nDone!")
