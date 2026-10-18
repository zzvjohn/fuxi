import re, textwrap

def audit(path):
    print("="*60); print("COMPILE-AUDIT:", path); print("="*60)
    src = open(path, encoding='utf-8').read()
    pat = re.compile(r"'jq_compute_function':\s*'''(.*?)'''", re.DOTALL)
    bad = 0
    for m in pat.finditer(src):
        block = m.group(1)
        fn = re.search(r'def (\w+)\(', block)
        name = fn.group(1) if fn else '?'
        stripped = block.strip()
        ok = True; err = None
        try:
            compile(stripped, '<%s>'%name, 'exec')
        except IndentationError as e:
            ok = False; err = "Indent L%d:%s" % (e.lineno, e.msg)
        except SyntaxError as e:
            ok = False; err = "Syntax L%d:%s" % (e.lineno, e.msg)
        if not ok:
            bad += 1
            print(f"  <<< {name:32s} {err}")
    print(f"  >>> BROKEN BLOCKS: {bad}")
    return bad

audit('jq_generator/registry.py')
audit('jq_generator/registry_generated.py')
