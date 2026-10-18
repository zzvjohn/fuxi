import re, ast, textwrap

def audit(path):
    print("="*70)
    print("AUDIT:", path)
    print("="*70)
    src = open(path, encoding='utf-8').read()
    pat = re.compile(r"'jq_compute_function':\s*'''(.*?)'''", re.DOTALL)
    n_bad = 0
    for m in pat.finditer(src):
        block = m.group(1)
        fn = re.search(r'def (\w+)\(', block)
        name = fn.group(1) if fn else '?'
        stripped = block.strip()
        sim_ok = True; sim_err = None
        try:
            ast.parse(stripped)
        except SyntaxError as e:
            sim_ok = False; sim_err = f"L{e.lineno}:{e.msg}"
        ded = textwrap.dedent(block)
        ded_ok = True
        try:
            ast.parse(ded.strip())
        except SyntaxError as e:
            ded_ok = False
        key_line = src[:m.start()].splitlines()[-1]
        key_indent = len(key_line) - len(key_line.lstrip())
        flag = "" if (sim_ok and ded_ok) else "  <<< PROBLEM"
        if not (sim_ok and ded_ok):
            n_bad += 1
        print(f"  {name:30s} key_ind={key_indent:2d} strip_ok={sim_ok} {sim_err or ''} dedent_ok={ded_ok}{flag}")
    print(f"  >>> TOTAL PROBLEM BLOCKS: {n_bad}")
    return n_bad

audit('jq_generator/registry.py')
audit('jq_generator/registry_generated.py')
