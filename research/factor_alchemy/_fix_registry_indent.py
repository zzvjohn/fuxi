import re, textwrap

def fix_file(path):
    src = open(path, encoding='utf-8').read()
    pat = re.compile(r"('jq_compute_function':\s*''')" + r"(.*?)" + r"(''')", re.DOTALL)
    fixes = 0
    def repl(m):
        nonlocal fixes
        prefix, body, suffix = m.group(1), m.group(2), m.group(3)
        # key line indent
        start = m.start()
        pre = src[:start]
        key_line = pre.splitlines()[-1]
        key_ind = len(key_line) - len(key_line.lstrip())
        ded = textwrap.dedent(body).strip('\n')
        # only rewrite if indented beyond normal
        if body.strip() != ded:
            fixes += 1
        new = prefix + "\n" + ded + "\n" + (" " * key_ind) + suffix
        return new
    new_src = pat.sub(repl, src)
    open(path, 'w', encoding='utf-8').write(new_src)
    print(f"{path}: rewrote {fixes} blocks")

for p in ['jq_generator/registry.py', 'jq_generator/registry_generated.py']:
    fix_file(p)
