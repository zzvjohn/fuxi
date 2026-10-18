import psutil, os

markers = ["run_fa.py", "watchdog_causal_ga", "monitor_causal_ga"]
killed = []
for p in psutil.process_iter(['pid', 'name', 'cmdline', 'ppid']):
    try:
        cl = p.info.get('cmdline') or []
        cmd = " ".join(str(c) for c in cl)
    except Exception:
        continue
    if not cmd:
        continue
    if any(m in cmd for m in markers):
        try:
            proc = psutil.Process(p.info['pid'])
            children = proc.children(recursive=True)
            for c in children:
                try:
                    c.kill()
                except Exception:
                    pass
            proc.kill()
            killed.append((p.info['pid'], p.info['name'], cmd[:90]))
        except Exception as e:
            killed.append((p.info['pid'], 'ERR', str(e)[:60]))

print("KILLED_COUNT=%d" % len(killed))
for k in killed:
    print(k)
# 二次确认
remaining = []
for p in psutil.process_iter(['pid', 'name', 'cmdline']):
    try:
        cmd = " ".join(str(c) for c in (p.info.get('cmdline') or []))
    except Exception:
        continue
    if any(m in cmd for m in markers):
        remaining.append((p.info['pid'], cmd[:90]))
print("REMAINING=%d" % len(remaining))
for r in remaining:
    print(r)
