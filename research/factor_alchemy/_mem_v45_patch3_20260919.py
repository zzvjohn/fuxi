# -*- coding: utf-8 -*-
"""压缩 MEMORY.md 的「行尾」条目表述 (去掉 json.dumps 机理说明, 保留可执行规则)。

🔴 为什么单独落文件而不是内联 python -c:
  目标文本含**反引号** ⇒ 内联 shell 会先做命令替换 ⇒ **静默掏空**
  (本次实测: 锚点命中 0 次)。走文件通道不经 shell 解析。
"""
from pathlib import Path

MP = Path(r'e:/quant/.workbuddy/memory/MEMORY.md')
OLD = '（`json.dumps` 只产 `\\n`；要 CRLF 须 `newline=\'\\r\\n\'`）'
NEW = '（CRLF 须 `newline=\'\\r\\n\'`）'

with open(MP, 'r', encoding='utf-8') as fh:
    t = fh.read()
n = t.count(OLD)
print('锚点命中', n, '次')
assert n == 1, f'🔴 锚点不唯一 ({n})'

t = t.replace(OLD, NEW, 1)
with open(MP, 'w', encoding='utf-8', newline='\r\n') as fh:
    fh.write(t)

b = MP.read_bytes()
c2 = b.count(b'\r\n')
ba2 = b.count(b'\n') - c2
with open(MP, 'r', encoding='utf-8') as fh:
    t2 = fh.read()
nw = lambda s: len(''.join(s.split()))
print(f'回读: bytes={len(b)} CRLF={c2} 裸LF={ba2}')
print(f'去空白 = {nw(t2)} / 3000   余量 {3000 - nw(t2)}')
assert ba2 == 0, '🔴 引入裸 LF'
assert c2 > 0, '🔴 行尾风格变了'
for k in ["newline='\\r\\n'", '静默转 LF', 'd_plus_*', '7 类', 'legacy_e2e']:
    assert k in t2, f'🔴 缺锚点 {k}'
print()
print('--- 行尾条目 ---')
for ln in t2.splitlines():
    if ln.startswith('- **行尾**'):
        print(ln)
print()
print('--- 关键结论节 ---')
sec = False
for ln in t2.splitlines():
    if ln.startswith('## 🔴 关键结论'):
        sec = True
    elif sec and ln.startswith('## '):
        break
    if sec:
        print(ln)
print('PASS')
