# -*- coding: utf-8 -*-
"""MEMORY.md v45 补丁: 补入「写侧 newline='' ⇒ 静默转 LF」(本轮亲身踩到)。

背景: 我用 `io.open(PROP,'w',newline='')` 写提案账本 ⇒ 整文件 CRLF → **纯 LF**
  (CRLF 6209 → 0)。同族于 memory 已记的「Edit 会整文件转 LF」, 但机理不同:
  根因是 `json.dumps` 只产出 `\\n`, 而 `newline=''` 在**写侧不做任何翻译**。
  ⇒ 要写 CRLF 必须 `newline='\\r\\n'`。此规则每个 `_add_p*.py` 模板都需要。

腾空间: 移出「多通道入库」条目 —— 它已在 `_fuxi_landing_details.md`
  「### 多通道因子入库判据 (09-16 立)」有专属小节 (合规外链), 且不属于
  「违反即**静默**出错」(它导致结论错, 不是数字静默错)。
"""
from pathlib import Path

MP = Path(r'e:/quant/.workbuddy/memory/MEMORY.md')

OLD_FOOT = '**读须 `newline=\'\'`**。🔴 **Edit 会整文件转 LF**（静默）⇒ 必回读。'
NEW_FOOT = ('**读须 `newline=\'\'`**。🔴 **写侧 `newline=\'\'` ⇒ 静默转 LF**'
            '（`json.dumps` 只产 `\\n`；要 CRLF 须 `newline=\'\\r\\n\'`）。'
            '🔴 **Edit 会整文件转 LF** ⇒ 必回读。')

OLD_DROP = ('- **多通道入库**：必做**单通道剥离消融 + 逐截面配对 t**'
            '（通过≠归因正确）。\n')

raw = MP.read_bytes()
crlf, bare = raw.count(b'\r\n'), raw.count(b'\n') - raw.count(b'\r\n')
print(f'MEMORY.md: bytes={len(raw)} CRLF={crlf} 裸LF={bare}')
assert bare == 0, '🔴 入参文件已有裸 LF'
with open(MP, 'r', encoding='utf-8') as fh:
    t = fh.read()


def nw(s):
    return len(''.join(s.split()))


n0 = nw(t)
print('入参去空白 =', n0)

assert t.count(OLD_FOOT) == 1, '🔴 行尾锚点不唯一/未命中'
t = t.replace(OLD_FOOT, NEW_FOOT, 1)
print('  [+] 补入 newline 写侧机理 +%d' % (nw(NEW_FOOT) - nw(OLD_FOOT)))

assert t.count(OLD_DROP) == 1, '🔴 多通道锚点不唯一/未命中'
t = t.replace(OLD_DROP, '', 1)
print('  [-] 移出「多通道入库」条目 (已外链详情 §多通道因子入库判据) -%d'
      % nw(OLD_DROP))

with open(MP, 'w', encoding='utf-8', newline='\r\n') as fh:
    fh.write(t)

b = MP.read_bytes()
c2, ba2 = b.count(b'\r\n'), b.count(b'\n') - b.count(b'\r\n')
with open(MP, 'r', encoding='utf-8') as fh:
    t2 = fh.read()
n1 = nw(t2)
print()
print(f'回读: bytes={len(b)} CRLF={c2} 裸LF={ba2}')
print(f'去空白: {n0} -> {n1} (delta {n1-n0})  余量 {3000-n1}')
assert ba2 == 0, '🔴 引入裸 LF'
assert c2 > 0, '🔴 行尾风格被改变'
for k in ['newline=\'\\r\\n\'', '静默转 LF', '多通道入库判据']:
    if k == '多通道入库判据':
        continue
    assert k in t2, f'🔴 缺锚点 {k}'
assert '多通道入库' not in t2, '🔴 待移出条目仍在'
for k in ['d_plus_*', '7 类', 'legacy_e2e', '同族须穷举', '2026 腿']:
    assert k in t2, f'🔴 旧锚点丢失 {k}'
assert n1 <= 3000, '🔴 超限'
print('PASS: 机理已补入, 移出项已确认不在, 旧锚点无丢失')
