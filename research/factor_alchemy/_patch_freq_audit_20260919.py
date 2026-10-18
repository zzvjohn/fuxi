# -*- coding: utf-8 -*-
"""P-20260919-001 落地: `_freq_audit` 三处同步进 schema (dataclass + to_dict + from_dict)。

根因 (2026-09-19 实测):
  `freq_audit.py:147` 写 `e["_freq_audit"]` 到**原始 dict**, 再 json.dump 落盘。
  该键**不在** MiningTrajectory 白名单 ⇒ 下一轮任意 `_save()` 整对象重建即静默抹除。
  实测: 跑中 1 条 → 跑后 **0 条** (`lessons` 键频同轮由 0 → 2302, 证明 `_save()` 真跑过)。

本补丁只做**最小安全修复**: 三处同步 (与 9-18 `d_plus_*` 同法)。
  不做的事 (刻意): 不把 freq_audit 改道 TrajectoryLogger —— 那会把整份 2300 条
  重新 to_dict 序列化 (indent 1→2, 全文件 diff), 需完整 A/B 才可动, 见报告"后续"。

行尾: 🔴 **按文件实测** —— trajectory_logger.py / freq_audit.py 均为**纯 LF**
  (CRLF=0), 故本补丁保持 LF, 不转 CRLF (避免整文件 diff)。

🔴 行尾不变量 (本脚本 v2 更正):
  补丁会**新增行** ⇒ 裸 LF 计数**必然增加**, 故「裸 LF 不增加」在这里是**假 FAIL**
  (与 memory 里「写死 ==0 ⇒ 永远 FAIL」同族)。
  **正确不变量 = CRLF 计数不变 + 不引入 `\\r`** (即文件的行尾风格不变)。
"""
import py_compile
from pathlib import Path

ROOT = Path(r'e:/quant/research/factor_alchemy')

PATCHES = {
    'trajectory_logger.py': [
        (
            '    # v0.7 频率对称: S1 裁决口径标签 ("daily" | "weekly"), D+ 蒸馏按频率归因\n'
            '    natural_freq: str = "daily"\n',
            '    # v0.7 频率对称: S1 裁决口径标签 ("daily" | "weekly"), D+ 蒸馏按频率归因\n'
            '    natural_freq: str = "daily"\n'
            '\n'
            '    # 2026-09-19 (P-20260919-001): freq_audit 铁证修正的**归属留痕**。\n'
            '    # 此前由 freq_audit.py:147 直写原始 dict ⇒ 非白名单键 ⇒ 下一轮 _save()\n'
            '    # 整对象重建即**静默抹除** (实测: 跑中 1 条 → 跑后 0 条)。\n'
            '    # 🔴 与 d_plus_* 同法: 新字段必须**三处同步** —— dataclass / to_dict /\n'
            '    #    from_dict (只补 to_dict ⇒ 下一轮 _load() 又丢)。\n'
            '    _freq_audit: str = ""\n',
        ),
        (
            '            "natural_freq": self.natural_freq,  # v0.7 频率对称\n',
            '            "natural_freq": self.natural_freq,  # v0.7 频率对称\n'
            '            "_freq_audit": self._freq_audit,  # 2026-09-19 三处同步 #2/3\n',
        ),
        (
            '            natural_freq=d.get("natural_freq", "daily"),  # v0.7 频率对称\n',
            '            natural_freq=d.get("natural_freq", "daily"),  # v0.7 频率对称\n'
            '            _freq_audit=d.get("_freq_audit", "") or "",  # 2026-09-19 三处同步 #3/3\n',
        ),
    ],
    'freq_audit.py': [
        (
            '  3. 幂等: freq 已是 weekly 的条目跳过; 修正写 _freq_audit 字段留痕。\n',
            '  3. 幂等: freq 已是 weekly 的条目跳过; 修正写 _freq_audit 字段留痕。\n'
            '     🔴 该键必须同时在 MiningTrajectory 三处 (dataclass/to_dict/from_dict)\n'
            '        出现, 否则下一轮 _save() 会**静默抹除** (P-20260919-001, 已修)。\n',
        ),
    ],
}


def main():
    for fname, subs in PATCHES.items():
        p = ROOT / fname
        raw = p.read_bytes()
        crlf = raw.count(b'\r\n')
        bare = raw.count(b'\n') - crlf
        nl = '\r\n' if (crlf > 0 and bare == 0) else '\n'
        print('=' * 74)
        print(f'{fname}: bytes={len(raw)} CRLF={crlf} bareLF={bare} -> 目标行尾 {nl!r}')
        t = raw.decode('utf-8')
        applied = 0
        for old, new in subs:
            if new in t:
                print(f'  [SKIP 已落地] {old.strip()[:66]}')
                continue
            n = t.count(old)
            assert n == 1, f'🔴 锚点不唯一 ({n}): {old[:60]!r}'
            t = t.replace(old, new, 1)
            applied += 1
            print(f'  [APPLY] {old.strip()[:66]}')
        if applied == 0:
            print('  无待落地项 (幂等)')
        else:
            with open(p, 'w', encoding='utf-8', newline=nl) as fh:
                fh.write(t)
        # ── 回读 + 正确的不变量 ──
        b2 = p.read_bytes()
        c2 = b2.count(b'\r\n')
        ba2 = b2.count(b'\n') - c2
        t2 = b2.decode('utf-8')
        style_ok = (c2 == crlf) and (c2 > 0 or b'\r' not in b2)
        print(f'  回读: CRLF={c2} (原 {crlf}) 裸LF={ba2} (原 {bare}, 新增行 += {ba2-bare}) '
              f'行尾风格不变={style_ok}')
        assert style_ok, '🔴 行尾风格被改变 (引入了 \\r 或 CRLF 计数变化)'
        for old, new in subs:
            assert new in t2, '🔴 补丁未落盘'
            # 注意: 本补丁是「在锚点后追加」⇒ old 必然仍是 new 的子串,
            # 故不能断言 `old not in t2` (那是假 FAIL)。正确检查: 锚点仍恰 1 次。
            assert t2.count(old) == 1, f'🔴 锚点计数异常 {t2.count(old)}'
        print('  PASS')

    print('=' * 74)
    for fname in PATCHES:
        py_compile.compile(str(ROOT / fname), doraise=True)
        print(f'  py_compile OK: {fname}')
    print('ALL PASS')


if __name__ == '__main__':
    main()
