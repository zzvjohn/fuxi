# -*- coding: utf-8 -*-
"""更新 version_registry.json 中 v0.7.126~129 的共享 landmark（本日四批全述）。
纪律：纯 CRLF 保持、备份、只改 landmark 字段、回读复核。
"""
import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / 'data' / 'version_registry.json'

NEW_LANDMARK = (
    '本日四批 (v0.7.126/127/128/129; 同日版本共享本 landmark)。首批 v0.7.126 = P2 收尾: M4 经济距离适应度 '
    'shadow (FUXI_V1_FITNESS_TRIAGE 默认 off, 零行为变化) + EvidenceStore WAL 初始化竞态修复 (零默认变化, '
    '复跑 280 项全绿)。第二批 **v0.7.127** = 🔴 M6b 影子计价通道**翻默认** (P-20261006-003, 用户签核): '
    'FUXI_S5_HOLDINGS_MARK enforce→shadow_carry_forward —— 四关: ③a 三轮对拍逐位一致 8/8×3 / ③b 成本正交性 '
    '3 flips 记已知限制 (|Δ|≤0.019, 不涉 passed) / ④a 生产入口 E2E 12/12 / ④b 下游零误计 / ④c 指纹纯净+'
    'revert 字节级; 零晋升 (would_pass=0/167)。落地后现场实测: getter 裸测 6/6 + 裸跑冒烟 2 条 '
    '(measured_shadow, 与④a 对拍 26 字段 0 差异)。翻默认后缺格条目产出 measured_shadow 读数 (passed 恒钉死 '
    'False + 🧪 前缀), S5 覆盖率阻断 142→0 (诊断可读性收益); summarize.py 增 shadow_measured🧪 独立桶。'
    '回退 = 显式 FUXI_S5_HOLDINGS_MARK=enforce。第三批 **v0.7.128** = artifacts.publish 发布竞态修复 '
    '(P-20261006-004, M2 A/B niche 臂现场命中 [WinError 5]): _rename_into_place 有界重试 5 次/线性退避 '
    '0.25s×n/非 Windows 单次; 探针 T1–T4 ALL OK; 回归 280 项 OK; M2 A/B attempt2 重跑零命中; 零默认变化。'
    '第四批 **v0.7.129** = 🔴 **M4 FITNESS_TRIAGE 翻默认** (P-20261006-005, 用户签核): FUXI_V1_FITNESS_TRIAGE '
    'off→enforce —— 四关: ①getter 实测 4 项 / ②A/B R1–R6 全绿 failures=[] / ③默认≡显式 enforce 逐位 '
    '(decisions+results+config 行为面; 无 memory 四档签名一致; 默认≡显式 off) / ④全量 280 项 OK + resume '
    '拒绝/回退可用 + code_hashes 冻结自洽。同锚全 ED 与无 memory 场景零行为变更 (ED=IC/breakeven 单调线性); '
    '收益=排序语义升级 (缺 ED/无读数沉底) + 默认档写 fitness_triage 台账; 混合批次排序变化 by design 已披露。'
    '回退 = 显式 FUXI_V1_FITNESS_TRIAGE=off。同日另: M2 niche_mix 判据双 FAIL 维持 pareto (无版本变更); '
    '停复牌三通道补齐至 20260930 对账 ALL PASS + BJ universe 预注册 (均无版本变更/未动默认)。'
)

TARGETS = {'v0.7.126', 'v0.7.127', 'v0.7.128', 'v0.7.129'}

raw = T.read_bytes()
assert raw.count(b'\n') == raw.count(b'\r\n') and raw[:3] != b'\xef\xbb\xbf', 'not pure CRLF'
data = json.loads(raw.decode('utf-8'))
versions = data['versions']

changed = []
for v in versions:
    if v['version'] in TARGETS:
        assert v.get('date') == '2026-10-06', v['version']
        old = v.get('landmark')
        v['landmark'] = NEW_LANDMARK
        changed.append(v['version'])
        assert old != NEW_LANDMARK, 'no-op for %s' % v['version']
assert sorted(changed) == sorted(TARGETS), changed

text = json.dumps(data, ensure_ascii=False, indent=1)
payload = text.replace('\n', '\r\n').encode('utf-8')
assert payload.count(b'\n') == payload.count(b'\r\n')

stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
backup = T.with_name(T.name + '.bak_%s_landmark129' % stamp)
shutil.copy2(T, backup)

T.write_bytes(payload)
got = T.read_bytes()
assert got == payload
again = json.loads(got.decode('utf-8'))
for v in again['versions']:
    if v['version'] in TARGETS:
        assert v['landmark'] == NEW_LANDMARK
print('ok: landmark updated for %s（四版共享）| bytes %d -> %d | backup %s'
      % (','.join(sorted(changed)), len(raw), len(got), backup.name))
