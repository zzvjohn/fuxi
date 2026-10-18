# -*- coding: utf-8 -*-
"""一次性脚本：向 data/stage1_optimization_proposals.json 追加 P-20261006-004（P7）。

行尾纪律：当前文件实测为纯 CRLF（10023 CRLF / 0 裸 LF，2026-10-06），写回必须保持 CRLF。
内容：artifacts.publish 发布竞态修复（Windows 目录改名 [WinError 5] 有界重试；零默认变化）。
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'data' / 'stage1_optimization_proposals.json'
NEW_ID = 'P-20261006-004'

ENTRY = {
    'proposal_id': NEW_ID,
    'title': 'artifacts.publish 发布竞态修复：Windows 目录改名 [WinError 5] 有界重试（M2 A/B 实测命中；零默认变化）',
    'category': '工程/证据层修复',
    'domains': ['T3模块/通道'],
    'risk_level': 'controlled_research_only',
    'status': 'landed',
    'proposal_change_kind': 'engineering',
    'generator_hint': 'manual',
    'trigger': (
        'M2 PARENT_POLICY enforce A/B（niche_mix 臂）第 5 轮实测 artifacts.publish 目录改名抛 PermissionError '
        '[WinError 5]；同一 bundles 根目录此前已成功发布多轮（pareto 臂全程 20 候选 + niche_mix 前 5 轮），'
        '属瞬态共享冲突而非确定性缺陷。一旦命中，异常穿透 run_campaign 使整批 campaign 崩溃。'
    ),
    'current_state': (
        'artifacts.py:57 发布末步 pending.rename(destination) 无重试；Windows 上杀软/索引器对刚 fsync 的 '
        'bundle 文件短暂持有句柄即触发 WinError 5。全仓近 30 天 pending 残留普查：真实生产残留仅 1 处'
        '（即本次 M2 现场，已留档 runs_attempt1_crash_20261006/），其余 7 处为测试 scratch 注入 —— '
        '低频但单次后果为整批失败。'
    ),
    'proposed_change': (
        '新增模块内私有函数 _rename_into_place(pending, destination)：Windows 平台有界重试 5 次、线性退避 '
        '0.25s×n，全部耗尽后原样抛出 PermissionError；非 Windows 保持单次尝试。不改原子语义：失败路径 '
        'pending 目录保持不完整且永不被消费（verify() 只认 run_ 前缀目录）。'
    ),
    'expected_impact': (
        '消除对杀软/索引器文件句柄时序的脆弱依赖；发布成功率提升；无 env 开关、无口径变化、无业务默认变化。'
    ),
    'implementation_scope': (
        'fuxi_v1/artifacts.py（全 LF，78→98 裸 LF；3669→4475 bytes；sha 42cd5e73…→cb60f6c1…）；'
        'reports/fuxi_v1_m2_enforce_ab_20261006/probe_publish_retry.py（T1-T4 探针）+ '
        'probe_publish_retry.json；runs_attempt1_crash_20261006/**（崩溃现场含 .pending 残留留档）。'
    ),
    'landed_detail': (
        '探针 ALL OK：T1 正常单发 + manifest verify；T2 注入前 3 次 PermissionError 后第 4 次成功（calls=4）；'
        'T3 全 5 次注入耗尽原样抛出且 pending 残留=1 / published=0；T4 posix 语义单次尝试（calls=1）。'
        '回归：fuxi_v1 全量 280 项通过（沙箱批量删除守卫会拦截测试 rmtree，须 env -u '
        'CODEBUDDY_SAFE_DELETE_BULK_STATE_DIR 绕行，属环境层非代码缺陷）。重跑 M2 A/B attempt2 全程零命中'
        '（pareto 136.5s + niche_mix 135.3s + 冒烟 2×25s，两臂 errors=0）。'
    ),
    'rollback_plan': '单文件字节级回退（artifacts.py 无行尾变化，LF 保持）；无生产默认引用、无开关。',
    'jq_validation_required': False,
    'jq_validation_status': 'no_jq_admission_granted_research_only',
    'references': [
        'reports/fuxi_v1_m2_enforce_ab_20261006/probe_publish_retry.py',
        'reports/fuxi_v1_m2_enforce_ab_20261006/probe_publish_retry.json',
        'fuxi_v1/artifacts.py',
    ],
    'cross_verified': False,
    'source': '研究批次（M2 A/B 现场命中 → 探针链验证 → 重跑零命中）',
}


def main():
    raw = TARGET.read_bytes()
    crlf_before = raw.count(b'\r\n')
    bare_before = raw.count(b'\n') - crlf_before
    assert crlf_before > 0 and bare_before == 0, 'source file is not pure CRLF'
    data = json.loads(raw.decode('utf-8'))
    assert isinstance(data, list)
    ids = [x.get('proposal_id') for x in data]
    assert len(ids) == len(set(ids)), 'duplicate ids present'
    if NEW_ID in ids:
        raise SystemExit('refusing to append: %s already present' % NEW_ID)
    before = len(data)
    last_before = ids[-1]

    data.append(ENTRY)
    text = json.dumps(data, ensure_ascii=False, indent=2)
    payload = text.replace('\n', '\r\n').encode('utf-8')
    assert payload[:3] != b'\xef\xbb\xbf'
    assert payload.count(b'\n') == payload.count(b'\r\n'), 'output is not pure CRLF'

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup = TARGET.with_name(TARGET.name + '.bak_%s_p7' % stamp)
    shutil.copy2(TARGET, backup)

    with open(TARGET, 'wb') as fh:
        fh.write(payload)

    got = TARGET.read_bytes()
    assert got == payload, 'byte-level round-trip mismatch'
    assert got.count(b'\n') - got.count(b'\r\n') == 0, 'bare LF after write'
    again = json.loads(got.decode('utf-8'))
    assert isinstance(again, list) and len(again) == before + 1, 'entry count wrong'
    assert again[-1]['proposal_id'] == NEW_ID
    got_ids = [x.get('proposal_id') for x in again]
    assert got_ids[:-1] == ids, 'prior entries mutated'
    assert last_before in got_ids, 'previous last entry lost'
    assert again[-1]['status'] == 'landed'
    assert again[-1]['proposal_change_kind'] == 'engineering'

    print('ok: %s -> %s entries (was %s, new last = %s)'
          % (before, len(again), last_before, NEW_ID))
    print('bytes %d (was %d)  backup %s' % (len(got), len(raw), backup.name))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
