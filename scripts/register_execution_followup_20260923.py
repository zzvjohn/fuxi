"""Register and seal the bounded first-stage checkpoint without promoting it."""
import io
import json
import shutil
import subprocess
import sys
import unittest
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research/factor_alchemy'))
from run_contract import FuxiWriterLock
from fuxi_execution_lab_20260923 import read, write, digest
import register_remediation_20260922 as registry

OUT = ROOT / 'reports/execution_followup_20260923'


def append_preserving(path, content):
    old = path.read_bytes()
    newline = '\r\n' if b'\r\n' in old else '\n'
    with path.open('ab') as handle:
        handle.write(content.replace('\r\n', '\n').replace('\n', newline).encode('utf-8'))
    assert path.read_bytes().startswith(old)


def main():
    seal_only = '--seal-only' in sys.argv
    assert not (OUT / 'release_audit.json').exists()
    protocol = read(OUT / 'protocol.json')
    validation = read(OUT / 'validation.json')
    reference = read(OUT / 'independent_reference.json')
    assert digest(ROOT / 'scripts/fuxi_reference_cash_20260923.py') == reference['code_sha256']
    assert len(validation['new_actions']) == 4
    assert (OUT / 'validation.json').read_bytes() == (OUT / 'network_retry/validation.json').read_bytes()
    for name, expected in validation['raw_sha256'].items():
        assert digest(OUT / 'network_retry/raw' / name) == expected
    prior = ROOT / 'reports/execution_events_20260923'
    old_audit = read(prior / 'release_audit.json')
    for name, expected in old_audit['evidence_sha256'].items():
        assert digest(prior / name) == expected, name
    market = read(OUT / 'replay/market.json')
    assert len(market['rows']) == 4 and all(r['status'] == 'data_insufficient' for r in market['rows'])
    comparisons = []
    for row in market['rows']:
        name = row['signal']
        old = read(prior / 'replay/ledgers' / (name + '.json'))
        new = read(OUT / 'replay/ledgers' / (name + '.json'))
        n = len(old['days']) - 1
        assert n > 0 and new['days'][:n] == old['days'][:n]
        assert len(new['days']) > len(old['days'])
        comparisons.append({'signal': name, 'unchanged_pre_event_days': n})
    assert sum(r['days'] for r in reference['comparisons']) == 151
    assert sum(r['fills'] for r in reference['comparisons']) == 3853
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'research/factor_alchemy/tests'), pattern='test_reference_cash.py')
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    write(OUT / 'reference_tests.json', {'tests_run': result.testsRun, 'successful': result.wasSuccessful(),
                                       'output': stream.getvalue()})
    assert result.wasSuccessful() and result.testsRun == 3
    existing_tests = read(OUT / 'replay/tests.json')
    assert existing_tests['successful'] and existing_tests['tests_run'] == 75
    for name, expected in existing_tests['code_sha256'].items():
        assert digest(ROOT / name) == expected, name
    for name, expected in protocol['protected'].items():
        if seal_only and name in {'data/version_registry.json', 'data/stage1_optimization_proposals.json'}:
            continue
        assert digest(ROOT / name) == expected, name
    note = ('\n\n## M20260923.6 五阶段继续：现金事件、停牌身份与独立参考账本\n\n'
            '用户再次明确按五项顺序逐步完成。本批P-20260923-006为第一阶段隔离检查点；'
            '五阶段进入条件记录于research/fuxi_five_stage_acceptance_20260923.md，不等于完成。\n\n'
            '16次沙箱ProxyError失败保留，获网络执行审核后同清单16次成功，另2次身份补证。'
            '新增4笔实施现金事件，合计10笔；23未知股日中22个同代码全天停牌有证，剩余920680.BJ通过'
            '映射839680.BJ找到当日停牌，切换生效日仍待核。当前名录L/D/P=5567/339/0，不当PIT标签。\n\n'
            '75项原回归测试加3项独立手算测试；Decimal参考从目标/原始行情/事件自行生成成交，'
            '151有限净值位置及3853成交的股数/现金/应收/净值对账一致；上一批68有限日期位置未变。'
            '只认证现金事件有限前缀，非全窗、红股、摘牌或独立数据源认证。\n\n'
            '四信号仍data_insufficient，分别止于2025-02-10/01-20/04-23/03-05。'
            'LLM1阻断为600898.SH真实摘牌，仍持124000股，供应商名录与交易所公告一致；'
            '不得按末价虚拟退出或补零，后续转让/估值/权益证据待补。其余三条缺601868.SH、002014.SZ、603998.SH公司行动。'
            '全窗目标涉及4766只证券，下一批应做事件库存/日期批采，而不是机械逐证券全扫。\n\n'
            '版本按--bump登记，生产S5/默认/研究反馈/队列未改，定时写入保持false，新JQ0。'
            '报告reports/fuxi_execution_followup_20260923.md；证据reports/execution_followup_20260923/。\n')
    proposal = {
        'proposal_id': 'P-20260923-006', 'title': '执行事件续补与有限前缀独立参考账本核验',
        'category': '执行与生命周期证据', 'domains': ['T2数据口径', 'T3模块/通道'],
        'risk_level': 'high', 'status': 'landed', 'proposal_change_kind': 'shadow',
        'trigger': '用户明确按五建议顺序逐步完成，沿已确认单写入的手动整改继续。',
        'current_state': '四冻结信号因公司行动证据不足停止，第一阶段未验收。',
        'proposed_change': '限定补核四笔现金事件与23未知股日；采集证券名录/代码身份；独立Decimal重算有限现金账本。',
        'implementation_scope': 'scripts/fuxi_execution_followup_20260923.py; scripts/fuxi_reference_cash_20260923.py; scripts/fuxi_bse_state_probe_20260923.py',
        'expected_impact': '隔离执行证据进展，不更改生产S5或晋级，不作有效经济失败归因。',
        'verification': '75原测试+3手算测试；4事件核验；22同代码停牌股日+1旧代码身份补证；151有限净值/3853成交独立对账。',
        'rollback_plan': '停用新增隔离入口；baseline保存账本/版本/文档，不回滚历史事实与原始数据。',
        'landed_detail': '四信号仍data_insufficient；600898.SH摘牌后估值证据缺失，仍持124000股；五阶段协议已记录，尚未进入模型搜索或未来验证。',
        'landed_at': '2026-09-23', 'landed_artifact': 'reports/fuxi_execution_followup_20260923.md',
        'code_changed': True, 'jq_validation_required': True, 'cross_verified': True,
    }
    if not seal_only:
        with FuxiWriterLock(run_id='followup_register_' + datetime.now().strftime('%H%M%S_%f')):
            registry.NOW = datetime.now().astimezone().isoformat()
            registry.PROPOSALS = {'execution_followup': [proposal]}
            registry.register('execution_followup')
            append_preserving(ROOT / '.workbuddy/memory/2026-09-23.md', note)
            append_preserving(ROOT / '.workbuddy/memory/_fuxi_landing_details.md',
                              note.replace('## M20260923.6', '## 附录 V — 2026-09-23 第六批'))
            subprocess.run([sys.executable, '-X', 'utf8', 'scripts/version_bump.py', '--bump'], cwd=ROOT, check=True)
            subprocess.run([sys.executable, '-X', 'utf8', 'scripts/version_bump.py', '--check'], cwd=ROOT, check=True)
    else:
        assert 'M20260923.6' in (ROOT / '.workbuddy/memory/2026-09-23.md').read_text(encoding='utf-8')
    version = read(ROOT / 'data/version_registry.json')['versions'][-1]['version']
    allowed = {'data/version_registry.json', 'data/stage1_optimization_proposals.json'}
    for name, expected in protocol['protected'].items():
        if name not in allowed:
            assert digest(ROOT / name) == expected, name
    old_proposals = read(OUT / 'baseline/data/stage1_optimization_proposals.json')
    current = read(ROOT / 'data/stage1_optimization_proposals.json')
    assert current[:len(old_proposals)] == old_proposals and len(current) == len(old_proposals) + 1
    for name in ['.workbuddy/memory/2026-09-23.md', '.workbuddy/memory/_fuxi_landing_details.md']:
        old, new = (OUT / 'baseline' / name).read_bytes(), (ROOT / name).read_bytes()
        assert new.startswith(old)
        if b'\r\n' in old:
            assert new.replace(b'\r\n', b'').count(b'\n') == old.replace(b'\r\n', b'').count(b'\n')
        else:
            assert b'\r\n' not in new
    for path in OUT.rglob('manifest.json'):
        assert read(path)['status'] != 'running'
    names = ['scripts/fuxi_execution_followup_20260923.py', 'scripts/fuxi_reference_cash_20260923.py',
             'scripts/fuxi_bse_state_probe_20260923.py', 'scripts/register_execution_followup_20260923.py',
             'research/factor_alchemy/tests/test_reference_cash.py', 'reports/fuxi_execution_followup_20260923.md',
             'research/fuxi_five_stage_acceptance_20260923.md', 'scripts/version_bump.py', 'CHANGELOG.md',
             '.workbuddy/memory/2026-09-23.md', '.workbuddy/memory/_fuxi_landing_details.md',
             'data/version_registry.json', 'data/stage1_optimization_proposals.json']
    for name in names:
        target = OUT / 'release' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    write(OUT / 'release_audit.json', {'version': version, 'stage1_complete': False,
          'five_stages_complete': False, 'formal_eligibility': False,
          'tests': {'existing': 75, 'new_hand_calculated': 3}, 'pre_event_comparisons': comparisons,
          'independent_reference': reference, 'protected_data_and_production_unchanged': True,
          'evidence_sha256': {p.relative_to(OUT).as_posix(): digest(p) for p in OUT.rglob('*') if p.is_file()}})
    print(json.dumps({'version': version, 'sealed': True, 'stage1_complete': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
