"""Register isolated execution engineering after explicit owner confirmation."""
import json
import csv
import hashlib
import shutil
import subprocess
import sys
from datetime import datetime
from collections import Counter
from pathlib import Path

import register_remediation_20260922 as registry

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research/factor_alchemy'))
from run_contract import FuxiWriterLock, _atomic_json

OUT = ROOT / 'reports/execution_events_20260923'


def main():
    policy = json.loads((ROOT / 'codex_automations/fuxi_runtime_policy.json').read_text(encoding='utf-8'))
    assert policy['single_writer_confirmed'] and not policy['autonomous_writes_enabled']
    market = json.loads((OUT / 'replay/market.json').read_text(encoding='utf-8'))
    tests = json.loads((OUT / 'replay/tests.json').read_text(encoding='utf-8'))
    assert len(market['rows']) == 4 and tests['successful'] and tests['tests_run'] == 75
    with (ROOT / 'data/raw/kuaidao_gaps/dividend.csv').open(encoding='utf-8-sig', newline='') as handle:
        periods = Counter(row['end_date'][4:] for row in csv.DictReader(handle))
    collector = ROOT / 'scripts/once/_collect_kuaidao_gaps.py'
    assert periods == {'1231': 77656}
    assert "end_date='%d1231' % y" in collector.read_text(encoding='utf-8')
    _atomic_json(OUT / 'coverage_diagnosis.json', {
        'existing_dividend_period_suffix_counts': dict(periods),
        'collector': str(collector.relative_to(ROOT)), 'collector_line': 269,
        'collector_sha256': hashlib.sha256(collector.read_bytes()).hexdigest(),
        'finding': 'annual-period-only cache excludes interim distributions; not a complete execution event source',
        'scope': 'explains the six observed cache gaps, not the full system factor drought'})
    proposal = {
        'proposal_id': 'P-20260923-005', 'title': '影子执行账本与六笔现金分红事件证据通道',
        'category': '执行与数据事件契约', 'domains': ['T2数据口径', 'T3模块/通道'],
        'risk_level': 'high', 'status': 'landed', 'proposal_change_kind': 'shadow',
        'trigger': '用户批准顺序整改，并确认 WorkBuddy 日/周任务已经停用。',
        'current_state': '生产 S5 日等权近似未接管；上批影子账本因六笔公司行动证据缺口停止。',
        'proposed_change': '登记已验证的固定股数/现金/应收款隔离执行模块；'
                           '受控补采六证券事件，只采纳实施记录，并校验公告/登记/派发日期和原始除息参考价。',
        'implementation_scope': 'portfolio_execution_ledger.py; fuxi_execution_lab_20260923.py; '
                                'fuxi_event_evidence_20260923.py; execution/cash event tests',
        'expected_impact': '区分缺事件与经济失败；不放宽 S5，不生成 JQ 资格，不自动启用定时写入。',
        'verification': '75 项测试通过；6/6 现金事件通过实施/日期/原始参考价核对；'
                        '四固定信号下一开盘账本复算、流水守恒与事件前区间对账。',
        'rollback_plan': '停用隔离实验入口即可，不改生产 S5。适配器旧版位于 '
                         'reports/execution_ledger_20260923/release；确认前策略位于本批 baseline。'
                         '回退不得重写用户确认、原始价格或历史实验事实。',
        'landed_detail': '影子研究基础设施已实现，不等于完整执行层通过验收。'
                         '只将已核验现金事件注入隔离账本，继续未知事件时 fail-closed。'
                         'cash_div 是供应商税后字段，不等于账户个税精确模拟；开盘代理非真实撮合。'
                         '用户确认单写入方，保持 autonomous_writes_enabled=false。',
        'landed_at': '2026-09-23', 'landed_artifact': 'reports/fuxi_execution_events_20260923.md',
        'code_changed': True, 'jq_validation_required': True, 'cross_verified': True,
    }
    registry.NOW = datetime.now().astimezone().isoformat()
    registry.PROPOSALS = {'execution_events': [proposal]}
    shutil.copy2(Path(__file__), OUT / 'registration_source.py')
    with FuxiWriterLock(run_id='execution_events_registration_20260923'):
        registry.register('execution_events')
        subprocess.run([sys.executable, '-B', '-X', 'utf8', 'scripts/version_bump.py', '--bump'],
                       cwd=ROOT, check=True)
        subprocess.run([sys.executable, '-B', '-X', 'utf8', 'scripts/version_bump.py', '--check'],
                       cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
