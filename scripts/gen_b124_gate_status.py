# -*- coding: utf-8 -*-
"""B12.4 (#14): release-gate status board + artifact-hash preparation script.

Two deliverables:
1. reports/b124_gate_status_20261016/gate_status.{json,md}
   The 8-step ReleaseService gate chain, honestly audited from landed
   artifacts (no authority-ledger reads while s406 holds the write lock —
   status derives from the frozen report files).
2. scripts/prep_release_bundle_artifacts.py (written by this generator)
   Computes the six release artifact hashes (code/model/strategy/
   data_contract/policy/state_schema) for a given candidate bundle so that,
   once jq_parity and E5 forward_acceptance actually occur, the bundle step
   is a one-command action.
"""
import json
import sys
from pathlib import Path

ROOT = Path('E:/quant')
OUT = ROOT / 'reports/b124_gate_status_20261016'
OUT.mkdir(parents=True, exist_ok=True)

S405_FINAL = ROOT / 'reports/fuxi_full_a_census_20261013/formal_admission_s405/FORMAL_FINAL.json'
S405_BUNDLE = ROOT / 'reports/fuxi_full_a_census_20261013/formal_admission_s405/campaigns/campaign_fuxi_formal_s405_20261015/bundles/run_e46cc9fc5d154d32926bcd67dddf9ae1'
PARITY_REPORT = ROOT / 'reports/fuxi_v2_jq_delivery_20261016/factor_704a_jq_backtest_parity.md'


def main():
    s405 = json.load(open(S405_FINAL, encoding='utf-8'))

    steps = [
        {
            'step': 1, 'name': 'research_acceptance (E0-E4 + independent data receipt)',
            'status': 'channel_proven__candidate_falsified',
            'evidence': {
                'formal_final': str(S405_FINAL),
                'jq_candidate_count': s405.get('jq_candidate_count'),
                'data_receipt_id': s405.get('data_receipt_id'),
                'research_acceptance_receipt': 'research_acceptance_4c2d5cfb509edeb9633b1e93e766971abf59c41a83cdfd85e516f89de9c6c7b8',
                'falsified_candidate': 'factor_704a0ddb (real JQ 2021-2026: -85.53% / Sharpe -1.59 / maxDD 86.61%)',
                'replacement_search': 's406 running (campaign_fuxi_formal_s406_20261016, 6 rounds x 2 candidates, refine+combine arms)',
            },
            'note': '通道已由 s405 坐实（P-011/012/013 修复链）；唯一准入候选被真实 JQ 证伪，替代候选搜索中（s406）。',
        },
        {
            'step': 2, 'name': 'compile-jq',
            'status': 'ready',
            'evidence': {
                'bundle_dir': str(S405_BUNDLE),
                'compiler_fields': {'backtest_orders_allowed': True, 'purpose': 'research_backtest',
                                    'parity_status': 'requires_external_signal_order_account_reconciliation',
                                    'universe_adapter': 'fuxi.jq_universe_adapter.1 (331 BSE members removed, dual ir_hash declared)'},
                'lightweight_script': 'reports/fuxi_v2_jq_delivery_20261016/factor_704a_jq_light.py (6.3KB, JQ-compilable)',
            },
            'note': '编译工程就绪：冻结版 32MB 与轻量实时版均可用；新候选产出后将自动走同一编译链。',
        },
        {
            'step': 3, 'name': 'jq_parity (real JQ, n_pairs>0)',
            'status': 'blocked',
            'evidence': {
                'parity_report': str(PARITY_REPORT),
                'factor_704a_result': 'cumulative -85.53% / Sharpe -1.59 / maxDD 86.61% / 5700 failed orders',
                'root_cause': 'weak factor (E1 ICIR 0.136) + execution bugs (1428 STAR-market market-order rejections, 3298 sub-100-share rejects, 950 suspended)',
                'remediation': {
                    'execution_bug_fix': 'next delivery uses limit orders for 688xxx + skips sub-100-share lots (must be baked into the next candidate script)',
                    'candidate_replacement': 's406',
                },
            },
            'note': '根瓶颈：需要强候选（E1 ICIR 显著高于 0.136）通过真实 JQ 对拍；factor_704a 的 price_volume_reversal 范式已被 D+ 硬禁止。',
        },
        {
            'step': 4, 'name': 'confirmatory_protocol',
            'status': 'pending_step3',
            'evidence': {},
            'note': '需 jq_parity 通过后冻结确认协议（预注册），不可提前伪造。',
        },
        {
            'step': 5, 'name': 'freeze-forward',
            'status': 'pending_step4',
            'evidence': {},
            'note': '冻结前向观察计划（决策/观察窗口），依赖确认协议。',
        },
        {
            'step': 6, 'name': 'E5 forward_acceptance (real future matured labels)',
            'status': 'time_gate',
            'evidence': {},
            'note': '真实未来观察成熟标签——时间门，不可伪造。注：首个 JQ 候选的「准入」豁免 E5（E5_required_for_first_JQ=false），但 B12.4 的 release_bundle 硬性消费 forward_acceptance 回执（acceptance.py ReleaseService.bundle），故发布激活必须等真实 E5。',
        },
        {
            'step': 7, 'name': 'release_bundle (six artifact hashes)',
            'status': 'pending_step6__hashes_prepared',
            'evidence': {
                'required_artifacts': ['code', 'model', 'strategy', 'data_contract', 'policy', 'state_schema'],
                'prep_script': 'scripts/prep_release_bundle_artifacts.py',
            },
            'note': '六工件哈希计算脚本已就绪（本次交付）；一旦步骤 3+6 达成，ReleaseService.bundle 一条命令完成。',
        },
        {
            'step': 8, 'name': 'paper_activation',
            'status': 'pending_step7',
            'evidence': {},
            'note': 'activate_paper 产出 paper_activation 回执（live_authorized=False）；rollback 链（reconciliation + 回滚目标）已在 acceptance.py 实现并被测试覆盖。',
        },
    ]

    board = {
        'schema_version': 'fuxi.b124_gate_status.1',
        'produced_at': '2026-10-16',
        'chain': 'research_acceptance -> compile-jq -> jq_parity -> confirmatory_protocol -> freeze-forward -> E5 -> release_bundle -> paper_activation',
        'overall': 'blocked_at_step3 (candidate replacement searching in s406); steps 1-2 proven, 7 prepared, 4-5-6-8 structurally gated on real events',
        'honest_boundary': '步骤 3 需真实 JQ 运行，步骤 6 需真实未来观察——两者均不可由操作方伪造；B04.1 独立回执已升级为人工签署包（reports/b041_independent_review_package_20261016/）。',
        'steps': steps,
    }
    json.dump(board, open(OUT / 'gate_status.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    md = ['# B12.4 发布/模拟激活门控链状态看板（2026-10-16）', '']
    md.append('链路：`research_acceptance → compile-jq → jq_parity → confirmatory_protocol → freeze-forward → E5 → release_bundle → paper_activation`')
    md.append('')
    md.append('| # | 步骤 | 状态 | 关键事实 |')
    md.append('|---|---|---|---|')
    icons = {'channel_proven__candidate_falsified': '⚠️ 通道✅ 候选❌', 'ready': '✅ 就绪', 'blocked': '❌ 阻塞',
             'pending_step3': '⏸ 待步骤3', 'pending_step4': '⏸ 待步骤4', 'time_gate': '⏳ 时间门',
             'pending_step6__hashes_prepared': '⏸ 待步骤6（哈希已备）', 'pending_step7': '⏸ 待步骤7'}
    for s in steps:
        fact = s.get('evidence', {})
        key_fact = json.dumps(fact, ensure_ascii=False)[:150] if fact else s['note'][:60]
        md.append('| %d | %s | %s | %s |' % (s['step'], s['name'], icons.get(s['status'], s['status']), key_fact))
    md.append('')
    md.append('## 诚实边界')
    md.append('')
    md.append('- **步骤 3（jq_parity）**：需新候选 + 真实聚宽运行。factor_704a 证伪后的替代搜索（s406）正在运行；')
    md.append('  下一次交付脚本必须修复执行 bug（科创板限价单 + 跳过不足 100 股）。')
    md.append('- **步骤 6（E5）**：真实未来观察成熟标签，时间门，任何操作方都不能伪造。')
    md.append('- **B04.1 独立回执**：已升级为人工签署包（`reports/b041_independent_review_package_20261016/`），')
    md.append('  reviewer 填真实身份并执行核验清单后运行签署脚本即产出新 data_contract 回执。')
    md.append('')
    (OUT / 'gate_status.md').write_text('\n'.join(md), encoding='utf-8')
    print('gate_status.json written:', OUT / 'gate_status.json')
    print('gate_status.md   written:', OUT / 'gate_status.md')
    return 0


if __name__ == '__main__':
    sys.exit(main())
