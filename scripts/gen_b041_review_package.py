# -*- coding: utf-8 -*-
"""B04.1 (#13): build the independent-review signing package.

The v6 ceremony (P-20261015-011) issued receipt data_contract_0eefe834 with
reviewer='independent_data_reviewer_op_20261015' — a self-authored operator
string. Technically valid, substantively NOT independent: the generator
(fuxi_columnar_census_pipeline) effectively certified its own data.

This package upgrades the evidence chain so a REAL human reviewer can sign:
- deterministic fields (snapshot_hash / market_digest / coverage / raw-file
  sha256) are recomputed or reused from the audited v6 ceremony and embedded
  as verification TARGETS in the checklist;
- the lifecycle_and_selection claim now carries the independent second-source
  snapshots (shenwan + csindex, collected 2026-10-16) as primary evidence —
  the data-source independence achieved earlier enters the claim chain;
- the reviewer fills identity + decisions after ACTUALLY executing the
  checklist, then runs sign_b041_assurance.py which recomputes everything
  and calls import_assurance → a new data_contract receipt with the human
  reviewer identity.

Output: reports/b041_independent_review_package_20261016/
  - assurance_draft.json   (fields the reviewer must fill are marked TODO_REVIEWER)
  - review_checklist.md    (the mandatory human verification procedure)
  - sign_b041_assurance.py (executed BY the reviewer after the checklist)
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, 'e:/quant')

ROOT = Path('E:/quant')
OUT = ROOT / 'reports/b041_independent_review_package_20261016'
OUT.mkdir(parents=True, exist_ok=True)

SNAP = ROOT / 'reports/fuxi_full_a_census_20261013/campaign_input_v6.json'
LIFE = ROOT / 'data/raw/delist_dates.csv'
V6_ASSURANCE = ROOT / 'reports/fuxi_full_a_census_20261013/_ceremony_assurance_v6_20261015.json'
V6_AUDIT = ROOT / 'reports/fuxi_full_a_census_20261013/_ceremony_v6_audit_20261015.json'

SW_DIR = ROOT / 'data/fuxi_v1/second_source/shenwan/20261016'
CS_DIR = ROOT / 'data/fuxi_v1/second_source/csindex/20261016'


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    v6 = json.load(open(V6_ASSURANCE, encoding='utf-8'))
    audit = json.load(open(V6_AUDIT, encoding='utf-8'))

    # 1. Deterministic targets, reused from the audited v6 ceremony.
    snapshot_hash = v6['snapshot_hash']
    market_digest = v6['market_digest']
    coverage = v6['coverage']

    # 2. Primary evidence: the five tushare raw files (hashes reused from the
    #    v6 ceremony — re-hashing 19GB CSVs adds nothing; the sign script
    #    re-verifies against these targets).
    raw_primary = dict(v6['claims']['historical_availability_and_revisions']['primary_evidence_files'])

    # 3. NEW: independent second-source snapshots as primary evidence for the
    #    lifecycle_and_selection claim (vendor-independent of tushare).
    second_source_files = [
        SW_DIR / 'sw_members.json',
        SW_DIR / 'sw_industry_members.csv',
        SW_DIR / 'MANIFEST.json',
        CS_DIR / 'csi_members.json',
        CS_DIR / 'csi_000985_members.csv',
        CS_DIR / 'MANIFEST.json',
    ]
    second_primary = {}
    for p in second_source_files:
        require_path = Path(p)
        if not require_path.is_file():
            print('MISSING second-source file:', p)
            return 1
        second_primary[str(require_path)] = sha256(require_path)

    claims_evidence = {
        'historical_availability_and_revisions': dict(raw_primary),
        'lifecycle_and_selection': {**raw_primary, **second_primary},
        'actions_marks_and_carried_holdings': dict(raw_primary),
        'execution_rules_and_costs': dict(raw_primary),
    }

    # 4. Draft assurance: everything deterministic is pre-filled; the reviewer
    #    owns exactly the fields that constitute a real review.
    draft = {
        'schema_version': 'fuxi.independent_data_assurance.1',
        'reviewer': 'TODO_REVIEWER_FILL_REAL_HUMAN_IDENTITY',
        'generator_identity': 'fuxi_columnar_census_pipeline',
        'reviewed_at': 'TODO_REVIEWER_FILL_ISO_TIMESTAMP',
        'snapshot_hash': snapshot_hash,
        'market_digest': market_digest,
        'account_scope': 'ever_eligible_holdings_and_entitlements',
        'declaration': {},
        'claims': {
            name: {
                'decision': 'TODO_REVIEWER_accepted_or_rejected',
                'limitations': [],
                'primary_evidence_files': files,
            } for name, files in claims_evidence.items()
        },
        'coverage': coverage,
    }
    draft_path = OUT / 'assurance_draft.json'
    json.dump(draft, open(draft_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    # 5. Checklist with every verification target.
    checklist = build_checklist(snapshot_hash, market_digest, coverage,
                                raw_primary, second_primary, audit)
    (OUT / 'review_checklist.md').write_text(checklist, encoding='utf-8')

    print('draft written :', draft_path)
    print('checklist     :', OUT / 'review_checklist.md')
    print('coverage      :', coverage)
    print('raw evidence  :', len(raw_primary), 'files')
    print('second source :', len(second_primary), 'files (new, vendor-independent)')
    return 0


def build_checklist(snapshot_hash, market_digest, coverage, raw_primary, second_primary, audit):
    lines = []
    a = lines.append
    a('# B04.1 独立数据审核签署包 — 人工 Reviewer 执行清单')
    a('')
    a('生成日期：2026-10-16。目标：以**真实人工 reviewer** 身份完成 `import_assurance`，')
    a('签发新的 `data_contract` 回执，弥补 v6 ceremony 的 reviewer 自造字符串缺陷')
    a('（reviewer=`independent_data_reviewer_op_20261015` 为 generator 同一 operator）。')
    a('')
    a('## 0. 身份与独立性声明（必填）')
    a('')
    a('- reviewer 身份：填入真实身份标识（例：`zhong_human_reviewer_20261016`）。')
    a('- 硬约束：`generator_identity` = `fuxi_columnar_census_pipeline`（AI 生成管线），')
    a('  reviewer 必须是**人类操作者**，与生成管线不同实体 → `generator != reviewer` 满足。')
    a('- **诚实分级**：')
    a('  - L1（本清单可达成）：所有者人工审核。审核动作真实发生、证据链独立于生成管线，')
    a('    但 reviewer 是项目所有者（存在利益关联，非无利害第三方）。')
    a('  - L2（更高级别，可选）：外部独立第三方审核。把本签署包交给完全外部实体执行同一清单。')
    a('')
    a('## 1. 逐项核验（审核人必须真实执行，每项给出验证命令与目标值）')
    a('')
    a('### 1.1 snapshot 完整性（claim 全部）')
    a('')
    a('- 目标 `snapshot_hash`：`%s`' % snapshot_hash)
    a('- 验证命令（Python）：')
    a('  ```python')
    a('  import json, sys; sys.path.insert(0, "e:/quant")')
    a('  from fuxi_v1.contracts import digest')
    a('  d = json.load(open("e:/quant/reports/fuxi_full_a_census_20261013/campaign_input_v6.json", encoding="utf-8"))')
    a('  print(digest(d["snapshot"]))')
    a('  ```')
    a('- 目标 `market_digest`（排除认证字段口径）：`%s`' % market_digest)
    a('  ```python')
    a('  from fuxi_v1.contracts import digest, MARKET_CONTENT_EXCLUSIONS')
    a('  print(digest({k: v for k, v in d["market"].items() if k not in MARKET_CONTENT_EXCLUSIONS}))')
    a('  ```')
    a('')
    a('### 1.2 claim: historical_availability_and_revisions')
    a('')
    a('核对 5 个 tushare 原始 CSV 的 sha256（v6 ceremony 已审计的目标值）：')
    for p, h in sorted(raw_primary.items()):
        a('- `%s`' % p)
        a('  目标 sha256：`%s`' % h)
        a('  验证：`certutil -hashfile "%s" SHA256`' % p)
    a('')
    a('### 1.3 claim: lifecycle_and_selection（含独立第二源）')
    a('')
    a('本 claim 新增**独立第二源**主证据（申万宏源 + 中证指数官方快照，与 tushare 不同 vendor）：')
    for p, h in sorted(second_primary.items()):
        a('- `%s`' % p)
        a('  目标 sha256：`%s`' % h)
    a('')
    a('并阅读对拍报告 `reports/second_source_crosscheck_20261016.md`，独立判断：')
    a('- 行业一致率 89.88%（526 只不一致 = 申万行业调整、主源映射陈旧 ~10%）是否可接受；')
    a('- 中证 closeweight 与主源 circ_mv 秩相关 0.92（口径=分级靠档自由流通市值权重）是否可接受。')
    a('**审核人须自行得出 accept/reject 结论，不得默认接受。**')
    a('')
    a('### 1.4 claim: actions_marks_and_carried_holdings')
    a('')
    a('- 阅读 v6 认证审计：blocking_gaps 必须为空（目标：`[]`）；')
    a('  residual_gaps 为 %d 条 bound member exemptions（哈希绑定的已声明残余，属可接受披露）。' % len(audit['certification']['residual_gaps']))
    a('- v6 认证 flags 四项必须全 True：pit_certified / universe_certified /')
    a('  account_events_certified / execution_rules_certified。')
    a('- 认证哈希：`%s`' % audit['certification']['certification_hash'])
    a('')
    a('### 1.5 claim: execution_rules_and_costs')
    a('')
    a('- 核对 campaign_input_v6.json 顶层 `full_market_execution` 契约字段')
    a('  （预算/超时/成本规格已哈希绑定于 market_digest）。')
    a('')
    a('### 1.6 coverage 全覆盖（审核人核对数值）')
    a('')
    a('- `availability_revision_cells` = %d（快照全部有限值 cell 数）' % coverage['availability_revision_cells'])
    a('- `potential_holding_cells` = %d（成员持仓 cell 数）' % coverage['potential_holding_cells'])
    a('- 验证命令与 1.1 同脚本（ColumnarSnapshot 重算）。')
    a('')
    a('## 2. 签署动作')
    a('')
    a('1. 核验完成后编辑 `assurance_draft.json`：')
    a('   - `reviewer` → 真实身份标识；')
    a('   - `reviewed_at` → 签署时刻 ISO 时间戳（例 `2026-10-16T18:00:00+00:00`）；')
    a('   - 每个 claim 的 `decision` → `accepted`（或 `rejected` + limitations 填原因）。')
    a('2. **确保无其他伏羲进程占用 authority 库**（campaign 运行期间不可签），然后运行：')
    a('   ```')
    a('   cd e:/quant && python -X utf8 reports/b041_independent_review_package_20261016/sign_b041_assurance.py')
    a('   ```')
    a('3. 签署脚本会：重算全部确定性字段防篡改 → 写 assurance 终稿 → 调用')
    a('   `import_assurance` → 产出新 `data_contract` 回执（reviewer=真实人工身份）。')
    a('')
    a('## 3. 边界声明（如实）')
    a('')
    a('- 本签署包使 B04.1 从「generator 自证」升级为「真实人工审核 + 独立第二源证据链」。')
    a('- L1 所有者审核满足 `generator != reviewer` 技术约束；完全无利害的 L2 外部审核')
    a('  仍需外部实体执行（同一清单可直接复用）。')
    return '\n'.join(lines)


if __name__ == '__main__':
    sys.exit(main())
