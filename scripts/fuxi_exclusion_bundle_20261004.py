"""A prospective no-new-entry bundle from preserved source/account evidence."""
import json,sys
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fuxi_v1.contracts import load_json,require
from fuxi_v1.datasets import file_hash
from fuxi_v1.a_share import canonical_code
from fuxi_v1.runtime import controlled_writer
OUT=ROOT/'reports/fuxi_v1_concentration_20261004'
DELIVERY=ROOT/'deliverables/fuxi_concentration_20261004'


def main():
    outcome=load_json(OUT/'attempt1/OUTCOME.json');later=load_json(OUT/'fallback_diagnostic/OUTCOME.json')
    closure=load_json(OUT/'closure_diagnostic/OUTCOME.json')
    v2,v3=load_json(OUT/'FRACTIONAL_RULES_v2.json')['rules'],load_json(OUT/'FRACTIONAL_RULES_v3.json')['rules']
    require(v2==v3,'new source rules require source-corrected accounting before exclusions')
    now=datetime.now(timezone.utc);rows=load_json(ROOT/'reports/fuxi_v1_all_a_20261003/EXCLUSIONS.json')['exclusions']
    evidence={'account_outcome':file_hash(OUT/'attempt1/OUTCOME.json'),'diagnostic_outcome':file_hash(OUT/'fallback_diagnostic/OUTCOME.json'),
        'closure_outcome':file_hash(OUT/'closure_diagnostic/OUTCOME.json'),
        'source_revision':file_hash(OUT/'source_revision1/RESULT.json')}
    instruments=sorted(set(outcome['root_instruments'])|set(later['root_instruments'])|set(closure['root_instruments']))
    for code in instruments:
        gaps=[g for g in outcome['root_gaps']+later['root_gaps']+closure['root_gaps'] if g['instrument']==code]
        rows.append({'instrument':canonical_code(code),'reason':'historical_entitlement_or_action_evidence_incomplete',
            'observed_at':now.isoformat(),'effective_at':now.isoformat(),'review_after':(now+timedelta(days=7)).isoformat(),
            'evidence_ids':list(evidence.values()),'evidence_files':[str(OUT/'attempt1/OUTCOME.json'),str(OUT/'fallback_diagnostic/OUTCOME.json'),str(OUT/'closure_diagnostic/OUTCOME.json'),str(OUT/'source_revision1/RESULT.json')],
            'event_dates':sorted({g['session'] for g in gaps}),'event_reasons':sorted({g['reason'] for g in gaps}),
            'new_entries_only':True,'retain_existing_positions_and_entitlements':True,
            'reentry_condition':'archive source-supported event/coverage evidence or independently reviewed broker/settlement statement; reconcile existing shares/cash/NAV before clearing this row'})
    bundle={'schema_version':'fuxi.prospective_exclusions.1','exclusions':rows,'historical_accounts_not_rewritten':True,
        'historical_exclusion_would_be_posthoc':True,'new_blocked_instruments':len(instruments),
        'consumer':'decision_with_exclusions.py -> fuxi_v1.online.daily_decision(exclusions=...)',
        'legacy_S5_default_unchanged':True,'formal_JQ':0,'live':0,'evidence':evidence}
    with (DELIVERY/'EXCLUSIONS.json').open('x',encoding='utf-8') as f:json.dump(bundle,f,ensure_ascii=False,indent=2)
    print(json.dumps({'total_exclusions':len(rows),'new_exclusions':len(instruments),'history_preserved':True},ensure_ascii=False))


if __name__=='__main__':
    with controlled_writer(OUT):main()
