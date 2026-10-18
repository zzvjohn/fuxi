"""Apply the evidenced full-halt cash-reference correction before a new freeze."""
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/fuxi_v1_announcements_20261003/before_halt_repair'
OUT.mkdir(parents=True,exist_ok=True)
for name in ('historical_adapter.py','__init__.py'):
    source=ROOT/'fuxi_v1'/name
    with (OUT/name).open('xb') as h:h.write(source.read_bytes())

p=ROOT/'fuxi_v1/historical_adapter.py';s=p.read_bytes().decode('utf-8')
assert '\r\n' not in s
s=s.replace('    drift_audit=[]','    drift_audit=[];halt_reference_audit=[]')
old="""            mark=raw['close'];mark_source='observed_close'
            if mark is None and full and not conflict and code in last_quotes:
                mark=last_quotes[code];mark_source='declared_frozen_last_close_on_vendor_halt'
                if adj is not None and prior is not None and adj==prior and term is None:
                    classification='unchanged_factor_on_recorded_full_halt'
"""
new="""            mark=raw['close'];mark_source='observed_close';halt_reference=None
            if mark is None and full and not conflict and code in last_quotes:
                mark=last_quotes[code];mark_source='declared_frozen_last_close_on_vendor_halt'
                if term is not None:
                    # A halted price is a declared valuation reference, never a raw
                    # observation. Cash paid during the halt must leave that price
                    # before cash/receivables enter NAV; otherwise income is doubled.
                    # Noncash/differentiated rights require their own reference model.
                    from decimal import Decimal
                    supported_cash=(term['status']=='cash_terms_supported' and
                        term.get('share_distribution')==0 and mark is not None and
                        finite(term.get('gross_cash_per_share')) and term['gross_cash_per_share']>=0)
                    adjusted=(Decimal(str(mark))-Decimal(str(term['gross_cash_per_share']))) if supported_cash else None
                    halt_reference={'session':day,'instrument':instrument,'previous_declared_mark':mark,
                        'gross_cash_per_share':term.get('gross_cash_per_share'),'action_evidence_id':term['evidence_id'],
                        'scope':'declared halted cash-ex valuation; not an observed quote or independently certified price'}
                    if adjusted is not None and adjusted>0:
                        mark=float(adjusted);last_quotes[code]=mark;last_prices[code]=mark
                        mark_source='declared_ex_cash_reference_on_vendor_halt'
                        halt_reference['status']='declared_cash_reference'
                    else:
                        mark=None;last_quotes[code]=None
                        mark_source='halt_action_reference_not_identifiable'
                        halt_reference['status']='data_insufficient'
                    halt_reference['reference_mark']=mark;halt_reference_audit.append(halt_reference)
                if adj is not None and prior is not None and adj==prior and term is None:
                    classification='unchanged_factor_on_recorded_full_halt'
"""
assert s.count(old)==1;s=s.replace(old,new)
anchor="            for f in fields:values[f].append(raw[f])"
assert s.count(anchor)==1
s=s.replace(anchor,"            if halt_reference is not None:\n                bars[-1]['halt_reference']=halt_reference\n                if mark is not None:\n                    bars[-1]['sizing_reference_close']=mark\n                    bars[-1]['sizing_reference_evidence']=digest(halt_reference)\n            "+anchor.strip())
s=s.replace("    pd.DataFrame(drift_audit).to_csv(out/'adjustment_drift.csv',index=False,lineterminator='\\n')", "    pd.DataFrame(drift_audit).to_csv(out/'adjustment_drift.csv',index=False,lineterminator='\\n')\n    pd.DataFrame(halt_reference_audit).to_csv(out/'halt_reference_adjustments.csv',index=False,lineterminator='\\n')")
s=s.replace("'action_terms':dict(Counter(t['status'] for t in terms))", "'halt_reference_adjustments':dict(Counter(r['status'] for r in halt_reference_audit)),\n           'action_terms':dict(Counter(t['status'] for t in terms))")
compile(s,str(p),'exec');p.write_bytes(s.encode('utf-8'))
p=ROOT/'fuxi_v1/__init__.py';p.write_bytes(p.read_bytes().replace(b'1.0.0-rc8',b'1.0.0-rc9'))
for oldname,newname in [('fuxi_notice_tests_20261003.py','fuxi_notice_tests_v2_20261003.py'),
                        ('fuxi_notice_ten_rounds_20261003.py','fuxi_notice_ten_rounds_v2_20261003.py')]:
    old=(ROOT/'scripts'/oldname).read_bytes().decode('utf-8')
    if 'ten_rounds' in newname:
        old=old.replace("20261003/ten_rounds'","20261003/ten_rounds_v2'")
        old=old.replace("OUT.parent/'TEST_RESULT.json'","OUT.parent/'tests_v2/TEST_RESULT.json'")
    else:
        old=old.replace("announcements_20261003'","announcements_20261003/tests_v2'")
        old=old.replace("    target=OUT/'TEST_RESULT.json'", "    OUT.mkdir(parents=True,exist_ok=True)\n    target=OUT/'TEST_RESULT.json'")
    with (ROOT/'scripts'/newname).open('xb') as h:h.write(old.encode('utf-8'))
print('rc9 halt-reference correction applied; old snapshots and rc8 test evidence preserved')
