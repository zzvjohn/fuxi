"""Preserve audit 2 and reconstruct its exact code solely when recorded hashes match."""
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fuxi_v1.contracts import require
from fuxi_v1.datasets import file_hash
from fuxi_v1.runtime import controlled_writer
OUT=ROOT/'reports/fuxi_v1_all_a_20261003'


def between(text,start,end,replacement=''):
    i=text.index(start);j=text.index(end,i)
    return text[:i]+replacement+text[j:]


def old_code(name,raw):
    text=raw.decode('utf-8')
    if name=='market_catalog.py':
        text=between(text,'def factor_corroboration(','def load_table(')
    elif name=='historical_adapter.py':
        text=text.replace("    if catalog_manifest is not None:\n        from .market_catalog import factor_corroboration\n        crosscheck,crosscheck_sources=factor_corroboration(root,catalog_manifest,codes,dates)\n",'')
        text=text.replace("        provenance['catalog_metadata_hash']=metadata['metadata_hash']\n        provenance['factor_crosscheck_sources']=crosscheck_sources\n","        provenance['catalog_metadata_hash']=metadata['metadata_hash']\n")
        text=text.replace('    if remediation is not None or crosscheck:\n','    if remediation is not None:\n')
        text=between(text,'    elif crosscheck:\n','    # Freeze only after')
    elif name=='panel_audit.py':
        text=between(text,'    from .market_catalog import factor_corroboration\n','    known=np.zeros(shape,bool)')
        text=text.replace('    last_quote_day=np.full(len(codes),None,dtype=object)\n','')
        text=between(text,"            elif not known[i,j] and masks['daily'][i,j] and crosscheck:\n",'        have=np.isfinite(close)&active[i]')
        text=text.replace('        last_quote_day[have]=day\n','')
        text=text.replace(',factor_constant_corroborated_vendor_proxy=corroborated','')
        text=between(text,"    pd.DataFrame(corroboration,columns=",'    panel_values=dict(numbers)')
        text=between(text,"        'factor_corroboration':", "        'independent_assurance':False,'historical_exposure'")
    elif name=='simulation.py':
        text=text.replace('from .contracts import digest, require, timestamp, finite','from .contracts import digest, require, timestamp')
        text=between(text,"                        elif not unlimited and (not finite(bar['up_limit'])",'                        elif bar["corporate_action_known"]')
        text=text.replace('"market_rule_or_quote_missing", "market_rule_or_quote_invalid", "unknown_event"','"market_rule_or_quote_missing", "unknown_event"')
    return text.encode('utf-8')


def old_test(raw):
    text=raw.decode('utf-8')
    text=text.replace("('none','future_name','halt_quote_conflict','factor_precision','factor_changed','cross_quote_conflict','declared_share_event')","('none','future_name','halt_quote_conflict')")
    text=between(text,"                    if issue.startswith('factor_')",'                    frame.to_parquet(path,index=False)')
    text=between(text,"                if issue.startswith('factor_')", "                manifest['manifest_hash']=digest(manifest)")
    text=between(text,"                dividends=[", "                metadata={'manifest_hash'", "                pd.DataFrame(columns=['ts_code','div_proc','ex_date','record_date','pay_date','stk_div','cash_div_tax','div_listdate','imp_ann_date']).to_csv(root/'dividend.csv',index=False)\n")
    text=text.replace("1 if issue in ('none','factor_precision') else 0","1 if issue=='none' else 0")
    text=text.replace("                self.assertEqual(result['factor_corroboration']['accepted_cells'],1 if issue=='factor_precision' else 0)\n",'')
    text=between(text,'    def test_invalid_observed_price_bounds_block_account(self):','    def test_missing_limits_never_imply_unbounded_prices(self):')
    return text.encode('utf-8')


def main():
    target=OUT/'audit_attempt2';require(not target.exists(),'audit 2 preservation already exists')
    tests=json.loads((OUT/'TEST_RESULT_FINAL.json').read_text(encoding='utf-8'))
    audit=json.loads((OUT/'DATA_AUDIT.json').read_text(encoding='utf-8'))
    staged={}
    for name,expected in tests['code_hashes'].items():
        raw=(ROOT/'fuxi_v1'/name).read_bytes()
        if file_hash(ROOT/'fuxi_v1'/name)!=expected:raw=old_code(name,raw)
        import hashlib
        require(hashlib.sha256(raw).hexdigest()==expected,'old code reconstruction mismatch: '+name)
        staged['code/'+name]=raw
    for name,expected in tests['test_hashes'].items():
        raw=(ROOT/'fuxi_v1/tests'/name).read_bytes()
        if file_hash(ROOT/'fuxi_v1/tests'/name)!=expected:raw=old_test(raw)
        require(hashlib.sha256(raw).hexdigest()==expected,'old test reconstruction mismatch: '+name)
        staged['tests/'+name]=raw
    require(all(tests['code_hashes'][n]==h for n,h in audit['code_hashes'].items()),'audit/test source mismatch')
    for name,expected in audit['artifact_hashes'].items():require(file_hash(OUT/name)==expected,'audit source asset changed: '+name)
    target.mkdir()
    for name,raw in staged.items():
        path=target/name;path.parent.mkdir(exist_ok=True);path.write_bytes(raw)
    shutil.copy2(OUT/'LOCAL_MASKS.npz',target/'LOCAL_MASKS.npz')
    for name in [*audit['artifact_hashes'],'DATA_AUDIT.json','TEST_RESULT_FINAL.json']:
        source=OUT/name
        require(source.resolve().parent==OUT.resolve(),'preservation scope')
        source.rename(target/name)
    body={'source_code':'reconstructed by exact inverse edits; every byte hash matched original 205-test receipt before any asset move',
        'code_hashes':tests['code_hashes'],'test_hashes':tests['test_hashes'],'audit_manifest_hash':audit['manifest_hash'],
        'artifact_hashes':{p.name:file_hash(p) for p in target.glob('*') if p.is_file()}}
    (target/'PRESERVATION.json').write_text(json.dumps(body,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'preserved_audit':str(target),'code_modules':len(tests['code_hashes']),'test_modules':len(tests['test_hashes'])}),flush=True)


if __name__=='__main__':
    with controlled_writer(OUT):main()
