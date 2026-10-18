"""Review the draft probe before delivery; keep v1 unchanged."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=(ROOT/'scripts/fuxi_notice_make_jq_probe_20261004.py').read_bytes().decode('utf-8')
s=s.replace("DEST/'jq_readonly_coverage_probe.py'","DEST/'jq_readonly_coverage_probe_v2.py'")
s=s.replace("DEST/'PROBE_RECEIPT.json'","DEST/'PROBE_RECEIPT_v2.json'")
s=s.replace("    for r in master:",
    "    coverage_path=ROOT/'reports/fuxi_v1_all_a_20261003/LOCAL_COVERAGE.json'\n    ordinary_codes=set(load_json(coverage_path)['codes']);require(len(ordinary_codes)==5909,'ordinary A scope changed')\n    for r in master:\n        if r['ts_code'] not in ordinary_codes:continue")
s=s.replace("'master_sha256':file_hash(master_path),'mapping_catalog_sha256':file_hash(catalog),",
    "'master_sha256':file_hash(master_path),'ordinary_scope_sha256':file_hash(coverage_path),'mapping_catalog_sha256':file_hash(catalog),")
s=s.replace("source=\"# Read-only JQ strategy probe; user submits manually. No order APIs.\\nimport json\\nSPEC=json.loads(\"",
            "source=\"# Read-only JQ strategy probe; user submits manually. No order APIs.\\nfrom jqdata import *\\nimport json\\nSPEC=json.loads(\"")
s=s.replace("    listed = get_all_securities(types=['stock'], date=SPEC['asof'])\n    buckets = {}\n    for code in listed.index:",
    "    index_error = None\n    try:\n        listed = get_all_securities(types=['stock'], date=SPEC['asof'])\n        platform_codes = list(listed.index)\n    except Exception as exc:\n        index_error = str(exc)\n        platform_codes = []\n    buckets = {}\n    for code in platform_codes:")
s=s.replace("        rows.append(dict(target, platform_codes=hits, status=status))",
    "        if index_error is not None:\n            status = 'provider_index_query_failed'\n        rows.append(dict(target, platform_codes=hits, status=status))")
s=s.replace("'platform_stock_index_count':len(listed.index),'platform_suffix_counts':dict(Counter(str(c).split('.')[-1] for c in listed.index)),",
    "'platform_stock_index_count':len(platform_codes),'platform_index_error':index_error,'platform_suffix_counts':dict(Counter(str(c).split('.')[-1] for c in platform_codes)),")
s=s.replace("def fixture(index,quote_failure=False,day='2026-10-09'):","def fixture(index,quote_failure=False,day='2026-10-09',index_failure=False):")
s=s.replace("        exec(compile(source,'jq_readonly_probe.py','exec'),env)",
    "        def securities(**kwargs):\n            if index_failure:raise RuntimeError('fixture service unavailable')\n            return types.SimpleNamespace(index=index)\n        module=types.ModuleType('jqdata');module.get_all_securities=securities;module.get_price=price\n        from unittest.mock import patch\n        with patch.dict(sys.modules,{'jqdata':module}):exec(compile(source,'jq_readonly_probe.py','exec'),env)")
s=s.replace("    checks.append('pre-asof execution emits no future query')",
    "    checks.append('pre-asof execution emits no future query')\n    _,_,export,calls=fixture([],index_failure=True)\n    require(export[0]['mapping_counts']=={'provider_index_query_failed':len(targets)} and not calls,'global service failure disguised as stock exclusions')\n    checks.append('global provider failure retains all targets and zero price calls')")
s=s.replace("if __name__=='__main__':main()",
    "if __name__=='__main__':\n    from fuxi_v1.runtime import controlled_writer\n    with controlled_writer(DEST):main()")
assert 'from jqdata import *' in s and "'provider_index_query_failed'" in s
compile(s,'jq_probe_v2_generator.py','exec')
with (ROOT/'scripts/fuxi_notice_make_jq_probe_v2_20261004.py').open('xb') as h:h.write(s.encode('utf-8'))
print('v2 probe generator prepared; execute only after active campaign releases runtime lock')
