"""Compare observed platform values with explicit identity, time and null semantics."""
from .contracts import require, finite, digest


def compare(expected, observed, *, tolerance=1e-8):
    require(observed.get('schema_version') == 'fuxi.jq_observed_diagnostic.1' and
            observed.get('observed_on_jq') is True and observed.get('raw_export_sha256'), 'real platform export identity required')
    require(expected.get('protocol_hash') == observed.get('protocol_hash'), 'platform protocol mismatch')
    required = ('signal_session','instrument')
    def index(rows):
        result = {}
        for row in rows:
            require(all(row.get(k) for k in required), 'parity key missing')
            key = tuple(row[k] for k in required)
            require(key not in result,'duplicate parity key')
            result[key] = row
        return result
    a,b = index(expected['rows']), index(observed['rows'])
    require(a and b, 'zero platform pairs')
    missing = sorted(set(a)^set(b))
    pairs, differences, nulls = 0, [], []
    for key in sorted(set(a)&set(b)):
        for field in ('factor_value','target_weight','decision_time','execution_session'):
            x,y = a[key].get(field), b[key].get(field)
            if x is None or y is None:
                nulls.append({'key':key,'field':field,'both_null':x is y})
                if x is not y:
                    differences.append({'key':key,'field':field,'reason':'missing_pattern_changed'})
            elif field in ('factor_value','target_weight'):
                require(finite(x) and finite(y),'nonfinite platform reading')
                pairs += 1
                if abs(x-y)>tolerance:
                    differences.append({'key':key,'field':field,'delta':y-x})
            elif x != y:
                differences.append({'key':key,'field':field,'reason':'time_mismatch'})
    require(pairs>0,'zero numeric platform pairs')
    return {'eval_status':'valid','diagnostic_match':not missing and not differences,
            'n_pairs':pairs,'missing_keys':missing,'differences':differences,'nulls':nulls,
            'observed_on_jq':True,'export_hash':digest(observed),'formal_parity_receipt_granted':False,
            'account_parity':'requires separate observed account/rights export','jq_candidate_allowed':False,'live_execution_allowed':False}
