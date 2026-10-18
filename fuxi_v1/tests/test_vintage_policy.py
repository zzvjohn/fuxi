"""Version choice, frozen identities and economic-date alignment at native consumers."""
import unittest
from pathlib import Path
from copy import deepcopy
from fuxi_v1.tests.test_kernel import scratch_directory
from fuxi_v1.system_fixture import synthetic_campaign
from fuxi_v1.campaign import write_new
from fuxi_v1.contracts import ContractError,load_json
from fuxi_v1.vintage_policy import freeze_latest_style_policy,verify_latest_style_policy
from fuxi_v1.remaining_study import freeze,proxy_styles
from fuxi_v1.tests.test_remaining_study import compact


class VintagePolicyTests(unittest.TestCase):
    def assets(self,root):
        payload=compact(synthetic_campaign(140,8));code=payload['snapshot']['instruments'][0]
        day=payload['snapshot']['sessions'][0].replace('-','')
        source={'captured_at':'2030-01-01T00:00:00+08:00','queries':[{'code':code[:6]+'.SH',
            'rows':[{'ts_code':code[:6]+'.SH','l1_code':'801010.SI','in_date':day,'out_date':None}]}]}
        market=root/'market.json';old=root/'old.json';new=root/'new.json'
        write_new(market,payload);write_new(new,source)
        older=deepcopy(source);older['captured_at']='2029-01-01T00:00:00+08:00';write_new(old,older)
        return payload,source,market,old,new

    def test_latest_capture_choice_is_order_independent(self):
        with scratch_directory() as temp:
            p,s,m,o,n=self.assets(Path(temp));cut='2031-01-01T00:00:00+08:00'
            a=freeze_latest_style_policy(m,[o,n],cut);b=freeze_latest_style_policy(m,[n,o],cut)
            self.assertEqual(a,b);self.assertEqual(a['industry_selected']['path'],str(n.resolve()))
            self.assertFalse(a['historical_PIT_certified']);verify_latest_style_policy(a,p,s)

    def test_missing_capture_cannot_use_file_mtime(self):
        with scratch_directory() as temp:
            p,s,m,o,n=self.assets(Path(temp));s.pop('captured_at');write_new(Path(temp)/'missing.json',s)
            with self.assertRaises(ContractError):freeze_latest_style_policy(m,[Path(temp)/'missing.json'],'2031-01-01T00:00:00+08:00')

    def test_future_capture_and_equal_time_conflicts_refused(self):
        with scratch_directory() as temp:
            p,s,m,o,n=self.assets(Path(temp))
            with self.assertRaises(ContractError):freeze_latest_style_policy(m,[n],'2029-01-01T00:00:00+08:00')
            changed=deepcopy(s);changed['queries'][0]['rows'][0]['l1_code']='other'
            other=Path(temp)/'conflict.json';write_new(other,changed)
            with self.assertRaises(ContractError):freeze_latest_style_policy(m,[n,other],'2031-01-01T00:00:00+08:00')

    def test_frozen_source_and_market_tampering_refused(self):
        with scratch_directory() as temp:
            p,s,m,o,n=self.assets(Path(temp));policy=freeze_latest_style_policy(m,[o,n],'2031-01-01T00:00:00+08:00')
            p2=deepcopy(p);p2['snapshot']['columns']['close'][0][0]+=1
            with self.assertRaises(ContractError):verify_latest_style_policy(policy,p2,s)
            o.write_bytes(o.read_bytes()+b' ')
            with self.assertRaises(ContractError):verify_latest_style_policy(policy,p,s)

    def test_latest_version_keeps_effective_dates_and_missing_cap(self):
        with scratch_directory() as temp:
            p,s,m,o,n=self.assets(Path(temp));policy=freeze_latest_style_policy(m,[n],'2031-01-01T00:00:00+08:00')
            old_panels,old_prov=proxy_styles(p,s);panels,prov=proxy_styles(p,s,data_policy=policy)
            self.assertEqual(panels,old_panels);self.assertGreater(prov['missing_industry_cells'],0)
            self.assertGreater(prov['missing_cap_cells'],0);self.assertFalse(prov['PIT_industry_certified'])
            self.assertFalse(prov['older_vintage_recovery_is_diagnostic_prerequisite'])
            self.assertTrue(all(v is None for x in panels['log_free_float_market_cap'].values() for v in x.values()))

    def test_native_freeze_selects_registered_latest_and_binds_protocol(self):
        with scratch_directory() as temp:
            p,s,m,o,n=self.assets(Path(temp));q=deepcopy(s);q['captured_at']='2001-01-01T00:00:00+08:00'
            n.write_text(__import__('json').dumps(q),'utf8');r=deepcopy(q);r['captured_at']='2000-01-01T00:00:00+08:00'
            o.write_text(__import__('json').dumps(r),'utf8');days=p['snapshot']['sessions']
            req={'research_start':days[85],'research_end':days[-1],'style_data_policy':'latest_available_snapshot',
                 'industry_candidates':[str(o),str(n)]}
            scope=freeze(m,Path(temp)/'run',req,industry=o)
            self.assertEqual(scope['inputs']['industry']['path'],str(n.resolve()))
            self.assertEqual(scope['style_data_policy']['industry_selected']['path'],str(n.resolve()))
            panels,prov=proxy_styles(p,q,data_policy=scope['style_data_policy'])
            self.assertTrue(panels);self.assertFalse(scope['formal_qualification_allowed'])

    def test_forged_mode_or_qualification_is_refused(self):
        with scratch_directory() as temp:
            p,s,m,o,n=self.assets(Path(temp));policy=freeze_latest_style_policy(m,[n],'2031-01-01T00:00:00+08:00')
            policy['formal_qualification_allowed']=True
            from fuxi_v1.contracts import digest
            policy['policy_hash']=digest({k:v for k,v in policy.items() if k!='policy_hash'})
            with self.assertRaises(ContractError):proxy_styles(p,s,data_policy=policy)
