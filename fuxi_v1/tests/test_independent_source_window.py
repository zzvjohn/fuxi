"""Frozen quote windows must not be widened or confused with the event calendar."""
import json,zlib,unittest
from pathlib import Path
from fuxi_v1.tests.test_kernel import scratch_directory
from fuxi_v1.contracts import ContractError
from fuxi_v1.datasets import file_hash
from fuxi_v1.campaign import write_new
from fuxi_v1.action_evidence import validate_independent_wire,load_independent_events,independent_no_event


def response(kind,start,end,rows):
    columns=list(rows[0])
    fields=['0','ok',kind,'anonymous','1','1',json.dumps({'record':[[r[n] for n in columns] for r in rows]}),'sz.000026']
    if kind=='query_history_k_data_plus':
        fields+= [','.join(columns),start,end,'d','3'];method='96'
        body=zlib.compress('\x01'.join(fields).encode())
    else:
        fields+= [start,end,','.join(columns)];method='16'
        body='\x01'.join(fields).encode()
    return ('00.9.10\x01'+method+'\x01'+str(len(body)).zfill(10)).encode()+body+b'<![CDATA[]]>\n'


class IndependentWindowTests(unittest.TestCase):
    def raw(self,start='2024-06-24',end='2024-06-26'):
        rows=[{'date':'2024-06-24','code':'sz.000026','close':'10.0','preclose':'9.9','volume':'10000','tradestatus':'1'},
              {'date':'2024-06-25','code':'sz.000026','close':'10.2','preclose':'10.0','volume':'12000','tradestatus':'1'}]
        return {'api':'query_history_k_data_plus','code':'000026.SZ','rows':rows},response('query_history_k_data_plus',start,end,rows)

    def test_scoped_response_requires_exact_frozen_window(self):
        record,wire=self.raw()
        validate_independent_wire(record,wire,expected_window=['2024-06-24','2024-06-26'])
        with self.assertRaises(ContractError):validate_independent_wire(record,wire)
        with self.assertRaises(ContractError):validate_independent_wire(record,wire,expected_window=['2024-06-24','2024-06-25'])

    def test_legacy_window_remains_default_and_parsed_rows_must_match(self):
        record,wire=self.raw('2021-01-04','2026-09-24')
        validate_independent_wire(record,wire)
        record['rows'][0]['close']='11'
        with self.assertRaises(ContractError):validate_independent_wire(record,wire)

    def test_scoped_loader_keeps_full_event_calendar_and_real_quote_checks(self):
        with scratch_directory() as temp:
            root=Path(temp);raw,wire=self.raw()
            write_new(root/'wire_raw.response',wire);raw['response_sha256']=file_hash(root/'wire_raw.response')
            events=[{'code':'sz.000026','dividOperateDate':'2024-06-12','adjustFactor':'1.1'}]
            action={'api':'query_adjust_factor','code':'000026.SZ','rows':events}
            write_new(root/'wire_action.response',response('query_adjust_factor','2021-01-04','2026-09-24',events))
            action['response_sha256']=file_hash(root/'wire_action.response')
            write_new(root/'000026.SZ.raw.json',raw);write_new(root/'000026.SZ.adjustment.json',action)
            write_new(root/'AMENDMENT.json',{'codes':['000026.SZ'],'raw_windows':{'000026.SZ':['2024-06-24','2024-06-26']}})
            write_new(root/'COMPLETE.json',{'codes':1})
            quotes,calendar,sources=load_independent_events(root)
            self.assertEqual(calendar['000026.SZ'],{'2024-06-12'})
            before=quotes[('000026.SZ','2024-06-24')];after=quotes[('000026.SZ','2024-06-25')]
            self.assertTrue(independent_no_event(before,after,calendar['000026.SZ'],'2024-06-25',10.,10.,10.2))
            self.assertFalse(independent_no_event(before,after,{'2024-06-25'},'2024-06-25',10.,10.,10.2))
            self.assertFalse(independent_no_event(before,after,calendar['000026.SZ'],'2024-06-25',10.,10.,10.3))
            self.assertNotIn(('000026.SZ','2024-06-26'),quotes)
            with (root/'wire_raw.response').open('ab') as handle:handle.write(b'changed')
            with self.assertRaises(ContractError):load_independent_events(root)

    def test_invalid_scope_order_and_history_expansion_rejected(self):
        record,wire=self.raw()
        for window in [['2024-06-26','2024-06-24'],['2020-01-01','2026-09-24'],['2024-06-24','2027-01-01']]:
            with self.subTest(window=window),self.assertRaises(ContractError):
                validate_independent_wire(record,wire,expected_window=window)

    def test_successful_empty_event_result_differs_from_failed_or_missing_data(self):
        fields=['0','success','query_adjust_factor','anonymous','1','2000','',
                'sz.000026','2021-01-04','2026-09-24','code,dividOperateDate,adjustFactor']
        body='\x01'.join(fields).encode()
        wire=('00.9.10\x0116\x01'+str(len(body)).zfill(10)).encode()+body+b'<![CDATA[]]>\n'
        record={'api':'query_adjust_factor','code':'000026.SZ','rows':[]}
        validate_independent_wire(record,wire)
        failed=wire.replace(b'0\x01success',b'1\x01success',1)
        with self.assertRaises(ContractError):validate_independent_wire(record,failed)
        record['rows']=[{'code':'sz.000026','dividOperateDate':'2024-06-25','adjustFactor':'1.1'}]
        with self.assertRaises(ContractError):validate_independent_wire(record,wire)


if __name__=='__main__':unittest.main()
