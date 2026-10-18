"""Existing JQData credential read-only probe; never strategy submission or renewal."""
import contextlib
from datetime import datetime,timezone
import io
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fuxi_v1.runtime import controlled_writer
from fuxi_v1.datasets import file_hash
OUT=ROOT/'reports/fuxi_v1_all_a_20261003'


def main():
    result={'observed_at':datetime.now(timezone.utc).isoformat(),'runner_sha256':file_hash(Path(__file__)),
        'scope':'existing JQData data-service credential and read-only permission; not JoinQuant strategy account or parity',
        'jqdata_authenticated':False,'actual_strategy_platform_observed':False,'orders_sent':0,'renewal_attempted':False}
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        try:
            import jqdatasdk as jq
            # Installed SDK tests the presence of enable_auth_prompt before each
            # key, so bundling options silently ignores timeout/retry settings.
            jq.set_params(request_timeout=15)
            jq.set_params(request_attempt_count=1)
            jq.set_params(enable_auth_prompt=False)
            from credentials import get_jqdata_auth
            get_jqdata_auth()
            result['jqdata_authenticated']=bool(jq.is_auth())
            if result['jqdata_authenticated']:
                result['query_count']={k:v for k,v in jq.get_query_count().items() if k in ('total','spare') and type(v) is int}
                info=jq.get_account_info()
                result['permission_dates']={}
                if isinstance(info,dict):
                    for key,value in info.items():
                        if any(x in key.lower() for x in ('expire','start','end','valid')):
                            dates=re.findall(r'\d{4}-\d{2}-\d{2}',str(value))
                            if dates:result['permission_dates'][key]=dates
                result['state']='authenticated_read_only_metadata_observed'
            else:result['state']='authentication_not_observed'
        except Exception as exc:
            message=str(exc).lower()
            result['state']='permission_expired' if any(x in message for x in ('过期','到期','expired','expire')) else 'credential_or_permission_rejected' if any(x in message for x in ('密码','账户','账号','权限','permission','password')) else 'transport_or_sdk_failure'
            result['exception_type']=type(exc).__name__
            result['dates_in_sanitized_category']=re.findall(r'\d{4}-\d{2}-\d{2}',message)
        if not result['jqdata_authenticated']:
            try:
                from jqdatasdk.client import JQDataClient
                from credentials import _load_config
                from urllib.parse import quote
                config=_load_config()
                http=JQDataClient(username=config.get('joinquant','username'),password=config.get('joinquant','password'))
                response=http.request_http({'method':'get_current_token','mob':http.username,'pwd':quote(http.password)})
                token=response.text.strip()
                result['http_fallback_status']=response.status_code
                if re.fullmatch(r'[0-9a-fA-F]{16,128}',token):
                    result['http_credential_authenticated']=True
                    response=http.request_http({'method':'get_query_count','token':token})
                    content=response.text.lower()
                    result['http_quota_response_observed']=response.status_code==200 and not content.startswith('error')
                    result['http_service_state']='quota_observed' if result['http_quota_response_observed'] else 'permission_expired' if any(x in content for x in ('过期','到期','expire')) else 'permission_or_region_rejected'
                else:
                    result['http_credential_authenticated']=False
                    result['http_service_state']='permission_expired' if any(x in token.lower() for x in ('过期','到期','expire')) else 'credential_or_region_rejected'
                    result['http_error_dates']=re.findall(r'\d{4}-\d{2}-\d{2}',token)
            except Exception as exc:
                result['http_service_state']='transport_or_sdk_failure';result['http_exception_type']=type(exc).__name__
    with (OUT/'JQDATA_ACCESS_PROBE.json').open('x',encoding='utf-8') as handle:json.dump(result,handle,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False),flush=True)


def direct_http():
    import requests
    from credentials import _load_config
    from urllib.parse import quote
    from jqdatasdk.client import AUTH_API_URL
    result={'observed_at':datetime.now(timezone.utc).isoformat(),'runner_sha256':file_hash(Path(__file__)),
        'parent_probe_sha256':file_hash(OUT/'JQDATA_ACCESS_PROBE.json'),'endpoint':AUTH_API_URL,
        'transport':'HTTPS session trust_env=False; no global proxy changes',
        'scope':'read-only JQData service; not strategy platform or order account','orders_sent':0,'renewal_attempted':False}
    config=_load_config();session=requests.Session();session.trust_env=False
    def call(body):
        with session.post(AUTH_API_URL,data=json.dumps(body),timeout=(10,15),stream=True) as response:
            raw=response.raw.read(1000001,decode_content=True)
            if len(raw)>1000000:raise ValueError('response size')
            return response.status_code,raw.decode('utf-8')
    try:
        status,token=call({'method':'get_current_token','mob':config.get('joinquant','username'),
            'pwd':quote(config.get('joinquant','password'))})
        token=token.strip();result['auth_http_status']=status
        result['http_credential_authenticated']=status==200 and bool(re.fullmatch(r'[0-9a-fA-F]{16,128}',token))
        if result['http_credential_authenticated']:
            status,content=call({'method':'get_query_count','token':token})
            result['quota_http_status']=status
            try:quota=json.loads(content)
            except (ValueError,TypeError):quota=None
            result['quota_projection']=quota if type(quota) is int and quota>=0 else {
                k:v for k,v in quota.items() if k in ('total','spare') and type(v) is int and v>=0} if isinstance(quota,dict) else None
            result['http_quota_response_observed']=status==200 and result['quota_projection'] is not None and result['quota_projection']!={}
        else:content=token
        result['http_service_state']='quota_observed' if result.get('http_quota_response_observed') else 'permission_expired' if any(x in content.lower() for x in ('过期','到期','expire')) else 'region_rejected' if any(x in content.lower() for x in ('region','中国大陆','地区')) else 'credential_or_permission_rejected'
        result['permission_dates']=re.findall(r'\d{4}-\d{2}-\d{2}',content) if not result.get('http_quota_response_observed') else []
    except Exception as exc:
        result.update(http_service_state='transport_failure',exception_type=type(exc).__name__)
    finally:session.close()
    with (OUT/'JQDATA_ACCESS_PROBE_V2.json').open('x',encoding='utf-8') as handle:json.dump(result,handle,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    with controlled_writer(OUT):
        if len(sys.argv)>1 and sys.argv[1]=='direct-http':direct_http()
        else:main()
