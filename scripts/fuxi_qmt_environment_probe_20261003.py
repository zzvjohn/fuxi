"""Broker-bundled Python ABI probe. No trader object, connect, subscribe or order call."""
import datetime
import hashlib
import json
import os
import platform
import sys


def main():
    result = {'schema_version': 'fuxi.qmt_environment_probe.1',
        'observed_at': datetime.datetime.utcnow().isoformat()+'Z',
        'python': platform.python_version(), 'executable': sys.executable,
        'scope': 'import and method inventory only; no account connection or credentials',
        'account_authenticated': False, 'platform_parity_observed': False,
        'order_calls': 0, 'live_orders': 0}
    try:
        from xtquant.xttrader import XtQuantTrader
        import xtquant.xttrader as module
        methods = ['query_account_infos','query_account_status','query_stock_asset',
            'query_stock_positions','query_stock_orders','query_stock_trades']
        result.update(sdk_import='successful',
            read_method_availability={name:hasattr(XtQuantTrader,name) for name in methods},
            sdk_path=module.__file__)
        with open(module.__file__,'rb') as handle:
            result['sdk_source_sha256'] = hashlib.sha256(handle.read()).hexdigest()
    except Exception as exc:
        result.update(sdk_import='unavailable', error_type=type(exc).__name__)
    with open(sys.argv[1],'x',encoding='utf-8') as handle:
        json.dump(result,handle,ensure_ascii=False,indent=2)


if __name__ == '__main__':main()
