"""Bounded, resumable vendor reads. Credentials and vendor messages never enter logs."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from .contracts import digest, require


class VendorFailure(RuntimeError):
    def __init__(self, category, api, code=None):
        super().__init__(category + ':' + api)
        self.category, self.api, self.code = category, api, code


class TushareSource:
    def __init__(self, root, token, *, max_calls=30000, interval=1.3, transport=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.token, self.max_calls, self.interval = token, max_calls, interval
        self.calls, self.last, self.transport = 0, 0., transport
        self.session = None

    def _http(self, payload):
        import requests
        if self.session is None:self.session=requests.Session()
        with self.session.post('https://api.tushare.pro',json=payload,timeout=(10,30),stream=True) as response:
            response.raise_for_status()
            raw=response.raw.read(16000001,decode_content=True)
        require(len(raw) <= 16000000, 'response size limit')
        return raw

    def page(self, api, params, fields, *, limit=5000, offset=0):
        identity = {'api': api, 'params': params, 'fields': fields, 'limit': limit, 'offset': offset}
        target = self.root / (digest(identity) + '.json')
        if target.exists():
            saved = json.loads(target.read_text(encoding='utf-8'))
            require(saved['request'] == identity and saved['rows_hash'] == digest(saved['rows']), 'vendor cache mismatch')
            return saved
        require(self.calls < self.max_calls, 'vendor request budget exhausted')
        for attempt in range(3):
            require(self.calls < self.max_calls, 'vendor retry budget exhausted')
            time.sleep(max(0., self.interval - (time.monotonic() - self.last)))
            self.last = time.monotonic()
            self.calls += 1
            payload = {'api_name': api, 'token': self.token, 'params': dict(params, limit=limit, offset=offset), 'fields': fields}
            try:
                raw = (self.transport or self._http)(payload)
                response = json.loads(raw)
            except Exception as exc:
                category = 'transport_' + type(exc).__name__
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                    continue
                self._failure(identity, category)
                raise VendorFailure(category, api) from None
            if not isinstance(response,dict):
                self._failure(identity,'response_contract')
                raise VendorFailure('response_contract',api)
            if response.get('code') != 0:
                # Interpret the message in memory, persist only a finite category.
                msg = str(response.get('msg', ''))
                category = ('permission' if any(x in msg for x in ('权限', '积分', 'token', 'Token', '无效'))
                            else 'rate_limit' if any(x in msg for x in ('频率', '分钟', '每小时', '每天', '访问次数'))
                            else 'vendor_rejected')
                if category == 'rate_limit' and attempt < 2:
                    time.sleep(20)
                    continue
                self._failure(identity, category, response.get('code'))
                raise VendorFailure(category, api, response.get('code'))
            data = response.get('data') or {}
            if not isinstance(data,dict):
                self._failure(identity,'field_contract')
                raise VendorFailure('field_contract',api)
            names, items = data.get('fields'), data.get('items')
            if not (isinstance(names,list) and isinstance(items,list) and
                    set(fields.split(','))<=set(names) and len(set(names))==len(names) and
                    all(isinstance(x,list) and len(x)==len(names) for x in items)):
                self._failure(identity,'field_contract')
                raise VendorFailure('field_contract',api)
            rows = [dict(zip(names, x)) for x in items]
            saved = {'request': identity, 'retrieved_at': datetime.now(timezone.utc).isoformat(),
                     'rows': rows, 'rows_hash': digest(rows), 'response_sha256': hashlib.sha256(raw).hexdigest(),
                     'status': 'observed_vendor_response_not_independent_assurance'}
            with target.open('x', encoding='utf-8') as handle:
                json.dump(saved, handle, ensure_ascii=False)
            return saved

    def _failure(self, identity, category, code=None):
        failure = {'request': identity, 'category': category, 'vendor_code': code,
                   'at': datetime.now(timezone.utc).isoformat()}
        with (self.root / ('failure_' + str(time.time_ns()) + '.json')).open('x', encoding='utf-8') as handle:
            json.dump(failure, handle, ensure_ascii=False)

    def query(self, api, params, fields, *, page_size=5000, max_pages=100):
        rows, evidence, previous = [], [], None
        for page in range(max_pages):
            record = self.page(api, params, fields, limit=page_size, offset=page * page_size)
            fingerprint = record['rows_hash']
            require(not record['rows'] or fingerprint != previous, 'vendor ignored pagination')
            rows.extend(record['rows'])
            evidence.append({k: record[k] for k in ('request', 'retrieved_at', 'rows_hash', 'response_sha256')})
            if len(record['rows']) < page_size:
                return {'rows': rows, 'evidence': evidence, 'retrieved_now_not_historical_revision_archive': True}
            previous = fingerprint
        raise VendorFailure('pagination_budget_exhausted', api)
