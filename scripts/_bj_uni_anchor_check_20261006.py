# -*- coding: utf-8 -*-
"""锚点唯一性验证"""
raw = open(r"E:\quant\research\factor_alchemy\s5_joint_filter.py",
           "r", encoding="utf-8", newline="").read()

A1 = ("_AXIS1_CALL_RE = re.compile(r'\\.\\s*\\w+\\s*\\([^()]*\\baxis\\s*=\\s*1\\b')"
      "\r\n\r\n\r\ndef _s5_holdings_mark() -> str:")
print("A1 (crlf form) count:", raw.count(A1))

A2 = ('        price = self._price_df[self._price_df["ts_code"].isin(self._universe)]['
      "\r\n            [c for c in _pcols if c in self._price_df.columns]"
      "\r\n        ].copy()")
print("A2 count:", raw.count(A2))

print("--- lines containing 'price = self._price_df' ---")
for i, ln in enumerate(raw.split("\r\n"), 1):
    if "price = self._price_df" in ln:
        print(i, "|", ln.strip()[:110])

print("--- lines containing 'def _s5_holdings_mark' ---")
for i, ln in enumerate(raw.split("\r\n"), 1):
    if "def _s5_holdings_mark" in ln:
        print(i, "|", ln.strip()[:110])

print("--- lines containing 'isin(self._universe)' ---")
for i, ln in enumerate(raw.split("\r\n"), 1):
    if "isin(self._universe)" in ln:
        print(i, "|", ln.strip()[:110])
