# -*- coding: utf-8 -*-
"""打印 patch 锚点精确文本（repr）"""
raw = open(r"E:\quant\research\factor_alchemy\s5_joint_filter.py",
           "r", encoding="utf-8", newline="").read()

def show(label, needle, before=0, after=0):
    i = raw.find(needle)
    print(f"=== {label} | idx={i} ===")
    if i < 0:
        print("  NOT FOUND")
        return
    seg = raw[max(0, i - before): i + len(needle) + after]
    print(repr(seg))
    print()

# 锚点 2: _mark 行 + 后继注释
show("A_mark", "        _mark = _s5_holdings_mark()\r\n        # M6b:", after=60)

# 锚点 3: _switches 尾部
show("A_switch", '                     "holdings_mark": _mark}', before=60, after=40)

# 锚点 4: fixed_shares guard
show("A_guard", '        if _s5_ledger_mode() != "proxy":\r\n            return self._backtest_factor_fixed_shares(factor_df)')

# 锚点 5: price 裁剪点
show("A_price", '''        price = self._price_df[self._price_df["ts_code"].isin(self._universe)][
            [c for c in _pcols if c in self._price_df.columns]
        ].copy()'''.replace("\n", "\r\n"), after=80)

# 锚点 1: 函数区插入点（完整）
show("A_func", "_AXIS1_CALL_RE = re.compile(r'\\.\\s*\\w+\\s*\\([^()]*\\baxis\\s*=\\s*1\\b')",
     after=80)
