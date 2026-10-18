# -*- coding: utf-8 -*-
"""Fix jq_fields and jq_window for the 8 Lasso factors to match their real implementations."""
import re
import os

FIXES = {
    'atr_14': {
        'jq_fields': "['close', 'high', 'low']",
        'jq_window': '30',
    },
    'dollar_vol_20d': {
        'jq_fields': "['close', 'volume']",
        'jq_window': '25',
    },
    'bp': {
        'jq_fields': "[]",
        'jq_window': '0',
    },
    'ep': {
        'jq_fields': "[]",
        'jq_window': '0',
    },
    'volume_stability': {
        'jq_fields': "['volume']",
        'jq_window': '25',
    },
    'range_consistency': {
        'jq_fields': "['high', 'low']",
        'jq_window': '25',
    },
    'high_low_range': {
        'jq_fields': "['close', 'high', 'low']",
        'jq_window': '25',
    },
    'ret_6m': {
        'jq_fields': "['close']",
        'jq_window': '135',
    },
}


def main():
    filepath = os.path.join(os.path.dirname(__file__), 'registry_generated.py')
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    for fname, fixes in FIXES.items():
        for key, new_val in fixes.items():
            # Match: 'key': <anything>, within the factor's dict
            pattern = (
                r"('" + re.escape(fname) + r"':\s*\{.*?)"
                r"'" + re.escape(key) + r"':\s*[^,\n]+"
            )
            replacement = r"\g<1>'" + key + "': " + new_val
            new_content, n = re.subn(pattern, replacement, content, count=1, flags=re.DOTALL)
            if n > 0:
                content = new_content
                print(f'  [OK] {fname}.{key} -> {new_val}')
            else:
                print(f'  [FAIL] Could not find {fname}.{key}')

    with open(filepath, 'w', encoding='utf-8', newline='\n') as f:
        f.write(content)

    # Verify
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from jq_generator.registry import JQ_FACTOR_REGISTRY
    for fname in FIXES:
        m = JQ_FACTOR_REGISTRY[fname]
        print(f'  {fname}: jq_window={m["jq_window"]}, jq_fields={m["jq_fields"]}, data={m["jq_data"]}')


if __name__ == '__main__':
    main()
