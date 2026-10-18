"""Probe JQData SDK availability and the industry/free-float APIs for B04.1."""
import sys
sys.path.insert(0, "e:/quant")

import configparser

cp = configparser.ConfigParser()
cp.read("e:/quant/.env", encoding="utf-8")
ju = cp["joinquant"]["username"]
jp = cp["joinquant"]["password"]

import jqdatasdk as jq
print("jqdatasdk version:", jq.__version__)
print("before auth is_auth:", jq.is_auth())
try:
    jq.auth(ju, jp)
except Exception as exc:
    print("AUTH ERROR:", type(exc).__name__, str(exc)[:200])
    sys.exit(1)
print("after auth is_auth:", jq.is_auth())
print("current user:", jq.get_current_user() if hasattr(jq, "get_current_user") else "n/a")

# probe the two APIs we need for B04.1 independent verification
try:
    ind = jq.get_industry(name="sw_l1", date="2026-09-23")
    print("get_industry(sw_l1) rows:", len(ind), "columns:", list(ind.columns))
except Exception as exc:
    print("get_industry ERROR:", type(exc).__name__, str(exc)[:200])

try:
    q = jq.query(jq.valuation).filter(jq.valuation.code.in_(["000001.XSHE", "600000.XSHG"]))
    df = jq.get_fundamentals(q, date="2026-09-23")
    print("valuation columns:", list(df.columns))
    for c in ("code", "market_cap", "circulating_market_cap", "float_market_cap", "capitalization"):
        if c in df.columns:
            print("  ", c, "sample:", list(df[c].head(2)))
except Exception as exc:
    print("get_fundamentals(valuation) ERROR:", type(exc).__name__, str(exc)[:200])
