"""Common A-share identity, exchange quantity contracts and prospective exclusions."""
from decimal import Decimal, ROUND_FLOOR
import re
from .contracts import digest, require, timestamp
from .pit_universe import canonical_code

RULE_SOURCES = {
 'main_sh': 'https://www.sse.com.cn/lawandrules/sselawsrules2025/fund/trading/c/c_20260424_10817739.shtml',
 'sz': 'https://docs.static.szse.cn/www/lawrules/rule/trade/current/W020260424690713155663.pdf',
 'star': 'https://edu.sse.com.cn/tib/qa/c/4866268.shtml',
 'bj': 'https://www.bse.cn/jygl_list/200028217.html',
}

BSE_PILOT_OLD_CODES = frozenset({'833819.BJ','830799.BJ','831445.BJ','430489.BJ','839167.BJ','834682.BJ'})
BSE_CODE_SOURCES = {
    'pilot': 'https://www.bse.cn/important_news/200025603.html',
    'pilot_members': 'https://www.bse.cn/important_news/200025487.html',
    'bulk': 'https://www.bse.cn/important_news/200026735.html',
    'pairs': 'https://www.bse.cn/service/code_mapping.html',
}


def identity_aliases(mapping):
    """Stable research identities with dated venue symbols; no broker receipt claim."""
    aliases = {}
    for row in mapping:
        old, new = canonical_code(row['o_code']), canonical_code(row['n_code'])
        require(board(old) == board(new) == 'bj' and old != new, 'invalid BSE identity pair')
        require(old not in aliases and new not in {r['stable_code'] for r in aliases.values()}, 'duplicate BSE identity pair')
        pilot = old in BSE_PILOT_OLD_CODES
        aliases[old] = {'stable_code':new,'effective_session':'2025-05-06' if pilot else '2025-10-09',
            'mapping_evidence_id':digest(row),'source_urls':[BSE_CODE_SOURCES['pairs'],
                BSE_CODE_SOURCES['pilot' if pilot else 'bulk']],
            'scope':'vendor code relation and exchange code-change clock; broker ownership receipt not asserted'}
    require(not (set(aliases) & {r['stable_code'] for r in aliases.values()}), 'ambiguous BSE alias chain')
    return aliases


def venue_code_at(stable_code, session, aliases):
    stable_code = canonical_code(stable_code)
    matches = [(old,row) for old,row in aliases.items() if row['stable_code'] == stable_code]
    require(len(matches) <= 1, 'ambiguous venue identity')
    if not matches:return stable_code
    old,row = matches[0]
    return old if session < row['effective_session'] else stable_code


def board(code):
    code = canonical_code(code)
    for name, pattern in [('main_sh', r'60\d{4}\.SH'), ('main_sz', r'00\d{4}\.SZ'),
                          ('chinext', r'30\d{4}\.SZ'), ('star', r'688\d{3}\.SH'),
                          ('bj', r'(?:[48]\d{5}|92\d{4})\.BJ')]:
        if re.fullmatch(pattern, code): return name
    return None


def rule_for(code):
    kind = board(code)
    require(kind is not None, 'not a supported common A-share identity: ' + str(code))
    source = RULE_SOURCES[kind if kind in RULE_SOURCES else 'sz']
    return {'schema_version': 'fuxi.a_share_quantity_rule.1', 'board': kind,
        'min_buy': 200 if kind == 'star' else 100,
        'min_sell': 200 if kind == 'star' else 100,
        'buy_step': 1 if kind in ('star', 'bj') else 100,
        'sell_step': 1 if kind in ('star', 'bj') else 100,
        'max_limit_order': 100000 if kind == 'star' else 300000 if kind == 'chinext' else 1000000,
        'max_market_order': 50000 if kind == 'star' else 150000 if kind == 'chinext' else 1000000,
        'settlement_sessions': 1, 'tick': '0.01', 'source_url': source,
        'scope': 'quantity contract only; dynamic price limits, price collars, permissions and fees require separate dated evidence'}


def instrument_rule(execution, code):
    explicit = execution.get('instrument_rules')
    if explicit is None:
        # Preserve the existing mainboard replay without silently treating another board as mainboard.
        require(board(code) in ('main_sh', 'main_sz') or board(code) is None,
                'non-mainboard requires explicit instrument rules')
        return {'min_buy': execution['buy_lot'], 'min_sell': execution['sell_lot'],
                'buy_step': execution['buy_lot'], 'sell_step': execution['sell_lot'],
                'max_limit_order': 10**18, 'settlement_sessions': execution['settlement_sessions'], 'legacy': True}
    key = canonical_code(code)
    require(key in explicit and explicit[key] == rule_for(key), 'missing/conflicting instrument rule')
    return explicit[key]


def order_cap(rule, order_type):
    require(order_type in ('limit','market'),'unsupported order type')
    return rule['max_market_order' if order_type=='market' else 'max_limit_order']


def buy_quantity(quantity, rule, *, order_type='limit'):
    require(type(quantity) is int and quantity >= 0, 'invalid intended buy quantity')
    capped = min(quantity, order_cap(rule,order_type))
    result = capped // rule['buy_step'] * rule['buy_step']
    return result if result >= rule['min_buy'] else 0


def sell_quantity(quantity, held, settled, capacity, rule, *, order_type='limit'):
    require(all(type(x) is int and x >= 0 for x in (quantity, held, settled, capacity)), 'invalid sell capacity')
    maximum = min(quantity, held, settled, capacity, order_cap(rule,order_type))
    if maximum == held and held < rule['min_sell']: return held
    result = maximum // rule['sell_step'] * rule['sell_step']
    # A full residual may be sold in one order, including mainboard odd shares.
    if maximum == held: return held
    return result if result >= rule['min_sell'] else 0


def desired_shares(value, price, rule):
    quantity = int((Decimal(str(value))/Decimal(str(price))).to_integral_value(rounding=ROUND_FLOOR))
    quantity = max(0, quantity)//rule['buy_step']*rule['buy_step']
    return quantity if quantity >= rule['min_buy'] else 0


def exclude_new_entries(universe, exclusions, decision_time):
    """Exclusions are dated restrictions on new entries, never a liquidation or a past rewrite."""
    clock = timestamp(decision_time)
    excluded = set()
    for row in exclusions:
        require(row.get('reason') and row.get('evidence_ids') and row.get('review_after'), 'unreviewable exclusion')
        observed, effective = timestamp(row['observed_at']), timestamp(row['effective_at'])
        require(effective >= observed, 'exclusion predates evidence')
        require(timestamp(row['review_after'])>=effective,'exclusion review predates effect')
        if row.get('resolved_at'):
            resolved=timestamp(row['resolved_at'])
            require(resolved>=effective and row.get('resolution_evidence_ids'),'unreviewed re-entry')
            if clock>=resolved:continue
        if clock >= effective: excluded.add(canonical_code(row['instrument']))
    return [code for code in universe if canonical_code(code) not in excluded]


def exclusion_facts_at(exclusions, decision_time):
    clock=timestamp(decision_time);facts=[]
    for row in exclusions:
        if timestamp(row['observed_at'])>clock:continue
        fact={key:row[key] for key in ('instrument','reason','evidence_ids','observed_at','effective_at','review_after')}
        if row.get('resolved_at') and timestamp(row['resolved_at'])<=clock:
            fact.update(resolved_at=row['resolved_at'],resolution_evidence_ids=row['resolution_evidence_ids'])
        facts.append(fact)
    return sorted(facts,key=lambda r:(r['instrument'],r['observed_at']))


def delivery_verdict(data_complete, account_identifiable, data_assured, platform_observed, live_authorized):
    values = (data_complete, account_identifiable, data_assured, platform_observed, live_authorized)
    require(all(type(x) is bool for x in values), 'explicit delivery verdict required')
    return {'data_complete': data_complete, 'account_identifiable': account_identifiable,
            'investment_data_assured': data_assured, 'platform_observed': platform_observed,
            'research_delivery_ready': all(values[:3]), 'live_delivery_ready': all(values)}
