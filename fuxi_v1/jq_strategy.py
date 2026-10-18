"""A single StrategyIR compiled into a JQ historical replay and parity export.

Diagnostic scripts log intent only. Backtest orders require an authority receipt,
and the generated script refuses simulation/live run types.
"""

import hashlib
from pathlib import Path
from .contracts import canonical, digest, require


RUNTIME = '''
DAY_EXPORT = {}

def planned_targets(signal_session, prior_weights=None):
    require(digest(FROZEN) == FROZEN_HASH, "frozen plan changed")
    rows = [r for r in FROZEN["ir"]["decisions"] if r["signal_session"] == signal_session]
    require(len(rows) == 1, "unregistered/duplicate signal session")
    return target_weights(rows[0]["scores"], FROZEN["ir"]["strategy"], prior_weights)

def _set_execution_cost(day):
    cost = FROZEN["execution"]
    rate = cost["sell_tax"]
    for effective, scheduled in sorted(cost.get("sell_tax_schedule", [])):
        if effective <= day:
            rate = scheduled
    set_order_cost(OrderCost(open_tax=0, close_tax=float(rate),
                            open_commission=float(cost["buy_commission"]), close_commission=float(cost["sell_commission"]),
                            min_commission=float(cost["minimum_commission"])), type="stock")

def initialize(context):
    require(context.run_params.type in ("simple_backtest", "full_backtest"),
            "this frozen historical plan only supports backtests")
    set_option("avoid_future_data", True)
    set_option("use_real_price", True)
    cost = FROZEN["execution"]
    _set_execution_cost(str(context.current_dt.date()))
    set_slippage(PriceRelatedSlippage(2*float(cost["slippage_bps"])/10000))
    run_daily(rebalance, time="open")
    log.info("Fuxi frozen IR; orders enabled=" + str(FROZEN["backtest_orders_allowed"]))

def rebalance(context):
    global DAY_EXPORT
    require(digest(FROZEN) == FROZEN_HASH, "frozen plan changed")
    day = str(context.current_dt.date())
    _set_execution_cost(day)
    DAY_EXPORT = {"key":day,"signal":None,"target_weights":None,"orders":[]}
    rows = [r for r in FROZEN["ir"]["decisions"] if r["execution_session"] == day]
    if not rows:
        return
    row = rows[0]
    require(row["signal_session"] < day, "future or same-day signal")
    inverse = {v:k for k,v in FROZEN["instrument_mapping"].items()}
    require(set(context.portfolio.positions) <= set(inverse), "unregistered position mapping")
    positions = {inverse[c]:p for c,p in context.portfolio.positions.items()}
    codes = sorted(set(row["scores"]) | set(positions))
    references, volumes = {}, {}
    for code in codes:
        frame = get_price(FROZEN["instrument_mapping"][code], start_date=row["signal_session"], end_date=row["signal_session"],
                          frequency="daily", fields=["close", "volume"], skip_paused=False, fq=None)
        require(frame is not None and len(frame) == 1, "sizing reference missing")
        references[code], volumes[code] = float(frame.iloc[0]["close"]), float(frame.iloc[0]["volume"])
    nav = float(context.portfolio.available_cash) + sum(float(p.total_amount)*references[c] for c,p in positions.items())
    require(nav > 0, "nonpositive sizing NAV")
    prior = {c: float(p.total_amount)*references[c]/nav for c,p in positions.items()}
    weights = planned_targets(row["signal_session"], prior)
    DAY_EXPORT.update(signal=row["scores"],target_weights=weights)
    if weights is None:
        return
    cost, intents = FROZEN["execution"], []
    for code in sorted(set(weights) | set(positions)):
        require(references[code] > 0, "invalid sizing reference")
        current = positions[code].total_amount if code in positions else 0
        rule=instrument_rule(cost,code) if cost.get('instrument_rules') is not None else None
        target = desired_shares(nav*weights.get(code,0),references[code],rule) if rule else int(nav*weights.get(code,0)/references[code]/cost["buy_lot"])*cost["buy_lot"]
        delta = target-current
        cap = int(volumes[code]*float(cost["participation"])/cost["buy_lot"])*cost["buy_lot"]
        delta = max(-cap, min(cap, delta))
        if rule:
            raw_cap=int(volumes[code]*float(cost['participation']))
            if delta>0:delta=buy_quantity(min(target-current,raw_cap),rule,order_type='market')
            elif delta<0:
                settled=int(positions[code].closeable_amount) if code in positions else 0
                delta=-board_sell_quantity(current-target,int(current),settled,raw_cap,rule,order_type='market')
        if delta:
            intents.append((code, delta))
    for code, delta in sorted(intents, key=lambda x:(x[1]>0,x[0])):
        log.info(json.dumps({"kind":"order_intent","session":day,"instrument":code,"shares":delta,
                             "ir_hash":FROZEN["ir"]["ir_hash"]}, sort_keys=True))
        if FROZEN["backtest_orders_allowed"]:
            receipt = order(FROZEN["instrument_mapping"][code], delta)
            log.info("order_receipt="+str(receipt))
            DAY_EXPORT["orders"].append({"instrument":code,"requested_shares":delta,
              "order_id":str(getattr(receipt,"order_id","unavailable")),
              "filled":getattr(receipt,"filled",None),"price":getattr(receipt,"price",None)})

def after_trading_end(context):
    inverse = {v:k for k,v in FROZEN["instrument_mapping"].items()}
    row = {**DAY_EXPORT,"key":str(context.current_dt.date()),"session":str(context.current_dt.date()), "cash":float(context.portfolio.available_cash),
           "nav":float(context.portfolio.total_value),
           "holdings":{inverse[c]:int(p.total_amount) for c,p in context.portfolio.positions.items()},
           "receivables":None,"receivable_evidence_status":"requires_independent_event_reconciliation",
           "ir_hash":FROZEN["ir"]["ir_hash"]}
    log.info("FUXI_ACCOUNT_EXPORT "+json.dumps(row, sort_keys=True))
'''


def _py27_compat(source):
    """Shim Python-3-only syntax/API for JoinQuant's Python 2.7 backtest engine.

    The frozen plan inlines fuxi_v1 module sources that are written in Python 3;
    JoinQuant's cloud backtest runs Python 2.7, so a handful of constructs would
    fail to parse (``def f(*, kw)`` keyword-only args, dict unpacking ``{**d}``,
    ``raise ... from``) or resolve (``collections.abc``, ``math.isfinite``,
    ``re.fullmatch``) and surface as JQ's ``编译失败: undefined``. We also
    normalise ``context.portfolio.positions`` dict keys from ``Security`` objects
    to code strings, because JQ keys positions by ``Security`` not ``str``.
    """
    source = source.replace(
        "from collections.abc import Mapping",
        "from collections import Mapping")
    source = source.replace(
        "and math.isfinite(value)",
        "and not (math.isinf(value) or math.isnan(value))")
    source = source.replace(
        'raise ContractError("invalid ISO timestamp") from exc',
        'raise ContractError("invalid ISO timestamp")')
    # keyword-only argument marker (PEP 3102) -> plain positional-with-default
    source = source.replace(", *, ", ", ")
    source = source.replace(
        "re.fullmatch(pattern, code)",
        're.match(pattern + r"\\Z", code)')
    # JQ keys positions by Security objects, not code strings.
    source = source.replace(
        'require(set(context.portfolio.positions) <= set(inverse), "unregistered position mapping")\n'
        '    positions = {inverse[c]:p for c,p in context.portfolio.positions.items()}',
        'require(set(str(c) for c in context.portfolio.positions) <= set(inverse), "unregistered position mapping")\n'
        '    positions = {inverse[str(c)]:p for c,p in context.portfolio.positions.items()}')
    # dict unpacking ``{**DAY_EXPORT, ...}`` -> copy + update (Python 2.7 safe)
    source = source.replace(
        'row = {**DAY_EXPORT,"key":str(context.current_dt.date()),"session":str(context.current_dt.date()), "cash":float(context.portfolio.available_cash),\n'
        '           "nav":float(context.portfolio.total_value),\n'
        '           "holdings":{inverse[c]:int(p.total_amount) for c,p in context.portfolio.positions.items()},\n'
        '           "receivables":None,"receivable_evidence_status":"requires_independent_event_reconciliation",\n'
        '           "ir_hash":FROZEN["ir"]["ir_hash"]}',
        'row = dict(DAY_EXPORT)\n'
        '    row.update({"key":str(context.current_dt.date()),"session":str(context.current_dt.date()), "cash":float(context.portfolio.available_cash),\n'
        '           "nav":float(context.portfolio.total_value),\n'
        '           "holdings":{inverse[str(c)]:int(p.total_amount) for c,p in context.portfolio.positions.items()},\n'
        '           "receivables":None,"receivable_evidence_status":"requires_independent_event_reconciliation",\n'
        '           "ir_hash":FROZEN["ir"]["ir_hash"]})')
    return source


def compile_jq_strategy(ir, market, store=None, acceptance_receipt=None, research_ir=None):
    require(ir["execution_hash"] == digest(market["execution"]), "JQ execution contract mismatch")
    require(ir["strategy"]["industry_cap"] == 1., "JQ industry constraint adapter not admitted")
    allowed = False
    mapping = {}
    for code in {s for row in ir["decisions"] for s in row["scores"]}:
        require(len(code.split(".")[0]) == 6 and code.split(".")[0].isdigit(),"invalid A-share instrument")
        if code.endswith((".XSHE",".XSHG")):
            mapping[code] = code
        elif code.endswith(".SZ"):
            mapping[code] = code[:-3]+".XSHE"
        elif code.endswith(".SH"):
            mapping[code] = code[:-3]+".XSHG"
        else:
            raise ValueError("unsupported JQ venue mapping")
    require(len(set(mapping.values())) == len(mapping),"ambiguous instrument aliases")
    bound_ir = research_ir if research_ir is not None else ir
    if acceptance_receipt is not None:
        require(store is not None, "authority store required")
        receipt = store.consume_receipt(acceptance_receipt, "research_acceptance")
        require(receipt["ir_hash"] == bound_ir["ir_hash"] and receipt["jq_candidate_allowed"], "JQ acceptance/IR mismatch")
        allowed = True
    frozen = {"ir": ir, "execution": market["execution"], "backtest_orders_allowed": allowed,
              "instrument_mapping":mapping,
              "acceptance_receipt": acceptance_receipt, "purpose": "research_backtest" if allowed else "strategy_diagnostic",
              "research_ir_hash": bound_ir["ir_hash"],
              "observed_on_jq": False, "live_execution_allowed": False}
    root, sources = Path(__file__).parent, []
    for name in ("contracts.py", "portfolio.py"):
        source = (root/name).read_text(encoding="utf-8")
        source = "\n".join(line for line in source.splitlines() if not line.startswith("from ."))
        # mix_scores is a research helper whose local import is never used by the adapter.
        source = source.split("\ndef mix_scores(")[0]
        sources.append(source)
    if market['execution'].get('instrument_rules') is not None:
        require(market['execution'].get('order_type')=='market','explicit instrument rules require the supported market-order JQ adapter')
        import inspect
        from .pit_universe import canonical_code
        from .a_share import instrument_rule
        for code in mapping:instrument_rule(market['execution'],code)
        source=(root/'a_share.py').read_text(encoding='utf-8')
        source='\n'.join(line for line in source.splitlines() if not line.startswith('from .'))
        sources += [inspect.getsource(canonical_code),source,'board_sell_quantity=sell_quantity']
    source = "\n\n".join(sources)+"\nFROZEN=json.loads("+repr(canonical(frozen).decode())+")\nFROZEN_HASH="+repr(digest(frozen))+"\n"+RUNTIME
    source = _py27_compat(source)
    compile(source, "jq_strategy.py", "exec")
    return source.encode(), {"purpose": frozen["purpose"], "ir_hash": ir["ir_hash"],
                             "script_sha256": hashlib.sha256(source.encode()).hexdigest(),
                             "backtest_orders_allowed": allowed, "observed_on_jq": False,
                             "parity_status": "requires_external_signal_order_account_reconciliation",
                             "model_serving": "frozen_historical_walk_forward_plan_not_live_retraining"}
