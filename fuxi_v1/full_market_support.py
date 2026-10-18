"""Explicit all-market transport, identities, and diagnostic platform handling."""
from collections.abc import Mapping
import hashlib
from pathlib import Path
import numpy as np
from .contracts import digest, require


def hydrate_payload(payload):
    from .columnar import hydrate_market
    if payload['market'].get('schema_version') == 'fuxi.columnar_market.1':
        payload = dict(payload, market=hydrate_market(payload['market']))
    return payload


def pack_grids(value, directory):
    """Transport grids as verified arrays; StrategyIR score rows stay logical JSON."""
    from .columnar import ScoreGrid, save_score_grid
    cache = {}
    def visit(row):
        if isinstance(row, ScoreGrid):
            key=id(row)
            if key not in cache:
                cache[key]=save_score_grid(row, directory, 'grid_'+str(len(cache)))
            return cache[key]
        if isinstance(row, dict):return {k:visit(v) for k,v in row.items()}
        if isinstance(row, list):return [visit(v) for v in row]
        if isinstance(row, tuple):return [visit(v) for v in row]
        return row
    return visit(value)


def unpack_grids(value):
    from .columnar import SCORE_SCHEMA, load_score_grid
    cache = {}
    def visit(row):
        if isinstance(row, dict) and row.get('schema_version') == SCORE_SCHEMA:
            key=row['reference_hash']
            if key not in cache:cache[key]=load_score_grid(row)
            return cache[key]
        if isinstance(row, dict):return {k:visit(v) for k,v in row.items()}
        if isinstance(row, list):return [visit(v) for v in row]
        return row
    return visit(value)


def ordinal_identity(scores, start, end):
    """New typed full-section ordinal identity, independent of formula and paths."""
    from .columnar import ScoreGrid, rank_vector, selected_by_dates
    require(isinstance(scores, ScoreGrid), 'columnar scores required')
    selected=selected_by_dates(scores,start,end)
    content=hashlib.sha256();support=hashlib.sha256()
    for i in range(selected.start, selected.end):
        values=np.array(scores.matrix[i],dtype='<f8',copy=True)
        values[~scores.membership[i]]=np.nan
        ordinal=rank_vector(values).astype('<f8',copy=False)
        # Canonical quiet NaNs make physical null encoding independent of input NaN bits.
        ordinal[~np.isfinite(ordinal)]=np.nan
        content.update(ordinal.tobytes(order='C'))
        support.update(np.asarray(scores.membership[i],dtype=np.uint8).tobytes(order='C'))
    require(len(selected)>0, 'empty ordinal comparison')
    return digest({'schema_version':'fuxi.full_section_ordinal_identity.1',
        'sessions':list(selected), 'instruments':scores.instruments,
        'ordinal_bytes_sha256':content.hexdigest(),'membership_bytes_sha256':support.hexdigest(),
        'semantics':'complete-declared-eligible-section average ties; storage nulls retained'})


def strip_unsupported_venues(ir, venues):
    """Derive a JQ-executable IR by dropping venues the compiler cannot map.

    Scores and per-decision universes are stripped from a derived execution IR;
    the acceptance receipt keeps binding to the research IR via its unchanged
    ir_hash. P-20261015-013: the adapter is declared, never silently granted.
    """
    venues=set(venues)
    decisions=[]
    for row in ir['decisions']:
        scores={s:v for s,v in row['scores'].items() if s not in venues}
        universe=[s for s in row['universe'] if s not in venues]
        decisions.append({**row,'scores':scores,'universe':universe})
    derived={k:v for k,v in ir.items() if k!='ir_hash'}
    derived['decisions']=decisions
    derived['ir_hash']=digest(derived)
    return derived


def diagnostic_jq(ir, market, admission, store):
    from .jq_strategy import compile_jq_strategy
    unsupported=sorted({s for row in ir['decisions'] for s in row['scores']
                        if not s.endswith(('.XSHG','.XSHE','.SH','.SZ'))})
    if unsupported:
        if admission.get('jq_candidate_allowed'):
            # P-20261015-013: explicit JQ universe adapter. The acceptance receipt
            # still binds the research IR; the exported frozen plan strips venues
            # the JQ compiler cannot map (README: BSE mapping unverified, the
            # platform must not be claimed full-market deliverable).
            execution_ir=strip_unsupported_venues(ir,unsupported)
            script,compiler=compile_jq_strategy(execution_ir,market,store,
                                                admission['receipt_id'],research_ir=ir)
            compiler.update({'universe_adapter':{
                'schema_version':'fuxi.jq_universe_adapter.1',
                'research_ir_hash':ir['ir_hash'],
                'execution_ir_hash':execution_ir['ir_hash'],
                'unsupported_instruments':unsupported,
                'research_members_removed':len(unsupported),
                'status':'jq_venue_mapping_unverified_research_scope_declared'}})
            return script,compiler
        body={'purpose':'strategy_diagnostic','status':'unsupported_venue',
              'ir_hash':ir['ir_hash'],'unsupported_instruments':unsupported,
              'backtest_orders_allowed':False,'observed_on_jq':False,
              'research_members_removed':0,
              'reason':'full A-share research retained; JQ venue adapter not certified'}
        script=b'# Full A-share diagnostic: JQ export blocked by unsupported venues.\nraise RuntimeError("JQ venue adapter not certified for this frozen IR")\n'
        body['script_sha256']=hashlib.sha256(script).hexdigest()
        return script, body
    return compile_jq_strategy(ir,market,store,admission['receipt_id'])


def grid_bundle_files(scores, grid_reference):
    """Include exact array bytes in each immutable bundle and retain the transport binding."""
    from .columnar import ScoreGrid
    require(isinstance(scores,ScoreGrid),'columnar score bundle required')
    portable=dict(grid_reference,schema_version='fuxi.score_grid_bundle.1',
        native_transport_hash=grid_reference['reference_hash'])
    portable['values']=dict(grid_reference['values'],path='score_values.npy')
    portable['membership']=dict(grid_reference['membership'],path='score_membership.npy')
    portable['reference_hash']=digest({k:v for k,v in portable.items() if k!='reference_hash'})
    files={'scores.json':portable,
           'score_values.npy':Path(grid_reference['values']['path']).read_bytes(),
           'score_membership.npy':Path(grid_reference['membership']['path']).read_bytes()}
    return files


def load_grid_bundle(bundle):
    """Consume a self-contained immutable score bundle after relocation."""
    from .artifacts import verify,safe_name
    from .contracts import load_json
    from .columnar import load_score_grid
    root=Path(bundle).resolve();verify(root)
    packed=load_json(root/'scores.json')
    require(packed.get('schema_version')=='fuxi.score_grid_bundle.1', 'portable score schema required')
    require(packed['reference_hash']==digest({k:v for k,v in packed.items() if k!='reference_hash'}),
            'portable score binding changed')
    reference={k:v for k,v in packed.items() if k not in {'reference_hash','native_transport_hash'}}
    reference['schema_version']='fuxi.score_grid_ref.1'
    for field in ('values','membership'):
        filename=packed[field]['path'];safe_name(filename)
        path=(root/filename).resolve()
        require(path.is_relative_to(root),'portable score escaped bundle')
        reference[field]=dict(packed[field],path=str(path))
    reference['reference_hash']=digest(reference)
    return load_score_grid(reference)


def full_market_drift(reference,current,account_results=()):
    from .columnar import ScoreGrid
    require(isinstance(reference,ScoreGrid) and isinstance(current,ScoreGrid),'columnar drift scope required')
    def describe(grid):
        mask=grid.membership[grid.start:grid.end]
        values=grid.matrix[grid.start:grid.end]
        good=mask & np.isfinite(values)
        found=values[good]
        require(int(mask.sum())>0,'empty drift comparison')
        return len(found)/int(mask.sum()),float(np.mean(found)) if len(found) else None,\
            np.quantile(found,[.1,.5,.9]).tolist() if len(found) else None
    a,b=describe(reference),describe(current)
    unknown=sum(r['eval_status']!='valid' for r in account_results)
    return {'reference_coverage':a[0],'current_coverage':b[0],
            'mean_shift':b[1]-a[1] if a[1] is not None and b[1] is not None else None,
            'reference_quantiles':a[2],'current_quantiles':b[2],
            'account_unidentifiable':unknown,'automatic_historical_champion_switch':False,
            'recommended_action':'halt_new_orders_and_reconcile_if_active' if unknown else 'research_review'}


def execution_options(payload):
    """Research batch resource/portfolio parameters are explicit, never global defaults."""
    config=payload.get('full_market_execution')
    if config is None:return None
    require(payload['snapshot']['schema_version']=='fuxi.columnar_snapshot.1',
            'all-market execution options require the new columnar input')
    require(set(config)=={'schema_version','top_n','worker_timeout','worker_memory_mb',
                        'model_timeout','portfolio_timeout','budget_units'},'all-market execution fields differ')
    require(config['schema_version']=='fuxi.full_market_execution.1','unknown all-market execution contract')
    require(all(type(config[k]) is int and config[k]>0 for k in config if k!='schema_version'),
            'invalid all-market resource or portfolio setting')
    return config


def complete_generation(proposals, fields, count, seen, seed):
    """Preregistered quota repair for finite emitters; no outcome-based selection."""
    from .search import generate
    require(len(proposals)<=count and len({p['formula_hash'] for p in proposals})==len(proposals),
            'invalid initial generation quota')
    require(not ({p['formula_hash'] for p in proposals}&set(seen)),'repeated initial formula')
    missing=count-len(proposals);receipt={'initial_generated':len(proposals),
        'supplemented':0,'supplement_generator':None,'seed':seed,'generation_attempts':0}
    proposals=list(proposals)
    receipt.update(max_batches=64,batch_seed_stride=65537,executed_batches=0)
    if missing:
        for batch in range(64):
            counters={}
            extra=generate(fields,arm='random_ast',count=count-len(proposals),seed=seed+65537*batch,
                           seen=set(seen)|{p['formula_hash'] for p in proposals},counters=counters)
            proposals.extend(extra)
            receipt['supplemented']+=len(extra)
            receipt['generation_attempts']+=counters['generation_attempts']
            receipt.update(supplement_generator='random_ast',executed_batches=batch+1)
            if len(proposals)==count:break
    require(len(proposals)==count and len({p['formula_hash'] for p in proposals})==count,
            'all-market round generation quota not fulfilled')
    return proposals,receipt
