"""Diagnostic style regression. Residuals are never silently substituted into portfolio signals."""

from statistics import mean
from .contracts import require, digest, finite
from .evaluation import correlation
from .dsl import ranks


def style_attribution(scores, style_panels, labels, snapshot, research_clock, *, provenance=None):
    if snapshot.payload.get("schema_version") == "fuxi.columnar_snapshot.1":
        from .columnar import style_attribution_columnar
        return style_attribution_columnar(scores, style_panels, labels, snapshot, research_clock, provenance=provenance)
    import numpy as np
    from .contracts import timestamp
    names = sorted(style_panels)
    require(names, "explicit style exposures required")
    rows = []
    clock = timestamp(research_clock)
    for day, cross in scores.items():
        sample = []
        for code, value in cross.items():
            features = [style_panels[name].get(day, {}).get(code) for name in names]
            target = labels.value(day, code, clock)
            if value is not None and target is not None and all(v is not None for v in features):
                sample.append((value, target, features, code))
        if len(sample) <= len(names)+2:
            rows.append({"session": day, "n_pairs": len(sample), "status": "data_insufficient"})
            continue
        values = np.asarray([r[0] for r in sample])
        y = [r[1] for r in sample]
        x = np.asarray([r[2] for r in sample])
        x = x-x.mean(axis=0)
        scale = x.std(axis=0)
        x = x/np.where(scale>0,scale,1)
        design = np.column_stack([np.ones(len(x)),x])
        coefficients, _, rank, _ = np.linalg.lstsq(design,values,rcond=None)
        explained = design@coefficients
        residual = values-explained
        raw_ranks = list(ranks(dict(enumerate(values.tolist()))).values())
        y_ranks = list(ranks(dict(enumerate(y))).values())
        residual_identified = np.linalg.norm(residual) > 1e-12*max(1.,np.linalg.norm(values))
        residual_ranks = list(ranks(dict(enumerate(residual.tolist()))).values()) if residual_identified else None
        explained_ranks = list(ranks(dict(enumerate(explained.tolist()))).values())
        rows.append({"session":day,"n_pairs":len(sample),"status":"valid","design_rank":int(rank),
                     "design_columns":int(design.shape[1]), "rank_deficient":int(rank)<design.shape[1],
                     "raw_spearman_ic":correlation(raw_ranks,y_ranks),
                     "residual_spearman_ic":correlation(residual_ranks,y_ranks) if residual_ranks else None,
                     "raw_pearson_ic":correlation(values.tolist(),y),
                     "explained_spearman_ic":correlation(explained_ranks,y_ranks),
                     "explained_pearson_ic":correlation(explained.tolist(),y),
                     "decomposition_max_error":float(np.max(np.abs(values-explained-residual))),
                     "residual_pearson_ic":correlation(residual.tolist(),y) if np.linalg.norm(residual) > 1e-12*max(1.,np.linalg.norm(values)) else None,
                     "support_instruments":[r[3] for r in sample],
                     "support_hash":digest([r[3] for r in sample]),
                     "raw_signal":{r[3]:float(v) for r,v in zip(sample,values)},
                     "explained_signal":{r[3]:float(v) for r,v in zip(sample,explained)},
                     "residual_signal":{r[3]:float(v) for r,v in zip(sample,residual)},
                     "style_exposures":{name:{r[3]:float(r[2][i]) for r in sample} for i,name in enumerate(names)},
                     "coefficient_identification":"minimum_norm_nonunique" if int(rank)<design.shape[1] else "full_column_rank",
                     "coefficients":{name:float(v) for name,v in zip(["intercept"]+names,coefficients)}})
    return {"purpose":"attribution_only_not_investment_neutralization","style_names":names,"sections":rows,
            "n_pairs":sum(r["n_pairs"] for r in rows),
            "missing_styles":[n for n in ("PIT_industry","PIT_market_cap")
                              if not (provenance or {}).get(n+"_certified",False)],
            "style_provenance":provenance or {"status":"caller supplied diagnostic panels; PIT unverified"},
            "causal_identification":False}


def diagnostic_style_residual(scores, style_panels, *, objective):
    """Fit on decision-date signals/styles only. Labels cannot select trading support."""
    import numpy as np
    require(objective == 'diagnostic_style_residual', 'explicit diagnostic residual objective required')
    names=sorted(style_panels)
    require(scores and names, 'nonempty signal/style panels required')
    raw, residual, sections = {}, {}, []
    for day, cross in sorted(scores.items()):
        raw[day]={s:None for s in cross};residual[day]={s:None for s in cross}
        sample=[]
        for code,value in sorted(cross.items()):
            features=[style_panels[n].get(day,{}).get(code) for n in names]
            require(value is None or finite(value), 'nonfinite diagnostic signal')
            require(all(v is None or finite(v) for v in features), 'nonfinite diagnostic exposure')
            if value is not None and all(v is not None for v in features):sample.append((code,value,features))
        row={'session':day,'n_pairs':len(sample),'input_universe_count':len(cross),
             'support_instruments':[r[0] for r in sample], 'support_hash':digest([r[0] for r in sample]),
             'support_selected_using_labels':False}
        if len(sample)<=len(names)+2:
            row.update(status='data_insufficient',reason='insufficient_cross_section_for_styles')
            sections.append(row);continue
        values=np.asarray([r[1] for r in sample],dtype=float);x=np.asarray([r[2] for r in sample],dtype=float)
        centered=x-x.mean(axis=0);scale=centered.std(axis=0)
        design=np.column_stack([np.ones(len(x)),centered/np.where(scale>0,scale,1.)])
        beta,_,rank,_=np.linalg.lstsq(design,values,rcond=None)
        explained=design@beta;errors=values-explained
        identified=np.linalg.norm(errors)>1e-12*max(1.,np.linalg.norm(values))
        # Numerical remnants of a completely explained signal cannot become a rank strategy.
        if identified:
            for (code,value,_),error in zip(sample,errors):raw[day][code]=value;residual[day][code]=float(error)
        exposure={}
        for i,name in enumerate(names):
            column=x[:,i].tolist();rank_x=list(ranks(dict(enumerate(column))).values())
            def readings(v):
                return {'pearson':correlation(v,column),
                        'spearman':correlation(list(ranks(dict(enumerate(v))).values()),rank_x)}
            exposure[name]={'raw':readings(values.tolist()),'residual':readings(errors.tolist()) if identified else None}
        row.update(status='valid' if identified else 'data_insufficient',
            reason='identified_residual' if identified else 'signal_fully_explained_no_residual_strategy',
            design_rank=int(rank),design_columns=int(design.shape[1]),rank_deficient=int(rank)<design.shape[1],
            coefficient_identification='minimum_norm_nonunique' if int(rank)<design.shape[1] else 'full_column_rank',
            raw_signal={r[0]:float(v) for r,v in zip(sample,values)},
            explained_signal={r[0]:float(v) for r,v in zip(sample,explained)},
            residual_signal={r[0]:float(v) for r,v in zip(sample,errors)},
            style_exposures={name:{r[0]:float(r[2][i]) for r in sample} for i,name in enumerate(names)},
            signal_style_correlations=exposure,
            decomposition_max_error=float(np.max(np.abs(values-explained-errors))))
        sections.append(row)
    require(sum(r['n_pairs'] for r in sections)>0, 'zero diagnostic style support')
    require(all({s for s,v in raw[d].items() if v is not None}==
                {s for s,v in residual[d].items() if v is not None} for d in raw),'residual account support differs')
    return {'schema_version':'fuxi.diagnostic_residual.1','objective':objective,'style_names':names,
            'raw_common_support':raw,'residual_common_support':residual,'sections':sections,
            'n_pairs':sum(r['n_pairs'] for r in sections),'valid_sections':sum(r['status']=='valid' for r in sections),
            'support_hash':digest({d:[s for s,v in sorted(x.items()) if v is not None] for d,x in raw.items()}),
            'labels_used_for_fit_or_support':False,'production_signal_substitution':False,
            'investment_neutralization_certified':False,'formal_eligibility':False,'causal_claim_allowed':False}
