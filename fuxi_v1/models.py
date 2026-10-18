"""Train actual competing models on mature labels; emit explicit walk-forward provenance."""

import importlib.util
import json
import math
from .contracts import digest, require, timestamp, finite


def capabilities():
    return {"ridge": "supported", "lightgbm": "supported" if importlib.util.find_spec("lightgbm") else "unsupported",
            "xgboost": "supported" if importlib.util.find_spec("xgboost") else "unsupported"}


def validate_frozen_model(state):
    """Verify both the fit parameters and optional native walk-forward provenance."""
    require(isinstance(state,dict), 'frozen model object required')
    algorithm=state.get('algorithm')
    names={'ridge':('coef','intercept','alpha'),'lightgbm':('model_text',),'xgboost':('model_json',)}
    require(algorithm in names, 'frozen model algorithm invalid')
    core=('algorithm','center','scale')+names[algorithm]
    require(all(k in state for k in core) and state.get('model_hash')==digest({k:state[k] for k in core}),
            'frozen model changed')
    if 'bundle_hash' in state:
        require(state['bundle_hash']==digest({k:v for k,v in state.items() if k!='bundle_hash'}),
                'frozen training provenance changed')
        require(state.get('features')==sorted(set(state.get('features',[])))
                and len(state['features'])==len(state['center']), 'frozen model feature order invalid')
        require(timestamp(state['latest_training_label_mature_at'])<=timestamp(state['training_cutoff']),
                'frozen model used immature training labels')
    else:
        require(set(state)==set(core)|{'model_hash'}, 'unbound frozen model metadata')
    return state


def predict_frozen_model(state, rows):
    """Consume the JSON state emitted by fit_model, without refitting or pickle."""
    import numpy as np
    validate_frozen_model(state)
    algorithm=state['algorithm']
    require(capabilities().get(algorithm) == 'supported', 'frozen model backend unavailable')
    x=np.asarray(rows,dtype=float);center=np.asarray(state['center'],dtype=float)
    scale=np.asarray(state['scale'],dtype=float)
    require(x.ndim==2 and len(x)>0 and len(center)>0 and x.shape[1]==len(center)
            and scale.shape==center.shape and np.isfinite(x).all()
            and np.isfinite(center).all() and np.isfinite(scale).all()
            and (scale>0).all(), 'frozen model input/normalization invalid')
    z=(x-center)/scale
    if algorithm=='ridge':
        coef=np.asarray(state['coef'],dtype=float)
        require(coef.shape==center.shape and np.isfinite(coef).all()
                and finite(state['intercept']), 'frozen ridge coefficients invalid')
        result=z@coef+state['intercept']
    elif algorithm=='lightgbm':
        from lightgbm import Booster
        result=Booster(model_str=state['model_text']).predict(z)
    else:
        from xgboost import Booster, DMatrix
        model=Booster();model.load_model(bytearray(state['model_json'],'utf8'))
        result=model.predict(DMatrix(z))
    require(np.asarray(result).shape==(len(x),) and np.isfinite(result).all(),
            'frozen model prediction invalid')
    return [float(v) for v in result]


def validate_axis(parent, child, axis):
    require(set(parent) == set(child), "unregistered comparison key")
    changed = {key for key in parent if parent[key] != child[key]}
    require(changed == {axis}, "comparison must change exactly one declared axis")
    if axis == "features":
        require(len(set(parent[axis]) ^ set(child[axis])) == 1, "feature axis must add/remove exactly one feature; rename is two")


def fit_model(algorithm, x, y, seed=0):
    import numpy as np
    require(algorithm in capabilities() and capabilities()[algorithm] == "supported", "model backend unavailable")
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    require(x.ndim == 2 and len(x) == len(y) and len(y) > x.shape[1]+1, "insufficient training matrix")
    require(np.isfinite(x).all() and np.isfinite(y).all(), "nonfinite training inputs")
    center, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale > 0, scale, 1.)
    z, mean_y = (x-center)/scale, float(y.mean())
    if algorithm == "ridge":
        alpha = 1.
        coef = np.linalg.solve(z.T@z + alpha*np.eye(z.shape[1]), z.T@(y-mean_y))
        state = {"algorithm": algorithm, "center": center.tolist(), "scale": scale.tolist(),
                 "coef": coef.tolist(), "intercept": mean_y, "alpha": alpha}
        predictor = lambda rows: ((np.asarray(rows)-center)/scale)@coef+mean_y
    elif algorithm == "lightgbm":
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(n_estimators=40, max_depth=3, num_leaves=7, min_child_samples=30,
                              learning_rate=.05, verbosity=-1, random_state=seed, n_jobs=1,
                              deterministic=True, force_col_wise=True)
        model.fit(z, y)
        state = {"algorithm": algorithm, "center": center.tolist(), "scale": scale.tolist(),
                 "model_text": model.booster_.model_to_string()}
        predictor = lambda rows: model.booster_.predict((np.asarray(rows)-center)/scale)
    else:
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=40, max_depth=3, learning_rate=.05, subsample=1.,
                             colsample_bytree=1., random_state=seed, n_jobs=1, tree_method="hist")
        model.fit(z, y)
        state = {"algorithm": algorithm, "center": center.tolist(), "scale": scale.tolist(),
                 "model_json": bytes(model.get_booster().save_raw(raw_format="json")).decode("utf-8")}
        predictor = lambda rows: model.predict((np.asarray(rows)-center)/scale)
    state["model_hash"] = digest(state)
    return state, predictor


def walk_forward(features, snapshot, labels, *, algorithm="ridge", train_start=None, predict_start=None,
                 retrain_every=20, min_train_sessions=40, lookback_sessions=160, gap_sessions=1, seed=0,
                 predict_end=None):
    require(features and retrain_every > 0 and gap_sessions >= 1, "invalid training protocol")
    if snapshot.payload.get("schema_version") == "fuxi.columnar_snapshot.1":
        from .columnar import walk_forward_columnar
        return walk_forward_columnar(features, snapshot, labels, algorithm=algorithm,
            train_start=train_start, predict_start=predict_start, retrain_every=retrain_every,
            min_train_sessions=min_train_sessions, lookback_sessions=lookback_sessions,
            gap_sessions=gap_sessions, seed=seed, predict_end=predict_end)
    names = sorted(features)
    days = snapshot.sessions[snapshot.start:]
    predict_start = predict_start or days[min_train_sessions+gap_sessions]
    require(predict_end is None or predict_end in days and predict_end>=predict_start,'prediction end outside declared scope')
    scores = {d: {s: None for s in snapshot.universe[d]} for d in days}
    states, provenance, fit = [], [], None
    last_fit = -retrain_every
    for index, day in enumerate(days):
        if predict_end is not None and day>predict_end:break
        if day < predict_start:
            continue
        cutoff = snapshot.clocks[day]
        if fit is None or index-last_fit >= retrain_every:
            train_days = [d for d in days[max(0,index-lookback_sessions):max(0,index-gap_sessions)]
                          if train_start is None or d >= train_start]
            x, y, maturity = [], [], []
            used_days = set()
            for training_day in train_days:
                for code in snapshot.universe[training_day]:
                    row = [features[name].get(training_day, {}).get(code) for name in names]
                    label = labels.value(training_day, code, cutoff)
                    if label is None or any(v is None for v in row):
                        continue
                    label_row = labels.rows[(training_day, code)]
                    require(timestamp(label_row["label_end"]) < cutoff, "training label overlaps prediction time")
                    x.append(row)
                    y.append(label)
                    maturity.append(label_row["mature_at"])
                    used_days.add(training_day)
            if len(used_days) < min_train_sessions or len(y) <= len(names)+1:
                provenance.append({"session": day, "status": "data_insufficient", "training_sessions": len(used_days)})
                continue
            state, fit = fit_model(algorithm, x, y, seed)
            state.update({"features": names, "training_cutoff": cutoff.isoformat(),
                          "latest_training_label_mature_at": max(maturity, key=timestamp),
                          "training_days": sorted(used_days), "training_rows": len(y),
                          "feature_snapshot_hash": snapshot.hash, "labels_hash": labels.hash,
                          "gap_sessions": gap_sessions, "score_origin": "walk_forward"})
            state["bundle_hash"] = digest(state)
            states.append(state)
            last_fit = index
        codes, matrix = [], []
        for code in snapshot.universe[day]:
            row = [features[name].get(day, {}).get(code) for name in names]
            if all(v is not None for v in row):
                codes.append(code)
                matrix.append(row)
        predicted = fit(matrix) if matrix else []
        for code, value in zip(codes, predicted):
            require(timestamp(states[-1]["latest_training_label_mature_at"]) <= timestamp(states[-1]["training_cutoff"]) <= cutoff,
                    "walk-forward provenance violation")
            scores[day][code] = float(value) if math.isfinite(float(value)) else None
        provenance.append({"session": day, "status": "valid", "model_hash": states[-1]["bundle_hash"],
                           "training_cutoff": states[-1]["training_cutoff"],
                           "latest_training_label_mature_at": states[-1]["latest_training_label_mature_at"],
                           "score_origin": "walk_forward", "prediction_count": len(codes)})
    return {"scores": scores, "models": states, "prediction_provenance": provenance,
            "algorithm": algorithm, "trained_model_count": len(states),
            "eval_status": "valid" if states else "data_insufficient"}


def model_experiment(payload, features, policy, baseline_scores, top_n, algorithm, seed):
    from .data import Snapshot, Labels
    from .evaluation import factor_statistics, DEFAULT_POLICY
    from .portfolio import strategy_spec, compile_strategy
    from .simulation import run_account
    from .research_evaluation import paired_accounts
    snapshot=Snapshot(payload["snapshot"])
    labels=Labels(payload["labels"],snapshot)
    request=policy["request"]
    fitted=walk_forward(features,snapshot,labels,algorithm=algorithm,predict_start=request["research_start"],
                        min_train_sessions=30,lookback_sessions=120,retrain_every=40,seed=seed)
    metric=factor_statistics({d:r for d,r in fitted["scores"].items() if d >= request["research_start"]},
                            snapshot,labels,DEFAULT_POLICY,payload["research_clock"])
    model_bundle={"algorithm":algorithm,"states":fitted["models"],"source_snapshot_hash":snapshot.hash,
                  "features":sorted(features),"protocol_hash":policy["policy_hash"]}
    model_id="model_"+digest(model_bundle)
    horizon=labels.payload["label_spec"]["horizon_sessions"]
    ir=compile_strategy(strategy_spec(model_id,top_n=top_n,max_weight=1/top_n,rebalance_sessions=horizon),
                        fitted["scores"],snapshot,digest(payload["market"]["execution"]),score_origin="walk_forward",
                        model_hash=digest(fitted["models"]))
    base_ir=compile_strategy(strategy_spec("model_baseline",top_n=top_n,max_weight=1/top_n,rebalance_sessions=horizon),
                            baseline_scores,snapshot,digest(payload["market"]["execution"]))
    account=run_account(ir,payload["market"],start=request["research_start"],end=request["research_end"])
    baseline=run_account(base_ir,payload["market"],start=request["research_start"],end=request["research_end"])
    e3=paired_accounts(account,baseline,request)
    if snapshot.payload["exposure"] == "synthetic":
        e3.update(economic_reward_eligible=False,evidence_scope="diagnostic",decision="not_applicable",reject_class=None)
    artifact={**fitted,"evaluation":metric,"strategy_ir":ir,"account":account,"baseline_account":baseline,"economic_evaluation":e3}
    summary={"status":fitted["eval_status"],"algorithm":algorithm,"features":sorted(features),
             "trained_model_count":fitted["trained_model_count"],"mean_rank_ic":metric["diagnostics"]["spearman"]["mean"],
             "scope":"walk_forward_on_research_exposed_history","portfolio_account_status":e3["eval_status"],"strategy_ir_hash":ir["ir_hash"]}
    return {"artifact":artifact,"model_bundle":model_bundle,"model_summary":summary}
