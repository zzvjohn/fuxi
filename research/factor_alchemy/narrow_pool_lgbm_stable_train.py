# -*- coding: utf-8 -*-
"""
M4 窄池 LightGBM 名单稳定性重训 — fwd5 标签慢化信号 (2026-08-23)
====================================================================
动机: fwd3 版日换手 78.9% 死成本 (净 -92.71%)。参考策略唯一未证伪的能力是
名单稳定性 (日换手 12.6%)。重训方向: 标签 3日→5日, 让模型学"5日反弹"而非
"3日反弹", 信号自然慢变 → 名单稳定 → 换手可压。
与 fwd3 版共用特征工程 (29 特征), 唯一差异 = 标签 FWD_DAYS=5。
输出: narrow_pool_lgbm_fwd5_oos_preds.csv (逐日池内 LGBM 分数, 无未来泄漏)
"""
import sys
import json
import warnings
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
warnings.filterwarnings("ignore")

from narrow_pool_lgbm_train import (
    build_features, load_mf_wide, build_pool_membership, make_dataset, eval_oos)

RAW_DIR = BASE.parent.parent / "data" / "raw"
OUT_DIR = BASE.parent / "data"
PRED_CSV = OUT_DIR / "narrow_pool_lgbm_fwd5_oos_preds.csv"
IMP_CSV = OUT_DIR / "narrow_pool_lgbm_fwd5_feature_importance.csv"
OUT_JSON = OUT_DIR / "narrow_pool_lgbm_fwd5_train.json"

POOL_N = 1000
TOP_K = 8
FWD_DAYS = 5                      # <-- 与 fwd3 版唯一实质差异
TRAIN_LEN, TEST_LEN, STEP = 500, 250, 250
PARAMS = dict(
    objective="regression", metric="rmse",
    num_leaves=31, learning_rate=0.05,
    min_data_in_leaf=200, feature_fraction=0.8,
    bagging_fraction=0.8, bagging_freq=1,
    lambda_l2=1.0, verbosity=-1, num_threads=8,
)
NUM_ROUNDS = 400


def main():
    print("=" * 68)
    print("  窄池 LightGBM — fwd5 慢信号重训 (池=超跌1000, 标签=fwd5)")
    print("=" * 68)
    from p0_conditional_line import load_wide_light
    close_w, open_w, vol_w, lhb_w = load_wide_light()
    price_csv = RAW_DIR / "daily_prices.csv"
    df = pd.read_csv(price_csv, usecols=["ts_code", "trade_date", "high", "low", "amount"],
                     low_memory=False)
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="mixed", errors="coerce")
    df = df.dropna(subset=["trade_date"]).drop_duplicates(
        subset=["ts_code", "trade_date"], keep="last")
    dates = close_w.index
    high_w = df.pivot(index="trade_date", columns="ts_code", values="high").sort_index() \
        .reindex(index=dates, columns=close_w.columns)
    low_w = df.pivot(index="trade_date", columns="ts_code", values="low").sort_index() \
        .reindex(index=dates, columns=close_w.columns)
    amount_w = df.pivot(index="trade_date", columns="ts_code", values="amount").sort_index() \
        .reindex(index=dates, columns=close_w.columns)
    del df
    gc.collect()
    print(f"  宽表: {len(dates)} 交易日 | {close_w.shape[1]} 只 | "
          f"{dates[0].date()} ~ {dates[-1].date()}")
    mf_w = load_mf_wide(dates, close_w.columns)

    feats = build_features(close_w, open_w, high_w, low_w, vol_w, amount_w, lhb_w, mf_w)
    print(f"  特征 {len(feats)} 个: {list(feats.keys())}")

    fwd5 = close_w.shift(-FWD_DAYS) / close_w - 1.0
    pool_by_day = build_pool_membership(feats["dev_ma20"], dates)
    labels = fwd5
    print(f"  池构造完成, 标签 = fwd{FWD_DAYS}")

    n = len(dates)
    fold_summary = []
    all_imp = None
    oos_frames = []
    fi = 0
    while fi + TRAIN_LEN + TEST_LEN <= n:
        tr = (fi, fi + TRAIN_LEN)
        te = (fi + TRAIN_LEN, min(fi + TRAIN_LEN + TEST_LEN, n))
        X, y, d, c = make_dataset(feats, pool_by_day, labels, tr, dates)
        print(f"\n  fold{fi // STEP}: train {dates[tr[0]].date()}~{dates[tr[1]-1].date()} "
              f"({len(X)} 样本) → test {dates[te[0]].date()}~{dates[te[1]-1].date()}")
        if len(X) < 10000:
            print("    [SKIP] 样本不足")
            fi += STEP
            continue
        ds = lgb.Dataset(X, label=y)
        model = lgb.train(PARAMS, ds, num_boost_round=NUM_ROUNDS)
        ic_a, ex_a, yearly, rows, preds = eval_oos(
            model, feats, pool_by_day, labels, dates, te)
        ic_mean = float(ic_a.mean()) if len(ic_a) else np.nan
        ic_t = float(ic_mean / ic_a.std() * np.sqrt(len(ic_a))) if len(ic_a) > 1 and ic_a.std() > 0 else 0.0
        ex_mean = float(ex_a.mean())
        ex_sharpe = float(ex_mean / ex_a.std() * np.sqrt(240)) if ex_a.std() > 0 else 0.0
        yrs = {str(k): round(100 * float(np.mean(v)), 3) for k, v in yearly.items()}
        pos_y = sum(1 for v in yearly.values() if np.mean(v) > 0)
        print(f"    OOS 池内 IC={ic_mean:.4f} (t={ic_t:.1f}) | top{TOP_K} 超额 "
              f"{100 * ex_mean:.4f}%/日 → 年化 {240 * 100 * ex_mean:.1f}% | "
              f"日Sharpe {ex_sharpe:.2f} | 分年 {yrs} | {pos_y}/{len(yearly)} 正")
        fold_summary.append({
            "fold": fi // STEP,
            "train": [str(dates[tr[0]].date()), str(dates[tr[1] - 1].date())],
            "test": [str(dates[te[0]].date()), str(dates[te[1] - 1].date())],
            "ic": round(ic_mean, 4), "ic_t": round(ic_t, 1),
            "top8_excess_daily": round(100 * ex_mean, 4),
            "ann_excess": round(100 * 240 * ex_mean, 1),
            "sharpe": round(ex_sharpe, 2),
            "yearly_daily_excess": yrs,
            "positive_years": f"{pos_y}/{len(yearly)}",
        })
        imp = pd.DataFrame({
            "feature": model.feature_name(),
            "gain": model.feature_importance("gain"),
        }).sort_values("gain", ascending=False)
        all_imp = imp if all_imp is None else all_imp.merge(
            imp, on="feature", how="outer", suffixes=("", f"_f{fi // STEP}"))
        rec = []
        for i in rows:
            pool = pool_by_day[i]
            rec.append(pd.DataFrame({
                "trade_date": np.repeat(dates[i], len(pool)),
                "ts_code": np.array(pool),
                "pred": preds[i].astype(np.float32),
            }))
        oos_frames.append(pd.concat(rec, ignore_index=True))
        fi += STEP

    oos_all = pd.concat(oos_frames, ignore_index=True)
    oos_all.to_csv(PRED_CSV, index=False)
    print(f"\n  OOS 预测已存: {PRED_CSV} ({len(oos_all)} 行)")
    all_imp.to_csv(IMP_CSV, index=False)
    print(f"  特征重要性已存: {IMP_CSV}")
    payload = {
        "generated_at": str(pd.Timestamp.now()),
        "pool_n": POOL_N, "top_k": TOP_K, "fwd_days": FWD_DAYS,
        "params": PARAMS, "num_rounds": NUM_ROUNDS,
        "train_len": TRAIN_LEN, "test_len": TEST_LEN, "step": STEP,
        "n_features": len(feats),
        "folds": fold_summary,
    }
    json.dump(payload, open(OUT_JSON, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=str)
    print(f"  训练汇总已存: {OUT_JSON}")


if __name__ == "__main__":
    main()
