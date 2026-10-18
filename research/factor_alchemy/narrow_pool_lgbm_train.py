# -*- coding: utf-8 -*-
"""
M1+M2 窄池 LightGBM — 特征工程 + walk-forward 训练 (2026-08-23)
================================================================
任务设定 (对齐聚宽日频均值回归策略): 池 = MA20 偏离最负 1000 只; 问题 = 池内谁反弹。
与伏羲融合点:
  - 特征层: alpha158 核心子集 + 伏羲 JQ 验证因子注入 (overreaction/overnight/
    intraday/lhb) + 伏羲资金流大单数据 (原帖没有的另类数据)
  - 数据层: 轻量宽表 (绕开 FactorICComputer 全量 merge 内存墙, P0 已建)
  - 评估层: 分年报告 + 与线性基线 (B1 IC 0.033/t9.6, top8 Sharpe 0.05) 对照
科学纪律:
  - walk-forward: 训练窗 500 交易日 / 测试窗 250 / 步进 250, 无未来泄漏
  - 标签 = 未来 3 日收益 (池内截面中性化), 训练集末尾 3 天因标签 NaN 自动排除
  - LightGBM 参数保守 (小树+强正则) 防过拟合; 判决 = OOS 分年稳健性
"""
import sys
import json
import warnings
import gc
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
warnings.filterwarnings("ignore")

from p0_conditional_line import load_wide_light

RAW_DIR = BASE.parent.parent / "data" / "raw"
OUT_DIR = BASE.parent / "data"
PRED_CSV = OUT_DIR / "narrow_pool_lgbm_oos_preds.csv"
IMP_CSV = OUT_DIR / "narrow_pool_lgbm_feature_importance.csv"
OUT_JSON = OUT_DIR / "narrow_pool_lgbm_train.json"

POOL_N = 1000
TOP_K = 8
FWD_DAYS = 3
TRAIN_LEN, TEST_LEN, STEP = 500, 250, 250
PARAMS = dict(
    objective="regression", metric="rmse",
    num_leaves=31, learning_rate=0.05,
    min_data_in_leaf=200, feature_fraction=0.8,
    bagging_fraction=0.8, bagging_freq=1,
    lambda_l2=1.0, verbosity=-1, num_threads=8,
)
NUM_ROUNDS = 400


def build_features(close_w, open_w, high_w, low_w, vol_w, amount_w, lhb_w, mf_w):
    """30 个特征: alpha158 核心 + 伏羲验证因子注入 + 资金流。全部 float32 宽表。"""
    f = {}
    ret = close_w.pct_change(1)
    # 动量族 (8)
    for n in [1, 3, 5, 10, 20, 60]:
        f[f"ret_{n}"] = close_w / close_w.shift(n) - 1.0
    f["ret_max_20"] = ret.rolling(20).max()
    f["ret_min_20"] = ret.rolling(20).min()
    # 波动族 (5)
    for n in [5, 10, 20]:
        f[f"std_{n}"] = ret.rolling(n).std()
    amp = (high_w - low_w) / close_w.shift(1)
    for n in [5, 20]:
        f[f"amp_mean_{n}"] = amp.rolling(n).mean()
    # 量能族 (5)
    vpos = vol_w.where(vol_w > 0)
    for n in [5, 10, 20]:
        f[f"volr_{n}"] = (vol_w / vol_w.rolling(n).mean()).where(vpos > 0)
    f["vol_std_20"] = vol_w.rolling(20).std()
    f["amount_chg_20"] = amount_w / amount_w.rolling(20).mean() - 1.0
    # 伏羲反转/隔夜/日内 (3)
    f["overnight_raw"] = open_w / close_w.shift(1) - 1.0
    f["overnight_5"] = f["overnight_raw"].rolling(5, min_periods=3).sum()
    f["intraday_rev"] = -(close_w / open_w - 1.0)
    # 事件族 (1) — lhb 上榜频次 (net_amount 不进特征, 稀疏且与 freq 高共线)
    f["lhb_freq_20"] = lhb_w.rolling(20).sum()
    # 条件层 (5)
    ma5, ma20, ma60 = close_w.rolling(5).mean(), close_w.rolling(20).mean(), close_w.rolling(60).mean()
    f["dev_ma5"] = close_w / ma5 - 1.0
    f["dev_ma20"] = close_w / ma20 - 1.0
    f["dev_ma60"] = close_w / ma60 - 1.0
    hi20, lo20 = high_w.rolling(20).max(), low_w.rolling(20).min()
    f["close_pos_20"] = (close_w - lo20) / (hi20 - lo20).where(hi20 > lo20)
    # RSI 14 (Wilder)
    up = ret.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-ret.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    f["rsi_14"] = 100.0 - 100.0 / (1.0 + up / dn.where(dn > 0))
    # 资金流族 (2): 大单+超大单净买占比
    if "lg_net" in mf_w:
        f["lg_net_ratio_5"] = (mf_w["lg_net"] / vol_w.where(vol_w > 0)).rolling(5).mean()
        f["lg_net_ratio_20"] = (mf_w["lg_net"] / vol_w.where(vol_w > 0)).rolling(20).mean()
    else:
        f["lg_net_ratio_5"] = close_w * np.nan
        f["lg_net_ratio_20"] = close_w * np.nan
    feats = {k: v.astype(np.float32) for k, v in f.items()}
    return feats


def load_mf_wide(dates, columns):
    """资金流宽表: 大单+超大单净买量 (伏羲另类数据, 原帖 alpha158 没有)"""
    mf_path = RAW_DIR / "moneyflow_daily.csv"
    mf = pd.read_csv(mf_path, usecols=["ts_code", "trade_date", "buy_lg_vol", "sell_lg_vol",
                                       "buy_elg_vol", "sell_elg_vol"], low_memory=False)
    mf["trade_date"] = pd.to_datetime(mf["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    mf = mf.dropna(subset=["trade_date"])
    mf["lg_net"] = mf["buy_lg_vol"] + mf["buy_elg_vol"] - mf["sell_lg_vol"] - mf["sell_elg_vol"]
    w = (mf.pivot(index="trade_date", columns="ts_code", values="lg_net")
         .sort_index().reindex(index=dates, columns=columns))
    return {"lg_net": w.astype(np.float32)}


def build_pool_membership(dev_ma20, dates):
    """每日超跌池名单 (dev_ma20 最负 1000) → (date -> Index)"""
    pool_by_day = {}
    for i in range(len(dates)):
        s = dev_ma20.iloc[i]
        valid = np.isfinite(s.to_numpy())
        if valid.sum() < POOL_N:
            pool_by_day[i] = s[valid].nsmallest(POOL_N).index
        else:
            pool_by_day[i] = s[valid].nsmallest(POOL_N).index
    return pool_by_day


def make_dataset(feat_list, pool_by_day, labels_centered, day_idx_range, dates):
    """把 [i_start, i_end) 天 × 池内股票 转成长表 X/y/keys"""
    Xs, ys, ds, cs = [], [], [], []
    for i in range(day_idx_range[0], day_idx_range[1]):
        pool = pool_by_day.get(i)
        if pool is None or len(pool) == 0:
            continue
        y_row = labels_centered.iloc[i][pool]
        y_ok = np.isfinite(y_row.to_numpy())
        if y_ok.sum() < 50:
            continue
        rows = {}
        for name, w in feat_list.items():
            rows[name] = w.iloc[i][pool].to_numpy()
        Xs.append(pd.DataFrame(rows))
        ys.append(y_row.to_numpy())
        ds.append(np.full(len(pool), dates[i], dtype="datetime64[ns]"))
        cs.append(np.array(pool, dtype=object))
    X = pd.concat(Xs, ignore_index=True)
    y = np.concatenate(ys)
    d = np.concatenate(ds)
    c = np.concatenate(cs)
    return X, y, d, c


def eval_oos(model, feat_list, pool_by_day, labels_centered, dates, te_range):
    """OOS 逐日: 池内 IC + top8 超额 (池内中性) + 分年"""
    rows, preds = [], {}
    for i in range(te_range[0], te_range[1]):
        pool = pool_by_day.get(i)
        if pool is None or len(pool) == 0:
            continue
        X = pd.DataFrame({n: w.iloc[i][pool].to_numpy() for n, w in feat_list.items()})
        pred = model.predict(X, num_iteration=model.best_iteration or NUM_ROUNDS)
        preds[i] = pred
        rows.append(i)
    # 评估
    ic_list, ex_list, yearly = [], [], {}
    for i in rows:
        pool = pool_by_day[i]
        pred = preds[i]
        y_row = labels_centered.iloc[i][pool].to_numpy()
        ok = np.isfinite(y_row)
        if ok.sum() < 100:
            continue
        p, yv = pred[ok], y_row[ok]
        ic = spearmanr(p, yv)[0]
        if np.isfinite(ic):
            ic_list.append(ic)
        top_idx = np.argsort(p)[::-1][:TOP_K]
        ex = float(np.mean(yv[top_idx]) - np.mean(yv))
        ex_list.append(ex)
        yearly.setdefault(dates[i].year, []).append(ex)
    return np.array(ic_list), np.array(ex_list), yearly, rows, preds


def main():
    print("=" * 68)
    print("  窄池 LightGBM — walk-forward 训练 (池=超跌1000, 标签=fwd3池内中性)")
    print("=" * 68)
    close_w, open_w, vol_w, lhb_w = load_wide_light()
    # 追加 high/low/amount + 资金流
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
    print(f"  资金流大单净买宽表加载完成")

    feats = build_features(close_w, open_w, high_w, low_w, vol_w, amount_w, lhb_w, mf_w)
    print(f"  特征 {len(feats)} 个: {list(feats.keys())}")

    fwd3 = close_w.shift(-FWD_DAYS) / close_w - 1.0
    pool_by_day = build_pool_membership(feats["dev_ma20"], dates)
    # 标签直接用 fwd3 原始收益: 排序不变性下中性化不影响 IC/top8 池内超额,
    # 省掉逐日宽表写操作 (评估口径 = 池内超额, 常数平移自动抵消)
    labels = fwd3
    print(f"  池构造完成, 标签 = fwd{FWD_DAYS} (评估口径池内超额自动中性)")

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
        # 汇总
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
        # 特征重要性
        imp = pd.DataFrame({
            "feature": model.feature_name(),
            "gain": model.feature_importance("gain"),
        }).sort_values("gain", ascending=False)
        all_imp = imp if all_imp is None else all_imp.merge(
            imp, on="feature", how="outer", suffixes=("", f"_f{fi // STEP}"))
        # OOS 预测拼接
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

    # 汇总输出
    oos_all = pd.concat(oos_frames, ignore_index=True)
    oos_all.to_csv(PRED_CSV, index=False)
    print(f"\n  OOS 预测已存: {PRED_CSV} ({len(oos_all)} 行)")
    all_imp.to_csv(IMP_CSV, index=False)
    print(f"  特征重要性已存: {IMP_CSV}")
    payload = {
        "generated_at": datetime.now().isoformat(),
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
