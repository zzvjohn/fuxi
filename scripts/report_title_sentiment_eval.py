# -*- coding: utf-8 -*-
"""
P-20260830-005: 研报标题情绪 LLM 打分因子评估实验
================================================================================
国联民生 (8-18): 分析师标题因子对传统预期因子明显增量; 调研文本因子全市场
RankIC 3.2%/ICIR 2.8/胜率 80%。本地 report_rc 68.7K 研报含标题 = 零采集成本文本源。

本脚本 (评估实验, 不入主线):
  1. 读 data/raw/report_rc.csv, 过滤近 2 年 + 有标题
  2. 按日期分层抽样 N 条标题 → DeepSeek 批量打分 (sentiment 0/1/2 + surprise 0/1)
  3. 按 (ts_code, 月末) 聚合为截面因子: senti_mean / surprise_ratio / n_reports
  4. Rank IC 检验 vs 次月收益 (daily_prices close)
  5. 输出 output/report_title_sentiment_eval.csv + 终端报告

LLM 调用预算: N=300 时 6 批 × 1 次调用 (每批 50 条标题合一个 prompt)。
--dry-run: 不调 API, 用 rating 字段映射伪打分, 验证管道 (零成本冒烟)。

用法:
  python scripts/report_title_sentiment_eval.py --sample 300
  python scripts/report_title_sentiment_eval.py --dry-run
"""
import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_FA_DIR = _PROJECT_ROOT / "research" / "factor_alchemy"
_OUT = _PROJECT_ROOT / "output" / "report_title_sentiment_eval.csv"

random.seed(42)

SENTIMENT_PROMPT_TMPL = (
    "你是A股券商研报情绪标注员。以下是{0}条研报标题(已编号), 请对每条输出JSON:\n"
    "sentiment: 0=悲观/看空/下调, 1=中性/维持, 2=乐观/看多/上调/超预期;\n"
    "surprise: 1=含\"超预期/大超/超指引/超wind一致预期\"等超预期表述, 0=无。\n"
    "标题列表:\n{1}\n"
    "只输出JSON数组, 形如: [{{\"i\":0,\"sentiment\":2,\"surprise\":1}}, ...]\n"
)


def _parse_llm_json(text: str, n: int):
    """从 LLM 回复中提取 JSON 数组 (容忍 markdown 包裹)。"""
    import re
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return None
    out = {}
    for item in arr:
        if isinstance(item, dict) and "i" in item:
            out[int(item["i"])] = {
                "sentiment": int(item.get("sentiment", 1)),
                "surprise": int(item.get("surprise", 0)),
            }
    return out


def score_titles_llm(titles, batch=50, dry_run=False, rating_map=None):
    """titles: list[str] → {idx: {sentiment, surprise}}"""
    scores = {}
    for start in range(0, len(titles), batch):
        chunk = titles[start:start + batch]
        if dry_run:
            # 伪打分: rating 字段映射 (冒烟用)
            rm = rating_map or {}
            for i, t in enumerate(chunk):
                r = str(rm.get(start + i, "") or "")
                s = 2 if r in ("买入", "增持", "强烈推荐", "推荐") else (0 if r in ("卖出", "减持") else 1)
                scores[start + i] = {"sentiment": s, "surprise": 0}
            continue
        numbered = "\n".join(f"{i}. {t}" for i, t in enumerate(chunk))
        prompt = SENTIMENT_PROMPT_TMPL.format(len(chunk), numbered)
        sys.path.insert(0, str(_FA_DIR))
        from llm_client import get_llm_client
        client = get_llm_client()
        resp = client.chat_with_system(
            system_prompt="你是金融文本标注引擎, 只输出JSON数组, 不解释。",
            user_prompt=prompt, temperature=0.0, max_tokens=4096,
        )
        parsed = _parse_llm_json(resp, len(chunk))
        if parsed is None:
            print(f"  [LLM] 批 {start // batch + 1} 解析失败, 该批置中性")
            for i in range(len(chunk)):
                scores[start + i] = {"sentiment": 1, "surprise": 0}
        else:
            for i in range(len(chunk)):
                scores[start + i] = parsed.get(i, {"sentiment": 1, "surprise": 0})
    return scores


def load_monthly_close(prices_path, codes):
    """daily_prices.csv (data/raw) → {code: Series[月末 close]} (只取需要股票的月末行)。"""
    try:
        head = pd.read_csv(prices_path, nrows=0)
        cols = head.columns.tolist()
        date_col = "trade_date" if "trade_date" in cols else "date"
        code_col = "ts_code" if "ts_code" in cols else ("stock_code" if "stock_code" in cols else "code")
        usecols = [date_col, code_col, "close"]
    except Exception:
        return {}
    reader = pd.read_csv(prices_path, usecols=usecols, chunksize=2_000_000,
                         dtype={code_col: str})
    frames = []
    for chunk in reader:
        chunk = chunk[chunk[code_col].isin(codes)]
        if chunk.empty:
            continue
        chunk[date_col] = pd.to_datetime(chunk[date_col])
        frames.append(chunk)
    if not frames:
        return {}
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=[date_col, code_col], keep="last")
    df["ym"] = df[date_col].dt.to_period("M")
    # 月末收盘 = 每月最后交易日的 close
    last = df.sort_values(date_col).groupby([code_col, "ym"], as_index=False).tail(1)
    out = {}
    for code, g in last.groupby(code_col):
        g = g.sort_values("ym")
        out[code] = pd.Series(g["close"].values, index=g["ym"].values)
    return out


def main():
    ap = argparse.ArgumentParser(description="研报标题情绪因子评估 (P-20260830-005)")
    ap.add_argument("--sample", type=int, default=300, help="抽样标题条数")
    ap.add_argument("--batch", type=int, default=50, help="每批 LLM 条数")
    ap.add_argument("--dry-run", action="store_true", help="伪打分冒烟, 不调 API")
    ap.add_argument("--prices", type=str,
                    default=str(_PROJECT_ROOT / "data" / "raw" / "daily_prices.csv"))
    args = ap.parse_args()

    rc_path = _PROJECT_ROOT / "data" / "raw" / "report_rc.csv"
    df = pd.read_csv(rc_path, usecols=["ts_code", "report_date", "report_title", "rating"],
                     dtype={"ts_code": str})
    # report_date 为 %Y%m%d 整型字符串 (如 20240331), 必须显式 format 否则解析失败
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce", format="%Y%m%d")
    # 评估窗口 = 最近 12 个月 (样本密度优先于时间跨度: 每月 40+ 只股票截面
    # 才支撑有意义的 Rank IC 检验; 24 个月分散抽样导致每月 <12 只不可用)
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=365)
    df = df[df["report_date"] >= cutoff]
    df = df[df["report_title"].notna() & (df["report_title"].str.strip() != "")]
    print(f"[Eval] 近12个月有标题研报: {len(df)} 条")

    if len(df) > args.sample:
        per_month = max(2, args.sample // 12)
        df = df.groupby(df["report_date"].dt.to_period("M"), group_keys=False).apply(
            lambda g: g.sample(n=min(len(g), per_month), random_state=42)
        )
        if len(df) > args.sample:
            df = df.sample(n=args.sample, random_state=42)
    print(f"[Eval] 抽样: {len(df)} 条")

    titles = df["report_title"].tolist()
    rating_map = {i: r for i, r in enumerate(df["rating"].tolist())}
    scores = score_titles_llm(titles, batch=args.batch, dry_run=args.dry_run,
                              rating_map=rating_map)
    df["sentiment"] = [scores.get(i, {"sentiment": 1, "surprise": 0})["sentiment"]
                       for i in range(len(df))]
    df["surprise"] = [scores.get(i, {"sentiment": 1, "surprise": 0})["surprise"]
                      for i in range(len(df))]

    # 月末聚合因子
    df["ym"] = df["report_date"].dt.to_period("M")
    agg = df.groupby(["ts_code", "ym"]).agg(
        senti_mean=("sentiment", "mean"),
        surprise_ratio=("surprise", "mean"),
        n_reports=("sentiment", "count"),
    ).reset_index()

    # 次月收益
    codes = agg["ts_code"].unique().tolist()
    mclose = load_monthly_close(args.prices, codes)
    fwd_rows = []
    for code, s in mclose.items():
        if len(s) < 2:
            continue
        ret = s.pct_change().shift(-1)  # 次月收益
        fwd_rows.append(pd.DataFrame({"ts_code": code, "ym": ret.index, "fwd_ret": ret.values}))
    fwd = pd.concat(fwd_rows, ignore_index=True) if fwd_rows else pd.DataFrame(
        columns=["ts_code", "ym", "fwd_ret"])
    fwd["ym"] = fwd["ym"].astype("period[M]")  # 与 agg.ym 类型对齐
    merged = agg.merge(fwd, on=["ts_code", "ym"], how="inner")
    print(f"[Eval] 聚合后截面样本: {len(merged)} 条 (股票×月), "
          f"覆盖 {merged['ts_code'].nunique()} 只股票")

    # Rank IC 检验
    results = []
    for factor in ("senti_mean", "surprise_ratio", "n_reports"):
        ics = []
        for ym, g in merged.groupby("ym"):
            if len(g) < 8:
                continue
            ic = g[factor].rank().corr(g["fwd_ret"].rank())
            if not np.isnan(ic):
                ics.append(ic)
        if ics:
            a = np.array(ics)
            icir = a.mean() / a.std() if a.std() > 0 else 0.0
            results.append({
                "factor": factor, "n_months": len(ics),
                "rank_ic": round(float(a.mean()), 4),
                "icir": round(float(icir), 3),
                "pos_ratio": round(float((a > 0).mean()), 3),
            })
        else:
            results.append({"factor": factor, "n_months": 0,
                            "rank_ic": None, "icir": None, "pos_ratio": None})

    print("\n[Eval] Rank IC 检验 (vs 次月收益):")
    for r in results:
        if r["n_months"]:
            print(f"  {r['factor']:16s} IC={r['rank_ic']:+.4f} ICIR={r['icir']:+.3f} "
                  f"胜率={r['pos_ratio']:.1%} (n={r['n_months']}月)")
        else:
            print(f"  {r['factor']:16s} 样本不足")

    merged.to_csv(_OUT, index=False, encoding="utf-8-sig")
    print(f"\n[Eval] 因子明细 → {_OUT}")


if __name__ == "__main__":
    main()
