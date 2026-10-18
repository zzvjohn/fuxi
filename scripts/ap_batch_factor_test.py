#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AlphaPurify 批量因子归因 — FA v7.3 全量因子
=============================================
1. 预计算周频价格 + 风格代理 (一次性)
2. 遍历所有因子CSV → PureExposures A/B组对比
3. 输出 alpha纯度排名表 (CSV + HTML)

用法:
  python ap_batch_factor_test.py              # 全量
  python ap_batch_factor_test.py --resume     # 断点续跑
  python ap_batch_factor_test.py --workers 4  # 并行数
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys, io, json, time, argparse, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

warnings.filterwarnings("ignore")
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = Path(__file__).parent.parent
FACTOR_DIR = BASE / "research" / "factor_alchemy" / "output"
PRICE_FILE = BASE / "data" / "raw" / "daily_prices.csv"
OUTPUT_DIR = BASE / "output" / "ap_batch"
CACHE_DIR = OUTPUT_DIR / "cache"

START_DATE = "2021-01-01"
END_DATE = "2026-06-25"
MIN_WEEKS = 50  # 最少需要50周数据

STYLE_COLS = ["momentum_12_1", "vol_60", "log_size"]


def setup_dirs():
    for d in [OUTPUT_DIR, CACHE_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def load_price_weekly():
    """预计算周频价格数据 + 风格代理，缓存"""
    cache_path = CACHE_DIR / "weekly_prices.parquet"
    if cache_path.exists():
        print(f"  从缓存加载: {cache_path}")
        return pd.read_parquet(cache_path)

    print("  计算周频价格数据...")
    prices = pd.read_csv(PRICE_FILE, encoding="utf-8")
    prices["trade_date"] = pd.to_datetime(prices["trade_date"], format="mixed")
    prices = prices[prices["trade_date"].between(START_DATE, END_DATE)]
    prices = prices.sort_values(["ts_code", "trade_date"])

    # 日收益
    prices["daily_ret"] = prices.groupby("ts_code")["close"].pct_change()

    # → 周频
    prices["week"] = prices["trade_date"].dt.to_period("W").dt.end_time.dt.normalize()
    weekly = prices.groupby(["ts_code", "week"], observed=True).agg(
        close=("close", "last"),
        vol=("vol", "sum"),
        amount=("amount", "sum"),
    ).reset_index()
    weekly = weekly.rename(columns={"week": "trade_date", "ts_code": "stock_code"})
    weekly = weekly.sort_values(["stock_code", "trade_date"])

    # 前向收益
    weekly["fwd_ret"] = weekly.groupby("stock_code")["close"].pct_change().shift(-1)

    # 风格代理
    print("  计算 momentum_12_1...")
    weekly["momentum_12_1"] = weekly.groupby("stock_code")["close"].transform(
        lambda x: x.pct_change(52)
    )
    print("  计算 vol_60...")
    weekly["r"] = weekly.groupby("stock_code")["close"].pct_change()
    weekly["vol_60"] = weekly.groupby("stock_code")["r"].transform(
        lambda x: x.rolling(12, min_periods=6).std()
    )
    weekly = weekly.drop(columns=["r"])

    print("  计算 log_size...")
    weekly["log_size"] = np.log(
        weekly["close"].abs() * weekly["amount"].clip(lower=1) + 1e-8
    )
    weekly["log_size"] = weekly["log_size"].replace([np.inf, -np.inf], np.nan)

    weekly = weekly.dropna(subset=["fwd_ret", "close"])
    weekly.to_parquet(cache_path, index=False)
    print(f"  缓存已保存: {cache_path}")
    return weekly


def discover_factors():
    """发现所有因子CSV文件 → {factor_name: csv_path}"""
    factors = {}
    for csv_path in FACTOR_DIR.glob("factor_*.csv"):
        name = csv_path.stem.replace("factor_", "")
        factors[name] = str(csv_path)
    return factors


def load_factor(csv_path):
    """加载单个因子CSV → long format DataFrame"""
    df = pd.read_csv(csv_path)
    if str(df.columns[0]).startswith("Unnamed"):
        df.rename(columns={df.columns[0]: "trade_date"}, inplace=True)
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]

    factor_name = csv_path.stem.replace("factor_", "") if hasattr(csv_path, "stem") else Path(csv_path).stem.replace("factor_", "")

    df = df.melt(id_vars="trade_date", var_name="stock_code", value_name=factor_name)
    df["trade_date"] = pd.to_datetime(
        df["trade_date"].astype(str).str.replace(r"\.0$", "", regex=True),
        format="mixed", errors="coerce")
    df = df.dropna(subset=[factor_name])
    df = df[df["trade_date"].between(START_DATE, END_DATE)]
    return df, factor_name


def analyze_single_factor(args):
    """
    单因子PureExposures分析 (在子进程中运行)
    返回 dict with alpha purity metrics
    """
    csv_path, weekly_pd, factor_name = args

    try:
        from alphapurify import AlphaPurifier, PureExposures

        # 加载+合并
        df_fac, _ = load_factor(csv_path)
        if df_fac.empty or df_fac["trade_date"].nunique() < MIN_WEEKS:
            return {"factor": factor_name, "error": f"数据不足 ({df_fac['trade_date'].nunique()}周)"}

        merged = df_fac.merge(
            weekly_pd[["stock_code", "trade_date", "close", "fwd_ret"] + STYLE_COLS],
            on=["trade_date", "stock_code"], how="inner"
        )
        if merged.empty or merged["trade_date"].nunique() < MIN_WEEKS:
            return {"factor": factor_name, "error": f"合并后数据不足 ({merged['trade_date'].nunique()}周)"}

        result = {
            "factor": factor_name,
            "n_weeks": merged["trade_date"].nunique(),
            "n_stocks": merged["stock_code"].nunique(),
            "n_rows": len(merged),
        }

        # A组: MAD + Zscore
        t0 = time.time()
        pa = AlphaPurifier(merged, factor_name=factor_name,
                          trade_date_col="trade_date", symbol_col="stock_code")
        df_a = pa.winsorize("mad").standardize("zscore").to_result()

        pure_a = PureExposures(
            base_df=df_a, trade_date_col="trade_date", symbol_col="stock_code",
            price_col="close", factor_name=factor_name, exposure_cols=STYLE_COLS,
        )
        pure_a.run()

        # B组: MAD + Ridge(市值+波动) + Zscore
        pb = AlphaPurifier(merged, factor_name=factor_name,
                          trade_date_col="trade_date", symbol_col="stock_code")
        df_b = pb.winsorize("mad").neutralize(
            "ridge", STYLE_COLS
        ).standardize("zscore").to_result()

        pure_b = PureExposures(
            base_df=df_b, trade_date_col="trade_date", symbol_col="stock_code",
            price_col="close", factor_name=factor_name, exposure_cols=STYLE_COLS,
        )
        pure_b.run()

        elapsed = time.time() - t0

        # 提取归因指标
        for tag, pure in [("a_no_neut", pure_a), ("b_ridge_neut", pure_b)]:
            rdf = pure.result_df.to_pandas() if hasattr(pure.result_df, 'to_pandas') else pure.result_df
            cdf = pure.corr_df.to_pandas() if hasattr(pure.corr_df, 'to_pandas') else pure.corr_df

            port_col = f"{factor_name}_cum_ret"
            cum_cols = [c for c in rdf.columns if c.endswith("_cum_ret")]
            alpha_col = [c for c in rdf.columns if "alpha" in c.lower() or "Alpha" in c]

            # 总收益
            total_ret = rdf[port_col].dropna().iloc[-1] if port_col in rdf.columns else 0

            # Alpha贡献
            alpha_ret = rdf[alpha_col[0]].dropna().iloc[-1] if alpha_col else 0

            # 各风格贡献
            style_rets = {}
            for col in cum_cols:
                if col != port_col:
                    vals = rdf[col].dropna()
                    style_rets[col.replace("_cum_ret", "")] = float(vals.iloc[-1]) if len(vals) > 0 else 0

            # Alpha纯度 = |alpha| / (|alpha| + Σ|style|)
            style_abs_sum = sum(abs(v) for v in style_rets.values())
            purity = abs(alpha_ret) / (abs(alpha_ret) + style_abs_sum) if (abs(alpha_ret) + style_abs_sum) > 0.0001 else 0.5

            # 风格相关性
            corrs = {}
            for c in STYLE_COLS:
                if c in cdf.columns:
                    corrs[c] = float(cdf[c].mean())

            result[f"{tag}_total_ret"] = float(total_ret)
            result[f"{tag}_alpha_ret"] = float(alpha_ret)
            result[f"{tag}_purity"] = float(purity)
            for k, v in style_rets.items():
                result[f"{tag}_style_{k}"] = v
            for k, v in corrs.items():
                result[f"{tag}_corr_{k}"] = v
            result[f"{tag}_elapsed"] = elapsed

        return result

    except Exception as e:
        return {"factor": factor_name, "error": str(e), "traceback": traceback.format_exc()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="断点续跑")
    parser.add_argument("--workers", type=int, default=4, help="并行worker数")
    parser.add_argument("--max-factors", type=int, default=0, help="限制测试因子数(0=全部)")
    args = parser.parse_args()

    setup_dirs()

    # Step 1: 预计算价格数据
    print("=" * 60)
    print("Step 1: 预计算价格 + 风格因子")
    print("=" * 60)
    weekly_pd = load_price_weekly()
    print(f"  周频数据: {len(weekly_pd):,} 条, {weekly_pd['stock_code'].nunique()} 只股票, "
          f"{weekly_pd['trade_date'].nunique()} 周\n")

    # Step 2: 发现因子
    all_factors = discover_factors()
    print("=" * 60)
    print(f"Step 2: 发现 {len(all_factors)} 个因子")
    print("=" * 60)

    # 断点续跑
    results_path = OUTPUT_DIR / "batch_results.csv"
    completed = set()
    if args.resume and results_path.exists():
        existing = pd.read_csv(results_path)
        completed = set(existing["factor"].tolist())
        print(f"  已完成的因子: {len(completed)}")

    to_process = [(csv_path, weekly_pd, name)
                  for name, csv_path in all_factors.items()
                  if name not in completed]
    if args.max_factors > 0:
        to_process = to_process[:args.max_factors]

    print(f"  待处理: {len(to_process)} 个因子, 并行workers: {args.workers}\n")

    if not to_process:
        print("没有需要处理的因子!")
        return

    # Step 3: 并行处理
    print("=" * 60)
    print("Step 3: 批量PureExposures归因")
    print("=" * 60)

    all_results = list(completed_results if args.resume and results_path.exists() else [])
    start_time = time.time()
    completed_count = len(all_results)

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(analyze_single_factor, task): task[2]
                   for task in to_process}
        for future in as_completed(futures):
            fac_name = futures[future]
            try:
                res = future.result(timeout=600)
                all_results.append(res)
                completed_count += 1

                if "error" in res:
                    status = f"ERROR: {res['error'][:60]}"
                else:
                    purity_a = res.get("a_no_neut_purity", 0)
                    purity_b = res.get("b_ridge_neut_purity", 0)
                    status = f"α纯度: {purity_a:.1%}→{purity_b:.1%}"

                elapsed_total = time.time() - start_time
                rate = completed_count / (elapsed_total / 60) if elapsed_total > 0 else 0
                print(f"  [{completed_count}/{len(all_factors)}] {fac_name:35s} | {status} "
                      f"| {rate:.1f} 因子/min")

                # 增量保存
                pd.DataFrame(all_results).to_csv(results_path, index=False, encoding="utf-8-sig")

            except Exception as e:
                all_results.append({"factor": fac_name, "error": str(e)})
                print(f"  [{completed_count}] {fac_name:35s} | TIMEOUT/CRASH: {e}")

    print(f"\n  完成! 总耗时: {(time.time() - start_time) / 60:.1f} 分钟")

    # Step 4: 生成排名
    print("\n" + "=" * 60)
    print("Step 4: 生成因子纯度排名")
    print("=" * 60)

    df_results = pd.DataFrame(all_results)
    if "error" in df_results.columns:
        df_ok = df_results[df_results["error"].isna()].copy()
        df_err = df_results[df_results["error"].notna()]
        print(f"  成功: {len(df_ok)} 个, 失败: {len(df_err)} 个")
    else:
        df_ok = df_results.copy()

    if len(df_ok) == 0:
        print("没有成功结果!")
        return

    # 选择最优纯度 (取B组Ridge中性化后)
    df_ok["best_purity"] = df_ok["b_ridge_neut_purity"].fillna(df_ok["a_no_neut_purity"])
    df_ok["best_total_ret"] = df_ok["b_ridge_neut_total_ret"].fillna(df_ok["a_no_neut_total_ret"])
    df_ok["purity_change"] = df_ok["b_ridge_neut_purity"] - df_ok["a_no_neut_purity"]

    # 分级
    def grade(p):
        if pd.isna(p): return "?"
        if p >= 0.85: return "A (高纯度)"
        if p >= 0.60: return "B (中等)"
        return "C (低纯度)"

    df_ok["grade"] = df_ok["best_purity"].apply(grade)
    df_ok = df_ok.sort_values("best_purity", ascending=False)

    # 汇总输出
    summary_cols = ["factor", "n_weeks", "n_stocks",
                    "a_no_neut_purity", "b_ridge_neut_purity", "purity_change",
                    "best_purity", "best_total_ret", "grade"]
    summary = df_ok[[c for c in summary_cols if c in df_ok.columns]]

    print(f"\n  因子纯度排名 Top 20:")
    print(f"  {'Rank':<5} {'Factor':<32} {'Before':>8} {'After':>8} {'Δ':>7} {'Grade':12} {'Ret':>8}")
    print(f"  {'─' * 90}")
    for i, row in summary.head(20).iterrows():
        before = row.get("a_no_neut_purity", 0) or 0
        after = row.get("b_ridge_neut_purity", 0) or 0
        delta = after - before
        ret = row.get("best_total_ret", 0) or 0
        print(f"  {i+1:<5} {row['factor']:<32} {before:>7.1%} {after:>7.1%} "
              f"{delta:>+6.1%} {row['grade']:12} {ret:>+7.1%}")

    # 分级统计
    print(f"\n  分级统计:")
    for g in ["A (高纯度)", "B (中等)", "C (低纯度)"]:
        cnt = (df_ok["grade"] == g).sum()
        print(f"    {g}: {cnt} 个因子")

    # 中性化改善统计
    improved = (df_ok["purity_change"] > 0.01).sum()
    worsened = (df_ok["purity_change"] < -0.01).sum()
    unchanged = len(df_ok) - improved - worsened
    print(f"\n  Ridge中性化效果: ↑{improved}个 / ↓{worsened}个 / →{unchanged}个")

    # 保存
    summary.to_csv(OUTPUT_DIR / "factor_purity_ranking.csv", index=False, encoding="utf-8-sig")
    df_ok.to_csv(OUTPUT_DIR / "factor_purity_full.csv", index=False, encoding="utf-8-sig")

    # HTML 报告
    html = generate_html_report(df_ok, summary)
    html_path = OUTPUT_DIR / "factor_purity_report.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"\n  HTML报告: {html_path}")

    return html_path


def generate_html_report(df: pd.DataFrame, summary: pd.DataFrame) -> str:
    """生成交互式HTML报告"""
    # 简化版: 纯文本表格
    rows_html = ""
    for _, row in summary.iterrows():
        before = row.get("a_no_neut_purity", 0) or 0
        after = row.get("b_ridge_neut_purity", 0) or 0
        delta = after - before
        ret = row.get("best_total_ret", 0) or 0
        grade = row.get("grade", "?")

        # 颜色
        if grade.startswith("A"):
            gcolor = "#27ae60"
        elif grade.startswith("B"):
            gcolor = "#f39c12"
        else:
            gcolor = "#e74c3c"

        purity_color = "#27ae60" if after > 0.7 else ("#f39c12" if after > 0.4 else "#e74c3c")
        delta_color = "#27ae60" if delta > 0.01 else ("#e74c3c" if delta < -0.01 else "#95a5a6")
        ret_color = "#e74c3c" if ret < 0 else "#27ae60"

        rows_html += f"""<tr>
            <td>{row['factor']}</td>
            <td>{int(row.get('n_weeks', 0))}</td>
            <td style="color:{purity_color}">{before:.1%}</td>
            <td style="color:{purity_color}">{after:.1%}</td>
            <td style="color:{delta_color}">{delta:+.1%}</td>
            <td style="color:{ret_color}">{ret:+.2%}</td>
            <td style="color:{gcolor};font-weight:bold">{grade}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><title>FA v7.3 因子Alpha纯度排名</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }}
  h1 {{ color: #4fc3f7; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 15px; }}
  th {{ background: #16213e; padding: 10px; text-align: left; border-bottom: 2px solid #4fc3f7; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid #333; }}
  tr:hover {{ background: #16213e; }}
  .stats {{ background: #16213e; padding: 15px; border-radius: 8px; margin: 10px 0; }}
  .note {{ color: #95a5a6; font-size: 0.85em; margin-top: 20px; }}
</style></head>
<body>
<h1>🔬 FA v7.3 — 因子 Alpha 纯度排名</h1>
<div class="stats">
  <strong>方法:</strong> PureExposures 收益归因 | 暴露因子: momentum_12_1, vol_60, log_size<br>
  <strong>预处理:</strong> B组 = MAD缩尾 + Ridge中性化(市值+波动率) + Zscore标准化<br>
  <strong>纯度 =</strong> |Alpha累积| / (|Alpha| + Σ|风格贡献|)
</div>
<table>
<thead><tr>
  <th>因子</th><th>周数</th><th>纯化前</th><th>纯化后</th><th>改善</th><th>总收益</th><th>等级</th>
</tr></thead>
<tbody>{rows_html}</tbody>
</table>
<div class="note">
  生成时间: {time.strftime('%Y-%m-%d %H:%M')} | AlphaPurify v1.0.5 | 数据范围: 2021-2026/06
</div>
</body></html>"""


if __name__ == "__main__":
    completed_results = []  # 断点续跑时填充
    html_path = main()
    if html_path:
        print(f"\n✅ 全部完成! 最终报告: {html_path}")