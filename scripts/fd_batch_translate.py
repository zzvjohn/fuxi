# -*- coding: utf-8 -*-
"""
factors.directory 量价类因子 latex -> pandas 批量翻译 (P-20260905-001 落地)
==================================================================================
- 输入: data/factors_directory_factors.json (量价类 tech/emotion/volatility/momentum/liquidity)
- 输出: data/_fd_tmp/translations.json ({slug: {formula_pandas, direction, rationale, skip_reason}})
- 断点续跑: 已翻译 slug 自动跳过
- LLM: DeepSeekClient (research/factor_alchemy/llm_client.py), 每批 12 个因子
- 用法: python scripts/fd_batch_translate.py [--batch-size 12] [--start 0] [--limit 0]
"""
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "research" / "factor_alchemy"))
from llm_client import DeepSeekClient

FD_PATH = BASE / "data" / "factors_directory_factors.json"
OUT_PATH = BASE / "data" / "_fd_tmp" / "translations.json"
QTY_CATS = {"tech", "emotion", "volatility", "momentum", "liquidity"}
FUND_CATS = {"basic-surface", "growth", "quality", "value", "scale"}

SYSTEM_PROMPT = """你是量化因子公式翻译专家。任务: 把学术研报的 LaTeX 因子公式翻译为可直接求值的 pandas 表达式 (formula_pandas)。

硬性规则:
1. 只使用 pandas Series 链式方法: .rolling(N).mean()/.std()/.skew()/.kurt()/.sum()/.min()/.max()/.corr()/.rank(pct=True)、.shift(N)、.diff(N)、.pct_change(N)、.div()、.mul()、.add()、.sub()、.abs()、.clip()、.where()、.cumsum()、.ewm(span=N).mean()。禁止自定义函数/apply/lambda。
2. 可用字段(列)白名单: close, open, high, low, volume, amount, pre_close, overnight, amplitude, returns, turnover, hl_ratio, circ_mv。禁止使用白名单外的任何字段。
3. 高阶矩用 .rolling(N).skew()/.kurt(); 横截面排序用 .rank(pct=True); 求和用 .rolling(N).sum() 或 .cumsum()。
4. 所有时间窗口 N 必须是整数常量, 直接写进 .rolling(N)。
5. 若因子机制是负向预期(如高换手负溢价/超买回落), 必须把负号显式写在公式开头 (如 -( ... )), 保证 formula_pandas 的期望 IC 为正。
6. 无法翻译的因子(需要分钟级数据/财务报表数据/跨股票配对/无明确公式), formula_pandas 设为 null, 并在 skip_reason 用中文说明原因。
7. formula_pandas 必须是单个可求值表达式, 不要多行, 不要赋值语句, 不要注释。
8. 公式里不要出现除数为零的保护(评估框架会处理), 但需要防止负值取根号等问题(用 .clip(lower=小正数) 替代)。

输出格式: 严格 JSON 数组, 每个元素:
{"slug": "<原slug>", "formula_pandas": "<pandas表达式或null>", "direction": 1 或 -1, "rationale": "<25字内中文机制简述>", "skip_reason": "<跳过原因或null>"}
只输出 JSON, 不要任何解释文字。"""

# 基本面批次专用白名单 (与 data_loader_ext 加载的 extra 字段一致):
# 行情: close/open/high/low/volume/amount/pre_close
# 规模: circ_mv (流通市值)
# 财务(季频前向填充到日频): roe/eps/eps_qoq/netprofit_yoy/roe_dt/tr_yoy
# 分析师: eps_disp/tp_disp/rating_disp/n_cover
FUND_SYSTEM_PROMPT = """你是量化因子公式翻译专家。任务: 把学术研报的 LaTeX 因子公式翻译为可直接求值的 pandas 表达式 (formula_pandas)。

硬性规则:
1. 只使用 pandas Series 链式方法: .rolling(N).mean()/.std()/.skew()/.kurt()/.sum()/.min()/.max()/.corr()/.rank(pct=True)、.shift(N)、.diff(N)、.pct_change(N)、.div()、.mul()、.add()、.sub()、.abs()、.clip()、.where()、.cumsum()、.ewm(span=N).mean()。禁止自定义函数/apply/lambda。
2. 可用字段(列)白名单: close, open, high, low, volume, amount, pre_close, circ_mv, roe, eps, eps_qoq, netprofit_yoy, roe_dt, tr_yoy, eps_disp, tp_disp, rating_disp, n_cover。禁止使用白名单外的任何字段。
3. 财务字段 (roe/eps 等) 已是季频值前向填充到日频的序列, 可直接用 .shift(N)/.diff(N) 表示季度变化; N 是交易日数 (约 63 为一季度, 252 为一年)。已有现成同比字段 (netprofit_yoy/roe_dt/tr_yoy/eps_qoq) 时直接复用, 不要重复构造。
4. 因子若需要总资产/总负债/股东权益/营业收入/毛利/净利/研发费用/现金流等资产负债表或利润表字段 → 白名单外, formula_pandas 设为 null, skip_reason 写 "需资产负债表/利润表字段(XX), 待财务数据接入"。
5. 市值类因子用 circ_mv (流通市值); 规模效应通常取对数或直接横截面 rank, 公式可直接写 circ_mv。
6. 若因子机制是负向预期(如高估值负溢价/高杠杆负溢价), 必须把负号显式写在公式开头, 保证 formula_pandas 的期望 IC 为正。
7. formula_pandas 必须是单个可求值表达式, 不要多行, 不要赋值语句, 不要注释。
8. 公式里不要出现除数为零的保护(评估框架会处理), 但需要防止负值取根号等问题(用 .clip(lower=小正数) 替代)。

输出格式: 严格 JSON 数组, 每个元素:
{"slug": "<原slug>", "formula_pandas": "<pandas表达式或null>", "direction": 1 或 -1, "rationale": "<25字内中文机制简述>", "skip_reason": "<跳过原因或null>"}
只输出 JSON, 不要任何解释文字。"""


def build_user_prompt(batch):
    lines = []
    for x in batch:
        lines.append("### slug: %s | 中文名: %s | 分类: %s" % (x["slug"], x["name_zh"], x["category"]))
        for fm in x.get("formulas", []):
            lines.append("- text: %s" % fm.get("text", ""))
            lines.append("  latex: %s" % fm.get("latex", ""))
        desc = (x.get("description") or "").strip()
        lines.append("- 描述: %s" % desc[:200])
        lines.append("")
    return "\n".join(lines)


def robust_json_parse(text: str):
    """提取 JSON 数组, 容错处理 markdown 包裹/截断。"""
    t = text.strip()
    # 去掉 markdown 代码块
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    # 去掉前导说明文字, 找第一个 [
    idx = t.find("[")
    if idx > 0:
        t = t[idx:]
    # 尝试直接解析
    for attempt in range(3):
        try:
            return json.loads(t)
        except json.JSONDecodeError as e:
            if attempt == 0:
                # 尝试补全结尾: 截断处回退到最后一个完整对象
                t = t[: e.pos].rstrip(", \n") + "]"
            elif attempt == 1:
                # 逐行修复: 去掉所有换行重试
                t = re.sub(r"\n\s*", " ", t)
            else:
                return None
    return None


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="0=全部")
    ap.add_argument("--dry", action="store_true", help="只统计不调用 API")
    ap.add_argument("--cats", type=str, default="qty",
                    help="qty=量价类(tech/emotion/volatility/momentum/liquidity), "
                         "fund=基本面类(basic-surface/growth/quality/value/scale), "
                         "all=全部")
    args = ap.parse_args()

    fd = json.load(open(FD_PATH, encoding="utf-8"))
    cat_set = {"all": QTY_CATS | FUND_CATS, "qty": QTY_CATS, "fund": FUND_CATS}[args.cats]
    prompt = FUND_SYSTEM_PROMPT if args.cats == "fund" else SYSTEM_PROMPT
    factors = [x for x in fd["factors"] if x["category"] in cat_set]
    if args.limit:
        factors = factors[args.start: args.start + args.limit]
    else:
        factors = factors[args.start:]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUT_PATH.exists():
        done = json.load(open(OUT_PATH, encoding="utf-8"))
    else:
        done = {}

    todo = [x for x in factors if x["slug"] not in done]
    print(f"总量价因子: {len(factors)} | 已翻译: {len(done)} | 待翻译: {len(todo)}")
    if args.dry:
        print("dry run, 退出")
        return

    client = DeepSeekClient()
    n_batch = 0
    n_ok = 0
    n_skip = 0
    for i in range(0, len(todo), args.batch_size):
        batch = todo[i: i + args.batch_size]
        user_prompt = build_user_prompt(batch)
        try:
            resp = client.chat(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=6000,
            )
        except Exception as e:
            print(f"[批{i // args.batch_size}] API 调用失败: {e}, 3s 后继续")
            time.sleep(3)
            continue

        parsed = robust_json_parse(resp)
        if parsed is None:
            print(f"[批{i // args.batch_size}] JSON 解析失败, 原始响应前 200 字符: {resp[:200]}")
            # 落盘原始响应供人工修复
            raw_path = OUT_PATH.parent / f"raw_batch_{i // args.batch_size}.txt"
            raw_path.write_text(resp, encoding="utf-8")
            continue

        for item in parsed:
            slug = item.get("slug")
            if not slug:
                continue
            fp = item.get("formula_pandas")
            done[slug] = {
                "formula_pandas": fp,
                "direction": item.get("direction", 1),
                "rationale": item.get("rationale", ""),
                "skip_reason": item.get("skip_reason"),
                "name_zh": next((x["name_zh"] for x in batch if x["slug"] == slug), ""),
                "category": next((x["category"] for x in batch if x["slug"] == slug), ""),
                "translated_at": datetime.now().isoformat(),
            }
            if fp:
                n_ok += 1
            else:
                n_skip += 1
        n_batch += 1
        # 每批落盘
        json.dump(done, open(OUT_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"[批{n_batch}] 完成 {len(parsed)} 条 | 累计 ok={n_ok} skip={n_skip} | {len(done)}/{len(factors)}")

    print(f"完成: 翻译 {len(done)}/{len(factors)} | formula 有效 {n_ok} | 跳过 {n_skip}")
    print(f"落盘: {OUT_PATH}")


if __name__ == "__main__":
    main()
