# -*- coding: utf-8 -*-
"""解析 JQ 回测日志 (FUXI_ACCOUNT_EXPORT + rebalance + ERROR) 产出结构化结果。

用法: python scripts/parse_jq_backtest_log.py <log_path> [--out out.json]
日志编码: GBK (聚宽回测引擎输出), 读取时 errors=replace。
"""
import sys
import re
import json
import html
import argparse
from collections import Counter

def _clean(s):
    """聚宽日志 JSON 被 HTML 实体转义 (&quot; → "), 反转义后再解析。"""
    return html.unescape(s)

def parse(path):
    with open(path, "r", encoding="gbk", errors="replace") as f:
        text = f.read()

    # FUXI_ACCOUNT_EXPORT {json} 行 (JSON 为 HTML 实体转义, 单行, 按行解析避免嵌套 {} 截断)
    nav_rows = []
    for line in text.splitlines():
        if "FUXI_ACCOUNT_EXPORT" not in line:
            continue
        idx = line.find("FUXI_ACCOUNT_EXPORT")
        payload = line[idx + len("FUXI_ACCOUNT_EXPORT"):].strip()
        try:
            d = json.loads(_clean(payload))
            nav_rows.append(d)
        except Exception:
            continue

    # rebalance 行
    rebalances = [l for l in text.splitlines() if "&quot;kind&quot;: &quot;rebalance&quot;" in l]

    # ERROR 行
    error_lines = re.findall(r'- ERROR - (.*?)$', text, flags=re.M)

    # 提取 nav 序列
    nav_seq = []
    for d in nav_rows:
        try:
            key = d.get("key") or d.get("session")
            nav = float(d.get("nav"))
            nav_seq.append((key, nav, d.get("n_hold", 0)))
        except Exception:
            continue

    # 计算收益/Sharpe/maxDD
    stats = {}
    if nav_seq:
        navs = [x[1] for x in nav_seq]
        init = navs[0]
        final = navs[-1]
        stats["init_nav"] = init
        stats["final_nav"] = final
        stats["total_return"] = final / init - 1
        stats["n_days"] = len(navs)
        stats["start"] = nav_seq[0][0]
        stats["end"] = nav_seq[-1][0]

        # 日收益
        rets = []
        for i in range(1, len(navs)):
            rets.append(navs[i] / navs[i-1] - 1)
        if rets:
            mean = sum(rets) / len(rets)
            var = sum((r - mean) ** 2 for r in rets) / len(rets)
            std = var ** 0.5
            stats["sharpe_annual"] = (mean / std * (252 ** 0.5)) if std > 0 else None
            stats["daily_mean"] = mean
            stats["daily_std"] = std

        # maxDD
        peak = navs[0]
        maxdd = 0
        for n in navs:
            if n > peak:
                peak = n
            dd = (peak - n) / peak
            if dd > maxdd:
                maxdd = dd
        stats["maxdd"] = maxdd

    # 错误分类
    err_counter = Counter()
    for line in error_lines:
        if "科创板" in line or "688" in line or "市价" in line:
            err_counter["科创板市价单需限价"] += 1
        elif "小于 100" in line or "小于100" in line:
            err_counter["买入数量小于100股"] += 1
        elif "资金" in line or "余额" in line:
            err_counter["资金不足"] += 1
        elif "停牌" in line or "暂停" in line:
            err_counter["停牌/不可交易"] += 1
        else:
            err_counter["其他"] += 1

    return {
        "stats": stats,
        "n_nav_rows": len(nav_seq),
        "n_rebalances": len(rebalances),
        "n_error_lines": len(error_lines),
        "error_breakdown": dict(err_counter),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_path")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    result = parse(args.log_path)
    txt = json.dumps(result, ensure_ascii=False, indent=2)
    print(txt)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(txt)
        print("\n[written]", args.out)

if __name__ == "__main__":
    main()
