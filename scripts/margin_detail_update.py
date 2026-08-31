# -*- coding: utf-8 -*-
"""
两融明细补采脚本 (P-20260814-001 落地)
=====================================
margin_detail.csv 存量数据最新 2025-12-31, 补采 2026-01-01 → 最近交易日。

Tushare margin_detail 接口: 按 trade_date 逐日拉取全市场两融明细
字段: trade_date/ts_code/rzye(融资余额)/rqye(融券余额)/rzmre(融资买入额)/rqyl(融券余量)/rqchl(融券偿还量)

用法:
  python scripts/margin_detail_update.py [--start 20260101] [--max-days 200]
"""
import sys
import os
import time
import argparse
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd

RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
MARGIN_CSV = os.path.join(RAW, "margin_detail.csv")

FIELDS = ["trade_date", "ts_code", "rzye", "rqye", "rzmre", "rqyl", "rqchl"]


def get_pro():
    try:
        from credentials import get_tushare_api
        return get_tushare_api()
    except Exception:
        import tushare as ts
        import credentials
        token = getattr(credentials, "TS_TOKEN", getattr(credentials, "TOKEN", ""))
        if not token:
            raise RuntimeError("credentials.py 无可用 token")
        return ts.pro_api(token)


def load_existing():
    if not os.path.exists(MARGIN_CSV):
        return pd.DataFrame(columns=FIELDS)
    df = pd.read_csv(MARGIN_CSV, dtype={"trade_date": str, "ts_code": str})
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20260101", help="补采起始日 YYYYMMDD")
    ap.add_argument("--max-days", type=int, default=200)
    ap.add_argument("--sleep", type=float, default=0.35, help="每两次调用间隔秒")
    args = ap.parse_args()

    pro = get_pro()
    existing = load_existing()
    print(f"存量: {len(existing)} 行, 最新日期: {existing['trade_date'].max() if len(existing) else 'N/A'}")

    # 交易日历
    start_iso = "%s-%s-%s" % (args.start[:4], args.start[4:6], args.start[6:8])
    end_iso = datetime.now().strftime("%Y-%m-%d")
    cal = pro.trade_cal(exchange="SSE", start_date=args.start, end_date=end_iso.replace("-", ""))
    trade_days = sorted(
        cal[cal["is_open"] == 1]["cal_date"].tolist()
    )
    # 只补存量中缺失的日期
    have_dates = set(existing["trade_date"].unique()) if len(existing) else set()
    todo = [d for d in trade_days if d not in have_dates]
    todo = todo[:args.max_days]
    print(f"需补采: {len(todo)} 个交易日 ({todo[0] if todo else '-'} → {todo[-1] if todo else '-'})")

    if not todo:
        print("无缺失日期, 已是最新")
        return

    new_rows = []
    fail_days = []
    for i, d in enumerate(todo):
        try:
            df = pro.margin_detail(trade_date=d)
        except Exception as e:
            msg = str(e)[:60]
            # 触发频率限制: 等待 61 秒后重试一次
            if "每分钟" in msg or "frequency" in msg.lower() or "limit" in msg.lower():
                print(f"  [{d}] 频率受限, 等待 62s 重试...")
                time.sleep(62)
                try:
                    df = pro.margin_detail(trade_date=d)
                except Exception as e2:
                    fail_days.append((d, str(e2)[:60]))
                    print(f"  [{d}] 重试失败: {str(e2)[:60]}")
                    continue
            else:
                fail_days.append((d, msg))
                print(f"  [{d}] 失败: {msg}")
                continue
        if df is not None and len(df) > 0:
            df = df[[c for c in FIELDS if c in df.columns]]
            new_rows.append(df)
        else:
            fail_days.append((d, "空数据"))
        if (i + 1) % 30 == 0:
            print(f"  进度 {i+1}/{len(todo)}, 累计 {sum(len(r) for r in new_rows)} 行")
        time.sleep(args.sleep)

    if new_rows:
        added = pd.concat(new_rows, ignore_index=True)
        added = added.drop_duplicates(subset=["trade_date", "ts_code"])
        merged = pd.concat([existing, added], ignore_index=True)
        merged = merged.drop_duplicates(subset=["trade_date", "ts_code"])
        merged = merged.sort_values(["trade_date", "ts_code"])
        merged.to_csv(MARGIN_CSV, index=False)
        print(f"完成: 新增 {len(added)} 行, 总计 {len(merged)} 行, 最新 {merged['trade_date'].max()}")
    else:
        print("无新数据写入")
    if fail_days:
        print(f"失败 {len(fail_days)} 天: {fail_days[:5]}{'...' if len(fail_days) > 5 else ''}")


if __name__ == "__main__":
    main()
