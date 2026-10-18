"""
Stage 0 — fina_indicator 增量采集器
=====================================
背景: Tushare fina_indicator 接口强制要求 ts_code 逐股查询, 无法按全市场日期批量拉取。
      单股调用 ~0.11s, 返回 94 字段 (含 ROE/净利润增速/毛利率/负债率等质量因子核心)。

设计:
  - 逐股循环 (stock_basic 全量 ~5500 只)
  - 拉取近 WINDOW_YEARS 年窗口 (捕获重述 + 最新报告期)
  - 按 (ts_code, end_date) 去重追加, 幂等可重复运行
  - 非财报日运行 -> 零新增 (仅 API 调用成本)
  - 断点续传: 记录已处理股票, 失败可续
  - 频率控制: 0.1s 间隔 + 失败重试

用法:
  python -u scripts/stage0_fina_indicator.py            # 增量 (近2年窗口)
  python -u scripts/stage0_fina_indicator.py --full     # 全历史 (start_date=20000101)
  python -u scripts/stage0_fina_indicator.py --limit 50 # 仅前50只 (测试)
"""
import sys
sys.path.insert(0, 'e:/quant')
from credentials import get_tushare_api
import pandas as pd
import numpy as np
import time
import os
import json
import argparse
from datetime import datetime, timedelta

BASE = 'e:/quant/data/raw'
PATH = os.path.join(BASE, 'fina_indicator.csv')
STOCK_CACHE = os.path.join(BASE, '.fina_stocks.json')
PROGRESS_CACHE = os.path.join(BASE, '.fina_progress.json')
WINDOW_YEARS = 2  # 增量窗口: 近2年
SLEEP = 0.10      # 调用间隔
MAX_RETRY = 3

# 质量因子核心字段 (curated, 覆盖 ROE/增速/毛利率/净利率/杠杆)
# ⚠️ 注意: report_type / comp_type / ebit_yoy 已刻意排除 —— 实测 Tushare 在 fields 中含
#    这两个字段时会忽略过滤、直接返回全部 94 列默认 schema (严重 schema 错位风险)。
#    end_date 已能区分年报(1231)/季报(0331/0630/0930), 合并报表为本接口默认口径,
#    故无需 report_type/comp_type。下方 24 字段为 Tushare 稳定返回的子集。
FIELDS = [
    'ts_code', 'ann_date', 'end_date',
    'roe', 'roe_dt', 'roa', 'roic',
    'netprofit_yoy', 'dt_netprofit_yoy',          # 净利润增速
    'or_yoy', 'tr_yoy', 'op_yoy',                  # 营收/营业/利润增速
    'grossprofit_margin', 'netprofit_margin',      # 毛利率/净利率
    'debt_to_assets', 'assets_to_eqt',             # 资产负债率/权益乘数
    'ocf_yoy', 'q_roe', 'q_op_qoq',                # 经营现金流增速/单季ROE
    'bps_yoy', 'equity_yoy',                        # 每股净资产/权益增速
    'current_ratio', 'quick_ratio',                # 流动/速动比率
    'npta',                                        # 总资产净利润率
]

FLUSH = lambda: sys.stdout.flush()

def get_stock_list(pro):
    """获取全量股票列表 (带缓存)"""
    if os.path.exists(STOCK_CACHE):
        with open(STOCK_CACHE, 'r') as f:
            cached = json.load(f)
        # 缓存 < 1 天则复用
        if time.time() - cached.get('ts', 0) < 86400:
            return cached['codes']
    print("  拉取股票列表 (stock_basic)...", flush=True)
    codes = []
    for status in ['L', 'D', 'P']:  # 上市/退市/暂停
        try:
            df = pro.stock_basic(exchange='', list_status=status,
                                 fields='ts_code')
            if df is not None:
                codes.extend(df['ts_code'].tolist())
        except Exception as e:
            print(f"    ⚠️ stock_basic({status}) 失败: {str(e)[:80]}", flush=True)
        time.sleep(0.2)
    codes = sorted(set(codes))
    with open(STOCK_CACHE, 'w') as f:
        json.dump({'ts': time.time(), 'codes': codes}, f)
    print(f"  股票列表: {len(codes)} 只", flush=True)
    return codes

def load_progress():
    if os.path.exists(PROGRESS_CACHE):
        with open(PROGRESS_CACHE, 'r') as f:
            return json.load(f)
    return {'done': [], 'last_run': None}

def save_progress(prog):
    with open(PROGRESS_CACHE, 'w') as f:
        json.dump(prog, f)

def get_existing_keys(path):
    """读取已有 (ts_code, end_date) 集合用于去重"""
    if not os.path.exists(path) or os.path.getsize(path) < 50:
        return set()
    try:
        df = pd.read_csv(path, usecols=['ts_code', 'end_date'],
                         encoding='utf-8-sig', dtype=str)
        return set(zip(df['ts_code'], df['end_date']))
    except Exception:
        return set()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true', help='全历史')
    parser.add_argument('--limit', type=int, default=0, help='仅处理前N只(测试)')
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print(" fina_indicator 增量采集器", flush=True)
    print(f" 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f" 模式: {'全历史' if args.full else f'增量(近{WINDOW_YEARS}年)'}"
          + (f" | 限制{args.limit}只" if args.limit else ''), flush=True)
    print("=" * 70, flush=True)

    pro = get_tushare_api()
    time.sleep(0.3)

    # --full: 清空旧文件 + 进度, 洁净重建 (避免 schema 错配)
    if args.full and os.path.exists(PATH):
        os.remove(PATH)
        print(f"  [--full] 已清空旧文件: {PATH}", flush=True)
    if args.full and os.path.exists(PROGRESS_CACHE):
        os.remove(PROGRESS_CACHE)
        print(f"  [--full] 已重置进度缓存", flush=True)

    codes = get_stock_list(pro)
    if args.limit:
        codes = codes[:args.limit]

    # 现有数据键
    existing = get_existing_keys(PATH)
    print(f" 现有数据: {len(existing)} 条 (ts_code,end_date) 键", flush=True)

    start_date = '20000101' if args.full else \
        (datetime.now() - timedelta(days=365 * WINDOW_YEARS)).strftime('%Y%m%d')

    prog = load_progress()
    done = set(prog.get('done', []))
    # 仅处理未完成的 (续传)
    todo = [c for c in codes if c not in done]
    print(f" 待处理: {len(todo)} 只 (已完成 {len(done)})", flush=True)

    total_new = 0
    total_rows = 0
    failed = []
    t0 = time.time()

    for i, code in enumerate(todo):
        try:
            df = pro.fina_indicator(ts_code=code, start_date=start_date,
                                    fields=','.join(FIELDS))
            time.sleep(SLEEP)
        except Exception as e:
            err = str(e)
            if any(k in err for kw in ['频率', '限流', 'frequency', '超限']):
                print(f"\n  ⏳ 频率限制, 等待60s...", flush=True)
                time.sleep(60)
                try:
                    df = pro.fina_indicator(ts_code=code, start_date=start_date,
                                            fields=','.join(FIELDS))
                except Exception as e2:
                    failed.append(code)
                    print(f"    ❌ {code}: {str(e2)[:60]}", flush=True)
                    continue
            else:
                failed.append(code)
                if '必填' in err:
                    print(f"    ⚠️ {code}: 参数错误跳过", flush=True)
                continue

        if df is None or len(df) == 0:
            done.add(code)
            continue

        # 去重: 仅保留 (ts_code,end_date) 不在 existing 的
        df['ts_code'] = df['ts_code'].astype(str)
        df['end_date'] = df['end_date'].astype(str)
        mask = ~df.apply(lambda r: (r['ts_code'], r['end_date']) in existing, axis=1)
        new_rows = df[mask]
        if len(new_rows) > 0:
            write_header = (os.path.getsize(PATH) < 50) if os.path.exists(PATH) else True
            new_rows.to_csv(PATH, mode='a', index=False, header=write_header,
                            encoding='utf-8-sig')
            total_new += len(new_rows)
            # 更新 existing 键
            for _, r in new_rows.iterrows():
                existing.add((r['ts_code'], r['end_date']))
        done.add(code)

        # 进度显示
        if (i + 1) % 200 == 0 or (i + 1) == len(todo):
            el = time.time() - t0
            print(f"  [{i+1}/{len(todo)}] +{total_new} 新增 | "
                  f"耗时{el:.0f}s | 速度{(i+1)/max(el,1):.1f}只/s", flush=True)

    # 保存进度
    prog['done'] = sorted(done)
    prog['last_run'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    save_progress(prog)

    # 统计
    if os.path.exists(PATH):
        final = pd.read_csv(PATH, usecols=['ts_code', 'end_date'],
                            encoding='utf-8-sig', dtype=str)
        total_rows = len(final)
        max_ed = final['end_date'].max()
    else:
        total_rows = 0
        max_ed = 'N/A'

    print("\n" + "=" * 70, flush=True)
    print(f" fina_indicator 完成", flush=True)
    print(f"  本次新增: {total_new} 行", flush=True)
    print(f"  总计: {total_rows} 行 | 最新报告期: {max_ed}", flush=True)
    print(f"  失败: {len(failed)} 只", flush=True)
    if failed:
        print(f"  失败列表: {failed[:20]}", flush=True)
    print(f"  耗时: {time.time()-t0:.0f}s", flush=True)
    print("=" * 70, flush=True)

if __name__ == '__main__':
    main()
