#!/usr/bin/env python
"""
LLM 提案 → K-fold OOS 筛选 → JQ 策略生成  (全链路, 复用本地CSV+因子缓存)
"""
import sys, os, gc, json, time, warnings, pickle
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.loader import load_all_data
from llm_factor_pairs import get_all_proposals
from kfold_oos import evaluate_all_llm_pairs_kfold

# ── 配置 ──
N_FOLDS = 5
MIN_ABS_ICIR = 0.30
STABILITY_GRADES = {"A", "B"}
TOP_N_PAIRS = 6
JQ_STOCK_NUM = 30
PRICE_WINDOW = 130

BANNED = {"vm_diff", "bargaining_power_proxy", "relative_spread_proxy",
          "avg_turnover_3m", "min_ret_1m", "atr_14"}

OUTPUT_DIR = ROOT / "output"
OUTPUT_META = OUTPUT_DIR / f"llm_pairs_kfold_result_{time.strftime('%Y%m%d_%H%M%S')}.json"


def main():
    print("=" * 70)
    print("LLM 提案 → K-fold OOS 筛选 → JQ 生成")
    print("=" * 70)
    
    # ── 1. 价格数据 (本地 CSV, 无 API 调用) ──
    t0 = time.time()
    print(f"\n[1/5] 加载价格数据 (本地 CSV)...")
    data = load_all_data('2021-01-01', '2026-06-25')
    close = data['price_data']['close']
    close_weekly = close.resample('W').last()
    close_weekly.index = pd.DatetimeIndex(close_weekly.index)
    forward_returns = close_weekly.shift(-1) / close_weekly - 1
    print(f"  forward_returns: {forward_returns.shape} "
          f"({forward_returns.index[0].strftime('%Y-%m-%d')} ~ {forward_returns.index[-1].strftime('%Y-%m-%d')}) "
          f"({time.time()-t0:.0f}s)")
    
    # ── 2. 因子缓存 ──
    cache = ROOT / "output" / "factor_cache.pkl"
    with open(cache, "rb") as f:
        factor_dfs = pickle.load(f)
    for bf in BANNED:
        factor_dfs.pop(bf, None)
    print(f"\n[2/5] 因子缓存: {len(factor_dfs)} 个 (已排禁投)")
    
    # 对齐因子日期到 forward_returns
    fr_dates = set(forward_returns.index)
    for name in list(factor_dfs.keys()):
        df = factor_dfs[name]
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        keep_dates = [d for d in df.index if d in fr_dates]
        if len(keep_dates) < 50:
            del factor_dfs[name]
        else:
            factor_dfs[name] = df.loc[keep_dates]
    print(f"  对齐后: {len(factor_dfs)} 个")
    
    # ── 3. LLM 提案 ──
    proposals = get_all_proposals()
    print(f"\n[3/5] LLM 提案: {len(proposals)} 个")
    
    # ── 4. K-fold OOS ──
    results = evaluate_all_llm_pairs_kfold(
        proposals, factor_dfs, forward_returns,
        n_folds=N_FOLDS, verbose=True,
    )
    
    # ── 5. 筛选 + 生成 ──
    print(f"\n[4/5] 筛选 (|ICIR|>={MIN_ABS_ICIR} & grade∈{STABILITY_GRADES})")
    
    filtered = [r for r in results
                if r.get("stability_grade") in STABILITY_GRADES
                and abs(r.get("avg_icir", 0)) >= MIN_ABS_ICIR]
    filtered.sort(key=lambda x: abs(x.get("avg_icir", 0)), reverse=True)
    top = filtered[:TOP_N_PAIRS]
    
    if not top:
        # 宽松重试
        print("  [WARN] 无配对通过! 尝试放宽...")
        relaxed = [r for r in results
                   if r.get("stability_grade") in {"A", "B", "C"}
                   and abs(r.get("avg_icir", 0)) >= 0.15]
        relaxed.sort(key=lambda x: abs(x.get("avg_icir", 0)), reverse=True)
        top = relaxed[:TOP_N_PAIRS]
    
    if not top:
        print("  [FAIL] 仍无配对通过, 退出")
        return
    
    print(f"\n  {len(top)} 个配对入选:")
    print(f"  {'配对':38s} {'avg_ICIR':>9s} {'cv_ICIR':>9s} {'Grade':>7s} {'类型':>10s}")
    print(f"  {'-'*78}")
    for p in top:
        print(f"  {p['pair']:38s} {p['avg_icir']:>+9.4f} {p['cv_icir']:>9.4f} "
              f"{p['stability_grade']:>7s} {p.get('pairing_type','?'):>10s}")
    
    # ── 保存元数据 ──
    output_meta = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_folds": N_FOLDS, "min_abs_icir": MIN_ABS_ICIR,
        "stability_grades": list(STABILITY_GRADES),
        "total_evaluated": len(results), "selected": len(top),
        "selected_pairs": [],
    }
    for p in top:
        pair_str = p["pair"]
        parts = pair_str.split(" × ")
        fa = parts[0]
        fb = parts[1] if len(parts) > 1 else "__STANDALONE__"
        output_meta["selected_pairs"].append({
            "pair": pair_str, "factor_a": fa, "factor_b": fb,
            "avg_icir": float(p["avg_icir"]), "cv_icir": float(p["cv_icir"]),
            "grade": p["stability_grade"], "type": p.get("pairing_type",""),
            "proposal_id": p.get("proposal_id",""), "rationale": p.get("rationale",""),
        })
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_META, "w", encoding="utf-8") as f:
        json.dump(output_meta, f, indent=2, ensure_ascii=False)
    print(f"\n  元数据: {OUTPUT_META}")
    
    # ── 生成 JQ ──
    jq_path = generate_jq(top, OUTPUT_DIR)
    print(f"  JQ 策略: {jq_path}")
    
    # 语法验证
    with open(jq_path, encoding="utf-8") as f:
        compile(f.read(), str(jq_path), 'exec')
    print("  语法: OK")
    
    print(f"\n{'='*70}")
    print(f"完成! ({time.time()-t0:.0f}s total)")
    print(f"{'='*70}")
    return jq_path, top


# ======================================================================
# JQ 策略生成器
# ======================================================================

def _factor_jq_impl(bf):
    """为单个因子生成 JQ compute 函数 (真实实现, 非通用z-score)."""
    # ═══ volume_climax_reversal: 天量反转 ═══
    if bf == "volume_climax_reversal":
        return '''
def compute_volume_climax_reversal(stocks, price_data):
    """天量反转: -(vol_lag1/vol_ma20_lag1 - 1).clip(lower=0)"""
    volume = price_data.get("volume", {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        v = volume.get(s)
        if v is None or len(v) < 22:
            continue
        v_arr = np.array(v[-22:], dtype=float)
        v_ma = np.mean(v_arr)
        if v_ma < 1:
            continue
        v_lag = v_arr[-2]  # vol_shift(1)
        ratio = v_lag / v_ma - 1.0
        arr[i] = -max(ratio, 0)
        valid[i] = True
    return arr, valid
'''
    # ═══ earnings_volume_drift: 缩量上涨 ═══
    if bf == "earnings_volume_drift":
        return '''
def compute_earnings_volume_drift(stocks, price_data):
    """缩量上涨: clip(ret_5d, [-0.3,0.3]) * clip(-vol_chg, [-1,2]), ret_5d>0 only"""
    close = price_data.get("close", {})
    volume = price_data.get("volume", {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        c = close.get(s); v = volume.get(s)
        if c is None or v is None or len(c) < 12 or len(v) < 12:
            continue
        c_arr = np.array(c[-12:], dtype=float)
        v_arr = np.array(v[-12:], dtype=float)
        ret_5d = c_arr[-1] / max(c_arr[-6], 1e-8) - 1.0
        if ret_5d <= 0:
            arr[i] = 0; valid[i] = True; continue
        m5 = np.mean(v_arr[-5:])
        m5p = np.mean(v_arr[-11:-6]) if len(v_arr) >= 11 else np.mean(v_arr[-10:-5])
        if m5p < 1:
            continue
        vol_chg = m5 / m5p - 1.0
        signal = np.clip(ret_5d, -0.3, 0.3) * np.clip(-vol_chg, -1.0, 2.0)
        arr[i] = signal; valid[i] = True
    return arr, valid
'''
    # ═══ opening_gap_momentum: 开盘跳空动量 ═══
    if bf == "opening_gap_momentum":
        return '''
def compute_opening_gap_momentum(stocks, price_data):
    """开盘跳空动量: (open/close.shift(1)-1).rolling(3).mean()"""
    close = price_data.get("close", {})
    open_p = price_data.get("open", {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        c = close.get(s); o = open_p.get(s)
        if c is None or o is None or len(c) < 5 or len(o) < 5:
            continue
        c_arr = np.array(c[-5:], dtype=float)
        o_arr = np.array(o[-5:], dtype=float)
        gaps = o_arr[1:] / c_arr[:-1] - 1.0  # 最近4个跳空
        if len(gaps) < 3:
            continue
        arr[i] = np.mean(gaps[-3:]); valid[i] = True
    return arr, valid
'''
    # ═══ tvma_20: 20日成交额均线 ═══
    if bf == "tvma_20":
        return '''
def compute_tvma_20(stocks, price_data):
    """20日成交额均线: MA20(volume*close)"""
    close = price_data.get("close", {})
    volume = price_data.get("volume", {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        c = close.get(s); v = volume.get(s)
        if c is None or v is None or len(c) < 20 or len(v) < 20:
            continue
        c_arr = np.array(c[-20:], dtype=float)
        v_arr = np.array(v[-20:], dtype=float)
        amount = v_arr * c_arr
        arr[i] = np.mean(amount); valid[i] = True
    return arr, valid
'''
    # ═══ 默认: 通用 z-score ═══
    return f'''
def compute_{bf}(stocks, price_data):
    """{bf} — rank-pct 标准化"""
    close = price_data.get("close", {{}})
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        c = close.get(s)
        if c is None or len(c) < 10:
            continue
        ret = np.diff(np.log(np.maximum(np.array(c, dtype=float), 1e-8)))
        if len(ret) < 5:
            continue
        mu, sigma = np.mean(ret[-52:]), np.std(ret[-52:])
        if sigma < 1e-10:
            continue
        arr[i] = (ret[-1] - mu) / sigma
        valid[i] = True
    return arr, valid
'''


def _pair_jq_impl(pair_name, fa, fb):
    """为配对复合生成 JQ compute_pair 函数 (纯ASCII)."""
    return f'''
def compute_pair_{pair_name}(stocks, price_data):
    """{fa} x {fb} -> rank-product"""
    a, va = compute_{fa}(stocks, price_data)
    b, vb = compute_{fb}(stocks, price_data)
    n = len(stocks)
    ra = np.full(n, 0.5); rb = np.full(n, 0.5)
    if va.any():
        order_a = np.argsort(np.argsort(a[va]))
        ra[va] = order_a / (va.sum() - 1) if va.sum() > 1 else ra[va]
    if vb.any():
        order_b = np.argsort(np.argsort(b[vb]))
        rb[vb] = order_b / (vb.sum() - 1) if vb.sum() > 1 else rb[vb]
    arr = ra * rb
    valid = np.isfinite(arr) & (arr > 0)
    return arr, valid
'''


def generate_jq(top_pairs, output_dir):
    ts = time.strftime('%Y%m%d_%H%M%S')
    jq_path = output_dir / f"fa_llm_pairs_jq_{ts}.py"
    n = len(top_pairs)
    w = round(1.0 / n, 4)
    
    # 解析配对字符串 → factor_a, factor_b, is_standalone
    parsed = []
    for p in top_pairs:
        pair_str = p["pair"]
        parts = pair_str.split(" × ")
        fa = parts[0]
        fb = parts[1] if len(parts) > 1 else "__STANDALONE__"
        is_standalone = (fb == "__STANDALONE__")
        parsed.append({"fa": fa, "fb": fb, "standalone": is_standalone, "pair_str": pair_str})
    
    # 收集所有 base 因子
    base = sorted(set(
        [pp["fa"] for pp in parsed] + 
        [pp["fb"] for pp in parsed if pp["fb"] != "__STANDALONE__"]
    ))
    
    # 配对名 (用于 JQ 函数名)
    pair_names = []
    for pp in parsed:
        pn = f"{pp['fa']}__{pp['fb']}" if not pp["standalone"] else pp["fa"]
        pair_names.append(pn)
    
    src = f'''# -*- coding: utf-8 -*-
# FA LLM Pairs -- LLM 提案 + K-fold OOS 筛选
# {n} 个配对复合, 等权 rank-product Ensemble
# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
#
# 入选配对 (K-fold avg_ICIR, cv, grade):
'''
    for p, pn in zip(top_pairs, pair_names):
        src += f"#   {pn:40s} {p['avg_icir']:>+8.4f}  cv={p['cv_icir']:.4f}  {p['stability_grade']}  ({p.get('proposal_id','')})\n"
    
    src += f'''
# 因子: {', '.join(base)}

import datetime
import numpy as np
import pandas as pd
import gc

BENCHMARK = '000300.XSHG'
TOP_N = {JQ_STOCK_NUM}
REBALANCE_DAY = 5
PRICE_WINDOW = {PRICE_WINDOW}
PRICE_FIELDS = ['close', 'open', 'volume']
CHUNK = 500

# ── 因子实现 ──
'''
    
    for bf in base:
        src += _factor_jq_impl(bf)
    
    src += '\n# ── 配对复合 & 单因子 ──\n'
    for pp, pn in zip(parsed, pair_names):
        if pp["standalone"]:
            # 单因子: 直接 rank-pct
            src += f'''
def compute_score_{pn}(stocks, price_data):
    """{pp['fa']} -- standalone rank-pct"""
    a, va = compute_{pp['fa']}(stocks, price_data)
    n = len(stocks)
    ra = np.full(n, 0.5)
    if va.any():
        order = np.argsort(np.argsort(a[va]))
        ra[va] = order / (va.sum() - 1) if va.sum() > 1 else ra[va]
    arr = ra
    valid = np.isfinite(arr) & (arr > 0)
    return arr, valid
'''
        else:
            src += _pair_jq_impl(pn, pp["fa"], pp["fb"])
    
    src += f'''
# ── Ensemble ──

def compute_ensemble(stocks, price_data):
    n_stocks = len(stocks)
    composite = np.ones(n_stocks)
    score_funcs = [
'''
    for pn in pair_names:
        src += f"        (compute_score_{pn},),\n"
    
    src += f'''    ]
    for (fn,) in score_funcs:
        arr, valid = fn(stocks, price_data)
        if not valid.any():
            continue
        arr = np.where(valid, arr, np.nan)
        arr = np.nan_to_num(arr, nan=0.5)
        composite *= np.power(arr + 1e-8, {w})
    return composite


# ── 数据加载 (V2模式) ──

def load_price_data(stocks, end_dt, fields, window):
    result = {{f: {{}} for f in fields}}
    PanelClass = getattr(pd, 'Panel', None)
    for i in range(0, len(stocks), CHUNK):
        batch = stocks[i:i+CHUNK]
        for field in fields:
            try:
                px = get_price(batch, end_date=end_dt, count=window,
                               fields=field, frequency='daily', fq='pre',
                               skip_paused=False)
                if PanelClass is not None and isinstance(px, PanelClass):
                    df = px[field] if field in px.items else px.minor_xs(field)
                else:
                    df = px
                if isinstance(df, pd.DataFrame):
                    result[field].update({{c: df[c].values for c in df.columns if c in batch}})
            except Exception:
                continue
        del batch; gc.collect()
    return result


# ── 初始化和调仓 ──

def initialize(context):
    set_benchmark(BENCHMARK)
    set_option('use_real_price', True)
    set_option('avoid_future_data', True)
    log.set_level('order', 'info')
    log.set_level("system", "error")  # P-20260902-016: 抑制 order 函数内部仓位查询的 WARNING ("在 positions 中不存在" 刷屏)
    set_order_cost(OrderCost(open_tax=0, close_tax=0.001,
        open_commission=0.0001, close_commission=0.0001,
        close_today_commission=0, min_commission=0), type='stock')
    set_slippage(FixedSlippage(0.003))
    g.cycle = 0
    log.info("[Init] FA LLM Pairs -- {n} pairs, K-fold OOS screened")
    run_weekly(rebalance, weekday=REBALANCE_DAY, time='14:50')


def rebalance(context):
    g.cycle += 1
    prev_date = context.previous_date

    all_stocks = list(get_all_securities(['stock'], prev_date).index)
    log.info(f"[W{{g.cycle}}] Step1: {{len(all_stocks)}} stocks")

    cd = get_current_data()
    universe = []
    npaused = nnew = 0
    for s in all_stocks:
        try:
            if cd[s].paused:
                npaused += 1; continue
            if 'ST' in get_security_info(s).display_name:
                continue
            si = get_security_info(s)
            if si and si.start_date:
                if (prev_date - si.start_date).days < 252:
                    nnew += 1; continue
            universe.append(s)
        except Exception:
            pass
    log.info(f"[W{{g.cycle}}] Step2: {{len(universe)}} kept (p:{{npaused}} n:{{nnew}})")

    if len(universe) < TOP_N:
        log.warn(f"[W{{g.cycle}}] < TOP_N SKIP"); return

    price_data = load_price_data(universe, prev_date, PRICE_FIELDS, PRICE_WINDOW)
    nclose = len(price_data.get('close', {{}})); nvol = len(price_data.get('volume', {{}}))
    log.info(f"[W{{g.cycle}}] Step3: close={{nclose}} vol={{nvol}}")
    if nclose == 0:
        log.warn(f"[W{{g.cycle}}] ZERO data SKIP"); return

    composite = compute_ensemble(universe, price_data)
    valid = np.isfinite(composite)
    nvalid = int(np.sum(valid))
    if nvalid < TOP_N:
        log.warn(f"[W{{g.cycle}}] valid={{nvalid}} < TOP_N SKIP"); return

    scores = pd.Series(composite, index=universe)
    scores = scores[valid].dropna()
    top = scores.nlargest(TOP_N).index.tolist()

    cur = list(context.portfolio.positions.keys())
    wgt = 1.0 / TOP_N
    for s in cur:
        if s not in top:
            try: order_target_value(s, 0)
            except Exception: pass
    for s in top:
        try:
            if s in cd and cd[s].high_limit <= cd[s].last_price:
                continue
            order_target_value(s, context.portfolio.total_value * wgt)
        except Exception:
            pass

    if g.cycle <= 4 or g.cycle % 13 == 0:
        log.info(f"[W{{g.cycle}}] Hold {{len(top)}} | Val {{context.portfolio.total_value:.0f}}")
'''
    
    os.makedirs(os.path.dirname(jq_path), exist_ok=True)
    with open(jq_path, "w", encoding="utf-8") as f:
        f.write(src)
    
    return str(jq_path)


if __name__ == "__main__":
    main()
