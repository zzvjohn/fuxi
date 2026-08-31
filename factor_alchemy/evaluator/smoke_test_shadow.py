# -*- coding: utf-8 -*-
"""
smoke_test_shadow.py — 本地烟雾测试 shadow_eval_v4.py 的因子函数与袖子逻辑
用合成随机游走价格数据验证:
  1. 每个 compute_xxx 能跑通, 返回 (arr, valid) 形状正确
  2. valid 覆盖率合理 (>50%)
  3. 因子值有截面区分度 (std > 0)
  4. 袖子 rank 乘积 + top20 选择逻辑正常
  5. 不同袖子的选股不完全相同 (信号多样性)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

SHADOW_FILE = Path(__file__).resolve().parents[1] / "output" / "shadow_eval_v4.py"


def make_synthetic_data(n_stocks=200, n_days=130, seed=42):
    """生成合成价格数据: 几何随机游走 + 个别股票注入趋势/跳水"""
    rng = np.random.RandomState(seed)
    stocks = [f"{600000+i}.XSHG" for i in range(n_stocks)]
    price_data = {f: {} for f in ["close", "open", "high", "low", "volume"]}
    for i, s in enumerate(stocks):
        drift = rng.normal(0.0002, 0.0005)
        vol = abs(rng.normal(0.02, 0.005))
        rets = rng.normal(drift, vol, n_days)
        if i % 17 == 0:   # 注入趋势股
            rets += 0.002
        if i % 23 == 0:   # 注入跳水股
            rets[-10:] -= 0.03
        close = 20 * np.exp(np.cumsum(rets))
        open_ = close * (1 + rng.normal(0, 0.003, n_days))
        high = np.maximum(close, open_) * (1 + abs(rng.normal(0, 0.008, n_days)))
        low = np.minimum(close, open_) * (1 - abs(rng.normal(0, 0.008, n_days)))
        volume = abs(rng.normal(5e6, 2e6, n_days)) * (1 + 20 * np.abs(rets))
        price_data["close"][s] = close
        price_data["open"][s] = open_
        price_data["high"][s] = high
        price_data["low"][s] = low
        price_data["volume"][s] = volume
    return stocks, price_data


def main():
    # 动态加载生成的影子文件 (只取函数, 不执行 JQ 部分)
    import importlib.util
    spec = importlib.util.spec_from_file_location("shadow_eval", SHADOW_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    stocks, price_data = make_synthetic_data()
    print(f"合成数据: {len(stocks)} 股 × 130 天\n")

    # 1. 逐因子测试
    print(f"{'因子':<38} {'valid%':>7} {'std':>10} {'min':>10} {'max':>10}")
    print("-" * 80)
    factor_out = {}
    n_fail = 0
    for name, func in mod.FACTOR_FUNCS.items():
        try:
            arr, valid = func(stocks, price_data)
            assert len(arr) == len(stocks) and len(valid) == len(stocks)
            cov = valid.mean()
            std = np.nanstd(arr[valid]) if valid.any() else 0.0
            vmin = np.nanmin(arr[valid]) if valid.any() else np.nan
            vmax = np.nanmax(arr[valid]) if valid.any() else np.nan
            flag = "✓" if cov > 0.5 and std > 1e-12 else "⚠️"
            print(f"{flag} {name:<36} {cov:7.1%} {std:10.4f} {vmin:10.4f} {vmax:10.4f}")
            factor_out[name] = (arr, valid)
            if cov <= 0.5 or std <= 1e-12:
                n_fail += 1
        except Exception as e:
            print(f"✗ {name:<36} ERROR: {e}")
            n_fail += 1

    # 2. 袖子逻辑测试
    print(f"\n袖子选股测试 (top20):")
    print("-" * 80)
    selections = {}
    for sid, a_name, b_name in mod.SLEEVE_DEFS:
        a, va = factor_out[a_name]
        b, vb = factor_out[b_name]
        valid = va & vb
        ra = mod.rank_pct(a, valid)
        rb = mod.rank_pct(b, valid)
        score = np.where(valid, ra * rb, -1.0)
        top_idx = np.argsort(score)[-20:][::-1]
        hold = [stocks[i] for i in top_idx if score[i] > -1]
        selections[sid] = set(hold)
        print(f"  {sid:<28} 持仓={len(hold)} 首只={hold[0] if hold else 'N/A'}")

    # 3. 多样性: 袖子间持仓重叠
    king_avg_overlap = []
    sids = [s for s, _, _ in mod.SLEEVE_DEFS]
    inj_sids = [s for s in sids if s.startswith("inj_")]
    king_sids = [s for s in sids if s.startswith("king_")]
    for i_sid in inj_sids:
        overlaps = [len(selections[i_sid] & selections[k]) / 20.0 for k in king_sids]
        king_avg_overlap.append(np.mean(overlaps))
    print(f"\n注入袖子 vs 王者袖子 平均持仓重叠: {np.mean(king_avg_overlap):.1%}")
    print(f"(重叠<50% = 信号有独立性; 全部~100% = 因子等价, 有问题)")

    print(f"\n{'='*80}")
    print(f"烟雾测试: {len(mod.FACTOR_FUNCS) - n_fail}/{len(mod.FACTOR_FUNCS)} 因子通过")
    if n_fail > 0:
        print(f"⚠️ {n_fail} 个因子需检查")
    else:
        print("✅ 全部通过, 可以上传 JQ 运行")


if __name__ == "__main__":
    main()
