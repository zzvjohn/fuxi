"""验证 walkforward_stability._run_windows_parallel 的候选子集路径不卡死。
模拟 Phase 7 Anchored: 父进程全局设 50 因子, specs 候选只用 15 个,
n_workers=4。修复后 initargs 只传 15 因子子集 (而非 50 全量)。

★ 必须用 if __name__=='__main__' 保护: Windows spawn 子进程会重新
import 本脚本, 若无保护会递归重跑整个脚本 → 无限 spawn 卡死。
"""
import sys, time

def main():
    sys.path.insert(0, 'e:/quant/research/factor_alchemy')
    print(f"[{time.strftime('%H:%M:%S')}] importing numpy/pandas...", flush=True)
    import numpy as np, pandas as pd
    print(f"[{time.strftime('%H:%M:%S')}] importing walkforward_stability...", flush=True)
    from evaluation import walkforward_stability as wf
    print(f"[{time.strftime('%H:%M:%S')}] imported. building data...", flush=True)

    rng = np.random.default_rng(42)
    n_weeks, n_stocks = 150, 400
    dates = pd.date_range('2021-01-01', periods=n_weeks, freq='W')
    stocks = [f'S{i:04d}' for i in range(n_stocks)]

    # 50 个因子 (模拟全量因子空间)
    all_factors = {}
    for j in range(50):
        base = rng.normal(0, 1, size=(n_weeks, n_stocks))
        all_factors[f'factor_{j}'] = pd.DataFrame(base, index=dates, columns=stocks)

    # forward_returns
    fr = pd.DataFrame(rng.normal(0, 0.02, size=(n_weeks, n_stocks)), index=dates, columns=stocks)

    # ic_summary (ICIR)
    ic_summary = pd.DataFrame(
        {'ICIR': [abs(rng.normal(0.3, 0.1)) for _ in range(50)]},
        index=[f'factor_{j}' for j in range(50)],
    )

    # specs: 候选只用 15 个因子 (模拟 anchored 的 top10-15)
    candidate_names = [f'factor_{j}' for j in range(15)]
    specs = []
    for i in range(8):  # 8 个窗口
        te = min((i + 1) * 15 + 10, n_weeks)
        specs.append({
            'i': i,
            'train_dates': dates[:te - 5],
            'test_dates': dates[te - 5:te],
            'candidate_names': candidate_names,
            'n_select': 5,
        })

    # 父进程侧设置全局 (模拟 anchored_walkforward line 240-242)
    wf._WF_FACTOR_DFS = all_factors
    wf._WF_FORWARD_RETURNS = fr
    wf._WF_IC_SUMMARY = ic_summary

    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] calling _run_windows_parallel (n_workers=4)...", flush=True)
    res = wf._run_windows_parallel(specs, wf._anchored_window_task, n_workers=4)
    dt = time.time() - t0
    print(f"[{time.strftime('%H:%M:%S')}] 并行完成: {len(res)} 窗口, 耗时 {dt:.1f}s")
    print("样例 Win0:", {k: res[0][k] for k in ['window', 'icir_test', 'factors']})
    assert len(res) == 8
    assert all(abs(r['icir_test']) <= 1.0 for r in res)
    print("✅ 候选子集路径验证通过 (无 hang, 结果合理)")


if __name__ == '__main__':
    main()
