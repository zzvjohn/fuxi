# -*- coding: utf-8 -*-
"""_verify_lion_factors.py — 狮子锚点因子族接入后验证 (fuxi-factor-onboarding §5)

判据: 日频通道 |IC| >= 0.02 且 日频 ICIR >= 0.10, 且 n_days >= 60。
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, 'e:/quant')
sys.path.insert(0, 'e:/quant/scripts')

import numpy as np
import daily_factor_hypothesis as dfh

print('=' * 74)
print('[1] load_data(n_days=270)')
close, open_, high, low, vol, extra = dfh.load_data(n_days=270)
print(f'  面板 {close.shape[0]} 交易日 x {close.shape[1]} 只')

print('')
print('[2] 通道字段注册与非空率 (last / mid / first_valid 中位位置)')
NAMES = ['lion_bias', 'lion_lu_days', 'lion_lu_cnt']
n = close.shape[0]
for k in NAMES:
    if k not in extra:
        print(f'  ❌ {k}: 未注册 (NameError 风险)')
        continue
    w = extra[k]
    last_r = int(w.iloc[-1].notna().sum())
    mid_r = int(w.iloc[n // 2].notna().sum())
    nn = w.notna()
    pos = [int(np.argmax(nn.values[:, j])) for j in range(nn.shape[1]) if nn.values[:, j].any()]
    med = float(np.median(pos)) if pos else float('nan')
    print(f'  ✅ {k:14s} last {last_r:5d} / mid {mid_r:5d} / {w.shape[1]} 只  '
          f'first_valid 中位 #{int(med)}/{n} ({med/n*100:.0f}% 处)')
    # 逐日 (前6 / 中位 / 末)
    daily = nn.sum(axis=1).values
    print(f'       逐日非空: 前6 {list(daily[:6])}  中位日 {daily[n//2]}  末日 {daily[-1]}')

print('')
print('[3] 公式预检 + 实算 IC')
fwd5 = close.shift(-5) / close - 1
lib = {f['name']: f for f in dfh.NOVEL_FACTOR_LIBRARY}
for name in NAMES:
    f = lib.get(name)
    if f is None:
        print(f'  ❌ {name}: NOVEL_FACTOR_LIBRARY 中不存在')
        continue
    print(f'  --- {name}  formula={f["formula"]!r}  direction={f["direction"]!r}')
    try:
        ok = dfh.validate_and_preprocess_formula(f['formula'], f['name'])
        print(f'      预检: {ok}')
    except Exception as e:
        print(f'      ❌ 预检异常: {type(e).__name__}: {e}')
        continue
    if isinstance(ok, tuple):          # (ok, formula) 或 (ok, msg, formula)
        okv = ok[0]
    else:
        okv = ok
    if not okv:
        print('      ❌ 预检未通过, 跳过实算')
        continue
    try:
        r = dfh.evaluate_factor(f, close, open_, high, low, vol, fwd5, extra=extra)
        if isinstance(r, dict):
            ic = r.get('ic', r.get('IC'))
            icir = r.get('icir', r.get('ICIR'))
            nd = r.get('n_days', r.get('n'))
            print(f'      实算: IC={ic}  ICIR={icir}  n_days={nd}')
            if ic is not None and icir is not None:
                try:
                    gi = abs(float(ic)); gc = abs(float(icir))
                    g = 'PASS' if (gi >= 0.02 and gc >= 0.10) else 'fail'
                    print(f'      S1 闸门: |IC|={gi:.4f} |ICIR|={gc:.4f} → {g}')
                except Exception:
                    pass
            extra_keys = {k: v for k, v in r.items() if k not in ('ic', 'icir', 'n_days', 'n')}
            if extra_keys:
                print(f'      其他: {extra_keys}')
        else:
            print(f'      实算返回: {r}')
    except Exception as e:
        import traceback
        print(f'      ❌ evaluate_factor 异常: {type(e).__name__}: {e}')
        traceback.print_exc()

print('')
print('[4] 是否能被 select_novel_hypotheses 选用 (blocked 检查)')
try:
    blocked = dfh.get_blocked_names()
    for name in NAMES:
        b = name in blocked
        print(f'  {name:14s} blocked={b}' + ('  ⚠️ 被阻塞, 不会进候选' if b else '  ✅ 可进候选'))
except Exception as e:
    print(f'  blocked 检查失败: {e}')

print('')
print('DONE')
