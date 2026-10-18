#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成 v7.11 校准原生三份聚宽代码:
  - fa_v711_large_dominant.py  (large_dominant 分支, SIGNED 权重)
  - fa_v711_small_bull.py      (small_bull 分支, SIGNED 权重)
  - fa_v711_merged.py          (regime-aware 纯切换 100/0)

权重方向: earnings_stability ICIR=-0.179 -> 校准代理内部翻转 -> 传负值权重.
其余因子 ICIR>0 -> 正权重. JQGenerator 用 composite += w*z, 负权重即翻转.
"""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jq_generator import JQGenerator

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       '..', 'strategies', 'drafts')
os.makedirs(OUT_DIR, exist_ok=True)

# ── SIGNED 权重 (按 ICIR 符号重建, earnings_stability 翻转) ──
LD_WEIGHTS = {
    'earnings_stability': -0.5134,   # ICIR=-0.179 -> 翻转
    'atr_14': 0.4253,
    'earnings_season_vol_div': 0.0613,
}
SB_WEIGHTS = {
    'relative_spread_proxy': 0.5673,
    'sp': 0.2661,
    'atr_14': 0.1374,
    'ret_6m': 0.0292,
}
# merged 生成用 union (保证所有6因子 compute 函数都在文件里)
UNION_WEIGHTS = dict(LD_WEIGHTS)
UNION_WEIGHTS.update(SB_WEIGHTS)

print("生成 large_dominant ...")
gen_ld = JQGenerator(LD_WEIGHTS, name='fa_v711_large_dominant', mcap_min=5,
                     stock_num=30, max_turnover=0.60)
code_ld = gen_ld.generate(os.path.join(OUT_DIR, 'fa_v711_large_dominant.py'))

print("生成 small_bull ...")
gen_sb = JQGenerator(SB_WEIGHTS, name='fa_v711_small_bull', mcap_min=5,
                     stock_num=30, max_turnover=0.60)
code_sb = gen_sb.generate(os.path.join(OUT_DIR, 'fa_v711_small_bull.py'))

print("生成 merged (regime-aware 纯切换) ...")
gen_merge = JQGenerator(UNION_WEIGHTS, name='fa_v711_merged', mcap_min=5,
                        stock_num=30, max_turnover=0.60)
code_merge = gen_merge.generate()  # 先拿字符串, 后处理

# ── 后处理: 单 WEIGHTS -> WEIGHTS_LD / WEIGHTS_SB + regime 切换 ──
# 1) 替换 WEIGHTS = {...} 块
def dict_block(name, w):
    lines = ",\n".join(f"    '{f}': {v:.4f}" for f, v in w.items())
    return f"{name} = {{\n{lines}\n}}"

weights_block = (
    dict_block('WEIGHTS_LD', LD_WEIGHTS) + "\n\n" +
    dict_block('WEIGHTS_SB', SB_WEIGHTS) + "\n\n"
    "REGIME_LOOKBACK_DAYS = 30   # 近30日(~6周)用于 regime 检测\n"
)
code_merge = re.sub(r"WEIGHTS = \{[^}]*\}\n", weights_block, code_merge, count=1)

# 2) initialize 中 g.weights = WEIGHTS -> 默认 LD (main 内切换)
code_merge = code_merge.replace(
    "g.weights        = WEIGHTS",
    "g.weights        = WEIGHTS_LD  # 实际权重在 main() 按 regime 切换")

# 3) 注入 detect_regime 函数 (在 def main 之前)
DETECT = '''# ==================== Regime 检测 ====================
def detect_regime(context):
    """CSI500-CSI300 风格价差 + 全A趋势 -> small_bull / large_dominant (纯切换)."""
    indices = {'CSI300': '000300.XSHG', 'CSI500': '000905.XSHG', 'CSI_ALL': '000985.XSHG'}
    rets = {}
    for name, code in indices.items():
        try:
            px = get_price(code, end_date=context.previous_date, count=REGIME_LOOKBACK_DAYS,
                           fields=['close'], skip_paused=False, frequency='daily')
            if px is None:
                rets[name] = 0.0; continue
            if isinstance(px, pd.DataFrame):
                closes = px['close'].values
            elif isinstance(px, dict):
                df = px[code] if code in px else None
                if df is not None and len(df) > 0:
                    col = 'close' if 'close' in df.columns else df.columns[0]
                    closes = df[col].values
                else:
                    rets[name] = 0.0; continue
            else:
                rets[name] = 0.0; continue
            if len(closes) >= 21:
                rets[name] = closes[-1] / closes[-21] - 1.0
            else:
                rets[name] = 0.0
        except Exception:
            rets[name] = 0.0
    style_spread = rets['CSI500'] - rets['CSI300']
    trend = rets['CSI_ALL']
    if style_spread > 0.02 and trend > -0.01:
        return 'small_bull'
    return 'large_dominant'


'''

code_merge = code_merge.replace("def main(context):", DETECT + "def main(context):", 1)

# 4) main() 内注入 regime 切换 (在 stocks = get_stock_pool 之前)
SWITCH = """    # ── Regime 检测 + 权重切换 (纯切换 100/0) ──
    regime = detect_regime(context)
    g.weights = WEIGHTS_LD if regime == 'large_dominant' else WEIGHTS_SB
    if g.debug:
        log.info('[%s] REGIME=%s -> %d factors' % (date_str, regime, len(g.weights)))

"""
code_merge = code_merge.replace("    stocks = get_stock_pool(context)",
                                SWITCH + "    stocks = get_stock_pool(context)", 1)

with open(os.path.join(OUT_DIR, 'fa_v711_merged.py'), 'w', encoding='utf-8') as f:
    f.write(code_merge)
print("  -> fa_v711_merged.py")

# ── 校验: 用 compile() (非 ast.parse!) 检查三层缩进 ──
import py_compile, tempfile
for fn in ['fa_v711_large_dominant.py', 'fa_v711_small_bull.py', 'fa_v711_merged.py']:
    fp = os.path.join(OUT_DIR, fn)
    try:
        with open(fp, 'r', encoding='utf-8') as f:
            src = f.read()
        compile(src, fp, 'exec')   # 捕获 IndentationError / SyntaxError
        print(f"  [OK] {fn}: compile() 通过")
    except (SyntaxError, IndentationError) as e:
        print(f"  [FAIL] {fn}: {e}")

print("\n完成! 输出目录:", os.path.abspath(OUT_DIR))
