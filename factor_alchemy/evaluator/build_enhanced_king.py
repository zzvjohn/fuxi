"""
A路线: 将影子验证的两个赢家因子注入王者JQ策略
- 替换 comp3 (money_flow×ret_3m) → idio_tail_hedge × dollar_vol_20d
- 替换 comp6 (nf_02a304×money_flow) → capital_efficiency × dollar_vol_20d
- 保留 comp1/comp2/comp4/comp5 不变
"""

import re

KING_FILE = r"E:\quant\research\factor_alchemy\output\fa_alpha_agent_v3_jq_20260802_130337.py"
OUT_FILE = r"E:\quant\research\factor_alchemy\output\fa_enhanced_v1_jq.py"

# ── 新增辅助函数 ──
MKT_RET_HELPER = '''
def _ensure_mkt_ret(stocks, price_data, need):
    """等权市场日收益, 缓存到 price_data['_mkt_ret']"""
    import numpy as np
    cached = price_data.get('_mkt_ret')
    if cached is not None and len(cached) >= need - 1:
        return cached
    close = price_data.get('close', {})
    rets = []
    for s in stocks[:800]:
        c = close.get(s)
        if c is None or len(c) < need:
            continue
        c = np.array(c[-need:], dtype=float)
        r = np.diff(c) / np.where(c[:-1] == 0, np.nan, c[:-1])
        rets.append(r)
    if len(rets) < 20:
        price_data['_mkt_ret'] = None
        return None
    mkt = np.nanmean(np.array(rets), axis=0)
    price_data['_mkt_ret'] = mkt
    return mkt

'''

# ── 新增因子实现 (来自 shadow_eval_v4, JQ实测通过) ──
IDIO_TAIL_FACTOR = '''
def compute_idiosyncratic_tail_hedge_premium(stocks, price_data):
    """CAPM残差尾部: 60日beta回归残差的最差5日均值取负.
    影子v4验证: +141.1%/Sharpe0.67, 全17袖子冠军"""
    import numpy as np
    n = len(stocks)
    close = price_data.get('close', {})
    mkt = _ensure_mkt_ret(stocks, price_data, 61)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        c = close.get(s)
        if c is None or len(c) < 61 or mkt is None:
            continue
        c = np.array(c[-61:], dtype=float)
        ri = np.diff(c) / np.where(c[:-1] == 0, np.nan, c[:-1])
        rm = mkt[-60:]
        k = min(len(ri), len(rm))
        if k < 40:
            continue
        ri, rm = ri[-k:], rm[-k:]
        ok = ~(np.isnan(ri) | np.isnan(rm))
        if int(np.sum(ok)) < 40:
            continue
        ri, rm = ri[ok], rm[ok]
        mvar = np.var(rm)
        if mvar < 1e-12:
            continue
        beta = np.cov(ri, rm)[0, 1] / mvar
        resid = ri - beta * rm
        worst5 = np.mean(np.sort(resid)[:5])
        arr[i] = -worst5
        valid[i] = True
    return arr, valid

'''

CAPITAL_EFF_FACTOR = '''
def compute_capital_efficiency_proxy(stocks, price_data):
    """资本效率: |ret_20d| / mean(dollar_vol_20d) * 1e8.
    单位成交额产出的价格变动效率(Amihud反面). 影子v4亚军: +78.7%/Sharpe0.52"""
    import numpy as np
    n = len(stocks)
    close = price_data.get('close', {})
    volume = price_data.get('volume', {})
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        c, v = close.get(s), volume.get(s)
        if c is None or v is None or len(c) < 41 or len(v) < 41:
            continue
        c = np.array(c[-41:], dtype=float)
        v = np.array(v[-41:], dtype=float)
        ret20 = c[-1] / c[-21] - 1.0 if c[-21] > 0 else np.nan
        dv = np.nanmean(v[-20:] * c[-20:])
        if not np.isfinite(ret20) or not np.isfinite(dv) or dv <= 0:
            continue
        arr[i] = abs(ret20) / dv * 1e8
        valid[i] = True
    return arr, valid

'''

# ── 新增复合函数 ──
COMP3_NEW = '''
def compute_score_comp3_idio_tail_hedge_dollar(stocks, price_data):
    """特质尾部对冲x成交额: 影子v4冠军组合. CAPM残差最差5日捕捉尾部风险溢价,
    成交额确保流动性. 互补评分=8(异质+流动性)"""
    a, va = compute_idiosyncratic_tail_hedge_premium(stocks, price_data)
    b, vb = compute_dollar_vol_20d(stocks, price_data)
    valid = va & vb
    ra = rank_pct(a, valid)
    rb = rank_pct(b, valid)
    score = np.where(valid, ra * rb, 0.5)
    return score, valid

'''

COMP6_NEW = '''
def compute_score_comp6_capital_eff_dollar(stocks, price_data):
    """资本效率x成交额: 影子v4亚军组合. 单位成交额的价格效率捕捉资金利用效率,
    成交额过滤低流动性噪音. 互补评分=8(质量+流动性)"""
    a, va = compute_capital_efficiency_proxy(stocks, price_data)
    b, vb = compute_dollar_vol_20d(stocks, price_data)
    valid = va & vb
    ra = rank_pct(a, valid)
    rb = rank_pct(b, valid)
    score = np.where(valid, ra * rb, 0.5)
    return score, valid

'''

# ── 更新后的 ensemble ──
ENSEMBLE_NEW = '''def compute_ensemble(stocks, price_data):
    import numpy as np
    n = len(stocks)
    composite = np.ones(n) * 0.5
    valid = np.ones(n, dtype=bool)
    score_funcs = [
        (compute_score_comp1_overnight_tvma,),
        (compute_score_comp2_dollar_turnover,),
        (compute_score_comp3_idio_tail_hedge_dollar,),
        (compute_score_comp4_tvma_20,),
        (compute_score_comp5_dollar_vol_20d,),
        (compute_score_comp6_capital_eff_dollar,),
    ]
    count = 0
    for (func,) in score_funcs:
        score, v = func(stocks, price_data)
        n_v = int(np.sum(v))
        if n_v > 0:
            composite[v] *= score[v]
            count += 1
            valid = valid & v
    return composite, valid'''


def main():
    with open(KING_FILE, encoding='utf-8') as f:
        src = f.read()

    # 1. 在 compute_dollar_vol_20d 之后插入辅助 + 两个新因子
    #    插入位置: 在 def compute_overnight_5d 之前 (第一个使用price_data的因子之后)
    insert_marker = "def compute_overnight_5d(stocks, price_data):"
    insert_code = MKT_RET_HELPER + IDIO_TAIL_FACTOR + CAPITAL_EFF_FACTOR
    src = src.replace(insert_marker, insert_code + insert_marker)

    # 2. 替换 comp3
    old_comp3 = r'''def compute_score_comp3_moneyflow_ret3m(stocks, price_data):
    """资金流x动量反转: V3原value_momentum简化版"""
    a, va = compute_money_flow_20(stocks, price_data)
    b, vb = compute_ret_3m(stocks, price_data)
    valid = va & vb
    ra = rank_pct(a, valid)
    rb = rank_pct(b, valid)
    score = np.where(valid, ra * rb, 0.5)
    return score, valid'''
    src = src.replace(old_comp3, COMP3_NEW.strip())

    # 3. 替换 comp6
    old_comp6 = r'''def compute_score_comp6_money_flow_20(stocks, price_data):
    """截面取负((取负(20日最小收益))/波动率)。极端下行风险相对总风险的定价。高 x money_flow_20: 互补"""
    a, va = compute_nf_02a304(stocks, price_data)
    b, vb = compute_money_flow_20(stocks, price_data)
    valid = va & vb
    ra = rank_pct(a, valid)
    rb = rank_pct(b, valid)
    score = np.where(valid, ra * rb, 0.5)
    return score, valid'''
    src = src.replace(old_comp6, COMP6_NEW.strip())

    # 4. 替换 ensemble
    old_ensemble = r'''def compute_ensemble(stocks, price_data):
    import numpy as np
    n = len(stocks)
    composite = np.ones(n) * 0.5
    valid = np.ones(n, dtype=bool)
    score_funcs = [
        (compute_score_comp1_overnight_tvma,),
        (compute_score_comp2_dollar_turnover,),
        (compute_score_comp3_moneyflow_ret3m,),
        (compute_score_comp4_tvma_20,),
        (compute_score_comp5_dollar_vol_20d,),
        (compute_score_comp6_money_flow_20,),
    ]
    count = 0
    for (func,) in score_funcs:
        score, v = func(stocks, price_data)
        n_v = int(np.sum(v))
        if n_v > 0:
            composite[v] *= score[v]
            count += 1
            valid = valid & v
    return composite, valid'''
    src = src.replace(old_ensemble, ENSEMBLE_NEW.strip())

    # 5. 写入
    with open(OUT_FILE, 'w', encoding='utf-8') as f:
        f.write(src)

    # 6. 验证
    import ast
    ast.parse(src)
    print(f"✅ 增强版王者策略已生成: {OUT_FILE}")
    print(f"   文件大小: {len(src)/1024:.1f}KB")
    print(f"   新增: _ensure_mkt_ret, idio_tail_hedge_premium, capital_efficiency_proxy")
    print(f"   替换: comp3(idio_tail×dollar) comp6(capital_eff×dollar)")
    print(f"   保留: comp1(overnight×tvma) comp2(dollar×turnover) comp4(nf_ff7913×tvma) comp5(f×dollar)")

if __name__ == "__main__":
    main()
