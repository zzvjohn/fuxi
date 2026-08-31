# 聚宽因子库 → 因子炼金术 覆盖映射
========================================
创建日期: 2026-06-07
数据来源: https://www.joinquant.com/help/api/help#name:factor_values

## 总览

| 聚宽品类 | 因子数(约) | 已覆盖 | 本次补充 | 无法覆盖 | 说明 |
|----------|:--:|:--:|:--:|:--:|------|
| 风格因子 (Barra) | 11 | 8 | 0 | 3 | 需本地jqdatasdk/复杂正交化 |
| 风格因子PRO | 16 | 0 | 0 | 16 | 仅本地jqdatasdk可用 |
| 质量因子 | ~40 | 15+ | 0 | ~20 | 部分需Tushare fina_indicator |
| 基础因子 | ~30 | 10+ | 0 | ~20 | 部分需Tushare三表 |
| 成长因子 | 9 | 6+ | 0 | ~3 | |
| 每股因子 | 14 | 0 | 0 | 14 | 需Tushare总股本数据 |
| 情绪因子 | 25+ | 0 | 5 | ~20 | 本次重点补充 |
| 动量因子 | 25+ | 10 | 7 | ~10 | 趋势延伸+形态 |
| 技术因子 | 20+ | 3 | 4 | ~15 | MA/MACD类可由boll/rsi覆盖 |
| 风险因子 | 12 | 0 | 4 | 8 | 偏度/峰度/Sharpe |
| 行业因子 | ~50 | 0 | 0 | ~50 | 需聚宽行业分类API |
| **合计** | **~250** | **~52** | **+20** | **~180** | **覆盖核心品类** |

## 一、风格因子 (Barra) — 11个

| 聚宽因子 | 对应的因子炼金术因子 | 状态 |
|----------|---------------------|:--:|
| size (市值) | ln_mcap | ✅ |
| beta (贝塔) | beta | ✅ |
| momentum (动量) | ret_12m, ret_1m_skip1m | ✅ 近似 |
| residual_volatility (残差波动) | idio_vol, downside_vol | ✅ 近似 |
| non_linear_size (非线性市值) | — | ❌ 需正交化 |
| book_to_price_ratio (账面市值比) | bp | ✅ |
| liquidity (流动性) | avg_turnover_1m, dollar_vol_20d | ✅ 近似 |
| earnings_yield (盈利收益) | ep, cfp | ✅ 近似 |
| growth (成长) | earnings_growth_yoy, rev_growth_yoy | ✅ 近似 |
| leverage (杠杆) | — | ❌ 需负债/权益数据 |
| 行业因子 | — | ❌ 需聚宽API |

**结论**: 11个Barra因子中覆盖8个核心方向, 3个因数据/计算复杂度暂不覆盖。

## 二、情绪因子 (Sentiment) — 本次重点补充

| 聚宽因子 | 因子炼金术 | 计算逻辑 | 方向 |
|----------|-----------|----------|:--:|
| VR (成交量比率) | vr_26 | AVS+0.5CVS / BVS+0.5CVS | 负向 |
| VROC12 (量变动速率) | vroc_12 | (Vol_t-Vol_{t-12})/Vol_{t-12} | 负向 |
| PSY (心理线) | psy_12 | 上涨天数/12 | 负向 |
| money_flow_20 (资金流量) | money_flow_20 | TP*Vol / MA(TP*Vol) | 负向 |
| DAVOL20 (换手异动) | davol_20 | 20日换手/120日换手 | 负向 |
| VROC6 | — | — | vroc_12可代表 |
| VMACD/VDIFF/VDEA | vm_diff | EMA(Vol,12)-EMA(Vol,26) | 负向 |
| VOSC | — | — | vm_diff可代表 |
| WVAD | — | — | 与money_flow类似 |
| MAWVAD | — | — | |
| VSTD10/20 | — | — | 有 turnover_std |
| TVSTD6/20 | — | — | |
| TVMA6/20 | tvma_20 | 成交额/20日均成交额 | 负向 |
| VEMA5/10/12/26 | — | — | vm_diff涵盖 |
| AR/BR/ARBR | — | — | 与VR类似 |
| VOL5/10/20/60/120/240 | — | — | 有 avg_turnover_1m/3m |
| DAVOL5/10 | — | — | davol_20可代表 |
| ATR6/14 | atr_14 | EMA(TR,14)/Close | 负向 |
| turnover_volatility | — | — | 有 turnover_std |

**补充**: 5个全新情绪因子 + 3个成交量结构因子 = 8个

## 三、动量因子 (Momentum/Price) — 趋势延伸+形态

| 聚宽因子 | 因子炼金术 | 计算逻辑 | 方向 |
|----------|-----------|----------|:--:|
| BIAS5/10/20/60 | bias_20 | (Close-MA20)/MA20 | 负向 |
| ROC6/12/20/60/120 | roc_20 | (Close_t-Close_{t-20})/Close_{t-20} | 负向 |
| CCI10/14/20/88 | cci_14 | (TP-MA)/0.015/MD | 负向 |
| PLRC6/12/24 | plrc_12 | OLS斜率 | 正向 |
| Price1M/3M/1Y | price_1m | Close/MA(Close,21)-1 | 负向 |
| Rank1M | rank_1m | Rank(20日收益)/N | 正向 |
| fifty_two_week_close_rank | high_52w_rank | Close/max(Close,250) | 负向 |
| bull_power | bull_power | (High-EMA13)/Close | 负向 |
| bear_power | bear_power | (Low-EMA13)/Close | 正向 |
| aroon_up/down_25 | — | — | 与plrc类似 |
| BBIC | — | — | |
| CR20 | — | — | |
| MASS | — | — | 计算复杂 |
| TRIX5/10 | — | — | 三重平滑, 噪音高 |
| single_day_VPT | (vpt) | — | 已有vpt |
| Volume1M | — | — | |
| MACDC | — | — | boll_pct_b类似 |
| EMA/MAC系列 | — | — | boll_pct_b/price_1m覆盖 |

**补充**: 5个趋势延伸 + 4个形态 = 9个

## 四、风险因子 (Distribution) — 收益分布

| 聚宽因子 | 因子炼金术 | 计算逻辑 | 方向 |
|----------|-----------|----------|:--:|
| Variance20/60/120 | — | — | 有 vol_1m/3m |
| Skewness20/60/120 | skewness_20 | 收益三阶矩 | 负向 |
| Kurtosis20/60/120 | kurtosis_20 | 收益四阶矩 | 负向 |
| sharpe_ratio_20/60/120 | sharpe_20 | (μ-Rf)/σ | 正向 |

**补充**: 3个全新(偏度/峰度/夏普) + 1个ATR = 4个

## 五、未覆盖的高价值因子 (Future Work)

以下因子当前未实现, 但具有潜在Alpha, 后续可考虑:

### 基本面因子 (需Tushare三表+总股本):
- net_profit_to_total_operate_revenue_ttm (净利润率TTM) → 已有 net_margin 近似
- cfo_to_ev (现金流/企业价值) → 数据结构复杂
- roe_ttm/roa_ttm/roic_ttm → 已有 roe/roa/roic
- DEGM (毛利率增长) → 已有 gross_margin
- current_ratio / quick_ratio (流动性比率) → 新数据需求
- debt_to_equity_ratio / MLEV (杠杆) → 新数据需求
- LVGI / SGI / GMI / DSRI (Beniesh M-Score) → 部分已实现(f_score)
- 每股因子 → 需Tushare 总股本

### 风格因子PRO (仅本地jqdatasdk):
- btop, divyild, earnqlty, earnvar, earnyild
- financial_leverage, invsqlty, liquidty(pro)
- long_growth, ltrevrsl, market_beta, market_size
- midcap, profit, relative_momentum, resvol

### 行业因子:
- 聚宽 jq_l1/l2, sw_l1/l2/l3, zjw, A01/HY007 等
- 需要行业分类 → 可从 Tushare sw_daily 生成 dummy

## 六、因子优先级排序 (用于GA海选)

按预期Alpha贡献排序 (基于Phase 2研究 + A股经济学逻辑):

| # | 因子 | 类别 | 预期Alpha | 理由 |
|:--:|------|------|:--:|------|
| 1 | streak | momentum | 🔴🔴🔴🔴 | Phase2 ICIR=+2.16, 最强单因子 |
| 2 | short_rev_5d | momentum | 🔴🔴🔴🔴 | A股短期反转极强 |
| 3 | bias_20 | momentum | 🔴🔴🔴 | 乖离率反转, A股习性 |
| 4 | cci_14 | momentum | 🔴🔴🔴 | 顺势反转, 经典技术 |
| 5 | roc_20 | momentum | 🔴🔴🔴 | 变动速率反转 |
| 6 | skewness_20 | distribution | 🔴🔴🔴 | 彩票型识别, 学术验证 |
| 7 | high_52w_rank | pattern | 🔴🔴 | 52周锚定效应 |
| 8 | rank_1m | pattern | 🔴🔴 | 排名反转 |
| 9 | rsi_14 | momentum | 🔴🔴 | RSI反转 |
| 10| vr_26 | sentiment | 🔴🔴 | 量比情绪 |
| 11| psy_12 | sentiment | 🔴🔴 | 心理线 |
| 12| atr_14 | distribution | 🔴🔴 | 波动异象 |
| 13| money_flow_20 | sentiment | 🔴🔴 | 资金流 |
| 14| kurtosis_20 | distribution | 🔴 | 峰度风险 |
| 15| sharpe_20 | distribution | 🔴 | 夏普动量 |
| 16| price_1m | momentum | 🔴 | 价格位置 |
| 17| plrc_12 | momentum | 🔴 | 趋势斜率 |
| 18| bull_power | pattern | 🔴 | 多头力道 |
| 19| bear_power | pattern | 🔴 | 空头力道 |
| 20| vroc_12 | sentiment | 🔴 | 量变动 |
| 21| davol_20 | sentiment | 🔴 | 换手异动 |
| 22| vm_diff | volume_structure | 🔴 | 量MACD |
| 23| tvma_20 | volume_structure | ⚪ | 量能 |
| 24| high_low_range | volatility | 🔴 | 日内振幅 |
| 25| vpt | liquidity | ⚪ | 量价趋势 |
| 26| boll_pct_b | volatility | 🔴 | 布林带 |
| 27| volume_ratio | turnover | ⚪ | 量比 |

🔴🔴🔴🔴 = 期望ICIR > 1.5 | 🔴🔴🔴 = 0.8-1.5 | 🔴🔴 = 0.5-0.8 | 🔴 = 0.3-0.5 | ⚪ = 待验证
