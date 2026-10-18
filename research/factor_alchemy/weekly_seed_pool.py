# -*- coding: utf-8 -*-
"""
P-20260901-004 周频原生种子子池
================================================================================
背景: v0.7 分频裁决转正后 weekly lane 本地→JQ 折扣近似 0 (8-31 五个周频校准
因子 JQ 全灭)。模板池 141 条几乎全为日频思维, 周频原生因子缺失 →
生成侧与裁决侧频率错配 (周频裁决器数据源 weekly_prices.parquet)。

本模块:
  1. WEEKLY_NATIVE_SEEDS — 13 条周频原生种子 (周频采样下自然表达, 单行 pandas
     公式, 字段=周频宽表 close/open/high/low/volume/amount/returns/momentum_12_1/
     vol_60/log_size, 与 WeeklyLaneJudge._wide_env 一致)。
  2. inject_weekly_seeds() — 落盘 data/weekly_native_seeds.json (持久化子池)。
  3. evaluate_seeds() — 用 WeeklyLaneJudge 批量本地周频裁决 (只报告, 不直接入库;
     通过 τ_w 的种子由用户/D+ 流程决定是否进 JQ)。
  4. format_prompt_reference() — 给 ralph_loop LLM 周频语境提供 few-shot 参考。

回滚: 子池独立于日频模板池, 清空 data/weekly_native_seeds.json 即回滚。
"""
import json
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SEEDS_PATH = _PROJECT_ROOT / "data" / "weekly_native_seeds.json"

# 周频原生种子: 语义在周频采样下才自然成立 (周内聚合 / 周频趋势 / 周频微观结构)
# 公式必须单行 (WeeklyLaneJudge.judge 为单行 eval), pandas 风格, 三重约束安全。
WEEKLY_NATIVE_SEEDS = [
    {
        "name": "w_native_mom4",
        "label": "周频4周动量",
        "category": "weekly_momentum",
        "hypothesis": "周频采样下动量效应更纯净(剔除日频噪声), 4周=月度动量中频带",
        "logic": "周频趋势延续: 月度动量在A股周频层面显著(与日频反转分层)",
        "formula": "close.pct_change(4)",
        "direction": "long",
        "natural_freq": "weekly",
    },
    {
        "name": "w_native_reversal8",
        "label": "周频8周反转",
        "category": "weekly_reversal",
        "hypothesis": "8周(2个月)回调后存在均值回复, 与4周动量形成频率分层",
        "logic": "周频中期反转: 机构调仓周期(季初/季末)引发的过度反应修复",
        "formula": "-(close.pct_change(8))",
        "direction": "long",
        "natural_freq": "weekly",
    },
    {
        "name": "w_native_vol_meanrev",
        "label": "周频波动率均值回复",
        "category": "weekly_volatility",
        "hypothesis": "高波动周后波动率收敛, 低波动率股后续收益更高(波动率之谜)",
        "logic": "周频波动率异象: 周线级低波动溢价, 日频噪声剔除后更稳",
        "formula": "-(returns.rolling(8).std())",
        "direction": "long",
        "natural_freq": "weekly",
    },
    {
        "name": "w_native_price_vol_diverge",
        "label": "周频价跌缩量背离",
        "category": "weekly_price_volume",
        "hypothesis": "价格下跌但周成交量萎缩=抛压衰竭, 后续反弹概率高",
        "logic": "周频量价背离: 缩量下跌表明卖方枯竭, 反转条件形成",
        "formula": "-(close.pct_change(4) * (1 - volume / volume.rolling(12).mean()))",
        "direction": "long",
        "natural_freq": "weekly",
    },
    {
        "name": "w_native_ideal_amp_reversal",
        "label": "周频理想振幅反转(单笔金额加权)",
        "category": "weekly_behavioral",
        "hypothesis": "以周均单笔成交金额为权重切割反转, 高单笔金额周的信息含量更高",
        "logic": "开源金工理想振幅因子周频版: W式切割反转, 信息优势方执行周贡献更高",
        "formula": "-((close.pct_change() * (amount / volume)).rolling(8).sum() / (amount / volume).rolling(8).sum())",
        "direction": "long",
        "natural_freq": "weekly",
    },
    {
        "name": "w_native_hl_range_reversal",
        "label": "周频振幅反转",
        "category": "weekly_behavioral",
        "hypothesis": "周内高低振幅扩大=情绪过热, 4周均值后回归",
        "logic": "周频振幅反转: 大振幅周=分歧剧烈, 后续收敛伴随价格修复",
        "formula": "-((high - low).rolling(4).mean() / close)",
        "direction": "long",
        "natural_freq": "weekly",
    },
    {
        "name": "w_native_channel_pos",
        "label": "52周通道位置",
        "category": "weekly_trend",
        "hypothesis": "周线52周高低通道位置度量长期趋势健康度, 中位以上趋势延续",
        "logic": "周频通道: 周线级别趋势跟踪, 52周=年线通道与机构年度调仓周期对齐",
        "formula": "(close - close.rolling(52).min()) / (close.rolling(52).max() - close.rolling(52).min() + 1e-8)",
        "direction": "long",
        "natural_freq": "weekly",
    },
    {
        "name": "w_native_parkinson_vol_rev",
        "label": "周频Parkinson波动率反转",
        "category": "weekly_volatility",
        "hypothesis": "周内高低价极差隐含波动率扩大后反转, Parkinson估计量比收益std更高效",
        "logic": "Parkinson(1980)高低价波动率: 周频极差波动率高→恐慌过度→修复",
        "formula": "-(((np.log(high / low) ** 2).rolling(8).sum() / (4 * np.log(2) * 8)) ** 0.5)",
        "direction": "long",
        "natural_freq": "weekly",
    },
    {
        "name": "w_native_obv_ratio",
        "label": "周频OBV占比",
        "category": "weekly_price_volume",
        "hypothesis": "周频能量潮(OBV)占比度量资金累积方向, 正向累积趋势延续",
        "logic": "周频OBV: 涨周成交量累积占比, 剔除日频换手噪声的资金流方向",
        "formula": "(np.sign(returns) * volume).rolling(12).sum() / volume.rolling(12).sum()",
        "direction": "long",
        "natural_freq": "weekly",
    },
    {
        "name": "w_native_vol_spike_reversal",
        "label": "周频量比突变反转",
        "category": "weekly_behavioral",
        "hypothesis": "周成交量相对4周均值突变(>1倍)伴随价格波动=情绪峰值, 反转",
        "logic": "周频量比突变: 放量周+动量方向反转, 捕捉情绪极值",
        "formula": "-((volume / volume.rolling(4).mean()).clip(0, 5) * close.pct_change(4))",
        "direction": "long",
        "natural_freq": "weekly",
    },
    {
        "name": "w_native_mom_lowvol",
        "label": "周频动量质量比",
        "category": "weekly_quality",
        "hypothesis": "单位波动率携带的动量=动量质量, 低波高质量动量延续性更强",
        "logic": "周频动量/波动率比: 夏普式动量, 过滤高波动的伪动量",
        "formula": "close.pct_change(8) / (returns.rolling(12).std() + 1e-8)",
        "direction": "long",
        "natural_freq": "weekly",
    },
    {
        "name": "w_native_ret_skew_rev",
        "label": "周频收益率偏度反转",
        "category": "weekly_behavioral",
        "hypothesis": "正偏度周频收益=彩票型偏好推高价格, 后续反转",
        "logic": "周频彩票股反转: 高偏度股票被散户追捧, 12周后均值回复",
        "formula": "-(returns.rolling(12).skew())",
        "direction": "long",
        "natural_freq": "weekly",
    },
    {
        "name": "w_native_size_mom_interact",
        "label": "周频市值分层动量",
        "category": "weekly_interaction",
        "hypothesis": "市值相对自身12周均值的偏移放大/抑制动量效应(小市值+动量更强)",
        "logic": "周频市值×动量交互: 市值漂移期动量分化, log_size偏离均值作调节器",
        "formula": "close.pct_change(4) * (log_size - log_size.rolling(12).mean())",
        "direction": "long",
        "natural_freq": "weekly",
    },

    # ============================================================
    # C1 补充 (2026-09-10, 状态条件化 + 因子动量 + 量价背离族)
    # 与已有 13 条区分: 不再用裸动量/裸反转主体, 全部为状态加权或复合结构
    # ============================================================
    {
        "name": "w_native_amount_spike_rev8",
        "label": "周频金额比突增加权8周反转",
        "category": "weekly_conditional",
        "hypothesis": "成交金额比突增(周频拥挤代理)的股票8周收益反转更强",
        "logic": "金额比偏离(amount/12周均amount-1)加权8周反转, 周频拥挤度条件化",
        "formula": "-(close.pct_change(8) * ((amount / amount.rolling(12).mean()) - 1))",
        "direction": "long",
        "natural_freq": "weekly",
    },
    {
        "name": "w_native_mom_accel_4w",
        "label": "周频动量加速度",
        "category": "weekly_momentum",
        "hypothesis": "4周动量快于8周动量的股票(动量加速)收益延续更强",
        "logic": "动量二阶导: 短窗动量-长窗动量, 刻画趋势在加速而非钝化",
        "formula": "close.pct_change(4) - close.pct_change(8)",
        "direction": "long",
        "natural_freq": "weekly",
    },
    {
        "name": "w_native_price_amt_corr_rev",
        "label": "周频量价相关背离反转",
        "category": "weekly_conditional",
        "hypothesis": "12周量价相关性过高的股票后续反转(量价一致透支), 背离股修复",
        "logic": "手工构造12周量价相关(避免.corr依赖): 收益×金额收益协动/各自std, 取负",
        "formula": "-((close.pct_change() * amount.pct_change()).rolling(12).mean() / (close.pct_change().rolling(12).std() * amount.pct_change().rolling(12).std() + 1e-8))",
        "direction": "long",
        "natural_freq": "weekly",
    },
    {
        "name": "w_native_vol_state_rev8",
        "label": "周频波动状态加权8周反转",
        "category": "weekly_conditional",
        "hypothesis": "60日波动率相对自身12周均值升高的股票8周收益反转更强",
        "logic": "波动状态偏离(vol_60/12周均vol_60-1)加权8周反转, 波动扩张=情绪过热代理",
        "formula": "-(close.pct_change(8) * ((vol_60 / vol_60.rolling(12).mean()) - 1))",
        "direction": "long",
        "natural_freq": "weekly",
    },
    {
        "name": "w_native_factor_mom_4w",
        "label": "周频因子动量",
        "category": "weekly_momentum",
        "hypothesis": "12-1动量因子自身4周上升的股票(因子动能)收益延续",
        "logic": "预计算因子momentum_12_1的4周差分, 刻画alpha信号本身的动量",
        "formula": "momentum_12_1 - momentum_12_1.shift(4)",
        "direction": "long",
        "natural_freq": "weekly",
    },
    {
        "name": "w_native_close_pos_8",
        "label": "周频8周通道位置",
        "category": "weekly_momentum",
        "hypothesis": "8周通道位置高的股票短期延续(短通道比52周通道更敏感)",
        "logic": "8周min-max归一化位置, 与52周版互补: 捕捉更近期的新高/新低结构",
        "formula": "(close - close.rolling(8).min()) / (close.rolling(8).max() - close.rolling(8).min() + 1e-8)",
        "direction": "long",
        "natural_freq": "weekly",
    },
    {
        "name": "w_native_avg_price_spike_rev",
        "label": "周频均价比突增加权4周反转",
        "category": "weekly_conditional",
        "hypothesis": "周均单价(amount/vol)相对12周均值突增的股票4周反转",
        "logic": "均价比偏离加权4周反转, 均价比突增=大单成交占比上升(资金结构变化)代理",
        "formula": "-(close.pct_change(4) * (((amount / volume.clip(lower=1)) / (amount / volume.clip(lower=1)).rolling(12).mean()) - 1))",
        "direction": "long",
        "natural_freq": "weekly",
    },
    {
        "name": "w_native_size_reversal_4",
        "label": "周频市值偏离加权4周反转",
        "category": "weekly_interaction",
        "hypothesis": "市值高于自身12周均值的股票4周反转更强(市值漂移后均值回归)",
        "logic": "log_size正向偏离加权4周反转, 与 size_mom_interact 反向下注对照",
        "formula": "-(close.pct_change(4)) * (log_size - log_size.rolling(12).mean())",
        "direction": "long",
        "natural_freq": "weekly",
    },
    {
        "name": "w_native_big_move_rev",
        "label": "周频单周大波动反转",
        "category": "weekly_conditional",
        "hypothesis": "单周收益绝对值远超自身12周均值的股票下周反转",
        "logic": "波动比(本周|ret|/12周均|ret|)加权单周反转, 极端波动后的均值回复",
        "formula": "-(close.pct_change() * (close.pct_change().abs() / (close.pct_change().abs().rolling(12).mean() + 1e-8)))",
        "direction": "long",
        "natural_freq": "weekly",
    },
]


def inject_weekly_seeds(verbose: bool = True) -> list:
    """落盘周频原生种子子池 (幂等: 同名覆盖刷新)。返回种子列表。"""
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "count": len(WEEKLY_NATIVE_SEEDS),
        "seeds": WEEKLY_NATIVE_SEEDS,
    }
    _SEEDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_SEEDS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    if verbose:
        print(f"[WeeklySeedPool] 子池已落盘: {_SEEDS_PATH} ({len(WEEKLY_NATIVE_SEEDS)} 条)")
    return WEEKLY_NATIVE_SEEDS


def evaluate_seeds(tau_w: float = 0.15, verbose: bool = True) -> dict:
    """用 WeeklyLaneJudge 批量本地周频裁决种子 (只报告, 不入库)。

    tau_w: lane_calibration 当前生效 τ_w (默认 0.15, 调用方可传动态值)。
    """
    from weekly_lane import WeeklyLaneJudge

    judge = WeeklyLaneJudge(icir_threshold=tau_w)
    results = []
    for s in WEEKLY_NATIVE_SEEDS:
        r = judge.judge(s["formula"], s["name"])
        r["name"] = s["name"]
        r["label"] = s["label"]
        r["category"] = s["category"]
        results.append(r)
        status = "PASS" if r.get("passed") else ("eval失败" if not r.get("eval_ok") else "未过")
        if verbose:
            print(f"  {s['name']}: weekly_ic={r.get('weekly_ic', 0):+.4f} "
                  f"weekly_icir={r.get('weekly_icir', 0):.3f} (τ_w={tau_w}) → {status}")
    passed = [r for r in results if r.get("passed")]
    out = {
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tau_w": tau_w,
        "total": len(results),
        "passed_count": len(passed),
        "results": results,
    }
    eval_path = _PROJECT_ROOT / "data" / "weekly_native_seeds_eval.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    if verbose:
        print(f"[WeeklySeedPool] 评估报告: {eval_path} ({len(passed)}/{len(results)} 过 τ_w)")
    return out


def format_prompt_reference(max_n: int = 6) -> str:
    """给 ralph_loop LLM 周频语境提供 few-shot 参考文本。"""
    lines = [
        "以下是已定义的周频原生种子公式 (周频宽表字段: close/open/high/low/"
        "volume/amount/returns/momentum_12_1/vol_60/log_size), 供风格参考:"
    ]
    for s in WEEKLY_NATIVE_SEEDS[:max_n]:
        lines.append(f"- {s['name']}: {s['formula']}  # {s['logic'][:40]}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    inject_weekly_seeds()
    tau = float(sys.argv[1]) if len(sys.argv) > 1 else 0.15
    evaluate_seeds(tau_w=tau)
