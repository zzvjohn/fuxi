"""
GRPO Data Builder — 从因子池构建 SFT/RL 训练数据

从 passed_factor_pool.csv + injected_factors.json + V3 composites
构建 instruction-output 格式的 SFT 数据集。

输出格式 (Alpaca-style):
{
    "instruction": "生成一个[范式]类因子公式...",
    "input": "上下文约束信息",
    "output": "factor_name: xxx\nexpression: ...\nrationale: ..."
}

用法:
    python research/factor_alchemy/grpo_data_builder.py --output data/grpo_sft_data.json
"""

import json
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── 范式→提示词模板 ──────────────────────────────────────
PARADIGM_PROMPTS = {
    "流动性": "生成一个A股流动性类因子公式，捕捉成交额/换手率/买卖价差的截面差异。"
              "因子应能区分高流动性和低流动性股票的未来收益差异。"
              "使用WorldQuant风格表达式，允许 ts_sum/ts_delta/ts_rank 等算子。",
    "动量": "生成一个A股动量效应因子公式，捕捉过去N日收益对未来收益的预测能力。"
            "可考虑多周期动量叠加、动量加速度、或截面动量偏离。"
            "使用WorldQuant风格表达式。",
    "反转": "生成一个A股短期反转效应因子公式，捕捉过度反应后的均值回归。"
            "可基于日内波动、极端收益、或成交量异常。"
            "使用WorldQuant风格表达式。",
    "波动率": "生成一个A股波动率类因子公式，捕捉波动率与预期收益的关系。"
              "可考虑已实现波动率、波动率变化率、或波动率区制转换。"
              "使用WorldQuant风格表达式。",
    "资金流": "生成一个A股资金流类因子公式，用价量关系代理资金进出。"
              "可考虑量价背离、资金流入强度、或主力资金痕迹。"
              "使用WorldQuant风格表达式。",
    "行为金融": "生成一个A股行为金融类因子公式，捕捉投资者非理性行为。"
                "可考虑处置效应、锚定效应、过度自信、注意力驱动交易。"
                "使用WorldQuant风格表达式。",
    "微观结构": "生成一个A股微观结构类因子公式，基于高频/日内的价格形态。"
                "可考虑开盘/收盘位置、日内路径、或订单流不平衡。"
                "使用WorldQuant风格表达式。",
    "隔夜": "生成一个A股隔夜效应因子公式，基于隔夜收益与日内收益的分离。"
            "可考虑隔夜动量、隔夜反转、或隔夜-日内收益比。"
            "使用WorldQuant风格表达式。",
    "尾部风险": "生成一个A股尾部风险类因子公式，捕捉极端事件的预测信号。"
                "可考虑最大回撤、偏度、峰度、VaR/CVaR、暴跌概率。"
                "使用WorldQuant风格表达式。",
    "趋势/动量": "生成一个A股趋势跟踪类因子公式，捕捉中长期趋势延续性。"
                "可考虑移动平均交叉、趋势强度、或趋势一致性。"
                "使用WorldQuant风格表达式。",
    "筹码分布": "生成一个A股筹码分布类因子公式，基于量价形态推断持仓结构。"
                "可考虑成交密集区、浮筹比例、或筹码锁定程度。"
                "使用WorldQuant风格表达式。",
    "基本面/成长": "生成一个A股基本面代理类因子公式，用量价数据代理盈利质量。"
                   "可考虑盈利动量(量价)、资本效率、或收入增长代理。"
                   "使用WorldQuant风格表达式。",
    # 通用
    "通用": "生成一个A股因子公式，要求有清晰的经济学直觉。"
            "使用WorldQuant风格表达式，允许 ts_sum/ts_delta/ts_rank/ts_max/ts_min 等算子。",
}

# ── V3 基准因子 (正样本加强) ─────────────────────────────
V3_BASE_FACTORS = {
    "dollar_vol_x_turnover_std": {
        "expression": "close_p * vol_p / 100000000 * (vol_p / vol_p.rolling(20).mean()).rolling(20).std()",
        "rationale": "流动性与换手率波动的乘积，同时捕捉成交活跃度和交易行为的不稳定性。高值意味着高流动性+交易波动，对短期收益有显著预测力。"
    },
    "overnight_x_tvma": {
        "expression": "(close_p / open_p - 1) * (close_p.rolling(5).mean() / close_p.rolling(20).mean())",
        "rationale": "隔夜收益与趋势强度交叉：隔夜收益反映信息吸收效率，趋势强度反映方向一致性。两者交互捕捉隔夜信息+趋势共振效应。"
    },
    "money_flow_x_value_momentum": {
        "expression": "(close_p * vol_p - close_p.shift(1) * vol_p.shift(1)).rolling(10).sum() / (close_p * vol_p).rolling(10).sum() * (close_p / close_p.shift(60) - 1)",
        "rationale": "资金流指标(CMF)与中长期价值动量的乘积：资金持续流入+股价趋势上行=强信号。同时满足流动性和动量两个维度的要求。"
    },
    "ret_open_x_skewness": {
        "expression": "(close_p / open_p - 1) * (close_p.pct_change().rolling(20).skew())",
        "rationale": "开盘强度(日内收益)与收益偏度的交互：开盘强+正偏度=乐观情绪延续，开盘弱+负偏度=恐慌抛售。情绪+尾部双维度。"
    },
    "chip_distribution_ratio": {
        "expression": "(close_p - close_p.rolling(60).min()) / (close_p.rolling(60).max() - close_p.rolling(60).min()).replace(0, np.nan) * (vol_p.rolling(20).mean() / vol_p.rolling(60).mean())",
        "rationale": "筹码分布位置(价格在60日区间的相对位置)与近期/远期成交量比的乘积：高价位+近期放量=筹码向上转移，低价位+缩量=筹码锁定。"
    },
    "tail_enhanced_momentum": {
        "expression": "close_p.pct_change(20).rank(axis=1, pct=True) * (1 - close_p.pct_change(60).rolling(20).skew().abs())",
        "rationale": "20日动量排名与60日偏度调整的乘积：动量排名捕捉相对强度，偏度调整惩罚尾部风险暴露。高动量+低偏度=稳健趋势。"
    },
}


def build_sft_dataset(
    pool_csv_path: Path,
    injected_json_path: Path,
    output_path: Path,
    include_v3: bool = True,
) -> Dict:
    """构建完整的 SFT 训练数据集。

    Returns:
        dict with 'data' (list of training samples) and 'stats' (metadata)
    """
    samples = []

    # ── 1. 从 passed_factor_pool.csv 提取 ─────────────────
    df = pd.read_csv(pool_csv_path, encoding='utf-8-sig')

    for _, row in df.iterrows():
        formula = row.get('formula', '')
        label = row.get('label', '')
        category = row.get('category', '')
        status = row.get('status', '')
        icir = row.get('icir', row.get('daily_icir', 0))

        if pd.isna(formula) or not str(formula).strip():
            continue
        formula = str(formula).strip()

        # 根据类别匹配范式
        paradigm = _map_category_to_paradigm(category)
        prompt = PARADIGM_PROMPTS.get(paradigm, PARADIGM_PROMPTS["通用"])

        # 构建质量标签
        quality = _estimate_quality(status, icir)
        rationale = row.get('logic', row.get('hypothesis', ''))
        if pd.isna(rationale):
            rationale = f"{label}: {paradigm}维度因子"

        output = _format_output(label, formula, str(rationale), icir, quality)

        sample = {
            "instruction": prompt,
            "input": f"目标范式: {paradigm}, 类别: {category}, 方向: {row.get('direction', 'long')}",
            "output": output,
            "metadata": {
                "source": "passed_factor_pool",
                "quality": quality,
                "icir": float(icir) if not pd.isna(icir) else 0.0,
                "paradigm": paradigm,
                "category": str(category),
            }
        }
        samples.append(sample)

    # ── 2. 从 injected_factors.json 提取 ──────────────────
    if injected_json_path.exists():
        with open(injected_json_path, 'r', encoding='utf-8') as f:
            injected = json.load(f)

        for factor in injected.get('injected', []):
            name = factor['factor_name']
            label = factor.get('factor_label', name)
            fri = factor.get('fri', 0)
            fri_grade = factor.get('fri_grade', '')
            cat = factor.get('category', '')
            rationale = factor.get('rationale', '')

            # 需要从 passed_factor_pool 回查公式
            formula = _find_formula_by_name(df, name)
            if not formula:
                continue

            paradigm = _map_category_to_paradigm(cat)
            prompt = PARADIGM_PROMPTS.get(paradigm, PARADIGM_PROMPTS["通用"])

            quality = "high" if fri_grade == 'A' else "medium"
            output = _format_output(label, formula, rationale[:300], 0, quality)
            output += f"\nFRI: {fri:.3f} ({fri_grade})"

            sample = {
                "instruction": prompt,
                "input": f"FRI验证因子, paradigm={paradigm}, fri_grade={fri_grade}",
                "output": output,
                "metadata": {
                    "source": "injected_factors",
                    "quality": quality,
                    "fri": fri,
                    "paradigm": paradigm,
                }
            }
            samples.append(sample)

    # ── 3. V3 基准因子 (高质量正样本) ──────────────────────
    if include_v3:
        for name, info in V3_BASE_FACTORS.items():
            # 根据名称推断范式
            if 'turnover' in name or 'dollar' in name:
                paradigm = "流动性"
            elif 'overnight' in name:
                paradigm = "隔夜"
            elif 'money_flow' in name:
                paradigm = "资金流"
            elif 'skewness' in name or 'tail' in name:
                paradigm = "尾部风险"
            elif 'chip' in name:
                paradigm = "筹码分布"
            elif 'value' in name or 'momentum' in name:
                paradigm = "趋势/动量"
            else:
                paradigm = "通用"

            prompt = PARADIGM_PROMPTS.get(paradigm, PARADIGM_PROMPTS["通用"])
            output = _format_output(name, info['expression'], info['rationale'], 0, "high")
            output += "\n来源: V3王者策略基准因子"

            sample = {
                "instruction": prompt,
                "input": f"V3王者策略基准因子, paradigm={paradigm}",
                "output": output,
                "metadata": {
                    "source": "v3_base",
                    "quality": "high",
                    "paradigm": paradigm,
                }
            }
            # 正样本加倍 (重要)
            samples.append(sample)
            samples.append(sample)

    # ── 统计 ──────────────────────────────────────────────
    stats = {
        "total_samples": len(samples),
        "by_source": {},
        "by_paradigm": {},
        "by_quality": {},
    }
    for s in samples:
        src = s['metadata']['source']
        stats['by_source'][src] = stats['by_source'].get(src, 0) + 1
        par = s['metadata']['paradigm']
        stats['by_paradigm'][par] = stats['by_paradigm'].get(par, 0) + 1
        q = s['metadata']['quality']
        stats['by_quality'][q] = stats['by_quality'].get(q, 0) + 1

    result = {
        "version": "2.0.0",
        "description": "GRPO SFT training data from factor pool",
        "stats": stats,
        "data": samples,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


# ── 辅助函数 ──────────────────────────────────────────────

def _map_category_to_paradigm(category: str) -> str:
    """将因子类别映射到 12 范式之一"""
    if not category or pd.isna(category):
        return "通用"

    cat = str(category).lower()
    mapping = {
        # ── 标准类别 ──────────────────────────────────
        'volume_structure': '流动性',
        'liquidity_micro': '微观结构',
        'return_decomposition': '隔夜',
        'momentum_dynamics': '趋势/动量',
        'price_pattern': '反转',
        'volatility_structure': '波动率',
        'return_distribution': '尾部风险',
        'behavioral': '行为金融',
        'correlation_linkage': '通用',
        'information_flow': '微观结构',
        'extreme_events': '尾部风险',
        'event_driven': '基本面/成长',
        'multi_period': '趋势/动量',
        'relative_value': '基本面/成长',
        'trend_quality': '趋势/动量',
        'growth_quality': '基本面/成长',
        'mid_report_divergence': '趋势/动量',
        '学术异象库': '尾部风险',
        # ── 补充映射 (94个未匹配类别) ──────────────────
        'academic_anomalies': '尾部风险',
        'academic_anomaly': '尾部风险',
        'earnings_season_divergence': '基本面/成长',
        'earnings_anomaly': '基本面/成长',
        'earnings_drift': '基本面/成长',
        'earnings_behavior': '基本面/成长',
        'chip_stratification_ai': '筹码分布',
        'chip_stratification': '筹码分布',
        'chip_structure_ai': '筹码分布',
        'chip_structure': '筹码分布',
        'fundamental_quality_proxy': '基本面/成长',
        'market_microstructure': '微观结构',
        'microstructure_liquidity': '流动性',
        'cf_microstructure': '微观结构',
        'research_driven': '通用',
        'asymmetry': '尾部风险',
        'tail_risk': '尾部风险',
        'higher_moment': '尾部风险',
        'behavioral_v2': '行为金融',
        'alpha_dynamics': '趋势/动量',
        'supply_demand': '资金流',
        'price_efficiency': '反转',
        'regime_adaptive': '通用',
        'momentum': '趋势/动量',
        'sentiment_technical': '行为金融',
        'sentiment_crowding': '行为金融',
        'network': '通用',
        'calendar_anomaly': '行为金融',
        'small_bull_breakthrough': '趋势/动量',
        'low_turnover_mid_freq': '流动性',
        '中报预增分化因子': '基本面/成长',
        '筹码分层ai因子': '筹码分布',
    }
    return mapping.get(cat, '通用')


def _estimate_quality(status: str, icir) -> str:
    """根据状态和 ICIR 估计因子质量"""
    if str(status) == 'candidate':
        return 'high'
    if str(status) == 'reserve':
        try:
            icir_val = float(icir) if not pd.isna(icir) else 0
            if abs(icir_val) >= 0.5:
                return 'high'
            elif abs(icir_val) >= 0.2:
                return 'medium'
        except (ValueError, TypeError):
            pass
        return 'medium'
    return 'low'


def _format_output(name: str, expression: str, rationale: str,
                   icir, quality: str) -> str:
    """格式化为训练输出文本"""
    parts = [
        f"factor_name: {name}",
        f"expression: {expression}",
    ]
    if rationale:
        parts.append(f"rationale: {rationale[:300]}")
    return '\n'.join(parts)


def _find_formula_by_name(df: pd.DataFrame, name: str) -> Optional[str]:
    """在 factor pool 中按名称查找公式"""
    # 尝试 label 匹配
    matches = df[df['label'].str.contains(name, case=False, na=False)]
    if len(matches) > 0:
        formula = matches.iloc[0].get('formula', '')
        if pd.notna(formula) and str(formula).strip():
            return str(formula).strip()

    # 尝试 name 列匹配
    if 'name' in df.columns:
        matches = df[df['name'].str.contains(name, case=False, na=False)]
        if len(matches) > 0:
            formula = matches.iloc[0].get('formula', '')
            if pd.notna(formula) and str(formula).strip():
                return str(formula).strip()

    return None


# ── CLI ───────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Build GRPO SFT dataset')
    parser.add_argument('--output', default='data/grpo_sft_data.json',
                        help='Output JSON path')
    parser.add_argument('--no-v3', action='store_true',
                        help='Exclude V3 base factors')
    args = parser.parse_args()

    pool_path = PROJECT_ROOT / 'data' / 'passed_factor_pool.csv'
    injected_path = PROJECT_ROOT / 'data' / 'injected_factors.json'
    output_path = PROJECT_ROOT / args.output

    if not pool_path.exists():
        print(f"ERROR: {pool_path} not found")
        sys.exit(1)

    result = build_sft_dataset(
        pool_path, injected_path, output_path,
        include_v3=not args.no_v3,
    )

    print(f"Built SFT dataset: {result['stats']['total_samples']} samples")
    print(f"  By source: {result['stats']['by_source']}")
    print(f"  By paradigm: {result['stats']['by_paradigm']}")
    print(f"  By quality: {result['stats']['by_quality']}")
    print(f"  Saved to: {output_path}")
