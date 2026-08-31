"""
RAG ↔ GA 双向知识链接 (v8.0)
===========================
提供:
  1. RAG → GA: 搜索领域知识, 注入先验 (regime-因子关联, 有效模式)
  2. GA → RAG: JQ验证策略入库, 失败模式总结
"""
import os
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# RAG 调用接口 (如果 zsxq-rag 可用)
_RAG_AVAILABLE = False
try:
    _rag = os.environ.get("RAG_DIR")
    if _rag:
        sys.path.insert(0, _rag)
    from rag_api import recall_for_prompt, ingest as rag_ingest
    _RAG_AVAILABLE = True
except ImportError:
    pass


def query_rag_prior(query: str, k: int = 5) -> str:
    """
    从 RAG 知识库检索领域先验知识
    
    Parameters
    ----------
    query : str
        自然语言查询, 如 "A股小盘股在 bull market 下哪些因子最有效"
    k : int
        返回 top-k 个相关文档
    
    Returns
    -------
    str
        格式化的知识片段, 可直接注入 prompt
    """
    if not _RAG_AVAILABLE:
        return ""
    try:
        ctx = recall_for_prompt(query, k=k)
        return ctx
    except Exception:
        return ""


def get_regime_factor_prior(regime: str) -> str:
    """检索特定 regime 下的因子有效性先验"""
    query = f"A股 {regime} 市场环境下 因子选股 有效因子 ICIR"
    return query_rag_prior(query, k=4)


def ingest_strategy_to_rag(
    strategy_name: str,
    regime: str,
    factors: Dict[str, float],
    jq_sharpe: float,
    jq_maxdd: float,
    jq_return: float,
    notes: str = "",
):
    """
    将 JQ 验证通过的策略反向写入 RAG
    
    Parameters
    ----------
    strategy_name : str
        策略名称, 如 "V2: FA Composite v2"
    regime : str
        有效市场环境
    factors : dict
        {factor_name: weight}
    jq_sharpe, jq_maxdd, jq_return : float
        JQ 回测指标
    notes : str
        策略备注
    """
    if not _RAG_AVAILABLE:
        return

    factor_str = ", ".join(f"{k}({v:.2f})" for k, v in sorted(
        factors.items(), key=lambda x: -abs(x[1]))
    )
    title = f"JQ验证策略: {strategy_name}"
    category = "策略/已验证"
    text = f"""
# {strategy_name}

## 因子组合
{", ".join(f"{k}({v:.3f})" for k, v in sorted(factors.items(), key=lambda x: -abs(x[1])))}

## JQ回测
- 年化收益: {jq_return:+.1%}
- Sharpe: {jq_sharpe:.2f}
- MaxDD: {jq_maxdd:.1%}
- 有效regime: {regime}

## 备注
{notes}
"""
    try:
        rag_ingest(title, category, text)
        print(f"  [RAG] 策略 {strategy_name} 已入库")
    except Exception as e:
        print(f"  [RAG] 策略入库失败: {e}")


def ingest_failure_pattern(
    pattern_name: str,
    description: str,
    jq_evidence: str = "",
):
    """
    将 JQ 验证失败的模式写入 RAG 作为避坑规则
    
    Parameters
    ----------
    pattern_name : str
        失败模式名称, 如 "XGB-learned因子过拟合"
    description : str
        详细描述, 包含原因分析
    jq_evidence : str
        JQ 回测证据
    """
    if not _RAG_AVAILABLE:
        return

    title = f"JQ证伪模式: {pattern_name}"
    category = "避坑规则"
    text = f"""
# {pattern_name}

## 描述
{description}

## JQ证据
{jq_evidence}

## 时间
{datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
    try:
        rag_ingest(title, category, text)
        print(f"  [RAG] 失败模式 {pattern_name} 已入库")
    except Exception as e:
        print(f"  [RAG] 失败模式入库失败: {e}")


def build_ga_initialization_context() -> str:
    """
    为 GA 初始化构建知识上下文
    综合 RAG 中的所有已验证策略、避坑规则、regime-因子先验
    
    Returns
    -------
    str
        结构化知识上下文
    """
    parts = []
    
    # 1. 已验证策略
    strategies = query_rag_prior("JQ验证策略 rank-product 因子组合 高Sharpe", k=5)
    if strategies:
        parts.append(f"## 已验证策略 (JQ)\n{strategies}")
    
    # 2. 避坑规则
    failures = query_rag_prior("JQ证伪 过拟合 因子失败 避坑", k=5)
    if failures:
        parts.append(f"## 已知失败模式\n{failures}")
    
    return "\n\n".join(parts) if parts else ""


if __name__ == "__main__":
    print(f"RAG available: {_RAG_AVAILABLE}")
    ctx = build_ga_initialization_context()
    if ctx:
        print(f"\n=== GA初始化知识上下文 ===\n{ctx[:500]}...")
    else:
        print("No RAG context available")
