# -*- coding: utf-8 -*-
"""
rag_context.py — RAG 知识注入 G 阶段 (v0.5)
============================================
将本地 RAG 量化知识库的前沿研究自动注入到 LLM 因子生成 prompt 中，
使 G 阶段能够利用积累的前沿因子/策略/失败模式知识。

使用方式:
    from rag_context import retrieve_for_generation
    knowledge = retrieve_for_generation(paradigm="动量", top_k=5)
    llm_gen.receive_context(rag_knowledge=knowledge)

设计原则:
  - 不阻塞主循环：RAG 不可用时静默降级，不影响因子生成
  - 查询改写：将范式名映射为 RAG 友好的检索 query
  - 缓存：同一 query 5 分钟内复用结果
"""
import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

# 缓存
_cache: Dict[str, tuple] = {}  # query → (timestamp, results)
_CACHE_TTL = 300  # 5 分钟


def _get_rag_api():
    """延迟加载 RAG API，避免启动时依赖。"""
    rag_dir = os.environ.get("RAG_DIR", "")
    if rag_dir not in sys.path:
        sys.path.insert(0, rag_dir)
    try:
        from rag_api import recall_for_prompt
        return recall_for_prompt
    except ImportError:
        return None


# 范式→检索关键词映射（桥接 G 阶段术语与 RAG 中文语料）
PARADIGM_QUERY_MAP = {
    "动量": "动量因子 趋势 alpha A股 ICIR",
    "反转": "反转因子 均值回复 短期反转 A股",
    "均值回复": "均值回复 反转因子 超跌反弹 A股",
    "波动": "波动率因子 低波 高波 风险溢价",
    "价量关系": "价量因子 量价关系 换手率 成交额 alpha",
    "流动性": "流动性因子 微观结构 买卖价差 资金流",
    "微观结构": "高频微观结构 订单流 tick 因子 A股",
    "行为金融": "行为金融 投资者情绪 处置效应 过度自信",
    "情绪": "投资者情绪 市场情绪 情绪因子 alpha NLP",
    "基本面": "基本面因子 财务质量 盈利 ROE 估值因子",
    "质量": "质量因子 盈利质量 应计利润 财务健康",
    "价值": "价值因子 低估值 股息率 安全边际",
    "成长": "成长因子 营收增速 盈利增速 PEG",
    "行业轮动": "行业轮动 板块动量 行业配置 alpha A股",
    "北向资金": "北向资金 沪深港通 外资流向 因子 alpha",
    "两融": "融资融券 两融余额 杠杆资金 alpha A股",
    "大宗交易": "大宗交易 折溢价 机构行为 alpha",
    "筹码分布": "筹码分布 筹码结构 持仓集中度 VWAP",
    "高频": "高频因子 tick数据 微观结构 订单簿 alpha",
    "跨资产": "跨资产联动 商品股指 债券股票 相关性 alpha",
    "截面交互": "截面因子 股票关联 网络效应 概念板块",
    "事件驱动": "事件驱动 公告效应 业绩预告 分红 alpha",
    "尾部风险": "尾部风险 极端事件 VaR 崩盘风险 偏度",
}

# 通用前沿检索（不指定范式时用）
DEFAULT_QUERY = "前沿因子 A股量化 最新alpha 因子挖掘 2026"


def retrieve_for_generation(
    paradigm: str = "",
    mab_direction: str = "",
    top_k: int = 5,
    include_failures: bool = True,
) -> Dict:
    """
    从 RAG 检索与当前生成任务相关的前沿知识。

    Args:
        paradigm: 目标范式名(如"动量")
        mab_direction: MAB 方向名(如"趋势__equal_weight_rank")
        top_k: 检索数量
        include_failures: 是否包含失败模式知识

    Returns:
        {"frontier_research": [...], "failure_patterns": [...], "query": str}
    """
    rag_api = _get_rag_api()
    if not rag_api:
        return {"frontier_research": [], "failure_patterns": [], "query": "", "error": "RAG API 不可用"}

    # 构造查询
    query = _build_query(paradigm, mab_direction)

    # 缓存检查
    cache_key = f"{query}:{top_k}:{include_failures}"
    if cache_key in _cache:
        ts, result = _cache[cache_key]
        if time.time() - ts < _CACHE_TTL:
            return result

    results = {"frontier_research": [], "failure_patterns": [], "query": query}

    try:
        # 主查询：前沿研究
        raw = rag_api(query, k=min(top_k + 2, 8))
        if raw:
            # recall_for_prompt 返回格式化文本字符串，用【片段N】切分
            if isinstance(raw, str):
                chunks = _parse_recall_text(raw)
            elif isinstance(raw, list):
                chunks = []
                for item in raw:
                    if isinstance(item, dict):
                        chunks.append(item.get("text", str(item)))
                    elif isinstance(item, str):
                        chunks.append(item)
            else:
                chunks = []

            for chunk in chunks:
                if not chunk or len(str(chunk)) < 20:
                    continue
                chunk_text = str(chunk)[:500]
                if _is_failure_pattern(chunk_text):
                    results["failure_patterns"].append(chunk_text)
                else:
                    results["frontier_research"].append(chunk_text)

        # 如果包含失败模式，额外检索一次
        if include_failures and len(results["failure_patterns"]) < 2:
            fail_query = f"A股 量化策略 失败 过拟合 因子衰减 {paradigm}"
            try:
                raw2 = rag_api(fail_query, k=3)
                if raw2:
                    for item in raw2:
                        chunk = item.get("text", str(item)) if isinstance(item, dict) else str(item)
                        if len(chunk) > 20 and _is_failure_pattern(chunk):
                            results["failure_patterns"].append(chunk[:500])
            except Exception:
                pass

    except Exception as e:
        results["error"] = str(e)

    # 写入缓存
    _cache[cache_key] = (time.time(), results)
    return results


def _build_query(paradigm: str, mab_direction: str) -> str:
    """构造 RAG 检索 query。"""
    parts = []

    # 范式映射
    if paradigm:
        for key, mapped in PARADIGM_QUERY_MAP.items():
            if key in paradigm:
                parts.append(mapped)
                break
    if not parts and paradigm:
        parts.append(f"{paradigm} 因子 alpha A股")

    # MAB 方向提取关键词
    if mab_direction:
        direction_keywords = mab_direction.replace("__", " ").replace("_", " ")
        parts.append(direction_keywords)

    if not parts:
        parts.append(DEFAULT_QUERY)

    # 加限定词提高精度
    query = " ".join(parts[:2])
    return f"{query} 2026 前沿 策略"


def _is_failure_pattern(text: str) -> bool:
    """判断检索到的文本是否为失败模式/警告类。"""
    failure_keywords = [
        "失败", "衰减", "拥挤", "过拟合", "失效", "崩溃", "亏损",
        "腰斩", "恶化", "泡沫", "陷阱", "禁止", "淘汰",
        "crowding", "decay", "failure", "overfit", "collapse",
    ]
    text_lower = text.lower()[:300]
    return any(kw in text_lower for kw in failure_keywords)


def _parse_recall_text(raw_text: str) -> List[str]:
    """
    解析 recall_for_prompt 返回的格式化文本。
    格式: 【片段N｜类别｜标题】 ... 原文: url
    按【片段N】分界切分，清理标题前缀和 URL 尾巴。
    """
    import re
    chunks = []
    # 按【片段 开头分割
    parts = re.split(r'(?:^|\n)\s*【片段\d*[｜|]', raw_text)
    for part in parts:
        part = part.strip()
        if not part or "以下是知识库" in part:
            continue
        # 清理: 去掉 "类别｜标题】" 行首
        part = re.sub(r'^[^】\n]*】\s*', '', part)
        # 去掉 "原文: https://..." 行
        part = re.sub(r'\n?\s*原文:\s*https?://\S+', '', part)
        # 合并多余空行
        part = re.sub(r'\n{3,}', '\n\n', part)
        part = part.strip()
        if len(part) >= 30:
            chunks.append(part)
    return chunks


def format_for_prompt(rag_result: Dict) -> str:
    """
    将 RAG 检索结果格式化为可注入 LLM prompt 的文本。

    返回格式:
        ## 🔬 RAG 知识库: 相关前沿研究
        - [因子研究前沿] ...
        - [ML/算法进展] ...

        ## ⚡ RAG 知识库: 失败模式警示
        - ...
    """
    parts = []

    frontier = rag_result.get("frontier_research", [])
    if frontier:
        parts.append("## 🔬 RAG 知识库: 相关前沿研究")
        parts.append(f"以下是知识库中与当前探索方向相关的最新研究 ({rag_result.get('query', '')}):")
        parts.append("")
        for i, chunk in enumerate(frontier[:5]):
            # 截断过长文本
            text = chunk[:400]
            if len(chunk) > 400:
                text += "..."
            parts.append(f"{i+1}. {text}")
            parts.append("")
        parts.append("请在设计因子时参考上述前沿思路，但必须保持公式简洁（≤15节点）。")
        parts.append("")

    failures = rag_result.get("failure_patterns", [])
    if failures:
        parts.append("## ⚡ RAG 知识库: 失败模式警示")
        parts.append("以下是已被证伪或需要避开的方向:")
        parts.append("")
        for i, chunk in enumerate(failures[:3]):
            text = chunk[:300]
            if len(chunk) > 300:
                text += "..."
            parts.append(f"- ❌ {text}")
        parts.append("")

    return "\n".join(parts)


def rag_status() -> Dict:
    """返回 RAG 连接状态（供诊断用）。"""
    rag_api = _get_rag_api()
    return {
        "available": rag_api is not None,
        "cache_size": len(_cache),
        "paradigm_map_entries": len(PARADIGM_QUERY_MAP),
    }
