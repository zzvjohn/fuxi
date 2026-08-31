# -*- coding: utf-8 -*-
"""
JQ Code Generator — 因子炼金术 → 聚宽策略代码自动生成器

P0.1: Factor Registry — 每个因子绑定本地公式 / JQ等价实现 / 数据需求
P0.2: JQGenerator — 给定 {factor_name: weight} 产出完整 JQ 策略 .py 文件

设计原则:
  - 以 v7.6 均衡 (fa_v76_balanced_jq.py) 为模板骨架 — 唯一在 JQ 上正收益验证过的架构
  - 因子公式逐行对账本地源码 (factors/*.py), 不做手动翻译
  - 生成代码可通过模板变量 {WEIGHTS} / {FACTOR_COMPUTE_BLOCKS} / {DISPATCH_BLOCK} 任意注入
"""

from .registry import JQ_FACTOR_REGISTRY, get_factor_jq_meta, list_available_factors
from .generator import JQGenerator

__all__ = ['JQ_FACTOR_REGISTRY', 'get_factor_jq_meta', 'list_available_factors', 'JQGenerator']
