"""
因子炼金术 (Factor Alchemy) v1.0
==================================
完全独立于 v5b 的因子研究体系。

架构: 5层流水线
  Phase 1: 数据采集 (Tushare hfq + fina_indicator_rich)
  Phase 2: 8大类40+因子计算
  Phase 3: 单因子评估 (ICIR / 十分位 / 双重排序 / 相关性)
  Phase 4: GA因子组合 (染色体=权重向量, 适应度=四维度)
  Phase 5: 虚拟组合周频回测
"""
__version__ = "1.0.0"
