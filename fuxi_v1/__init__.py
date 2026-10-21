"""Independent Fuxi research kernel. No legacy runtime or market IO at import."""

# v2.0 产品定版（2026-10-18）：kernel 语义版本随之定为 2.0.0。
# v2.1 功能增量（2026-10-20）：慢循环落地 + E1 单调性/适用性/regime/市值分段 + 预算提速。
# 注意与 data/version_registry.json 的 v0.7.x 工程批次账本是两个坐标系：
# v2.0 是产品版本，v0.7.x 是内部逐批编号；详见 ARCHITECTURE.md 第 1.3 节。
__version__ = "2.2.0"
