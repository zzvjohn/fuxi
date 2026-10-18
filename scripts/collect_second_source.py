# -*- coding: utf-8 -*-
"""独立第二源采集统一入口 (P-20261016-second-source, 第 2 级)。

用法:
  python scripts/collect_second_source.py              # 申万 + 中证 全跑
  python scripts/collect_second_source.py shenwan      # 仅申万
  python scripts/collect_second_source.py csindex      # 仅中证

产出累积框架 (data/fuxi_v1/second_source/):
  shenwan/{YYYYMMDD}/  SW2021 一级行业分类 + 成分股映射 + MANIFEST
  csindex/{YYYYMMDD}/  中证全指成分股 + 自由流通权重 + MANIFEST
  各源 index.json       累积索引 (快照时间序列, 供 B04.1 cross-check)
"""
import sys
import time


def main():
    args = sys.argv[1:]
    only = args[0] if args else None

    import collect_second_source_shenwan as sw
    import collect_second_source_csindex as csi

    t0 = time.time()
    if only in (None, "shenwan"):
        sw.run()
    if only in (None, "csindex"):
        csi.run()
    print("\n独立第二源采集完成, 用时 %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
