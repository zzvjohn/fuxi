# -*- coding: utf-8 -*-
"""重放前置检查：锚点存在性 + 负对照候选侦察"""
raw = open(r"E:\quant\research\factor_alchemy\s5_joint_filter.py",
           "r", encoding="utf-8", newline="").read()

# 1) M17 stage2 的采集锚点
ANC = ("                for day in missing[missing > 0].index:\r\n"
       "                    if len(_mark_diag['marking_samples']) < 12:\r\n"
       "                        _mark_diag['marking_samples'].append({'date': str(day.date()), 'missing': int(missing.loc[day])})")
print("M17 ANCHOR count:", raw.count(ANC))
i = raw.find(ANC)
print("  at idx:", i)

# 2) prev_codes 锚点
PC = "            prev_codes = selected_codes"
print("prev_codes anchor count:", raw.count(PC))

# 3) _MISS_CELLS 名字冲突检查
print("_MISS_CELLS in file:", raw.count("_MISS_CELLS"))
print("_S2_PERIODS in file:", raw.count("_S2_PERIODS"))
