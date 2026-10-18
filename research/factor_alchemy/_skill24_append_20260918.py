# -*- coding: utf-8 -*-
"""把 §24 追加入 fuxi-dead-wiring-audit SKILL.md，并扩充 description/触发词。
保持 CRLF；写前备份；写后断言裸 LF=0 与关键串存在。"""
import io, os, shutil, datetime

SK = r"C:\Users\Zhong\.workbuddy\skills\fuxi-dead-wiring-audit\SKILL.md"
FRAG = r"e:\quant\research\factor_alchemy\_skill24_fragment_20260918.md"

bak = SK + ".bak_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
shutil.copy2(SK, bak)

txt = open(SK, "rb").read().decode("utf-8")
txt = txt.replace("\r\n", "\n")

frag = open(FRAG, "rb").read().decode("utf-8").replace("\r\n", "\n")

# 1) 触发词扩展
old_trig = "影子读数, 硬编码结论, 合法空态, 读先于写, 误诊, 假修复, 逐单元复核, Edit转LF, 行尾被改写, 吞return, 锚def, factor=收尾。"
new_trig = ("影子读数, 硬编码结论, 合法空态, 读先于写, 误诊, 假修复, 逐单元复核, Edit转LF, 行尾被改写, 吞return, 锚def, factor=收尾,\n"
            "  死通道, 三源字段核对, 对照因子, 严格增量, ctrl因子, uniq, std, 死因子, 塌截面, 事件样本, fillna(0), 单边判据, 漏报方向,\n"
            "  反方向样本, 窗口依赖, lookback, 量纲错配, deferred_premise_flawed, 放大任意性, 跨层轴, rank(axis=1), 逐股Series。")
assert txt.count(old_trig) == 1, "触发词锚点未命中: %d" % txt.count(old_trig)
txt = txt.replace(old_trig, new_trig)

# 2) description 补一句
old_sent = "以及**并行前先测内存**（实例内存 × 并发数 vs 可用内存，并用实测耗时而非估算）。"
new_sent = ("以及**并行前先测内存**（实例内存 × 并发数 vs 可用内存，并用实测耗时而非估算）；\n"
            "  并含**死通道的分层核对**（三源字段集合求差 → 带对照因子的端到端 A/B「逐位不变」证严格增量 → 新因子 eval 看 uniq/std 排死因子 →\n"
            "  事件型因子禁 fillna(0) → 单边判据=漏报 → 影子读数窗口依赖必须落 caliber → 前提量纲错配应 deferred 而非调参 → `.rank(axis=1)` 跨层轴缺陷）。")
assert txt.count(old_sent) == 1, "description 锚点未命中: %d" % txt.count(old_sent)
txt = txt.replace(old_sent, new_sent)

# 3) 追加 §24
assert "## 二十四、" not in txt, "§24 已存在，中止"
if not txt.endswith("\n"):
    txt += "\n"
txt += "\n" + frag.strip("\n") + "\n"

out = txt.replace("\n", "\r\n")
open(SK, "wb").write(out.encode("utf-8"))

# 4) 复核
b = open(SK, "rb").read()
print("backup      =", os.path.basename(bak))
print("bytes       =", len(b))
print("CRLF        =", b.count(b"\r\n"))
print("bareLF      =", b.count(b"\n") - b.count(b"\r\n"))
s = b.decode("utf-8")
for k in ["## 二十四、", "### 24.10", "死通道, 三源字段核对", "对照因子", "uniq", "跨层轴"]:
    print("HAS %-24s = %d" % (k, s.count(k)))
print("OK" if (b.count(b"\n") - b.count(b"\r\n")) == 0 else "FAIL bareLF")
