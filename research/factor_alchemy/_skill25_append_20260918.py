# -*- coding: utf-8 -*-
"""把 §25 追入 fuxi-dead-wiring-audit SKILL.md（CRLF 保真, 写前备份, 写后断言）。"""
import io, os, shutil, datetime

SK   = r"C:\Users\Zhong\.workbuddy\skills\fuxi-dead-wiring-audit\SKILL.md"
FRAG = r"e:\quant\research\factor_alchemy\_skill25_fragment_20260918.md"

bak = SK + ".bak_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
shutil.copy2(SK, bak)

txt = open(SK, "rb").read().decode("utf-8").replace("\r\n", "\n")
frag = open(FRAG, "rb").read().decode("utf-8").replace("\r\n", "\n")

# 1) 触发词
old_trig = "反方向样本, 窗口依赖, lookback, 量纲错配, deferred_premise_flawed, 放大任意性, 跨层轴, rank(axis=1), 逐股Series。"
new_trig = ("反方向样本, 窗口依赖, lookback, 量纲错配, deferred_premise_flawed, 放大任意性, 跨层轴, rank(axis=1), 逐股Series,\n"
            "  白名单序列化, to_dict白名单, 键频普查, 1/N签名, 存不住, 被抹掉, 标记失效, 重放, 计数器虚增, 假修复, 误诊顺序。")
assert txt.count(old_trig) == 1, "触发词锚点 %d" % txt.count(old_trig)
txt = txt.replace(old_trig, new_trig)

# 2) description 补一句
old_sent = "事件型因子禁 fillna(0) → 单边判据=漏报 → 影子读数窗口依赖必须落 caliber → 前提量纲错配应 deferred 而非调参 → `.rank(axis=1)` 跨层轴缺陷）。"
new_sent = ("事件型因子禁 fillna(0) → 单边判据=漏报 → 影子读数窗口依赖必须落 caliber → 前提量纲错配应 deferred 而非调参 → `.rank(axis=1)` 跨层轴缺陷）；\n"
            "  并含**「白名单序列化」类静默抹除**（dataclass `to_dict()` 白名单收缩 ⇒ 外部脚本直写的键存活不过下一次 `_save()`；\n"
            "  用**键频普查的「1/N 签名」**定性、内存往返零副作用复现、扩 schema 必须 `to_dict`+`from_dict` 双向补、\n"
            "  「顺序」类症状先普查否证再定修法以防假修复、标记失效导致重放使 `+= 1` 计数器虚增）。")
assert txt.count(old_sent) == 1, "description 锚点 %d" % txt.count(old_sent)
txt = txt.replace(old_sent, new_sent)

# 3) 追加 §25
assert "## 二十五、" not in txt, "§25 已存在，中止"
if not txt.endswith("\n"):
    txt += "\n"
txt += "\n" + frag.strip("\n") + "\n"

open(SK, "wb").write(txt.replace("\n", "\r\n").encode("utf-8"))

b = open(SK, "rb").read()
s = b.decode("utf-8")
print("backup  =", os.path.basename(bak))
print("bytes   = %d (was 90657)" % len(b))
print("CRLF    =", b.count(b"\r\n"), "| bareLF =", b.count(b"\n") - b.count(b"\r\n"))
for k in ["## 二十五、", "### 25.6", "1/N签名", "to_dict白名单"]:
    print("HAS %-16s = %d" % (k, s.count(k)))
print("OK" if (b.count(b"\n") - b.count(b"\r\n")) == 0 else "FAIL bareLF")
