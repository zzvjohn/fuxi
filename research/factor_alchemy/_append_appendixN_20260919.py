# -*- coding: utf-8 -*-
"""追加 附录 N 到 `_fuxi_landing_details.md` (2026-09-19 批次)。
行尾: 按文件实测 —— 该文件为 CRLF (1414 CRLF / 0 裸 LF), 故 newline='\\r\\n'。
"""
from pathlib import Path

D = Path(r'e:/quant/.workbuddy/memory/_fuxi_landing_details.md')

APPENDIX = '''

## 附录 N — 2026-09-19 批次: `_freq_audit` 同族修复 + MEMORY v45 精简 + 短路清单 7 类

### N.1 `_freq_audit` 同族缺陷: 三方案对比与裁决

**问题**: `freq_audit.py:147` 把 `e["_freq_audit"]` 写进**原始 dict**, 再 `json.dump(indent=1)` 落盘。
该键不在 `MiningTrajectory` 白名单 ⇒ 下一轮任意 `_save()` 整对象重建即**静默抹除**。
实测: 跑中 1 条 → 跑后 **0 条**; 同轮 `lessons` 键频 0 → 2302 (证明 `_save()` 真的跑过)。
全库 grep: 该键**零读方** ⇒ 后果 = 归属证据丢失, **不影响裁决** (`natural_freq` 在白名单内)。

**关键事实**（决定了裁决方向）:

| 事实 | 读数 | 出处 |
|---|---|---|
| `_freq_audit` 写方 | 仅 `freq_audit.py:147`, 直写原始 dict | grep |
| `_freq_audit` 读方 | **零** (只有写方 + 验证脚本) | grep |
| 活跃的「直连 JSON 写方」 | 仅 `freq_audit.py` (另有 3 个一次性历史脚本) | grep |
| `lessons` 元素形状 | `{"lesson": str, "timestamp": str}` **dict** | `log_lesson()` L699 |
| trajectory 级 `lessons` 读方 | **零生产读方** (`experience_memory` 的 `lessons` 是 `SuccessPattern` 的 **str 列表**, 另一个对象) | grep |
| 抹除面 | 条目级非白名单键 + **根级**非白名单键 (`_save()` 根 dict 只有 4 键) | `_save()` L313 |
| 写入频率 | **至多一次/条** (幂 등: freq 已 weekly 即跳过) | `freq_audit.py:133` |

**三方案对比**:

| 方案 | 做法 | 判定 |
|---|---|---|
| A 白名单三处同步 | dataclass + `to_dict` + `from_dict` 加 `_freq_audit` | ✅ **采用** —— 与 9-18 `d_plus_*` 同法, 1 字段, 零裁决影响 |
| B 改由 `lessons` 承载 | 把修正归属写进 `lessons` 列表 | ❌ **否决** —— 见下三条理由 |
| C `to_dict` 保留未知键 | 加 `extra_fields` 兜住全部非白名单键 | ⚠️ 暂不采用 —— 覆盖全类但会让**陈旧/垃圾键永久滞留**, 并使「键频 1/N = 被抹签名」这个诊断失去仪器; 需完整 A/B |
| D 写方统一 | 让 `freq_audit` 走 `TrajectoryLogger._save()` | ⚠️ 后续待裁决 —— 会整份 2302 条重新序列化 (indent 1→2, 全文件 diff), 需 A/B |

**为何否决 B（用户提问的候选）**:

1. **不解决「类」, 只换载体**。塌陷点是「`to_dict()` 白名单全量重写 ⇒ 非白名单键被抹」。
   把 `_freq_audit` 塞进 `lessons` 只让**这一个键**恰好落在白名单里;
   下一个直连写方的新字段照样被抹。**本轮即是实证**: `d_plus_*` 修好后
   `_freq_audit` 立刻补位成为新的被抹项 ⇒ 按字段搬进白名单是**打地鼠**。
2. **结构化信息降级 + 违反「判据锚在实现」**。`_freq_audit` 是 **str**
   (含 old→new freq / 锚点名 / 提案号), `lessons` 元素是 **dict**
   ⇒ 搬过去要么把结构化信息压成一句话塞进 `lesson` 字段, 要么新造 shape;
   无论哪种, 读侧都只能**字符串扫描** —— 而 §9.7 已因「锚消息串前缀」造过**假 FAIL**。
3. **通道污染 + 新的膨胀风险**。`lessons` 是**教训/知识**通道, `_freq_audit` 是**归属留痕**。
   今天 trajectory 级 `lessons` 恰好无生产读方 ⇒ **「暂时」无害**, 不是「无害」
   (D+ 蒸馏正是会读该通道的地方)。且 `freq_audit` 每轮都跑, 一旦锚点重复命中,
   `lessons` 会**逐轮膨胀** —— `_append_dplus_rc_20260918.py` 已就 `attempts[].lessons`
   膨胀立过案（数据没坏, 权重被灌水）。

**验证 (`_verify_freq_audit_fix_20260919.py`, 11/11 PASS)**:

在 `data/trajectory_log.json` 的**临时副本**上跑完整链路（写方 → `_save()` → 重载）。
🔴 **含对照组** —— 只证「`_freq_audit` 存活」**不足以**证明修复（也可能是我把白名单机制
整个停掉了, 那是 fail-open）。故同时注入非白名单键 `_zz_control_key`, 断言**它仍被抹除**:

```
处理组: _freq_audit 穿越 log_lesson()→_save() **存活**  ✅
对照组: _zz_control_key = None               **仍被抹除**  ✅
```

其余读数: 写方 `n_fixed=2`（含 1 条**生产数据里本就存在**的未修正条目
`traj_2561b7d3c712`）; `from_dict` 正确恢复; 二次重载一致; 全库键频 = 0 + 2。
落地 `P-20260919-001` / **v0.7.43**。

🔴 **未做整轮 ralph 端到端**, 理由: 该路径**不可能被触发** (`freq_audit` 幂等, 生产库已无
可修正条目) ⇒ 跑整轮只会得到「0 条修正」, 对本改动**零信息量**。真正的端到端由**下一轮**
自然给出 (见 N.1 末的遗留条目)。

**遗留**: `traj_2561b7d3c712` 处于「可铁证修正但尚未修正」状态 (推测为轮内新建、
未被轮首审计覆盖) ⇒ 其 `_freq_audit` 将在下一个 ralph 轮次写入, 需回读确认存活。

### N.2 🔴 `validate_factor` 的「pre-metrics 短路」共 **7 类**（从实现枚举）

起因: 2026-09-19 送 31 条 axis=1 因子做**冒烟**时, `co_movement_intensity` 返回
`🧹 常数因子拦截: 有效值仅 1 个离散水平` ⇒ 两腿超额**同时恰为 0.0**。
而 S5 技能 §9.7 的 `PRE_EVAL_PREFIXES` 只列了 **2** 类 ⇒ **未分类** ⇒ 按 §9.7 自身的
`C2h` 判据应判 FAIL / 人查。逐条枚举 `validate_factor` (s5_joint_filter.py 3071→3235):

| # | 短路出口 | 行 | 是否已求值 | excess |
|---|---|---|---|---|
| 1 | `价格数据不可用` | L3104 | 否 | 0 (构造) |
| 2 | `数据加载失败: {e}` | L3111 | 否 | 0 (构造) |
| 3 | `🧹 公式合理性拦截: …` | L3123 | 否 | 0 (构造) |
| 4 | `因子公式评估无有效数据` | L3154 | 求值但空 | 0 (构造) |
| 5 | `因子值全为 NaN` | L3161a | **是** | 0 (构造) |
| 6 | `🧹 常数因子拦截: 唯一值 ≤5` | L3161b | **是** | 0 (构造) |
| 7 | `🧹 价格水平代理拦截: 与 close 相关 >0.95` | L3161c | **是** | 0 (构造) |

⇒ **技能 §9.7 的白名单自己就是它警告的那种「字面白名单」**（漏 5/7）。
漏掉的第 6/7 类恰恰**最容易命中真实候选**。已在技能里更正为
`PRE_METRICS_REASONS`（7 条）+ 附「枚举命令」替代靠记忆维护。

概念澄清: 第 5~7 类**已经求过值**了（只是没过 `_value_guards` 就返回）,
但对「位移/散布」统计而言与「求值前」**等价** —— 读数由构造决定, 不是测量。

### N.3 MEMORY.md v45 精简台账（2995 → 2955, 余量 5 → 45）

**新增 4 条**（均为本轮亲历/实测）:
1. 短路清单**禁锚前缀串** + 「实测 7 类, 旧白名单只列 2」+ 锚数值签名/原因全集/排除台账可证性;
2. **同族须穷举所有直连 JSON 写方**（修一个不算修）;
3. 判「走没走 `_save()`」看 `lessons` 键频 + **缩进**（`_save`=2 / `freq_audit`=1）, **不看 mtime**;
4. **写侧 `newline=''` ⇒ 静默转 LF**（CRLF 须 `newline='\\r\\n'`）。

**折叠进积压项**: `--caliber legacy` 只覆盖 S5 内部 6 键 ⇒ **非端到端**, 端到端须 `legacy_e2e`。

**移出 1 条**: 「多通道入库」（已在本文档「### 多通道因子入库判据 (09-16 立)」有专属小节;
且它导致**结论错**而非**数字静默错**, 不满足「违反即静默出错」的收录门槛）。

**压缩**: 「数据/产物五坑」④⑤ 的公式截断/基准错配细节（→ 技能 §9.6/§9.5/§9.4）。

行尾: 保持 **CRLF**(29 CRLF / 0 裸 LF); 落盘脚本含 50 项锚点校验 + 逐行往返断言。

### N.4 🔴 写侧 `newline=''` 静默转 LF（本轮亲身踩到）

用 `io.open(PROP, 'w', encoding='utf-8', newline='')` 写提案账本 ⇒ 整文件
**CRLF 6209 → 纯 LF**。机理: `newline=''` 在**写侧不做任何翻译**, 而 `json.dumps`
只产出 `\\n`。**要写 CRLF 必须 `newline='\\r\\n'`**。
同族于已记的「Edit 会整文件转 LF」, 但机理不同（那个是工具行为, 这个是自己的 I/O 参数）。
⇒ 每个 `_add_p*.py` 模板都需要注意（`_add_p20260917_028.py` 就是 `newline=""`, 其
docstring 还写着「文件为 LF」—— 而实测该账本是 CRLF）。

### N.5 行尾「不变量」的两类假 FAIL（本轮各犯一次）

补丁会**新增行** ⇒ 行尾统计量**必然变化**。两次都因写死等式而造出假 FAIL:

| 场景 | 错误断言 | 为何是假 FAIL | 正确不变量 |
|---|---|---|---|
| `_patch_freq_audit` | `bareLF 与写前相等` | 新增 9 行 ⇒ 裸 LF 884→893 | **CRLF 计数不变 + 不引入 `\\r`** |
| `_skill97_patch` | `CRLF 计数不变` | 新增 32 行 ⇒ CRLF 703→735 | **裸 LF **计数不变** + 行尾风格不变** |

统一表述: **「新增的行必须用原风格」** ⇒ 若文件是 CRLF, 断言「裸 LF 不增加」;
若文件是纯 LF, 断言「不引入 `\\r`」。**不要断言任何一种计数相等**。
'''

raw = D.read_bytes()
crlf, bare = raw.count(b'\r\n'), raw.count(b'\n') - raw.count(b'\r\n')
print(f'details: bytes={len(raw)} CRLF={crlf} 裸LF={bare}')
assert bare == 0, '🔴 入参已有裸 LF'
with open(D, 'r', encoding='utf-8') as fh:
    t = fh.read()
assert '## 附录 N' not in t, '🔴 附录 N 已存在'
t2 = t.rstrip('\n') + APPENDIX + '\n'
with open(D, 'w', encoding='utf-8', newline='\r\n') as fh:
    fh.write(t2)

b = D.read_bytes()
c2, ba2 = b.count(b'\r\n'), b.count(b'\n') - b.count(b'\r\n')
with open(D, 'r', encoding='utf-8') as fh:
    t3 = fh.read()
print(f'回读: bytes={len(b)} (+{len(b)-len(raw)}) CRLF={c2} 裸LF={ba2}')
assert ba2 == 0, '🔴 引入裸 LF'
for k in ['## 附录 N', 'N.1', 'N.2', 'N.3', 'N.4', 'N.5', '_zz_control_key',
          'traj_2561b7d3c712', 'PRE_METRICS_REASONS', '打地鼠']:
    assert k in t3, f'🔴 缺锚点 {k}'
print('PASS: 附录 N 已归档 (N.1~N.5)')
