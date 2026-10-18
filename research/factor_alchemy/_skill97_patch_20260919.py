# -*- coding: utf-8 -*-
"""修 skill `fuxi-s5-submission-and-topn-attribution` §9.7:
「前置短路白名单」实测只列 2/7 类 —— 它本身就是它自己警告的那种字面白名单。

证据 (2026-09-19, 送 31 条 axis=1 因子时冒烟即暴露):
  `co_movement_intensity` 返回 `🧹 常数因子拦截: 有效值仅 1 个离散水平` ⇒ 两腿超额
  同时恰为 0.0。而 §9.7 的 PRE_EVAL_PREFIXES 只有 2 条 ⇒ **未分类** ⇒ 按 §9.7 自身
  的 `C2h` 判据应判 FAIL/人查。逐条枚举 validate_factor(3071→3235) 得 **7 条**出口。

本补丁: 把清单补全为从实现穷举的 7 条 + 加「枚举命令」替代靠记忆抄 + 记录这次更正。
"""
from pathlib import Path

SK = Path(r'C:/Users/Zhong/.workbuddy/skills/fuxi-s5-submission-and-topn-attribution/SKILL.md')

OLD = '''**改法（锚在实现）**：

```python
# ① 已知前置短路类**白名单**（穷举 + 注释来源），而不是一条前缀
PRE_EVAL_PREFIXES = (
    "因子公式评估无有效数据",   # 求值通道无数据 ⇒ _empty
    "🧹 公式合理性拦截",        # 合理性门: **求值之前**拦下
)

def never_eval(r):
    s = str(r.get("new_reason") or "")
    for p in PRE_EVAL_PREFIXES:
        if s.startswith(p):
            return True, "reason::" + p
    # ② **数值签名**: 基准在该年为真实正值 ⇒ 被真实评估的组合不可能**恰好**落在 0.0
    if r.get("new_excess_25") == 0.0 and r.get("new_excess_26") == 0.0:
        return True, "numeric_signature_only"      # ← 单独命中 = 未分类短路, 判 FAIL
    return False, ""
```
'''

NEW = '''**改法（锚在实现）**：

🔴 **2026-09-19 更正 —— 这份白名单自己就是「字面白名单」**：它最初只列 **2** 类，
而实现里 `S5JointFilter.validate_factor`（`s5_joint_filter.py` 3071→3235）在
`_backtest_and_metrics` 之前共有 **7 条** `return result` 出口 ⇒ **漏了 5/7**。
漏掉的 `🧹 常数因子拦截` / `🧹 价格水平代理拦截` 恰恰**最容易**命中真实候选
（稀疏/常数因子、价格水平类因子），而漏判的后果正是本节的 `C2h` 判 FAIL：
**任何一条走短路的条目，两年超额都由构造留在 `0.0`** ⇒ 对位移/散布**零信息量**。

```python
# ① 求值前 / 指标前的短路类 —— 从 s5_joint_filter.validate_factor **穷举**
#    (2026-09-19; 每一类都会把 excess_2025/2026 留在默认 0.0)
PRE_METRICS_REASONS = (
    "价格数据不可用",            # L3104  数据未就绪
    "数据加载失败",              # L3111  _load_data 抛错
    "🧹 公式合理性拦截",         # L3123  **求值之前**拦下 (含"过度简化")
    "因子公式评估无有效数据",     # L3154  求值返回 None/空
    "因子值全为 NaN",            # L3161a _value_guards
    "🧹 常数因子拦截",           # L3161b _value_guards: 唯一值 ≤ 5
    "🧹 价格水平代理拦截",        # L3161c _value_guards: 与 close rank 相关 > 0.95
)

def never_eval(r):
    s = str(r.get("new_reason") or "")
    for p in PRE_METRICS_REASONS:
        if s.startswith(p):
            return True, "reason::" + p            # 有前置原因 ⇒ 排除可证
    # ② **数值签名**: 该年基准为真实正值 ⇒ 被真实评估的组合不可能**恰好**落在 0.0
    if r.get("new_excess_25") == 0.0 and r.get("new_excess_26") == 0.0:
        return True, "numeric_signature_only"      # ← 单独命中 = 未分类短路, 判 FAIL
    return False, ""
```

🔴 **别靠记忆维护这份清单 —— 每次改动后用「枚举命令」重生成**（锚在实现）：

```python
import re
from pathlib import Path
lines = Path("research/factor_alchemy/s5_joint_filter.py").read_text(
    encoding="utf-8", newline="").split("\\n")
lo = next(i for i, l in enumerate(lines) if re.match(r"    def validate_factor\\(", l))
hi = next(i for i in range(lo + 1, len(lines)) if re.match(r"    def ", lines[i]))
metrics = next(i for i in range(lo, hi) if "_backtest_and_metrics(" in lines[i])
for i in range(lo, metrics):            # metrics 之前的每个 "return result" 都要登记
    if "return result" in lines[i]:
        print(i + 1, lines[i - 1].strip()[:80])
```

（本清单是「**求值前/指标前**」概念；注意它与「**求值失败**」不是一回事 ——
`🧹 常数因子拦截` 其实**已经求过值**了，只是没过 `_value_guards` 就返回。
但对「位移/散布」统计而言二者等价：**读数由构造决定，不是测量**。）
'''

raw = SK.read_bytes()
crlf = raw.count(b'\r\n')
bare = raw.count(b'\n') - crlf
nl = '\r\n' if (crlf > 0 and bare == 0) else '\n'
print(f'SKILL.md: bytes={len(raw)} CRLF={crlf} bareLF={bare} -> 行尾 {nl!r}')

t = raw.decode('utf-8')
# 🔴 该技能是 CRLF 文件: 直接对 bytes 解码后, 内容里是 '\r\n' ⇒ 用 '\n' 写的锚点
#    永远匹配不上 (实测命中 0 次)。故改用**通用换行**读入, 在 '\n' 空间做替换。
with open(SK, 'r', encoding='utf-8') as fh:
    t = fh.read()
n = t.count(OLD)
print('锚点命中(旧块)', n, '次')
if NEW in t:
    print('  [SKIP] 目标块已存在 ⇒ 幂等跳过')
    t2 = t
else:
    assert n == 1, f'🔴 锚点不唯一 ({n})'
    t2 = t.replace(OLD, NEW, 1)
    with open(SK, 'w', encoding='utf-8', newline=nl) as fh:
        fh.write(t2)

# ── 回读 ──
b = SK.read_bytes()
c2 = b.count(b'\r\n')
ba2 = b.count(b'\n') - c2
with open(SK, 'r', encoding='utf-8') as fh:
    t3 = fh.read()
print(f'回读: bytes={len(b)} CRLF={c2} (原 {crlf}) 裸LF={ba2} (原 {bare})')
# 🔴 不变量 = 「裸 LF 计数不变」(新增行须全部转成 CRLF) + 行尾风格不变。
#    写死 `c2 == crlf` 是**假 FAIL** —— 补丁新增 32 行, CRLF 计数必然增加。
assert ba2 == bare, '🔴 引入了新的裸 LF'
assert (c2 > 0 or b'\r' not in b), '🔴 行尾风格被改变'
assert t3.count(OLD) == 0, '🔴 旧块仍在'
for k in ['PRE_METRICS_REASONS', '价格数据不可用', '因子值全为 NaN',
          '🧹 常数因子拦截', '🧹 价格水平代理拦截', '枚举命令', '不是一回事']:
    assert k in t3, f'🔴 缺锚点 {k}'
print('PASS: 7 类清单 + 枚举命令 + 更正注记 均已落盘')
print('  新增 bytes =', len(b) - len(raw))
