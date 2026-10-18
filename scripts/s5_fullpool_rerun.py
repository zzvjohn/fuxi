# -*- coding: utf-8 -*-
"""
P-20260909-005 补充: S5 全量池重裁历史被拒因子
====================================================
背景: S5 已切换全量池 (5763 股, 基准 000985 中证全指)。历史被 Top-200 大盘池
      拒绝的因子可能存在池错杀, 需全量池重裁。

候选来源 (三源合并去重):
  A. pass_to_jq_bridge_result.json — 桥接首跑 17 个 S5 被拒 (公式从池 CSV 取)
  B. s135_candidate_check_result.json — 09-08 s135 检查 26 个 S5 被拒
  C. trajectory_log.json — Ralph 轨迹中 S5 真实回测被拒 (excess 非 null)

用法:
    python scripts/s5_fullpool_rerun.py                    # 重裁 (代码默认口径)
    python scripts/s5_fullpool_rerun.py --dry-run          # 只盘点不跑
    python scripts/s5_fullpool_rerun.py --caliber legacy   # 修正前口径 (**S5 内部**回归)
    python scripts/s5_fullpool_rerun.py --caliber legacy_e2e  # 端到端零回归 (S5+上游)
    python scripts/s5_fullpool_rerun.py --caliber p0       # 修正后推荐口径

--caliber 用于**全池翻转率复跑**: 同一候选集跑两遍口径再 diff, 量化
「翻默认会改写多少历史裁决」。结果按口径分文件落盘, 互不覆盖:
    s5_fullpool_rerun_result.json          (default = 代码默认)
    s5_fullpool_rerun_result_legacy.json   (= v0.7.23 之前读数, 应逐位复现)
    s5_fullpool_rerun_result_p0.json
"""
import sys
import os
import re
import json
import time
import argparse
from pathlib import Path

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / "research" / "factor_alchemy"))

DATA_DIR = BASE / "data"
POOL_CSV = DATA_DIR / "passed_factor_pool.csv"
BRIDGE_JSON = DATA_DIR / "pass_to_jq_bridge_result.json"
S135_JSON = DATA_DIR / "s135_candidate_check_result.json"
TRAJ_JSON = DATA_DIR / "trajectory_log.json"
OUT_JSON = DATA_DIR / "s5_fullpool_rerun_result.json"

# ── S5 口径 (v0.7.26 加): 供全池翻转率复跑做「同集双跑 diff」 ──
#   default = 不动 env, 用代码默认值 (v0.7.26 起第 1 批三项已翻默认)
#   legacy  = 显式设回修正前值 ⇒ 应逐位复现 v0.7.23 之前的读数
#   p0      = 修正后推荐口径: 第 1 批三项 + 第 2 批两项
#             (TRADABLE_GATE 与 ADJ_PRICE **必须捆绑**, 单独开 ADJ_PRICE 会
#              制造假 PASS —— 它恢复的正是「建仓日封涨停/买不进」那段收益);
#             FIRSTDAY 保持 off (已被 ADJ_PRICE 取代, 见 s5_joint_filter.py 硬约束)
# ── S-1 (P-20260918-019) 开关分档显式化 —— 让命名不再撒谎 ──
# 🔴 语义问题 (外部审计 §8 S-1 实测): 原名 `ALL_SWITCH_ENVS` 暗示"全部开关", 但实际
#   只覆盖 **S5 内部** 6 键。CHANGELOG 里另有 3 个**已翻默认**的开关属于**上游候选项
#   口径** —— 它们决定"哪些候选被送进 S5", 却不在本清单里:
#     FUXI_ELIGIBLE_REQUIRE_S5   on   (原 off)     multi_stage_validator.py:1089
#     FUXI_MF_MISSING            nan  (原 zero)    scripts/data_loader_ext.py:157
#     FUXI_MAB_PARADIGM_BROADCAST on  (原 shadow)  ralph_loop.py:4475
#   ⇒ `--caliber legacy` 是 **S5 内部回归**, 不是**端到端回归**: 喂进 S5 的那批候选
#     本身已经不是当年的那批了 ⇒ 零回归基线在**静默**漂移。
#   修法: 正典名 S5_SWITCH_ENVS; 上游清单单列; 新增 legacy_e2e 档 (两套都挂回)。
S5_SWITCH_ENVS = ["FUXI_S5_TRADABLE_GATE", "FUXI_S5_FIRSTDAY", "FUXI_S5_TIE_RULE",
                  "FUXI_S5_COST_BPS", "FUXI_S5_MIN_COVERAGE", "FUXI_S5_ADJ_PRICE"]

# 上游候选口径键: **不在**本脚本的 pop 范围内 (见 _CALIBER_POP_EXTRA)。
UPSTREAM_SWITCH_ENVS = ["FUXI_ELIGIBLE_REQUIRE_S5", "FUXI_MF_MISSING",
                        "FUXI_MAB_PARADIGM_BROADCAST"]
# 翻默认前的旧值 (回退语义自 v0.7.2x 起已变: 「不设 env = 修正前」**不再成立**
# ⇒ 必须显式设回旧值, 否则 legacy_e2e 挂不回去)
UPSTREAM_LEGACY = {"FUXI_ELIGIBLE_REQUIRE_S5": "off",        # P-20260916-011 翻 on
                   "FUXI_MF_MISSING": "zero",                # P-20260916-012 翻 nan
                   "FUXI_MAB_PARADIGM_BROADCAST": "shadow"}  # P-20260915-015 翻 on

# 🔴 弃用别名 (向后兼容): 既有静态验证脚本按此名做 `ast.literal_eval`
#   (_verify_p0917_landing.py / _verify_auxlag_declare_20260918.py / _patch_*).
#   刻意写成**重复字面量**而非 `= S5_SWITCH_ENVS`: 后者会让 ast.literal_eval 取不到
#   list ⇒ 那些静态断言**静默失效** (fail-open, 正是 memory 里"降级须留痕"的禁项)。
#   下面的 assert 保证两处不漂移 (漂移 ⇒ import 期 loud 失败, 不静默)。
ALL_SWITCH_ENVS = ["FUXI_S5_TRADABLE_GATE", "FUXI_S5_FIRSTDAY", "FUXI_S5_TIE_RULE",
                   "FUXI_S5_COST_BPS", "FUXI_S5_MIN_COVERAGE", "FUXI_S5_ADJ_PRICE"]
assert ALL_SWITCH_ENVS == S5_SWITCH_ENVS, \
    "弃用别名漂移: ALL_SWITCH_ENVS != S5_SWITCH_ENVS (请同步两处字面量)"

# 🔴 CALIBERS 必须是**纯字面量字典** (不得用 dict(...)/** 展开):
#   同上, 静态验证脚本靠 ast.literal_eval 取它。故 legacy_e2e 的 9 键全部显式写出。
CALIBERS = {
    "default": {},
    "legacy":  {"FUXI_S5_TRADABLE_GATE": "off", "FUXI_S5_FIRSTDAY": "off",
                "FUXI_S5_TIE_RULE": "legacy", "FUXI_S5_COST_BPS": "0",
                "FUXI_S5_MIN_COVERAGE": "0", "FUXI_S5_ADJ_PRICE": "off",
                # 窗口也必须挂回: FUXI_S5_DATE_CEIL 于 v0.7.34 翻默认 auto
                # ⇒ 不挂回则 legacy 会**跟着漂移**, 零回归基线失效 (历史读数
                # 再也复现不出来)。见 P-20260917-020。
                "FUXI_S5_DATE_CEIL": "2026-08-28",
                # P-20260916-010: 新增通道口径键 ⇒ 旧基线须**显式挂回 off**
                # (「不设 env = 修正前」自 v0.7.27 起已不成立)。
                "FUXI_S5_AUX_LAG": "off",
                "FUXI_S5_WEEKLY_CHANNEL": "shadow", "FUXI_S5_RETURN_ALIGN": "off",
                "FUXI_S5_SANITY_AST": "legacy", "FUXI_WEEKLY_MISSING_CONTRACT": "legacy",
                "FUXI_S5_RANK_AXIS": "legacy", "FUXI_S5_HOLDINGS_MARK": "legacy"},
    # 🔴 S-1 新增: **端到端**零回归 = S5 内部 6 键 + 上游 3 键 + 正交 2 键。
    #   想要"当年那批候选 + 当年那套 S5"就用本档。legacy 档语义**保持不变**。
    "legacy_e2e": {"FUXI_S5_TRADABLE_GATE": "off", "FUXI_S5_FIRSTDAY": "off",
                   "FUXI_S5_TIE_RULE": "legacy", "FUXI_S5_COST_BPS": "0",
                   "FUXI_S5_MIN_COVERAGE": "0", "FUXI_S5_ADJ_PRICE": "off",
                   "FUXI_S5_DATE_CEIL": "2026-08-28",
                   "FUXI_S5_AUX_LAG": "off",
                   "FUXI_S5_WEEKLY_CHANNEL": "shadow", "FUXI_S5_RETURN_ALIGN": "off",
                   "FUXI_S5_SANITY_AST": "legacy", "FUXI_WEEKLY_MISSING_CONTRACT": "legacy",
                   "FUXI_S5_RANK_AXIS": "legacy", "FUXI_S5_HOLDINGS_MARK": "legacy",
                   "FUXI_ELIGIBLE_REQUIRE_S5": "off",
                   "FUXI_MF_MISSING": "zero",
                   "FUXI_MAB_PARADIGM_BROADCAST": "shadow"},
    "p0":      {"FUXI_S5_TRADABLE_GATE": "on", "FUXI_S5_FIRSTDAY": "off",
                "FUXI_S5_TIE_RULE": "on", "FUXI_S5_COST_BPS": "15",
                "FUXI_S5_MIN_COVERAGE": "0.9", "FUXI_S5_ADJ_PRICE": "on",
                # P-20260916-010: p0 档同样显式声明 (默认 off, 零回归)
                "FUXI_S5_AUX_LAG": "off",
                "FUXI_S5_WEEKLY_CHANNEL": "enforce", "FUXI_S5_RETURN_ALIGN": "on",
                "FUXI_S5_SANITY_AST": "enforce", "FUXI_WEEKLY_MISSING_CONTRACT": "enforce",
                "FUXI_S5_RANK_AXIS": "enforce", "FUXI_S5_HOLDINGS_MARK": "enforce"},
}
# 防漂移: legacy_e2e 必须恰好等于 legacy ∪ UPSTREAM_LEGACY
assert CALIBERS["legacy_e2e"] == {**CALIBERS["legacy"], **UPSTREAM_LEGACY}, \
    "legacy_e2e 与 legacy+UPSTREAM_LEGACY 不一致 (漂移)"

# 🔴 只有**声明了上游键**的口径档才 pop 上游键。
#   若无条件把 UPSTREAM_SWITCH_ENVS 塞进 pop 列表 ⇒ `--caliber legacy` 会多清掉
#   外部传入的上游 env = 改变既有行为 (违反零破坏)。
_CALIBER_POP_EXTRA = {"legacy_e2e": UPSTREAM_SWITCH_ENVS}

# 数据窗口径: **不参与** apply_caliber 的清空-重设, 改用 setdefault 语义。
#   🔴 为何特殊对待: 数据窗 A/B 的语义是「同口径 × 不同窗口」—— 口径与窗口
#   必须**正交**。若把 DATE_CEIL 放进 ALL_SWITCH_ENVS (先 pop 再赋值), 则
#   `--caliber legacy` 会**静默覆盖**外部传入的 FUXI_S5_DATE_CEIL=auto ⇒
#   窗口 A/B 跑成"两边同窗" ⇒ 报出「无任何变化」的**假阴性**。
#   故: 外部显式传入者优先 (setdefault 只在该键缺失时才落 legacy 的挂回值)。
_WINDOW_KEYS = {"FUXI_S5_DATE_CEIL"}

# P-20260916-010 (`FUXI_S5_AUX_LAG`): 辅助通道「按**交易日历**滞后 1 交易日」
#   同样是**与口径档正交**的通道口径键 —— 提案判据 (a) 要求在
#   `legacy × {off,on}` 两个口径下重测同一批因子。
#   🔴 故必须与 DATE_CEIL 同族走 **setdefault**: 若误塞进
#   `ALL_SWITCH_ENVS` (先 pop 再赋值), 外部 `FUXI_S5_AUX_LAG=on` 会被
#   静默吞掉 ⇒ 两臂同 env ⇒ A/B 报「无变化」的**假阴性**
#   (与 P-20260917-021 的 `_AB_KEYS` 同机理)。
_SETDEFAULT_KEYS = _WINDOW_KEYS | {"FUXI_S5_AUX_LAG", "FUXI_S5_WEEKLY_CHANNEL",
    "FUXI_S5_RETURN_ALIGN", "FUXI_S5_SANITY_AST", "FUXI_WEEKLY_MISSING_CONTRACT", "FUXI_S5_RANK_AXIS",
    "FUXI_S5_HOLDINGS_MARK"}

# P-20260917-021: **A/B 臂键** — 与口径/窗口均正交, 同样只 setdefault。
#   🔴 为何不能放进 ALL_SWITCH_ENVS: apply_caliber 先 pop 再赋值 ⇒ 会静默吞掉
#   外部显式传入的 FUXI_S5_IDX_ALIGN ⇒ 两臂跑成同一个 env ⇒ A/B 报
#   「无任何变化」的**假阴性** (P-20260916-016 已就同一机理立过案)。
_AB_KEYS = {"FUXI_S5_IDX_ALIGN"}

SEMANTIC_DEFAULTS = {
    "FUXI_S5_WEEKLY_CHANNEL": "enforce", "FUXI_S5_RETURN_ALIGN": "on",
    "FUXI_S5_SANITY_AST": "enforce", "FUXI_WEEKLY_MISSING_CONTRACT": "enforce",
    "FUXI_S5_RANK_AXIS": "enforce",
    "FUXI_S5_HOLDINGS_MARK": "enforce",
}


def semantic_contracts():
    return {key: (os.environ.get(key, default) or default).strip().lower()
            for key, default in SEMANTIC_DEFAULTS.items()}


def apply_caliber(name):
    """先清空本档作用域内的开关再按口径设置 ⇒ 各档互斥、可复现。

    ``FUXI_S5_DATE_CEIL`` 例外: 只 ``setdefault`` (外部显式传入优先)
    ⇒ 保证「口径档 × 数据窗」可自由正交组合 (A/B 的前提)。

    S-1: pop 范围 = ``S5_SWITCH_ENVS`` + 本档在 ``_CALIBER_POP_EXTRA`` 里声明的
    额外键 (目前只有 ``legacy_e2e`` 声明上游 3 键)。其余档**不动**上游键
    ⇒ 既有 legacy/p0 行为逐位不变 (零破坏)。
    """
    for k in list(S5_SWITCH_ENVS) + list(_CALIBER_POP_EXTRA.get(name, ())):
        os.environ.pop(k, None)
    _lifts = []
    for k, v in CALIBERS[name].items():
        if k in _SETDEFAULT_KEYS:
            if k not in os.environ:          # 外部未传 ⇒ 落口径的挂回值
                os.environ[k] = v
                _lifts.append((k, v))
        else:
            os.environ[k] = v
    if _lifts:
        print("[CALIBER] 正交键挂回 (外部未传 ⇒ 用口径值): %s" % _lifts)

    # ── S-1: 把「口径覆盖缺口」从**静默**变成**显式** ──
    #   legacy 档的语义没有变 (仍是 S5 内部回归), 但必须当场说清它**不含**什么。
    if name == "legacy":
        print("[CALIBER] NOTE: legacy = **S5 内部回归** (只挂回 %d 个 S5 键); "
              "上游候选口径键 %s **不在覆盖范围** ⇒ 候选集口径仍是新的。"
              % (len(S5_SWITCH_ENVS), UPSTREAM_SWITCH_ENVS))
        print("[CALIBER]       需要**端到端**零回归请用 --caliber legacy_e2e。")
        _up = {k: os.environ[k] for k in UPSTREAM_SWITCH_ENVS if k in os.environ}
        if _up:
            print("[CALIBER] 上游键当前被外部显式设置 (本档不会覆盖): %s" % _up)
    # 记录本档**实际**生效的上游键值 (供结果 JSON 披露; 只读, 不改 env)
    _up_obs = {k: os.environ.get(k, "(unset→代码默认)") for k in UPSTREAM_SWITCH_ENVS}
    print("[CALIBER] 上游键实测: %s" % _up_obs)
    return dict(CALIBERS[name])


def caliber_out_path(name, idx_align=None, limit=0):
    """口径 → 结果文件。A/B 臂与 --limit 各自独立命名 ⇒ 互不覆盖、可并存比对。"""
    if name == "default":
        p = OUT_JSON
        stem = "s5_fullpool_rerun_result"
    else:
        stem = "s5_fullpool_rerun_result_%s" % name
        p = DATA_DIR / ("%s.json" % stem)
    if idx_align in ("on", "off"):
        p = DATA_DIR / ("%s_IDXALIGN%s.json" % (stem, idx_align))
    if limit and limit > 0:
        p = p.with_name(p.name.replace(".json", "_limit%d.json" % limit))
    nominal = {key: CALIBERS[name].get(key, default) for key, default in SEMANTIC_DEFAULTS.items()}
    actual = semantic_contracts()
    if actual != nominal:
        import hashlib
        suffix = hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest()[:10]
        p = p.with_name(p.stem + '_sem' + suffix + p.suffix)
    return p


# ── P0-1 (P-20260918-017): 复跑指纹须含**代码维度** ──
# 🔴 动机: 原指纹只有 (caliber, date_ceil, idx_align) —— 三个都是**参数**, 没有**代码**。
#   若全池复跑跑到一半, 另一个进程 (或 AI 自己) 改了 s5_joint_filter.py, 这轮与上一轮
#   的读数差异会被记成"同口径下的差异" —— 与 P-015「重放拿残片 ⇒ SyntaxError 静默记
#   FAIL」同一类不可见失效。冻结清单哈希把「代码是否同一版」变成**可检测事实**。
#   来源: external_audit_quant_autoresearch_20260918.md §7 P0-1。
CODE_FINGERPRINT_FILES = [
    "scripts/s5_fullpool_rerun.py",
    "research/factor_alchemy/s5_joint_filter.py",
    "research/factor_alchemy/weekly_panel.py",
    "research/factor_alchemy/forge/primitives.py",
    "research/factor_alchemy/forge/expression.py",
    "research/factor_alchemy/multi_stage_validator.py",
    "research/factor_alchemy/ralph_loop.py",
    "scripts/data_loader_ext.py",
]


def _sha256_file(p, nbytes=1 << 22):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(nbytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def code_fingerprint():
    """冻结清单的文件哈希 (截断 16 hex)。缺失/不可读 ⇒ 记 'MISSING'/'ERR:*' 而非跳过
    —— 跳过会让"文件被删"看起来和"文件没变"一样。"""
    out = {}
    for rel in CODE_FINGERPRINT_FILES:
        p = BASE / rel
        if not p.exists():
            out[rel] = "MISSING"
            continue
        try:
            out[rel] = _sha256_file(p)[:16]
        except Exception as e:
            out[rel] = "ERR:%s" % type(e).__name__
    return out


# ── P1-2 (P-20260918-018): 每 attempt 一目录 + result.json 三态留痕 ──
# 🔴 **双写保兼容**: 旧的聚合结果文件 `s5_fullpool_rerun_result*.json` 结构**一字不改**
#   (已有消费者读 results / new_passed / new_excess_26 / new_reason)。新增目录仅作留痕。
#   价值: P-015 那类"静默记 FAIL"在**目录层面**即肉眼可见 —— 每条 attempt 必带三态。
ATTEMPT_ROOT = DATA_DIR / "s5_runs"
_CODE_FP = {}          # 由 main() 填入, 供 attempt 留痕引用


def _attempt_status(rec):
    """三态 (禁止静默): pass / reject / degraded(+归因)。

    与 memory 的「未走 / 通 / 降级+归因」对应: 本函数只处理"已走"的两种,
    未走 (resume 跳过 / --limit 截断) 由调用侧不落目录来表达。
    """
    if rec.get("error"):
        return "degraded", "运行异常: %s" % rec["error"]
    if rec.get("new_passed"):
        return "pass", (rec.get("new_reason") or "")[:200]
    return "reject", (rec.get("new_reason") or "")[:200]


def write_attempt(run_id, seq, name, rec, formula_full, caliber, date_ceil, idx_align):
    """落一个 attempt 目录。返回相对路径; **失败不抛** (留痕失败不得中断复跑)。"""
    try:
        slug = re.sub(r"[^0-9A-Za-z_]+", "_", str(name))[:64] or "unnamed"
        d = ATTEMPT_ROOT / run_id / ("%04d_%s" % (seq, slug))
        d.mkdir(parents=True, exist_ok=True)
        status, attrib = _attempt_status(rec)
        metrics = {k: rec.get(k) for k in
                   ("new_passed", "new_excess_25", "new_excess_26",
                    "new_calmar_25", "new_calmar_26", "icir", "source")}
        metrics["sparse_flags"] = rec.get("sparse_flags")
        metrics["sparse_warning"] = rec.get("sparse_warning")
        with open(d / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=1, default=_json_safe)
        result = {
            "attempt": seq, "run_id": run_id, "factor_name": name,
            "status": status, "attribution": attrib, "error": rec.get("error") or None,
            "caliber": caliber, "date_ceil": date_ceil, "idx_align": idx_align,
            # 🔴 不造假的 orders.csv: S5 的 validate_factor 只返回
            #   (portfolio_returns, turnover_history), **不暴露持仓序列**
            #   ⇒ 无法产出订单。这里显式记录"不可得", 而不是落一个空文件
            #   (空文件会被下游当成"零持仓合法结果" —— 那才是真正的假通道)。
            "orders_csv": "unavailable__validate_factor_exposes_no_holdings",
            "code_fingerprint": _CODE_FP,
            "semantic_contracts": semantic_contracts(),
        }
        with open(d / "result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=1, default=_json_safe)
        with open(d / "log.txt", "w", encoding="utf-8") as f:
            f.write("factor_name: %s\nstatus: %s\nattribution: %s\n"
                    "caliber: %s | date_ceil: %s | idx_align: %s\n\n"
                    "formula:\n%s\n" % (name, status, attrib, caliber, date_ceil,
                                         idx_align, formula_full))
        return str(d.relative_to(BASE))
    except Exception as e:
        print("    [WARN] attempt 留痕失败 (不影响复跑): %s" % e)
        return None


def _resume_decision(prev, fp, code_fp):
    """续跑判据 (P-20260916-016 窗口 + P-20260917-021 A/B臂 + P-20260918-017 代码)。

    返回 ``(reuse_count, msgs)``: ``reuse_count is None`` ⇒ 不复用; 否则为可复用条数。

    🔴 抽成纯函数的原因 (S-4): 这是**新增的失效路径** —— 若埋在 main() 里, 只能靠
       真跑一次 3h 复跑来验证「指纹不符是否真的拒绝复用」, 等于不验证。纯函数可单测。
    """
    msgs = []
    prev_fp = {"caliber": prev.get("caliber"),
               "date_ceil": prev.get("date_ceil"),
               "idx_align": prev.get("idx_align", "off")}
    if 'semantic_contracts' in fp:
        prev_fp['semantic_contracts'] = prev.get('semantic_contracts')
    if "idx_align" not in prev:
        msgs.append("⚠️ 旧文件无 idx_align 键 ⇒ 按 'off' 归一 "
                    "(该轮 harness 未设置该 env = 代码默认 off)")
    # ── P0-1: 代码指纹比对。缺字段 ⇒ 兼容复用 (零破坏, 不强制重跑 3h);
    #          有字段且不符 ⇒ 拒绝复用 (防「代码改了却把读数差异记成同口径差异」)。──
    prev_code = prev.get("code_fingerprint")
    if prev_code is None:
        code_ok = 'semantic_contracts' not in fp
        msgs.append('旧文件无 code_fingerprint: ' + ('旧调用兼容复用' if code_ok else '新语义契约拒绝复用'))
    elif prev_code == code_fp:
        code_ok = True
    else:
        code_ok = False
        msgs.append("🔴 代码指纹不符 ⇒ **拒绝复用** (P0-1):")
        for k in sorted(set(prev_code) | set(code_fp)):
            if prev_code.get(k) != code_fp.get(k):
                msgs.append("      %s: %s -> %s" % (k, prev_code.get(k), code_fp.get(k)))
    if prev_fp != fp:
        msgs.append("指纹不符 (已有 %s != 本次 %s), 从零开始" % (prev_fp, fp))
        return None, msgs
    if not code_ok:
        msgs.append("代码指纹不符 ⇒ 从零开始")
        return None, msgs
    n = len(prev.get("results", []))
    msgs.append("命中同口径+同数据窗 %s 已有结果 %d 条, 将跳过" % (fp, n))
    return n, msgs


from s135_candidate_check import _is_pandas_compatible  # noqa: E402,F401  (仅 JQ codegen 用, S5 重跑不预过滤)


def _json_safe(o):
    """float32/np 标量 → JSON 可序列化 (np.float32 JSON: default=_json_safe_default 铁律)"""
    import numpy as _np
    if isinstance(o, (_np.float32, _np.float64, _np.integer)):
        return float(o)
    if isinstance(o, _np.bool_):
        return bool(o)
    return str(o)


def load_pool_formulas():
    """池 CSV: name → {formula, icir, direction, status}"""
    import csv
    out = {}
    with open(POOL_CSV, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            out[r["name"]] = {
                "formula": str(r.get("formula", "") or "").strip(),
                "icir": float(r.get("icir") or 0),
                "direction": str(r.get("direction", "long") or "long"),
                "status": r.get("status", ""),
            }
    return out


def collect_candidates(pool):
    """三源合并: S5 历史被拒因子 (去重)。

    注意: 不做 pandas 兼容预过滤 —— _is_pandas_compatible 是 JQ codegen 翻译
    检查 (拦 .rank(/returns/DSL), 会误杀 S5 本地可跑公式 (S5 有 returns 衍生
    字段 + 横截面 rank 通道 + DSL 自动转换)。S5 eval 失败自然捕获记录。
    """
    cands = {}  # name → dict

    def from_record_file(path, label):
        if not path.exists():
            return
        d = json.load(open(path, encoding="utf-8"))
        det = d.get("detail", [])
        for x in det:
            if x.get("s5_passed"):
                continue
            if "excess" not in str(x.get("s5_reason", "")).lower():
                continue
            name = x.get("factor_name", "")
            if name in cands or name not in pool:
                continue
            f = pool[name]["formula"]
            if not f or f.startswith("# TODO"):
                continue
            cands[name] = {
                "factor_name": name, "formula": f,
                "icir": pool[name]["icir"], "source": label,
                "old_reason": str(x.get("s5_reason", ""))[:120],
            }

    from_record_file(BRIDGE_JSON, "bridge_0909")
    from_record_file(S135_JSON, "s135_0908")

    # 源 C: trajectory_log (Ralph 运行内 S5 拒绝, 公式在 _eval_log 里)
    if TRAJ_JSON.exists():
        d = json.load(open(TRAJ_JSON, encoding="utf-8"))
        for t in d.get("trajectories", []):
            elogs = t.get("_eval_log") or []
            e5s = [e for e in elogs if e.get("stage") == "S5"]
            if not e5s:
                continue
            e5 = e5s[-1]
            if e5.get("passed"):
                continue
            if e5.get("excess_25") is None and e5.get("excess_26") is None:
                continue
            name = e5.get("factor_name", "")
            formula = e5.get("formula") or ""
            if not formula:
                # 尝试轨迹 expression
                formula = (t.get("expression") or {}).get("content") or ""
            if not formula:
                # 尝试轨迹 expression
                formula = (t.get("expression") or {}).get("content") or ""
            if not formula:
                continue
            # 名字可能与池内重复 (池优先已有), 用 (name, formula 前缀) 去重
            if name in cands:
                continue
            # forge 公式与池内名字撞车时跳过 (保留池版本)
            key = f"{name}||{formula[:60]}"
            cands[name] = {
                "factor_name": name, "formula": formula,
                "icir": float(e5.get("icir") or 0), "source": "trajectory",
                "old_reason": (f"ex25={e5.get('excess_25')}, ex26={e5.get('excess_26')}"),
                "_dedup_key": key,
            }
    return cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--caliber", choices=sorted(CALIBERS.keys()), default="default",
                    help="S5 口径: default(代码默认) | legacy(修正前) | p0(修正后捆绑)")
    ap.add_argument("--idx-align", choices=["default", "off", "on"], default="default",
                    help="P-20260917-021 A/B: FUXI_S5_IDX_ALIGN 基准同窗裁剪 "
                         "(default=不设 env, 走代码默认 off)")
    ap.add_argument("--limit", type=int, default=0,
                    help="只跑前 N 条 (按 icir 降序, 与主循环同序); 0=全部。"
                         "用于廉价守卫/冒烟, 自动禁用 resume 并另存文件")
    args = ap.parse_args()

    applied = apply_caliber(args.caliber)
    # P0-1: 跑前取一次代码指纹 (跑后不再取 —— 中途被改要在**下一轮**才暴露,
    #       本轮只负责把"起跑时的代码版本"钉进结果)
    _CODE_FP.clear()
    _CODE_FP.update(code_fingerprint())
    print("[CODE_FP] %s" % json.dumps(_CODE_FP, ensure_ascii=False))
    # A/B 臂: 在 apply_caliber **之后**设置 (后者不碰 _AB_KEYS)
    if args.idx_align != "default":
        os.environ["FUXI_S5_IDX_ALIGN"] = args.idx_align
    _ia_fp = (os.environ.get("FUXI_S5_IDX_ALIGN", "off") or "off").strip().lower()
    if _ia_fp not in ("on", "off"):
        _ia_fp = "off"
    out_json = caliber_out_path(args.caliber, _ia_fp, args.limit)

    # 断点续跑 (v0.7.27): 同口径已有结果直接复用。
    # 动机: 单条 validate_factor 实测 175~185s (60 条 ≈ 3h), 中途中断/并行
    # 都会想重跑 ⇒ 无续跑键时只能从头再来。
    #
    # 🔴 2026-09-16 P-20260916-016 **复用指纹必须含数据窗**:
    # 原实现只比 ``caliber`` ⇒ 改 ``FUXI_S5_DATE_CEIL`` 重跑(数据窗 A/B) 会
    # **静默命中旧窗口的全部结果**、0 条真跑, 于是 A/B 报出「无任何变化」
    # 这个**假阴性** —— 与「静默截断」同类的不可见失效。
    # 现指纹 = (caliber, date_ceil); 旧落盘文件无 ``date_ceil`` 字段 ⇒
    # 视为不匹配 ⇒ 从零跑 (安全方向: 宁重跑不复用错窗读数)。
    _ceil_fp = (os.environ.get("FUXI_S5_DATE_CEIL", "auto")
                or "auto").strip()
    # 🔴 P-20260917-021: **A/B 臂必须入指纹**。否则 `--idx-align on` 会静默复用
    # off 臂结果 ⇒ 0 条真跑 ⇒ 报「两臂一致」的假阴性 (同 P-20260916-016 机理)。
    # 旧文件**无 idx_align 键** ⇒ 归一为 "off": 那一轮 harness 从未设置该 env
    # ⇒ 实际走的就是代码默认 off。该假设已由 --limit 守卫跑逐值核对。
    _fp = {"caliber": args.caliber, "date_ceil": _ceil_fp, "idx_align": _ia_fp,
           "semantic_contracts": semantic_contracts()}
    done: dict = {}
    if args.limit and args.limit > 0:
        print("[RESUME] --limit>0 ⇒ 禁用续跑 (残缺结果不可当完整臂)")
    elif (not args.dry_run) and out_json.exists():
        try:
            _prev = json.loads(out_json.read_text(encoding="utf-8"))
            _n_reuse, _msgs = _resume_decision(_prev, _fp, _CODE_FP)
            for _m in _msgs:
                print("[RESUME] %s" % _m)
            if _n_reuse is not None:
                done = {r["factor_name"]: r for r in _prev.get("results", [])}
        except Exception as _re:
            print(f"[RESUME] 读取失败, 从零开始: {type(_re).__name__}: {_re}")

    # P1-2: 本轮 attempt 目录的 run_id (含口径/窗口/臂 + 起跑时刻)
    _run_id = "%s__%s__ceil%s__ia%s" % (
        time.strftime("%Y%m%d_%H%M%S"), args.caliber,
        re.sub(r"[^0-9A-Za-z]+", "", _ceil_fp or "auto"), _ia_fp)
    _attempt_rows = []
    print("[ATTEMPT] 留痕目录 data/s5_runs/%s/ (双写: 旧聚合文件结构不变)" % _run_id)

    pool = load_pool_formulas()
    cands = collect_candidates(pool)

    _ordered = sorted(cands.items(), key=lambda kv: -kv[1]["icir"])
    if args.limit and args.limit > 0:
        _ordered = _ordered[:args.limit]
        cands = dict(_ordered)
    print(f"[Collect] 池 {len(pool)} | S5 历史被拒可重跑: {len(cands)}"
          + ("  (--limit %d 截断)" % args.limit if args.limit else ""))
    print(f"[CALIBER] {args.caliber} | idx_align={_ia_fp} "
          f"| date_ceil={_ceil_fp} "
          f"| env={applied if applied else '(代码默认)'} "
          f"| out={out_json.name}")
    by_src = {}
    for c in cands.values():
        by_src[c["source"]] = by_src.get(c["source"], 0) + 1
    print(f"  来源分布: {by_src}")

    if args.dry_run:
        for name, c in sorted(cands.items(), key=lambda kv: -kv[1]["icir"]):
            print(f"  [{c['source']:>12}] {name[:50]:<50} icir={c['icir']:.3f}")
            print(f"               {c['formula'][:100]}")
        return

    # 全量池 S5 重裁 (默认已是全量池; 双保险显式清 env)
    os.environ.pop("FUXI_S5_UNIVERSE", None)
    from s5_joint_filter import S5JointFilter
    s5 = S5JointFilter(top_n=80)
    print(f"[CFG] sample_stocks={s5.sample_stocks}, index={s5.index_path.name}")

    t0 = time.time()
    s5._load_data()
    print(f"[LOAD] {time.time()-t0:.0f}s universe={len(s5._universe)} rows={len(s5._price_df)}")

    results = []
    n_pass = 0
    for i, (name, c) in enumerate(_ordered):
        if name in done:
            rec = done[name]
            results.append(rec)
            if rec.get("new_passed"):
                n_pass += 1
            print(f"  [{i+1}/{len(cands)}] SKIP(resume) {name[:48]} "
                  f"({len(results)}/{len(cands)} 已装载)")
            continue
        t1 = time.time()
        try:
            r = s5.validate_factor(name, c["formula"])
            # P-20260916-018: 稀疏结构体检**必须走结构化字段**。
            # 旧写法 `(r.reason or "")[:200]` 会把 P-014a 追加在 reason **末尾**的
            # ⚠️ 稀疏结构注记静默截掉 (基础 reason 已 ~110 字符 + 注记 ~120 字符 > 200)
            # —— 与 `trajectory_logger.py:417` `formula[:200]` 同一类「写入即截断」陷阱。
            _sd = getattr(r, "sparse_diag", None) or {}
            rec = {
                "factor_name": name,
                "formula": c["formula"][:300],
                "icir": c["icir"],
                "source": c["source"],
                "old_s5_reason": c["old_reason"],
                "new_passed": r.passed,
                "new_excess_25": round(r.excess_2025, 4),
                "new_excess_26": round(r.excess_2026, 4),
                "new_calmar_25": round(r.calmar_2025, 3),
                "new_calmar_26": round(r.calmar_2026, 3),
                "new_reason": (r.reason or "")[:400],
                # ── 结构化稀疏体检 (P-014a) — 不受字符串截断影响 ──
                "sparse_flags": list(_sd.get("flags") or []),
                "sparse_flag_basis": _sd.get("decile_flag_basis"),
                "sparse_argmax_recent": _sd.get("decile_argmax_recent"),
                "sparse_monotone_recent": _sd.get("decile_monotone_pairs_recent"),
                "sparse_sel_zero_frac_recent": _sd.get("sel_zero_frac_mean_recent"),
                "sparse_n_pos_median_recent":
                    (_sd.get("n_pos_recent") or {}).get("median"),
                "sparse_pct_pos_below_topn_recent":
                    _sd.get("pct_periods_pos_below_topn_recent"),
                "sparse_warning": _sd.get("warning"),
            }
        except Exception as e:
            rec = {
                "factor_name": name, "formula": c["formula"][:300],
                "icir": c["icir"], "source": c["source"],
                "old_s5_reason": c["old_reason"],
                "new_passed": False, "error": f"{type(e).__name__}: {e}"[:150],
            }
        results.append(rec)
        if rec.get("new_passed"):
            n_pass += 1
        # P1-2: attempt 目录留痕 (双写; 失败不影响复跑)
        _ap = write_attempt(_run_id, i + 1, name, rec, c["formula"],
                            args.caliber, _ceil_fp, _ia_fp)
        _st = _attempt_status(rec)[0]
        _attempt_rows.append({"seq": i + 1, "factor_name": name, "status": _st,
                              "path": _ap})
        # 增量落盘 (每条): 崩溃/序列化异常不丢已跑结果
        try:
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump({"run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                           "caliber": args.caliber, "caliber_env": applied,
                           "date_ceil": _ceil_fp, "idx_align": _ia_fp,
                           "limit": args.limit,
                           "universe": "full_pool_5763", "benchmark": "000985",
                           "progress": f"{i+1}/{len(cands)}",
                           "code_fingerprint": _CODE_FP,
                           "semantic_contracts": _fp['semantic_contracts'],
                           "attempt_run_id": _run_id,
                           "results": results},
                          f, ensure_ascii=False, indent=1, default=_json_safe)
        except Exception as werr:
            print(f"    [WARN] incremental save failed: {werr}")
        dt = time.time() - t1
        tag = "PASS" if rec.get("new_passed") else "rej"
        ex = f"ex25={rec.get('new_excess_25')} ex26={rec.get('new_excess_26')}"
        print(f"  [{i+1}/{len(cands)}] {tag} {name[:48]:<48} {ex} ({dt:.0f}s)")

    # 汇总
    flipped = [r for r in results if r.get("new_passed")]
    # P1-2: attempt manifest (三态计数可见; 只覆盖**本轮真跑**的 attempt)
    _stat = {}
    for _r in _attempt_rows:
        _stat[_r["status"]] = _stat.get(_r["status"], 0) + 1
    _manifest = {
        "run_id": _run_id, "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "caliber": args.caliber, "date_ceil": _ceil_fp, "idx_align": _ia_fp,
        "code_fingerprint": _CODE_FP,
        "semantic_contracts": _fp['semantic_contracts'],
        "upstream_env_observed": {k: os.environ.get(k, "(unset→代码默认)")
                                  for k in UPSTREAM_SWITCH_ENVS},
        "n_ran_this_run": len(_attempt_rows),
        "n_resumed_or_skipped": len(results) - len(_attempt_rows),
        "status_counts": _stat,
        "upstream_scope_note": (
            "本脚本从**静态 JSON** 取候选 (bridge/s135/trajectory), 故上游 3 键"
            "对本脚本输出**无直接影响**; 它们只决定上游 validator 往 S5 送什么。"
            "legacy 档不覆盖它们 ⇒ 是 S5 内部回归; legacy_e2e 档才挂回。"),
        "attempts": _attempt_rows,
    }
    try:
        _mdir = ATTEMPT_ROOT / _run_id
        _mdir.mkdir(parents=True, exist_ok=True)
        with open(_mdir / "run_manifest.json", "w", encoding="utf-8") as f:
            json.dump(_manifest, f, ensure_ascii=False, indent=1, default=_json_safe)
        print("[ATTEMPT] manifest 落盘 %s (三态 %s)"
              % (str((_mdir / "run_manifest.json").relative_to(BASE)), _stat))
    except Exception as e:
        print("[ATTEMPT] [WARN] manifest 落盘失败: %s" % e)

    out = {
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "caliber": args.caliber,
        "caliber_env": applied,
        "date_ceil": _ceil_fp,
        "idx_align": _ia_fp,
        "limit": args.limit,
        "universe": "full_pool_5763",
        "benchmark": "000985",
        # ── P0-1 / P1-2 新增字段 (只增不改名 ⇒ 既有消费者不受影响) ──
        "code_fingerprint": _CODE_FP,
        "semantic_contracts": _fp['semantic_contracts'],
        "attempt_run_id": _run_id,
        "upstream_switch_envs": UPSTREAM_SWITCH_ENVS,
        "total_rerun": len(results),
        "flipped_to_pass": len(flipped),
        "flipped_names": [r["factor_name"] for r in flipped],
        "results": results,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=_json_safe)

    print(f"\n[DONE] 重跑 {len(results)} | 翻盘 PASS {len(flipped)} | "
          f"口径 {args.caliber} | 结果落盘 {out_json.name}")
    if flipped:
        print("翻盘因子:")
        for r in flipped:
            print(f"  ✅ {r['factor_name']} | ex25={r['new_excess_25']} "
                  f"ex26={r['new_excess_26']} c25={r['new_calmar_25']} c26={r['new_calmar_26']}")
    print(f"[TOTAL] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
