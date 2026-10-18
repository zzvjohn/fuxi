# -*- coding: utf-8 -*-
"""
L2 修复回路 — 代码类失败的 LLM 自动修复
========================================
伏羲 agentic 演进 L2: 主攻 460 轨迹中最弱环节 (代码生成, ~41%)。

设计原则 (2026-08-14):
- 旁路"维修间": 不碰主循环, 只修 E 阶段分流出来的代码类失败
- 沙箱纪律: 备份 → 写修复 → 跑测试门禁 → 全绿才合并, 失败自动回滚
- 失败不重试: 同一条目最多 MAX_ATTEMPTS 次, 超限记 WarningDirection
- 局限: 只修本地 Python (可完整验证); JQ 平台代码仅静态验证 + 建议文件

与 v0.5.1 退避分工: 退避=换引擎(方向问题), L2=修代码(实现问题), 互补不冲突。

用法:
  python repair_agent.py --check      # 查看队列状态
  python repair_agent.py --run        # 处理 pending 条目 (最多 --limit 条)
  python repair_agent.py --retry ID   # 强制重试某条目
"""

import json
import re
import shutil
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

FACTOR_ALCHEMY_DIR = Path(__file__).parent
DATA_DIR = FACTOR_ALCHEMY_DIR.parent.parent / "data"
REPAIR_QUEUE_PATH = DATA_DIR / "repair_queue.json"
REPAIR_OUTPUT_DIR = DATA_DIR / "repair_output"
BACKUP_SUFFIX = ".bak_l2"

MAX_ATTEMPTS = 2          # 同一修复条目最多尝试次数
DEFAULT_TESTS = [         # 修复后默认测试门禁 (快测试)
    "scripts/test_v08_seed_recheck.py",
    "scripts/test_v09_jq_autogen.py",
]
ALLOWED_EXT = (".py",)    # 只修 .py
ALLOWED_ROOT = FACTOR_ALCHEMY_DIR  # 只修本目录树内文件

CODE_ERROR_KEYWORDS = (
    "syntaxerror", "nameerror", "keyerror", "attributeerror", "typeerror",
    "编译", "compile", "代码", "execution", "无法执行", "执行失败",
    "invalid syntax", "not defined", "unexpected token", "eval fail",
)


# ═══════════════════════════════════════════════════════════
# 队列管理
# ═══════════════════════════════════════════════════════════

def load_queue() -> Dict:
    if REPAIR_QUEUE_PATH.exists():
        with open(REPAIR_QUEUE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"queue": [], "updated_at": ""}


def save_queue(q: Dict):
    q["updated_at"] = datetime.now().isoformat()
    REPAIR_QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPAIR_QUEUE_PATH, "w", encoding="utf-8") as f:
        json.dump(q, f, ensure_ascii=False, indent=2)


def is_code_failure(reason: str) -> bool:
    """判定失败原因是否属代码类 (供 ralph_loop 失败分类调用)"""
    r = str(reason or "").lower()
    return any(k in r for k in CODE_ERROR_KEYWORDS)


def enqueue_repair(
    factor_name: str,
    formula: str,
    stage: str,
    reason: str,
    kind: str = "local_python",
) -> bool:
    """代码类失败入队 (幂等: 同因子同 stage 不重复入队)。供 ralph_loop._phase_evaluate 调用。"""
    q = load_queue()
    for item in q["queue"]:
        if (
            item.get("factor_name") == factor_name
            and item.get("stage") == stage
            and item.get("status") in ("pending", "in_progress")
        ):
            return False

    repair_id = f"rp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(q['queue']):03d}"
    q["queue"].append({
        "repair_id": repair_id,
        "kind": kind,
        "status": "pending",
        "factor_name": factor_name,
        "formula": (formula or "")[:300],
        "stage": stage,
        "reason": (reason or "")[:500],
        "file_path": "",          # 空 = 由 LLM 根据错误信息自行定位
        "tests": DEFAULT_TESTS,
        "created_at": datetime.now().isoformat(),
        "attempts": 0,
        "last_error": "",
    })
    save_queue(q)
    print(f"  [L2] 修复队列入队: {repair_id} ({factor_name}, {stage}, kind={kind})")
    return True


# ═══════════════════════════════════════════════════════════
# LLM 修复
# ═══════════════════════════════════════════════════════════

_SYSTEM_PROMPT = (
    "你是资深 Python 量化系统修复工程师。任务: 修复伏羲因子挖掘系统中一个代码类失败。"
    "你必须输出严格 JSON: {\"files\": [{\"path\": \"相对路径\", \"content\": \"修复后完整文件内容\"}]}。"
    "规则: "
    "1. 只允许修改 research/factor_alchemy/ 目录树内的 .py 文件; "
    "2. 每个文件输出完整内容, 不是 diff; "
    "3. 不要改动因子公式本身的数学含义, 只修代码错误; "
    "4. 不要添加新依赖; "
    "5. 如果无法定位问题, 输出 {\"files\": []}。"
    "不要输出 JSON 以外的任何内容。"
)


def _build_repair_prompt(item: Dict) -> str:
    lines = ["以下因子在验证管线中失败, 请定位并修复代码问题。", ""]
    lines.append(f"因子名: {item.get('factor_name', '?')}")
    lines.append(f"公式: {item.get('formula', '')[:300]}")
    lines.append(f"失败阶段: {item.get('stage', '?')}")
    lines.append(f"失败原因: {item.get('reason', '')}")
    if item.get("last_error"):
        lines.append(f"上次修复尝试错误: {item['last_error'][:300]}")
    lines.append("")
    lines.append(
        "常见代码类失败模式: pandas 属性遮蔽(如变量名 'open' 遮蔽内置)、"
        "dict 字段类型演进后旧切片代码、Forge→pandas 翻译遗漏、"
        "numpy 版本不兼容 (nan_to_num 参数)、Rolling.rank 不支持。"
    )
    lines.append("")
    lines.append("请先定位相关文件, 读取并分析, 然后输出修复 JSON。")
    return "\n".join(lines)


def _extract_files(text: str) -> List[Dict]:
    """解析 LLM 输出的 JSON → files 列表, 校验路径安全"""
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return []
        obj = json.loads(m.group(0))
        files = obj.get("files", [])
    except Exception:
        return []

    safe = []
    for f in files:
        path = str(f.get("path", ""))
        content = f.get("content", "")
        if not path or not isinstance(content, str) or len(content) < 20:
            continue
        full = (FACTOR_ALCHEMY_DIR / path).resolve()
        # 路径安全: 必须在允许目录树内且为 .py
        if not str(full).startswith(str(ALLOWED_ROOT.resolve())):
            print(f"  [L2] ⚠️ 拒绝越界路径: {path}")
            continue
        if not full.name.endswith(ALLOWED_EXT):
            print(f"  [L2] ⚠️ 拒绝非 .py 文件: {path}")
            continue
        safe.append({"path": str(full), "content": content})
    return safe


def _run_tests(item: Dict) -> List[str]:
    """运行测试门禁, 返回失败信息列表 (空=全绿)"""
    import subprocess
    tests = item.get("tests") or DEFAULT_TESTS
    failures = []
    workspace_root = DATA_DIR.parent
    for t in tests:
        test_path = workspace_root / t
        if not test_path.exists():
            continue
        try:
            r = subprocess.run(
                [sys.executable, str(test_path)],
                capture_output=True, text=True, timeout=600,
                cwd=str(workspace_root),
            )
            if r.returncode != 0:
                tail = (r.stdout or "")[-800:] + (r.stderr or "")[-800:]
                failures.append(f"{t} 失败: {tail[-600:]}")
        except Exception as e:
            failures.append(f"{t} 执行异常: {e}")
    return failures


# ═══════════════════════════════════════════════════════════
# 主处理流程
# ═══════════════════════════════════════════════════════════

def _note_warning_direction(item: Dict, reason: str):
    """修复失败超限 → 记 WarningDirection (不重试)"""
    try:
        from experience_memory import get_memory
        mem = get_memory()
        wid = f"l2_repair_failed::{item.get('factor_name', '?')}"
        warnings = mem.data.setdefault("warning_directions", [])
        for w in warnings:
            if w.get("direction_id") == wid:
                w["failed_attempts"] = w.get("failed_attempts", 0) + 1
                mem._save()
                return
        warnings.append({
            "direction_id": wid,
            "description": f"L2修复失败: {item.get('factor_name')} "
                           f"({item.get('stage')}) → {reason[:120]}",
            "severity": "soft",
            "reason": "l2_repair_failed",
            "failed_attempts": 1,
            "added_at": datetime.now().isoformat(),
        })
        mem._save()
    except Exception:
        pass


def _note_success_pattern(item: Dict, files: List[Dict]):
    """修复成功 → attempts lessons 追加 (供未来相似失败参考)"""
    try:
        from experience_memory import get_memory
        mem = get_memory()
        for a in mem.data.get("attempts", []):
            if a.get("factor_name") == item.get("factor_name"):
                a.setdefault("lessons", [])
                a["lessons"].append(
                    f"L2修复: {item.get('stage')} 代码错误已修复 "
                    f"({', '.join(Path(f['path']).name for f in files)})"
                )
                mem._save()
                break
    except Exception:
        pass


def _process_local_python(item: Dict, use_llm: bool = True) -> bool:
    """处理一条 local_python 修复: 返回是否 merged"""
    item["status"] = "in_progress"
    item["attempts"] += 1
    _queue_with(item)

    files = []
    if use_llm:
        try:
            from llm_client import get_llm_client
            client = get_llm_client()
            text = client.chat_with_system(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=_build_repair_prompt(item),
                temperature=0.2,
                max_tokens=8192,
                response_format={"type": "json_object"},
            )
            files = _extract_files(text)
        except Exception as e:
            item["last_error"] = str(e)
            print(f"  [L2] LLM 调用失败: {e}")
            files = []

    if not files:
        _mark_status(item, "failed", "LLM 无法定位修复文件")
        _note_warning_direction(item, "无法定位")
        return False

    # ── 沙箱: 备份 → 写入 → 测试 → 合并/回滚 ──
    backups = []
    for f in files:
        src = Path(f["path"])
        if not src.exists():
            _mark_status(item, "failed", f"目标文件不存在: {src.name}")
            return False
        bak = src.with_suffix(src.suffix + BACKUP_SUFFIX)
        shutil.copy2(src, bak)
        backups.append((src, bak))

    try:
        for f in files:
            with open(Path(f["path"]), "w", encoding="utf-8") as fp:
                fp.write(f["content"])
        # 语法自检
        for f in files:
            with open(Path(f["path"]), "r", encoding="utf-8") as fp:
                compile(fp.read(), str(f["path"]), "exec")
        failures = _run_tests(item)
    except Exception as e:
        failures = [f"写入/编译异常: {e}"]

    if failures:
        # 回滚
        for src, bak in backups:
            shutil.copy2(bak, src)
        for _, bak in backups:
            bak.unlink(missing_ok=True)
        err_text = "\n".join(failures)[:500]
        item["last_error"] = err_text
        print(f"  [L2] ✗ 测试门禁未过, 已回滚: {err_text[:200]}")
        if item["attempts"] >= MAX_ATTEMPTS:
            _mark_status(item, "failed", err_text[:300])
            _note_warning_direction(item, err_text[:120])
            return False
        _mark_status(item, "pending", "测试失败, 待重试")
        return False

    # 全绿 → 合并成功, 清理备份
    for _, bak in backups:
        bak.unlink(missing_ok=True)
    item["file_path"] = ", ".join(Path(f["path"]).name for f in files)
    _mark_status(item, "merged", "")
    _note_success_pattern(item, files)
    print(f"  [L2] ✓ 修复合并: {item['file_path']}")
    return True


def _process_jq_code(item: Dict, use_llm: bool = True) -> bool:
    """JQ 平台代码: 仅静态验证 + 生成修复建议 (行为验证必须送 JQ 平台)"""
    item["status"] = "in_progress"
    item["attempts"] += 1
    _queue_with(item)
    suggestion = ""
    if use_llm:
        try:
            from llm_client import get_llm_client
            client = get_llm_client()
            text = client.chat_with_system(
                system_prompt=(
                    "你是聚宽(JQ)平台策略代码修复专家。输出修复后的完整策略代码 "
                    "(纯代码, 不要 JSON 包装, 不要解释)。"
                ),
                user_prompt=_build_repair_prompt(item),
                temperature=0.2,
                max_tokens=8192,
            )
            suggestion = text
        except Exception as e:
            item["last_error"] = str(e)
    if not suggestion or len(suggestion) < 50:
        _mark_status(item, "failed", "LLM 无法生成修复代码")
        return False
    # 静态编译验证
    try:
        compile(suggestion, f"<jq_repair_{item.get('repair_id')}>", "exec")
    except SyntaxError as e:
        item["last_error"] = f"SyntaxError: {e}"
        _mark_status(item, "failed", f"SyntaxError: {e}")
        return False
    # 写建议文件 (不自动合并 — JQ 行为验证需平台执行)
    REPAIR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPAIR_OUTPUT_DIR / f"{item.get('repair_id')}_jq_fixed.py"
    with open(out, "w", encoding="utf-8") as fp:
        fp.write(suggestion)
    item["file_path"] = str(out)
    _mark_status(item, "patched", "JQ 修复建议已生成, 待平台行为验证")
    print(f"  [L2] ✓ JQ 修复建议: {out} (静态编译通过, 待送 JQ 验证)")
    return True


def _queue_with(item: Dict) -> Dict:
    q = load_queue()
    for i in q["queue"]:
        if i.get("repair_id") == item.get("repair_id"):
            i.update(item)
            break
    save_queue(q)
    return q


def _mark_status(item: Dict, status: str, note: str):
    item["status"] = status
    item["updated_at"] = datetime.now().isoformat()
    if note:
        item["last_error"] = note if status in ("failed",) else item.get("last_error", "")
        item["note"] = note
    _queue_with(item)


def run(limit: int = 3, use_llm: bool = True, retry_id: str = "") -> Dict:
    q = load_queue()
    pending = []
    for item in q["queue"]:
        if retry_id:
            if item.get("repair_id") == retry_id:
                item["status"] = "pending"
                item["attempts"] = 0
                pending.append(item)
            continue
        if item.get("status") in ("pending",) and item.get("attempts", 0) < MAX_ATTEMPTS:
            pending.append(item)
        elif item.get("status") == "pending" and item.get("attempts", 0) >= MAX_ATTEMPTS:
            _mark_status(item, "failed", "attempts 超限")

    pending = pending[:limit]
    print(f"[L2] 待处理: {len(pending)} 条")

    stats = {"merged": 0, "patched": 0, "failed": 0, "retry_pending": 0}
    for item in pending:
        print(f"\n  → {item.get('repair_id')} ({item.get('kind')}): "
              f"{item.get('factor_name')} [{item.get('stage')}]")
        if item.get("kind") == "jq_code":
            ok = _process_jq_code(item, use_llm=use_llm)
            stats["patched" if ok else "failed"] += 1
        else:
            ok = _process_local_python(item, use_llm=use_llm)
            if ok:
                stats["merged"] += 1
            elif item.get("status") == "pending":
                stats["retry_pending"] += 1
            else:
                stats["failed"] += 1

    save_queue(q)
    print(f"\n[L2] 本轮: {stats}")
    return stats


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--check" in args or not args:
        q = load_queue()
        print(f"修复队列: {len(q['queue'])} 条")
        for item in q["queue"]:
            print(f"  [{item.get('status')}] {item.get('repair_id')} "
                  f"{item.get('factor_name', '?')} [{item.get('stage', '?')}] "
                  f"attempts={item.get('attempts', 0)} kind={item.get('kind', '?')}")
    elif "--retry" in args:
        rid = args[args.index("--retry") + 1] if len(args) > args.index("--retry") + 1 else ""
        run(retry_id=rid)
    elif "--run" in args:
        limit = 3
        if "--limit" in args:
            limit = int(args[args.index("--limit") + 1])
        run(limit=limit)
