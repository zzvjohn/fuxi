# -*- coding: utf-8 -*-
"""P-20260919-001 决定性验证: `_freq_audit` 能否穿越 写方 → `_save()` → 重载。

为何需要对照组:
  「`_freq_audit` 存活」本身**不足以**证明修复 —— 也可能是我误把 `_save()` 的
  白名单机制整个停掉了 (那会掩盖问题, 属 fail-open)。
  ⇒ 必须同时证明: **另一个非白名单键仍被抹除** (白名单机制仍在工作, 只是多了一个成员)。

在**临时副本**上做, 绝不碰 `data/trajectory_log.json`。
"""
import io
import json
import shutil
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
ROOT = Path(r'e:/quant')
sys.path.insert(0, str(ROOT / "research" / "factor_alchemy"))

REAL = ROOT / "data" / "trajectory_log.json"
CAL = ROOT / "data" / "lane_calibration.json"
TMP = ROOT / "data" / "_tmp_traj_freqaudit_20260919.json"

results = []


def chk(ok, label, detail=""):
    results.append((bool(ok), label, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}"
          + (f"   {detail}" if detail else ""))
    return bool(ok)


def raw_load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def find_entry(path, tid):
    for e in raw_load(path).get("trajectories", []):
        if e.get("trajectory_id") == tid:
            return e
    return None


def main():
    print("=" * 76)
    print("0. 准备: 拷贝副本 + 选一个 weekly 校准锚点")
    print("=" * 76)
    shutil.copy2(REAL, TMP)
    print(f"   {REAL.name} -> {TMP.name}  ({TMP.stat().st_size} bytes)")

    cal = raw_load(CAL)
    anchors = [p for p in cal.get("points", [])
               if str(p.get("natural_freq") or "").lower() == "weekly"
               and str(p.get("formula") or "").strip()]
    chk(len(anchors) > 0, "校准集有可用 weekly 锚点", f"n={len(anchors)}")
    anchor = anchors[0]
    print(f"   锚点: name={anchor.get('name')!r}  formula={str(anchor.get('formula'))[:70]!r}")

    # ── 前置: 现状应是「0 条带 _freq_audit」(9-18 写入后已被抹) ──
    pre = raw_load(TMP)
    n_pre = sum(1 for e in pre["trajectories"] if e.get("_freq_audit"))
    print(f"   修复前: 带 _freq_audit 的条目 = {n_pre} / {len(pre['trajectories'])}")

    print()
    print("=" * 76)
    print("1. 注入一条「可被铁证修正」的条目 + 一个对照组键")
    print("=" * 76)
    tgt = pre["trajectories"][0]
    tid = tgt["trajectory_id"]
    tgt["natural_freq"] = "daily"          # 触发 freq in ('daily','')
    tgt["jq_validated"] = True             # 触发 _is_jq_validated
    tgt["jq_return"] = 0.123
    tgt["expression"] = {"content": str(anchor["formula"])}   # 与锚点公式一致
    tgt["seed_factor"] = str(anchor.get("name") or "")
    tgt["_zz_control_key"] = "SHOULD_BE_WIPED"   # 🔴 对照组: 非白名单键, 必须被抹
    tgt.pop("_freq_audit", None)
    TMP.write_text(json.dumps(pre, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"   目标条目 trajectory_id = {tid}")
    print("   已置: natural_freq=daily, jq_validated=True, 公式=锚点公式")
    print("   对照组键 _zz_control_key = 'SHOULD_BE_WIPED'")

    print()
    print("=" * 76)
    print("2. 跑真实写方 audit_trajectory_freqs(fix=True)")
    print("=" * 76)
    from freq_audit import audit_trajectory_freqs
    rep = audit_trajectory_freqs(traj_path=TMP, cal_path=CAL, fix=True)
    print(f"   n_total={rep['n_total']} n_fixed={rep['n_fixed']} "
          f"traj_updated={rep['traj_updated']}")
    print(f"   fixed_names={rep['fixed_names'][:3]}")
    chk(rep["n_fixed"] >= 1, "写方命中并执行了铁证修正", f"n_fixed={rep['n_fixed']}")

    e = find_entry(TMP, tid)
    fa_written = str(e.get("_freq_audit") or "")
    chk(e.get("natural_freq") == "weekly", "natural_freq 已改为 weekly",
        f"={e.get('natural_freq')!r}")
    chk(bool(fa_written), "写方已写入 _freq_audit", f"={fa_written[:72]!r}")

    print()
    print("=" * 76)
    print("3. 🔴 决定性: _save() 整对象重建后, _freq_audit 是否存活")
    print("=" * 76)
    from trajectory_logger import TrajectoryLogger
    lg = TrajectoryLogger(TMP)
    tr = lg.trajectories.get(tid)
    chk(tr is not None, "from_dict 载入目标条目")
    chk(getattr(tr, "_freq_audit", None) == fa_written,
        "from_dict 恢复了 _freq_audit (三处同步 #3)",
        f"字段值={getattr(tr, '_freq_audit', None)!r}")
    chk(getattr(tr, "natural_freq", None) == "weekly",
        "from_dict 恢复了 natural_freq")

    lg.log_lesson(tid, "P-20260919-001 探针")   # 触发一次 _save() 全对象重建
    print("   已调用 log_lesson() ⇒ 触发 _save()")

    e2 = find_entry(TMP, tid)
    fa_after = str(e2.get("_freq_audit") or "")
    ctrl_after = e2.get("_zz_control_key")
    print()
    print("   -- 处理组 vs 对照组 --")
    chk(fa_after == fa_written,
        "🔴 修复目标: _freq_audit 穿越 _save() 存活",
        f"={fa_after[:72]!r}")
    chk(ctrl_after is None,
        "🔴 对照组: 其它非白名单键**仍被抹除** (白名单机制未被停掉)",
        f"_zz_control_key={ctrl_after!r}")

    print()
    print("=" * 76)
    print("4. 二次重载 (幂等/稳定性)")
    print("=" * 76)
    lg2 = TrajectoryLogger(TMP)
    tr2 = lg2.trajectories.get(tid)
    chk(getattr(tr2, "_freq_audit", None) == fa_written,
        "二次重载 _freq_audit 仍一致")
    keyfreq = sum(1 for x in raw_load(TMP)["trajectories"] if x.get("_freq_audit"))
    # 期望 = 修复前已有 + 本轮写方实际修正的条数 (🔴 不是写死 +1:
    # 实测 n_fixed=2 —— 生产数据里本就存在一条「可铁证修正但尚未修正」的条目)
    chk(keyfreq == n_pre + rep["n_fixed"],
        f"全库键频 = 修复前({n_pre}) + 本轮修正({rep['n_fixed']})",
        f"{n_pre} -> {keyfreq}")

    print()
    print("=" * 76)
    n_fail = sum(1 for ok, _, _ in results if not ok)
    print(f"RESULT: {len(results)-n_fail}/{len(results)} PASS"
          f"{'  🔴 HAS FAILURE' if n_fail else '  ✅ ALL GREEN'}")
    print("=" * 76)

    # 清理副本
    TMP.unlink(missing_ok=True)
    print(f"已删除临时副本 {TMP.name}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
