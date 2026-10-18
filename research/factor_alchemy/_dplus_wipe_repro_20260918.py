# -*- coding: utf-8 -*-
"""零副作用复现: d_plus_applied 在 MiningTrajectory 往返中被抹掉。
全程内存操作 + 只用临时文件, 绝不触碰 data/trajectory_log.json。"""
import io, json, sys, os, tempfile, datetime
sys.path.insert(0, r"e:\quant\research\factor_alchemy")
from trajectory_logger import MiningTrajectory, TrajectoryLogger

TRAJ = r"e:\quant\data\trajectory_log.json"

print("=" * 78)
print("A. 那唯一 1 条含富字段的轨迹是谁")
print("=" * 78)
d = json.load(io.open(TRAJ, encoding="utf-8"))
rich = [t for t in d["trajectories"] if "d_plus_applied" in t]
for t in rich:
    print("  trajectory_id :", t.get("trajectory_id"))
    print("  seed_factor   :", t.get("seed_factor"))
    print("  phase         :", t.get("phase"))
    print("  final_outcome :", t.get("final_outcome"))
    print("  d_plus_applied:", t.get("d_plus_applied"), "| d_plus_at:", t.get("d_plus_at"))
    print("  富字段        :", {k: t[k] for k in ("jq_maxdd","jq_ic","jq_icir","jq_rating",
                                                  "jq_plus_ic_pct","jq_backtest_period","jq_notes")
                                if k in t})
print("  (该条即 2026-09-18 直写 JSON 的那条; 全库其余 2297 条均无这些键)")

print()
print("=" * 78)
print("B. 零副作用复现 —— 往返 to_dict() 看键位存亡")
print("=" * 78)
sample = rich[0]
print("  [输入] 原始 JSON 键数 :", len(sample))
KEEP = ["d_plus_applied","d_plus_at","jq_maxdd","jq_ic","jq_icir","jq_rating","jq_notes"]
print("  [输入] 富字段        :", {k: sample.get(k) for k in KEEP})

# 模拟 TrajectoryLogger 的一次 _save(): 载入 → to_dict() 序列化
obj = MiningTrajectory.from_dict(sample)
out = obj.to_dict()
print("  [输出] to_dict 键数  :", len(out))
lost = [k for k in KEEP if k not in out]
print("  [输出] 🔴 被丢弃的键  :", lost)
print("  [输出] 仍保留的 jq 键:", [k for k in out if k.startswith("jq_")])

print()
print("=" * 78)
print("C. 通过真实 TrajectoryLogger 往返 (临时路径, 不碰真数据)")
print("=" * 78)
tmp = os.path.join(tempfile.gettempdir(), "_traj_wipe_probe_20260918.json")
tmpdir = os.path.dirname(tmp)
_p = type(obj.__class__.__mro__)  # noop
logger = TrajectoryLogger.__new__(TrajectoryLogger)   # 绕过 __init__/_load, 不读真文件
logger.path = __import__("pathlib").Path(tmp)
logger.trajectories = {obj.trajectory_id: obj}
logger._save()                       # ← 一次 _save()
after = json.load(io.open(tmp, encoding="utf-8"))["trajectories"][0]
print("  _save() 之后落盘键数 :", len(after))
print("  🔴 富字段存活情况    :", {k: (k in after) for k in KEEP})
print("  文件大小             :", os.path.getsize(tmp), "bytes ->", tmp)
os.remove(tmp)
print("  已清理临时文件")

print()
print("=" * 78)
print("D. _save() 的 8 个调用点 = 抹除发生频率")
print("=" * 78)
print("  start_trajectory / log_hypothesis / log_expression / log_backtest(x2)")
print("  / line451 / log_lesson / finalize  —— 每次调用都是一次全量白名单重写")
print("  ⇒ trigger_d_plus 写入的 d_plus_applied 存活上限 = 到下一次 _save() 为止")
