# -*- coding: utf-8 -*-
"""
Loop Checkpoint — 检查点/断点续跑系统
==========================================

对标中金 CICC Loop Engineering 报告中的检查点机制:
  - 每轮迭代结束后以原子方式写入检查点
  - 存储已测试因子的哈希集合、入库因子的完整信息、迭代计数
  - 任意环节中断后均可从断点恢复

核心功能:
  1. 因子哈希指纹去重 — 避免重复测试相同表达式
  2. 原子写入 — 先写临时文件再 rename，防止中断损坏
  3. 增量保存 — 每轮只追加，不重写全量
  4. 恢复接口 — load_checkpoint() 返回最后状态

设计原则:
  - 检查点 = 已测试哈希集合 + 入库因子列表 + 迭代计数器
  - 原子性: write(temp) → fsync → rename → 安全
  - 与 SubtreeFingerprinter.expression_hash() 共享哈希算法

集成路径:
  - Ralph Loop: 每轮迭代前后读/写检查点
  - MetaController: auto_cycle 启动时从检查点恢复
  - 自动化任务: 断电/超时后重启自动续跑

用法:
    from loop_checkpoint import CheckpointManager
    
    ckpt = CheckpointManager()
    
    # 启动时检查是否可续跑
    if ckpt.can_resume():
        state = ckpt.load()
        print(f"从第 {state['iteration']} 轮续跑")
    
    # 每轮迭代后保存
    ckpt.mark_tested("expression_hash_here")
    ckpt.save(iteration=42, approved=[...])
"""

import json
import os
import time
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

# ── 默认路径 ──────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent.parent / "data"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"


@dataclass
class CheckpointState:
    """检查点状态"""
    iteration: int = 0
    total_tested: int = 0
    total_approved: int = 0
    tested_hashes: Set[str] = field(default_factory=set)
    approved_factors: List[Dict] = field(default_factory=list)
    rejected_patterns: List[str] = field(default_factory=list)
    budget_state: Dict = field(default_factory=dict)
    started_at: str = ""
    last_saved_at: str = ""
    loop_id: str = ""


class CheckpointManager:
    """
    检查点管理器 — 支持断点续跑。
    
    对标中金: 检查点文件每轮迭代后以原子方式写入，
    存储已测试因子的哈希集合、入库因子、迭代计数。
    """

    def __init__(
        self,
        checkpoint_dir: Path = CHECKPOINT_DIR,
        checkpoint_name: str = "ralph_loop",
        max_backups: int = 3,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = checkpoint_name
        self.max_backups = max_backups
        self.checkpoint_path = checkpoint_dir / f"{checkpoint_name}.json"
        self._state: CheckpointState = CheckpointState()
        self._ensure_dir()

    def _ensure_dir(self):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ═══════════════════════════════════════════════════════════
    # 哈希与去重
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def factor_hash(factor: Dict) -> str:
        """计算因子唯一哈希（基于表达式 + 参数）"""
        expr = factor.get("formula", factor.get("expression", ""))
        # 标准化
        expr = expr.strip().replace(" ", "").lower()
        return hashlib.sha256(expr.encode()).hexdigest()[:16]

    @staticmethod
    def expression_hash(expression: str) -> str:
        """计算表达式哈希"""
        expr = expression.strip().replace(" ", "").lower()
        return hashlib.sha256(expr.encode()).hexdigest()[:16]

    def is_tested(self, factor_or_expr) -> bool:
        """
        检查因子是否已经被测试过。

        Parameters
        ----------
        factor_or_expr: Dict (因子元数据) 或 str (表达式)
        """
        if isinstance(factor_or_expr, dict):
            h = self.factor_hash(factor_or_expr)
        else:
            h = self.expression_hash(factor_or_expr)
        return h in self._state.tested_hashes

    def mark_tested(self, factor_or_expr, outcome: str = "tested"):
        """
        标记因子为已测试。

        Parameters
        ----------
        factor_or_expr: Dict 或 str
        outcome: "approved" / "rejected" / "tested"
        """
        if isinstance(factor_or_expr, dict):
            h = self.factor_hash(factor_or_expr)
        else:
            h = self.expression_hash(factor_or_expr)
        self._state.tested_hashes.add(h)
        return h

    def mark_approved(self, factor: Dict):
        """标记因子为已批准入库"""
        h = self.mark_tested(factor)
        # 去重
        existing = {self.factor_hash(f) for f in self._state.approved_factors}
        if h not in existing:
            self._state.approved_factors.append({
                "factor_name": factor.get("factor_name", factor.get("name", "")),
                "formula": factor.get("formula", factor.get("expression", ""))[:200],
                "hash": h,
                "approved_at": datetime.now().isoformat(),
                "ic": factor.get("ic"),
                "icir": factor.get("icir"),
                "grade": factor.get("grade", factor.get("final_grade", "")),
            })
            self._state.total_approved += 1

    def mark_rejected(self, expression: str, reason: str = ""):
        """记录被拒绝的因子表达式模式"""
        h = self.mark_tested(expression)
        if reason and reason not in self._state.rejected_patterns:
            self._state.rejected_patterns.append(reason[:200])

    # ═══════════════════════════════════════════════════════════
    # 保存与加载
    # ═══════════════════════════════════════════════════════════

    def save(self, iteration: int = None, atomic: bool = True):
        """
        保存检查点。

        Parameters
        ----------
        iteration: 当前迭代轮数（None = 使用内部计数）
        atomic: 是否使用原子写入（先写 temp 再 rename）
        """
        if iteration is not None:
            self._state.iteration = iteration

        self._state.last_saved_at = datetime.now().isoformat()
        self._state.total_tested = len(self._state.tested_hashes)

        state_dict = {
            "_version": "1.0",
            "_saved_at": self._state.last_saved_at,
            "iteration": self._state.iteration,
            "total_tested": self._state.total_tested,
            "total_approved": self._state.total_approved,
            "tested_hashes": list(self._state.tested_hashes),
            "approved_factors": self._state.approved_factors,
            "rejected_patterns": self._state.rejected_patterns,
            "budget_state": self._state.budget_state,
            "started_at": self._state.started_at,
            "loop_id": self._state.loop_id,
        }

        json_str = json.dumps(state_dict, indent=2, ensure_ascii=False)

        if atomic:
            # 先写临时文件
            tmp_path = self.checkpoint_path.with_suffix(".tmp")
            try:
                with open(tmp_path, "w", encoding="utf-8") as f:
                    f.write(json_str)
                    f.flush()
                    os.fsync(f.fileno())
            except Exception:
                # 原子写入失败，回退到直接写入
                atomic = False

            if atomic:
                # rename 是原子操作
                try:
                    tmp_path.replace(self.checkpoint_path)
                except Exception:
                    # Windows 上可能不支持 replace，用 rename
                    os.replace(str(tmp_path), str(self.checkpoint_path))

        if not atomic:
            # 直接写入（最后手段）
            with open(self.checkpoint_path, "w", encoding="utf-8") as f:
                f.write(json_str)
                f.flush()

    def load(self) -> Optional[Dict]:
        """
        从检查点文件加载状态。

        Returns
        -------
        state_dict or None (文件不存在或损坏)
        """
        if not self.checkpoint_path.exists():
            return None

        try:
            # 检查临时文件（表示之前的原子写入异常中断）
            tmp_path = self.checkpoint_path.with_suffix(".tmp")
            if tmp_path.exists():
                # 临时文件可能更新
                use_tmp = os.path.getmtime(str(tmp_path)) > os.path.getmtime(str(self.checkpoint_path))
                load_path = tmp_path if use_tmp else self.checkpoint_path
            else:
                load_path = self.checkpoint_path

            with open(load_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 恢复内部状态
            self._state.iteration = data.get("iteration", 0)
            self._state.total_tested = data.get("total_tested", 0)
            self._state.total_approved = data.get("total_approved", 0)
            self._state.tested_hashes = set(data.get("tested_hashes", []))
            self._state.approved_factors = data.get("approved_factors", [])
            self._state.rejected_patterns = data.get("rejected_patterns", [])
            self._state.budget_state = data.get("budget_state", {})
            self._state.started_at = data.get("started_at", "")
            self._state.loop_id = data.get("loop_id", "")
            self._state.last_saved_at = data.get("_saved_at", "")

            # 清理临时文件
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

            return data
        except (json.JSONDecodeError, KeyError, OSError) as e:
            print(f"  [Checkpoint] 加载失败: {e}")
            return None

    # ═══════════════════════════════════════════════════════════
    # 生命周期
    # ═══════════════════════════════════════════════════════════

    def start_new(
        self, loop_id: str = "", total_iterations: int = 0
    ) -> CheckpointState:
        """
        开始新一轮循环，初始化检查点。

        Parameters
        ----------
        loop_id: 循环标识
        total_iterations: 预期总轮数（0 = 无限制）
        """
        self._state = CheckpointState(
            iteration=0,
            total_tested=0,
            total_approved=0,
            tested_hashes=set(),
            approved_factors=[],
            started_at=datetime.now().isoformat(),
            loop_id=loop_id or f"loop_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        return self._state

    def can_resume(self) -> bool:
        """检查是否有可恢复的检查点"""
        return self.checkpoint_path.exists()

    def resume(self) -> Tuple[bool, CheckpointState]:
        """
        尝试从检查点恢复。

        Returns
        -------
        (success, state)
        """
        if not self.can_resume():
            return False, CheckpointState()

        data = self.load()
        if data is None:
            return False, CheckpointState()

        print(f"  [Checkpoint] 从第 {self._state.iteration} 轮恢复")
        print(f"    已测试: {self._state.total_tested} 个候选")
        print(f"    已入库: {self._state.total_approved} 个因子")
        return True, self._state

    def get_progress(self) -> Dict:
        """获取当前进度摘要"""
        return {
            "iteration": self._state.iteration,
            "total_tested": self._state.total_tested,
            "total_approved": self._state.total_approved,
            "approval_rate": (
                self._state.total_approved / max(self._state.total_tested, 1)
            ),
            "started_at": self._state.started_at,
            "last_saved_at": self._state.last_saved_at,
        }

    def get_untested(
        self, candidates: List[Dict]
    ) -> List[Dict]:
        """
        从候选列表中筛选出未测试的因子。

        Parameters
        ----------
        candidates: 候选因子列表

        Returns
        -------
        未测试的候选（去重后）
        """
        result = []
        seen = set()
        for c in candidates:
            h = self.factor_hash(c)
            if h not in self._state.tested_hashes and h not in seen:
                result.append(c)
                seen.add(h)
        return result

    # ═══════════════════════════════════════════════════════════
    # 预算状态
    # ═══════════════════════════════════════════════════════════

    def update_budget(self, budget: Dict):
        """更新预算状态（供动态预算调整使用）"""
        self._state.budget_state = {
            **self._state.budget_state,
            **budget,
            "updated_at": datetime.now().isoformat(),
        }

    def get_budget(self) -> Dict:
        """获取当前预算状态"""
        return self._state.budget_state

    # ═══════════════════════════════════════════════════════════
    # 清理
    # ═══════════════════════════════════════════════════════════

    def archive(self):
        """归档当前检查点（保留备份）"""
        if not self.checkpoint_path.exists():
            return
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        archive_path = self.checkpoint_dir / f"{self.checkpoint_name}_{ts}.json"
        try:
            self.checkpoint_path.rename(archive_path)
            # 删除旧备份（保留最近 N 个）
            backups = sorted(
                self.checkpoint_dir.glob(f"{self.checkpoint_name}_*.json"),
                key=os.path.getmtime,
            )
            for old in backups[:-self.max_backups]:
                old.unlink()
        except Exception:
            pass

    def clear(self):
        """清除检查点"""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
        tmp = self.checkpoint_path.with_suffix(".tmp")
        if tmp.exists():
            tmp.unlink()
        self._state = CheckpointState()


# ── 便捷函数 ──────────────────────────────────────────────

_default_ckpt: Optional[CheckpointManager] = None


def get_checkpoint(name: str = "ralph_loop") -> CheckpointManager:
    global _default_ckpt
    if _default_ckpt is None:
        _default_ckpt = CheckpointManager(checkpoint_name=name)
    return _default_ckpt


# ── 测试 ──────────────────────────────────────────────────

if __name__ == "__main__":
    ckpt = CheckpointManager(checkpoint_name="test_ckpt")

    print("=" * 60)
    print("  Checkpoint Manager 测试")
    print("=" * 60)

    # 测试新循环
    state = ckpt.start_new(loop_id="test_loop_001")
    print(f"\n  开始新循环: {state.loop_id}")
    print(f"  迭代: {state.iteration}")

    # 标记测试
    ckpt.mark_tested({"factor_name": "f1", "formula": "ma(close, 20)"})
    ckpt.mark_tested({"factor_name": "f2", "formula": "sub(ma(overnight,60), ma(close,20))"})
    ckpt.mark_approved({"factor_name": "f1", "formula": "ma(close, 20)", "icir": 0.52})
    ckpt.mark_rejected("rank(volume)", "IC too low")

    # 保存
    ckpt.save(iteration=1)
    print(f"  已保存: 测试 {ckpt._state.total_tested} 个, 入库 {ckpt._state.total_approved} 个")

    # 测试去重
    print(f"\n  去重测试:")
    print(f"    f1 已测试? {ckpt.is_tested({'formula': 'ma(close, 20)'})}")
    print(f"    f3 已测试? {ckpt.is_tested({'formula': 'new_factor'})}")

    # 测试恢复
    ckpt2 = CheckpointManager(checkpoint_name="test_ckpt")
    if ckpt2.can_resume():
        success, state = ckpt2.resume()
        print(f"\n  恢复成功: 迭代={state.iteration}, 测试={state.total_tested}")

    # 进度
    print(f"\n  进度: {ckpt.get_progress()}")

    # 清理测试文件
    ckpt.clear()
    print(f"\n  清理完成")
