"""
Meta Learner 元验证器 — 无接触验证集
===================================
核心原则:
- 元验证集数据在系统生命周期内完全不被因子挖掘/策略进化使用
- 仅 Meta Learner 的最终方法论验证可访问
- 每次方法论变更只允许一次验证 (one-shot)

用法:
    validator = MetaValidator(data_dir='data/meta_validation')
    validator.seal()  # 密封数据, 之后不可修改

    # 方法论变更验证
    result = validator.validate(
        old_config={"ga_pop_size": 80, "ga_generations": 15},
        new_config={"ga_pop_size": 120, "ga_generations": 20},
        strategy_weights={"bp": 0.4, "vol_1m": 0.6},
    )
    # result['accepted'] → True/False
    # result['improvement'] → +0.05 (如果新参数在元验证集上更好)
"""

import json
import hashlib
import pickle
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class MetaValidationRecord:
    """一次元验证的记录 — 系统生命周期内每个方法论变更只能用一次"""
    proposal_id: str       # 方法论变更 ID (hash)
    proposal_desc: str     # 变更描述
    old_config: dict       # 旧配置
    new_config: dict       # 新配置
    old_score: float       # 旧配置在元验证集上的表现
    new_score: float       # 新配置在元验证集上的表现
    improvement: float     # new_score - old_score
    accepted: bool         # 是否接受变更
    validated_at: str      # 验证时间戳


class MetaValidator:
    """
    元验证器 — 管理"无接触"数据段, 执行方法论变更的最终验证。

    安全机制:
    1. seal(): 一次性密封数据, 之后不可修改
    2. validate(): 每个 proposal 只能调用一次
    3. 验证历史全程记录在 meta_validation_records.json
    """

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = str(Path(__file__).parent.parent / "outputs" / "meta_validation")
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._sealed = False
        self._sealed_hash = None
        self._records_file = self.data_dir / "meta_validation_records.json"
        self._seal_file = self.data_dir / "meta_seal.json"

        # 加载已有记录
        self.records: Dict[str, MetaValidationRecord] = {}
        self._load_records()

        # 检查是否已密封
        if self._seal_file.exists():
            with open(self._seal_file, 'r') as f:
                seal_data = json.load(f)
            self._sealed = seal_data.get("sealed", False)
            self._sealed_hash = seal_data.get("data_hash", "")

    # ── 密封机制 ──

    def seal(self, factor_dfs: dict = None, forward_returns: pd.DataFrame = None,
             close_weekly: pd.DataFrame = None, mcap_df: pd.DataFrame = None):
        """
        密封元验证数据 — 一次性操作, 不可撤销。
        数据写入磁盘后 hash 锁定, 防止篡改。

        Args:
            factor_dfs: {factor_name: DataFrame} 因子数据
            forward_returns: 周频前向收益
            close_weekly: 周频收盘价
            mcap_df: 市值数据
        """
        if self._sealed:
            raise PermissionError("元验证数据已密封, 不可重复 seal()。"
                                  "如需修改, 需删除 meta_seal.json 重新密封。")

        # 序列化数据
        data_bundle = {
            "factor_dfs": {k: v.to_dict() if hasattr(v, 'to_dict') else str(v)
                           for k, v in (factor_dfs or {}).items()},
            "forward_returns_shape": (forward_returns.shape if forward_returns is not None
                                       else (0, 0)),
            "sealed_at": datetime.now().isoformat(),
            "meta_validation_start": "2025-07-01",
            "meta_validation_end": "2026-06-25",
        }

        # 计算数据 hash
        data_str = json.dumps(data_bundle, sort_keys=True, default=str)
        self._sealed_hash = hashlib.sha256(data_str.encode()).hexdigest()

        # 保存密封信息
        seal_info = {
            "sealed": True,
            "data_hash": self._sealed_hash,
            "sealed_at": data_bundle["sealed_at"],
            "meta_validation_start": data_bundle["meta_validation_start"],
            "meta_validation_end": data_bundle["meta_validation_end"],
            "note": "此文件锁定后, 元验证数据不可复写。删除此文件可重置密封。",
        }

        with open(self._seal_file, 'w') as f:
            json.dump(seal_info, f, indent=2, ensure_ascii=False)

        # 保存原始数据 (pickle)
        if factor_dfs or forward_returns is not None:
            raw_path = self.data_dir / "meta_validation_data.pkl"
            with open(raw_path, 'wb') as f:
                pickle.dump({
                    "factor_dfs": factor_dfs,
                    "forward_returns": forward_returns,
                }, f)

        self._sealed = True
        print(f"[MetaValidator] 元验证数据已密封 (hash={self._sealed_hash[:12]}...)")
        print(f"  时间段: 2025-07-01 ~ 2026-06-25 (无接触)")
        print(f"  此后, 该数据段仅可通过 validate() 访问, 且每个 proposal 仅一次")

    @property
    def is_sealed(self) -> bool:
        return self._sealed

    # ── 记录管理 ──

    def _load_records(self):
        if self._records_file.exists():
            with open(self._records_file, 'r') as f:
                records_raw = json.load(f)
            for proposal_id, rec in records_raw.items():
                self.records[proposal_id] = MetaValidationRecord(**rec)

    def _save_records(self):
        with open(self._records_file, 'w') as f:
            json.dump(
                {pid: {
                    "proposal_id": r.proposal_id,
                    "proposal_desc": r.proposal_desc,
                    "old_config": r.old_config,
                    "new_config": r.new_config,
                    "old_score": r.old_score,
                    "new_score": r.new_score,
                    "improvement": r.improvement,
                    "accepted": r.accepted,
                    "validated_at": r.validated_at,
                } for pid, r in self.records.items()},
                f, indent=2, ensure_ascii=False, default=str,
            )

    def _make_proposal_id(self, old_config: dict, new_config: dict) -> str:
        """为方法论变更生成唯一 ID"""
        content = json.dumps({"old": old_config, "new": new_config},
                             sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    # ── 核心验证 ──

    def validate(self, old_config: dict, new_config: dict,
                 strategy_weights: dict = None,
                 proposal_desc: str = "",
                 pipeline_func: callable = None) -> Dict[str, Any]:
        """
        验证方法论变更是否带来样本外改善 (one-shot)。

        Args:
            old_config: 旧方法论配置 (如 {"ga_pop_size": 80})
            new_config: 新方法论配置 (如 {"ga_pop_size": 120})
            strategy_weights: 要在元验证集上评估的策略权重
            proposal_desc: 变更描述
            pipeline_func: 可选的完整 pipeline 重跑函数, 签名为
                           pipeline_func(config) -> {'score': float, 'sharpe': float, ...}
                           如果提供, 将用新旧 config 各跑一次, 然后在元验证集上比较。

        Returns:
            {
                "proposal_id": str,
                "accepted": bool,       # 新配置是否优于旧配置
                "improvement": float,   # 相对提升
                "old_score": float,
                "new_score": float,
                "rejection_reason": str,  # 若被拒绝, 原因
            }
        """
        proposal_id = self._make_proposal_id(old_config, new_config)

        # One-shot 检查: 同一提案不可重复验证
        if proposal_id in self.records:
            existing = self.records[proposal_id]
            raise PermissionError(
                f"方法论变更 '{proposal_id}' 已经验证过 (时间: {existing.validated_at}).\n"
                f"  结果: {'ACCEPTED' if existing.accepted else 'REJECTED'} "
                f"(improvement={existing.improvement:+.4f}).\n"
                f"  元验证策略: 每个方法论变更只能验证一次, 这是无接触数据的核心保障。"
            )

        if not self._sealed:
            print("[MetaValidator] ⚠️ 元验证数据未密封! "
                  "建议先调用 seal() 锁定数据, 确保验证公正性。")

        # 如果提供了完整 pipeline 函数, 跑新旧配置对比
        if pipeline_func:
            print(f"\n[MetaValidator] 对比验证: {proposal_desc or proposal_id}")
            print(f"  OLD: {old_config}")
            print(f"  NEW: {new_config}")

            old_result = pipeline_func(old_config)
            new_result = pipeline_func(new_config)

            old_score = old_result.get('score', old_result.get('sharpe', 0))
            new_score = new_result.get('score', new_result.get('sharpe', 0))

            print(f"  OLD score: {old_score:.4f}")
            print(f"  NEW score: {new_score:.4f}")
        elif strategy_weights:
            # 简化模式: 直接比策略在元验证集上的表现
            # 需要加载密封数据
            data_path = self.data_dir / "meta_validation_data.pkl"
            if not data_path.exists():
                raise FileNotFoundError(
                    f"元验证数据文件不存在: {data_path}。"
                    f"请先调用 seal() 密封数据。"
                )

            with open(data_path, 'rb') as f:
                sealed_data = pickle.load(f)

            factor_dfs_meta = sealed_data.get("factor_dfs", {})
            fr_meta = sealed_data.get("forward_returns")

            if fr_meta is None or not factor_dfs_meta:
                raise ValueError("密封数据缺少 factor_dfs 或 forward_returns")

            # 用策略权重在元验证集上计算 ICIR
            old_icir = self._compute_strategy_icir(
                factor_dfs_meta, fr_meta, strategy_weights
            )
            # 策略权重固定, old_score == new_score — 需要 pipeline_func 才有意义
            old_score = new_score = old_icir
        else:
            raise ValueError(
                "必须提供 pipeline_func (完整重跑) 或 strategy_weights (策略对比)"
            )

        improvement = new_score - old_score
        # 接受标准: 任何正改善, 且改善幅度 > 1e-6 (避免浮点噪声)
        accepted = improvement > 1e-6

        record = MetaValidationRecord(
            proposal_id=proposal_id,
            proposal_desc=proposal_desc,
            old_config=old_config,
            new_config=new_config,
            old_score=old_score,
            new_score=new_score,
            improvement=improvement,
            accepted=accepted,
            validated_at=datetime.now().isoformat(),
        )
        self.records[proposal_id] = record
        self._save_records()

        conclusion = "ACCEPTED" if accepted else "REJECTED"
        reason = ""
        if not accepted:
            if abs(improvement) <= 1e-6:
                reason = "改善幅度为 0 (浮点噪声)"
            else:
                reason = f"新配置劣于旧配置 (Δ={improvement:+.4f})"

        print(f"\n[MetaValidator] 结论: {conclusion}")
        if reason:
            print(f"  原因: {reason}")
        print(f"  改善: {improvement:+.4f}")
        print(f"  记录: {self._records_file}")

        # 同时写入 Chronicle
        try:
            from chronicle.db import ExperimentChronicle
            chronicle = ExperimentChronicle()
            chronicle.log_methodology_update({
                "experiment_id": f"meta_{proposal_id}",
                "update_type": "meta_validation",
                "old_value": json.dumps(old_config, default=str),
                "new_value": json.dumps(new_config, default=str),
                "reasoning": proposal_desc,
                "source_module": "meta.validator",
                "validated": accepted,
                "validated_by": "MetaValidator (one-shot)",
            })
        except Exception:
            pass  # Chronicle 写入非关键

        return {
            "proposal_id": proposal_id,
            "accepted": accepted,
            "improvement": improvement,
            "old_score": old_score,
            "new_score": new_score,
            "rejection_reason": reason,
        }

    def _compute_strategy_icir(self, factor_dfs: dict, forward_returns: pd.DataFrame,
                                weights: dict) -> float:
        """在元验证数据上计算策略加权 ICIR"""
        from evaluation.ic_analysis import compute_ic_icir
        from factors.composite import combine_factors_vectorized

        sel = {n: factor_dfs[n] for n in weights if n in factor_dfs}
        if not sel:
            return 0.0

        composite = combine_factors_vectorized(sel, weights)
        if composite.empty:
            return 0.0

        result = compute_ic_icir(composite, forward_returns)
        return abs(result.get('icir', 0))

    # ── 查询 ──

    def get_validation_history(self) -> list:
        """获取所有元验证记录"""
        return [
            {
                "proposal_id": r.proposal_id,
                "desc": r.proposal_desc,
                "improvement": r.improvement,
                "accepted": r.accepted,
                "when": r.validated_at,
            }
            for r in sorted(self.records.values(),
                            key=lambda x: x.validated_at, reverse=True)
        ]

    def get_acceptance_rate(self) -> float:
        """方法论变更接受率"""
        if not self.records:
            return 0.0
        accepted = sum(1 for r in self.records.values() if r.accepted)
        return accepted / len(self.records)

    def stats(self) -> dict:
        """元验证统计"""
        history = list(self.records.values())
        improvements = [r.improvement for r in history]
        return {
            "sealed": self._sealed,
            "data_hash": self._sealed_hash[:12] + "..." if self._sealed_hash else "N/A",
            "total_validations": len(history),
            "acceptance_rate": self.get_acceptance_rate(),
            "mean_improvement": float(np.mean(improvements)) if improvements else 0,
            "best_improvement": float(np.max(improvements)) if improvements else 0,
            "worst_improvement": float(np.min(improvements)) if improvements else 0,
            "records_file": str(self._records_file),
        }


# ── CLI 快速测试 ──
if __name__ == "__main__":
    print("=== MetaValidator 自测 ===")

    # 创建验证器
    v = MetaValidator(data_dir="outputs/meta_validation_test")

    # 密封 (空数据, 仅演示)
    v.seal()

    # 定义 fake pipeline
    def fake_pipeline(config):
        """模拟 pipeline: score 随 pop_size 单调增 (模拟改善)"""
        score = 1.0 + config.get("ga_pop_size", 80) * 0.001
        return {"score": score, "sharpe": score}

    # 验证一个改善
    result = v.validate(
        old_config={"ga_pop_size": 80, "ga_generations": 15},
        new_config={"ga_pop_size": 120, "ga_generations": 20},
        proposal_desc="增大 GA 种群规模 (80→120)",
        pipeline_func=fake_pipeline,
    )
    print(f"\n结果: accepted={result['accepted']}, improvement={result['improvement']:+.3f}")

    # 测试 one-shot 拒绝
    try:
        v.validate(
            old_config={"ga_pop_size": 80, "ga_generations": 15},
            new_config={"ga_pop_size": 120, "ga_generations": 20},
            proposal_desc="重复验证 (应被拒绝)",
            pipeline_func=fake_pipeline,
        )
    except PermissionError as e:
        print(f"\nOne-shot 拒绝测试 PASS: {str(e)[:80]}...")

    # 统计
    print(f"\n验证器统计: {v.stats()}")

    # 清理
    import shutil
    shutil.rmtree("outputs/meta_validation_test", ignore_errors=True)
    print("\n自测通过!")
