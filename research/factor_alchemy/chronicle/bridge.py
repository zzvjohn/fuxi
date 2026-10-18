"""
Chronicle ↔ Vault 双向桥接
===========================
提供跨两库的统一查询和自动同步。

核心理念:
- Chronicle 记录「发生了什么」(What happened)
- Vault    记录「因子现在是什么状态」(What is the current state)
- Bridge   确保两者一致同步

数据流:
  实验完成 → Chronicle.complete_experiment()
           → Bridge.sync_to_vault()
           → Vault 更新因子性能 + 血统 + regime

  因子状态变更(vault.deprecate_factor) → Bridge.log_to_chronicle()
                                       → Chronicle.log_methodology_note()
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any


class ChronicleVaultBridge:
    """Chronicle 和 Vault 之间的双向同步桥"""

    def __init__(self, chronicle, vault):
        """
        Args:
            chronicle: ExperimentChronicle 实例
            vault: FactorVault 实例
        """
        self.chronicle = chronicle
        self.vault = vault

    # ── Chronicle → Vault 同步 ──

    def sync_experiment_to_vault(self, experiment_id: str) -> dict:
        """将一次实验的所有因子性能同步到 Vault。

        Returns:
            dict: {"synced_factors": N, "new_factors": [...], "updated_factors": [...]}
        """
        result = {"synced_factors": 0, "new_factors": [], "updated_factors": []}

        # 从 Chronicle 获取该实验的因子性能
        factors = self.chronicle.get_top_factors(
            experiment_id, min_icir=-99, limit=200
        )
        if not factors:
            return result

        for f in factors:
            fid = f["factor_id"]
            existing = self.vault.get_factor(fid)

            if existing:
                # 更新已有因子
                perf = (json.loads(existing.get("performance", "{}"))
                        if isinstance(existing.get("performance"), str)
                        else existing.get("performance", {}) or {})
                perf["icir_mean"] = f.get("icir_mean", 0)
                perf["icir_std"] = f.get("icir_std", 0)
                perf["last_experiment"] = experiment_id
                self.vault.register_factor(factor_id=fid, performance=perf)
                result["updated_factors"].append(fid)
            else:
                # 新因子入库
                self.vault.register_factor(
                    factor_id=fid,
                    name_cn=f.get("factor_id", fid),  # 暂用 id 作名称
                    category=f.get("category", ""),
                    source="experiment",
                    performance={
                        "icir_mean": f.get("icir_mean", 0),
                        "icir_std": f.get("icir_std", 0),
                        "ic_mean": f.get("ic_mean", 0),
                        "last_experiment": experiment_id,
                    },
                )
                result["new_factors"].append(fid)

            result["synced_factors"] += 1

        # 同步策略结果
        strategies = self.chronicle.get_best_strategies(min_sharpe=-99, limit=50)
        for s in strategies:
            if s.get("experiment_id") != experiment_id:
                continue
            # 策略中的因子组合写血统
            weights = s.get("factor_weights", {})
            if isinstance(weights, str):
                weights = json.loads(weights)
            for fid in weights:
                self.vault.set_lineage(
                    factor_id=fid,
                    parent_id=f"strategy_{s.get('strategy_name', 'unknown')}",
                    derivation="combination",
                )

        return result

    def sync_regime_data(self, experiment_id: str,
                          regime_data: List[dict]):
        """批量同步 regime 数据到 Vault。

        regime_data: [{"factor_id": "...", "regime": "bull",
                        "icir": 0.5}, ...]
        """
        for item in regime_data:
            item["experiment_id"] = experiment_id
        self.vault.batch_annotate_regime(regime_data)

    # ── Vault → Chronicle 同步 ──

    def log_factor_deprecation(self, factor_id: str,
                                superseded_by: str,
                                reason: str = ""):
        """因子被弃用时，写一条方法论笔记到 Chronicle"""
        note = (f"[Vault] 因子弃用: {factor_id} → {superseded_by}\n"
                f"  原因: {reason or '性能衰退 / 被更优因子替代'}\n"
                f"  时间: {datetime.now().isoformat()}")
        self.chronicle.log_methodology_note(
            category="factor_deprecation",
            note=note,
            tags=["vault", "deprecation", factor_id],
        )

    def log_factor_status_change(self, factor_id: str,
                                  old_status: str, new_status: str):
        """因子状态变更记录到 Chronicle"""
        note = (f"[Vault] 因子状态变更: {factor_id}\n"
                f"  {old_status} → {new_status}\n"
                f"  时间: {datetime.now().isoformat()}")
        self.chronicle.log_methodology_note(
            category="factor_status",
            note=note,
            tags=["vault", "status_change", factor_id],
        )

    # ── 统一查询 ──

    def unified_query(self, factor_id: str) -> dict:
        """跨 Chronicle 和 Vault 查询某个因子的完整历史"""
        result = {
            "factor_id": factor_id,
            "vault_info": self.vault.get_factor(factor_id),
            "regime_profile": self.vault.get_regime_profile(factor_id),
            "experiment_history": self._get_factor_experiments(factor_id),
            "lineage_tree": self.vault.get_lineage_tree(factor_id, max_depth=5),
        }
        return result

    def _get_factor_experiments(self, factor_id: str) -> List[dict]:
        """查询因子在历史实验中的表现变化"""
        experiments = self.chronicle.get_recent_experiments(limit=50)
        history = []
        for exp in experiments:
            fid = exp.get("id", "")
            summary = self.chronicle.get_experiment_summary(fid)
            factor_icirs = summary.get("factor_icirs", {})
            if factor_id in factor_icirs:
                history.append({
                    "experiment_id": fid,
                    "icir": factor_icirs[factor_id],
                    "date": exp.get("started_at", ""),
                    "version": exp.get("version", ""),
                })
        return history

    def get_factor_trend(self, factor_id: str) -> List[dict]:
        """追踪因子 ICIR 的时间序列趋势"""
        history = self._get_factor_experiments(factor_id)
        return sorted(history, key=lambda x: x.get("date", ""))

    def cross_reference_strategies(self, factor_id: str) -> List[dict]:
        """查询哪些策略使用了该因子"""
        experiments = self.chronicle.get_recent_experiments(limit=50)
        usages = []
        for exp in experiments:
            fid = exp.get("id", "")
            strategies = self.chronicle.get_best_strategies(min_sharpe=-99, limit=50)
            for s in strategies:
                if s.get("experiment_id") != fid:
                    continue
                weights = s.get("factor_weights", {})
                if isinstance(weights, str):
                    weights = json.loads(weights)
                if factor_id in weights:
                    usages.append({
                        "experiment_id": fid,
                        "strategy": s.get("strategy_name", "unknown"),
                        "weight": weights[factor_id],
                        "phase5_sharpe": s.get("phase5_sharpe", 0),
                    })
        return usages

    # ── 批量同步 ──

    def full_sync(self, experiment_id: str,
                   regime_data: List[dict] = None) -> dict:
        """完整同步：实验 → Vault (性能+regime+血统)"""
        result = self.sync_experiment_to_vault(experiment_id)
        if regime_data:
            self.sync_regime_data(experiment_id, regime_data)
            result["regime_records"] = len(regime_data)
        return result


# ── 便捷工厂 ──

def create_bridge(chronicle_db_path: str = None,
                   vault_dir: str = None) -> ChronicleVaultBridge:
    """一行创建 bridge 实例"""
    from chronicle.db import ExperimentChronicle
    from vault import FactorVault

    if chronicle_db_path is None:
        chronicle_db_path = str(
            Path(__file__).parent.parent / "outputs" / "chronicle.db"
        )
    if vault_dir is None:
        vault_dir = str(
            Path(__file__).parent.parent / "data" / "vault"
        )

    chronicle = ExperimentChronicle(chronicle_db_path)
    vault = FactorVault(vault_dir)
    return ChronicleVaultBridge(chronicle, vault)
