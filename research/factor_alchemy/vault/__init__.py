"""
Factor Vault — 因子宝库
======================
因子全生命周期管理: 元数据存储、向量嵌入、相似度搜索、冗余检测。

存储后端: SQLite (元数据) + pickled numpy arrays (嵌入向量)
轻量化设计 — 无需 FAISS 安装也能跑 (fallback 到纯 numpy 余弦相似度)。

用法:
    from vault import FactorVault
    vault = FactorVault("data/vault")
    vault.import_from_pool("data/passed_factor_pool.csv")
    similar = vault.find_similar("vol_1m", top_k=5)
"""
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any
import sqlite3
from contextlib import contextmanager


class FactorVault:
    """因子宝库 — 主接口"""

    def __init__(self, vault_dir: str = None):
        if vault_dir is None:
            vault_dir = Path(__file__).parent.parent / "data" / "vault"
        self.vault_dir = Path(vault_dir)
        self.vault_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.vault_dir / "vault.db"
        self.embedding_path = self.vault_dir / "embeddings.pkl"
        self._embeddings: Dict[str, np.ndarray] = {}
        self._init_schema()
        self._load_embeddings()

        # 尝试加载 FAISS
        self._use_faiss = False
        try:
            import faiss
            self._faiss = faiss
            self._faiss_index = None
            self._faiss_id_map = {}
            self._use_faiss = True
        except ImportError:
            pass

    @contextmanager
    def _get_conn(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS factors (
                    factor_id TEXT PRIMARY KEY,
                    name_cn TEXT DEFAULT '',
                    category TEXT DEFAULT '',
                    source TEXT DEFAULT 'manual',
                    status TEXT DEFAULT 'active',
                    superseded_by TEXT,
                    version INTEGER DEFAULT 1,
                    expression JSON DEFAULT '{}',
                    performance JSON DEFAULT '{}',
                    regime_profile JSON DEFAULT '{}',
                    novelty_score REAL DEFAULT 0,
                    lineage JSON DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS factor_icir_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    factor_id TEXT NOT NULL,
                    experiment_id TEXT NOT NULL,
                    icir_mean REAL DEFAULT 0,
                    icir_std REAL DEFAULT 0,
                    regime TEXT DEFAULT 'all',
                    recorded_at TEXT NOT NULL,
                    FOREIGN KEY (factor_id) REFERENCES factors(factor_id)
                );

                CREATE INDEX IF NOT EXISTS idx_fv_category ON factors(category);
                CREATE INDEX IF NOT EXISTS idx_fv_status ON factors(status);
                CREATE INDEX IF NOT EXISTS idx_fv_source ON factors(source);
                CREATE INDEX IF NOT EXISTS idx_fih_factor ON factor_icir_history(factor_id);
            """)

    def _load_embeddings(self):
        if self.embedding_path.exists():
            with open(self.embedding_path, "rb") as f:
                self._embeddings = pickle.load(f)

    def _save_embeddings(self):
        with open(self.embedding_path, "wb") as f:
            pickle.dump(self._embeddings, f)

    # ── 因子元数据 CRUD ──

    def register_factor(self, factor_id: str, **kwargs) -> bool:
        """注册新因子 (或更新已有因子)"""
        now = datetime.now().isoformat()
        with self._get_conn() as conn:
            existing = conn.execute(
                "SELECT version FROM factors WHERE factor_id=?", (factor_id,)
            ).fetchone()

            if existing:
                version = existing["version"] + 1
                conn.execute(
                    """UPDATE factors SET
                       name_cn=?, category=?, source=?, status=?,
                       expression=?, performance=?, regime_profile=?,
                       novelty_score=?, lineage=?, version=?, updated_at=?
                       WHERE factor_id=?""",
                    (
                        kwargs.get("name_cn", ""),
                        kwargs.get("category", ""),
                        kwargs.get("source", "manual"),
                        kwargs.get("status", "active"),
                        json.dumps(kwargs.get("expression", {}), ensure_ascii=False),
                        json.dumps(kwargs.get("performance", {}), ensure_ascii=False),
                        json.dumps(kwargs.get("regime_profile", {}), ensure_ascii=False),
                        kwargs.get("novelty_score", 0),
                        json.dumps(kwargs.get("lineage", {}), ensure_ascii=False),
                        version, now, factor_id,
                    )
                )
            else:
                conn.execute(
                    """INSERT INTO factors (factor_id, name_cn, category, source, status,
                       expression, performance, regime_profile, novelty_score, lineage,
                       version, created_at, updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        factor_id,
                        kwargs.get("name_cn", ""),
                        kwargs.get("category", ""),
                        kwargs.get("source", "manual"),
                        kwargs.get("status", "active"),
                        json.dumps(kwargs.get("expression", {}), ensure_ascii=False),
                        json.dumps(kwargs.get("performance", {}), ensure_ascii=False),
                        json.dumps(kwargs.get("regime_profile", {}), ensure_ascii=False),
                        kwargs.get("novelty_score", 0),
                        json.dumps(kwargs.get("lineage", {}), ensure_ascii=False),
                        1, now, now,
                    )
                )
        return True

    def get_factor(self, factor_id: str) -> Optional[dict]:
        """获取单个因子的完整信息"""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM factors WHERE factor_id=?", (factor_id,)
            ).fetchone()
            if row:
                d = dict(row)
                for json_field in ["expression", "performance", "regime_profile", "lineage"]:
                    if isinstance(d.get(json_field), str):
                        d[json_field] = json.loads(d[json_field])
                return d
        return None

    def list_factors(self, category: str = None, status: str = "active",
                     min_icir: float = None, source: str = None) -> List[dict]:
        """按条件列出因子"""
        query = "SELECT factor_id, name_cn, category, source, status, performance, novelty_score FROM factors WHERE 1=1"
        params = []
        if category:
            query += " AND category=?"; params.append(category)
        if status:
            query += " AND status=?"; params.append(status)
        if source:
            query += " AND source=?"; params.append(source)
        query += " ORDER BY factor_id"

        with self._get_conn() as conn:
            rows = conn.execute(query, params).fetchall()
            results = []
            for r in rows:
                d = dict(r)
                perf = json.loads(d["performance"]) if isinstance(d["performance"], str) else d["performance"]
                icir = perf.get("icir_mean", 0) if isinstance(perf, dict) else 0
                if min_icir is not None and abs(icir) < min_icir:
                    continue
                d["icir_mean"] = icir
                results.append(d)
            return results

    def deprecate_factor(self, factor_id: str, superseded_by: str = None):
        """弃用一个因子"""
        with self._get_conn() as conn:
            if superseded_by:
                conn.execute(
                    "UPDATE factors SET status='deprecated', superseded_by=?, updated_at=? WHERE factor_id=?",
                    (superseded_by, datetime.now().isoformat(), factor_id)
                )
            else:
                conn.execute(
                    "UPDATE factors SET status='deprecated', updated_at=? WHERE factor_id=?",
                    (datetime.now().isoformat(), factor_id)
                )

    # ── 批量导入 ──

    def import_from_pool(self, pool_csv: str) -> int:
        """从 passed_factor_pool.csv 批量初始化 Vault"""
        df = pd.read_csv(pool_csv, encoding="utf-8-sig")
        count = 0
        for _, row in df.iterrows():
            fid = str(row.get("name", row.get("factor_id", row.get("factor", "")))).strip()
            if not fid or fid == "nan":
                continue

            status = str(row.get("status", "active")).strip()
            # 映射: reserve → experimental, pool → active
            status_map = {"reserve": "experimental", "pool": "active"}
            status = status_map.get(status, status)

            icir_val = float(row.get("icir", row.get("daily_icir", 0)) or 0)

            self.register_factor(
                factor_id=fid,
                name_cn=str(row.get("label", row.get("name_cn", ""))).strip(),
                category=str(row.get("category", "")).strip(),
                source="manual",
                status=status,
                performance={
                    "icir_mean": icir_val,
                    "icir_std": float(row.get("ic_std", 0) or 0),
                    "ic_mean": float(row.get("ic_mean", 0) or 0),
                    "positive_ic_pct": float(row.get("+ic_pct", 0) or 0),
                },
            )
            count += 1
        return count

    def import_from_chronicle(self, chronicle_db: "ExperimentChronicle",
                               experiment_id: str):
        """从 Chronicle 某次实验导入因子性能"""
        factors = chronicle_db.get_top_factors(experiment_id, min_icir=-99, limit=200)
        for f in factors:
            fid = f["factor_id"]
            existing = self.get_factor(fid)
            if existing:
                perf = json.loads(existing.get("performance", "{}")) if isinstance(existing.get("performance"), str) else existing.get("performance", {})
                perf["icir_mean"] = f.get("icir_mean", 0)
                perf["last_updated"] = experiment_id
                self.register_factor(factor_id=fid, performance=perf)
            else:
                self.register_factor(
                    factor_id=fid,
                    category=f.get("category", ""),
                    performance={"icir_mean": f.get("icir_mean", 0),
                                 "last_updated": experiment_id},
                )

    # ── 向量嵌入 & 相似度 ──

    def build_embeddings(self, factor_performance_vectors: Dict[str, np.ndarray]):
        """从因子性能向量构建嵌入。key=factor_id, value=性能向量(ICIR, TO, 稳定性等)"""
        for fid, vec in factor_performance_vectors.items():
            vec = np.asarray(vec, dtype=np.float32)
            self._embeddings[fid] = vec / (np.linalg.norm(vec) + 1e-8)
        self._save_embeddings()

        if self._use_faiss and len(self._embeddings) > 0:
            dim = next(iter(self._embeddings.values())).shape[0]
            self._faiss_index = self._faiss.IndexFlatIP(dim)
            vectors = []
            ids = []
            for i, (fid, vec) in enumerate(self._embeddings.items()):
                vectors.append(vec)
                ids.append(fid)
                self._faiss_id_map[i] = fid
            self._faiss_index.add(np.array(vectors, dtype=np.float32))

    def find_similar(self, factor_id: str, top_k: int = 5) -> List[dict]:
        """找与 factor_id 最相似的因子"""
        if factor_id not in self._embeddings:
            return []

        query_vec = self._embeddings[factor_id]

        if self._use_faiss and self._faiss_index is not None:
            D, I = self._faiss_index.search(query_vec.reshape(1, -1), top_k + 1)
            results = []
            for dist, idx in zip(D[0], I[0]):
                fid = self._faiss_id_map.get(idx, "")
                if fid and fid != factor_id:
                    results.append({"factor_id": fid, "similarity": float(dist)})
            return results[:top_k]
        else:
            similarities = {}
            for fid, vec in self._embeddings.items():
                if fid == factor_id:
                    continue
                sim = float(np.dot(query_vec, vec))
                similarities[fid] = sim
            sorted_factors = sorted(similarities.items(), key=lambda x: -x[1])[:top_k]
            return [{"factor_id": fid, "similarity": sim} for fid, sim in sorted_factors]

    def check_novelty(self, factor_id: str, threshold: float = 0.80) -> dict:
        """检测因子是否新颖。相似度 >= threshold 被认为不新颖。"""
        similar = self.find_similar(factor_id, top_k=3)
        if not similar:
            return {"novel": True, "novelty_score": 1.0, "similar_factors": []}

        max_sim = similar[0]["similarity"]
        novel = max_sim < threshold
        return {
            "novel": novel,
            "novelty_score": 1.0 - max_sim,
            "similar_factors": similar,
        }

    # ── 活跃因子集导出 ──

    def get_active_set(self, min_icir: float = 0.1,
                        exclude_status: List[str] = None) -> List[str]:
        """获取活跃因子 ID 列表 (供 Crucible 使用)"""
        if exclude_status is None:
            exclude_status = ["deprecated"]
        with self._get_conn() as conn:
            placeholders = ",".join(["?"] * len(exclude_status))
            rows = conn.execute(
                f"SELECT factor_id, performance FROM factors "
                f"WHERE status NOT IN ({placeholders})",
                exclude_status
            ).fetchall()
            active = []
            for r in rows:
                perf = json.loads(r["performance"]) if isinstance(r["performance"], str) else r["performance"]
                icir = perf.get("icir_mean", 0) if isinstance(perf, dict) else 0
                if abs(icir) >= min_icir:
                    active.append(r["factor_id"])
            return active

    # ── 统计 ──

    # ── Regime 标注 ──

    def annotate_regime(self, factor_id: str, regime: str,
                         icir: float, experiment_id: str = "manual"):
        """标注因子在特定市场regime下的表现。

        regime 取值: 'bull'(牛市), 'bear'(熊市), 'volatile'(高波动),
                     'calm'(低波动), 'all'(全样本)

        同时更新 factor_icir_history 和 factors.regime_profile。
        """
        now = datetime.now().isoformat()
        with self._get_conn() as conn:
            # 写入历史记录
            conn.execute(
                """INSERT INTO factor_icir_history (factor_id, experiment_id,
                   icir_mean, regime, recorded_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (factor_id, experiment_id, icir, regime, now)
            )
            # 更新 regime_profile
            existing = conn.execute(
                "SELECT regime_profile FROM factors WHERE factor_id=?",
                (factor_id,)
            ).fetchone()
            if existing:
                profile = json.loads(existing[0]) if existing[0] else {}
                profile[regime] = {
                    "icir_mean": icir,
                    "last_experiment": experiment_id,
                    "recorded_at": now,
                }
                conn.execute(
                    "UPDATE factors SET regime_profile=?, updated_at=? WHERE factor_id=?",
                    (json.dumps(profile, ensure_ascii=False), now, factor_id)
                )

    def batch_annotate_regime(self, factor_regime_data: List[dict]):
        """批量标注regime。

        factor_regime_data: [{"factor_id": "...", "regime": "bull",
                               "icir": 0.5, "experiment_id": "v7.9"}, ...]
        """
        for item in factor_regime_data:
            self.annotate_regime(
                item["factor_id"], item["regime"],
                item.get("icir", 0), item.get("experiment_id", "batch")
            )

    def get_best_by_regime(self, regime: str, top_k: int = 20,
                           min_icir: float = 0.1) -> List[dict]:
        """获取指定regime下ICIR最强的因子"""
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT f.factor_id, f.name_cn, f.category, f.performance,
                   h.icir_mean as regime_icir, h.experiment_id
                   FROM factors f
                   JOIN factor_icir_history h ON f.factor_id = h.factor_id
                   WHERE h.regime = ? AND f.status != 'deprecated'
                     AND abs(h.icir_mean) >= ?
                   ORDER BY abs(h.icir_mean) DESC LIMIT ?""",
                (regime, min_icir, top_k * 2)
            ).fetchall()

            results = []
            seen = set()
            for r in rows:
                if r["factor_id"] in seen:
                    continue
                seen.add(r["factor_id"])
                d = dict(r)
                perf = json.loads(d["performance"]) if isinstance(d["performance"], str) else d["performance"]
                d["icir_mean"] = perf.get("icir_mean", 0) if isinstance(perf, dict) else 0
                results.append(d)
                if len(results) >= top_k:
                    break
            return results

    def get_regime_profile(self, factor_id: str) -> dict:
        """获取因子在不同regime下的完整表现画像"""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT regime_profile FROM factors WHERE factor_id=?",
                (factor_id,)
            ).fetchone()
            if row and row[0]:
                return json.loads(row[0])

            # 回退: 从历史表聚合
            rows = conn.execute(
                "SELECT regime, AVG(icir_mean) as avg_icir, MAX(experiment_id) as latest_exp "
                "FROM factor_icir_history WHERE factor_id=? GROUP BY regime",
                (factor_id,)
            ).fetchall()
            return {r["regime"]: {"icir_mean": r["avg_icir"],
                                   "last_experiment": r["latest_exp"]}
                    for r in rows}

    # ── 血统追踪 ──

    def set_lineage(self, factor_id: str, parent_id: str,
                    derivation: str = "mutation", generation: int = None):
        """记录因子血统: 哪个因子衍生出了当前因子。

        derivation: 'mutation'(GP变异), 'crossover'(GP交叉),
                    'manual'(人工改造), 'combination'(多因子融合)
        """
        now = datetime.now().isoformat()
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT lineage FROM factors WHERE factor_id=?", (factor_id,)
            ).fetchone()
            if not row:
                return
            lineage = json.loads(row[0]) if row[0] else {}
            lineage["parent"] = parent_id
            lineage["derivation"] = derivation
            lineage["recorded_at"] = now
            if generation is not None:
                lineage["generation"] = generation

            conn.execute(
                "UPDATE factors SET lineage=?, updated_at=? WHERE factor_id=?",
                (json.dumps(lineage, ensure_ascii=False), now, factor_id)
            )

    def get_lineage_tree(self, factor_id: str, max_depth: int = 5) -> dict:
        """向上追溯因子血统树"""
        tree = {"factor_id": factor_id, "info": self.get_factor(factor_id),
                "parents": []}
        if max_depth <= 0:
            return tree

        lineage = tree["info"].get("lineage", {}) if tree["info"] else {}
        parent_id = lineage.get("parent")
        if parent_id:
            tree["parents"].append(
                self.get_lineage_tree(parent_id, max_depth - 1)
            )
        return tree

    def list_auto_generated(self) -> List[dict]:
        """列出所有自动生成的因子 (Forge产出)"""
        return self.list_factors(source="auto_gen", status=None)

    # ── FAISS 高级索引管理 ──

    def ensure_faiss_index(self, dim: int = None):
        """确保 FAISS 索引已构建。如果已有嵌入向量但索引为空则自动构建。"""
        if not self._use_faiss:
            return False
        if self._faiss_index is not None:
            return True
        if len(self._embeddings) == 0:
            return False

        if dim is None:
            dim = next(iter(self._embeddings.values())).shape[0]

        self._faiss_index = self._faiss.IndexFlatIP(dim)
        vectors = []
        for i, (fid, vec) in enumerate(self._embeddings.items()):
            vectors.append(vec.astype(np.float32))
            self._faiss_id_map[i] = fid
        if vectors:
            self._faiss_index.add(np.array(vectors, dtype=np.float32))
        return True

    def find_similar_by_vector(self, query_vec: np.ndarray,
                                top_k: int = 5) -> List[dict]:
        """根据外部向量找最相似因子 (Forge 产出的新因子用)"""
        query_vec = np.asarray(query_vec, dtype=np.float32)
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)

        if self._use_faiss and self._faiss_index is not None:
            D, I = self._faiss_index.search(query_vec.reshape(1, -1), top_k)
            results = []
            for dist, idx in zip(D[0], I[0]):
                fid = self._faiss_id_map.get(idx, "")
                if fid:
                    results.append({"factor_id": fid, "similarity": float(dist)})
            return results
        else:
            similarities = {}
            for fid, vec in self._embeddings.items():
                sim = float(np.dot(query_vec, vec))
                similarities[fid] = sim
            return [{"factor_id": fid, "similarity": sim}
                    for fid, sim in sorted(similarities.items(),
                                           key=lambda x: -x[1])[:top_k]]

    # ── 统计 ──

    def stats(self) -> dict:
        with self._get_conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM factors").fetchone()[0]
            active = conn.execute(
                "SELECT COUNT(*) FROM factors WHERE status='active'"
            ).fetchone()[0]
            deprecated = conn.execute(
                "SELECT COUNT(*) FROM factors WHERE status='deprecated'"
            ).fetchone()[0]
            auto_gen = conn.execute(
                "SELECT COUNT(*) FROM factors WHERE source='auto_gen'"
            ).fetchone()[0]
            categories = conn.execute(
                "SELECT category, COUNT(*) as cnt FROM factors GROUP BY category ORDER BY cnt DESC"
            ).fetchall()
            return {
                "total": total,
                "active": active,
                "deprecated": deprecated,
                "auto_generated": auto_gen,
                "by_category": {r["category"]: r["cnt"] for r in categories},
            }
