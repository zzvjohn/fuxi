"""
Chronicle SQLite 数据库引擎
============================
所有表的 CRUD 操作 + schema 自动迁移。
"""
import sqlite3
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any
from contextlib import contextmanager


DB_VERSION = 1


class ExperimentChronicle:
    """实验编年史 — 主数据库接口"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            from pathlib import Path as _P
            db_path = str(_P(__file__).parent.parent / "outputs" / "chronicle.db")
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = None
        self._init_schema()

    @contextmanager
    def _get_conn(self):
        """延迟连接，支持多线程安全（每线程独立连接）"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        """创建所有表（幂等: IF NOT EXISTS）"""
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    version TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    config_snapshot TEXT NOT NULL DEFAULT '{}',
                    status TEXT DEFAULT 'running',
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    total_duration_s REAL,
                    phase_timings TEXT DEFAULT '{}',
                    error_message TEXT
                );

                CREATE TABLE IF NOT EXISTS factor_performance (
                    experiment_id TEXT NOT NULL,
                    factor_id TEXT NOT NULL,
                    category TEXT DEFAULT '',
                    icir_mean REAL DEFAULT 0,
                    icir_std REAL DEFAULT 0,
                    ic_mean REAL DEFAULT 0,
                    rank_autocorr REAL DEFAULT 0,
                    top_decile_retention REAL DEFAULT 0,
                    cv_ic REAL DEFAULT 0,
                    ic_half_life REAL DEFAULT 0,
                    estimated_turnover REAL DEFAULT 0,
                    decile_monotonicity REAL DEFAULT 0,
                    wfa_survival_rate REAL DEFAULT 0,
                    regime_bull_icir REAL,
                    regime_bear_icir REAL,
                    regime_volatile_icir REAL,
                    regime_calm_icir REAL,
                    passed_screening INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'active',
                    PRIMARY KEY (experiment_id, factor_id),
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                );

                CREATE TABLE IF NOT EXISTS strategy_results (
                    experiment_id TEXT NOT NULL,
                    strategy_label TEXT NOT NULL,
                    factor_weights TEXT DEFAULT '{}',
                    quality_obj REAL DEFAULT 0,
                    stability_obj REAL DEFAULT 0,
                    sharpe_obj REAL DEFAULT 0,
                    icir_train REAL DEFAULT 0,
                    estimated_turnover REAL DEFAULT 0,
                    turnover_penalty REAL DEFAULT 0,
                    passed_dimensions INTEGER DEFAULT 0,
                    phase5_total_return REAL,
                    phase5_cagr REAL,
                    phase5_sharpe REAL,
                    phase5_max_dd REAL,
                    phase5_avg_turnover REAL,
                    phase55_icir REAL,
                    phase56_sharpe_2026 REAL,
                    oos_icir_mean REAL,
                    wf_anchored_score REAL,
                    wf_rolling_score REAL,
                    overfitting_risk TEXT DEFAULT 'UNKNOWN',
                    PRIMARY KEY (experiment_id, strategy_label),
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                );

                CREATE TABLE IF NOT EXISTS ga_runs (
                    experiment_id TEXT PRIMARY KEY,
                    pop_size INTEGER DEFAULT 0,
                    generations INTEGER DEFAULT 0,
                    n_factors INTEGER DEFAULT 0,
                    final_pareto_size INTEGER DEFAULT 0,
                    pareto_q_range TEXT DEFAULT '',
                    pareto_s_range TEXT DEFAULT '',
                    pareto_c_range TEXT DEFAULT '',
                    convergence_q TEXT DEFAULT '[]',
                    convergence_s TEXT DEFAULT '[]',
                    convergence_c TEXT DEFAULT '[]',
                    ga_duration_s REAL DEFAULT 0,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                );

                CREATE TABLE IF NOT EXISTS methodology_updates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT,
                    update_type TEXT NOT NULL,
                    old_value TEXT DEFAULT '',
                    new_value TEXT DEFAULT '',
                    reasoning TEXT DEFAULT '',
                    source_module TEXT DEFAULT '',
                    validated INTEGER DEFAULT 0,
                    validated_by TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                );

                CREATE INDEX IF NOT EXISTS idx_fp_icir ON factor_performance(experiment_id, icir_mean);
                CREATE INDEX IF NOT EXISTS idx_fp_factor ON factor_performance(factor_id);
                CREATE INDEX IF NOT EXISTS idx_sr_label ON strategy_results(experiment_id, strategy_label);
                CREATE INDEX IF NOT EXISTS idx_exp_status ON experiments(status);
                CREATE INDEX IF NOT EXISTS idx_exp_version ON experiments(version);
            """)

    # ── 实验生命周期 ──

    def start_experiment(self, experiment_id: str, config: dict) -> bool:
        """开始实验: 创建记录，状态 = running"""
        now = datetime.now().isoformat()
        with self._get_conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO experiments (id, version, description, config_snapshot, status, started_at)
                   VALUES (?, ?, ?, ?, 'running', ?)""",
                (
                    experiment_id,
                    config.get("version", "unknown"),
                    config.get("description", ""),
                    json.dumps(config, ensure_ascii=False),
                    now,
                )
            )
        return True

    def complete_experiment(self, experiment_id: str, success: bool = True,
                            error_msg: str = "", phase_timings: dict = None,
                            duration_s: float = 0.0):
        """完成实验: 更新状态、耗时、错误"""
        now = datetime.now().isoformat()
        status = "completed" if success else "failed"
        with self._get_conn() as conn:
            conn.execute(
                """UPDATE experiments SET status=?, completed_at=?, total_duration_s=?,
                   error_message=?, phase_timings=?
                   WHERE id=?""",
                (status, now, duration_s, error_msg,
                 json.dumps(phase_timings or {}, ensure_ascii=False),
                 experiment_id)
            )

    def update_phase_timing(self, experiment_id: str, phase: str, duration_s: float):
        """实时更新某个阶段的耗时"""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT phase_timings FROM experiments WHERE id=?", (experiment_id,)
            ).fetchone()
            if row:
                timings = json.loads(row[0]) if row[0] else {}
                timings[phase] = duration_s
                conn.execute(
                    "UPDATE experiments SET phase_timings=? WHERE id=?",
                    (json.dumps(timings, ensure_ascii=False), experiment_id)
                )

    # ── 因子性能写入 ──

    def log_factor_performances(self, experiment_id: str,
                                 records: List[dict]) -> int:
        """批量写入因子性能。每条 record = dict，key 对应 factor_performance 列。"""
        count = 0
        with self._get_conn() as conn:
            for r in records:
                conn.execute(
                    """INSERT OR REPLACE INTO factor_performance
                       (experiment_id, factor_id, category, icir_mean, icir_std, ic_mean,
                        rank_autocorr, top_decile_retention, cv_ic, ic_half_life,
                        estimated_turnover, decile_monotonicity, wfa_survival_rate,
                        regime_bull_icir, regime_bear_icir, regime_volatile_icir, regime_calm_icir,
                        passed_screening, status)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        experiment_id, r.get("factor_id", ""),
                        r.get("category", ""), r.get("icir_mean", 0), r.get("icir_std", 0),
                        r.get("ic_mean", 0), r.get("rank_autocorr", 0),
                        r.get("top_decile_retention", 0), r.get("cv_ic", 0),
                        r.get("ic_half_life", 0), r.get("estimated_turnover", 0),
                        r.get("decile_monotonicity", 0), r.get("wfa_survival_rate", 0),
                        r.get("regime_bull_icir"), r.get("regime_bear_icir"),
                        r.get("regime_volatile_icir"), r.get("regime_calm_icir"),
                        1 if r.get("passed_screening") else 0,
                        r.get("status", "active"),
                    )
                )
                count += 1
        return count

    # ── 策略结果写入 ──

    def log_strategy_result(self, experiment_id: str, strategy_label: str,
                            result: dict) -> bool:
        """写入单个策略结果"""
        with self._get_conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO strategy_results
                   (experiment_id, strategy_label, factor_weights,
                    quality_obj, stability_obj, sharpe_obj, icir_train,
                    estimated_turnover, turnover_penalty, passed_dimensions,
                    phase5_total_return, phase5_cagr, phase5_sharpe,
                    phase5_max_dd, phase5_avg_turnover,
                    phase55_icir, phase56_sharpe_2026,
                    oos_icir_mean, wf_anchored_score, wf_rolling_score, overfitting_risk)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    experiment_id, strategy_label,
                    json.dumps(result.get("factor_weights", {}), ensure_ascii=False),
                    result.get("quality_obj", 0), result.get("stability_obj", 0),
                    result.get("sharpe_obj", 0), result.get("icir_train", 0),
                    result.get("estimated_turnover", 0), result.get("turnover_penalty", 0),
                    result.get("passed_dimensions", 0),
                    result.get("phase5_total_return"),
                    result.get("phase5_cagr"),
                    result.get("phase5_sharpe"),
                    result.get("phase5_max_dd"),
                    result.get("phase5_avg_turnover"),
                    result.get("phase55_icir"),
                    result.get("phase56_sharpe_2026"),
                    result.get("oos_icir_mean"),
                    result.get("wf_anchored_score"),
                    result.get("wf_rolling_score"),
                    result.get("overfitting_risk", "UNKNOWN"),
                )
            )
        return True

    # ── GA 运行日志 ──

    def log_ga_run(self, experiment_id: str, ga_log: dict) -> bool:
        """写入 NSGA-II 运行摘要"""
        with self._get_conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO ga_runs
                   (experiment_id, pop_size, generations, n_factors,
                    final_pareto_size, pareto_q_range, pareto_s_range, pareto_c_range,
                    convergence_q, convergence_s, convergence_c, ga_duration_s)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    experiment_id,
                    ga_log.get("pop_size", 0),
                    ga_log.get("generations", 0),
                    ga_log.get("n_factors", 0),
                    ga_log.get("final_pareto_size", 0),
                    ga_log.get("pareto_q_range", ""),
                    ga_log.get("pareto_s_range", ""),
                    ga_log.get("pareto_c_range", ""),
                    json.dumps(ga_log.get("convergence_q", [])),
                    json.dumps(ga_log.get("convergence_s", [])),
                    json.dumps(ga_log.get("convergence_c", [])),
                    ga_log.get("ga_duration_s", 0),
                )
            )
        return True

    # ── 方法论更新 ──

    def log_methodology_update(self, update: dict) -> int:
        """记录方法论更新"""
        with self._get_conn() as conn:
            cur = conn.execute(
                """INSERT INTO methodology_updates
                   (experiment_id, update_type, old_value, new_value, reasoning,
                    source_module, validated, validated_by, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    update.get("experiment_id", ""),
                    update.get("update_type", ""),
                    update.get("old_value", ""),
                    update.get("new_value", ""),
                    update.get("reasoning", ""),
                    update.get("source_module", ""),
                    1 if update.get("validated") else 0,
                    update.get("validated_by", ""),
                    update.get("created_at", datetime.now().isoformat()),
                )
            )
            return cur.lastrowid

    # ── 查询 ──

    def get_recent_experiments(self, limit: int = 10) -> List[dict]:
        """获取最近 N 次实验摘要"""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT id, version, status, started_at, total_duration_s, error_message "
                "FROM experiments ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_top_factors(self, experiment_id: str = None, min_icir: float = 0.3,
                         limit: int = 20) -> List[dict]:
        """获取高 ICIR 因子"""
        with self._get_conn() as conn:
            if experiment_id:
                rows = conn.execute(
                    "SELECT factor_id, category, icir_mean, estimated_turnover, wfa_survival_rate "
                    "FROM factor_performance WHERE experiment_id=? AND abs(icir_mean)>=? "
                    "ORDER BY abs(icir_mean) DESC LIMIT ?",
                    (experiment_id, min_icir, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT factor_id, category, avg(abs(icir_mean)) as avg_icir "
                    "FROM factor_performance GROUP BY factor_id "
                    "HAVING avg(abs(icir_mean))>=? "
                    "ORDER BY avg_icir DESC LIMIT ?",
                    (min_icir, limit)
                ).fetchall()
            return [dict(r) for r in rows]

    def get_best_strategies(self, min_sharpe: float = 0.5,
                             limit: int = 10) -> List[dict]:
        """获取历史上表现最好的策略"""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT experiment_id, strategy_label, factor_weights, "
                "phase5_sharpe, phase5_cagr, phase5_max_dd, oos_icir_mean, overfitting_risk "
                "FROM strategy_results "
                "WHERE phase5_sharpe IS NOT NULL AND phase5_sharpe >= ? "
                "ORDER BY phase5_sharpe DESC LIMIT ?",
                (min_sharpe, limit)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_factor_trend(self, factor_id: str) -> List[dict]:
        """获取某因子的 ICIR 历史趋势"""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT e.id, e.version, e.started_at, fp.icir_mean, fp.estimated_turnover "
                "FROM factor_performance fp "
                "JOIN experiments e ON fp.experiment_id=e.id "
                "WHERE fp.factor_id=? AND e.status='completed' "
                "ORDER BY e.started_at ASC",
                (factor_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def experiment_count(self) -> int:
        with self._get_conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]

    def factor_count(self) -> int:
        with self._get_conn() as conn:
            return conn.execute(
                "SELECT COUNT(DISTINCT factor_id) FROM factor_performance"
            ).fetchone()[0]

    # ── 实验摘要快照 (给 Agent 用) ──

    def get_experiment_summary(self, experiment_id: str) -> dict:
        """单次实验的完整摘要，包含所有表的数据"""
        with self._get_conn() as conn:
            exp = conn.execute("SELECT * FROM experiments WHERE id=?", (experiment_id,)).fetchone()
            if not exp:
                return {}
            exp_d = dict(exp)

            factors = conn.execute(
                "SELECT factor_id, icir_mean, estimated_turnover, passed_screening "
                "FROM factor_performance WHERE experiment_id=? "
                "ORDER BY abs(icir_mean) DESC LIMIT 30",
                (experiment_id,)
            ).fetchall()

            strategies = conn.execute(
                "SELECT * FROM strategy_results WHERE experiment_id=?", (experiment_id,)
            ).fetchall()

            ga = conn.execute(
                "SELECT * FROM ga_runs WHERE experiment_id=?", (experiment_id,)
            ).fetchone()

            return {
                "experiment": exp_d,
                "top_factors": [dict(f) for f in factors],
                "strategies": [dict(s) for s in strategies],
                "ga": dict(ga) if ga else {},
            }
