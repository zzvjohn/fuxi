# -*- coding: utf-8 -*-
"""
S1-b: 因子-模型联合优化 — 模型维度"实体化"（免回测代理评估）
=========================================================================
背景（2026-09-12 取证结论）
--------------------------
MAB 臂空间**已经含模型维度**（41 臂中 38 个形如 `范式__模型`），但
`.model_type` **在全代码库零消费者**：
  - `ralph_loop._mab_select_direction` 只读 `pulls/expected_reward/paradigm/status`
  - `cooptim_results.json` **不存在**（`record_result()` 从未被调用）
  - `ModelConfig` 只被 `print(len(cooptim.models))` 计数
⇒ 同范式的 3 个模型臂在数学上**完全等价**，等于把 1 个方向拆成 3 个相同臂。

本模块的最小可行方案（无前视、免回测）
--------------------------------------
对**同一批已算出 IC 的候选因子**，用 3 个廉价模型各生成一个组合信号，以
"**组合信号 vs fwd20 的逐日 Rank IC**"作为模型臂的**即时奖励**：

  1. `equal_weight_rank` — 因子截面 rank-pct 等权平均（王者当前方式的近似）
  2. `ridge_linear`      — 逐日截面 Ridge（walk-forward 重拟合）
  3. `lightgbm_rank`     — 逐日截面 LightGBM 回归（同上）

🔴 前视防护（硬约束）
--------------------
所有监督模型的训练集**只允许使用 `s <= t - horizon` 的样本**（这些日期的
fwd20 在 t 日已实现），预测 t 日信号后与 t 日的 fwd20 求 IC。禁止用同日
标签拟合同日信号。

🔴 量纲纪律（S2-c iii）
---------------------
`proxy_reward = clip(ic_mean / PROXY_REWARD_REF_IC, -1, 1)`，
基准 `PROXY_REWARD_REF_IC = 0.05`（IC 0.05 → reward 1.0）。
该量纲与 JQ 奖励 `clip((jq_return-182.57)/100*5, -1, 1)` **不同源**，
因此 MAB 侧以 `reward_source` 区分、**禁止混算**（见 `MABScheduler.update_reward`）。

开关
----
`FUXI_MODEL_PROXY_EVAL = off | shadow(默认) | enforce`
  - off     : 完全不运行（零成本）
  - shadow  : 运行评估、写 `cooptim_results.json` 与 `ModelConfig.performance`，
              **不改**任何 MAB 奖励（零裁决影响）
  - enforce : 额外把 proxy reward 写进对应 `范式__模型` 臂的 `expected_reward`

创建: 2026-09-12 (S1-b 落地)
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── 常量 ────────────────────────────────────────────────────
PROXY_HORIZON = 20              # fwd 收益期 (交易日)
try:
    # 量纲基准唯一权威定义在 factor_model_cooptim (避免两处漂移)
    from factor_model_cooptim import PROXY_REWARD_REF_IC
except Exception:               # 独立运行时兜底
    PROXY_REWARD_REF_IC = 0.05  # IC 0.05 → reward 1.0

MODEL_IDS = ("equal_weight_rank", "ridge_linear", "lightgbm_rank")
MAX_FACTORS_DEFAULT = 12
MIN_STOCKS_PER_DAY = 30
# S1-b (2026-09-12): 代理奖励的最小有效日数。低于此值 IC 无统计意义
# (冒烟实测: 未做缺失填补时监督模型只剩 n_days=2, IC=0.277 属纯噪声)
PROXY_MIN_DAYS = 30


def get_mode() -> str:
    """读 FUXI_MODEL_PROXY_EVAL (off|shadow|enforce, 默认 shadow)。"""
    v = os.environ.get("FUXI_MODEL_PROXY_EVAL", "shadow").strip().lower()
    return v if v in ("off", "shadow", "enforce") else "shadow"


def proxy_reward(ic_mean: float) -> float:
    """把组合信号 IC 映射到 MAB 可用的 [-1, 1] 奖励 (量纲显式声明)。"""
    if ic_mean is None or not np.isfinite(ic_mean):
        return 0.0
    return float(np.clip(ic_mean / PROXY_REWARD_REF_IC, -1.0, 1.0))


class ModelProxyEvaluator:
    """S1-b 免回测模型代理评估器。

    用法:
        ev = ModelProxyEvaluator()
        res = ev.evaluate(candidates)   # candidates: [{formula, factor_name, paradigm}]
        # res["models"][model_id] = {ic, icir, n_days, reward, degraded}
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        horizon: int = PROXY_HORIZON,
        max_factors: int = MAX_FACTORS_DEFAULT,
        min_stocks_per_day: int = MIN_STOCKS_PER_DAY,
        refit_step: int = 10,
        train_min_days: int = 40,
        max_train_rows: int = 120_000,
    ):
        self.data_dir = Path(data_dir) if data_dir else (
            Path(__file__).resolve().parent.parent.parent / "data")
        self.horizon = int(horizon)
        self.max_factors = int(max_factors)
        self.min_stocks_per_day = int(min_stocks_per_day)
        self.refit_step = max(1, int(refit_step))
        self.train_min_days = int(train_min_days)
        self.max_train_rows = int(max_train_rows)
        self._ic_comp = None

    # ── 对外主入口 ──────────────────────────────────────────

    def evaluate(self, candidates: List[Dict]) -> Dict:
        """评估一批候选因子在 3 个模型下的组合表现。

        Returns:
            {
              "ok": bool, "reason": str, "horizon": int, "n_factors": int,
              "n_days": int, "models": {model_id: {...}}, "factor_names": [...],
              "timestamp": str
            }
        """
        out = {
            "ok": False, "reason": "", "horizon": self.horizon,
            "n_factors": 0, "n_days": 0, "models": {},
            "factor_names": [], "timestamp": datetime.now().isoformat(),
        }
        formulae = []
        for c in (candidates or []):
            if isinstance(c, str):
                f = c.strip()
                nm = ""
            else:
                f = str(c.get("formula", c.get("expression", "")) or "").strip()
                nm = str(c.get("factor_name", c.get("name", "")) or "")
            if f and f not in [x[0] for x in formulae]:
                formulae.append((f, nm))
            if len(formulae) >= self.max_factors:
                break

        if len(formulae) < 2:
            out["reason"] = f"候选不足 (需≥2, 实得 {len(formulae)}) — 组合信号无意义"
            return out

        try:
            rank_panels, fwd_wide, names = self._build_panels(formulae)
        except Exception as e:
            out["reason"] = f"面板构建失败: {e}"
            return out

        if not rank_panels or fwd_wide is None:
            out["reason"] = "面板为空 (公式全部求值失败或数据不足)"
            return out

        out["n_factors"] = len(rank_panels)
        out["factor_names"] = names

        # ── 1. 等权 rank 平均 (纯横截面, 无拟合 ⇒ 无前视) ──
        try:
            eq = self._equal_weight_signal(rank_panels, fwd_wide)
            out["models"]["equal_weight_rank"] = self._score(eq, fwd_wide)
        except Exception as e:
            out["models"]["equal_weight_rank"] = {"error": str(e)}

        # ── 2/3. 监督模型 (walk-forward, 严防前视) ──
        for mid in ("ridge_linear", "lightgbm_rank"):
            try:
                sig, degraded, note = self._walk_forward_signal(rank_panels, fwd_wide, mid)
                sc = self._score(sig, fwd_wide)
                sc["degraded"] = degraded
                if note:
                    sc["note"] = note
                out["models"][mid] = sc
            except Exception as e:
                out["models"][mid] = {"error": str(e)}

        # ── S1-b (2026-09-12): 有效日数闸门 ──
        # n_days < PROXY_MIN_DAYS 的 IC 属噪声, 强制 reward=0 并标 degraded,
        # 否则会把 "2 天样本算出的 IC=0.28" 当成模型臂的强奖励写进 MAB。
        for mid, sc in out["models"].items():
            if not isinstance(sc, dict) or "n_days" not in sc:
                continue
            if sc["n_days"] < PROXY_MIN_DAYS:
                sc["degraded"] = True
                sc["note"] = (f"n_days={sc['n_days']} < {PROXY_MIN_DAYS} "
                              f"⇒ IC 无统计意义, reward 归零")
                sc["raw_reward"] = sc.get("reward", 0.0)
                sc["reward"] = 0.0

        out["n_days"] = max(
            [m.get("n_days", 0) for m in out["models"].values()
             if isinstance(m, dict)] or [0])
        out["ok"] = any(m.get("n_days", 0) > 0 for m in out["models"].values()
                        if isinstance(m, dict))
        if not out["ok"]:
            out["reason"] = "三个模型均无有效评估日"
        return out

    # ── 面板构建 ────────────────────────────────────────────

    def _get_ic_comp(self):
        if self._ic_comp is None:
            from factor_ic_computer import FactorICComputer
            self._ic_comp = FactorICComputer(forward_period=self.horizon)
        return self._ic_comp

    def _build_panels(self, formulae: List[Tuple[str, str]]):
        """因子公式 → (list[rank宽表], fwd宽表, 因子名列表)。"""
        ic = self._get_ic_comp()
        # 触发一次加载 (全市场 csv, 分钟级)
        if not getattr(ic, "_loaded", False):
            ic._load_data()

        px = ic._price_df[["ts_code", "trade_date", "close"]].drop_duplicates(
            subset=["ts_code", "trade_date"])
        close_wide = px.pivot(index="trade_date", columns="ts_code",
                              values="close").sort_index()
        # 🔴 前向收益: 沿日期轴 shift(-h) (逐股等价于 groupby().shift(-h))
        fwd_wide = close_wide.shift(-self.horizon) / close_wide - 1.0

        rank_panels, names = [], []
        for f, nm in formulae:
            try:
                df = ic._eval_formula(f, nm or "model_proxy")
            except Exception:
                df = None
            if df is None or len(df) == 0:
                continue
            try:
                w = df.pivot_table(index="trade_date", columns="ts_code",
                                   values="factor_value", aggfunc="last")
            except Exception:
                continue
            w = w.reindex(index=fwd_wide.index, columns=fwd_wide.columns)
            # 🔴 预处理强制 axis=1 (截面), 裸 .rank(pct=True) 默认 axis=0 = 前视
            r = w.rank(axis=1, pct=True)
            if r.notna().sum().sum() < 100:
                continue
            # S1-b (2026-09-12): 缺失以截面中位数 0.5 填补 (rank-pct ∈ [0,1] 的中性值)。
            # 不做填补时监督模型用的是 "all(axis=1) isfinite" 行掩码 ⇒ 多因子交集使
            # 每日有效股票数塌缩到 <30, 只剩 n_days=2 的噪声 IC。
            r = r.fillna(0.5)
            rank_panels.append(r)
            names.append(nm or f[:40])

        return rank_panels, fwd_wide, names

    # ── 三个模型 ────────────────────────────────────────────

    @staticmethod
    def _equal_weight_signal(rank_panels, fwd_wide):
        """等权 rank 平均 (横截面, 无拟合 ⇒ 天然无前视)。"""
        stack = np.stack([p.values.astype("float32") for p in rank_panels], axis=0)
        with np.errstate(invalid="ignore"):
            sig = np.nanmean(stack, axis=0)
        import pandas as pd
        return pd.DataFrame(sig, index=fwd_wide.index, columns=fwd_wide.columns)

    def _walk_forward_signal(self, rank_panels, fwd_wide, model_id: str):
        """监督模型组合信号 — 严格 walk-forward。

        训练样本: 日期 s <= t - horizon (其 fwd20 在 t 日已实现)
        预测目标: t 日截面
        重拟合: 每 refit_step 天一次 (成本控制)
        """
        import pandas as pd

        dates = fwd_wide.index
        n_dates = len(dates)

        # 逐日截面: X = 因子 rank 堆叠 (n_stock, k), y = fwd 的截面 rank-pct
        X_by_date, y_by_date = [], []
        for i in range(n_dates):
            x = np.stack([p.iloc[i].values for p in rank_panels], axis=1).astype("float32")
            y = fwd_wide.iloc[i].values
            # 目标用截面 rank-pct (抗离群, 与 IC 口径一致)
            yv = pd.Series(y).rank(pct=True).values.astype("float32")
            X_by_date.append(x)
            y_by_date.append(yv)

        degraded, note = False, ""
        model = None
        need_fit = False
        start = self.train_min_days + self.horizon
        if n_dates <= start + 5:
            degraded, note = True, f"有效日不足 (n_dates={n_dates} <= {start + 5})"
            return pd.DataFrame(np.nan, index=dates, columns=fwd_wide.columns), degraded, note

        def _make_model():
            if model_id == "ridge_linear":
                from sklearn.linear_model import Ridge
                return Ridge(alpha=1.0)
            # lightgbm_rank
            from lightgbm import LGBMRegressor
            return LGBMRegressor(
                n_estimators=100, num_leaves=15, learning_rate=0.05,
                min_child_samples=200, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.9, verbose=-1, n_jobs=2, random_state=42,
            )

        model = _make_model()
        out = np.full((n_dates, fwd_wide.shape[1]), np.nan, dtype="float32")

        for i in range(start, n_dates):
            if need_fit or (i - start) % self.refit_step == 0:
                # 训练集: 日期 <= i - horizon
                last = i - self.horizon
                Xs, ys = [], []
                for s in range(0, last + 1):
                    x, y = X_by_date[s], y_by_date[s]
                    if x is None or len(x) == 0:
                        continue
                    ok = np.isfinite(x).all(axis=1) & np.isfinite(y)
                    if ok.sum() < self.min_stocks_per_day:
                        continue
                    Xs.append(x[ok])
                    ys.append(y[ok])
                if not Xs:
                    continue
                X = np.concatenate(Xs, axis=0)
                y = np.concatenate(ys, axis=0)
                if len(X) > self.max_train_rows:
                    idx = np.random.RandomState(42).choice(
                        len(X), self.max_train_rows, replace=False)
                    X, y = X[idx], y[idx]
                if model_id == "lightgbm_rank":
                    try:
                        model = _make_model()
                    except Exception as e:
                        degraded, note = True, f"lightgbm 不可用 → 降级 Ridge: {e}"
                        from sklearn.linear_model import Ridge
                        model = Ridge(alpha=1.0)
                try:
                    model.fit(X, y)
                    need_fit = False
                except Exception as e:
                    need_fit = True
                    note = f"fit 失败: {e}"
                    continue
            xi = X_by_date[i]
            if xi is None or len(xi) == 0:
                continue
            ok = np.isfinite(xi).all(axis=1)
            if ok.sum() < self.min_stocks_per_day:
                continue
            try:
                pred = model.predict(xi[ok])
            except Exception:
                continue
            col = np.full(xi.shape[0], np.nan, dtype="float32")
            col[ok] = pred
            out[i] = col

        return pd.DataFrame(out, index=dates, columns=fwd_wide.columns), degraded, note

    # ── 打分 ────────────────────────────────────────────────

    def _score(self, signal, fwd_wide) -> Dict:
        """组合信号 vs fwd 的逐日截面 Rank IC → mean/std/ICIR + proxy reward。"""
        ics = []
        for i in range(len(fwd_wide.index)):
            s = signal.iloc[i].values
            f = fwd_wide.iloc[i].values
            ok = np.isfinite(s) & np.isfinite(f)
            if ok.sum() < self.min_stocks_per_day:
                continue
            import pandas as pd
            ic = pd.Series(s[ok]).rank().corr(pd.Series(f[ok]).rank())
            if ic is not None and np.isfinite(ic):
                ics.append(float(ic))
        if not ics:
            return {"ic": 0.0, "icir": 0.0, "n_days": 0, "reward": 0.0}
        arr = np.array(ics, dtype="float64")
        mean_ic = float(np.nanmean(arr))
        std_ic = float(np.nanstd(arr))
        icir = abs(mean_ic / std_ic) if std_ic > 1e-12 else 0.0
        return {
            "ic": round(mean_ic, 6),
            "icir": round(icir, 4),
            "n_days": len(ics),
            "ic_std": round(std_ic, 6),
            "reward": round(proxy_reward(mean_ic), 4),
        }


# ═══════════════════════════════════════════════════════════════
# 写回 cooptim / MAB
# ═══════════════════════════════════════════════════════════════

def apply_to_cooptim(cooptim, ev: Dict, candidates: List[Dict],
                     enforce: Optional[bool] = None) -> Dict:
    """把代理评估结果写回 FactorModelCoOptimizer。

    - `ModelConfig.performance` 填充 (3 个模型)
    - `CoOptimResult` 落盘 → `data/cooptim_results.json`
    - enforce=True 时把 proxy reward 写入 `范式__模型` 臂的 `expected_reward`
      (source="model_proxy", 与 JQ 量纲隔离)
    """
    from factor_model_cooptim import CoOptimResult

    if enforce is None:
        enforce = (get_mode() == "enforce")

    models = ev.get("models", {}) or {}
    # 1) ModelConfig.performance
    for mid, sc in models.items():
        m = cooptim.models.get(mid)
        if m is None:
            continue
        m.n_factors_used = int(ev.get("n_factors", 0) or 0)
        m.performance = dict(sc)
        m.performance["horizon"] = ev.get("horizon")
        m.performance["ts"] = ev.get("timestamp")

    # 2) MAB 奖励 (仅 enforce)
    n_arms_updated = 0
    paradigms = []
    for c in (candidates or []):
        if isinstance(c, dict):
            p = c.get("paradigm") or c.get("_mab_paradigm") or ""
            if p and p not in paradigms:
                paradigms.append(p)

    if enforce and paradigms:
        for mid, sc in models.items():
            if not isinstance(sc, dict) or "reward" not in sc:
                continue
            rew = float(sc["reward"])
            for para in paradigms:
                did = f"{para}__{mid}"
                d = cooptim.mab.directions.get(did)
                if d is None:
                    continue
                # S2-c iii: 走量纲守卫 (换源会重置而非平均), 禁止与 JQ 奖励混算
                cooptim.mab.ensure_reward_source(d, "model_proxy")
                d.expected_reward = rew
                d.n_rewards = max(1, int(getattr(d, "n_rewards", 0) or 0))
                n_arms_updated += 1
        cooptim.mab.save_state()

    # 3) CoOptimResult + results 落盘
    try:
        res = CoOptimResult(
            round_id=f"cooptim_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            factor_changes=[str(ev.get("factor_names", [])[:8])],
            model_changes={
                mid: {
                    "ic": (sc or {}).get("ic"),
                    "icir": (sc or {}).get("icir"),
                    "n_days": (sc or {}).get("n_days"),
                    "reward": (sc or {}).get("reward"),
                    "degraded": (sc or {}).get("degraded", False),
                }
                for mid, sc in models.items()
            },
            jq_return=None, jq_sharpe=None, vs_baseline=0.0,
            insights=[
                f"模型代理评估 (horizon={ev.get('horizon')}, "
                f"n_factors={ev.get('n_factors')}, n_days={ev.get('n_days')})",
                f"enforce={enforce}; 更新臂数={n_arms_updated}; paradigms={paradigms[:4]}",
            ],
        )
        cooptim.record_result(res)
    except Exception as e:
        return {"ok": False, "reason": f"record_result 失败: {e}",
                "n_arms_updated": n_arms_updated}

    return {
        "ok": bool(ev.get("ok")),
        "enforce": enforce,
        "n_arms_updated": n_arms_updated,
        "n_factors": ev.get("n_factors"),
        "n_days": ev.get("n_days"),
        "paradigms": paradigms,
        "results_path": str(cooptim._results_path),
    }


def run_and_apply(candidates: List[Dict], cooptim=None, mode: Optional[str] = None) -> Dict:
    """便捷入口: 评估 + 写回。mode=None → 读 env。off 直接跳过。"""
    m = (mode or get_mode()).lower()
    if m == "off":
        return {"ok": True, "skipped": True, "mode": "off"}
    ev = ModelProxyEvaluator().evaluate(candidates)
    if cooptim is None:
        from factor_model_cooptim import create_default_cooptimizer
        cooptim = create_default_cooptimizer()
    rep = apply_to_cooptim(cooptim, ev, candidates, enforce=(m == "enforce"))
    rep["mode"] = m
    rep["eval"] = {k: v for k, v in ev.items() if k != "factor_names"}
    return rep
