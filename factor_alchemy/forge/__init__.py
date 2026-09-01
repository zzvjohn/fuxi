"""
Factor Forge — 因子锻造厂
===========================
遗传编程自动生成、测试、归档因子公式。

Voyager风格技能库:
  - 成功因子自动入库 (Vault)
  - 相似度检测避免重复
  - 血统追踪 (哪个公式变异而来)
  - 复杂度惩罚 (偏好简洁公式)

用法:
  from forge import FactorForge
  forge = FactorForge(data_dict, vault)
  results = forge.evolve(n_generations=10, pop_size=50)
  forge.export_to_vault()
"""

import time
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from collections import defaultdict, Counter

from forge.primitives import (
    Primitive, PRIMITIVE_BY_NAME, INPUT_PRIMITIVES,
    get_random_terminal, get_random_primitive, WINDOW_SIZES,
)
from forge.expression import (
    ExprNode, ExpressionEvaluator,
    grow_random_tree, full_random_tree,
    parse_expression, simplify_tree,
)


class FactorForge:
    """因子自动锻造引擎"""

    def __init__(self, data: Dict[str, np.ndarray],
                 vault=None, chronicle=None,
                 max_depth: int = 5, max_complexity: float = 12.0,
                 icir_threshold: float = 0.15,
                 forward_returns: np.ndarray = None,
                 seed: int = 42,
                 fsa=None, fsa_retry: int = 3,
                 paradigm_profiles: Optional[Dict[str, Dict]] = None,
                 penalizer: Optional[Any] = None,
                 edit_memory: Optional[Any] = None,
                 dim_prune: Optional[str] = None):  # P-20260831: Alpha2 维度预剪枝
        """
        Args:
            data: 输入数据 {'open','high','low','close','volume','vwap','returns'} (T,N)
                  ⚠️ P-022: 'returns' 必须是历史收益 (close.pct_change()),
                  不能是前向收益! 前向收益仅通过 forward_returns 传入用于 ICIR。
            vault: FactorVault 实例 (自动入库)
            chronicle: Chronicle 实例 (记录方法论)
            max_depth: 表达式树最大深度
            max_complexity: 复杂度上限 (超出淘汰)
            icir_threshold: 入库最低 ICIR
            forward_returns: (T,N) 前向收益 (仅用于 ICIR 计算, 不作为 GP 终端变量)
            seed: 随机种子
            fsa: SubtreeFingerprinter 实例 (v0.7 P-025: FSA 频繁子树规避接入)
            fsa_retry: FSA 拦截后重试次数 (重试失败兜底重生成, 由适应度淘汰)
            paradigm_profiles: v0.7 P-025: {范式名: {terminal_weights, op_weights}}
                              范式定向初始化 — 种群按范式轮流分配, 交叉/变异继承范式标签
            penalizer: P-007: SubstructurePenalizer 实例 (高频子结构软拒绝, 与 FSA 并列)
            dim_prune: P-20260831: off/shadow/enforce (None=读 FUXI_DIM_PRUNE, 默认 shadow)
                       维度一致性预剪枝 (Alpha2 移植, 影子→对照→转正)
        """
        self.data = data
        self.vault = vault
        self.chronicle = chronicle
        self.max_depth = max_depth
        self.max_complexity = max_complexity
        self.icir_threshold = icir_threshold
        self.forward_returns = forward_returns
        self.rng = np.random.RandomState(seed)
        self.evaluator = ExpressionEvaluator(data)

        # v0.7 P-025: FSA 接入
        self.fsa = fsa
        self.fsa_retry = max(0, int(fsa_retry))
        self.fsa_blocks: int = 0            # FSA 拦截次数
        self._run_forbidden: Set[str] = set()  # 运行内冻结骨架 (集中度≥阈值)
        self._paradigm_tags: List[Optional[str]] = []

        # P-007: 子结构频率惩罚器
        self.penalizer = penalizer
        self.sub_blocks: int = 0            # P-007: 软拒绝拦截次数 (审计)

        # P-20260827-001: SSPM 结构化编辑记忆 ((paradigm, edit_mode) → 残差信用)
        self.edit_memory = edit_memory
        self.sspm_vetos: int = 0            # SSPM 否决拦截次数 (审计)
        self.sspm_observed: int = 0         # SSPM 残差记录次数 (审计)

        # P-20260831: Alpha2 维度一致性预剪枝
        from forge import dimension_rules as _dr
        self.dim_prune = dim_prune if dim_prune is not None else _dr.get_mode()
        self.dim_hard: int = 0              # 整树硬违规拦截/计数
        self.dim_soft: int = 0              # 整树软标记计数

        # v0.7 P-025: 范式定向
        self.paradigm_profiles = paradigm_profiles or {}

        # 并行评估 worker 数 (FORGE_WORKERS 可覆盖, 上限 pop_size)
        self.n_workers = max(1, int(os.environ.get('FORGE_WORKERS', '4')))

        # 状态
        self.generation = 0
        self.skill_library: List[dict] = []  # 成功因子积累
        self.population: List[ExprNode] = []
        self.fitness_cache: Dict[str, float] = {}
        self.last_run_stats: Dict[str, Any] = {}  # v0.7: 每轮运行统计

    # ══════════════════════════════════════════════════
    # 核心进化循环
    # ══════════════════════════════════════════════════

    # ── v0.7 P-025: FSA + 范式定向辅助 ─────────────────

    def _make_tree(self, paradigm: Optional[str], kind: str = "grow") -> ExprNode:
        """按范式 profile 生成一棵树 (kind: grow/full)"""
        tw, ow = None, None
        if paradigm and paradigm in self.paradigm_profiles:
            prof = self.paradigm_profiles.get(paradigm) or {}
            tw = prof.get("terminal_weights")
            ow = prof.get("op_weights")
        if kind == "full":
            return full_random_tree(self.rng, self.max_depth,
                                    terminal_weights=tw, op_weights=ow,
                                    dim_prune=self.dim_prune)
        return grow_random_tree(self.rng, self.max_depth,
                                terminal_weights=tw, op_weights=ow,
                                dim_prune=self.dim_prune)

    def _dim_audit(self, tree: ExprNode) -> bool:
        """P-20260831: 整树维度审计。
        返回 clean (无硬违规)。shadow: 只计数; enforce: 违规者由调用方拒绝重生成。
        """
        if self.dim_prune == "off":
            return True
        from forge import dimension_rules as dr
        res = dr.audit_forge_node(tree, record=False)
        if res is None:
            return True
        if res.n_hard:
            self.dim_hard += 1
        if res.n_soft:
            self.dim_soft += 1
        return res.clean

    def _skeleton(self, tree: ExprNode) -> Optional[str]:
        """提取树的 FSA 结构骨架 (解析失败返回 None)"""
        if self.fsa is None:
            return None
        try:
            return self.fsa.parse_expression(tree.to_string())
        except Exception:
            return None

    def _is_fsa_forbidden(self, tree: ExprNode) -> bool:
        """FSA 双重检查: 外部库冻结骨架 ∪ 运行内集中度冻结骨架"""
        if self.fsa is None:
            return False
        skel = self._skeleton(tree)
        if skel and skel in self._run_forbidden:
            return True
        try:
            return self.fsa.check_expression(tree.to_string())
        except Exception:
            return False

    def _gen_tree_fsa_guarded(self, kind: str,
                              paradigm: Optional[str]) -> Optional[ExprNode]:
        """生成一棵避开 FSA 冻结骨架与高频子结构的树; 重试耗尽返回 None"""
        for _ in range(self.fsa_retry + 1):
            t = self._make_tree(paradigm, kind)
            if self._is_fsa_forbidden(t):
                continue
            # P-007: 高频子结构软拒绝 (高频∧JQ差 → 降采样; 高频但JQ好 → 保留)
            if self.penalizer and self.penalizer.should_reject(t.to_string(), self.rng):
                self.sub_blocks += 1
                continue
            # P-20260831: 维度硬违规 (enforce 拒绝重生成; shadow 只计数)
            if not self._dim_audit(t) and self.dim_prune == "enforce":
                continue
            return t
        self.fsa_blocks += 1
        return None

    def _normalize_ts_windows(self, node: ExprNode) -> ExprNode:
        """v0.7 P-025: 修复交叉/变异把非字面量子树塞进 ts 窗口位 —
        导致 pandas 翻译出 `rolling(表达式)` 非法语法 (S5: window must be an integer)。
        窗口位非数字字面量时替换为合法窗口 (与 numpy 求值语义一致)。
        """
        if node.is_leaf:
            return node
        new_children = []
        for i, c in enumerate(node.children):
            if node.primitive.name.startswith("ts_") and i == 1:
                ok = False
                if c.is_leaf:
                    try:
                        float(c.primitive.name)
                        ok = True
                    except ValueError:
                        ok = False
                if ok:
                    new_children.append(c.clone())
                else:
                    w = self.rng.choice(WINDOW_SIZES)
                    new_children.append(
                        ExprNode(Primitive(str(w), None, 0, 0, is_input=False)))
            else:
                new_children.append(self._normalize_ts_windows(c))
        return ExprNode(node.primitive, new_children)

    def _update_run_forbidden(self, pop_size: int):
        """运行内 FSA: 统计当前种群骨架集中度, 冻结 ≥15% 种群或 ≥3 次的骨架"""
        if self.fsa is None:
            return
        counter = Counter()
        for t in self.population:
            skel = self._skeleton(t)
            if skel:
                counter[skel] += 1
        threshold = max(3, int(0.15 * pop_size))
        for skel, cnt in counter.items():
            if cnt >= threshold:
                self._run_forbidden.add(skel)

    def _tournament_select_idx(self, fitnesses: List[float], k: int) -> int:
        """锦标赛选择 — 返回种群索引 (v0.7: 需要携带范式标签)"""
        idx = self.rng.choice(len(self.population), size=k, replace=False)
        return max(idx, key=lambda i: fitnesses[i])

    def evolve(self, n_generations: int = 10, pop_size: int = 50,
               tournament_size: int = 3, elite_count: int = 3,
               mutation_rate: float = 0.3,
               crossover_rate: float = 0.5,
               workers: int = None,
               verbose: bool = True) -> List[dict]:
        """GP 进化主循环

        Returns:
            [{name, expression, icir, fitness, generation, ...}, ...]
        """
        # 初始化种群 (50% Grow + 50% Full); v0.7: 范式轮流分配 + FSA 守卫
        self.population = []
        self._paradigm_tags = []
        self.fsa_blocks = 0
        self._run_forbidden = set()
        paradigm_keys = list(self.paradigm_profiles.keys())
        for i in range(pop_size):
            paradigm = paradigm_keys[i % len(paradigm_keys)] if paradigm_keys else None
            kind = "grow" if i < pop_size // 2 else "full"
            tree = self._gen_tree_fsa_guarded(kind, paradigm)
            if tree is None:
                tree = self._make_tree(paradigm, kind)  # 兜底: 入池后由适应度淘汰
            self.population.append(tree)
            self._paradigm_tags.append(paradigm)
        if verbose and paradigm_keys:
            print(f"  [Forge] 范式定向: {len(paradigm_keys)} 个范式轮流分配 "
                  f"(每范式 {pop_size // len(paradigm_keys)} 个体)")

        best_all = []
        t0 = time.time()
        child_parent_map = {}  # P-20260827-001: SSPM 残差回填登记表

        # 解析并行 worker 数
        if workers is None:
            workers = self.n_workers
        workers = max(1, min(workers, pop_size, (os.cpu_count() or 4)))
        if verbose and workers > 1:
            print(f"  [Forge] 并行评估 workers={workers} (FORGE_WORKERS 可覆盖)")

        for gen in range(n_generations):
            self.generation = gen + 1

            # 评估适应度 — v0.7: FSA 冻结骨架直接 -999 (不计算, 省算力且强制淘汰)
            fitnesses = [None] * len(self.population)
            to_compute = []
            for i, tree in enumerate(self.population):
                if self._is_fsa_forbidden(tree):
                    fitnesses[i] = -999.0
                else:
                    to_compute.append(i)

            if workers > 1 and len(to_compute) > 1:
                from concurrent.futures import ThreadPoolExecutor, as_completed
                cache_hit = 0
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    fut_to_idx = {}
                    for i in to_compute:
                        tree = self.population[i]
                        key = tree.to_string()
                        if key in self.fitness_cache:
                            fitnesses[i] = self.fitness_cache[key]
                            cache_hit += 1
                        else:
                            fut_to_idx[ex.submit(self._compute_fitness_raw, tree)] = i
                    for fut in as_completed(fut_to_idx):
                        i = fut_to_idx[fut]
                        fit = fut.result()
                        self.fitness_cache[self.population[i].to_string()] = fit
                        fitnesses[i] = fit
                if verbose:
                    frozen = len(self.population) - len(to_compute)
                    print(f"    [并行评估] workers={workers} 命中缓存={cache_hit}/"
                          f"{len(self.population)} 计算={len(fut_to_idx)} "
                          f"FSA冻结={frozen}")
            else:
                for i in to_compute:
                    fitnesses[i] = self._fitness(self.population[i])

            # P-20260827-001: SSPM 残差回填 — 本代 fitness 已算出,
            # child_parent_map 的 slot 索引即当前种群索引 (父本 fitness 生成时已存)
            if self.edit_memory is not None and child_parent_map:
                for slot, (mode, _pi, parent_fit, child_tag) in list(child_parent_map.items()):
                    if slot < len(fitnesses) and fitnesses[slot] is not None \
                            and fitnesses[slot] > -999.0:
                        self.edit_memory.record(
                            child_tag, mode, fitnesses[slot] - parent_fit)
                        self.sspm_observed += 1
                child_parent_map = {}

            # 排序 (v0.7: 按索引, 保留范式标签)
            order = sorted(range(len(self.population)),
                           key=lambda i: -fitnesses[i])

            # 记录最佳
            best_i = order[0]
            best_tree = self.population[best_i]
            best_fit = fitnesses[best_i]
            best_name = f"gp_gen{self.generation}_{best_tree.to_string()[:60]}"
            best_all.append({
                "generation": self.generation,
                "name": best_name,
                "expression": best_tree.to_string(),
                "pandas_expression": best_tree.to_pandas_string(),  # v3.2: fri.py 可执行
                "fitness": best_fit,
                "icir": self._evaluate_icir(best_tree),
                "complexity": best_tree.complexity,
                "size": best_tree.size(),
                "depth": best_tree.depth(),
                "terminals": best_tree.terminals(),
                "paradigm": self._paradigm_tags[best_i],  # v0.7 P-025
            })

            if verbose:
                p_tag = self._paradigm_tags[best_i] or "uniform"
                print(f"  Gen {self.generation:2d}/{n_generations}: "
                      f"best_fit={best_fit:.4f} "
                      f"ICIR={best_all[-1]['icir']:+.3f} "
                      f"[{p_tag}] expr={best_tree.to_string()[:40]}...")

            # 精英保留 (v0.7: 携带范式标签)
            next_pop = [self.population[i] for i in order[:elite_count]]
            next_tags = [self._paradigm_tags[i] for i in order[:elite_count]]

            # 填充下一代
            # P-20260827-001: child_parent_map = {slot: (mode, parent_idx, parent_fit, child_tag)}
            # 父本 fitness 已知, 子代 fitness 下一轮评估后回填残差 → SSPM 编辑记忆
            while len(next_pop) < pop_size:
                mode = "crossover" if self.rng.random() < crossover_rate else "mutate"
                if mode == "crossover":
                    # 交叉 (子代继承 p1 范式)
                    p1_i = self._tournament_select_idx(fitnesses, tournament_size)
                    p2_i = self._tournament_select_idx(fitnesses, tournament_size)
                    parent_idx = p1_i
                else:
                    # 变异 (子代继承父代范式)
                    p_i = self._tournament_select_idx(fitnesses, tournament_size)
                    p1_i = p2_i = parent_idx = p_i
                parent_tag = self._paradigm_tags[parent_idx]
                parent_fit = fitnesses[parent_idx]

                # P-20260827-001: SSPM 非对称否决 (hard_veto 关闭时行为与改造前一致)
                if (self.edit_memory is not None and
                        self.edit_memory.should_veto(parent_tag, mode)):
                    alt_mode = "mutate" if mode == "crossover" else "crossover"
                    if self.edit_memory.should_veto(parent_tag, alt_mode):
                        self.sspm_vetos += 1
                        child = self.population[parent_idx].clone()  # 双否决 → 父本克隆
                        child_tag = parent_tag
                        mode = "clone"
                    else:
                        self.sspm_vetos += 1
                        mode = alt_mode
                        if mode == "crossover":
                            p2_i = self._tournament_select_idx(fitnesses, tournament_size)

                if mode == "crossover":
                    child = self._crossover(self.population[p1_i],
                                            self.population[p2_i])
                    child_tag = parent_tag
                elif mode == "mutate":
                    child = self._mutate(self.population[p1_i])
                    child_tag = parent_tag
                # mode == "clone": child 已由否决分支生成

                # 检查复杂度上限
                child = simplify_tree(child, max_size=25)

                # v0.7: ts 窗口位字面量化 (交叉/变异可能塞入非数字子树)
                child = self._normalize_ts_windows(child)

                edited = True  # P-20260827-001: 追踪子代是否仍为该编辑模式产物
                # v0.7: FSA 守卫 — 冻结骨架子代重生成
                if self._is_fsa_forbidden(child):
                    alt = self._gen_tree_fsa_guarded("grow", child_tag)
                    if alt is None:
                        alt = self._make_tree(child_tag, "grow")
                    child = alt
                    edited = False
                # P-007: 子结构频率软拒绝 — 高频∧JQ差子代重生成 (JQ表现好不受影响)
                elif (self.penalizer and
                      self.penalizer.should_reject(child.to_string(), self.rng)):
                    self.sub_blocks += 1
                    alt = self._gen_tree_fsa_guarded("grow", child_tag)
                    if alt is not None:
                        child = alt
                    edited = False

                # P-20260831: 维度硬违规 (enforce: 子代重生成; shadow: 只计数)
                if not self._dim_audit(child) and self.dim_prune == "enforce":
                    alt = self._gen_tree_fsa_guarded("grow", child_tag)
                    if alt is None:
                        alt = self._make_tree(child_tag, "grow")
                    child = alt
                    edited = False

                next_pop.append(child)
                next_tags.append(child_tag)

                # P-20260827-001: 登记待回填残差 (被 FSA/penalizer 替换的不登记)
                if (self.edit_memory is not None and edited and mode != "clone"
                        and parent_fit is not None and parent_fit > -999.0):
                    child_parent_map[len(next_pop) - 1] = (
                        mode, parent_idx, parent_fit, child_tag)

            self.population = next_pop
            self._paradigm_tags = next_tags

            # v0.7: 运行内 FSA — 更新种群骨架集中度冻结表
            self._update_run_forbidden(pop_size)

        elapsed = time.time() - t0
        if verbose:
            print(f"  Forge 完成: {n_generations}代, {elapsed:.0f}s, "
                  f"最佳 ICIR={best_all[-1]['icir']:+.3f}")
            if self.dim_prune != "off" and (self.dim_hard or self.dim_soft):
                _verb = "拦截重生成" if self.dim_prune == "enforce" else "影子计数"
                print(f"  [DimPrune:{self.dim_prune}] hard={self.dim_hard} "
                      f"soft={self.dim_soft} (维度一致性审计, {_verb})")

        # ── v0.7 P-025: 暴露最终种群供挖掘 ──
        # 旧实现只导出每代最优 (5 条), 整个种群的次优解从未被利用。
        self.final_population = []
        for i, tree in enumerate(self.population):
            fit = fitnesses[i]
            if fit is None or fit <= -999.0:
                continue
            self.final_population.append({
                "expression": tree.to_string(),
                "pandas_expression": tree.to_pandas_string(),
                "fitness": fit,
                "paradigm": self._paradigm_tags[i],
                "size": tree.size(),
                "depth": tree.depth(),
                "complexity": tree.complexity,
                "icir": None,
            })
        self.final_population.sort(key=lambda x: -x["fitness"])
        # 只为 Top-30 补算 ICIR (全种群补算太慢; 报告用)
        for entry in self.final_population[:30]:
            try:
                t = parse_expression(entry["expression"])
                entry["icir"] = self._evaluate_icir(t) if t else 0.0
            except Exception:
                entry["icir"] = 0.0

        # 过滤入库级因子
        qualified = [b for b in best_all if abs(b["icir"]) >= self.icir_threshold]
        if verbose:
            print(f"  合格因子: {len(qualified)}/{len(best_all)} "
                  f"(ICIR >= {self.icir_threshold})")

        # 检查新颖性
        novel = []
        if self.vault:
            for b in qualified:
                novelty = self.vault.check_novelty(b["name"])
                if novelty["novel"]:
                    novel.append(b)
                    self.skill_library.append(b)
            if verbose:
                print(f"  新颖因子: {len(novel)}/{len(qualified)}")

        # v0.7: 运行统计 (FSA + 范式)
        per_paradigm = {}
        paradigm_keys = list(self.paradigm_profiles.keys())
        if paradigm_keys:
            for p in paradigm_keys:
                bests = [b for b in best_all if b.get("paradigm") == p]
                if bests:
                    top = max(bests, key=lambda b: b["fitness"])
                    per_paradigm[p] = {
                        "best_fitness": top["fitness"],
                        "best_icir": top["icir"],
                        "best_generation": top["generation"],
                        "best_expression": top["expression"][:100],
                    }
        self.last_run_stats = {
            "pop_size": pop_size,
            "n_generations": n_generations,
            "max_depth": self.max_depth,
            "fsa_blocks": self.fsa_blocks,
            "n_run_frozen_skeletons": len(self._run_forbidden),
            "run_frozen_skeletons": sorted(self._run_forbidden)[:15],
            "n_qualified": len(qualified),
            "n_novel": len(novel),
            "per_paradigm": per_paradigm,
            # P-20260831: 维度剪枝审计 (shadow 计数 / enforce 拦截)
            "dim_prune_mode": self.dim_prune,
            "dim_hard": self.dim_hard,
            "dim_soft": self.dim_soft,
        }
        if verbose and self.fsa is not None:
            print(f"  [FSA] 拦截 {self.fsa_blocks} 次, "
                  f"运行内冻结骨架 {len(self._run_forbidden)} 个")

        return best_all

    # ══════════════════════════════════════════════════
    # 适应度计算
    # ══════════════════════════════════════════════════

    @staticmethod
    def _row_rank_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """逐行 (截面日) rank 相关, NaN 行均值填充 (与 _compute_icir 同风格)"""
        av = np.where(np.isfinite(a), a, np.nan)
        bv = np.where(np.isfinite(b), b, np.nan)
        am = np.nanmean(av, axis=1, keepdims=True)
        bm = np.nanmean(bv, axis=1, keepdims=True)
        av = np.where(np.isfinite(av), av, np.broadcast_to(am, av.shape))
        bv = np.where(np.isfinite(bv), bv, np.broadcast_to(bm, bv.shape))
        from scipy.stats import rankdata
        ra = rankdata(av, axis=1)
        rb = rankdata(bv, axis=1)
        ca = ra - ra.mean(axis=1, keepdims=True)
        cb = rb - rb.mean(axis=1, keepdims=True)
        num = (ca * cb).sum(axis=1)
        den = np.sqrt((ca ** 2).sum(axis=1) * (cb ** 2).sum(axis=1))
        with np.errstate(all='ignore'):
            return num / np.where(den > 0, den, np.nan)

    def _compute_fitness_raw(self, tree: ExprNode) -> float:
        """综合适应度 = |ICIR| - 复杂度惩罚 - NaN惩罚 (无缓存, 线程安全)

        由 evolve() 的并行 worker 调用 — 不触碰 self.fitness_cache,
        结果由主线程顺序写回, 避免共享 dict 竞态。

        v0.7 P-025: 新增表达式质量门 — 常数/近常数/强离散化/价格水平代理因子
        会用虚假 |ICIR| 刷爆适应度 (如 sign(volume)=1 → 常数), 直接 -999。
        """
        try:
            factor_values = self.evaluator.evaluate(tree)
        except Exception:
            return -999.0

        if factor_values is None or np.all(np.isnan(factor_values)):
            return -999.0

        nan_ratio = np.mean(np.isnan(factor_values))
        if nan_ratio > 0.5:
            return -999.0

        # ── v0.7 表达式质量门: 常数/低基数因子检测 ──
        arr = np.asarray(factor_values, dtype=float)
        finite = np.isfinite(arr)
        if finite.any():
            sample = arr[finite]
            step = max(1, len(sample) // 100_000)
            sample = sample[::step]
            n_unique = len(np.unique(np.round(sample, 12)))
            if n_unique <= 5:
                return -999.0  # 全局常数/极低基数 → 虚假 ICIR
            with np.errstate(all='ignore'):
                row_std = np.nanstd(arr, axis=1)
            if np.nansum(row_std < 1e-8) > 0.3 * arr.shape[0]:
                return -999.0  # 大部分截面日无离散 → 近常数
            # 每截面日唯一值数 (抽样 200 列): 中位数 <10 → 无横截面区分度
            # (sign(volume) 全为 1 → 中位数 1; ts_rank 窗口离散但仍有 60 档 → 存活)
            sub = arr[:, : min(arr.shape[1], 200)]
            row_unq = np.array([
                len(np.unique(np.round(sub[t][np.isfinite(sub[t])], 12)))
                for t in range(sub.shape[0])
            ])
            if len(row_unq) and np.median(row_unq) < 10:
                return -999.0

        # ── v0.7 价格/规模水平代理检测 ──
        # 与原始 close/volume 水平高度共线 (中位 |rank corr| > 0.95) 的因子
        # 是价格/规模代理而非 alpha (如 ts_sum(close,20) ≈ 价格水平, inv(volume) ≈ 流动性水平)。
        for pname in ("close", "volume"):
            px = self.data.get(pname)
            if px is None or np.asarray(px).shape != arr.shape:
                continue
            cc = self._row_rank_corr(arr, np.asarray(px, dtype=float))
            ccv = cc[np.isfinite(cc)]
            if len(ccv) and np.median(np.abs(ccv)) > 0.95:
                return -999.0  # 水平代理因子, 非 alpha

        # 计算 ICIR
        icir = self._compute_icir(factor_values)

        # 复杂度惩罚
        complexity_penalty = 0.02 * tree.complexity

        # NaN 惩罚
        nan_penalty = 2.0 * nan_ratio

        return abs(icir) - complexity_penalty - nan_penalty

    def _fitness(self, tree: ExprNode) -> float:
        """带缓存的综合适应度 (串行逻辑, 供单树/调试调用)"""
        key = tree.to_string()
        if key in self.fitness_cache:
            return self.fitness_cache[key]
        fit = self._compute_fitness_raw(tree)
        self.fitness_cache[key] = fit
        return fit

    def _evaluate_icir(self, tree: ExprNode) -> float:
        """只计算 ICIR (用于报告)"""
        try:
            factor_values = self.evaluator.evaluate(tree)
            return self._compute_icir(factor_values)
        except Exception:
            return 0.0

    def _compute_icir(self, factor_values: np.ndarray) -> float:
        """跨截面 Spearman rank IC 均值 / std — 全向量化实现

        原实现逐时间点调用 scipy.stats.spearmanr (Python 循环 + GIL 瓶颈)。
        改为: 逐行 rankdata 排名 → 批量 Pearson 相关系数, 单次 C 级调用完成。
        NaN 用行均值填充 (仅影响缺失行, 有限值行排名基本不变, 与子集法差异可忽略)。
        """
        if self.forward_returns is None:
            return 0.0

        fwd = self.forward_returns
        if factor_values.shape != fwd.shape:
            return 0.0

        T, N = factor_values.shape
        if T < 10 or N < 30:
            return 0.0

        fvals = np.asarray(factor_values, dtype=float)
        rvals = np.asarray(fwd, dtype=float)

        # 有限值掩码: factor 与 return 同时有效
        valid = np.isfinite(fvals) & np.isfinite(rvals)
        valid_count = valid.sum(axis=1)

        # 行均值填充 NaN (避免 rankdata 崩溃 + 保留有限值行排名稳定性)
        fmean = np.where(valid, fvals, np.nan)
        rmean = np.where(valid, rvals, np.nan)
        fmean = np.nanmean(fmean, axis=1, keepdims=True)
        rmean = np.nanmean(rmean, axis=1, keepdims=True)
        fvals = np.where(valid, fvals, np.broadcast_to(fmean, fvals.shape))
        rvals = np.where(valid, rvals, np.broadcast_to(rmean, rvals.shape))

        from scipy.stats import rankdata
        fr = rankdata(fvals, axis=1)   # 逐行排名
        rr = rankdata(rvals, axis=1)

        # 逐行 Pearson 相关系数 = Spearman IC
        fc = fr - fr.mean(axis=1, keepdims=True)
        rc = rr - rr.mean(axis=1, keepdims=True)
        num = (fc * rc).sum(axis=1)
        den = np.sqrt((fc ** 2).sum(axis=1) * (rc ** 2).sum(axis=1))
        ics = np.divide(num, den, out=np.full(T, np.nan), where=den > 0)

        # 仅保留有效样本充足的行 (与原逻辑一致: mask.sum() >= 30)
        ics = np.where(valid_count >= 30, ics, np.nan)
        ics_valid = ics[np.isfinite(ics)]
        if len(ics_valid) < 10:
            return 0.0

        ic_mean = np.mean(ics_valid)
        ic_std = np.std(ics_valid, ddof=1)
        return ic_mean / max(ic_std, 0.001)

    # ══════════════════════════════════════════════════
    # 遗传操作
    # ══════════════════════════════════════════════════

    def _tournament_select(self, pop: List[ExprNode],
                            fitnesses: List[float],
                            k: int) -> ExprNode:
        """锦标赛选择"""
        idx = self.rng.choice(len(pop), size=k, replace=False)
        best_idx = max(idx, key=lambda i: fitnesses[i])
        return pop[best_idx].clone()

    def _crossover(self, p1: ExprNode, p2: ExprNode) -> ExprNode:
        """子树交叉: 随机选p1的子树, 替换为p2的随机子树"""
        child = p1.clone()
        nodes = self._all_subtrees(child)
        if nodes and self.rng.random() < 0.8:
            target = nodes[self.rng.randint(0, len(nodes))]
            donor_nodes = self._all_subtrees(p2)
            if donor_nodes:
                donor = donor_nodes[self.rng.randint(
                    0, len(donor_nodes))].clone()
                # 用donor替换target (在树上查找并替换)
                child = self._replace_subtree(child, target, donor)
        return child

    def _mutate(self, tree: ExprNode) -> ExprNode:
        """子树变异: 随机替换一个子树为全新的树"""
        mutant = tree.clone()
        nodes = self._all_subtrees(mutant)
        if nodes:
            target = nodes[self.rng.randint(0, len(nodes))]
            # 生成新子树
            new_subtree = grow_random_tree(
                self.rng,
                max_depth=min(3, self.max_depth),
                dim_prune=self.dim_prune,
            )
            mutant = self._replace_subtree(mutant, target, new_subtree)
        return mutant

    def _all_subtrees(self, node: ExprNode) -> List[ExprNode]:
        """收集树的所有子树 (包括自己)"""
        result = [node]
        for c in node.children:
            result.extend(self._all_subtrees(c))
        return result

    def _replace_subtree(self, root: ExprNode, target: ExprNode,
                          new_node: ExprNode) -> ExprNode:
        """在root中查找并替换target为new_node (按表达式字符串匹配)"""
        target_str = target.to_string()
        if root.to_string() == target_str:
            return new_node.clone()
        new_children = []
        for c in root.children:
            new_children.append(
                self._replace_subtree(c, target, new_node)
            )
        return ExprNode(root.primitive, new_children)

    # ══════════════════════════════════════════════════
    # 技能库导出
    # ══════════════════════════════════════════════════

    def export_to_vault(self) -> int:
        """将 skill_library 中的因子导出到 Vault"""
        if not self.vault or not self.skill_library:
            return 0

        count = 0
        for skill in self.skill_library:
            name = f"gp_{skill['expression'][:30]}"
            factor_id = f"forge_{self.generation}_{count:03d}"

            self.vault.register_factor(
                factor_id=factor_id,
                name_cn=f"GP自动生成: {skill['expression'][:40]}",
                category="auto_gen",
                source="auto_gen",
                status="experimental",
                expression={"formula": skill["expression"],
                            "generation": skill["generation"]},
                performance={
                    "icir_mean": skill["icir"],
                    "fitness": skill["fitness"],
                    "complexity": skill["complexity"],
                },
            )
            count += 1

        if count > 0:
            print(f"  [Forge] {count} 因子已导入 Vault")

        # 记录到 Chronicle
        if self.chronicle:
            self.chronicle.log_methodology_note(
                category="factor_forge",
                note=(f"Factor Forge 第{self.generation}代完成\n"
                      f"  产出: {count} 新颖因子\n"
                      f"  ICIR阈值: {self.icir_threshold}"),
                tags=["forge", "auto_gen", f"gen_{self.generation}"],
            )

        return count

    def save_skill_library(self, path: str = None):
        """持久化技能库到 JSON"""
        if path is None:
            path = str(
                Path(__file__).parent.parent / "data" / "forge_skill_library.json"
            )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.skill_library, f, ensure_ascii=False, indent=2)
        print(f"  [Forge] 技能库已保存: {path} ({len(self.skill_library)} 项)")

    def load_skill_library(self, path: str = None):
        """加载技能库"""
        if path is None:
            path = str(
                Path(__file__).parent.parent / "data" / "forge_skill_library.json"
            )
        if Path(path).exists():
            with open(path, "r", encoding="utf-8") as f:
                self.skill_library = json.load(f)
            print(f"  [Forge] 技能库已加载: {len(self.skill_library)} 项")
