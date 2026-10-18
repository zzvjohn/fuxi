"""Isolated quantum-inspired search challenger for FactorForge.

This is a classical probability-vector search, not quantum hardware.  It
decodes sampled bits into the *same* Forge ExprNode vocabulary and delegates
all factor scoring to the supplied FactorForge instance.  It neither changes
Forge defaults nor grants S5/JQ eligibility.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from forge.expression import ExprNode
from forge.primitives import (
    NON_INPUT_PRIMITIVES, Primitive, WINDOW_SIZES, available_input_primitives,
)


_BITS_PER_SLOT = 13  # independent stop bit, 8 choice bits, 4 window bits
_CONSTANTS = (-1.0, -0.5, 0.0, 0.5, 1.0, 2.0)


class QuantumForgeChallenger:
    """QEA-style independent qubits with separate islands to retain niches."""

    def __init__(self, forge: Any, *, seed: int, unique_budget: int,
                 population_size: int = 40, islands: int = 4,
                 rotation: float = 0.04, max_draw_multiplier: int = 30,
                 target_paradigm: str | None = None):
        if unique_budget <= 0 or population_size <= 0 or islands <= 0 or max_draw_multiplier <= 0:
            raise ValueError("budget, population_size, islands and draw multiplier must be positive")
        if not 0 <= rotation < math.pi / 8:
            raise ValueError("rotation must be in [0, pi/8)")
        self.forge = forge
        self.rng = np.random.RandomState(seed)
        self.seed = int(seed)
        self.unique_budget = int(unique_budget)
        self.population_size = int(population_size)
        self.islands = int(islands)
        self.rotation = float(rotation)
        self.max_draws = int(unique_budget * max_draw_multiplier)
        self.max_depth = int(forge.max_depth)
        if self.max_depth < 1 or self.max_depth > 8:
            raise ValueError("Forge depth outside challenger decoder range")
        self.slot_count = 2 ** (self.max_depth + 1) - 1
        self.bit_count = self.slot_count * _BITS_PER_SLOT
        self.angles = np.full((islands, self.bit_count), math.pi / 4)
        self.ops = list(NON_INPUT_PRIMITIVES)
        profiles = getattr(forge, "paradigm_profiles", {}) or {}
        if target_paradigm and target_paradigm not in profiles:
            raise ValueError("quantum challenger paradigm has no Forge profile")
        profile = profiles.get(target_paradigm, {})
        self.target_paradigm = target_paradigm
        self.op_weights = profile.get("op_weights", {})
        self.terminal_weights = profile.get("terminal_weights", {})
        inputs = [p for p in available_input_primitives() if p.name in forge.data]
        if not inputs:
            raise ValueError("no verified Forge input terminals are present")
        # Six repeats per input give constants approximately Forge's 15% rate.
        self.terminals = inputs * 6 + [
            Primitive(str(value), None, 0, 0, is_input=False)
            for value in _CONSTANTS
        ]

    @staticmethod
    def _number(bits: np.ndarray) -> int:
        return int(np.sum(bits.astype(np.int64) << np.arange(len(bits))))

    @staticmethod
    def _weighted_choice(options: list[Primitive], weights: dict, code: int) -> Primitive:
        if not 0 <= code < 256:
            raise ValueError("choice code must fit the independent 8-bit field")
        values = np.array([max(0.0, float(weights.get(p.name, 1.0)))
                           for p in options], dtype=float)
        if not np.isfinite(values).all() or values.sum() <= 0:
            raise ValueError("invalid Forge paradigm weights")
        cut = (code + 0.5) / 256 * values.sum()
        return options[min(int(np.searchsorted(np.cumsum(values), cut)), len(options) - 1)]

    def _decode(self, tape: np.ndarray) -> ExprNode:
        def node(slot: int, depth: int) -> ExprNode:
            start = slot * _BITS_PER_SLOT
            stop = bool(tape[start])
            choice = self._number(tape[start + 1:start + 9])
            if depth >= self.max_depth or (depth > 0 and stop):
                return ExprNode(self._weighted_choice(
                    self.terminals, self.terminal_weights, choice))
            op = self._weighted_choice(self.ops, self.op_weights, choice)
            left = node(2 * slot + 1, depth + 1)
            if op.name.startswith("ts_"):
                window_index = self._number(tape[start + 9:start + 13])
                window = WINDOW_SIZES[window_index % len(WINDOW_SIZES)]
                right = ExprNode(Primitive(str(window), None, 0, 0, is_input=False))
                return ExprNode(op, [left, right])
            if op.arity == 1:
                return ExprNode(op, [left])
            return ExprNode(op, [left, node(2 * slot + 2, depth + 1)])

        return node(0, 0)

    def _sample(self, island: int) -> np.ndarray:
        probability_one = np.sin(self.angles[island]) ** 2
        return (self.rng.random(self.bit_count) < probability_one).astype(np.uint8)

    def _rotate_toward(self, island: int, elite: np.ndarray) -> None:
        direction = np.where(elite == 1, 1.0, -1.0)
        self.angles[island] = np.clip(
            self.angles[island] + self.rotation * direction,
            0.10, math.pi / 2 - 0.10,
        )

    def run(self) -> dict:
        seen: set[str] = set()
        archive: list[dict] = []
        island_best: list[tuple[float, np.ndarray] | None] = [None] * self.islands
        draws = duplicates = rejected = 0
        rotation_batches = rotation_updates = 0

        def complete_draw() -> None:
            nonlocal rotation_batches, rotation_updates
            # A boundary after the final evaluation cannot influence another
            # draw, so it must not be counted as an effective update.
            if (draws % self.population_size == 0
                    and len(archive) < self.unique_budget and draws < self.max_draws):
                rotation_batches += 1
                for j, best in enumerate(island_best):
                    if best is not None and self.rotation > 0:
                        self._rotate_toward(j, best[1])
                        rotation_updates += 1

        while len(archive) < self.unique_budget and draws < self.max_draws:
            island = ((draws // self.population_size) + draws) % self.islands
            tape = self._sample(island)
            draws += 1
            tree = self._decode(tape)
            expression = tree.to_string()
            if expression in seen:
                duplicates += 1
                complete_draw()
                continue
            seen.add(expression)
            if tree.complexity > self.forge.max_complexity:
                rejected += 1
                complete_draw()
                continue
            if getattr(self.forge, "_is_fsa_forbidden", lambda _: False)(tree):
                rejected += 1
                complete_draw()
                continue
            if (getattr(self.forge, "dim_prune", "off") == "enforce" and
                    not self.forge._dim_audit(tree)):
                rejected += 1
                complete_draw()
                continue
            # Compilation is a shared Forge contract; invalid expressions do
            # not consume economic evaluations but remain visible as rejects.
            try:
                pandas_expression = tree.to_pandas_string()
            except (TypeError, ValueError):
                rejected += 1
                complete_draw()
                continue
            fitness = float(self.forge._fitness(tree))
            if not math.isfinite(fitness):
                fitness = -999.0
            row = {
                "expression": expression,
                "pandas_expression": pandas_expression,
                "fitness": fitness,
                "complexity": float(tree.complexity),
                "depth": int(tree.depth()),
                "terminals": sorted(tree.terminals()),
                "root_op": tree.primitive.name,
                "island": island,
            }
            archive.append(row)
            current = island_best[island]
            if fitness > -999 and (current is None or fitness > current[0]):
                island_best[island] = (fitness, tape)
            complete_draw()
        archive.sort(key=lambda row: (-row["fitness"], row["expression"]))
        probabilities = np.sin(self.angles) ** 2
        bit_entropy = -(probabilities * np.log2(probabilities)
                        + (1 - probabilities) * np.log2(1 - probabilities))
        return {
            "algorithm": ("classical_quantum_inspired_forge_challenger"
                          if self.rotation else "same_grammar_random_control"),
            "seed": self.seed,
            "unique_budget_requested": self.unique_budget,
            "unique_evaluated": len(archive),
            "draws": draws,
            "duplicates": duplicates,
            "pre_eval_rejected": rejected,
            "rotation_batches": rotation_batches,
            "rotation_updates": rotation_updates,
            "rotation": self.rotation,
            "angle_std": float(np.std(self.angles)),
            "mean_bit_entropy": float(np.mean(bit_entropy)),
            "root_ops_covered": sorted({row["root_op"] for row in archive}),
            "budget_met": len(archive) == self.unique_budget,
            "fitness_mode": self.forge.fitness_mode,
            "target_paradigm": self.target_paradigm,
            "max_depth": self.max_depth,
            "max_complexity": self.forge.max_complexity,
            "candidates": archive,
        }
