import unittest
from fuxi_v1.contracts import ContractError, digest
from fuxi_v1.effective_config import collect_effective_config, DOMAIN_KNOWN_KNOBS
from fuxi_v1.run_spec import (build_run_spec, validate_run_spec, run_spec_digest,
                              RUN_SPEC_SCHEMA)


def sha(value):
    from hashlib import sha256
    return sha256(repr(value).encode("utf-8")).hexdigest()


class EffectiveConfigTests(unittest.TestCase):
    def test_defaults_materialized(self):
        cfg = collect_effective_config({})
        self.assertEqual(cfg["schema_version"], "fuxi.effective_config.1")
        for name, default in DOMAIN_KNOWN_KNOBS.items():
            self.assertEqual(cfg["resolved"][name], default)

    def test_override_captured(self):
        env = {"FUXI_V1_GEN_EXPAND": "expanded"}
        cfg = collect_effective_config(env)
        self.assertEqual(cfg["resolved"]["FUXI_V1_GEN_EXPAND"], "expanded")
        self.assertEqual(cfg["resolved"]["FUXI_V1_PARENT_POLICY"], "pareto")

    def test_extra_fuxi_knobs_captured(self):
        cfg = collect_effective_config({"FUXI_S5_COST_BPS": "30"})
        self.assertEqual(cfg["extra_resolved"]["FUXI_S5_COST_BPS"], "30")

    def test_secrets_excluded(self):
        cfg = collect_effective_config({"FUXI_DEEPSEEK_API_KEY": "sk-secret",
                                        "FUXI_SOME_TOKEN": "tok"})
        self.assertNotIn("FUXI_DEEPSEEK_API_KEY", cfg["resolved"])
        self.assertNotIn("FUXI_DEEPSEEK_API_KEY", cfg["extra_resolved"])
        self.assertNotIn("FUXI_SOME_TOKEN", cfg["extra_resolved"])

    def test_unset_equals_explicit_default(self):
        unset = collect_effective_config({})
        explicit = collect_effective_config({k: v for k, v in DOMAIN_KNOWN_KNOBS.items()})
        self.assertEqual(digest(unset), digest(explicit))

    def test_gen_expand_changes_identity(self):
        base = {"request": {"input": "req"}, "code": {"a": "b"}, "limits": {"x": 1}}
        a = dict(base, effective_config=collect_effective_config({}))
        b = dict(base, effective_config=collect_effective_config({"FUXI_V1_GEN_EXPAND": "expanded"}))
        self.assertNotEqual(digest(a), digest(b))
        a2 = dict(base, effective_config=collect_effective_config({}))
        self.assertEqual(digest(a), digest(a2))


class RunSpecTests(unittest.TestCase):
    def _spec(self, **overrides):
        spec = {"schema_version": RUN_SPEC_SCHEMA, "task_kind": "candidate_research",
                "snapshot_ref": sha("snap"), "universe_ref": sha("uni"), "label_ref": sha("lab"),
                "split_ref": sha("split"),
                "generator_config": {"mode": "expanded", "operator_catalog_ref": sha("ops"),
                                     "allowed_actions": ["template", "edit", "combine", "hypothesis"],
                                     "seed": 17},
                "screen_policy_ref": sha("sp"), "evaluation_policy_ref": sha("ep"),
                "baseline_strategy_ref": sha("bs"), "execution_cost_ref": sha("ec"),
                "memory_view_ref": sha("mv"), "budget_ref": sha("bud"), "code_ref": sha("code"),
                "effective_config_ref": sha("cfg")}
        spec.update(overrides)
        return spec

    def test_valid_and_stable_digest(self):
        a = run_spec_digest(self._spec())
        b = run_spec_digest(self._spec())
        self.assertEqual(a, b)

    def test_every_semantic_field_changes_digest(self):
        base = run_spec_digest(self._spec())
        fields = ("snapshot_ref", "universe_ref", "label_ref", "split_ref", "screen_policy_ref",
                  "evaluation_policy_ref", "baseline_strategy_ref", "execution_cost_ref",
                  "memory_view_ref", "budget_ref", "code_ref")
        for field in fields:
            changed = self._spec(**{field: sha(field + "-changed")})
            self.assertNotEqual(base, run_spec_digest(changed), field)

    def test_generator_config_fields_change_digest(self):
        base = run_spec_digest(self._spec())
        for patch in ({"mode": "legacy"}, {"seed": 18},
                      {"allowed_actions": ["template"]},
                      {"operator_catalog_ref": sha("ops2")}):
            gen = dict(self._spec()["generator_config"]); gen.update(patch)
            self.assertNotEqual(base, run_spec_digest(self._spec(generator_config=gen)), patch)

    def test_effective_config_ref_changes_digest(self):
        base = run_spec_digest(self._spec())
        changed = self._spec(effective_config_ref=sha("cfg2"))
        self.assertNotEqual(base, run_spec_digest(changed))

    def test_missing_field_rejected(self):
        spec = self._spec(); del spec["label_ref"]
        with self.assertRaises(ContractError):
            validate_run_spec(spec)

    def test_bad_ref_rejected(self):
        with self.assertRaises(ContractError):
            validate_run_spec(self._spec(snapshot_ref="not-a-hex"))

    def test_unknown_key_rejected(self):
        with self.assertRaises(ContractError):
            validate_run_spec(self._spec(extra_key=1))

    def test_build_roundtrip(self):
        out = build_run_spec(task_kind="candidate_research", snapshot_ref=sha("snap"),
                             universe_ref=sha("uni"), label_ref=sha("lab"), split_ref=sha("split"),
                             generator_config=self._spec()["generator_config"],
                             screen_policy_ref=sha("sp"), evaluation_policy_ref=sha("ep"),
                             baseline_strategy_ref=sha("bs"), execution_cost_ref=sha("ec"),
                             memory_view_ref=sha("mv"), budget_ref=sha("bud"), code_ref=sha("code"),
                             effective_config_ref=sha("cfg"))
        self.assertEqual(out["run_spec_digest"], run_spec_digest(self._spec()))


if __name__ == "__main__":
    unittest.main()
