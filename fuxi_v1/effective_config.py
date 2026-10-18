"""Resolved behaviour-affecting environment knobs for run identity (P0.1, v2.0 landing).

The R2 blueprint (BLUEPRINT.md section 7) requires: every behaviour-affecting
environment variable is resolved at startup into an ``effective_config`` that
enters the run identity digest, and workers consume the config snapshot instead
of hidden runtime overrides. The concrete defect this module fixes is that
``FUXI_V1_GEN_EXPAND`` (an opt-in generator switch) was absent from the
unified_runner identity, so two runs differing only in generator expansion
collided on the same run_id (CONTRACTS.md, "共同对象与身份").

Design constraints honoured here:

* Only the knobs actually read inside ``fuxi_v1/`` (the unified_runner domain)
  are declared with defaults. The v0.7 S5-domain knobs (``FUXI_S5_*`` etc.) are
  resolved by the S5 runner itself and belong to that identity domain; they are
  still captured if present in the environment, but no defaults are assumed.
* Defaults are materialized, so "unset on machine A" and "explicitly set to the
  default on machine B" resolve to the same identity.
* Secrets (API keys / tokens / passwords) never enter the digest.
* Any FUXI_* variable not in the declared map but present in the environment is
  still captured (future knobs cannot silently escape the identity), under
  ``extra_resolved`` so a knob added to the code later is also covered.
"""

import os

EFFECTIVE_CONFIG_SCHEMA = "fuxi.effective_config.1"

# Declared defaults for the knobs read by fuxi_v1 modules (campaign.py / search.py).
# Keep in sync when a new env knob is read in this package.
DOMAIN_KNOWN_KNOBS = {
    "FUXI_V1_PARENT_POLICY": "pareto",    # campaign.py:564
    "FUXI_V1_MOTIF_MEMORY": "off",        # campaign.py:566
    "FUXI_V1_FITNESS_TRIAGE": "enforce",  # campaign.py:568
    "FUXI_V1_MIN_EXPLORATION": "enforce", # campaign.py:570
    "FUXI_V1_GEN_EXPAND": "legacy",       # campaign.py:572 / search.py:121
    "FUXI_V1_P5_SHADOW": "on",            # campaign.py:576 / landing_shadow.py (P5, v0.7.186翻on)
    "FUXI_V1_SCORE_REF_KEY": "on",        # campaign.py:578 / dsl.py (P5.1, v0.7.186翻on)
}

SECRET_MARKERS = ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "PRIVATE_KEY")


def _is_secret(name):
    upper = name.upper()
    return any(marker in upper for marker in SECRET_MARKERS)


def collect_effective_config(environ=None):
    """Resolve env knobs into a canonical, digestible config snapshot.

    ``environ`` defaults to ``os.environ``; pass a dict in tests/replay. The
    returned mapping is JSON-canonicalisable and is intended to be embedded in
    the run ``experiment`` dict before digesting (unified_runner.run).
    """
    source = os.environ if environ is None else environ
    resolved = {name: source.get(name, default)
                for name, default in sorted(DOMAIN_KNOWN_KNOBS.items())}
    extra = {name: value for name, value in sorted(source.items())
             if name.startswith("FUXI_") and name not in resolved and not _is_secret(name)}
    return {"schema_version": EFFECTIVE_CONFIG_SCHEMA,
            "domain": "unified_runner",
            "resolved": resolved,
            "extra_resolved": extra,
            "note": ("resolved knobs materialise declared defaults; extra_resolved captures "
                     "any additional FUXI_* present in the environment; secrets excluded")}
