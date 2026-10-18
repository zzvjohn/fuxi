# Fuxi v2.0 — A-Share Factor Research Closed Loop

> **Product version**: **v2.0.0** (aligned with the product name, defined 2026-10-18) | **Kernel version**: 2.0.0 (`fuxi_v1/__init__.py::__version__`)
> **Engineering ledger**: v0.7.218 (`data/version_registry.json`, 219 batch entries)
> **Language**: Python 3.10+ | **Backtest platform**: JoinQuant (manual upload / raw-result hand-back)
> Chinese version: [README.md](README.md) | Architecture: [ARCHITECTURE.md](ARCHITECTURE.md)

---

## Why it is worth a try

In one line: **it turns hand-tuning factors into an automatic search that is reproducible,
auditable, and actually learns from its own history — without promising returns.**

| The usual pain in factor research | What Fuxi does | Where to check |
|---|---|---|
| Results change with the seed; nothing reproduces | Same seed + snapshot + config reproduces the same trajectory; parallel results merge by `candidate_id`, never by completion order | `unified_runner.py` / `campaign.py` |
| Great in backtest, dead in production | Graded evidence E0–E4: contract → statistics → independence → economic increment → stress; the JQ backtest is the only source of truth | `research_evaluation.py` |
| Every new factor looks like the last one | Cross-family pairing constraint + E2 in-pool redundancy gate (high correlation must show orthogonal residual alpha) | `ga_ops.py` / `research_evaluation.py` |
| Tried 200 times, picked the best, significance is inflated | The E1 hurdle tightens with the number of trials: `t = z(1 - alpha/(2*max_trials))` | `research_protocol.py` |
| A 2026 paper guiding a 2020 backtest | Knowledge cards must record `knowledge_available_at` — when the knowledge existed, not when you found it | `scripts/knowledge_card.py` |
| Sampling drifted quietly, backtest still looks great | Hash sampling + SSE/SZSE coverage assertion; `sorted(codes)[:N]` is forbidden | `datasets.py` |
| Concurrent writers overwrite state | Writer authority + real process-lease lock | `governance.py` |

**Performance, not marketing** (32 cores / 34 GB, 1212-session x 64-instrument snapshot):

- single candidate 63.7 s → warm loop **31.2 s** (1.36x)
- 4-worker parallel **3.84x** (64.7 s wall vs 248.4 s serial)
- 20 rounds x 4 candidates: 83 minutes serial → **25–30 minutes**

**What it deliberately refuses to do** (this is why it can be trusted): it does not admit candidates,
does not touch a live account, does not modify the judge, does not publish strategies.
`live_execution_allowed` / `QMT_orders_enabled` default to `False`; crossing that line requires a
manual signature from the account owner.

**Three commands to start**:

```bash
pip install -r requirements-env-default.txt
python -X utf8 -m fuxi_v1 system-fixture --output reports/_smoke/synthetic.json
python -X utf8 -m fuxi_v1 campaign --input reports/_smoke/synthetic.json \
  --output reports/_smoke/campaigns --rounds 6 --candidates 2
```

> This project **promises no returns**. Every number in this README can be traced to an
> implementation or a measurement inside the repository; past backtests do not indicate future results.

---

## 0. In 30 seconds

Fuxi is a **closed-loop research system for A-share quant factors**: hypothesis generation,
expression search, graded evidence, JQ backtesting, and forward archiving — all driven by
*reproducible declarations plus transactional evidence*, never by filenames or human narrative.

Three non-negotiable principles:

1. **Evidence-driven** — every conclusion must be anchored to an implementation, a data column, or a structured field.
2. **No verdict without evidence** — an execution failure is not a falsified direction; insufficient data is not an invalid factor.
3. **Determinism first** — same seed, same snapshot, same config must reproduce the same learning trajectory.

---

## 1. Version coordinates (all three coexist)

| Coordinate | Meaning | Carrier |
|---|---|---|
| **v2.0.0** | **Product version = kernel version** (aligned, 2026-10-18) | `fuxi_v1/__init__.py` · `VERSION` |
| v2.0 | Product release architecture: the complete system described here | `ARCHITECTURE.md` |
| v0.7.x | Engineering iteration ledger (per-batch proposals), currently v0.7.218 | `data/version_registry.json` |

> The version number is now **2.0.0**, aligned with the product name v2.0 (`__version__` moved straight
> from `1.0.0-rc12` to `2.0.0`, with no stale value left behind).
> Seeing `v0.7.218` does **not** mean the system is still on the v0.7 architecture: v0.7.x is an
> **engineering batch number**, not the product version. Both are recorded in parallel by design.

---

## 2. Boundaries: what it does not do

`fuxi_v1.unified_runner.run()` is the single workflow entry point. It does exactly one thing:
produce auditable research evidence inside a frozen request and budget. It **cannot**:

- admit a candidate (admission is decided by E-gate evidence)
- submit to a JQ live account (export and execution are strictly separated)
- modify the judge or the evaluation policy
- publish a strategy

No live trading client is in scope. `live_execution_allowed` and `QMT_orders_enabled` default to
`False`; crossing that line requires the account owner to **sign manually**. The system never
lifts it automatically.

---

## 3. Layered architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Entry      cycle_orchestrator (daily) · unified_runner (one) │
├─────────────────────────────────────────────────────────────┤
│ Data       native64 snapshot · historical PIT · columnar grid│
├─────────────────────────────────────────────────────────────┤
│ Research   L0 archetypes → L1 hypotheses → L2 arms → L3 reflect│
├─────────────────────────────────────────────────────────────┤
│ Search     fixed / random_ast / refine / combine (bounded GA)│
├─────────────────────────────────────────────────────────────┤
│ Evaluation E0 contract → E1 stats → E2 independence → E3 → E4 → E5│
├─────────────────────────────────────────────────────────────┤
│ Evidence   evidence.sqlite (transactional facts) · governance│
├─────────────────────────────────────────────────────────────┤
│ Exit       JQ export · forward archive · data certification  │
└─────────────────────────────────────────────────────────────┘
```

Data flows one way: **data → research → search → evaluation → evidence → exit**. No layer may
rewrite a frozen artifact upstream.

---

## 4. Directory layout

```
e:\quant\
├── fuxi_v1/                  # v2.0 kernel (research main path)
│   ├── unified_runner.py     # single bounded workflow (source-keyed idempotency)
│   ├── __main__.py           # CLI command family
│   ├── campaign.py           # campaign loop / L1-L2 scheduling
│   ├── search.py ga_ops.py dsl.py edits.py          # search layer
│   ├── research_evaluation.py research_protocol.py  # E gates
│   ├── evidence.py governance.py monitoring.py      # evidence layer
│   ├── export.py jq_strategy.py forward_archive.py  # exit layer
│   └── tests/
├── scripts/                  # orchestration & ops
│   ├── cycle_orchestrator.py # daily loop
│   ├── rebuild_snapshot_1212.py
│   ├── knowledge_card.py     # knowledge cards: init/add/list/promote/verify
│   └── version_bump.py       # version ledger
├── research/factor_alchemy/  # legacy v0.7 library (LLM client still reused)
├── data/                     # raw data & snapshots (large, not committed)
│   ├── raw/
│   ├── fuxi_unified/snapshots/
│   └── version_registry.json
├── reports/                  # decision-facing outputs (daily / audit / verdicts)
├── ARCHITECTURE.md           # authoritative v2.0 architecture
├── AGENTS.md                 # project rules and runtime discipline
├── .env.example              # credential template (no real values)
└── credentials.py            # unified credential loader (Tushare / JQ / WorldQuant)
```

---

## 5. Quick start

### 5.1 Environment

Python 3.10+. The project convention is to reuse the **existing interpreter** rather than
creating a new environment (see `AGENTS.md`). On Windows, add `-X utf8` for scripts that
print non-ASCII output:

```bash
python -X utf8 -m fuxi_v1 --help
```

### 5.2 Dependencies

```bash
pip install -r requirements-env-default.txt
```

Minimal set for manual checks: `numpy pandas scipy tushare jqdatasdk`.

### 5.3 Credentials

**Channel 1 — data platforms (Tushare / JoinQuant / WorldQuant)**, loaded by `credentials.py`:

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

Section names in `.env` must not be changed:

```ini
[tushare]
token = YOUR_TUSHARE_TOKEN        # placeholder — put your own 40-hex token here

[joinquant]
username = YOUR_JQ_USERNAME
password = YOUR_JQ_PASSWORD

[worldquant]
email = YOUR_WQ_EMAIL             # optional, not on the main path
password = YOUR_WQ_PASSWORD
```

Self-check (output is auto-masked, first 3 + last 2 characters only):

```bash
python -X utf8 credentials.py
```

**Channel 2 — LLM (DeepSeek)**, resolved by
`research/factor_alchemy/llm_client.py::resolve_api_key_with_source` in this order:

| Priority | Source | Usage |
|---|---|---|
| 1 | Explicit argument | `explicit=` in code |
| 2 | Process env var | `DEEPSEEK_API_KEY=<DEEPSEEK_API_KEY>` |
| 3 | Local credential files | `.env` / `.env.local` / `config/secrets.local.json` |
| 4 | Windows user env var | read from registry, bypassing process snapshot |

```bash
# Linux / macOS
export DEEPSEEK_API_KEY="<DEEPSEEK_API_KEY>"
# Windows PowerShell
$env:DEEPSEEK_API_KEY = "<DEEPSEEK_API_KEY>"
```

Without a key the LLM arm degrades gracefully; GP and Forge paths are unaffected.

### 5.4 Prepare data

Build a snapshot from local raw CSVs — **hash sampling is mandatory**, never string-sorted sampling:

```bash
python -X utf8 -m fuxi_v1 snapshot-local \
  --raw-root data/raw --start 2024-01-01 --end 2026-09-30 \
  --max-instruments 64 --output data/fuxi_unified/snapshots/mysnap
```

> v2.0 fix: the old `sorted(codes)[:N]` let `000*.SZ` monopolize the sample. Sampling now uses
> `hash(salt, ts_code)` and asserts coverage on both SSE and SZSE. `--legacy-code-order` exists
> only to **explicitly reproduce** the old biased diagnostic sample — never use it in production.

### 5.5 Run the first pipeline

```bash
# 1) Synthetic fixture — verifies the engineering path without real market data
python -X utf8 -m fuxi_v1 system-fixture --output reports/_smoke/synthetic.json

# 2) Small campaign (6 rounds x 2 candidates)
python -X utf8 -m fuxi_v1 campaign \
  --input reports/_smoke/synthetic.json \
  --output reports/_smoke/campaigns \
  --rounds 6 --candidates 2 --mode adaptive

# 3) Verify artifacts
python -X utf8 -m fuxi_v1 verify reports/_smoke/campaigns
```

For real data pass `--input <snapshot path>`; optionally add `--llm`,
`--models ridge,lightgbm,xgboost` (empty string disables), or `--resume-id campaign_<ID>`.

### 5.6 Command reference

| Command | Purpose |
|---|---|
| `python -m fuxi_v1 demo --output <dir>` | Minimal demo pipeline |
| `python -m fuxi_v1 system-fixture --output <json>` | Synthetic fixture |
| `python -m fuxi_v1 campaign --input --output --rounds --candidates` | Main research loop |
| `python -m fuxi_v1 snapshot-local --raw-root --start --end --output` | Local snapshot |
| `python -m fuxi_v1 snapshot-history --plan --raw-root --supplement --output` | Historical PIT snapshot |
| `python -m fuxi_v1 certify-data --input --apply` | Derive data-contract flags |
| `python -m fuxi_v1 pit-cohort --raw-root --start --end --salt` | Declare a PIT cohort |
| `python -m fuxi_v1 forward-archive` | Forward decision archive |
| `python -m fuxi_v1 operate` | Operations (guarded by the governance lock) |
| `python -m fuxi_v1 work-packages` / `blueprint` | Work-package & blueprint state |
| `python scripts/version_bump.py --bump` | Register a version entry |

---

## 6. Configuration

### 6.1 Search and execution (`FUXI_V1_*`)

| Variable | Default | Effect |
|---|---|---|
| `FUXI_V1_GA_OPS` | `on` | refine/combine use bounded GA; `off` reverts to single-leaf mutation |
| `FUXI_V1_PARALLEL_CANDIDATES` | `off` | Evaluate candidates of a round in parallel |
| `FUXI_V1_PERSISTENT_POOL` | `off` | Persistent worker pool reusing the snapshot cache |
| `FUXI_V1_POOL_SIZE` | `2` | Pool size (4 recommended) |
| `FUXI_V1_POOL_MAX_REQUESTS` | `8` | Requests per slot before recycle (40 recommended) |

### 6.2 Budget constants (in code, `fuxi_v1/ga_ops.py`)

```python
GA_POPULATION_PER_GENERATION = 4      # new individuals per generation
GA_MIN_GENERATIONS = 10               # minimum actual GA generations
GA_TRIALS_BUDGET = (60, 120)          # target max_trials range
```

Recommended: `rounds=20`, `candidates_per_round=4` → `max_trials=80`, t-hurdle 3.42, ~10 GA generations.

### 6.3 Legacy S5 channel

The legacy v0.7 S5 channel is still controlled by `FUXI_*` switches (`COST_BPS=15`,
`MIN_COVERAGE=0.9`, `TRADABLE_GATE=on`, `ADJ_PRICE=on`, `DATE_CEIL=auto`, etc.). The full list
and the default-flip discipline live in `AGENTS.md`. **The v2.0 research main path (E gates)
does not read these switches** — do not conflate the two.

---

## 7. Artifacts and contracts

The two entry points produce different artifacts — do not conflate them:

**`campaign` command** (verified by an actual run):

```
<output>/
├── authority.sqlite          # cross-campaign authority ledger
└── campaign_<id>/
    ├── config.json input.json policy.json
    ├── code/                 # frozen source copy (edits during a run block the report)
    ├── evidence.sqlite       # transactional facts for this campaign
    ├── attempts/             # per-attempt request / result / process_guard
    ├── models/ bundles/
    └── report_<hash>.json    # one JSON line per round + a final summary line
```

**`unified_runner.run()`**:

| File | Content |
|---|---|
| `RUN_PROTOCOL.json` | code hash, `effective_config`, request, limits, identity |
| `PHASE_RESEARCH.json` | research report (candidate_results / rounds / archive) |
| `COMPLETE.json` | completion receipt |
| `research/RECEIPT.json` | delivery receipt pointing at report_path |
| `evidence.sqlite` | facts / campaigns / attempts / rewards / exposure / receipts |

Main schemas: `fuxi.campaign_input.1`, `fuxi.unified_run.1`, `fuxi.snapshot.2`,
`fuxi.columnar_snapshot.1`, `fuxi.knowledge_card.1`, `fuxi.cost_spec.1`, `fuxi.data_certification.1`.

---

## 8. Evaluation gates E0–E5

| Gate | What it decides |
|---|---|
| E0 | Whether data and PIT contracts support the candidate |
| E1 | Direction, yearly stability, **positive IC on both exchanges**, block-bootstrap lower bound > 0 |
| E2 | Independence: paired cross-sectional correlation + style attribution + single-field sub-channel ablation |
| E3 | Economic increment: paired-account net return difference, net of cost and capacity pressure |
| E4 | Robustness: cost-multiple / participation-multiple stress |
| E5 | Forward evidence: pre-registered look after JQ hand-back |

The E1 hurdle tightens with the number of trials:

```
t_hurdle = z(1 - alpha / (2 * max_trials))
min_detectable_ICIR = t_hurdle / sqrt(n_sections)
```

For a 1125-section research window, `max_trials` 80 gives t=3.42 and min detectable ICIR=0.102.

> **JQ is the only source of truth**: local IC may veto but never rank. Every JQ backtest must
> report per-factor Rank IC/ICIR.

---

## 9. Current status and known limits

This section only states readings that have been actually checked:

- **v2.0 is defined** (2026-10-18) and the E-gate research main path is usable; the engineering
  ledger is registered up to v0.7.218.
- **The G channel (G0 legality precheck / G1 diagnostic routing / G2 account seats) is still in
  shadow validation** (P5 shadow). It writes facts only and takes part in no decision, which is
  why it is excluded from the `ARCHITECTURE.md` main path.
- **Blueprint §9 statistical acceptance is still accumulating** (requires >=100 shadow rounds,
  independent repeats, and fixes for in-flight items). Shadow readings are not economic
  conclusions and are not reported here as passed.
- **Do not trust legacy environment variables**: `FUXI_DATA_DIR`, `FUXI_OUTPUT_DIR` and `RAG_DIR`
  claimed by early READMEs have **no consumer** in the current `fuxi_v1/` or `scripts/`. They were
  removed from this document; paths are passed explicitly via `--output` / `--raw-root` / `--input`.
- **Cost basis**: account replay still uses the 15bps proxy; the native `CostSpec` rate map exists
  at the contract layer but is not yet the production default.
- **The suspension/resumption table** only extends to 2026-04-30; corporate-action and PIT label
  gaps remain and must be disclosed explicitly in any conclusion.

---

## 10. Credentials and sanitization policy

This repository contains **no real credentials**. Sensitive values in docs, examples and logs are
replaced by these placeholders:

| Type | Placeholder | Forbidden |
|---|---|---|
| Tushare token | `YOUR_TUSHARE_TOKEN` | the real 40-hex value |
| JoinQuant account / password | `YOUR_JQ_USERNAME` / `YOUR_JQ_PASSWORD` | phone number / plaintext password |
| WorldQuant | `YOUR_WQ_EMAIL` / `YOUR_WQ_PASSWORD` | real email and password |
| DeepSeek | `<DEEPSEEK_API_KEY>` | any real `sk-` value |
| Account / capital | `<ACCOUNT_ID>` / `<INITIAL_CAPITAL>` | broker account id, real capital amount |

Rules:

1. Real values live only in `.env` / `.env.bt` / `config/secrets.local.json`, all git-ignored.
2. Pre-commit check: `grep -rnE "sk-[A-Za-z0-9]{20,}" --include=*.md --include=*.py .`
3. Historical credentials in notes and logs must be sanitized too — **history may be kept, plaintext secrets may not**.
4. A leaked key must be rotated at the provider; deleting it from a file is not remediation.

---

## 11. Disclaimer

This project is for quantitative research and learning only. All backtest figures are historical
simulations and **do not constitute investment advice**; past performance does not guarantee future
results. A-share markets carry risk.
