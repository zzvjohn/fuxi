# Fuxi — A Self-Evolving Alpha Factor Mining System for the A-Share Market

**A pipeline that discovers alpha on its own.** Starting from 109+ seed factor templates, Fuxi runs a closed loop of
*generate → multi-stage evaluate → portfolio screen → live backtest → knowledge distillation → feedback & re-evolve*,
**continuously and autonomously mining new quantitative factors** — turning failed experiments into distilled constraints
and successful patterns into new seeds, all without human intervention.

> Version: v0.7 (frequency-symmetric dual channel) · Language: Python 3.10+ · Backtest platform: JoinQuant (JQ)
> The Chinese README is the primary document; see [README.md](README.md) for the full version.

---

## Why "Self-Evolving"

Fuxi is not a backtesting tool for a fixed set of factors — it is an **evolutionary system that searches factor space,
self-corrects, and accumulates experience**:

| Capability | Mechanism |
|-----------|-----------|
| **Autonomous exploration** | Three parallel generation engines: GP genetic programming (60%) / LLM formula generation / Forge paradigm generator |
| **Learned scheduling** | A multi-armed bandit (MAB) allocates compute across engines based on their historical yield, balancing exploration vs. exploitation |
| **Multi-stage natural selection** | Five evaluation tiers S1–S5 (IC/ICIR → robustness → crowding gates → joint multi-signal screening), with elimination at each tier |
| **Experiential memory** | Three-layer Experience Memory (factor experience / paradigm state / trajectory pool) injects lessons learned into generation prompts |
| **Knowledge distillation** | Every real backtest automatically distills conclusions into "no-fly zone" constraints and new seed templates |
| **Feedback loop** | JoinQuant backtest results are written back to memory and trajectories (D+ stage), closing the full evolution loop |

Each evolution round makes the next generation **smarter at avoiding proven pitfalls** (backtest contamination,
crowded alpha, overfitting patterns) instead of repeating random trial-and-error in place.

---

## System Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │           Self-Evolution Main Loop (Ralph)     │
                    └──────────────────────────────────────────────┘
  Seed library ──► MAB scheduling ──► R: Experience Memory ──► G: Generation engines
 (109+ templates)  (explore/exploit)  (3-layer distillation)  ├─ gp_breed genetic programming (60%)
                                                               ├─ LLM formula generation
                                                               └─ Forge paradigm generator
        ▲                                                            │
        │                                                            ▼
   D+: JQ feedback injection                              E: Multi-stage eval S1~S5
 (results back to memory)                     (IC/ICIR → robustness → crowding → joint)
        ▲                                                            │
        │                                                            ▼
   D: 3-layer distillation ◄── JQ live backtest ◄── MMR portfolio screen ◄── S5 passed
```

### Core Design Principles

1. **JoinQuant is the single source of truth** — local IC only vetoes, never ranks;
2. **Evolution ≠ optimization** — exploring new factor space beats tuning inside old space;
3. **LLM only writes formulas, never selects factors** — statistical screening and portfolio optimization are rule-based;
4. **Domain pairing beats statistical screening** — every factor must map to investment logic or behavioral finance rationale;
5. **Mandatory per-factor IC/ICIR disclosure** — every JQ backtest must report per-factor Rank IC/ICIR.

### Pipeline Stages

| Stage | Module | Responsibility |
|-------|--------|----------------|
| Seed injection | `seed_injector.py` | Inject 109+ seed templates, maintain library diversity |
| MAB scheduling | `factor_model_cooptim.py` | Multi-armed bandit schedules the three engines |
| Memory retrieval | `experience_memory.py` | 3-layer memory: factor experience / paradigm state / trajectories |
| Generation | `gp_breed` (ga/) / `llm_generator.py` / `forge/` | Three engines explore in parallel |
| Evaluation S1–S5 | `factor_ic_computer.py`, `s5_joint_filter.py`, `multi_stage_validator.py` | IC/ICIR → robustness → crowding → joint screening |
| Portfolio | `mmr_selector.py` | Maximal Marginal Relevance selection, de-redundancy |
| JQ validation | `jq_generator/` | Auto-generate JoinQuant backtest code after S5 passes |
| Knowledge distillation | `experience_memory.py`, `trajectory_pool.py` | Distill results into factor library & no-fly zones |
| Feedback loop | `trigger_d_plus.py`, `trajectory_logger.py` | JQ results written back to memory & trajectories |

### v0.6 Experimental Guard Modules (disabled by default, toggled in `config.py`)

`holdout_boundary` / `behavior_homogeneity` / `multiple_testing_guard` (DSR·PBO·BH-FDR) /
`direction_campaign` / `budget_ledger` / `contamination_ledger` / `strategy_spec`

---

## Examples of Evolved Output

Factors autonomously discovered and validated by the system during unattended iterations
(JoinQuant historical backtests, shown only as capability demos — **not indicative of future performance**):

| Factor | Discovery Engine | Rationale | Backtest Return |
|--------|-----------------|-----------|-----------------|
| 002 price-volume relation | GP genetic programming | Mean reversion of price-volume divergence | +143.5% |
| Tail risk (ts_std composite) | Forge paradigm generator | Penalizing volatility-tail exposure | +91.9% |
| Broken-board pullback satellite v2 | LLM + crowding gates | Pullback momentum after limit-up streaks | +91.0% |
| Composite portfolio (AlphaAgent v3) | MMR multi-factor blend | Cross-paradigm de-redundant synthesis | +198.2% |

---

## Repository Layout

```
fuxi/
├── README.md / README_EN.md
├── LICENSE                  # MIT
├── .env.example             # Credentials template (copy to .env and fill in)
├── credentials.py           # Unified credential loader (Tushare / JoinQuant / WorldQuant)
├── factor_alchemy/          # Core system
│   ├── run_v4_pipeline.py   # Unified v4 entry point (--ralph starts self-evolution)
│   ├── ralph_loop.py        # Self-evolution main loop
│   ├── config.py            # System-wide config & experiment toggles
│   ├── forge/               # Forge paradigm generation engine
│   ├── ga/                  # GP genetic programming engine (NSGA-II)
│   ├── evaluation/          # Single-factor evaluation (IC/quantile/correlation/robustness)
│   ├── evaluator/           # Shadow evaluators
│   ├── factors/             # Factor library primitives
│   ├── portfolio/           # Portfolio simulator
│   ├── jq_generator/        # JoinQuant backtest code generator
│   └── ...
├── scripts/                 # Companion pipeline scripts (proposals/collection/dashboard/crowding)
└── docs/                    # Supplementary docs
```

---

## Quick Start

### 1. Environment

```bash
pip install numpy pandas scipy tushare jqdatasdk
```

### 2. Configure Credentials

```bash
cp .env.example .env
# Fill in .env:
#   [tushare]   token      — Tushare Pro token
#   [joinquant] username/password — JoinQuant JQData account
#   [worldquant] email/password   — (optional) WorldQuant BRAIN credentials
```

The LLM generation engine requires a DeepSeek API key (injected via environment variable):

```bash
export DEEPSEEK_API_KEY="sk-..."
```

### 3. Directory Conventions

- Market/factor data lives in `data/` by default (override with `FUXI_DATA_DIR`);
- Outputs go to `output/` (override with `FUXI_OUTPUT_DIR`);
- Optional: set `RAG_DIR` to a local research knowledge base for automatic prompt injection.

### 4. Run

```bash
cd factor_alchemy

# System snapshot
python run_v4_pipeline.py --view

# Full diagnostic report
python run_v4_pipeline.py --diagnose

# Start the self-evolution main loop (Ralph Loop)
python run_v4_pipeline.py --ralph
```

---

## Security

- This repository contains **no account credentials**; all keys are injected via `.env` / environment variables;
- `.env` is listed in `.gitignore` — never commit it.

## Disclaimer

This project is for quantitative research and education only. All backtest figures are historical simulations and
**do not constitute investment advice**; past performance does not guarantee future results.
The A-share market carries risk — invest with caution.
