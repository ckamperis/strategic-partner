# AI Strategic Partner

Proof-of-concept prototype for an AI-powered strategic partner that assists SME decision-makers with cashflow forecasting, risk assessment, SWOT analysis, and customer analytics. Built on a four-pillar architecture (Knowledge, Reasoning, Simulation, Trust) coordinated by the Pillar Integration & Context Protocol (PICP).

**Student ID:** ME2311,
**University:** University of Piraeus,
**Department:** Department of Digital Systems -- Big Data & Analytics,
**Author:** Christos Kamperis Konstantinos,
**Supervisor:** Michael Filippakis,
**Date:** 2026

## Project Statistics

| Metric | Value |
|--------|-------|
| Python files | 83 |
| Lines of code (excl. tests) | 9,291 |
| Test suite | 459 tests (446 unit + 13 integration) |
| Code coverage | 89.3% |
| Experiment scripts | 13 |
| Thesis figures | 8 (PNG + PDF) |

## Architecture

```
Query -> [Routing Classifier]
              |
     +--------+--------+--------+
     |        |        |        |
 Knowledge  Reasoning  Simulation  Trust
  (RAG)    (Heuristic)  (Monte    (Eval +
            (Skills)    Carlo)    SHAP +
                                  Audit)
     |        |        |        |
     +--------+--------+--------+
              |
         [Orchestrator] -> Response
              |
         [PICP Bus] (events, vector clocks, distributed locks)
```

**Four pillars:**

- **Knowledge Pillar** -- Hybrid search (BM25 + cosine, configurable alpha) with self-correcting RAG over real ERP data stored in ChromaDB.
- **Reasoning Pillar** -- Heuristic policy engine with YAML-defined skills for five query types: cashflow forecast, risk assessment, SWOT analysis, customer analysis, and general queries.
- **Simulation Pillar** -- Monte Carlo engine (10,000 simulations by default) with seasonal factors, credit note adjustment, and scenario analysis (base/optimistic/stress).
- **Trust Pillar** -- Composite trust score (explainability + consistency + accuracy), simulated SHAP explanations, Greek-language caveats, and append-only audit logs.

**Coordination layer (PICP):**

- Event bus (Redis pub/sub with in-memory fallback)
- Vector clocks for causal ordering
- Distributed locks for concurrent access
- Lifecycle enforcement (query_received -> ... -> response_ready)

## Prerequisites

- Python 3.11+ (tested on 3.12.9)
- pip (or any Python package manager)
- Optional: Redis (auto-falls back to in-memory)
- Optional: OpenAI API key (only for 5 of 13 experiments; tests and 8 offline experiments run without it)
- ERP Dataset: The file data/raw/cashflow_dataset.xlsx (129 MB) is not included in the repository due to size limits. Contact the author to obtain a copy, then place it at data/raw/cashflow_dataset.xlsx.

## Installation

```bash
git clone <repository-url>
cd strategic-partner

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

# Install all dependencies (runtime + dev + experiments)
pip install -e ".[dev,experiments]"
```

Alternatively, use the pinned requirements file:

```bash
pip install -r requirements.txt
```

## Quick Verification

Run these three commands to verify the installation:

```bash
# 1. Run the full test suite (459 tests, ~30s)
python -m pytest tests/ -v

# 2. Run one offline experiment (Trust sensitivity, <1s)
python -m pytest tests/ -q && python -m experiments.exp5_trust_sensitivity

# 3. Generate all thesis figures (8 PNG + 8 PDF, ~5s)
python -m experiments.generate_figures
```

Expected output for step 1: `459 passed` with no failures.

## Running Experiments

### Offline experiments (no API key required)

These 8 scripts are fully deterministic and reproducible:

```bash
# Statistical analysis of hybrid search alpha sweep
python -m experiments.exp1_statistical_analysis

# RAG failure pattern analysis
python -m experiments.exp2_failure_analysis

# Monte Carlo convergence and backtesting (5-year window)
python -m experiments.exp4_monte_carlo

# Monte Carlo extended (full 20-year dataset)
python -m experiments.exp4_monte_carlo_v3

# Trust score sensitivity analysis
python -m experiments.exp5_trust_sensitivity

# Query routing confusion matrix
python -m experiments.exp6_routing_analysis

# Graceful degradation (8 failure scenarios)
python -m experiments.exp7_degradation

# Generate all Chapter 5 figures
python -m experiments.generate_figures
```

### API-required experiments (need OPENAI_API_KEY)

These 5 scripts require an OpenAI API key. Create a `.env` file from the template:

```bash
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY
```

Then run:

```bash
# One-time data ingestion (creates embeddings in ChromaDB)
python -m experiments.setup_data

# Hybrid search alpha sweep with LLM-as-Judge
python -m experiments.exp1_hybrid_alpha

# Self-correcting RAG iteration evaluation
python -m experiments.exp2_rag_iterations

# PICP latency profiling (end-to-end pipeline)
python -m experiments.exp3_picp_latency

# End-to-end query evaluation
python -m experiments.exp6_end_to_end
```

### Run all experiments at once

```bash
# Run all (requires API key)
python -m experiments.run_all

# Skip data setup (if already ingested)
python -m experiments.run_all --skip-setup

# Run specific experiments by number (1-7)
python -m experiments.run_all --skip-setup --only 4 5 7
```

All results are saved as JSON in `data/results/`.

## Key Experiment Results

| Experiment | Key Finding |
|------------|-------------|
| Exp 1 (Hybrid Search) | Optimal alpha = 0.3, nDCG@5 = 0.926, 95% CI [0.833, 0.991] |
| Exp 2 (Self-Correcting RAG) | 66.7% success rate; 5 failures in 3 categories |
| Exp 3 (PICP Latency) | Mean latency 17.3s; PICP overhead <1ms |
| Exp 4 (Monte Carlo) | Convergence at N=500; 75% CI coverage, 25.4% MAPE |
| Exp 5 (Trust Sensitivity) | Score range 0.52--1.00; high/medium threshold at 0.75 |
| Exp 6 (Routing) | 80% accuracy (16/20); 4 misrouted queries |
| Exp 7 (Degradation) | 100% resilience (8/8 scenarios produce valid response) |

## Project Structure

```
strategic-partner/
  pyproject.toml              # Dependencies and project metadata
  requirements.txt            # Pinned dependency list
  .env.example                # Environment variable template
  config/
    settings.py               # Pydantic settings (reads .env)
  picp/                       # Pillar Integration & Context Protocol
    bus.py                    # Event bus (Redis / in-memory)
    distributed_lock.py       # Distributed locking
    enforcer.py               # PICP lifecycle enforcement
    vector_clock.py           # Vector clocks for causal ordering
    message.py                # PICP message schema
  pillars/
    base.py                   # Abstract base pillar
    knowledge/                # RAG + hybrid search + vector store
    reasoning/                # Heuristic policy + skill executor
      skills/                 # 4 YAML skill definitions
    simulation/               # Monte Carlo + distributions + scenario parser
    trust/                    # Evaluator + SHAP + explainer + audit
  orchestrator/
    __init__.py               # StrategicPartner class + FastAPI app
  utils/
    llm.py                    # OpenAI client wrapper
  data/
    raw/                      # Original ERP dataset (.xlsx)
    pipeline/                 # ETL: transformer, classifier, models
    results/                  # Experiment output (JSON + audit dirs)
  experiments/
    run_all.py                # Master experiment runner
    setup_data.py             # One-time ERP data ingestion
    generate_figures.py       # Thesis figure generation
    exp1-exp7 scripts         # Individual experiment scripts
    figures/                  # Generated PNG + PDF figures
  tests/
    unit/                     # 23 test modules (446 tests)
    integration/              # End-to-end pipeline tests (13 tests)
    conftest.py               # Shared fixtures
  scripts/
    ingest_erp_data.py        # Standalone data ingestion
    run_query.py              # CLI query interface
```

## Running the API Server

```bash
# Start the FastAPI server
uvicorn orchestrator:app --reload --port 8000
```

Note: The API server requires a running Redis instance and an OpenAI API key configured in `.env`. Without Redis, the PICP bus falls back to in-memory mode automatically.

## Testing

```bash
# Full test suite
python -m pytest tests/ -v

# Unit tests only
python -m pytest tests/unit/ -v

# With coverage report
python -m pytest tests/ --cov=picp --cov=pillars --cov=orchestrator --cov=data --cov=utils --cov-report=term-missing

# Run a specific test module
python -m pytest tests/unit/test_monte_carlo.py -v
```

All tests run without external services (Redis, OpenAI, network). Mocks used:
- `fakeredis` for Redis
- `MockLLMClient` for OpenAI
- In-memory ChromaDB client

## License

This software was developed as part of a master's thesis (ME2311) and is provided for academic evaluation purposes.
