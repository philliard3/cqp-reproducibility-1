# Conformal Query Processing (CQP / Mode U)
## Official Reproducibility & Artifact Evaluation Package
### ACM SIGMOD / PVLDB Research & Reproducibility Track

[![Rust Engine](https://img.shields.io/badge/Rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Safe Rust](https://img.shields.io/badge/Rust-%23!%5Bforbid(unsafe__code)%5D-brightgreen.svg)]()
[![Artifact Evaluation](https://img.shields.io/badge/Artifact%20Evaluation-Passed-success.svg)]()

This repository provides the official, complete, and reproducible implementation of **Conformal Query Processing (CQP / Mode U)**—a query execution framework that integrates finite-sample split-conformal prediction into relational and aggregation operators for in-database analytics over learned and uncertain data.

---

## 1. Executive Summary & Key Results

Conformal Query Processing (CQP) pushes distribution-free conformal calibration directly into the physical tuple iterator (`next()` / `pull()`) of a Volcano query engine. By evaluating dynamic interval bounds ($\hat{L}_g, \hat{U}_g$) during stream consumption, CQP stops stream ingestion as early as the top-$k$ set is statistically guaranteed to contain the true winners with probability $\ge 1 - \alpha$.

### Summary of Empirical Findings (Canonical V3 Suite: 25,200 Invocations, 562.5M Tuples):
- **Speedup vs. Sequential Baseline**: **$1.2\times$ to $8.1\times$** faster end-to-end execution across vision, text, and relational join workloads.
- **Speedup vs. Monte Carlo Baselines (MCDB-1000)**: **$5.4\times$ to $26\times$** faster than optimized 1,000-world vectorized Monte Carlo database simulation.
- **Stream Ingestion Savings**: **$85\%$ to $92.5\%$** reduction in stream reads for flat and nested monitoring queries; **$10.6\%$ to $47.1\%$** reduction for continuous revenue multi-table joins.
- **Statistical Coverage Compliance**: **$100.0\%$ empirical validity** ($\ge 1 - \alpha$) across all 1,032 evaluated parameter configurations ($\alpha \in \{0.01, 0.05, 0.10\}$) with **zero violations**.
- **Ablation Insight**: At $k \ge 2$, external pull coordinators experience **pruning collapse** ($0.0\%$ savings, running slower than baseline), whereas CQP's in-operator pushdown sustains $\sim 87\%$ savings regardless of $k$.

---

## 2. Supported Workloads & Query Portfolio

The benchmark suite evaluates six representative query archetypes across three distinct modalities:

| Query | Modality | Base Dataset | Base Model & Calibration | Groups $M$ | Semantics | Relational Joins |
| :--- | :--- | :--- | :--- | :---: | :--- | :--- |
| **Q1a** | Vision | MS COCO 2017 Val (Lin et al.) | YOLOv8n + Platt Scaling | 10 | Closed-World MutEx ($\sum p_c \le 1$) | Flat (0 Joins) |
| **Q1b** | Text | Jigsaw Toxicity (Borkan et al.) | Twitter-RoBERTa + Platt Scaling | 6 | Open-World Independent Bernoulli | Flat (0 Joins) |
| **Q2a** | Vision | MS COCO 2017 Val | YOLOv8n + Platt Scaling | 10 | 2-Stage Nested Day $\to$ Global Winner | Windowed (0 Joins) |
| **Q2b** | Text | Jigsaw Toxicity | Twitter-RoBERTa + Platt Scaling | 6 | 2-Stage Nested Day $\to$ Global Winner | Windowed (0 Joins) |
| **Q3**  | Relational | TPC-H Schema (Rev 3.0) | $p \sim \text{Beta}(0.5, 0.5)$ | 10 | Top-$k$ Continuous Revenue Sum | 2 Joins: $(\text{Cust} \bowtie \text{Orders}) \bowtie \text{Line}$ |
| **Q4**  | Relational | TPC-H Schema | $p \sim \text{Beta}(0.5, 0.5)$ | 10 | Top-$k$ Lost Revenue Sum | 3 Joins: $(\text{Cust} \bowtie \text{Orders} \bowtie \text{Line}) \bowtie \text{Nation}$ |

---

## 3. System Requirements & Evaluation Platform

- **Paper Evaluation Platform**:
  - AWS `g6e.xlarge` instance (4 vCPUs, 32 GiB RAM, Nvidia L40S GPU).
  - Ubuntu 24.04 LTS, Python 3.12.3, Rust 1.97.1 (`--release`).
- **Minimum Evaluator Requirements**:
  - **Operating System**: Linux (Ubuntu 20.04+, Debian 11+, RHEL 8+) or macOS (12.0+ on Intel or Apple Silicon).
  - **Hardware**: Standard x86_64 or ARM64 (aarch64) machine. Minimum 8 GB RAM, 2 GB free disk space. No GPU required for query evaluation or chart rendering.
- **Software Dependencies**:
  - **Rust**: Version $\ge 1.70$ (with `cargo`).
  - **Python**: Version $\ge 3.10$ (standard `python3`).
  - **Python Libraries**: `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn` (automatically installed into a local `.venv` by setup script).

---

## 4. Quickstart: 3-Step Verification (< 1 Minute)

### Step 1: Environment Setup (< 30 Seconds)
```bash
bash scripts/setup_env.sh
```
This script:
1. Compiles the native Rust Volcano engine in release mode with `#![forbid(unsafe_code)]`.
2. Automatically creates and populates a localized `.venv` environment if packages are missing.
3. Cryptographically checks base data tables (COCO and Jigsaw) and calibration models.

### Step 2: Automated Smoke Test (< 5 Seconds)
```bash
bash scripts/run_smoke_test.sh
```
Executes 13 Rust mathematical invariant unit tests and 5 Python end-to-end micro-queries. Output:
```
==============================================================================
 ALL SMOKE TESTS PASSED SUCCESSFULLY in 0.57s (<30s threshold).
==============================================================================
```

### Step 3: Reproduce Paper Figures & Table 1 (< 10 Seconds)
```bash
bash scripts/generate_paper_charts.sh
```
Instantly re-hydrates the bundled audited canonical dataset (26,400 runs) and renders all camera-ready vector PDF figures and Markdown/LaTeX tables:
- **Table 1**: `figures_v3/tables/table_1_coverage_validation.md`
- **Part A (Speedup)**: `figures_v3/paper/part_a/`
- **Part B (Top-k Scaling & U-Curve)**: `figures_v3/paper/part_b/`
- **Part C (Calibration Dynamics)**: `figures_v3/paper/part_c/`
- **Part D (Scale Comparison)**: `figures_v3/paper/part_d/`
- **Part E (Ablations & Pruning Collapse)**: `figures_v3/paper/part_e/`

---

## 5. Live Benchmark Evaluation (Optional)

To verify that the Volcano engine and conformal operators execute live queries and stream data on your system:

### Quick End-to-End Dry Run (< 1 Minute)
```bash
bash scripts/run_production_sweep.sh dry_run [output_dir]
```
Executes a rapid multi-query sweep across 2 seeds and verifies the complete query engine and telemetry pipeline.

### Paper-Focused Representative Sweep (~15–20 Minutes)
```bash
bash scripts/run_production_sweep.sh paper_focused [output_dir]
```
Executes representative evaluations for all 6 queries across key paper parameters.

*(Note: The full 25,200-invocation multi-hour canonical grid sweep is pre-compiled and bundled under `data/reference_results/canonical_v3/` for instant reproduction of all paper figures in Step 3).*

---

## 6. Paper Claims-to-Artifacts Mapping

| Paper Section & Claim | Paper Artifact | Reproducing Script | Generated Output Artifact |
| :--- | :--- | :--- | :--- |
| **§ Evaluation Overview**: End-to-End Speedup (1.2× to 8.1× vs Baseline; up to 26× vs MCDB) | Figure 1 | `bash scripts/generate_paper_charts.sh` | `figures_v3/paper/part_a/fig1_baseline_multi_query_bars.pdf` |
| **§ Workload Scalability**: Stream Savings across Modalities (85%–92% flat/nested, 10%–47% joins) | Figure 2 | `bash scripts/generate_paper_charts.sh` | `figures_v3/paper/part_b/fig2a_topk_quad_bars_with_baseline.pdf` |
| **§ Top-k Scaling Dynamics**: The $M=20$ U-Curve and Log-Weight Correlation ($r=0.868, 0.936$) | Figure 5 | `bash scripts/generate_paper_charts.sh` | `figures_v3/paper/part_b/fig2g_q3_m20_savings_and_delta_w_log_grouped.pdf` |
| **§ Ablation Studies**: External Pull Pruning Collapse at $k \ge 2$ ($0\%$ savings vs $87\%$ In-Operator) | Figure 6 | `bash scripts/generate_paper_charts.sh` | `figures_v3/paper/part_e/fig5_strategy_ablation_dual_panel.pdf` |
| **§ Conformal Validity**: Finite-Sample Coverage Compliance ($\ge 1-\alpha$ across 1,032 configs) | Table 1 | `bash scripts/generate_paper_charts.sh` | `figures_v3/tables/table_1_coverage_validation.md` |

For an exhaustive audit and step-by-step evaluation guide, see [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

---

## 7. Package Directory Structure

```
cqp-reproducibility/
├── README.md                    <- Primary documentation and overview (this file)
├── REPRODUCIBILITY.md           <- Formal ACM / PVLDB Artifact Evaluation guidebook
├── RELEASE_NOTES.md             <- Release provenance and audited metrics
├── LICENSE                      <- Apache 2.0 Open Source License
├── data/
│   ├── cached_tables/           <- Real-world base tables (MS COCO 2017, Jigsaw)
│   ├── calibration_artifacts/   <- Platt scaling model artifacts
│   └── reference_results/       <- Audited canonical sweep data (26,400 runs, 297 KB)
│       └── canonical_v3/        <- Unified benchmark dataset (26,400 runs)
├── engine/                      <- High-performance Volcano query engine in Safe Rust
│   ├── Cargo.toml               <- Crate definition (#![forbid(unsafe_code)])
│   └── src/
│       ├── bin/                 <- 6 Canonical query binaries (Q1a, Q1b, Q2a, Q2b, Q3, Q4)
│       └── operators/           <- Volcano streaming operators, joins, conformal state machines
├── python/                      <- Python calibration, sweep runners, and visualization
│   ├── lib/                     <- Conformal math, synthetic generators, schema codec
│   ├── presets/                 <- Preset sweep configurations (dry_run, paper_focused)
│   ├── runners/                 <- Workload runners (COCO, Jigsaw, TPC-H)
│   ├── generate_paper_charts_v3.py <- Publication figure renderer
│   └── run_production_sweep.py  <- Benchmark master orchestrator
├── scripts/                     <- Shell entry points for evaluators
│   ├── setup_env.sh             <- Compiles Rust release engine & verifies Python env
│   ├── run_smoke_test.sh        <- Fast 5s end-to-end regression validation
│   ├── generate_paper_charts.sh <- Renders Table 1 and Figures from reference data
│   └── run_production_sweep.sh  <- Benchmark sweep runner (dry_run, paper_focused)
└── tests/
    └── test_smoke_end_to_end.py <- Automated smoke test suite (<1s runtime)
```

---

## 8. License & Citation

This project is licensed under the Apache License, Version 2.0. See [`LICENSE`](LICENSE) for details.

If you use this software or empirical data in your research, please cite:
```bibtex
@article{cqp2026,
  title   = {Conformal Query Processing: Calibrated Early Stopping for In-Database Analytics over Learned and Uncertain Data},
  author  = {Anonymous Authors},
  journal = {Proceedings of the VLDB Endowment (PVLDB)},
  year    = {2026}
}
```
