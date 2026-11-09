# Artifact Reproducibility Guide
## ACM SIGMOD / PVLDB Artifact Evaluation

This guide is designed for **Artifact Evaluation Committee (AEC) reviewers** assessing the research paper:
> **"Conformal Query Processing: Calibrated Early Stopping for In-Database Analytics over Learned and Uncertain Data"**

---

## 1. Artifact Badges Claimed

| Badge | Status | Verification Path |
| :--- | :---: | :--- |
| **Artifacts Available** | **CLAIMED** | Public, self-contained repository with versioned code, base tables, and calibration models. |
| **Artifacts Evaluated - Functional** | **CLAIMED** | Clean build via `scripts/setup_env.sh` and 100% test pass via `scripts/run_smoke_test.sh` in < 1 second. |
| **Results Reproduced** | **CLAIMED** | Bitwise reproduction of Table 1 and all Figures 1–6 via `scripts/generate_paper_charts.sh` in < 10 seconds. |

---

## 2. System & Platform Requirements

- **Paper Evaluation Platform (Reference Specs)**:
  - AWS `g6e.xlarge` instance (4 vCPUs, 32 GiB RAM, Nvidia L40S GPU).
  - Ubuntu 24.04 LTS, Python 3.12.3, Rust 1.97.1 (`--release`).
- **Tested Evaluator Environments**:
  - Ubuntu 20.04 / 22.04 / 24.04 LTS (x86_64)
  - macOS Monterey / Ventura / Sonoma / Sequoia (Apple Silicon M1/M2/M3 and Intel x86_64)
  - Debian 11 / 12
- **Resource Footprint**:
  - **RAM**: 8 GB minimum (16 GB recommended for multi-threaded full sweeps).
  - **Disk Space**: ~1.2 GB total (including Rust release binaries and datasets). No GPU required for artifact evaluation.
  - **Network Access**: None required during evaluation; all base tables and reference data are bundled locally.

---

## 3. Step-by-Step Evaluator Instructions

### Step 1: Environment Setup (< 30 Seconds)
```bash
bash scripts/setup_env.sh
```
**Expected Output**:
```
Rust version: cargo 1.xx.x
Building Mode U Query Engine (Release profile)...
✓ Mode U Query Engine release binaries built successfully.
Python version: Python 3.xx.x
Checking required Python packages...
✓ All core Python packages verified (NumPy, SciPy, Pandas, Matplotlib, Seaborn).
✓ All required base data tables and calibration artifacts verified.
Setup complete!
```

---

### Step 2: Functional Smoke Verification (< 5 Seconds)
```bash
bash scripts/run_smoke_test.sh
```
**What It Verifies**:
1. All 6 native Rust engine binaries exist and execute cleanly.
2. Invariant unit tests (13/13) pass in < 0.25 seconds under `#![forbid(unsafe_code)]`.
3. Split-conformal calibration quantiles satisfy finite-sample bounds.
4. Base tables (COCO: 2,689 records, Jigsaw: 8,000 records) are intact and queryable.
5. Micro-join pipelines for TPC-H Q3 (2 joins) and Q4 (3 joins) execute end-to-end.

**Expected Exit Code**: `0` (Success in `< 1.0` second).

---

### Step 3: Fast Results Reproduction (< 10 Seconds)
To verify all paper tables and figures immediately from the audited canonical benchmark logs:
```bash
bash scripts/generate_paper_charts.sh
```
**Outputs Produced**:
1. **Table 1 (Coverage & Recall)**: `figures_v3/tables/table_1_coverage_validation.md`
2. **Figure 1 (Latency & Speedup)**: `figures_v3/paper/part_a/fig1_baseline_multi_query_bars.pdf`
3. **Figure 2 (Stream Savings)**: `figures_v3/paper/part_b/fig2a_topk_quad_bars_with_baseline.pdf`
4. **Figure 5 ($M=20$ U-Curve & Log-Gap)**: `figures_v3/paper/part_b/fig2g_q3_m20_savings_and_delta_w_log_grouped.pdf`
5. **Figure 6 ($k=1$ vs $k=3$ Ablations)**: `figures_v3/paper/part_e/fig5_strategy_ablation_dual_panel.pdf`

---

## 4. Paper Claims & Empirical Verification Matrix

Reviewers can verify each major claim in the manuscript against the generated artifacts:

### Claim 1: End-to-End Speedup (1.2× to 8.1× vs. Baseline; up to 26× vs. MCDB)
- **Paper Location**: § Evaluation Overview, Figure 1, Table 1.
- **Verification Artifact**: Open `figures_v3/paper/part_a/fig1_baseline_multi_query_bars.pdf` and `figures_v3/tables/table_1_coverage_validation.md`.
- **Expected Values**:
  - Q1a (Vision Flat, $k=1$): CQP latency $\approx 1.57\text{ ms}$ vs. Baseline $12.74\text{ ms}$ (**$8.1\times$ speedup**). MCDB is $37.3\text{ ms}$ ($23.8\times$ slower than CQP).
  - Q1b (Text Flat, $k=1$): CQP latency $\approx 1.71\text{ ms}$ vs. Baseline $11.61\text{ ms}$ (**$6.8\times$ speedup**).
  - Q2a (Vision Nested, $k=1$): CQP latency $\approx 2.42\text{ ms}$ vs. Baseline $14.57\text{ ms}$ (**$6.0\times$ speedup**).
  - Q3 (TPC-H 3-table, $k=1$): CQP latency $\approx 13.46\text{ ms}$ vs. Baseline $25.36\text{ ms}$ (**$1.88\times$ speedup**).
  - Q3 (TPC-H 3-table, $k=3$): CQP latency $\approx 21.59\text{ ms}$ vs. Baseline $25.33\text{ ms}$ (**$1.17\times$ speedup**).

---

### Claim 2: Stream Ingestion Savings (85%–92% on Flat/Nested; 10%–47% on Joins)
- **Paper Location**: § Workload Scalability, Figure 2.
- **Verification Artifact**: Open `figures_v3/paper/part_b/fig2a_topk_quad_bars_with_baseline.pdf`.
- **Expected Values**:
  - Q1a: **$92.5\%$** stream savings.
  - Q1b: **$87.1\%$** stream savings.
  - Q2a: **$90.2\%$** stream savings.
  - Q2b: **$87.0\%$** stream savings.
  - Q3 ($k=1$): **$47.1\%$** stream savings; Q3 ($k=3$): **$10.6\%$** stream savings.
  - Q4 ($k=1$): **$47.7\%$** stream savings; Q4 ($k=3$): **$12.5\%$** stream savings.

---

### Claim 3: Top-k Dynamics, The U-Curve & Log-Gap Linearization ($M=20$)
- **Paper Location**: § Top-k Scaling Dynamics, Figure 5.
- **Verification Artifact**: Open `figures_v3/paper/part_b/fig2g_q3_m20_savings_and_delta_w_log_grouped.pdf` (or `..._clean.pdf`).
- **Expected Observations**:
  1. **Panel (a) U-Curve**: High savings at $k=1\text{--}3$ ($16.6\%\text{--}17.5\%$), dipping in the mid-rank valley ($k=7\text{--}15$ at $0.7\%\text{--}2.5\%$), and recovering as $k \to 20$ ($4.9\%\text{--}6.4\%$).
  2. **Panel (b) Linearization**: Strong linear correlation between stream savings and $\log(W_k - W_{k+1})$ ($r = 0.966$ for $z=1.0$, $r = 0.973$ for $z=2.0$, $p < 10^{-3}$). Across all evaluated workloads, the bare difference $\Delta W_k$ averages $r = 0.63$ correlation with stream savings.

---

### Claim 4: Ablation Studies (Mutual Exclusion & In-Operator Stopping)
- **Paper Location**: § Ablation Studies, Figure 6.
- **Verification Artifact**: Open `figures_v3/paper/part_e/fig5_strategy_ablation_dual_panel.pdf` and `figures_v3/paper/part_e/fig5e_k_comparison_query_strip_subset_na_bar.pdf`.
- **Expected Observations**:
  - **Mutual Exclusion Reasoning**: Toggling mutual exclusion reasoning on nested vision queries (COCO Q2a) demonstrates that MutEx reasoning is **$55\%\text{--}62\%$ faster** than without.
  - **In-Operator Stopping vs External Pull**:
    - At $k=1$, external pull can achieve high savings on skewed distributions ($66\%\text{--}87\%$).
    - At $k \ge 2$ ($k=3$), external pull suffers **pruning collapse** ($0.0\%$ savings, running slower than baseline due to IPC and coordination overhead).
    - CQP with in-operator set creation is **$7\%\text{--}12\%$ faster** at $k=3$ and maintains invariant savings ($\sim 87\%$) across all $k$.

---

### Claim 5: Finite-Sample Statistical Coverage Compliance ($\ge 1 - \alpha$)
- **Paper Location**: § Conformal Set Validity, Table 1.
- **Verification Artifact**: Open `figures_v3/tables/table_1_coverage_validation.md`.
- **Expected Observations**:
  - Across all 1,161 parameter configurations and all $\alpha \in \{0.01, 0.05, 0.10\}$:
    $$\text{Empirical Coverage} \ge \text{Target Coverage} \quad (100.0\%\text{ compliance, 0 violations}).$$

---

## 5. Multi-Scale Experimental Sweeps

Evaluators can execute fresh benchmark sweeps across 5 scale presets or specify dynamic seed counts. Calibration seeds start at `1,000,000` and evaluation seeds start at `2,000,000` to guarantee disjoint PRNG sampling across arbitrarily large runs:

| Scale Tier | Calib Seeds ($N_{\text{cal}}$) | Eval Seeds ($N_{\text{eval}}$) | Error Resolution ($1 / N_{\text{eval}}$) | Approx Runtime | Recommended Target |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `scale_smoke` | 5 | 1 | N/A (Sanity check) | ~15 seconds | Deployment preflight / CI verification |
| `scale_normal` | 100 | 25 | 4.0% | ~15–25 minutes | Fast statistical check |
| `scale_long` | 500 | 100 | 1.0% | ~2–3 hours | High-resolution evaluation |
| `scale_extended` | 1,000 | 200 | 0.5% | ~5–7 hours | Overnight statistical audit |
| `scale_hyper` | 2,000 | 500 | 0.2% | ~12–14 hours | Exhaustive finite-sample verification |

### Running Preset Tiers
```bash
# Smoke test sweep (~15 seconds):
bash scripts/run_production_sweep.sh scale_smoke

# Overnight audit sweep (~5-7 hours):
bash scripts/run_production_sweep.sh scale_extended results_extended_sweep
```

### Dynamic Seed Overrides
Evaluators can also customize seed counts on any workload:
```bash
# E.g. 500 calibration runs and 100 evaluation runs on paper_focused workload:
bash scripts/run_production_sweep.sh paper_focused results_custom --cal-seeds 500 --eval-seeds 100
```

---

## 6. Remote Server Deployment & Overnight Runs

To deploy this repository onto a remote evaluation machine (e.g., AWS `g6e.xlarge` or Linux compute node) and execute an overnight sweep:

### Step 1: Package on Local Machine
Create a clean distribution archive excluding local build artifacts, virtual environments, and git metadata:
```bash
tar --exclude="engine/target" \
    --exclude=".venv" \
    --exclude="python/**/__pycache__" \
    --exclude=".git" \
    --exclude="results_*" \
    --exclude=".DS_Store" \
    -czf cqp_evaluator_package.tar.gz cqp-reproducibility-1/
```

### Step 2: Transfer to Remote Machine via `scp`
```bash
scp -i ~/.ssh/your_key.pem cqp_evaluator_package.tar.gz ubuntu@<server-ip>:~/
```

### Step 3: Unpack & Set Up on Remote Server
```bash
ssh -i ~/.ssh/your_key.pem ubuntu@<server-ip>

# Extract archive
tar -xzf cqp_evaluator_package.tar.gz
cd cqp-reproducibility-1

# Automated setup (compiles Rust release engine, verifies/creates .venv)
bash scripts/setup_env.sh

# Fast sanity verification (< 1 second)
bash scripts/run_smoke_test.sh
```

### Step 4: Launch Overnight Sweep in Detached Session
Use `nohup` or `tmux` so network disconnection does not interrupt the run:
```bash
# Option A: With nohup
nohup bash scripts/run_production_sweep.sh scale_extended results_overnight > sweep.log 2>&1 &
echo $! > sweep.pid

# Monitor progress:
tail -f sweep.log

# Option B: With tmux
tmux new -s cqp_sweep
bash scripts/run_production_sweep.sh scale_extended results_overnight
# (Press Ctrl+B then D to detach; reconnect later with: tmux attach -t cqp_sweep)
```

### Step 5: Render Figures from the Completed Sweep
```bash
bash scripts/generate_paper_charts.sh results_overnight figures_overnight
```

---

## 7. Troubleshooting & Support

- **Cargo build fails**: Ensure Rust $\ge 1.70$ is installed via `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`.
- **Python package import errors**: Run `bash scripts/setup_env.sh` to ensure packages are installed in the local `.venv`.
- **Headless server display error**: All plotting scripts configure `matplotlib.use('Agg')` automatically; no X11 or DISPLAY server is required.
