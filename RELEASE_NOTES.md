# CQP Reproducibility Package — Release Notes & Provenance

**Release**: `v1.0-camera-ready`  
**Date**: October 18, 2026  
**Conference**: ACM SIGMOD / PVLDB  
**Status**: VERIFIED ARTIFACT EVALUATION RELEASE  

---

## 1. Release Summary

This release contains the clean, self-contained, and verified artifact distribution for **Conformal Query Processing (CQP / Mode U)**.
All core engine binaries, benchmark drivers, synthetic data generators, and visualization pipelines have been compiled, tested, and synchronized against the canonical experimental findings in the paper.

---

## 2. Key Artifacts & Audit Highlights

### 2.1 Canonical Reference Dataset (26,400 Invocations, 297 KB)
- **Primary Canonical Suite**: 25,200 invocations processing 562.5M tuples across Vision (COCO), Text (Jigsaw), and Relational (TPC-H Q3 & Q4) workloads (`data/reference_results/canonical_v3/`).
- **Extended Q3 Top-k Suite**: Deep 12-point scaling sweep ($M=20, k \in \{1, 3, 5, 7, 9, 10, 11, 13, 15, 17, 19, 20\}$) exploring the U-curve and theoretical $\log \Delta W$ correlation (`data/reference_results/q3_extended_k_supplement/`).
- **Merged Canonical Dataset**: Unified 26,400-invocation dataset (`data/reference_results/merged_canonical_v3/`) analyzed across 1,032 configuration groups with zero statistical coverage violations.

### 2.2 Native Engine Compilation & Safety
- **Pure Safe Rust**: Enforces `#![forbid(unsafe_code)]` crate-wide.
- **Pruned Canonical Binaries**: Features strictly the 6 required binaries for Q1–Q4 evaluation and calibration.
- **Unit Test Parity**: 13/13 invariant unit tests pass in 0.20s (`cargo test`).
- **Smoke Tests**: 5/5 micro end-to-end queries pass in 0.57s (`bash scripts/run_smoke_test.sh`).

### 2.3 Publication Figures & Tables
- **Table 1**: Generated in `figures_v3/tables/table_1_coverage_validation.md`.
- **Figures 1–6**: Emitted across Part a, b, c, d, e in both `paper` (compact, serif typography) and `presentation` profiles in `figures_v3/`.
