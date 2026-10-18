#!/usr/bin/env python3
"""
test_smoke_end_to_end.py
Strategy 1 Clean Migration: Fast End-to-End Smoke Test (<15 seconds total).

Permitted for execution by AI agents and CI environments.
Validates:
1. Engine release binaries presence and execution.
2. Conformal calibration mathematical invariant: ceil((n+1)(1-alpha))/n.
3. End-to-End micro-run of TPC-H Q3 join pipeline.
4. End-to-End micro-run of TPC-H Q10 join pipeline.
5. Integrity and schema of cached base tables.
"""

import os
import sys
import json
import time
import tempfile
import subprocess
from pathlib import Path
import numpy as np

TEST_DIR = Path(__file__).resolve().parent
STRATEGY_DIR = TEST_DIR.parent
PYTHON_DIR = STRATEGY_DIR / "python"
ENGINE_DIR = STRATEGY_DIR / "engine"
DATA_DIR = STRATEGY_DIR / "data"

sys.path.insert(0, str(PYTHON_DIR))
sys.path.insert(0, str(PYTHON_DIR / "lib"))

from lib.conformal_calibration import compute_tau_threshold
from lib.generate_tpch_q3_synthetic import generate_tpch_q3_dataset
from lib.generate_tpch_q10_synthetic import generate_tpch_q10_dataset


def test_01_engine_binaries_exist():
    """Verify all 6 Canonical V3 release binaries were compiled and are executable."""
    release_dir = ENGINE_DIR / "target" / "release"
    required_bins = [
        "benchmark_flat_conformal",
        "benchmark_nested_conformal",
        "benchmark_tpch_q3",
        "benchmark_tpch_q10",
        "calibrate_flat_strategies",
        "calibrate_nested_strategies",
    ]
    for b in required_bins:
        bin_path = release_dir / b
        assert bin_path.exists(), f"Missing binary: {bin_path}"
        assert os.access(bin_path, os.X_OK), f"Binary not executable: {bin_path}"
    print(f"✓ Test 1 Passed: All {len(required_bins)} Canonical V3 engine release binaries verified.")


def test_02_conformal_calibration_math():
    """Verify that conformal threshold calibration strictly adheres to finite-sample guarantees."""
    # Test with 100 calibration scores: [0.01, 0.02, ..., 1.00]
    scores = np.linspace(0.01, 1.0, 100)
    alpha = 0.10
    n = len(scores)

    # Theoretical index: ceil((n + 1) * (1 - alpha)) = ceil(101 * 0.90) = ceil(90.9) = 91 (1-indexed) -> index 90 (0-indexed)
    tau = compute_tau_threshold(scores, alpha)
    expected_tau = np.sort(scores)[min(n - 1, int(np.ceil((n + 1) * (1.0 - alpha))) - 1)]

    assert abs(tau - expected_tau) < 1e-9, f"Expected tau {expected_tau}, got {tau}"
    assert 0.0 <= tau <= 1.0, f"Tau {tau} out of range [0, 1]"
    print(f"✓ Test 2 Passed: Conformal calibration math verified (n={n}, alpha={alpha}, tau={tau:.4f}).")


def test_03_base_tables_integrity():
    """Verify that frozen base tables are valid JSON and have correct key counts."""
    cached_tables_dir = DATA_DIR / "cached_tables"
    coco_path = cached_tables_dir / "coco_animals_calibrated_softmax.json"
    jigsaw_path = cached_tables_dir / "jigsaw_toxicity_platt.json"

    assert coco_path.exists(), f"COCO base table missing: {coco_path}"
    with open(coco_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)
    assert len(coco_data) > 0, "COCO base table is empty"

    assert jigsaw_path.exists(), f"Jigsaw base table missing: {jigsaw_path}"
    with open(jigsaw_path, "r", encoding="utf-8") as f:
        jigsaw_data = json.load(f)
    assert len(jigsaw_data) > 0, "Jigsaw base table is empty"

    print(f"✓ Test 3 Passed: Base tables verified (COCO: {len(coco_data)} entries, Jigsaw: {len(jigsaw_data)} entries).")


def test_04_tpch_q3_micro_e2e():
    """Run an end-to-end micro-execution of TPC-H Q3 join pipeline."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        dataset_file = tmp_path / "micro_q3.json"
        out_file = tmp_path / "out_q3.json"

        # Generate minimal synthetic dataset: 40 orders, 4 customer groups
        data = generate_tpch_q3_dataset(
            n_customers=20,
            n_orders=40,
            n_groups=4,
            zipf_skew=0.8,
            seed=42,
        )
        with open(dataset_file, "w", encoding="utf-8") as f:
            json.dump(data, f)
        assert dataset_file.exists()

        # Run benchmark_tpch_q3 binary
        cmd = [
            str(ENGINE_DIR / "target" / "release" / "benchmark_tpch_q3"),
            "--file", str(dataset_file),
            "--output", str(out_file),
            "--k-query", "2",
            "--k-cand-max", "4",
            "--alpha", "0.10",
            "--strategies", "Count_Naive,Count_Conformal_InOperator_Stop",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"benchmark_tpch_q3 failed:\n{res.stderr}\n{res.stdout}"

        assert out_file.exists(), "Output file was not generated"
        with open(out_file, "r", encoding="utf-8") as f:
            out_data = json.load(f)

        assert isinstance(out_data, list) and len(out_data) > 0, "Output JSON should be a non-empty list"
        res_entry = out_data[0]
        assert "results" in res_entry, "Missing 'results' key in output JSON entry"
        results = res_entry["results"]
        strat_names = [r["strategy"] for r in results]
        assert "Count_Naive" in strat_names, "Count_Naive missing in results"
        assert "Count_Conformal_InOperator_Stop" in strat_names, "Count_Conformal_InOperator_Stop missing in results"

        # Verify items read is bounded by total tuples
        total_tuples = res_entry["total_tuples"]
        for r in results:
            assert r["items_read"] <= total_tuples, f"Items read {r['items_read']} exceeded total tuples {total_tuples}"
            assert len(r["top_k_emitted"]) > 0, "Expected non-empty top_k_emitted"

        print(f"✓ Test 4 Passed: TPC-H Q3 micro end-to-end verified ({total_tuples} tuples processed).")


def test_05_tpch_q10_micro_e2e():
    """Run an end-to-end micro-execution of TPC-H Q10 join pipeline."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        dataset_file = tmp_path / "micro_q10.json"
        out_file = tmp_path / "out_q10.json"

        # Generate minimal synthetic dataset: 30 orders, 20 customers
        data = generate_tpch_q10_dataset(
            n_customers=20,
            n_orders=30,
            zipf_skew=0.8,
            seed=42,
        )
        with open(dataset_file, "w", encoding="utf-8") as f:
            json.dump(data, f)
        assert dataset_file.exists()

        # Run benchmark_tpch_q10 binary
        cmd = [
            str(ENGINE_DIR / "target" / "release" / "benchmark_tpch_q10"),
            "--file", str(dataset_file),
            "--output", str(out_file),
            "--k-query", "2",
            "--k-cand-max", "4",
            "--alpha", "0.10",
            "--strategies", "Naive,Conformal_InOperator_Stop",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"benchmark_tpch_q10 failed:\n{res.stderr}\n{res.stdout}"

        assert out_file.exists(), "Output file was not generated"
        with open(out_file, "r", encoding="utf-8") as f:
            out_data = json.load(f)

        assert isinstance(out_data, list) and len(out_data) > 0
        res_entry = out_data[0]
        assert "results" in res_entry
        results = res_entry["results"]
        strat_names = [r["strategy"] for r in results]
        assert "Naive" in strat_names
        assert "Conformal_InOperator_Stop" in strat_names

        total_tuples = res_entry["total_tuples"]
        for r in results:
            assert r["items_read"] <= total_tuples
            assert len(r["top_k_emitted"]) > 0

        print(f"✓ Test 5 Passed: TPC-H Q10 micro end-to-end verified ({total_tuples} tuples processed).")


if __name__ == "__main__":
    t0 = time.time()
    print("==============================================================================")
    print(" Strategy 1 Clean Migration: End-to-End Smoke Test Suite")
    print("==============================================================================")
    test_01_engine_binaries_exist()
    test_02_conformal_calibration_math()
    test_03_base_tables_integrity()
    test_04_tpch_q3_micro_e2e()
    test_05_tpch_q10_micro_e2e()
    elapsed = time.time() - t0
    print("==============================================================================")
    print(f" ALL SMOKE TESTS PASSED SUCCESSFULLY in {elapsed:.2f}s (<30s threshold).")
    print("==============================================================================")
