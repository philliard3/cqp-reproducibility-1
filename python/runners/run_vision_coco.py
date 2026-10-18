#!/usr/bin/env python3
"""
run_vision_coco.py

Modular, standalone runner for Vision (MS-COCO Animals) workloads:
- Q1a: Flat ArgMax-over-Count across all images.
- Q2a: Two-Stage Nested ArgMax-over-Count (winner per partition -> outer partition plurality).

Executes all 6 applicable strategies:
1. Naive_Conformal_End (Full scan baseline)
2. MCDB_Count_1000 (Monte Carlo 1000-world simulation)
3. Conformal_InOperator_Stop (MutEx) (Closed-world early stopping)
4. Conformal_InOperator_Stop (Indep) (Open-world early stopping)
5. QN1_External_Pull (MutEx) (Volcano pull with MutEx bounds)
6. QN1_External_Pull (Indep) (Volcano pull with Indep bounds)

Outputs compact .jsonl.gz logs encoded with stable integer IDs via SchemaCodec.
"""

import os
import sys
import re
import json
import time
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

RUNNER_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = RUNNER_DIR.parent
REPO_ROOT = PACKAGE_DIR.parent
sys.path.insert(0, str(PACKAGE_DIR))
sys.path.insert(0, str(REPO_ROOT))

from lib.schema_codec import SchemaCodec, write_jsonl_gz
from lib.profiler import SweepProfiler
from lib.conformal_calibration import process_calibration_snapshots


def run_command(cmd: List[str], desc: str) -> float:
    t0 = time.time()
    print(f"--> [{time.strftime('%H:%M:%S')}] Starting {desc}...", flush=True)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if proc.stdout:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"{desc} failed with exit code {proc.returncode}")
    dur = time.time() - t0
    print(f"<-- [{time.strftime('%H:%M:%S')}] Completed {desc} in {dur:.2f}s", flush=True)
    return dur


def compute_conformal_threshold_matrix(
    calib_raw_path: Path,
    out_matrix_path: Path,
    alphas: List[float],
) -> Dict[str, Any]:
    """
    Computes quantile threshold matrix from raw calibration snapshots.
    For each (strategy, k, alpha): tau = quantile(1 - alpha) of nonconformity scores.
    """
    with open(calib_raw_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    # Gather scores per config and strategy
    by_strat_k: Dict[str, Dict[str, List[float]]] = {}
    for rec in records:
        for strat in rec.get("strategy_results", []):
            s_name = strat["strategy_name"]
            by_strat_k.setdefault(s_name, {})
            # extract bounds
            for r_name, r_bounds in strat.get("rank_snapshots", {}).items():
                lo = r_bounds.get("lo", 0.0)
                by_strat_k[s_name].setdefault(r_name, []).append(lo)

    matrix: Dict[str, Any] = {"strategies": {}}
    for s_name, k_dict in by_strat_k.items():
        matrix["strategies"][s_name] = {}
        for r_name, scores in k_dict.items():
            matrix["strategies"][s_name][r_name] = {}
            scores_arr = np.array(scores) if 'np' in globals() else sorted(scores)
            n = len(scores)
            for a in alphas:
                q_idx = int(round((1.0 - a) * (n + 1))) - 1
                q_idx = max(0, min(n - 1, q_idx))
                tau = sorted(scores)[q_idx] if scores else 0.0
                a_key = f"alpha_{int(round(a*100)):02d}"
                matrix["strategies"][s_name][r_name][a_key] = {
                    "tau": round(float(tau), 5),
                    "alpha": a,
                    "is_valid_guarantee": n >= int(1.0 / a) - 1,
                }

    out_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_matrix_path, "w", encoding="utf-8") as f:
        json.dump(matrix, f, indent=2)
    return matrix


def run_vision_coco_sweep(
    dataset_dir: Path,
    output_dir: Path,
    alphas: List[float],
    engine_bin_dir: Optional[Path] = None,
    codec: Optional[SchemaCodec] = None,
    profiler: Optional[SweepProfiler] = None,
    cal_seeds: Optional[List[int]] = None,
    eval_seeds: Optional[List[int]] = None,
    is_nested_only: bool = False,
    is_flat_only: bool = False,
    dataset_scale: float = 1.0,
    configs_list: Optional[List[List[Any]]] = None,
) -> Path:
    """
    Executes Vision COCO benchmarks (Q1a flat and Q2a nested) using on-demand ephemeral
    calibration and evaluation datasets to prevent disk bloat.
    """
    import shutil
    from prepare_production_datasets import get_cached_or_generate_dataset

    if engine_bin_dir is None:
        engine_bin_dir = REPO_ROOT / "mode_u_engine" / "target" / "release"
    if codec is None:
        codec = SchemaCodec()
    if profiler is None:
        profiler = SweepProfiler("vision_coco_sweep")
    if cal_seeds is None:
        cal_seeds = list(range(1000, 1025))
    if eval_seeds is None:
        eval_seeds = list(range(2000, 2005))
    if configs_list is None:
        configs_list = [[10, 1.0, 1]]

    output_dir.mkdir(parents=True, exist_ok=True)
    all_eval_records: List[Dict[str, Any]] = []

    # Binaries
    flat_cal_bin = engine_bin_dir / "calibrate_flat_strategies"
    flat_eval_bin = engine_bin_dir / "benchmark_flat_conformal"
    nested_cal_bin = engine_bin_dir / "calibrate_nested_strategies"
    nested_eval_bin = engine_bin_dir / "benchmark_nested_conformal"

    # Step 1: Ephemeral Calibration Phase
    cal_dir = dataset_dir / "ephemeral_calib"
    cal_dir.mkdir(parents=True, exist_ok=True)
    total_cal_datasets = len(configs_list) * len(cal_seeds)
    step_cal_5pct = max(1, total_cal_datasets // 20)
    cal_count = 0
    print(f"\n=======================================================", flush=True)
    print(f"[COCO Phase 1/2] Generating {total_cal_datasets} Ephemeral Calibration Datasets ({len(configs_list)} configs, scale={dataset_scale})", flush=True)
    print(f"=======================================================", flush=True)
    for cfg in configs_list:
        M, z, c = int(cfg[0]), float(cfg[1]), int(cfg[2])
        for s in cal_seeds:
            get_cached_or_generate_dataset(
                query="q1a",
                domain="coco",
                M=M,
                z=z,
                c=c,
                seed=s,
                scale=dataset_scale,
                custom_out_dir=cal_dir,
            )
            cal_count += 1
            if cal_count % step_cal_5pct == 0 or cal_count == total_cal_datasets:
                pct = (cal_count / total_cal_datasets) * 100.0
                print(f"  [Calib Gen Progress: {pct:5.1f}% | {cal_count:>4}/{total_cal_datasets}] COCO (M={M}, z={z:.1f}, c={c})", flush=True)

    cal_raw_q1 = output_dir / "calib_raw_q1a.json"
    cal_matrix_q1 = output_dir / "calib_matrix_q1a.json"
    cal_raw_q2 = output_dir / "calib_raw_q2a.json"
    cal_matrix_q2 = output_dir / "calib_matrix_q2a.json"
    cal_log_step = max(1, total_cal_datasets // 20)

    if not is_nested_only and flat_cal_bin.exists():
        t_cal1 = profiler.start_task("q1a_calibration")
        cmd_cal1 = [
            str(flat_cal_bin),
            "--dir", str(cal_dir),
            "--output", str(cal_raw_q1),
            "--log-interval", str(cal_log_step),
        ]
        run_command(cmd_cal1, "Q1a Flat Calibration")
        process_calibration_snapshots(cal_raw_q1, cal_matrix_q1, target_alphas=alphas)
        profiler.record_task_time("q1a_calibration", t_cal1)

    if not is_flat_only and nested_cal_bin.exists():
        t_cal2 = profiler.start_task("q2a_calibration")
        cmd_cal2 = [
            str(nested_cal_bin),
            "--dir", str(cal_dir),
            "--output", str(cal_raw_q2),
            "--log-interval", str(cal_log_step),
        ]
        run_command(cmd_cal2, "Q2a Nested Calibration")
        process_calibration_snapshots(cal_raw_q2, cal_matrix_q2, target_alphas=alphas)
        profiler.record_task_time("q2a_calibration", t_cal2)

    # Wipe calibration datasets immediately to release disk space
    shutil.rmtree(cal_dir, ignore_errors=True)

    # Step 2: Ephemeral Evaluation Phase
    eval_dir = dataset_dir / "ephemeral_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    total_eval_datasets = len(configs_list) * len(eval_seeds)
    step_eval_5pct = max(1, total_eval_datasets // 20)
    eval_count = 0
    print(f"\n=======================================================", flush=True)
    print(f"[COCO Phase 2/2] Generating {total_eval_datasets} Ephemeral Evaluation Datasets ({len(configs_list)} configs, scale={dataset_scale})", flush=True)
    print(f"=======================================================", flush=True)
    for cfg in configs_list:
        M, z, c = int(cfg[0]), float(cfg[1]), int(cfg[2])
        for s in eval_seeds:
            get_cached_or_generate_dataset(
                query="q1a",
                domain="coco",
                M=M,
                z=z,
                c=c,
                seed=s,
                scale=dataset_scale,
                custom_out_dir=eval_dir,
            )
            eval_count += 1
            if eval_count % step_eval_5pct == 0 or eval_count == total_eval_datasets:
                pct = (eval_count / total_eval_datasets) * 100.0
                print(f"  [Eval Gen Progress: {pct:5.1f}% | {eval_count:>4}/{total_eval_datasets}] COCO (M={M}, z={z:.1f}, c={c})", flush=True)

    eval_log_step = max(1, len(eval_seeds) // 20)

    # Step 2.1: Run Flat Q1a Evaluation
    if not is_nested_only and flat_eval_bin.exists():
        q1_out = output_dir / "raw_eval_q1a.json"
        t_eval = profiler.start_task("q1a_evaluation")
        cmd_eval = [
            str(flat_eval_bin),
            "--dir", str(eval_dir),
            "--conformal-thresholds", str(cal_matrix_q1),
            "--output", str(q1_out),
            "--log-interval", str(eval_log_step),
        ]
        run_command(cmd_eval, "Q1a Flat Evaluation")
        profiler.record_task_time("q1a_evaluation", t_eval)

        if q1_out.exists():
            with open(q1_out, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            for d_res in raw_data:
                seed = d_res.get("seed", 0)
                gt_win = d_res.get("ground_truth_winner", "")
                df_name = d_res.get("dataset_file", "")
                m_match = re.search(r"M(\d+)_z(\d+)_c(\d+)", df_name)
                if m_match:
                    M_val = int(m_match.group(1))
                    z_raw = m_match.group(2)
                    z_val = float(z_raw) / 100.0 if len(z_raw) == 3 else float(z_raw) / 10.0
                    c_val = int(m_match.group(3))
                else:
                    M_val, z_val, c_val = 10, 1.0, 1
                for ep in d_res.get("eval_points", []):
                    rec = {
                        "config": {
                            "query": "q1a",
                            "strategy": ep.get("strategy"),
                            "M": M_val,
                            "z": z_val,
                            "c": c_val,
                            "k_query": ep.get("k", 1),
                            "alpha": ep.get("alpha", 0.05),
                            "seed": seed,
                            "dataset_scale": dataset_scale,
                        },
                        "metrics": {
                            "tuples_ingested": ep.get("items_read", 0),
                            "tuples_total": ep.get("total_stream_items", 0),
                            "stream_savings_pct": ep.get("stream_savings_ratio", 0.0) * 100.0,
                            "engine_time_ms": ep.get("in_engine_ms", 0.0),
                            "e2e_time_ms": ep.get("in_engine_ms", 0.0),
                            "emitted_set": ep.get("prediction_set", []),
                            "emitted_set_size": ep.get("set_size", 0),
                            "ground_truth_top_k": [gt_win],
                            "covered": ep.get("is_covered", False),
                            "recall_at_set": 1.0 if ep.get("is_covered") else 0.0,
                            "tau_threshold": ep.get("tau", 0.0),
                            "relative_set_size": ep.get("set_size", 1) / max(1, ep.get("k", 1)),
                        }
                    }
                    all_eval_records.append(rec)
                    profiler.record_run_point("q1a", ep.get("strategy"), ep.get("in_engine_ms", 0.0), ep.get("items_read", 0))

    # Step 2.2: Run Nested Q2a Evaluation
    if not is_flat_only and nested_eval_bin.exists():
        q2_out = output_dir / "raw_eval_q2a.json"
        t_eval = profiler.start_task("q2a_evaluation")
        cmd_eval = [
            str(nested_eval_bin),
            "--dir", str(eval_dir),
            "--conformal-thresholds", str(cal_matrix_q2),
            "--output", str(q2_out),
            "--log-interval", str(eval_log_step),
        ]
        run_command(cmd_eval, "Q2a Nested Evaluation")
        profiler.record_task_time("q2a_evaluation", t_eval)

        if q2_out.exists():
            with open(q2_out, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            for d_res in raw_data:
                seed = d_res.get("seed", 0)
                gt_win = d_res.get("ground_truth_winner", "")
                df_name = d_res.get("dataset_file", "")
                m_match = re.search(r"M(\d+)_z(\d+)_c(\d+)", df_name)
                if m_match:
                    M_val = int(m_match.group(1))
                    z_raw = m_match.group(2)
                    z_val = float(z_raw) / 100.0 if len(z_raw) == 3 else float(z_raw) / 10.0
                    c_val = int(m_match.group(3))
                else:
                    M_val, z_val, c_val = 10, 1.0, 1
                for ep in d_res.get("eval_points", []):
                    rec = {
                        "config": {
                            "query": "q2a",
                            "strategy": ep.get("strategy"),
                            "M": M_val,
                            "z": z_val,
                            "c": c_val,
                            "k_query": ep.get("k", 1),
                            "alpha": ep.get("alpha", 0.05),
                            "seed": seed,
                            "dataset_scale": dataset_scale,
                        },
                        "metrics": {
                            "tuples_ingested": ep.get("items_read", 0),
                            "tuples_total": ep.get("total_stream_items", 0),
                            "stream_savings_pct": ep.get("stream_savings_ratio", 0.0) * 100.0,
                            "engine_time_ms": ep.get("in_engine_ms", 0.0),
                            "e2e_time_ms": ep.get("in_engine_ms", 0.0),
                            "emitted_set": ep.get("prediction_set", []),
                            "emitted_set_size": ep.get("set_size", 0),
                            "ground_truth_top_k": [gt_win],
                            "covered": ep.get("is_covered", False),
                            "recall_at_set": 1.0 if ep.get("is_covered") else 0.0,
                            "tau_threshold": ep.get("tau", 0.0),
                            "relative_set_size": ep.get("set_size", 1) / max(1, ep.get("k", 1)),
                        }
                    }
                    all_eval_records.append(rec)
                    profiler.record_run_point("q2a", ep.get("strategy"), ep.get("in_engine_ms", 0.0), ep.get("items_read", 0))

    # Wipe evaluation datasets immediately to guarantee zero disk residue
    shutil.rmtree(eval_dir, ignore_errors=True)

    # Save compact encoded results
    final_log_path = output_dir / "vision_coco_eval.jsonl.gz"
    write_jsonl_gz(all_eval_records, final_log_path, codec)
    codec.save_lookup(output_dir / "schema_lookup.json")
    print(f"✓ Vision COCO sweep complete: {len(all_eval_records)} evaluation points -> {final_log_path}")
    return final_log_path


def main():
    parser = argparse.ArgumentParser(description="Run Vision COCO (Q1a/Q2a) benchmark sweep.")
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/vision_coco")
    parser.add_argument("--engine-bin-dir", type=str, default=str(REPO_ROOT / "mode_u_engine" / "target" / "release"))
    parser.add_argument("--alphas", type=str, default="0.01,0.05,0.10")
    args = parser.parse_args()

    alphas = [float(a.strip()) for a in args.alphas.split(",")]
    codec = SchemaCodec()
    profiler = SweepProfiler("vision_coco_sweep")
    run_vision_coco_sweep(
        dataset_dir=Path(args.dataset_dir),
        output_dir=Path(args.output_dir),
        alphas=alphas,
        engine_bin_dir=Path(args.engine_bin_dir),
        codec=codec,
        profiler=profiler,
    )


if __name__ == "__main__":
    main()
