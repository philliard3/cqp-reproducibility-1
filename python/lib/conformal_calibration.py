#!/usr/bin/env python3
"""
lib/conformal_calibration.py

Transforms raw calibration snapshots emitted by Rust binaries
(calibrate_flat_strategies and calibrate_nested_strategies)
into the structured CalibrationMatrix JSON expected by benchmark binaries
(benchmark_flat_conformal and benchmark_nested_conformal).
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np


def compute_tau_threshold(scores: List[float], alpha: float) -> float:
    """
    Computes finite-sample conformal quantile threshold for a score list.
    Formula: sorted_scores[min(n - 1, ceil((n + 1) * (1 - alpha)) - 1)]
    """
    n = len(scores)
    if n == 0:
        return 0.0
    sorted_scores = sorted(scores)
    q_idx = max(0, min(int(np.ceil((n + 1) * (1.0 - alpha))) - 1, n - 1))
    return float(sorted_scores[q_idx])


def compute_conformal_threshold(
    scores: List[float],
    alpha: float,
    n_seeds: int,
) -> Dict[str, Any]:
    """
    Computes finite-sample conformal quantile threshold.
    scores: list of length n_seeds, where score is score_lo(Y*) or -inf if Y* not in candidates.
    """
    sorted_scores = sorted(scores)
    # Standard conformal quantile index: floor((n + 1) * alpha) clamped to [0, n - 1]
    q_idx = max(0, min(int(np.floor((n_seeds + 1) * alpha)) - 1, n_seeds - 1))
    quantile_val = sorted_scores[q_idx]

    if quantile_val > -1e8:
        tau = float(quantile_val)
        is_valid = True
    else:
        finite_scores = [s for s in scores if s > -1e8]
        tau = float(min(finite_scores)) if finite_scores else 0.0
        is_valid = False

    covered = sum(1 for s in scores if s >= tau - 1e-9)
    calib_coverage = float(covered / n_seeds) if n_seeds > 0 else 0.0

    return {
        "tau": float(tau),
        "is_valid_guarantee": is_valid,
        "target_coverage": round(1.0 - alpha, 2),
        "calib_coverage": round(calib_coverage, 4),
        "avg_set_size": 1.0,
    }


def process_calibration_snapshots(
    input_path: Path,
    output_path: Path,
    target_alphas: List[float] = [0.01, 0.05, 0.10],
    ranks: List[int] = [1, 2, 3],
) -> Dict[str, Any]:
    """
    Parses raw calibration snapshots and writes structured CalibrationMatrix JSON.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        records: List[Dict[str, Any]] = json.load(f)

    if not records:
        raise ValueError(f"No calibration records found in {input_path}")

    # Group by config_name
    by_config: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        cfg = r.get("config_name", "z10_c1")
        by_config.setdefault(cfg, []).append(r)

    calibration_matrix: Dict[str, Any] = {}

    for cfg_name, cfg_records in by_config.items():
        n_seeds = len(cfg_records)
        cfg_dict: Dict[str, Any] = {}

        # Collect all strategies present in records
        strat_set = set()
        for r in cfg_records:
            for s in r.get("strategy_results", []):
                strat_set.add(s["strategy"])

        for strat in strat_set:
            strat_ranks_dict: Dict[str, Any] = {}

            for k in ranks:
                rank_key = f"rank_{k}"
                scores_for_k = []
                set_sizes_for_k = []
                recall_count = 0

                for r in cfg_records:
                    gt = r.get("ground_truth_winner", "")
                    strat_res = next((s for s in r.get("strategy_results", []) if s["strategy"] == strat), None)
                    if not strat_res or rank_key not in strat_res:
                        scores_for_k.append(-1e9)
                        continue

                    candidates_at_k = strat_res[rank_key]
                    cand_names = [c["candidate"] for c in candidates_at_k]

                    if gt and gt in cand_names:
                        recall_count += 1
                        cand_item = next(c for c in candidates_at_k if c["candidate"] == gt)
                        scores_for_k.append(cand_item.get("score_lo", 0.0))
                    else:
                        scores_for_k.append(-1e9)

                recall_at_k = recall_count / float(n_seeds) if n_seeds > 0 else 0.0
                k_entry: Dict[str, Any] = {
                    "recall_at_k": round(recall_at_k, 4),
                }

                for alpha in target_alphas:
                    alpha_key = f"alpha_{alpha:.2f}"
                    th_info = compute_conformal_threshold(scores_for_k, alpha, n_seeds)

                    # Compute average set size under this tau
                    tau = th_info["tau"]
                    sizes = []
                    for r in cfg_records:
                        strat_res = next((s for s in r.get("strategy_results", []) if s["strategy"] == strat), None)
                        if strat_res and rank_key in strat_res:
                            candidates_at_k = strat_res[rank_key]
                            filtered = [c for c in candidates_at_k if c.get("score_lo", 0.0) >= tau - 1e-9]
                            sizes.append(len(filtered))
                    th_info["avg_set_size"] = round(float(np.mean(sizes)), 2) if sizes else 1.0
                    k_entry[alpha_key] = th_info

                strat_ranks_dict[f"k_{k}"] = k_entry
                strat_ranks_dict[f"rank_{k}"] = k_entry

            # Store under primary strategy name
            cfg_dict[strat] = strat_ranks_dict

            # Store under aliases so both QN1 (MutEx) and QN1_MutEx match
            alias_underscore = strat.replace(" (", "_").replace(")", "")
            alias_paren = alias_underscore.replace("_", " (") + ")" if "_" in alias_underscore else alias_underscore
            cfg_dict[alias_underscore] = strat_ranks_dict
            cfg_dict[alias_paren] = strat_ranks_dict

        calibration_matrix[cfg_name] = cfg_dict

    # Populate canonical config aliases ("z10_c1", "unknown", etc.)
    if calibration_matrix:
        first_cfg = next(iter(calibration_matrix.values()))
        if "z10_c1" not in calibration_matrix:
            calibration_matrix["z10_c1"] = first_cfg
        if "unknown" not in calibration_matrix:
            calibration_matrix["unknown"] = first_cfg

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(calibration_matrix, f, indent=2)

    return calibration_matrix
