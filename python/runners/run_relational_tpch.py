#!/usr/bin/env python3
"""
run_relational_tpch.py

Modular, standalone runner for Relational (TPC-H Q3 & Q10) workloads:
- Q3: 2-Table Join (Orders |><| Lineitem) with shipping date predicate and continuous price variance.
- Q4: 4-Table Join (Customer |><| Orders |><| Lineitem |><| Nation) with nation filtering and return risk.

Executes the 4 applicable strategies:
1. Naive_Conformal_End (Full stream scan to exhaustion)
2. MCDB_Sum_1000 (Monte Carlo 1000-world continuous revenue simulation)
3. Conformal_InOperator_Stop (Indep) (In-operator pruning via calibrated tau)
4. QN1_External_Pull (Indep) (Volcano pull iterator with post-hoc conformal bounding)

Outputs compact .jsonl.gz logs encoded with stable integer IDs via SchemaCodec.
"""

import os
import sys
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
sys.path.insert(0, str(PACKAGE_DIR / "lib"))
sys.path.insert(0, str(REPO_ROOT))

from lib.schema_codec import SchemaCodec, write_jsonl_gz
from lib.profiler import SweepProfiler
try:
    from lib.generate_tpch_q3_synthetic import generate_tpch_q3_dataset
    from lib.generate_tpch_q10_synthetic import generate_tpch_q10_dataset
except ImportError:
    from generate_tpch_q3_synthetic import generate_tpch_q3_dataset
    from generate_tpch_q10_synthetic import generate_tpch_q10_dataset


def run_command(cmd: List[str], desc: str) -> float:
    t0 = time.time()
    print(f"  [Running] {desc}...", flush=True)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in iter(proc.stdout.readline, ''):
        if line:
            sys.stdout.write(f"    {line}")
            sys.stdout.flush()
    proc.stdout.close()
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"{desc} failed with exit code {ret}")
    return time.time() - t0


def calibrate_tpch(
    query: str,
    cal_seeds: List[int],
    M: int,
    z: float,
    c: int,
    target_k_list: List[int],
    alphas: List[float],
    out_summary_path: Path,
    scale: float = 1.0,
) -> Dict[str, Any]:
    """
    Computes split-conformal calibration thresholds for TPC-H queries across target alphas.
    """
    calib_records = []
    is_q10 = (query.lower() == "q4" or "q10" in query.lower())
    step_5pct = max(1, len(cal_seeds) // 20)
    target_tuples = int(round(50000 * scale))

    for idx, s in enumerate(cal_seeds):
        if is_q10:
            data = generate_tpch_q10_dataset(
                n_customers=M,
                target_stream_tuples=target_tuples,
                zipf_skew=z,
                seed=s,
                include_mc_bits=False,
            )
        else:
            data = generate_tpch_q3_dataset(
                target_stream_tuples=target_tuples,
                n_groups=M,
                zipf_skew=z,
                contender_c=c,
                seed=s,
                include_mc_bits=False,
            )

        gt_top = data["ground_truth"]["top_groups_expected"]
        v_max = data["metadata"]["v_max"]
        if "proxy_score_by_group" in data["ground_truth"]:
            score_by_grp = {
                g: data["ground_truth"]["proxy_score_by_group"][g] / v_max
                for g in range(data["metadata"]["n_groups"])
            }
        else:
            score_by_grp = {g: 0.0 for g in range(data["metadata"]["n_groups"])}
            for t in data["stream"]:
                score_by_grp[t["group_key"]] += (t.get("proxy_score", 0.0) / v_max)

        calib_records.append({"seed": s, "gt_top": gt_top, "score_by_grp": score_by_grp})

        if (idx + 1) % step_5pct == 0 or (idx + 1) == len(cal_seeds):
            pct = ((idx + 1) / len(cal_seeds)) * 100.0
            print(f"    [Calib Progress: {pct:5.1f}% | {idx+1:>3}/{len(cal_seeds)}] {query} in RAM (M={M}, z={z:.1f}, c={c})", flush=True)

    n_cal = len(calib_records)
    summary: Dict[str, Any] = {
        "metadata": {
            "query": query,
            "M": M, "z": z, "c": c,
            "n_cal_seeds": n_cal,
        },
        "thresholds": {}
    }

    for k in target_k_list:
        summary["thresholds"][f"k_{k}"] = {}
        # Min scores across top k ground truth
        mins = []
        for r in calib_records:
            top_k_gt = r["gt_top"][:k]
            m_val = min(r["score_by_grp"].get(g, 0.0) for g in top_k_gt) if top_k_gt else 0.0
            mins.append(m_val)
        mins.sort()

        for a in alphas:
            q_idx = max(0, min(int(np.floor((n_cal + 1) * a)) - 1, n_cal - 1))
            tau = float(mins[q_idx]) if mins else 0.0
            a_key = f"alpha_{int(round(a * 100)):02d}"
            summary["thresholds"][f"k_{k}"][a_key] = {
                "alpha": a,
                "tau": round(tau, 5),
                "is_valid_guarantee": n_cal >= int(1.0 / a) - 1,
            }
            z_key = f"z{int(round(z * 10)):02d}"
            active_key = f"active_k_{k}_{a_key}"
            summary.setdefault(active_key, {})[z_key] = round(tau, 5)
            summary[active_key]["default"] = round(tau, 5)
            summary.setdefault("active_thresholds", {})[z_key] = round(tau, 5)
            summary["active_thresholds"]["default"] = round(tau, 5)

    out_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def run_relational_tpch_sweep(
    query: str,
    dataset_dir: Path,
    output_dir: Path,
    M_list: List[int],
    z_list: List[float],
    c_list: List[int],
    k_list: List[int],
    alphas: List[float],
    cal_seeds: List[int],
    eval_seeds: Optional[List[int]] = None,
    engine_bin_dir: Optional[Path] = None,
    codec: Optional[SchemaCodec] = None,
    profiler: Optional[SweepProfiler] = None,
    configs_list: Optional[List[List[Any]]] = None,
    dataset_scale: float = 1.0,
) -> Path:
    """
    Executes TPC-H Q3 or Q4 benchmark sweeps across specified parameter configurations.
    Uses on-demand ephemeral evaluation dataset generation and immediate cleanup to guarantee
    sub-gigabyte peak disk storage throughout the sweep.
    """
    import shutil
    if eval_seeds is None:
        eval_seeds = list(range(2000, 2005))
    if engine_bin_dir is None:
        engine_bin_dir = REPO_ROOT / "mode_u_engine" / "target" / "release"
    if codec is None:
        codec = SchemaCodec()
    if profiler is None:
        profiler = SweepProfiler(f"relational_{query}_sweep")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_eval_records: List[Dict[str, Any]] = []

    is_q10 = (query.lower() == "q4" or "q10" in query.lower())
    bin_name = "benchmark_tpch_q10" if is_q10 else "benchmark_tpch_q3"
    engine_bin = engine_bin_dir / bin_name

    if not engine_bin.exists():
        raise FileNotFoundError(f"Missing required engine binary: {engine_bin}")

    if configs_list:
        resolved_configs = []
        for cfg in configs_list:
            m_val = int(cfg[0])
            z_val = float(cfg[1])
            c_val = int(cfg[2])
            sc_val = float(cfg[3]) if len(cfg) > 3 else dataset_scale
            resolved_configs.append((m_val, z_val, c_val, sc_val))
    else:
        resolved_configs = [(M, z, c, dataset_scale) for M in M_list for z in z_list for c in c_list]

    total_configs = len(resolved_configs)
    for cfg_idx, (M, z, c, cur_scale) in enumerate(resolved_configs, 1):
        cfg_tag = f"M{M}_z{int(round(z*100)):03d}_c{c}"
        if cur_scale != dataset_scale:
            cfg_tag += f"_scale{int(round(cur_scale*100)):03d}"
        pct_cfg = (cfg_idx / total_configs) * 100.0
        print(f"\n=======================================================", flush=True)
        print(f"[{query} Configuration {cfg_idx}/{total_configs} ({pct_cfg:.1f}%)] {cfg_tag} (scale={cur_scale})", flush=True)
        print(f"=======================================================", flush=True)
        calib_summary_path = output_dir / f"calib_summary_{cfg_tag}.json"

        # 1. Calibrate once for this (M, z, c) distribution across target k and alphas (in RAM)
        t_cal = profiler.start_task(f"{query}_calib_{cfg_tag}")
        calib_summary = calibrate_tpch(
            query=query,
            cal_seeds=cal_seeds,
            M=M,
            z=z,
            c=c,
            target_k_list=k_list,
            alphas=alphas,
            out_summary_path=calib_summary_path,
            scale=cur_scale,
        )
        profiler.record_task_time(f"{query}_calib_{cfg_tag}", t_cal)

        # 2. Ephemeral On-Demand Generation of Evaluation Datasets for this configuration
        cur_eval_dir = dataset_dir / f"ephemeral_eval_{cfg_tag}"
        cur_eval_dir.mkdir(parents=True, exist_ok=True)
        from prepare_production_datasets import get_cached_or_generate_dataset
        step_eval_5pct = max(1, len(eval_seeds) // 20)
        for s_idx, s in enumerate(eval_seeds):
            get_cached_or_generate_dataset(
                query=query,
                domain="tpch_q10" if is_q10 else "tpch_q3",
                M=M,
                z=z,
                c=c,
                seed=s,
                scale=cur_scale,
                cache_dir=dataset_dir,
                custom_out_dir=cur_eval_dir,
            )
            if (s_idx + 1) % step_eval_5pct == 0 or (s_idx + 1) == len(eval_seeds):
                pct = ((s_idx + 1) / len(eval_seeds)) * 100.0
                print(f"    [Eval Gen Progress: {pct:5.1f}% | {s_idx+1:>3}/{len(eval_seeds)}] {query} (M={M}, z={z:.1f}, c={c})", flush=True)

        # 3. Evaluate across each (k, alpha) combination (once across all evaluation datasets)
        for k in k_list:
            baseline_k_records = []
            for a_idx, a in enumerate(alphas):
                eval_json_out = output_dir / f"eval_{cfg_tag}_k{k}_a{int(round(a*100)):02d}.json"
                t_eval = profiler.start_task(f"{query}_eval_{cfg_tag}_k{k}_a{int(round(a*100)):02d}")
                
                # On first alpha, evaluate baselines (Naive, MCDB) alongside conformal strategies.
                # On subsequent alphas, only evaluate strategies that depend on alpha (tau_calib).
                if a_idx == 0:
                    strats_to_run = "Naive,Naive_Conformal_End,MCDB_Sum_1000,QN1_External_Pull,Conformal_InOperator_Stop"
                else:
                    strats_to_run = "Naive_Conformal_End,QN1_External_Pull,Conformal_InOperator_Stop"

                cmd = [
                    str(engine_bin),
                    "--dir", str(cur_eval_dir),
                    "--k", str(k),
                    "--alpha", str(a),
                    "--conformal-summary", str(calib_summary_path),
                    "--strategies", strats_to_run,
                    "--output", str(eval_json_out),
                ]
                run_command(cmd, f"{query} Eval ({cfg_tag}, k={k}, alpha={a})")
                profiler.record_task_time(f"{query}_eval_{cfg_tag}_k{k}_a{int(round(a*100)):02d}", t_eval)

                a_key = f"alpha_{int(round(a * 100)):02d}"
                tau_from_summary = calib_summary.get("thresholds", {}).get(f"k_{k}", {}).get(a_key, {}).get("tau", 0.0) if calib_summary else 0.0

                if eval_json_out.exists():
                    with open(eval_json_out, "r", encoding="utf-8") as f:
                        raw_results = json.load(f)
                    for d_res in raw_results:
                        seed = d_res.get("dataset_seed", 0)
                        for strat_res in d_res.get("results", []):
                            s_name = strat_res.get("strategy")
                            # Canonicalize name
                            canon_strat = s_name
                            if s_name == "Conformal_InOperator_Stop":
                                canon_strat = "Conformal_InOperator_Stop (Indep)"
                            elif s_name == "QN1_External_Pull":
                                canon_strat = "QN1_External_Pull (Indep)"

                            tau_val = strat_res.get("tau_calib", 0.0)
                            if tau_val == 0.0:
                                tau_val = tau_from_summary

                            rec = {
                                "config": {
                                    "query": query,
                                    "strategy": canon_strat,
                                    "M": M,
                                    "z": z,
                                    "c": c,
                                    "k_query": k,
                                    "alpha": a,
                                    "seed": seed,
                                    "dataset_scale": cur_scale,
                                },
                                "metrics": {
                                    "tuples_ingested": strat_res.get("items_read", 0),
                                    "tuples_total": strat_res.get("total_stream_items", 0),
                                    "stream_savings_pct": strat_res.get("stream_savings_ratio", 0.0) * 100.0,
                                    "engine_time_ms": strat_res.get("in_engine_ms", 0.0),
                                    "e2e_time_ms": strat_res.get("in_engine_ms", 0.0),
                                    "emitted_set": [e["group_key"] if isinstance(e, dict) else e for e in (strat_res.get("top_k_emitted") or strat_res.get("emitted_candidates") or [])],
                                    "emitted_set_size": len(strat_res.get("top_k_emitted") or strat_res.get("emitted_candidates") or []),
                                    "ground_truth_top_k": strat_res.get("ground_truth_top_k") or strat_res.get("gt_top_k", []),
                                    "covered": strat_res.get("gt_simultaneous_covered", strat_res.get("gt_set_coverage", False)),
                                    "recall_at_set": strat_res.get("gt_marginal_recall", 0.0),
                                    "tau_threshold": tau_val,
                                    "relative_set_size": len(strat_res.get("top_k_emitted") or strat_res.get("emitted_candidates") or []) / max(1, k),
                                }
                            }
                            all_eval_records.append(rec)
                            profiler.record_run_point(query, canon_strat, strat_res.get("in_engine_ms", 0.0), strat_res.get("items_read", 0))

                            # Cache alpha-independent baselines to replicate for subsequent alphas
                            if a_idx == 0 and canon_strat in ["Naive", "MCDB_Sum_1000"]:
                                baseline_k_records.append((canon_strat, seed, rec["metrics"]))

                    # Replicate cached alpha-independent baselines on subsequent alphas
                    if a_idx > 0:
                        for b_strat, b_seed, b_metrics in baseline_k_records:
                            all_eval_records.append({
                                "config": {
                                    "query": query,
                                    "strategy": b_strat,
                                    "M": M,
                                    "z": z,
                                    "c": c,
                                    "k_query": k,
                                    "alpha": a,
                                    "seed": b_seed,
                                    "dataset_scale": cur_scale,
                                },
                                "metrics": dict(b_metrics),
                            })
                            profiler.record_run_point(query, b_strat, b_metrics.get("engine_time_ms", 0.0), b_metrics.get("tuples_ingested", 0))

                    # Clean up temporary engine json to conserve disk
                    try:
                        eval_json_out.unlink()
                    except OSError:
                        pass

        # 4. Clean up ephemeral evaluation datasets for this configuration immediately
        shutil.rmtree(cur_eval_dir, ignore_errors=True)

    final_log_path = output_dir / f"{query}_eval.jsonl.gz"
    write_jsonl_gz(all_eval_records, final_log_path, codec)
    codec.save_lookup(output_dir / "schema_lookup.json")
    print(f"✓ {query} Relational sweep complete: {len(all_eval_records)} evaluation points -> {final_log_path}")
    return final_log_path


def main():
    parser = argparse.ArgumentParser(description="Run Relational TPC-H (Q3/Q4) benchmark sweep.")
    parser.add_argument("--query", choices=["q3", "q4"], default="q3")
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/relational")
    parser.add_argument("--engine-bin-dir", type=str, default=str(REPO_ROOT / "mode_u_engine" / "target" / "release"))
    parser.add_argument("--M", type=str, default="10")
    parser.add_argument("--z", type=str, default="1.0")
    parser.add_argument("--c", type=str, default="1")
    parser.add_argument("--k", type=str, default="1")
    parser.add_argument("--alphas", type=str, default="0.01,0.05,0.10")
    args = parser.parse_args()

    alphas = [float(a.strip()) for a in args.alphas.split(",")]
    M_list = [int(m.strip()) for m in args.M.split(",")]
    z_list = [float(z.strip()) for z in args.z.split(",")]
    c_list = [int(c.strip()) for c in args.c.split(",")]
    k_list = [int(k.strip()) for k in args.k.split(",")]

    codec = SchemaCodec()
    profiler = SweepProfiler(f"relational_{args.query}_sweep")
    run_relational_tpch_sweep(
        query=args.query,
        dataset_dir=Path(args.dataset_dir),
        output_dir=Path(args.output_dir),
        M_list=M_list,
        z_list=z_list,
        c_list=c_list,
        k_list=k_list,
        alphas=alphas,
        cal_seeds=list(range(1000, 1025)),
        engine_bin_dir=Path(args.engine_bin_dir),
        codec=codec,
        profiler=profiler,
    )


if __name__ == "__main__":
    main()
