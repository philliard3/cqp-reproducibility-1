#!/usr/bin/env python3
"""
run_production_sweep.py

Master Orchestrator for Conformal Query Processing Production Sweeps:
- One-command execution dispatching presets: 'dry_run', 'paper_focused', 'exhaustive_thesis'.
- Orchestrates dataset preparation, modular runners, result aggregation, and profiling.
- Emits detailed profiling telemetry and production scaling estimates.
- Triggers downstream analysis (Table 1) and camera-ready chart generation (Figures 1-6).
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
sys.path.insert(0, str(PACKAGE_DIR))
sys.path.insert(0, str(REPO_ROOT))

from lib.schema_codec import SchemaCodec, read_jsonl_gz, write_jsonl_gz
from lib.profiler import SweepProfiler
from runners.run_vision_coco import run_vision_coco_sweep
from runners.run_text_jigsaw import run_text_jigsaw_sweep
from runners.run_relational_tpch import run_relational_tpch_sweep


def run_orchestrated_sweep(
    preset_name: str,
    config_file: Path,
    output_base_dir: Path,
    cache_dir: Path,
    engine_bin_dir: Path,
    selected_queries: Optional[List[str]] = None,
):
    total_start = time.time()
    profiler = SweepProfiler(f"orchestrated_{preset_name}")
    codec = SchemaCodec(output_base_dir / "schema_lookup.json")

    with open(config_file, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    if preset_name not in config_data["presets"]:
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {list(config_data['presets'].keys())}")

    preset_cfg = config_data["presets"][preset_name]
    output_base_dir.mkdir(parents=True, exist_ok=True)

    print("==========================================================================")
    print(f" Conformal Database Sweep Orchestrator: Preset [{preset_name}]")
    print(f" Output Directory: {output_base_dir}")
    print(f" Cache Directory:  {cache_dir}")
    print(f" Target Alphas:    {preset_cfg.get('alphas')}")
    print("==========================================================================")

    # 1. Dataset Preflight Verification Step
    print("\n>>> Phase 1: Preflight Verification of Datasets and Generators...")
    t_prep = profiler.start_task("dataset_preparation")
    prep_cmd = [
        sys.executable,
        str(PACKAGE_DIR / "prepare_production_datasets.py"),
        "--preset", preset_name,
        "--config", str(config_file),
        "--cache-dir", str(cache_dir),
        "--preflight-only",
    ]
    subprocess.run(prep_cmd, check=True)
    profiler.record_task_time("dataset_preparation", t_prep)

    # 2. Modular Runner Dispatch
    queries_cfg = preset_cfg.get("queries", {})
    alphas = preset_cfg.get("alphas", [0.01, 0.05, 0.10])
    cal_seeds = list(range(1000, 1000 + preset_cfg.get("cal_seeds", 25)))
    eval_seeds = list(range(2000, 2000 + preset_cfg.get("eval_seeds", 5)))
    dataset_scale = float(preset_cfg.get("dataset_scale", 1.0))
    unified_records: List[Dict[str, Any]] = []

    # Run Vision COCO (Q1a / Q2a)
    if not selected_queries or any(q in ["q1a", "q2a"] for q in selected_queries):
        print("\n>>> Phase 2.1: Executing Vision COCO (Q1a & Q2a)...")
        t_coco = profiler.start_task("vision_coco_runner")
        coco_cache_dir = cache_dir / "vision_coco" if (cache_dir / "vision_coco").exists() else cache_dir / "coco"
        coco_out_dir = output_base_dir / "vision_coco"
        q1_needed = not selected_queries or "q1a" in selected_queries
        q2_needed = not selected_queries or "q2a" in selected_queries

        coco_cfg = queries_cfg.get("q1a", {}) or queries_cfg.get("q2a", {})
        log_path = run_vision_coco_sweep(
            dataset_dir=coco_cache_dir,
            output_dir=coco_out_dir,
            alphas=alphas,
            engine_bin_dir=engine_bin_dir,
            codec=codec,
            profiler=profiler,
            cal_seeds=cal_seeds,
            eval_seeds=eval_seeds,
            is_flat_only=(q1_needed and not q2_needed),
            is_nested_only=(q2_needed and not q1_needed),
            dataset_scale=dataset_scale,
            configs_list=coco_cfg.get("configs"),
        )
        unified_records.extend(read_jsonl_gz(log_path, codec))
        profiler.record_task_time("vision_coco_runner", t_coco)

    # Run Text Jigsaw (Q1b / Q2b)
    if not selected_queries or any(q in ["q1b", "q2b"] for q in selected_queries):
        print("\n>>> Phase 2.2: Executing Text Jigsaw (Q1b & Q2b)...")
        t_jig = profiler.start_task("text_jigsaw_runner")
        jig_cache_dir = cache_dir / "text_jigsaw" if (cache_dir / "text_jigsaw").exists() else cache_dir / "jigsaw"
        jig_out_dir = output_base_dir / "text_jigsaw"
        q1_needed = not selected_queries or "q1b" in selected_queries
        q2_needed = not selected_queries or "q2b" in selected_queries

        jig_cfg = queries_cfg.get("q1b", {}) or queries_cfg.get("q2b", {})
        log_path = run_text_jigsaw_sweep(
            dataset_dir=jig_cache_dir,
            output_dir=jig_out_dir,
            alphas=alphas,
            engine_bin_dir=engine_bin_dir,
            codec=codec,
            profiler=profiler,
            cal_seeds=cal_seeds,
            eval_seeds=eval_seeds,
            is_flat_only=(q1_needed and not q2_needed),
            is_nested_only=(q2_needed and not q1_needed),
            dataset_scale=dataset_scale,
            configs_list=jig_cfg.get("configs"),
        )
        unified_records.extend(read_jsonl_gz(log_path, codec))
        profiler.record_task_time("text_jigsaw_runner", t_jig)

    # Run TPC-H Q3
    if not selected_queries or "q3" in selected_queries:
        print("\n>>> Phase 2.3: Executing Relational TPC-H Q3...")
        t_q3 = profiler.start_task("relational_q3_runner")
        q3_cfg = queries_cfg.get("q3", {})
        q3_cache_dir = cache_dir / "tpch_q3"
        q3_out_dir = output_base_dir / "tpch_q3"

        log_path = run_relational_tpch_sweep(
            query="q3",
            dataset_dir=q3_cache_dir,
            output_dir=q3_out_dir,
            M_list=q3_cfg.get("M_list", [10]),
            z_list=q3_cfg.get("z_list", [1.0]),
            c_list=q3_cfg.get("c_list", [1]),
            k_list=q3_cfg.get("k_list", [1]),
            alphas=alphas,
            cal_seeds=cal_seeds,
            eval_seeds=eval_seeds,
            engine_bin_dir=engine_bin_dir,
            codec=codec,
            profiler=profiler,
            configs_list=q3_cfg.get("configs"),
            dataset_scale=dataset_scale,
        )
        unified_records.extend(read_jsonl_gz(log_path, codec))
        profiler.record_task_time("relational_q3_runner", t_q3)

    # Run TPC-H Q4
    if not selected_queries or "q4" in selected_queries:
        print("\n>>> Phase 2.4: Executing Relational TPC-H Q4 (TPC-H Q10)...")
        t_q4 = profiler.start_task("relational_q4_runner")
        q4_cfg = queries_cfg.get("q4", {})
        q4_cache_dir = cache_dir / "tpch_q10"
        q4_out_dir = output_base_dir / "tpch_q4"

        log_path = run_relational_tpch_sweep(
            query="q4",
            dataset_dir=q4_cache_dir,
            output_dir=q4_out_dir,
            M_list=q4_cfg.get("M_list", [10]),
            z_list=q4_cfg.get("z_list", [1.0]),
            c_list=q4_cfg.get("c_list", [1]),
            k_list=q4_cfg.get("k_list", [1]),
            alphas=alphas,
            cal_seeds=cal_seeds,
            eval_seeds=eval_seeds,
            engine_bin_dir=engine_bin_dir,
            codec=codec,
            profiler=profiler,
            configs_list=q4_cfg.get("configs"),
            dataset_scale=dataset_scale,
        )
        unified_records.extend(read_jsonl_gz(log_path, codec))
        profiler.record_task_time("relational_q4_runner", t_q4)

    # 3. Unify Results
    unified_log_path = output_base_dir / "unified_sweep_results.jsonl.gz"
    write_jsonl_gz(unified_records, unified_log_path, codec)
    codec.save_lookup(output_base_dir / "schema_lookup.json")
    print(f"\n✓ Unified Results Aggregated: {len(unified_records)} evaluation records in {unified_log_path}")

    # 4. Generate Profiling & Telemetry Report
    print("\n>>> Phase 3: Compiling Profiling Telemetry & Production Projections...")
    summary = profiler.generate_profile_summary(dataset_dir=cache_dir, results_dir=output_base_dir)
    projections = profiler.project_production_requirements(current_preset=preset_name, n_workers=4)

    prof_report = {
        "execution_summary": summary,
        "production_scaling_projections": projections,
    }

    report_json_path = output_base_dir / "profiling_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(prof_report, f, indent=2)

    # Formatted Markdown report
    report_md_path = output_base_dir / "profiling_report.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(f"# Sweep Profiling & Production Projection Report\n\n")
        f.write(f"**Preset Executed**: `{preset_name}`  \n")
        f.write(f"**Wall-Clock Elapsed**: {summary['total_wall_clock_sec']} s  \n")
        f.write(f"**Total Engine Invocations**: {summary['total_engine_invocations']}  \n")
        f.write(f"**Tuples Processed**: {summary['total_tuples_processed']:,} ({summary['overall_tuples_per_sec']} tuples/sec)  \n")
        f.write(f"**Peak Memory Usage**: {summary['peak_memory_mb']} MB  \n\n")

        f.write(f"## Disk Usage Audit\n\n")
        f.write(f"- **Cached Datasets**: {summary['dataset_storage'].get('total_mb', 0)} MB ({summary['dataset_storage'].get('file_count', 0)} files)\n")
        f.write(f"- **Compressed Run Results**: {summary['results_storage'].get('total_mb', 0)} MB\n\n")

        f.write(f"## Production Sweep Scaling Projections (on 4 vCPUs)\n\n")
        paper = projections["paper_focused_projection"]
        f.write(f"### Focused Paper Sweep:\n")
        f.write(f"- **Estimated Runs**: {paper['estimated_runs']:,}\n")
        f.write(f"- **Projected Wall-Clock**: {paper['projected_wall_clock_minutes']} min ({paper['projected_wall_clock_hours']} hrs)\n")
        f.write(f"- **Projected Dataset Storage**: {paper['projected_dataset_storage_gb']} GB (Budget: 60 GB - **{'PASS' if paper['within_storage_budget'] else 'FAIL'}**)\n")
        f.write(f"- **Projected Run Results**: {paper['projected_results_storage_mb']} MB (Budget: 4 GB)\n\n")

        thesis = projections["exhaustive_thesis_projection"]
        f.write(f"### Exhaustive Thesis Sweep:\n")
        f.write(f"- **Estimated Runs**: {thesis['estimated_runs']:,}\n")
        f.write(f"- **Projected Wall-Clock**: {thesis['projected_wall_clock_hours']} hrs\n")
        f.write(f"- **Projected Dataset Storage**: {thesis['projected_dataset_storage_gb']} GB\n")

    print(f"✓ Profiling report generated: {report_md_path}")

    # 5. Downstream Analysis & Charts Trigger
    print("\n>>> Phase 4: Triggering Statistical Audit (Table 1) and Chart Rendering...")
    try:
        subprocess.run([
            sys.executable,
            str(PACKAGE_DIR / "analyze_results.py"),
            "--results", str(unified_log_path),
            "--lookup", str(output_base_dir / "schema_lookup.json"),
            "--output-dir", str(output_base_dir / "analysis"),
        ], check=True)
    except Exception as e:
        print(f"Notice: analyze_results.py finished or skipped with: {e}")

    try:
        subprocess.run([
            sys.executable,
            str(PACKAGE_DIR / "generate_charts.py"),
            "--results", str(unified_log_path),
            "--lookup", str(output_base_dir / "schema_lookup.json"),
            "--output-dir", str(output_base_dir / "figures"),
        ], check=True)
    except Exception as e:
        print(f"Notice: generate_charts.py finished or skipped with: {e}")

    # If running canonical_v3, also render V3 paper figures automatically
    if preset_name == "canonical_v3" and (PACKAGE_DIR / "generate_paper_charts_v3.py").exists():
        try:
            print("\n>>> Generating Canonical V3 Publication Figures...")
            subprocess.run([
                sys.executable,
                str(PACKAGE_DIR / "generate_paper_charts_v3.py"),
                "--results-dir", str(output_base_dir),
                "--figures-dir", str(output_base_dir / "figures_v3"),
            ], check=True)
        except Exception as e:
            print(f"Notice: generate_paper_charts_v3.py finished or skipped with: {e}")

    print("==========================================================================")
    print(f"✓ Orchestrated Sweep Finished Successfully in {time.time() - total_start:.1f}s!")
    print("==========================================================================")


def resolve_engine_bin_dir(raw_path: Optional[str] = None) -> Path:
    if raw_path:
        p = Path(raw_path)
        if p.exists():
            return p
    candidates = [
        REPO_ROOT / "mode_u_engine" / "target" / "release",
        PACKAGE_DIR.parent / "mode_u_engine" / "target" / "release",
        Path.cwd() / "mode_u_engine" / "target" / "release",
        Path.cwd() / ".." / "mode_u_engine" / "target" / "release",
    ]
    for c in candidates:
        if c.exists():
            return c
    return REPO_ROOT / "mode_u_engine" / "target" / "release"


def resolve_config_path(raw_path: Optional[str] = None) -> Path:
    if raw_path:
        p = Path(raw_path)
        if p.exists():
            return p
    candidates = [
        PACKAGE_DIR / "sweep_config.json",
        Path.cwd() / "sweep_config.json",
        Path.cwd() / "production_sweep_package" / "sweep_config.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return PACKAGE_DIR / "sweep_config.json"


def main():
    parser = argparse.ArgumentParser(description="Master Orchestrator for Conformal DB sweeps.")
    parser.add_argument(
        "--preset",
        type=str,
        default="dry_run",
        help="Preset name from sweep_config.json (e.g. canonical_v3, paper_focused_quick, dry_run, paper_focused, exhaustive_thesis)",
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=str(PACKAGE_DIR / "results"))
    parser.add_argument("--cache-dir", type=str, default=str(REPO_ROOT / "experiment_datasets" / "production_cache"))
    parser.add_argument("--engine-bin-dir", type=str, default=None)
    parser.add_argument("--queries", type=str, default=None, help="Comma-separated query filter, e.g. q1a,q2a,q3")
    args = parser.parse_args()

    q_filter = [q.strip() for q in args.queries.split(",")] if args.queries else None
    run_orchestrated_sweep(
        preset_name=args.preset,
        config_file=resolve_config_path(args.config),
        output_base_dir=Path(args.output_dir),
        cache_dir=Path(args.cache_dir),
        engine_bin_dir=resolve_engine_bin_dir(args.engine_bin_dir),
        selected_queries=q_filter,
    )


if __name__ == "__main__":
    main()
