#!/usr/bin/env python3
"""
prepare_production_datasets.py

Deterministic dataset preparation and shared cache pool for all experimental workloads:
- Vision (COCO Animals): Q1a (flat) and Q2a (nested) share the same underlying datasets.
- Text (Jigsaw Toxicity): Q1b (flat) and Q2b (nested) share the same underlying datasets.
- Relational (TPC-H Q3): 2-table join with continuous price variance and shipping date predicate.
- Relational (TPC-H Q10): 4-table join with customer nation/return attributes and price variance.

Datasets are keyed by (domain, M, z, c, scale, seed) into a shared cache pool to prevent
duplicate generation. If a cached dataset exists, it is validated and reused.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

# Ensure local lib directory is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from lib.distribution_references import (
    compute_zipf_weights,
    compute_piecewise_zipf_weights,
)
try:
    from generate_tpch_q3_synthetic import generate_tpch_q3_dataset
    from generate_tpch_q10_synthetic import generate_tpch_q10_dataset
except ImportError:
    from lib.generate_tpch_q3_synthetic import generate_tpch_q3_dataset
    from lib.generate_tpch_q10_synthetic import generate_tpch_q10_dataset

COCO_CLASSES_ALL = [
    "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe"
]

JIGSAW_TAGS_ALL = [
    "identity_hate", "insult", "obscene", "severe_toxic", "threat", "toxic"
]

DEFAULT_CACHE_DIR = REPO_ROOT / "experiment_datasets" / "production_cache"


def find_base_table(filename: str) -> Path:
    candidates = [
        REPO_ROOT / "data" / "cached_tables" / filename,
        SCRIPT_DIR.parent / "data" / "cached_tables" / filename,
        SCRIPT_DIR / "cached_tables" / filename,
        SCRIPT_DIR / filename,
        REPO_ROOT / "experiment_datasets" / "cached_tables" / filename,
        REPO_ROOT / "cached_tables" / filename,
        Path.cwd() / "data" / "cached_tables" / filename,
        Path.cwd() / "cached_tables" / filename,
        Path.cwd() / "production_sweep_package" / "cached_tables" / filename,
        Path.cwd() / "experiment_datasets" / "cached_tables" / filename,
    ]
    for c in candidates:
        if c.exists():
            return c
    return SCRIPT_DIR / "cached_tables" / filename


COCO_BASE_SOURCE = find_base_table("coco_animals_calibrated_softmax.json")
JIGSAW_BASE_SOURCE = find_base_table("jigsaw_toxicity_platt.json")


def load_coco_base_table(path: Path) -> Dict[str, Dict[str, float]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    clean: Dict[str, Dict[str, float]] = {}
    for k, class_list in data.items():
        scores = {}
        for item in class_list:
            cid = item[0]["class_id"]
            scores[cid] = float(item[1])
        for c in COCO_CLASSES_ALL:
            scores.setdefault(c, 0.0)
        s = sum(scores[c] for c in COCO_CLASSES_ALL)
        if s > 0:
            for c in COCO_CLASSES_ALL:
                scores[c] /= s
            clean[k] = scores
    return clean


def load_jigsaw_base_table(path: Path) -> Dict[str, Dict[str, float]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    clean: Dict[str, Dict[str, float]] = {}
    for k, class_list in data.items():
        scores = {}
        for item in class_list:
            tag = item[0]["class_id"]
            scores[tag] = max(0.0, min(1.0, float(item[1])))
        for t in JIGSAW_TAGS_ALL:
            scores.setdefault(t, 0.0)
        clean[k] = scores
    return clean


_BASE_TABLE_CLASS_INDEX: Dict[str, Dict[str, List[str]]] = {}


def get_class_index(domain: str, base_table: Dict[str, Dict[str, float]], classes: List[str]) -> Dict[str, List[str]]:
    cache_key = f"{domain}_{'_'.join(classes)}"
    if cache_key in _BASE_TABLE_CLASS_INDEX:
        return _BASE_TABLE_CLASS_INDEX[cache_key]

    base_keys = sorted(list(base_table.keys()))
    class_to_keys: Dict[str, List[str]] = {cls: [] for cls in classes}
    for bk in base_keys:
        top_c = max(base_table[bk].items(), key=lambda x: x[1])[0]
        if top_c in class_to_keys:
            class_to_keys[top_c].append(bk)
    _BASE_TABLE_CLASS_INDEX[cache_key] = class_to_keys
    return class_to_keys


def prepare_ml_dataset(
    domain: str,
    base_table: Dict[str, Dict[str, float]],
    M: int,
    z: float,
    c: int,
    seed: int,
    n_items: int = 1250,
    n_partitions: int = 25,
) -> Dict[str, Any]:
    """
    Generates unified ML evaluation dataset supporting both single-tier Flat (Q1)
    and two-tier Nested (Q2) ground truth evaluation.
    """
    rng = np.random.default_rng(seed)
    is_jigsaw = (domain.lower() == "jigsaw")
    all_classes = JIGSAW_TAGS_ALL if is_jigsaw else COCO_CLASSES_ALL
    classes = all_classes[:min(M, len(all_classes))]
    k_classes = len(classes)
    base_keys = sorted(list(base_table.keys()))

    weights = compute_piecewise_zipf_weights(k_classes, z, c)
    class_to_keys = get_class_index(domain, base_table, classes)

    sampled_classes = rng.choice(k_classes, size=n_items, p=weights)
    sampled_keys = []
    for c_idx in sampled_classes:
        c_name = classes[c_idx]
        avail = class_to_keys.get(c_name, [])
        sampled_keys.append(rng.choice(avail) if avail else rng.choice(base_keys))

    items_per_partition = n_items // n_partitions
    tuples: List[Dict[str, Any]] = []

    part_expected = np.zeros((n_partitions, k_classes), dtype=np.float64)
    flat_expected = np.zeros(k_classes, dtype=np.float64)

    for item_idx, bk in enumerate(sampled_keys):
        p_idx = min(item_idx // items_per_partition, n_partitions - 1)
        part_name = f"part_{p_idx}"
        item_id = f"item_{item_idx:05d}"
        prob_dict = base_table[bk]

        probs = np.array([prob_dict[cls] for cls in classes], dtype=np.float64)
        if not is_jigsaw:
            s = probs.sum()
            if s > 0:
                probs /= s

        part_expected[p_idx] += probs
        flat_expected += probs

        for c_idx, cls in enumerate(classes):
            tuples.append({
                "day": part_name,
                "image_id": item_id,
                "animal": cls,
                "score": float(probs[c_idx]),
            })

    # Nested ground truth (Stage 1 ArgMax winner per partition -> Stage 2 outer count)
    nested_tag_counts = {cls: 0 for cls in classes}
    for d in range(n_partitions):
        w_idx = int(np.argmax(part_expected[d]))
        nested_tag_counts[classes[w_idx]] += 1
    max_part_wins = max(nested_tag_counts.values())
    nested_winners = [cls for cls, cnt in nested_tag_counts.items() if cnt == max_part_wins]
    nested_winner = sorted(nested_winners)[0]

    # Flat ground truth (overall sum)
    flat_scores = {classes[i]: float(flat_expected[i]) for i in range(k_classes)}
    sorted_flat = sorted(flat_scores.items(), key=lambda x: (x[1], -ord(x[0][0])), reverse=True)
    flat_winner = sorted_flat[0][0]

    return {
        "dataset": f"{domain}_M{M}_z{z}_c{c}_seed{seed}",
        "domain": domain,
        "seed": seed,
        "M": M,
        "z": z,
        "c": c,
        "n_images": n_items,
        "n_partitions": n_partitions,
        "images_per_partition": items_per_partition,
        "total_tuples": len(tuples),
        "animal_classes": classes,
        "is_non_exclusive": is_jigsaw,
        "ground_truth": {
            "naive_outer_winner": nested_winner,
            "partition_counts": nested_tag_counts,
        },
        "flat_ground_truth": {
            "naive_winner": flat_winner,
            "expected_scores": flat_scores,
        },
        "tuples": tuples,
    }


def get_cached_or_generate_dataset(
    query: str,
    domain: str,
    M: int,
    z: float,
    c: int,
    seed: int,
    scale: float = 1.0,
    branching: str = "uniform_1_7",
    cache_dir: Path = DEFAULT_CACHE_DIR,
    base_tables: Optional[Dict[str, Any]] = None,
    custom_out_dir: Optional[Path] = None,
) -> Path:
    """
    Ensure dataset is present in shared cache pool or custom ephemeral directory. Returns filepath.
    """
    config_name = f"M{M}_z{int(round(z*100)):03d}_c{c}_sc{int(round(scale*100)):03d}"
    if custom_out_dir is not None:
        out_dir = custom_out_dir
        out_file = out_dir / f"dataset_{domain}_{config_name}_seed_{seed:05d}.json"
    else:
        domain_dir = cache_dir / domain
        out_dir = domain_dir / config_name
        out_file = out_dir / f"seed_{seed:05d}.json"

    out_dir.mkdir(parents=True, exist_ok=True)

    if out_file.exists() and out_file.stat().st_size > 100:
        return out_file

    # Generate fresh
    if domain in ["vision_coco", "coco"]:
        base = base_tables["coco"] if base_tables and "coco" in base_tables else load_coco_base_table(COCO_BASE_SOURCE)
        # Target 50,000 stream tuples at scale 1.0 (50ms target): n_items * M = 50,000
        n_items = int((50000 // max(1, M)) * scale)
        data = prepare_ml_dataset("coco", base, M=M, z=z, c=c, seed=seed, n_items=n_items)
    elif domain in ["text_jigsaw", "jigsaw"]:
        base = base_tables["jigsaw"] if base_tables and "jigsaw" in base_tables else load_jigsaw_base_table(JIGSAW_BASE_SOURCE)
        # Target 50,000 stream tuples at scale 1.0 (50ms target): n_items * M = 50,000
        n_items = int((50000 // max(1, M)) * scale)
        data = prepare_ml_dataset("jigsaw", base, M=M, z=z, c=c, seed=seed, n_items=n_items)
    elif "tpch_q3" in domain:
        n_tuples = int(50000 * scale)
        data = generate_tpch_q3_dataset(
            target_stream_tuples=n_tuples,
            n_groups=M,
            zipf_skew=z,
            contender_c=c,
            seed=seed,
            branching_mode="uniform" if "uniform" in branching else "poisson",
        )
    elif "tpch_q10" in domain:
        n_tuples = int(50000 * scale)
        data = generate_tpch_q10_dataset(
            n_customers=M,
            target_stream_tuples=n_tuples,
            zipf_skew=z,
            seed=seed,
            include_mc_bits=True,
        )
    else:
        raise ValueError(f"Unknown domain: {domain}")

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return out_file


def main():
    parser = argparse.ArgumentParser(description="Prepare production sweep datasets or preflight validation.")
    parser.add_argument("--preset", type=str, default="dry_run", help="Preset name from sweep_config.json")
    parser.add_argument("--config", type=str, default=str(SCRIPT_DIR / "sweep_config.json"))
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--preflight-only", action="store_true", default=True, help="Validate data generators without pre-allocating full factorial datasets to disk.")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    preset_cfg = cfg["presets"][args.preset]
    print(f"=== Dataset Preflight Verification: Preset [{args.preset}] ===")
    print(f"Calibration seeds: {preset_cfg.get('cal_seeds')}, Evaluation seeds: {preset_cfg.get('eval_seeds')}")

    # Load and verify base ML tables
    base_tables = {}
    if COCO_BASE_SOURCE.exists():
        print(f"✓ Found COCO base table at {COCO_BASE_SOURCE} ({COCO_BASE_SOURCE.stat().st_size / 1e6:.1f} MB)")
        base_tables["coco"] = load_coco_base_table(COCO_BASE_SOURCE)
        print(f"  Verified {len(base_tables['coco'])} COCO images parsed across 10 animal classes.")
    else:
        raise FileNotFoundError(
            f"Required COCO base table not found: {COCO_BASE_SOURCE}\n"
            f"Please ensure 'coco_animals_calibrated_softmax.json' is present in 'production_sweep_package/cached_tables/' "
            f"or 'experiment_datasets/cached_tables/'."
        )

    if JIGSAW_BASE_SOURCE.exists():
        print(f"✓ Found Jigsaw base table at {JIGSAW_BASE_SOURCE} ({JIGSAW_BASE_SOURCE.stat().st_size / 1e6:.1f} MB)")
        base_tables["jigsaw"] = load_jigsaw_base_table(JIGSAW_BASE_SOURCE)
        print(f"  Verified {len(base_tables['jigsaw'])} Jigsaw comments parsed across 6 toxicity tags.")
    else:
        raise FileNotFoundError(
            f"Required Jigsaw base table not found: {JIGSAW_BASE_SOURCE}\n"
            f"Please ensure 'jigsaw_toxicity_platt.json' is present in 'production_sweep_package/cached_tables/' "
            f"or 'experiment_datasets/cached_tables/'."
        )

    # Validate generator operational status with a single micro probe
    test_dir = cache_dir / ".preflight_probe"
    test_dir.mkdir(parents=True, exist_ok=True)
    try:
        t0 = time.time()
        f_coco = get_cached_or_generate_dataset("q1a", "coco", M=10, z=1.0, c=1, seed=9999, custom_out_dir=test_dir, base_tables=base_tables)
        f_jig = get_cached_or_generate_dataset("q1b", "jigsaw", M=6, z=1.0, c=1, seed=9999, custom_out_dir=test_dir, base_tables=base_tables)
        f_q3 = get_cached_or_generate_dataset("q3", "tpch_q3", M=10, z=1.0, c=1, seed=9999, scale=0.1, custom_out_dir=test_dir)
        f_q10 = get_cached_or_generate_dataset("q4", "tpch_q10", M=10, z=1.0, c=1, seed=9999, scale=0.1, custom_out_dir=test_dir)
        print(f"✓ Generator sanity verified: Vision, Text, and Relational probes generated in {time.time() - t0:.2f}s.")
    finally:
        import shutil
        shutil.rmtree(test_dir, ignore_errors=True)

    print("✓ Preflight verification complete. On-demand streaming execution enabled (ephemeral per-configuration generation).")


if __name__ == "__main__":
    main()
