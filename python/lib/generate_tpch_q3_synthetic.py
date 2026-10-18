#!/usr/bin/env python3
"""
generate_tpch_q3_synthetic.py

Synthetic Data Generator for Probabilistic TPC-H Q3 (Shipping Priority Query).
Follows TPC-H schema and statistical characteristics with controlled:
- Zipf skew on customer market segments / grouping keys (z in {0.0, 0.5, 1.0, 1.5, 2.0})
- Join selectivity on date & segment predicates
- Order existence probability models p(o)
- Order-to-lineitem cardinality ratios and lineitem revenue distributions
- MCDB world realizations (bitvectors) across W worlds
- Stream ordering policies (Proxy-sorted by o_totalprice vs. Random)
"""

import os
import json
import argparse
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


MARKET_SEGMENTS = [
    "BUILDING",
    "AUTOMOBILE",
    "MACHINERY",
    "HOUSEHOLD",
    "FURNITURE",
]

# Standard date ranges for TPC-H Q3 predicates
# o_orderdate < date '1995-03-15'
# l_shipdate > date '1995-03-15'
DEFAULT_CUTOFF_DAY = 75  # Normalized day index 0..150


def sample_zipf(n_items: int, n_categories: int, alpha: float, rng: np.random.Generator) -> np.ndarray:
    """Sample category indices 0..n_categories-1 from a Zipf-Mandelbrot / power-law distribution."""
    if alpha <= 0.001:
        return rng.integers(0, n_categories, size=n_items)
    ranks = np.arange(1, n_categories + 1, dtype=np.float64)
    weights = 1.0 / (ranks ** alpha)
    probs = weights / np.sum(weights)
    return rng.choice(n_categories, size=n_items, p=probs)


def sample_piecewise_zipf(
    n_items: int,
    n_categories: int,
    alpha_r: float,
    c_contenders: int = 1,
    p_contender_mass: float = 0.70,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Two-Tier Piecewise Zipf distribution:
    - Top c items get mass p total, decaying gently with exponent alpha_c = 0.1 * alpha_r.
    - Remaining n_categories - c items get mass (1 - p) total, decaying with exponent alpha_r.
    """
    if rng is None:
        rng = np.random.default_rng()
    if c_contenders <= 1 or alpha_r <= 0.001:
        return sample_zipf(n_items, n_categories, alpha_r, rng)

    c = min(c_contenders, n_categories)
    alpha_c = 0.1 * alpha_r
    probs = np.zeros(n_categories, dtype=np.float64)

    raw_cont = 1.0 / (np.arange(1, c + 1, dtype=np.float64) ** alpha_c)
    probs[:c] = p_contender_mass * (raw_cont / np.sum(raw_cont))

    num_rem = n_categories - c
    if num_rem > 0:
        raw_rem = 1.0 / (np.arange(1, num_rem + 1, dtype=np.float64) ** alpha_r)
        probs[c:] = (1.0 - p_contender_mass) * (raw_rem / np.sum(raw_rem))
    else:
        probs[:c] = raw_cont / np.sum(raw_cont)

    probs /= np.sum(probs)
    return rng.choice(n_categories, size=n_items, p=probs)


def generate_tpch_q3_dataset(
    n_customers: Optional[int] = None,
    n_orders: Optional[int] = None,
    target_stream_tuples: Optional[int] = None,
    n_groups: int = 50,
    zipf_skew: float = 1.0,
    contender_c: int = 1,
    p_min: float = 0.1,
    p_max: float = 0.99,
    n_mc_worlds: int = 1000,
    target_segment: str = "BUILDING",
    cutoff_day: int = DEFAULT_CUTOFF_DAY,
    seed: int = 42,
    stream_sort: str = "proxy",
    include_mc_bits: bool = True,
    price_mode: str = "high",
    branching_mode: str = "poisson4",
    selectivity_mode: str = "default",
    prob_mode: str = "smooth_beta",
) -> Dict[str, Any]:
    """
    Generates a full synthetic probabilistic TPC-H Q3 dataset.
    
    If target_stream_tuples is specified (e.g. 12,500 or 50,000), n_orders and n_customers
    are derived working backwards from expected join & predicate selectivity.

    Ablation Parameters:
    - price_mode: "constant" ($10k uniform), "low" ($8k-$12k), "high" ($100-$5000/line, default)
    - branching_mode: "1to1" (1 lineitem/order), "uniform_tpch", "poisson4" (Poisson(4) lineitems, default)
    - selectivity_mode: "full" (100% qualify), "default" (~10% qualify, default)
    - prob_mode: "smooth_beta" (Beta(2, 2), default), "polarized_neural" (Bimodal 0.92 vs 0.03)
    """
    rng = np.random.Generator(np.random.PCG64(seed))

    # Determine base order & customer cardinalities
    if target_stream_tuples is not None:
        if selectivity_mode == "full":
            expected_selectivity = 1.0
            n_orders = target_stream_tuples
            if n_customers is None:
                n_customers = max(100, int(np.round(n_orders / 10.0)))
        else:
            p_seg = 0.20
            p_date = float(cutoff_day) / 150.0
            p_line = 0.985
            expected_selectivity = max(0.01, p_seg * p_date * p_line)
            n_orders = int(np.round(target_stream_tuples / expected_selectivity))
            if n_customers is None:
                n_customers = max(100, int(np.round(n_orders / 10.0)))
    else:
        if n_orders is None:
            n_orders = 15000
        if n_customers is None:
            n_customers = max(100, int(np.round(n_orders / 10.0)))

    # 1. Generate CUSTOMERS
    cust_segments = rng.choice(MARKET_SEGMENTS, size=n_customers, p=[0.2, 0.2, 0.2, 0.2, 0.2])
    
    # 2. Generate ORDERS
    order_custkeys = rng.integers(0, n_customers, size=n_orders)
    order_dates = rng.integers(0, 150, size=n_orders)
    order_shippriorities = rng.integers(0, 8, size=n_orders)
    
    # Group assignment with Zipf skew or Piecewise Contender skew across n_groups candidate groups
    order_group_keys = sample_piecewise_zipf(n_orders, n_groups, zipf_skew, c_contenders=contender_c, rng=rng)

    # Existence probabilities p(o)
    if prob_mode == "polarized_neural":
        # Bimodal neural logits: winning/contender groups have high probability mass,
        # distractors are predominantly low-confidence noise (< 0.05).
        is_contender = order_group_keys < contender_c
        p_high_prob = np.where(is_contender, 0.85, 0.05)
        is_high = rng.uniform(0.0, 1.0, size=n_orders) < p_high_prob
        high_draws = 0.85 + rng.beta(20.0, 2.0, size=n_orders) * 0.14  # [0.85, 0.99]
        low_draws = 0.001 + rng.beta(1.0, 30.0, size=n_orders) * 0.049 # [0.001, 0.05]
        order_probs = np.where(is_high, high_draws, low_draws)
    else:
        # Canonical smooth prior: Beta(2, 2) shifted to [p_min, p_max], mean ~ 0.55
        raw_probs = rng.beta(2.0, 2.0, size=n_orders)
        order_probs = p_min + raw_probs * (p_max - p_min)

    # 3. Generate LINEITEMS per order
    if branching_mode == "1to1":
        n_items_per_order = np.ones(n_orders, dtype=int)
    elif branching_mode in ["uniform_tpch", "uniform"]:
        # Canonical TPC-H Section 4.2.2.13: Discrete Uniform U[1, 7] (mean 4.0, variance 4.0)
        n_items_per_order = rng.integers(1, 8, size=n_orders)
    else:
        # Clamped Poisson legacy option: Poisson(4) clamped [1, 7]
        n_items_per_order = np.clip(rng.poisson(4, size=n_orders), 1, 7)
    
    stream_tuples = []
    ground_truth_world_revenues = np.zeros((n_groups, n_mc_worlds), dtype=np.float64)
    ground_truth_expected_revenues = np.zeros(n_groups, dtype=np.float64)
    ground_truth_proxy_scores = np.zeros(n_groups, dtype=np.float64)
    ground_truth_expected_counts = np.zeros(n_groups, dtype=np.float64)

    # Set v_max upper bound dynamically based on price & branching modes
    if price_mode == "constant":
        v_max = 10000.0 if branching_mode == "1to1" else 70000.0
    elif price_mode == "low":
        v_max = 12000.0 if branching_mode == "1to1" else 84000.0
    else:
        v_max = 50000.0

    n_mc_words = (n_mc_worlds + 63) // 64

    if selectivity_mode == "full":
        qualifying_indices = np.arange(n_orders)
    else:
        order_segs = cust_segments[order_custkeys]
        qualifying_mask = (order_dates < cutoff_day) & (order_segs == target_segment)
        qualifying_indices = np.flatnonzero(qualifying_mask)

    for o_idx in qualifying_indices:
        cust_id = order_custkeys[o_idx]
        seg = cust_segments[cust_id]
        o_date = int(order_dates[o_idx])
        p_exist = float(order_probs[o_idx])
        grp = int(order_group_keys[o_idx])
        priority = int(order_shippriorities[o_idx])

        # Line items for this qualifying order
        n_lines = n_items_per_order[o_idx]
        if price_mode == "constant":
            line_prices = np.full(n_lines, 10000.0)
            line_discounts = np.zeros(n_lines)
        elif price_mode == "low":
            line_prices = rng.uniform(8000.0, 12000.0, size=n_lines)
            line_discounts = np.zeros(n_lines)
        else:
            line_prices = rng.uniform(100.0, 5000.0, size=n_lines)
            line_discounts = rng.uniform(0.00, 0.10, size=n_lines)

        # Shipping date filter on lineitems
        if selectivity_mode == "full":
            qualifying_lines = np.ones(n_lines, dtype=bool)
        else:
            line_shipdates = rng.integers(o_date + 1, 150, size=n_lines)
            qualifying_lines = line_shipdates > cutoff_day
            if not np.any(qualifying_lines):
                continue

        filtered_revenue = float(np.sum(
            line_prices[qualifying_lines] * (1.0 - line_discounts[qualifying_lines])
        ))
        
        total_order_price = float(np.sum(line_prices * (1.0 - line_discounts)))
        
        # Proxy score: proxy is based on expected total price
        proxy_score = float(p_exist * total_order_price)
        
        # Expected revenue for this order
        expected_rev = float(p_exist * filtered_revenue)
        ground_truth_expected_revenues[grp] += expected_rev
        ground_truth_proxy_scores[grp] += proxy_score
        ground_truth_expected_counts[grp] += p_exist

        # MCDB realizations across W worlds
        if include_mc_bits:
            world_mask = (rng.uniform(0.0, 1.0, size=n_mc_worlds) < p_exist)
            ground_truth_world_revenues[grp, world_mask] += filtered_revenue

            pad_len = (64 - (n_mc_worlds % 64)) % 64
            padded = np.pad(world_mask, (0, pad_len), 'constant') if pad_len > 0 else world_mask
            bytes_data = np.packbits(padded, bitorder='little').tobytes()
            mc_words = np.frombuffer(bytes_data, dtype=np.uint64).tolist()
            packed_words = bytes_data.hex()
            active_worlds = int(np.sum(world_mask))
        else:
            mc_words = None
            packed_words = None
            active_worlds = 0

        normalized_score = min(1.0, max(0.0, proxy_score / v_max))
        unit_score = float(p_exist * 1.0)
        exact_score = float(expected_rev)

        stream_tuples.append({
            "order_id": int(o_idx),
            "group_key": grp,
            "group_label": f"Group_{grp:03d}",
            "order_date": o_date,
            "ship_priority": priority,
            "prob": round(p_exist, 6),
            "filtered_revenue": round(filtered_revenue, 4),
            "expected_revenue": round(expected_rev, 4),
            "normalized_score": round(normalized_score, 8),
            "proxy_score": round(proxy_score, 4),
            "unit_score": round(unit_score, 6),
            "exact_score": round(exact_score, 4),
            "mc_bits": mc_words,
            "mcdb_bits_hex": packed_words,
            "mcdb_active_worlds": active_worlds,
        })

    # Sort stream by chosen score space descending
    if stream_sort == "exact":
        sorted_stream = sorted(stream_tuples, key=lambda t: t["exact_score"], reverse=True)
    elif stream_sort == "unit":
        sorted_stream = sorted(stream_tuples, key=lambda t: t["unit_score"], reverse=True)
    else:
        # Default to proxy score space
        sorted_stream = sorted(stream_tuples, key=lambda t: t["proxy_score"], reverse=True)

    # Calculate ground truth winners
    # Expected value winners:
    sorted_gt_groups = np.argsort(-ground_truth_expected_revenues).tolist()
    sorted_proxy_groups = np.argsort(-ground_truth_proxy_scores).tolist()
    sorted_count_groups = np.argsort(-ground_truth_expected_counts).tolist()
    
    # MCDB world plurality winners:
    world_winning_groups = np.argmax(ground_truth_world_revenues, axis=0)
    mcdb_group_win_counts = np.bincount(world_winning_groups, minlength=n_groups)
    mcdb_sorted_groups = np.argsort(-mcdb_group_win_counts).tolist()

    return {
        "metadata": {
            "n_customers": n_customers,
            "n_orders": n_orders,
            "n_groups": n_groups,
            "qualifying_tuples": len(stream_tuples),
            "zipf_skew": zipf_skew,
            "target_segment": target_segment,
            "cutoff_day": cutoff_day,
            "n_mc_worlds": n_mc_worlds,
            "v_max": v_max,
            "seed": seed,
            "stream_sort": stream_sort,
            "contender_c": contender_c,
            "price_mode": price_mode,
            "branching_mode": branching_mode,
            "selectivity_mode": selectivity_mode,
            "prob_mode": prob_mode,
        },
        "ground_truth": {
            "expected_revenue_by_group": [round(float(v), 4) for v in ground_truth_expected_revenues],
            "proxy_score_by_group": [round(float(v), 4) for v in ground_truth_proxy_scores],
            "expected_counts_by_group": [round(float(v), 4) for v in ground_truth_expected_counts],
            "top_groups_expected": sorted_gt_groups,
            "top_groups_proxy": sorted_proxy_groups,
            "top_groups_count": sorted_count_groups,
            "mcdb_win_counts": [int(c) for c in mcdb_group_win_counts],
            "top_groups_mcdb": mcdb_sorted_groups,
        },
        "stream": sorted_stream,
    }


# Backwards-compatibility alias
generate_q3_instance = generate_tpch_q3_dataset


def _generate_and_save_q3_dataset(task_args: Tuple[float, int, int, int, str, int, bool, int, str]) -> Tuple[str, float, bool]:
    import time
    skew, seed, target_stream_tuples, n_groups, stream_sort, n_mc_worlds, include_mc_bits, contender_c, out_dir = task_args
    c_tag = f"_c{contender_c}" if contender_c > 1 else ""
    out_file = os.path.join(out_dir, f"tpch_q3_z{skew}_seed{seed}{c_tag}_{stream_sort}.json")
    if os.path.exists(out_file):
        return out_file, 0.0, True

    t0 = time.time()
    data = generate_tpch_q3_dataset(
        target_stream_tuples=target_stream_tuples,
        n_groups=n_groups,
        zipf_skew=skew,
        contender_c=contender_c,
        n_mc_worlds=n_mc_worlds,
        seed=seed,
        stream_sort=stream_sort,
        include_mc_bits=include_mc_bits,
    )
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f)
    elapsed = time.time() - t0
    return out_file, elapsed, False


def main():
    import time
    from concurrent.futures import ProcessPoolExecutor, as_completed

    parser = argparse.ArgumentParser(description="Generate synthetic TPC-H Q3 probabilistic benchmark data.")
    parser.add_argument("--out_dir", "--output-dir", "--out-dir", "--output_dir", type=str, default="experiment_datasets/tpch_q3_synthetic", dest="out_dir")
    parser.add_argument("--out_file", "--out-file", type=str, default=None, dest="out_file")
    parser.add_argument("--n_orders", "--n-orders", type=int, default=None, dest="n_orders")
    parser.add_argument("--target_stream_tuples", "--target-stream-tuples", type=int, default=None, dest="target_stream_tuples")
    parser.add_argument("--n_groups", "--n-groups", type=int, default=50, dest="n_groups")
    parser.add_argument("--zipf", "--zipf-skew", "--zipf_skew", type=float, default=1.0, dest="zipf")
    parser.add_argument("--contender_c", "--contender-c", type=int, default=1, dest="contender_c", help="Number of co-equal contenders in piecewise Zipfian distribution")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--stream_sort", "--score-space", "--stream-sort", "--score_space", type=str, default="proxy", choices=["proxy", "exact", "unit"], dest="stream_sort")
    parser.add_argument("--n_mc_worlds", "--n-mc-worlds", type=int, default=1000, dest="n_mc_worlds")
    parser.add_argument("--num_seeds", "--num-seeds", type=int, default=None, dest="num_seeds")
    parser.add_argument("--start_seed", "--start-seed", type=int, default=1000, dest="start_seed")
    parser.add_argument("--skews", type=str, default=None, help="Comma-separated list of skews e.g. '1.0,2.0'")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--no_mc_bits", "--no-mc-bits", action="store_true", default=False, dest="no_mc_bits")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.num_seeds is not None:
        skews = [float(z.strip()) for z in args.skews.split(",")] if args.skews else [args.zipf]
        tasks = []
        for z in skews:
            for s in range(args.start_seed, args.start_seed + args.num_seeds):
                tasks.append((z, s, args.target_stream_tuples, args.n_groups, args.stream_sort, args.n_mc_worlds, not args.no_mc_bits, args.contender_c, args.out_dir))

        total_tasks = len(tasks)
        print(f"==========================================================================")
        print(f"  PARALLEL TPC-H Q3 DATASET GENERATION ({args.workers} Workers)")
        print(f"  Total Tasks: {total_tasks} ({len(skews)} Skews x {args.num_seeds} Seeds) | Target Tuples: {args.target_stream_tuples}")
        print(f"==========================================================================")

        t_start = time.time()
        completed = 0
        if args.workers <= 1:
            for task in tasks:
                out_path, elapsed, cached = _generate_and_save_q3_dataset(task)
                completed += 1
                status = "cached" if cached else f"{elapsed:.2f}s"
                print(f"  [{completed:>3}/{total_tasks}] Seed {task[1]} (z={task[0]}) -> {status}")
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(_generate_and_save_q3_dataset, task): task for task in tasks}
                for fut in as_completed(futures):
                    out_path, elapsed, cached = fut.result()
                    completed += 1
                    status = "cached" if cached else f"{elapsed:.2f}s"
                    task = futures[fut]
                    if completed % 5 == 0 or completed == total_tasks or completed == 1:
                        total_elapsed = time.time() - t_start
                        rate = completed / max(0.001, total_elapsed)
                        print(f"  [{completed:>3}/{total_tasks}] Seed {task[1]} (z={task[0]}) -> {status} (rate: {rate:.1f} ds/s)")

        total_elapsed = time.time() - t_start
        print(f"✓ Completed {total_tasks} datasets in {total_elapsed:.2f}s ({total_tasks/max(0.001, total_elapsed):.1f} ds/s) into {args.out_dir}")
        return

    data = generate_tpch_q3_dataset(
        n_orders=args.n_orders,
        target_stream_tuples=args.target_stream_tuples,
        n_groups=args.n_groups,
        zipf_skew=args.zipf,
        n_mc_worlds=args.n_mc_worlds,
        seed=args.seed,
        stream_sort=args.stream_sort,
        include_mc_bits=not args.no_mc_bits,
    )

    if args.out_file:
        out_file = args.out_file
    else:
        suffix = f"_{args.stream_sort}" if args.stream_sort != "proxy" else ""
        out_file = os.path.join(args.out_dir, f"tpch_q3_z{int(args.zipf*10):02d}_seed{args.seed}{suffix}.json")

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f)

    print(f"Generated {data['metadata']['qualifying_tuples']} qualifying tuples for {args.n_groups} groups.")
    print(f"Top 5 expected revenue groups: {data['ground_truth']['top_groups_expected'][:5]}")
    print(f"Saved to: {out_file}")


if __name__ == "__main__":
    main()
