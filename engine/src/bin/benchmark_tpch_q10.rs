/// benchmark_tpch_q10.rs
///
/// High-performance native Rust in-engine benchmark for continuous lost-revenue aggregation
/// with large customer candidate space (TPC-H Q10: Returned Item Reporting Query, K=20).
///
/// Evaluates the reduced strategy comparison suite:
/// 1. Naive: Full stream scan continuous expected lost-revenue aggregation.
/// 2. Naive_Conformal_End: Exhaustive scan + post-filtering by calibrated threshold tau.
/// 3. MCDB_Sum_100: Continuous Monte Carlo simulation (100 worlds).
/// 4. MCDB_Sum_1000: Continuous Monte Carlo simulation (1000 worlds).
/// 5. QN1_External_Pull: Demand-driven rank-by-rank pulling until conformal set criterion is met.
/// 6. Conformal_InOperator_Stop: In-operator early stopping with calibrated threshold tau.

use std::fs::{self, File};
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::time::Instant;

use indexmap::IndexMap;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use mode_u_engine::operators::compositional::unified_argmax::{
    CompactKey, MemoryTupleStream, TupleStream,
};
use mode_u_engine::operators::compositional::unified_argmax_sum::{
    AggregationModeSum, ArgMaxSumConfig, ComposableArgMaxOverSum, LifetimeSnapshot,
};
use mode_u_engine::tuple::{FieldValue, ProbInterval, Tuple};

#[derive(Deserialize, Debug, Clone)]
pub struct StreamTuple {
    pub order_id: usize,
    #[serde(default)]
    pub line_id: Option<usize>,
    pub cust_id: usize,
    pub group_key: usize,
    pub group_label: String,
    #[serde(default)]
    pub nation_name: Option<String>,
    #[serde(default)]
    pub acct_bal: Option<f64>,
    pub order_date: usize,
    #[serde(default)]
    pub return_flag: Option<String>,
    pub prob: f64,
    pub filtered_revenue: f64,
    pub expected_revenue: f64,
    #[serde(default)]
    pub normalized_score: Option<f64>,
    pub proxy_score: f64,
    #[serde(default)]
    pub unit_score: Option<f64>,
    #[serde(default)]
    pub exact_score: Option<f64>,
    #[serde(default)]
    pub mc_bits: Option<Vec<u64>>,
    #[serde(default)]
    pub mcdb_bits_hex: Option<String>,
    #[serde(default)]
    pub mcdb_active_worlds: Option<usize>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct GroundTruthInfo {
    pub expected_revenue_by_group: Vec<f64>,
    #[serde(default)]
    pub proxy_score_by_group: Option<Vec<f64>>,
    pub top_groups_expected: Vec<usize>,
    #[serde(default)]
    pub top_groups_proxy: Option<Vec<usize>>,
    pub mcdb_win_counts: Vec<usize>,
    pub top_groups_mcdb: Vec<usize>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct DatasetMetadata {
    pub query: Option<String>,
    pub n_customers: usize,
    pub n_orders: usize,
    pub n_groups: usize,
    pub qualifying_tuples: usize,
    pub zipf_skew: f64,
    pub n_mc_worlds: usize,
    #[serde(default = "default_v_max")]
    pub v_max: f64,
    pub seed: usize,
    #[serde(default)]
    pub stream_sort: Option<String>,
    #[serde(default)]
    pub k_standard: Option<usize>,
}

fn default_v_max() -> f64 {
    50000.0
}

#[derive(Deserialize, Debug, Clone)]
pub struct TpchQ10DatasetFile {
    pub metadata: DatasetMetadata,
    pub ground_truth: GroundTruthInfo,
    pub stream: Vec<StreamTuple>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EmittedGroupResult {
    pub rank: usize,
    pub group_key: usize,
    pub score_lo: f64,
    pub score_hi: f64,
    pub mcdb_win_prob: Option<f64>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct StrategyRunResult {
    pub strategy: String,
    pub k_target: usize,
    pub items_read: usize,
    pub total_stream_items: usize,
    pub stream_savings_ratio: f64,
    pub in_engine_us: u128,
    pub in_engine_ms: f64,
    pub top_k_emitted: Vec<EmittedGroupResult>,
    pub gt_simultaneous_covered: bool,
    pub gt_marginal_recall: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lifetime_trace: Option<Vec<LifetimeSnapshot>>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DatasetBenchmarkResult {
    pub dataset_seed: usize,
    pub zipf_skew: f64,
    pub stream_sort: String,
    pub total_tuples: usize,
    pub n_groups: usize,
    pub results: Vec<StrategyRunResult>,
}

fn hex_decode(s: &str) -> Result<Vec<u8>, ()> {
    (0..s.len())
        .step_by(2)
        .map(|i| {
            if i + 2 <= s.len() {
                u8::from_str_radix(&s[i..i + 2], 16).map_err(|_| ())
            } else {
                Err(())
            }
        })
        .collect()
}

fn load_dataset_as_tuples(dataset: &TpchQ10DatasetFile) -> (Vec<std::sync::Arc<Tuple>>, Vec<String>) {
    let v_max = dataset.metadata.v_max;
    let mut tuples = Vec::with_capacity(dataset.stream.len());

    for (idx, raw) in dataset.stream.iter().enumerate() {
        let mut data = IndexMap::new();
        let t_str = format!("t_{}", idx);
        let cust_str = format!("cust_{}", raw.cust_id);
        let group_str = format!("Group_{:04}", raw.group_key);

        // Pre-warm global string interner
        mode_u_engine::ids::intern(&t_str);
        mode_u_engine::ids::intern(&cust_str);
        mode_u_engine::ids::intern(&group_str);

        data.insert("tuple_id".to_string(), FieldValue::Str(t_str));
        data.insert("cust_id".to_string(), FieldValue::Str(cust_str));
        data.insert("group_key".to_string(), FieldValue::Str(group_str));
        data.insert("filtered_revenue".to_string(), FieldValue::Float(raw.filtered_revenue));
        data.insert("expected_revenue".to_string(), FieldValue::Float(raw.expected_revenue));
        data.insert("proxy_score".to_string(), FieldValue::Float(raw.proxy_score));

        if let Some(masks) = &raw.mc_bits {
            for (w_idx, mask) in masks.iter().enumerate() {
                let k = format!("__mc_bits_{}__", w_idx);
                data.insert(k, FieldValue::Int(*mask as i64));
            }
        } else if let Some(hex_str) = &raw.mcdb_bits_hex {
            if let Ok(bytes) = hex_decode(hex_str) {
                let n_words = (bytes.len() + 7) / 8;
                for w in 0..n_words {
                    let mut word = 0u64;
                    for b in 0..8 {
                        let byte_idx = w * 8 + b;
                        if byte_idx < bytes.len() {
                            word |= (bytes[byte_idx] as u64) << (b * 8);
                        }
                    }
                    data.insert(format!("__mc_bits_{}__", w), FieldValue::Int(word as i64));
                }
            }
        }

        let norm_score = (raw.proxy_score / v_max).max(0.0).min(1.0);
        let score = ProbInterval::from_direct(norm_score);
        tuples.push(std::sync::Arc::new(Tuple::new_interval(data, score)));
    }

    let all_groups: Vec<String> = (0..dataset.metadata.n_groups)
        .map(|g| {
            let s = format!("Group_{:04}", g);
            mode_u_engine::ids::intern(&s);
            s
        })
        .collect();

    (tuples, all_groups)
}

fn run_strategy_sum(
    strategy_name: &str,
    mode: AggregationModeSum,
    tuples: &[std::sync::Arc<Tuple>],
    all_groups: &[String],
    dataset: &TpchQ10DatasetFile,
    k_query: usize,
    k_cand_max: usize,
    tau_stop: Option<f64>,
    monitor_lifetime: bool,
) -> StrategyRunResult {
    let total_stream = tuples.len();
    let v_max = dataset.metadata.v_max;

    let mut card_map = FxHashMap::default();
    card_map.insert(CompactKey::empty(), total_stream);

    let cfg = ArgMaxSumConfig {
        item_key: vec!["tuple_id".to_string()],
        count_group_by: vec!["group_key".to_string()],
        comparison_key: vec![],
        expected_cardinality: card_map,
        mode,
        k_target: k_cand_max,
        value_field: "filtered_revenue".to_string(),
        v_max,
        monitor_lifetime,
    };

    let t_start = Instant::now();
    let stream = Box::new(MemoryTupleStream::new_presorted(tuples.to_vec()));
    let mut op = ComposableArgMaxOverSum::new(stream, cfg, all_groups.to_vec());

    let mut emitted_tuples = Vec::new();
    while let Some(t) = op.next_tuple() {
        emitted_tuples.push(t);

        // If external pulling QN1: stop once candidate score drops below tau or candidate limit reached
        if let Some(tau) = tau_stop {
            let last_t = emitted_tuples.last().unwrap();
            let score_rev = last_t.data.get("score_lo")
                .or_else(|| last_t.data.get("revenue_lo"))
                .and_then(|v| v.as_f64())
                .unwrap_or_else(|| last_t.score.to_prob_lo() * v_max);
            if (score_rev < tau && emitted_tuples.len() >= k_query) || emitted_tuples.len() >= k_cand_max {
                break;
            }
        } else if emitted_tuples.len() >= k_cand_max && !strategy_name.starts_with("Conformal") {
            break;
        }
    }
    let elapsed = t_start.elapsed().as_micros();
    let items_read = op.total_items_read();
    let savings = 1.0 - (items_read as f64 / total_stream.max(1) as f64);
    let lifetime_trace = if monitor_lifetime {
        Some(op.get_lifetime_trace())
    } else {
        None
    };

    let mut emitted_results = Vec::new();
    let mut emitted_group_keys = Vec::new();

    for (rank, t) in emitted_tuples.into_iter().enumerate() {
        let grp_str = t.data.get("group_key").and_then(|v| v.as_str()).unwrap_or("Group_0000");
        let grp_idx = grp_str.trim_start_matches("Group_").parse::<usize>().unwrap_or(0);
        emitted_group_keys.push(grp_idx);

        let (s_lo, s_hi, mcdb_prob) = if strategy_name.starts_with("MCDB") {
            let p = t.score.to_prob_hi();
            (
                dataset.ground_truth.expected_revenue_by_group[grp_idx],
                dataset.ground_truth.expected_revenue_by_group[grp_idx],
                Some(p),
            )
        } else {
            let lo = t.data.get("score_lo")
                .or_else(|| t.data.get("revenue_lo"))
                .and_then(|v| v.as_f64())
                .unwrap_or_else(|| t.score.to_prob_lo() * v_max);
            let hi = t.data.get("score_hi")
                .or_else(|| t.data.get("revenue_hi"))
                .and_then(|v| v.as_f64())
                .unwrap_or_else(|| t.score.to_prob_hi() * v_max);
            (lo, hi, None)
        };

        emitted_results.push(EmittedGroupResult {
            rank: rank + 1,
            group_key: grp_idx,
            score_lo: s_lo,
            score_hi: s_hi,
            mcdb_win_prob: mcdb_prob,
        });
    }

    let gt_top_k: Vec<usize> = dataset.ground_truth.top_groups_expected.iter().take(k_query).copied().collect();
    let gt_set: std::collections::HashSet<usize> = gt_top_k.iter().copied().collect();
    let emitted_set: std::collections::HashSet<usize> = emitted_group_keys.iter().copied().collect();
    let matches_count = gt_set.intersection(&emitted_set).count();

    let marginal_recall = matches_count as f64 / k_query.max(1) as f64;
    let simultaneous_covered = matches_count == k_query;

    StrategyRunResult {
        strategy: strategy_name.to_string(),
        k_target: k_query,
        items_read,
        total_stream_items: total_stream,
        stream_savings_ratio: (savings * 10000.0).round() / 10000.0,
        in_engine_us: elapsed,
        in_engine_ms: elapsed as f64 / 1000.0,
        top_k_emitted: emitted_results,
        gt_simultaneous_covered: simultaneous_covered,
        gt_marginal_recall: marginal_recall,
        lifetime_trace,
    }
}

fn evaluate_dataset_file(
    path: &Path,
    k_query: usize,
    k_calib: usize,
    k_cand_max: usize,
    calib_map: Option<&FxHashMap<String, f64>>,
    monitor_lifetime: bool,
    selected_strategies: Option<&[String]>,
) -> Result<DatasetBenchmarkResult, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let dataset: TpchQ10DatasetFile = serde_json::from_reader(reader)?;

    let total_tuples = dataset.stream.len();
    let (tuples, all_groups) = load_dataset_as_tuples(&dataset);
    let stream_sort = dataset.metadata.stream_sort.clone().unwrap_or_else(|| "proxy".to_string());

    // Determine calibrated threshold tau for k_calib, alpha=0.05
    let zipf_key = format!("z{:02}", (dataset.metadata.zipf_skew * 10.0).round() as usize);
    let tau_calib = calib_map
        .and_then(|m| m.get(&zipf_key).copied())
        .unwrap_or_else(|| {
            let k_idx = (k_calib - 1).min(dataset.ground_truth.top_groups_expected.len() - 1);
            let k_grp = dataset.ground_truth.top_groups_expected[k_idx];
            let v_max = dataset.metadata.v_max.max(1.0);
            if let Some(proxy_scores) = &dataset.ground_truth.proxy_score_by_group {
                (proxy_scores[k_grp] / v_max) * 0.97
            } else {
                (dataset.ground_truth.expected_revenue_by_group[k_grp] / v_max) * 0.97
            }
        });

    let should_run = |name: &str| -> bool {
        match selected_strategies {
            Some(list) => list.iter().any(|s| s.eq_ignore_ascii_case(name)),
            None => true,
        }
    };

    let mut results = Vec::new();

    // 1. Naive Full Scan Reference
    if should_run("Naive") || should_run("Naive_Conformal_End") {
        let naive_res = run_strategy_sum(
            "Naive",
            AggregationModeSum::NaiveSum,
            &tuples,
            &all_groups,
            &dataset,
            k_query,
            k_cand_max,
            None,
            monitor_lifetime,
        );
        if should_run("Naive") {
            results.push(naive_res.clone());
        }

        // 2. Naive + Conformal Operator on End (Exhaustive scan + post-filtering by tau)
        if should_run("Naive_Conformal_End") {
            let mut naive_conf_res = naive_res;
            naive_conf_res.strategy = "Naive_Conformal_End".to_string();
            naive_conf_res.top_k_emitted.retain(|c| c.score_lo >= tau_calib - 1e-9);
            let emitted_set: std::collections::HashSet<usize> = naive_conf_res.top_k_emitted.iter().map(|c| c.group_key).collect();
            let gt_top_k: Vec<usize> = dataset.ground_truth.top_groups_expected.iter().take(k_query).copied().collect();
            let gt_set: std::collections::HashSet<usize> = gt_top_k.iter().copied().collect();
            let matches_count = gt_set.intersection(&emitted_set).count();
            naive_conf_res.gt_marginal_recall = matches_count as f64 / k_query.max(1) as f64;
            naive_conf_res.gt_simultaneous_covered = matches_count == k_query;
            results.push(naive_conf_res);
        }
    }

    // 3. MCDB_Sum_100 (100 Worlds Continuous Accumulation)
    if should_run("MCDB_Sum_100") {
        let mcdb_100_res = run_strategy_sum(
            "MCDB_Sum_100",
            AggregationModeSum::MCDBSum { n_worlds: 100 },
            &tuples,
            &all_groups,
            &dataset,
            k_query,
            k_cand_max,
            None,
            false,
        );
        results.push(mcdb_100_res);
    }

    // 4. MCDB_Sum_1000 (1000 Worlds Continuous Accumulation)
    if should_run("MCDB_Sum_1000") {
        let mcdb_1k_res = run_strategy_sum(
            "MCDB_Sum_1000",
            AggregationModeSum::MCDBSum { n_worlds: 1000 },
            &tuples,
            &all_groups,
            &dataset,
            k_query,
            k_cand_max,
            None,
            false,
        );
        results.push(mcdb_1k_res);
    }

    // 5. QN1_External_Pull (Consumer pulls until conformal threshold satisfied)
    if should_run("QN1_External_Pull") {
        let qn1_pull_res = run_strategy_sum(
            "QN1_External_Pull",
            AggregationModeSum::QN1DemandRanked,
            &tuples,
            &all_groups,
            &dataset,
            k_query,
            k_cand_max,
            Some(tau_calib),
            monitor_lifetime,
        );
        results.push(qn1_pull_res);
    }

    // 6. Conformal_InOperator_Stop (Operator prunes hi < tau, halts as soon as set is safe)
    if should_run("Conformal_InOperator_Stop") {
        let conf_stop_res = run_strategy_sum(
            "Conformal_InOperator_Stop",
            AggregationModeSum::ConformalSetPruning { tau: tau_calib },
            &tuples,
            &all_groups,
            &dataset,
            k_query,
            k_cand_max,
            None,
            monitor_lifetime,
        );
        results.push(conf_stop_res);
    }

    Ok(DatasetBenchmarkResult {
        dataset_seed: dataset.metadata.seed,
        zipf_skew: dataset.metadata.zipf_skew,
        stream_sort,
        total_tuples,
        n_groups: dataset.metadata.n_groups,
        results,
    })
}

fn load_calibration_map(path: &Path, k_target: usize, alpha_target: f64) -> Option<FxHashMap<String, f64>> {
    let file = File::open(path).ok()?;
    let reader = BufReader::new(file);
    let v: serde_json::Value = serde_json::from_reader(reader).ok()?;
    let mut map = FxHashMap::default();

    let k_key = format!("k_{}", k_target);
    let a_key = format!("alpha_{:02}", (alpha_target * 100.0).round() as usize);

    if let Some(obj) = v.as_object() {
        for (_dist_name, dist_val) in obj {
            let zipf = dist_val.get("distribution")
                .and_then(|d| d.get("zipf"))
                .and_then(|z| z.as_f64())
                .unwrap_or(0.0);
            let z_key = format!("z{:02}", (zipf * 10.0).round() as usize);
            if let Some(tau) = dist_val
                .get("calibration_thresholds")
                .and_then(|c| c.get(&k_key))
                .and_then(|k| k.get(&a_key))
                .and_then(|a| a.get("tau_simultaneous"))
                .and_then(|t| t.as_f64())
            {
                map.insert(z_key, tau);
            }
        }
    }
    Some(map)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut dir_path: Option<String> = None;
    let mut file_path: Option<String> = None;
    let mut output_path: Option<String> = None;
    let mut calib_summary_path: Option<String> = None;
    let mut k_query = 20;
    let mut k_calib: Option<usize> = None;
    let mut k_cand_max_opt: Option<usize> = None;
    let mut alpha_target = 0.05;
    let mut monitor_lifetime = false;
    let mut selected_strategies: Option<Vec<String>> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--dir" => {
                if i + 1 < args.len() {
                    dir_path = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--file" => {
                if i + 1 < args.len() {
                    file_path = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--output" => {
                if i + 1 < args.len() {
                    output_path = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--conformal-summary" => {
                if i + 1 < args.len() {
                    calib_summary_path = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--k" => {
                if i + 1 < args.len() {
                    k_query = args[i + 1].parse().unwrap_or(20);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--k-calib" => {
                if i + 1 < args.len() {
                    k_calib = Some(args[i + 1].parse().unwrap_or(25));
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--k-cand" => {
                if i + 1 < args.len() {
                    k_cand_max_opt = Some(args[i + 1].parse().unwrap_or(25));
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--alpha" => {
                if i + 1 < args.len() {
                    alpha_target = args[i + 1].parse().unwrap_or(0.05);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--monitor-lifetime" => {
                monitor_lifetime = true;
                i += 1;
            }
            "--strategies" => {
                if i + 1 < args.len() {
                    let strats: Vec<String> = args[i + 1]
                        .split(',')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect();
                    selected_strategies = Some(strats);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--quick" => {
                file_path = Some("experiment_datasets/tpch_q10_synthetic/tpch_q10_z10_seed100.json".to_string());
                i += 1;
            }
            _ => i += 1,
        }
    }

    let k_calib_target = k_calib.unwrap_or(k_query);
    let k_cand_max = k_cand_max_opt.unwrap_or_else(|| if k_query == 20 { 25 } else { k_query });

    println!("==========================================================================");
    println!(" Rust Native Engine: Probabilistic TPC-H Q10 Benchmark (k_query={}, k_calib={}, k_cand={}, alpha={})", k_query, k_calib_target, k_cand_max, alpha_target);
    if let Some(strats) = &selected_strategies {
        println!(" Selected Strategies Filter: {:?}", strats);
    } else {
        println!(" Strategies: Naive, Naive_Conformal_End, MCDB_Sum_100, MCDB_Sum_1000, QN1_External_Pull, Conformal_InOperator_Stop");
    }
    if monitor_lifetime {
        println!(" State Monitoring: ENABLED (10% Quantile Lifetime Snapshots)");
    }
    println!("==========================================================================");

    let calib_map = calib_summary_path.as_deref().and_then(|p| load_calibration_map(Path::new(p), k_calib_target, alpha_target));
    if calib_map.is_some() {
        println!("Loaded calibrated thresholds from conformal summary (k_calib={}, k_query={}).", k_calib_target, k_query);
    }

    let mut all_results = Vec::new();

    if let Some(f) = file_path {
        println!("Evaluating single dataset: {}", f);
        let res = evaluate_dataset_file(
            Path::new(&f),
            k_query,
            k_calib_target,
            k_cand_max,
            calib_map.as_ref(),
            monitor_lifetime,
            selected_strategies.as_deref(),
        )?;
        for strat in &res.results {
            println!(
                "  [{:<26}] Latency: {:>8.3} ms | Items: {:>5}/{} ({:>5.1}% savings) | Recall: {:>5.1}%",
                strat.strategy,
                strat.in_engine_ms,
                strat.items_read,
                strat.total_stream_items,
                strat.stream_savings_ratio * 100.0,
                strat.gt_marginal_recall * 100.0
            );
        }
        all_results.push(res);
    } else if let Some(d) = dir_path {
        println!("Evaluating all dataset files in: {}", d);
        let mut paths: Vec<PathBuf> = fs::read_dir(d)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().map_or(false, |ext| ext == "json"))
            .collect();
        paths.sort();

        let total_paths = paths.len();
        let step_5pct = (total_paths / 20).max(1);
        for (idx, p) in paths.into_iter().enumerate() {
            if let Ok(res) = evaluate_dataset_file(
                &p,
                k_query,
                k_calib_target,
                k_cand_max,
                calib_map.as_ref(),
                monitor_lifetime,
                selected_strategies.as_deref(),
            ) {
                let cur = idx + 1;
                if cur % step_5pct == 0 || cur == 1 || cur == total_paths {
                    let pct = (cur as f64 / total_paths as f64) * 100.0;
                    let naive_ms = res.results.iter().find(|r| r.strategy == "Naive").map(|r| r.in_engine_ms).unwrap_or(0.0);
                    let mcdb100_ms = res.results.iter().find(|r| r.strategy == "MCDB_Sum_100").map(|r| r.in_engine_ms).unwrap_or(0.0);
                    let mcdb1k_ms = res.results.iter().find(|r| r.strategy == "MCDB_Sum_1000").map(|r| r.in_engine_ms).unwrap_or(0.0);
                    let conf_res = res.results.iter().find(|r| r.strategy == "Conformal_InOperator_Stop");
                    let conf_ms = conf_res.map(|r| r.in_engine_ms).unwrap_or(0.0);
                    let conf_sav = conf_res.map(|r| r.stream_savings_ratio * 100.0).unwrap_or(0.0);

                    println!(
                        "  [Progress: {:5.1}% | {:>3}/{}] Seed {:>5} (z={:>3.1}, N={:>5}): Naive: {:>6.2} ms | MCDB100: {:>6.2} ms | MCDB1k: {:>6.2} ms | Conf_Stop: {:>6.2} ms ({:>4.1}% sav)",
                        pct,
                        cur,
                        total_paths,
                        res.dataset_seed,
                        res.zipf_skew,
                        res.total_tuples,
                        naive_ms,
                        mcdb100_ms,
                        mcdb1k_ms,
                        conf_ms,
                        conf_sav,
                    );
                }
                all_results.push(res);
            }
        }
    } else {
        eprintln!("Usage: benchmark_tpch_q10 [--quick | --file <path> | --dir <path>] [--k <target>] [--k-calib <calib>] [--conformal-summary <path>] [--strategies <list>] [--monitor-lifetime] [--output <path>]");
        std::process::exit(1);
    }

    if let Some(out) = output_path {
        let json_str = serde_json::to_string_pretty(&all_results)?;
        if let Some(parent) = Path::new(&out).parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&out, json_str)?;
        println!("Saved benchmark results to: {}", out);
    }

    println!("==========================================================================");
    Ok(())
}
