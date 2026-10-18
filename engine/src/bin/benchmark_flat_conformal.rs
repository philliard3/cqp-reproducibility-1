/// benchmark_flat_conformal.rs
///
/// Conformal prediction set evaluation binary for Flat Composable ArgMax-over-Count queries.
/// Evaluates:
///   "What is the top-k most popular animal / toxicity tag across the entire stream?"
///
/// Operates on the same dataset files as Nested queries, evaluating single-stage
/// `ComposableArgMaxOverCount` across all images/comments without partition nesting.
///
/// Evaluates across:
///   - Modalities: COCO (MutuallyExclusive budget) vs Jigsaw (Independent budget)
///   - Strategies:
///       * QN1 (Relative bound separation stopping)
///       * ConformalCountPruning (In-Operator early stopping at tau_alpha)
///       * Naive_Conformal_End (Exhaustive scan + post-filtering by tau)
///       * Naive (Exhaustive baseline)
///   - Ranks: k in {1, 2, 3}
///   - Significance levels: alpha in {0.01, 0.05, 0.10}
///   - Piecewise Contender Cliff distributions: z in {0.5, 1.0, 2.0}, c in {1, 2, 3, 4}

use std::fs::{self, File};
use std::io::BufReader;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use indexmap::IndexMap;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use mode_u_engine::tuple::{Tuple, FieldValue, ProbInterval};
use mode_u_engine::operators::compositional::unified_argmax::{
    ComposableArgMaxOverCount, ArgMaxConfig, AggregationMode, ItemBudgetSemantics, TupleStream, MemoryTupleStream, CompactKey
};

#[derive(Serialize, Deserialize, Debug, Clone)]
struct RawDatasetTuple {
    #[serde(default, alias = "feed", alias = "thread", alias = "partition_id")]
    day: String,
    #[serde(alias = "doc_id", alias = "comment_id", alias = "item_id")]
    image_id: String,
    #[serde(alias = "topic", alias = "tag", alias = "candidate_id")]
    animal: String,
    score: f64,
    #[serde(default)]
    mc_bits: Option<Vec<u64>>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct FlatGroundTruth {
    #[serde(default)]
    naive_winner: Option<String>,
    #[serde(default)]
    expected_scores: Option<FxHashMap<String, f64>>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct PreparedDatasetFile {
    total_tuples: usize,
    #[serde(default)]
    n_partitions: usize,
    #[serde(default, alias = "n_documents", alias = "n_comments")]
    n_images: usize,
    #[serde(default, alias = "docs_per_partition", alias = "comments_per_partition")]
    images_per_partition: usize,
    #[serde(default, alias = "class_names")]
    animal_classes: Vec<String>,
    #[serde(default)]
    flat_ground_truth: Option<FlatGroundTruth>,
    tuples: Vec<RawDatasetTuple>,
    #[serde(default)]
    is_non_exclusive: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AlphaThresholdEntry {
    pub tau: f64,
    pub is_valid_guarantee: bool,
    pub target_coverage: f64,
    pub calib_coverage: f64,
    #[serde(default)]
    pub avg_set_size: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RankThresholdEntry {
    pub recall_at_k: f64,
    #[serde(rename = "alpha_0.01")]
    pub alpha_0_01: AlphaThresholdEntry,
    #[serde(rename = "alpha_0.05")]
    pub alpha_0_05: AlphaThresholdEntry,
    #[serde(rename = "alpha_0.10")]
    pub alpha_0_10: AlphaThresholdEntry,
}

pub type CalibrationMatrix = FxHashMap<String, FxHashMap<String, FxHashMap<String, RankThresholdEntry>>>;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FlatEvalPoint {
    pub strategy: String,
    pub k: usize,
    pub alpha: f64,
    pub target_coverage: f64,
    pub tau: f64,
    pub is_valid_guarantee: bool,
    pub is_covered: bool,
    pub prediction_set: Vec<String>,
    pub set_size: usize,
    pub matches_top1: bool,
    pub items_read: usize,
    pub total_stream_items: usize,
    pub stream_savings_ratio: f64,
    pub in_engine_ms: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FlatDatasetEvalResult {
    pub dataset_file: String,
    pub config_name: String,
    pub seed: usize,
    pub ground_truth_winner: String,
    pub eval_points: Vec<FlatEvalPoint>,
}

fn load_dataset(path: &PathBuf) -> (PreparedDatasetFile, Vec<Arc<Tuple>>) {
    let file = File::open(path).unwrap_or_else(|e| panic!("Failed to open dataset {}: {}", path.display(), e));
    let reader = BufReader::new(file);
    let dataset: PreparedDatasetFile = serde_json::from_reader(reader)
        .unwrap_or_else(|e| panic!("Failed to parse JSON dataset {}: {}", path.display(), e));

    for c in &dataset.animal_classes {
        mode_u_engine::ids::intern(c);
    }

    let mut tuples = Vec::with_capacity(dataset.tuples.len());
    for raw in &dataset.tuples {
        mode_u_engine::ids::intern(&raw.image_id);
        mode_u_engine::ids::intern(&raw.animal);

        let mut data = IndexMap::new();
        data.insert("image_id".to_string(), FieldValue::Str(raw.image_id.clone()));
        data.insert("animal".to_string(), FieldValue::Str(raw.animal.clone()));

        let score = ProbInterval::from_direct(raw.score);
        tuples.push(Arc::new(Tuple::new_interval(data, score)));
    }

    (dataset, tuples)
}

fn infer_config_name(path: &PathBuf) -> (String, usize) {
    let full_path = path.to_str().unwrap_or("");
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let s = format!("{} {}", full_path, stem);
    let seed = if let Some(idx) = stem.rfind("seed") {
        stem[idx + 4..].trim_start_matches('_').parse::<usize>().unwrap_or(0)
    } else {
        0
    };

    let config_name = if s.contains("uniform") {
        "uniform".to_string()
    } else if s.contains("z05_c1") || s.contains("z0.5_c1") || s.contains("z050_c1") {
        "z05_c1".to_string()
    } else if s.contains("z10_c1") || s.contains("z1.0_c1") || s.contains("z100_c1") {
        "z10_c1".to_string()
    } else if s.contains("z15_c1") || s.contains("z1.5_c1") || s.contains("z150_c1") {
        "z15_c1".to_string()
    } else if s.contains("z20_c1") || s.contains("z2.0_c1") || s.contains("z200_c1") {
        "z20_c1".to_string()
    } else if s.contains("z05_c2") || s.contains("z0.5_c2") || s.contains("z050_c2") {
        "z05_c2".to_string()
    } else if s.contains("z10_c2") || s.contains("z1.0_c2") || s.contains("z100_c2") {
        "z10_c2".to_string()
    } else if s.contains("z15_c2") || s.contains("z1.5_c2") || s.contains("z150_c2") {
        "z15_c2".to_string()
    } else if s.contains("z20_c2") || s.contains("z2.0_c2") || s.contains("z200_c2") {
        "z20_c2".to_string()
    } else if s.contains("z05_c3") || s.contains("z0.5_c3") || s.contains("z050_c3") {
        "z05_c3".to_string()
    } else if s.contains("z10_c3") || s.contains("z1.0_c3") || s.contains("z100_c3") {
        "z10_c3".to_string()
    } else if s.contains("z20_c3") || s.contains("z2.0_c3") || s.contains("z200_c3") {
        "z20_c3".to_string()
    } else if s.contains("z05_c4") || s.contains("z0.5_c4") || s.contains("z050_c4") {
        "z05_c4".to_string()
    } else if s.contains("z10_c4") || s.contains("z1.0_c4") || s.contains("z100_c4") {
        "z10_c4".to_string()
    } else if s.contains("z20_c4") || s.contains("z2.0_c4") || s.contains("z200_c4") {
        "z20_c4".to_string()
    } else {
        "z10_c1".to_string()
    };

    (config_name, seed)
}

fn compute_flat_ground_truth(dataset: &PreparedDatasetFile) -> (String, FxHashMap<String, f64>) {
    if let Some(fgt) = &dataset.flat_ground_truth {
        if let (Some(w), Some(s)) = (&fgt.naive_winner, &fgt.expected_scores) {
            return (w.clone(), s.clone());
        }
    }

    let mut scores: FxHashMap<String, f64> = FxHashMap::default();
    for t in &dataset.tuples {
        *scores.entry(t.animal.clone()).or_insert(0.0) += t.score;
    }

    let mut sorted: Vec<(String, f64)> = scores.iter().map(|(k, v)| (k.clone(), *v)).collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let winner = sorted.first().map(|(k, _)| k.clone()).unwrap_or_else(|| "none".to_string());
    (winner, scores)
}

fn run_flat_strategy(
    mode: AggregationMode,
    budget: ItemBudgetSemantics,
    k: usize,
    tuples: &[Arc<Tuple>],
    dataset: &PreparedDatasetFile,
) -> (Vec<(String, f64, f64)>, usize, f64) {
    let mut card = FxHashMap::default();
    // Total items is number of distinct images or comments
    let n_items = if dataset.n_images > 0 {
        dataset.n_images
    } else {
        dataset.total_tuples / dataset.animal_classes.len().max(1)
    };
    card.insert(CompactKey::empty(), n_items);

    let leaf_stream = Box::new(MemoryTupleStream::new(tuples.to_vec()));

    let flat_cfg = ArgMaxConfig {
        item_key: vec!["image_id".to_string()],
        count_group_by: vec!["animal".to_string()],
        comparison_key: vec![],
        expected_cardinality: card,
        budget_semantics: budget,
        mode,
        k_target: k,
    };

    let t0 = Instant::now();
    let mut op = ComposableArgMaxOverCount::new(
        leaf_stream,
        flat_cfg,
        dataset.animal_classes.clone(),
    );

    let mut emitted: Vec<(String, f64, f64)> = Vec::new();
    while let Some(tuple) = op.next_tuple() {
        let cand = tuple.data.get("animal").unwrap().as_str().unwrap().to_string();
        let lo = tuple.score.to_prob_lo();
        let hi = tuple.score.to_prob_hi();
        emitted.push((cand, lo, hi));
    }
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let items_read = op.total_items_read();

    (emitted, items_read, elapsed_ms)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut dir_path = "experiment_datasets/eval_datasets".to_string();
    let mut calib_path = "experiment_datasets/calibration_artifacts/flat_calibration_thresholds.json".to_string();
    let mut out_path = "flat_conformal_eval_results.json".to_string();
    let mut log_interval = 25;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--dir" => { dir_path = args[i + 1].clone(); i += 2; }
            "--conformal-thresholds" => { calib_path = args[i + 1].clone(); i += 2; }
            "--output" => { out_path = args[i + 1].clone(); i += 2; }
            "--log-interval" => { log_interval = args[i + 1].parse().unwrap_or(25); i += 2; }
            _ => { i += 1; }
        }
    }

    println!("==========================================================================");
    println!("  FLAT COMPOSABLE ARGMAX-OVER-COUNT CONFORMAL BENCHMARK                   ");
    println!("  Evaluation Directory: {}", dir_path);
    println!("  Conformal Thresholds: {}", calib_path);
    println!("  Output File:          {}", out_path);
    println!("==========================================================================");

    let calib_file = File::open(&calib_path)
        .unwrap_or_else(|e| panic!("Failed to open calibration thresholds at {}: {}", calib_path, e));
    let calib_matrix: CalibrationMatrix = serde_json::from_reader(BufReader::new(calib_file))
        .unwrap_or_else(|e| panic!("Failed to parse calibration JSON at {}: {}", calib_path, e));

    let mut files: Vec<PathBuf> = fs::read_dir(&dir_path)
        .unwrap_or_else(|e| panic!("Failed to read dir {}: {}", dir_path, e))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|ext| ext == "json").unwrap_or(false))
        .collect();
    files.sort();

    println!("Found {} evaluation dataset files.", files.len());
    let alphas = [0.01, 0.05, 0.10];
    let ranks = [1, 2, 3];

    let mut results: Vec<FlatDatasetEvalResult> = Vec::with_capacity(files.len());
    let t_start = Instant::now();

    for (f_idx, path) in files.iter().enumerate() {
        let (config_name, seed) = infer_config_name(path);
        let (dataset, tuples) = load_dataset(path);
        let total_stream_items = tuples.len();
        let (gt_winner, _) = compute_flat_ground_truth(&dataset);

        let budget = if dataset.is_non_exclusive {
            ItemBudgetSemantics::Independent
        } else {
            ItemBudgetSemantics::MutuallyExclusive
        };

        let mut eval_points = Vec::new();

        // 1. Baseline Naive
        let (naive_emitted, naive_items, naive_ms) = run_flat_strategy(
            AggregationMode::Naive,
            budget,
            1,
            &tuples,
            &dataset,
        );
        let naive_winner = naive_emitted.first().map(|(c, _, _)| c.clone()).unwrap_or_default();

        eval_points.push(FlatEvalPoint {
            strategy: "Naive_Baseline".to_string(),
            k: 1,
            alpha: 0.0,
            target_coverage: 1.0,
            tau: 0.0,
            is_valid_guarantee: true,
            is_covered: naive_winner == gt_winner,
            prediction_set: vec![naive_winner.clone()],
            set_size: 1,
            matches_top1: naive_winner == gt_winner,
            items_read: naive_items,
            total_stream_items,
            stream_savings_ratio: 0.0,
            in_engine_ms: naive_ms,
        });

        // 1b. MCDB-1000 Baseline
        let (mcdb_emitted, mcdb_items, mcdb_ms) = run_flat_strategy(
            AggregationMode::MCDBBitvec { n_worlds: 1000 },
            budget,
            1,
            &tuples,
            &dataset,
        );
        let mcdb_winner = mcdb_emitted.first().map(|(c, _, _)| c.clone()).unwrap_or_default();

        eval_points.push(FlatEvalPoint {
            strategy: "MCDB-1000".to_string(),
            k: 1,
            alpha: 0.0,
            target_coverage: 1.0,
            tau: 0.0,
            is_valid_guarantee: true,
            is_covered: mcdb_winner == gt_winner,
            prediction_set: vec![mcdb_winner.clone()],
            set_size: 1,
            matches_top1: mcdb_winner == gt_winner,
            items_read: mcdb_items,
            total_stream_items,
            stream_savings_ratio: 0.0,
            in_engine_ms: mcdb_ms,
        });

        // 2. QN1 (Relative Bound Separation Stopping)
        for &k in &ranks {
            let (qn1_emitted, qn1_items, qn1_ms) = run_flat_strategy(
                AggregationMode::QN1,
                budget,
                k,
                &tuples,
                &dataset,
            );
            let pred_set: Vec<String> = qn1_emitted.iter().map(|(c, _, _)| c.clone()).collect();
            let is_covered = pred_set.contains(&gt_winner);
            let top1_match = pred_set.first() == Some(&gt_winner);
            let savings = 1.0 - (qn1_items as f64 / total_stream_items.max(1) as f64);

            eval_points.push(FlatEvalPoint {
                strategy: "QN1_Relative_Stop".to_string(),
                k,
                alpha: 0.0,
                target_coverage: 1.0,
                tau: 0.0,
                is_valid_guarantee: true,
                is_covered,
                prediction_set: pred_set.clone(),
                set_size: pred_set.len(),
                matches_top1: top1_match,
                items_read: qn1_items,
                total_stream_items,
                stream_savings_ratio: savings.max(0.0),
                in_engine_ms: qn1_ms,
            });
        }

        // 3. Conformal In-Operator Stopping via tau_alpha
        if let Some(strat_map) = calib_matrix.get(&config_name) {
            let strat_key = if dataset.is_non_exclusive { "QN1_Indep" } else { "QN1_MutEx" };
            if let Some(rank_map) = strat_map.get(strat_key) {
                for &k in &ranks {
                    let rank_entry_opt = rank_map.get(&format!("k_{}", k)).or_else(|| rank_map.get(&format!("rank_{}", k)));
                    if let Some(rank_entry) = rank_entry_opt {
                        for &alpha in &alphas {
                            let (tau, is_valid, target_cov) = if (alpha - 0.01f64).abs() < 1e-4 {
                                (rank_entry.alpha_0_01.tau, rank_entry.alpha_0_01.is_valid_guarantee, rank_entry.alpha_0_01.target_coverage)
                            } else if (alpha - 0.05f64).abs() < 1e-4 {
                                (rank_entry.alpha_0_05.tau, rank_entry.alpha_0_05.is_valid_guarantee, rank_entry.alpha_0_05.target_coverage)
                            } else {
                                (rank_entry.alpha_0_10.tau, rank_entry.alpha_0_10.is_valid_guarantee, rank_entry.alpha_0_10.target_coverage)
                            };

                            let (conf_emitted, conf_items, conf_ms) = run_flat_strategy(
                                AggregationMode::ConformalCountPruning { tau },
                                budget,
                                k,
                                &tuples,
                                &dataset,
                            );

                            let pred_set: Vec<String> = conf_emitted.iter().map(|(c, _, _)| c.clone()).collect();
                            let is_covered = pred_set.contains(&gt_winner);
                            let top1_match = pred_set.first() == Some(&gt_winner);
                            let savings = 1.0 - (conf_items as f64 / total_stream_items.max(1) as f64);

                            eval_points.push(FlatEvalPoint {
                                strategy: "Conformal_InOperator_Stop".to_string(),
                                k,
                                alpha,
                                target_coverage: target_cov,
                                tau,
                                is_valid_guarantee: is_valid,
                                is_covered,
                                prediction_set: pred_set.clone(),
                                set_size: pred_set.len(),
                                matches_top1: top1_match,
                                items_read: conf_items,
                                total_stream_items,
                                stream_savings_ratio: savings.max(0.0),
                                in_engine_ms: conf_ms,
                            });
                        }
                    }
                }
            }
        }

        results.push(FlatDatasetEvalResult {
            dataset_file: path.file_name().unwrap().to_str().unwrap().to_string(),
            config_name,
            seed,
            ground_truth_winner: gt_winner,
            eval_points,
        });

        if (f_idx + 1) % log_interval == 0 || f_idx + 1 == files.len() {
            println!("  [Progress] Processed {}/{} datasets ({:.1}s elapsed)",
                f_idx + 1, files.len(), t_start.elapsed().as_secs_f64());
        }
    }

    let out_file = File::create(&out_path)
        .unwrap_or_else(|e| panic!("Failed to create output file {}: {}", out_path, e));
    serde_json::to_writer_pretty(out_file, &results)
        .unwrap_or_else(|e| panic!("Failed to serialize results to {}: {}", out_path, e));

    println!("==========================================================================");
    println!("✓ Flat Conformal Evaluation Complete!");
    println!("  Total Datasets Evaluated: {}", results.len());
    println!("  Output written to: {}", out_path);
    println!("==========================================================================");
}
