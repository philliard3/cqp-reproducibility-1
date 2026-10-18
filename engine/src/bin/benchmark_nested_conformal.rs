/// benchmark_nested_conformal.rs
///
/// Conformal prediction set evaluation binary for Nested Query Q4.
/// Evaluates test datasets across:
/// - 5 strategies: QN1 (MutEx), QN1 (Indep), QPM (MutEx), QPI (MutEx), QPI (Indep)
/// - 3 ranks: k in {1, 2, 3}
/// - 3 significance levels: alpha in {0.01, 0.05, 0.10}
///
/// Evaluates:
/// - Empirical test coverage (guarantee >= 1 - alpha)
/// - Prediction set size |C|
/// - Early-exit wall-clock latency (ms) and items read
/// - Stream savings ratio

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
use mode_u_engine::statistical_estimation::{
    QPEngineType, QPUpdateScheme, DisambiguationCriterion
};

#[derive(Serialize, Deserialize, Debug, Clone)]
struct RawDatasetTuple {
    #[serde(alias = "feed", alias = "thread", alias = "partition_id")]
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
struct GroundTruthInfo {
    naive_outer_winner: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct PreparedDatasetFile {
    total_tuples: usize,
    n_partitions: usize,
    #[serde(default, alias = "n_documents", alias = "n_comments")]
    n_images: usize,
    #[serde(default, alias = "docs_per_partition", alias = "comments_per_partition")]
    images_per_partition: usize,
    #[serde(default, alias = "class_names")]
    animal_classes: Vec<String>,
    ground_truth: GroundTruthInfo,
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
pub struct ConformalEvalPoint {
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
pub struct DatasetConformalEvalResult {
    pub dataset_file: String,
    pub config_name: String,
    pub seed: usize,
    pub ground_truth_winner: String,
    pub eval_points: Vec<ConformalEvalPoint>,
}

fn load_dataset(path: &PathBuf) -> (PreparedDatasetFile, Vec<Arc<Tuple>>) {
    let file = File::open(path).unwrap_or_else(|e| panic!("Failed to open dataset {}: {}", path.display(), e));
    let reader = BufReader::new(file);
    let dataset: PreparedDatasetFile = serde_json::from_reader(reader)
        .unwrap_or_else(|e| panic!("Failed to parse JSON dataset {}: {}", path.display(), e));

    for c in &dataset.animal_classes {
        mode_u_engine::ids::intern(c);
    }
    for d in 0..dataset.n_partitions {
        mode_u_engine::ids::intern(&format!("day_{}", d));
        mode_u_engine::ids::intern(&format!("thread_{}", d));
    }

    let mut tuples = Vec::with_capacity(dataset.tuples.len());
    for raw in &dataset.tuples {
        mode_u_engine::ids::intern(&raw.day);
        mode_u_engine::ids::intern(&raw.image_id);
        mode_u_engine::ids::intern(&raw.animal);

        let mut data = IndexMap::new();
        data.insert("day".to_string(), FieldValue::Str(raw.day.clone()));
        data.insert("image_id".to_string(), FieldValue::Str(raw.image_id.clone()));
        data.insert("animal".to_string(), FieldValue::Str(raw.animal.clone()));

        if let Some(masks) = &raw.mc_bits {
            for (idx, mask) in masks.iter().enumerate() {
                let k = format!("__mc_bits_{}__", idx);
                data.insert(k, FieldValue::Int(*mask as i64));
            }
        }

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

fn run_strategy_k(
    mode: AggregationMode,
    stage1_budget: ItemBudgetSemantics,
    stage2_budget: ItemBudgetSemantics,
    k: usize,
    tuples: &[Arc<Tuple>],
    dataset: &PreparedDatasetFile,
) -> (Vec<(String, f64, f64)>, usize, f64) {
    let part_prefix = dataset.tuples.first()
        .and_then(|t| t.day.split('_').next())
        .unwrap_or("day");
    let mut card_stage1 = FxHashMap::default();
    for d in 0..dataset.n_partitions {
        let p_id = mode_u_engine::ids::intern(&format!("{}_{}", part_prefix, d));
        card_stage1.insert(CompactKey::single(p_id), dataset.images_per_partition);
        let day_id = mode_u_engine::ids::intern(&format!("day_{}", d));
        card_stage1.insert(CompactKey::single(day_id), dataset.images_per_partition);
    }
    let mut card_stage2 = FxHashMap::default();
    card_stage2.insert(CompactKey::empty(), dataset.n_partitions);

    let leaf_stream = Box::new(MemoryTupleStream::new(tuples.to_vec()));

    let stage1_cfg = ArgMaxConfig {
        item_key: vec!["image_id".to_string()],
        count_group_by: vec!["animal".to_string(), "day".to_string()],
        comparison_key: vec!["day".to_string()],
        expected_cardinality: card_stage1,
        budget_semantics: stage1_budget,
        mode: mode.clone(),
        k_target: 1,
    };

    let stage2_cfg = ArgMaxConfig {
        item_key: vec!["day".to_string()],
        count_group_by: vec!["animal".to_string()],
        comparison_key: vec![],
        expected_cardinality: card_stage2,
        budget_semantics: stage2_budget,
        mode: mode.clone(),
        k_target: k,
    };

    let t0 = Instant::now();
    let stage1 = Box::new(ComposableArgMaxOverCount::new(
        leaf_stream,
        stage1_cfg,
        dataset.animal_classes.clone(),
    ));
    let mut stage2 = ComposableArgMaxOverCount::new(
        stage1,
        stage2_cfg,
        dataset.animal_classes.clone(),
    );

    let mut emitted: Vec<(String, f64, f64)> = Vec::new();
    while let Some(tuple) = stage2.next_tuple() {
        let cand = tuple.data.get("animal").unwrap().as_str().unwrap().to_string();
        let lo = tuple.score.to_prob_lo();
        let hi = tuple.score.to_prob_hi();
        emitted.push((cand, lo, hi));
    }
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let items_read = stage2.total_items_read();

    (emitted, items_read, elapsed_ms)
}

fn run_strategy_in_operator_conformal(
    tau: f64,
    stage1_budget: ItemBudgetSemantics,
    tuples: &[Arc<Tuple>],
    dataset: &PreparedDatasetFile,
) -> (Vec<(String, f64, f64)>, usize, f64) {
    let part_prefix = dataset.tuples.first()
        .and_then(|t| t.day.split('_').next())
        .unwrap_or("day");
    let mut card_stage1 = FxHashMap::default();
    for d in 0..dataset.n_partitions {
        let p_id = mode_u_engine::ids::intern(&format!("{}_{}", part_prefix, d));
        card_stage1.insert(CompactKey::single(p_id), dataset.images_per_partition);
        let day_id = mode_u_engine::ids::intern(&format!("day_{}", d));
        card_stage1.insert(CompactKey::single(day_id), dataset.images_per_partition);
    }
    let mut card_stage2 = FxHashMap::default();
    card_stage2.insert(CompactKey::empty(), dataset.n_partitions);

    let leaf_stream = Box::new(MemoryTupleStream::new(tuples.to_vec()));

    let stage1_cfg = ArgMaxConfig {
        item_key: vec!["image_id".to_string()],
        count_group_by: vec!["animal".to_string(), "day".to_string()],
        comparison_key: vec!["day".to_string()],
        expected_cardinality: card_stage1,
        budget_semantics: stage1_budget,
        mode: AggregationMode::QN1,
        k_target: 1,
    };

    let stage2_cfg = ArgMaxConfig {
        item_key: vec!["day".to_string()],
        count_group_by: vec!["animal".to_string()],
        comparison_key: vec![],
        expected_cardinality: card_stage2,
        budget_semantics: ItemBudgetSemantics::Independent,
        mode: AggregationMode::ConformalCountPruning { tau },
        k_target: 1,
    };

    let t0 = Instant::now();
    let stage1 = Box::new(ComposableArgMaxOverCount::new(
        leaf_stream,
        stage1_cfg,
        dataset.animal_classes.clone(),
    ));
    let mut stage2 = ComposableArgMaxOverCount::new(
        stage1,
        stage2_cfg,
        dataset.animal_classes.clone(),
    );

    let mut emitted: Vec<(String, f64, f64)> = Vec::new();
    while let Some(tuple) = stage2.next_tuple() {
        let cand = tuple.data.get("animal").unwrap().as_str().unwrap().to_string();
        let lo = tuple.score.to_prob_lo();
        let hi = tuple.score.to_prob_hi();
        emitted.push((cand, lo, hi));
    }
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let items_read = stage2.total_items_read();

    (emitted, items_read, elapsed_ms)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let dir_arg = args.iter().position(|a| a == "--dir").and_then(|i| args.get(i + 1));
    let th_arg = args.iter().position(|a| a == "--conformal-thresholds").and_then(|i| args.get(i + 1));
    let out_arg = args.iter().position(|a| a == "--output").and_then(|i| args.get(i + 1));
    let log_interval_arg = args.iter().position(|a| a == "--log-interval").and_then(|i| args.get(i + 1)).and_then(|s| s.parse::<usize>().ok());

    let out_path = out_arg
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("nested_q4_conformal_eval_results.json"));

    let dir_path = dir_arg
        .map(PathBuf::from)
        .expect("--dir <directory> is required");

    let th_path = th_arg
        .map(PathBuf::from)
        .expect("--conformal-thresholds <path> is required");

    println!("================================================================================");
    println!("  NESTED Q4 STRATEGY CONFORMAL EVALUATION SUITE");
    println!("  Input Directory: {}", dir_path.display());
    println!("  Output Destination: {}", out_path.display());
    println!("================================================================================");

    // Load calibration thresholds matrix
    let th_file = File::open(&th_path).unwrap_or_else(|e| panic!("Failed to open thresholds {}: {}", th_path.display(), e));
    let calib_matrix: CalibrationMatrix = serde_json::from_reader(BufReader::new(th_file))
        .unwrap_or_else(|e| panic!("Failed to parse thresholds JSON: {}", e));

    let mut files: Vec<PathBuf> = Vec::new();
    if let Ok(entries) = fs::read_dir(&dir_path) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.extension().and_then(|e| e.to_str()) == Some("json") {
                files.push(p);
            }
        }
    }
    files.sort();

    let log_interval = log_interval_arg.unwrap_or_else(|| (files.len() / 20).max(1));

    println!("Found {} evaluation dataset files to process.", files.len());
    println!("Progress Logging Interval: every {} datasets (5% steps)", log_interval);
    let t_start = Instant::now();

    let strategies: Vec<(&str, AggregationMode, ItemBudgetSemantics, ItemBudgetSemantics)> = vec![
        ("Naive", AggregationMode::Naive, ItemBudgetSemantics::Independent, ItemBudgetSemantics::Independent),
        ("MCDB-100", AggregationMode::MCDBBitvec { n_worlds: 100 }, ItemBudgetSemantics::Independent, ItemBudgetSemantics::Independent),
        ("MCDB-1k", AggregationMode::MCDBBitvec { n_worlds: 1000 }, ItemBudgetSemantics::Independent, ItemBudgetSemantics::Independent),
        ("QN1 (MutEx)", AggregationMode::QN1, ItemBudgetSemantics::MutuallyExclusive, ItemBudgetSemantics::Independent),
        ("QN1 (Indep)", AggregationMode::QN1, ItemBudgetSemantics::Independent, ItemBudgetSemantics::Independent),
        (
            "QPM (MutEx, FocalLazy, Either)",
            AggregationMode::QPM {
                engine: QPEngineType::AdaptiveMultiTier,
                update_scheme: QPUpdateScheme::FocalLazy,
                criterion: DisambiguationCriterion::Either,
                alpha: 0.05,
            },
            ItemBudgetSemantics::MutuallyExclusive,
            ItemBudgetSemantics::Independent,
        ),
        (
            "QPI (MutEx, FocalLazy, Either)",
            AggregationMode::QPI {
                engine: QPEngineType::AdaptiveMultiTier,
                update_scheme: QPUpdateScheme::FocalLazy,
                criterion: DisambiguationCriterion::Either,
                alpha: 0.05,
            },
            ItemBudgetSemantics::MutuallyExclusive,
            ItemBudgetSemantics::Independent,
        ),
        (
            "QPI (Indep, FocalLazy, Either)",
            AggregationMode::QPI {
                engine: QPEngineType::AdaptiveMultiTier,
                update_scheme: QPUpdateScheme::FocalLazy,
                criterion: DisambiguationCriterion::Either,
                alpha: 0.05,
            },
            ItemBudgetSemantics::Independent,
            ItemBudgetSemantics::Independent,
        ),
    ];

    let ranks_arg = args.iter().position(|a| a == "--ranks" || a == "--k").and_then(|i| args.get(i + 1));
    let ranks: Vec<usize> = if let Some(r_str) = ranks_arg {
        r_str.split(',').filter_map(|s| s.trim().parse::<usize>().ok()).collect()
    } else {
        vec![1usize, 2, 3]
    };
    let alphas = vec![0.01f64, 0.05, 0.10];

    println!("  Target Evaluation Ranks (k): {:?}", ranks);
    println!("  Target Coverage Levels (1-alpha): {:?}", alphas.iter().map(|a| 1.0 - a).collect::<Vec<_>>());

    let mut dataset_results: Vec<DatasetConformalEvalResult> = Vec::new();

    for (idx, file_path) in files.iter().enumerate() {
        let (config_name, seed) = infer_config_name(file_path);
        let (dataset, tuples) = load_dataset(file_path);
        let gt_winner = dataset.ground_truth.naive_outer_winner.clone();
        let total_stream = dataset.total_tuples;

        let mut eval_points = Vec::new();

        let cfg_thresholds = calib_matrix.get(&config_name);

        let cli_non_exclusive = args.iter().any(|a| a == "--non-exclusive");
        let is_non_exclusive = cli_non_exclusive || dataset.is_non_exclusive;
        let is_lean = args.iter().any(|a| a == "--no-mcdb" || a == "--lean");

        let active_strategies: Vec<(&str, AggregationMode, ItemBudgetSemantics, ItemBudgetSemantics)> = if is_lean {
            if is_non_exclusive {
                vec![
                    ("Naive", AggregationMode::Naive, ItemBudgetSemantics::Independent, ItemBudgetSemantics::Independent),
                    ("QN1 (Indep)", AggregationMode::QN1, ItemBudgetSemantics::Independent, ItemBudgetSemantics::Independent),
                ]
            } else {
                vec![
                    ("Naive", AggregationMode::Naive, ItemBudgetSemantics::Independent, ItemBudgetSemantics::Independent),
                    ("QN1 (MutEx)", AggregationMode::QN1, ItemBudgetSemantics::MutuallyExclusive, ItemBudgetSemantics::Independent),
                    ("QN1 (Indep)", AggregationMode::QN1, ItemBudgetSemantics::Independent, ItemBudgetSemantics::Independent),
                ]
            }
        } else if is_non_exclusive {
            vec![
                ("Naive", AggregationMode::Naive, ItemBudgetSemantics::Independent, ItemBudgetSemantics::Independent),
                ("MCDB-100", AggregationMode::MCDBBitvec { n_worlds: 100 }, ItemBudgetSemantics::Independent, ItemBudgetSemantics::Independent),
                ("MCDB-1k", AggregationMode::MCDBBitvec { n_worlds: 1000 }, ItemBudgetSemantics::Independent, ItemBudgetSemantics::Independent),
                ("QN1 (Indep)", AggregationMode::QN1, ItemBudgetSemantics::Independent, ItemBudgetSemantics::Independent),
                (
                    "QPI (Indep, FocalLazy, Either)",
                    AggregationMode::QPI {
                        engine: QPEngineType::AdaptiveMultiTier,
                        update_scheme: QPUpdateScheme::FocalLazy,
                        criterion: DisambiguationCriterion::Either,
                        alpha: 0.05,
                    },
                    ItemBudgetSemantics::Independent,
                    ItemBudgetSemantics::Independent,
                ),
            ]
        } else {
            strategies.clone()
        };

        for (strat_name, mode, s1_b, s2_b) in &active_strategies {
            let strat_thresholds = cfg_thresholds.and_then(|c| c.get(*strat_name));

            for &k in &ranks {
                let (emitted_cands, items_read, in_engine_ms) = run_strategy_k(
                    mode.clone(),
                    *s1_b,
                    *s2_b,
                    k,
                    &tuples,
                    &dataset,
                );

                let savings = 1.0 - (items_read as f64 / total_stream.max(1) as f64);
                let top1_match = emitted_cands.first().map(|(c, _, _)| c == &gt_winner).unwrap_or(false);

                let k_th_entry = strat_thresholds.and_then(|s| s.get(&format!("k_{}", k)));

                for &alpha in &alphas {
                    let th_entry = match (alpha * 100.0).round() as usize {
                        1 => k_th_entry.map(|e| &e.alpha_0_01),
                        5 => k_th_entry.map(|e| &e.alpha_0_05),
                        10 => k_th_entry.map(|e| &e.alpha_0_10),
                        _ => None,
                    };

                    let tau = th_entry.map(|e| e.tau).unwrap_or(0.0);
                    let is_valid = th_entry.map(|e| e.is_valid_guarantee).unwrap_or(false);

                    let (prediction_set, set_size, is_covered) = if strat_name.starts_with("MCDB") {
                        let p_set: Vec<String> = emitted_cands.iter().map(|(c, _, _)| c.clone()).collect();
                        let cov = p_set.contains(&gt_winner);
                        let sz = p_set.len();
                        (p_set, sz, cov)
                    } else {
                        let p_set: Vec<String> = emitted_cands.iter()
                            .filter(|(_, lo, _)| *lo >= tau - 1e-9)
                            .map(|(c, _, _)| c.clone())
                            .collect();
                        let cov = p_set.contains(&gt_winner);
                        let sz = p_set.len();
                        (p_set, sz, cov)
                    };

                    eval_points.push(ConformalEvalPoint {
                        strategy: strat_name.to_string(),
                        k,
                        alpha,
                        target_coverage: 1.0 - alpha,
                        tau,
                        is_valid_guarantee: is_valid,
                        is_covered,
                        prediction_set,
                        set_size,
                        matches_top1: top1_match,
                        items_read,
                        total_stream_items: total_stream,
                        stream_savings_ratio: savings,
                        in_engine_ms,
                    });
                }
            }
        }

        // In-Operator Conformal Stop
        let in_op_combos: Vec<(&str, &str, ItemBudgetSemantics)> = if is_non_exclusive {
            vec![
                ("Conformal QN1 (Indep, In-Operator Stop)", "QN1 (Indep)", ItemBudgetSemantics::Independent),
            ]
        } else {
            vec![
                ("Conformal QN1 (MutEx, In-Operator Stop)", "QN1 (MutEx)", ItemBudgetSemantics::MutuallyExclusive),
                ("Conformal QN1 (Indep, In-Operator Stop)", "QN1 (Indep)", ItemBudgetSemantics::Independent),
            ]
        };

        for (strat_label, th_key, s1_budget) in in_op_combos {
            let qn1_th = cfg_thresholds.and_then(|c| c.get(th_key));
            for &k in &ranks {
                let k_th_entry = qn1_th.and_then(|s| s.get(&format!("k_{}", k)));
                for &alpha in &alphas {
                    let th_entry = match (alpha * 100.0).round() as usize {
                        1 => k_th_entry.map(|e| &e.alpha_0_01),
                        5 => k_th_entry.map(|e| &e.alpha_0_05),
                        10 => k_th_entry.map(|e| &e.alpha_0_10),
                        _ => None,
                    };
                    let tau = th_entry.map(|e| e.tau).unwrap_or(0.0);
                    let is_valid = th_entry.map(|e| e.is_valid_guarantee).unwrap_or(false);

                    let (emitted_cands, items_read, in_engine_ms) = run_strategy_in_operator_conformal(
                        tau,
                        s1_budget,
                        &tuples,
                        &dataset,
                    );
                    let savings = 1.0 - (items_read as f64 / total_stream.max(1) as f64);
                    let top1_match = emitted_cands.first().map(|(c, _, _)| c == &gt_winner).unwrap_or(false);
                    let prediction_set: Vec<String> = emitted_cands.iter().map(|(c, _, _)| c.clone()).collect();
                    let is_covered = prediction_set.contains(&gt_winner);
                    let set_size = prediction_set.len();

                    eval_points.push(ConformalEvalPoint {
                        strategy: strat_label.to_string(),
                        k,
                        alpha,
                        target_coverage: 1.0 - alpha,
                        tau,
                        is_valid_guarantee: is_valid,
                        is_covered,
                        prediction_set,
                        set_size,
                        matches_top1: top1_match,
                        items_read,
                        total_stream_items: total_stream,
                        stream_savings_ratio: savings,
                        in_engine_ms,
                    });
                }
            }
        }

        dataset_results.push(DatasetConformalEvalResult {
            dataset_file: file_path.to_string_lossy().to_string(),
            config_name: config_name.clone(),
            seed,
            ground_truth_winner: gt_winner,
            eval_points,
        });

        if (idx + 1) % log_interval == 0 || (idx + 1) == files.len() {
            let elapsed_sec = t_start.elapsed().as_secs_f64();
            let rate = (idx + 1) as f64 / elapsed_sec.max(0.001);
            println!(
                "  [Eval Progress] Evaluated {:4}/{} datasets ({:5.1}%) | Elapsed: {:.1}s | Rate: {:.1} ds/s (Config: {})",
                idx + 1,
                files.len(),
                ((idx + 1) as f64 / files.len() as f64) * 100.0,
                elapsed_sec,
                rate,
                config_name,
            );
        }
    }

    if let Some(parent) = out_path.parent() {
        let _ = fs::create_dir_all(parent);
    }

    let out_file = File::create(&out_path).unwrap_or_else(|e| panic!("Failed to create {}: {}", out_path.display(), e));
    serde_json::to_writer_pretty(out_file, &dataset_results).expect("Failed to write conformal evaluation JSON output");

    println!("================================================================================");
    println!("✓ Conformal evaluation completed in {:.2}s across {} datasets.", t_start.elapsed().as_secs_f64(), dataset_results.len());
    let total_points: usize = dataset_results.iter().map(|d| d.eval_points.len()).sum();
    println!("  Total evaluated conformal points: {}", total_points);
    println!("  Output saved to: {}", out_path.display());
    println!("================================================================================");
}
