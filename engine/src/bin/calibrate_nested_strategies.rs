/// calibrate_nested_strategies.rs
///
/// Runs calibration datasets across the 5 bounded/probabilistic strategies:
/// 1. QN1 (MutEx)
/// 2. QN1 (Indep)
/// 3. QPM (MutEx, FocalLazy, Either)
/// 4. QPI (MutEx, FocalLazy, Either)
/// 5. QPI (Indep, FocalLazy, Either)
///
/// Executes a single pass up to top-3 (plus ties) per seed and extracts snapshots
/// of candidate lower bounds at rank 1, rank 2, and rank 3.
///
/// Supports periodic logging (e.g. every 25 seeds) so progress is clearly visible.

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
pub struct CandidateSnapshot {
    pub candidate: String,
    pub score_lo: f64,
    pub score_hi: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct StrategyRankSnapshots {
    pub strategy: String,
    pub rank_1: Vec<CandidateSnapshot>,
    pub rank_2: Vec<CandidateSnapshot>,
    pub rank_3: Vec<CandidateSnapshot>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SeedCalibrationRecord {
    pub dataset_file: String,
    pub config_name: String,
    pub seed: usize,
    pub ground_truth_winner: String,
    pub strategy_results: Vec<StrategyRankSnapshots>,
}

fn load_dataset(path: &PathBuf) -> (PreparedDatasetFile, Vec<Arc<Tuple>>) {
    let file = File::open(path).unwrap_or_else(|e| panic!("Failed to open dataset {}: {}", path.display(), e));
    let reader = BufReader::new(file);
    let dataset: PreparedDatasetFile = serde_json::from_reader(reader)
        .unwrap_or_else(|e| panic!("Failed to parse JSON dataset {}: {}", path.display(), e));

    let mut tuples = Vec::with_capacity(dataset.tuples.len());
    for raw in &dataset.tuples {
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

fn run_single_calibration_strategy(
    strategy_name: &str,
    mode: AggregationMode,
    stage1_budget: ItemBudgetSemantics,
    stage2_budget: ItemBudgetSemantics,
    tuples: &[Arc<Tuple>],
    dataset: &PreparedDatasetFile,
) -> StrategyRankSnapshots {
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
        k_target: 3, // Target top 3 plus ties
    };

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

    let mut emitted_all: Vec<CandidateSnapshot> = Vec::new();
    while let Some(tuple) = stage2.next_tuple() {
        let cand = tuple.data.get("animal").unwrap().as_str().unwrap().to_string();
        let lo = tuple.score.to_prob_lo();
        let hi = tuple.score.to_prob_hi();
        emitted_all.push(CandidateSnapshot {
            candidate: cand,
            score_lo: lo,
            score_hi: hi,
        });
    }

    // Build rank snapshots:
    // rank_1: top-1 plus any candidates tied with top-1
    let mut rank_1 = Vec::new();
    if !emitted_all.is_empty() {
        let top_lo = emitted_all[0].score_lo;
        for c in &emitted_all {
            if (c.score_lo - top_lo).abs() < 1e-9 || rank_1.is_empty() {
                rank_1.push(c.clone());
            } else {
                break;
            }
        }
    }

    // rank_2: top-2 plus any candidates tied with 2nd place
    let mut rank_2 = Vec::new();
    if !emitted_all.is_empty() {
        let limit = 2.min(emitted_all.len());
        let threshold_lo = emitted_all[limit - 1].score_lo;
        for (idx, c) in emitted_all.iter().enumerate() {
            if idx < 2 || (c.score_lo - threshold_lo).abs() < 1e-9 {
                rank_2.push(c.clone());
            } else {
                break;
            }
        }
    }

    // rank_3: all emitted (which contains top 3 plus any tied with 3rd place)
    let rank_3 = emitted_all;

    StrategyRankSnapshots {
        strategy: strategy_name.to_string(),
        rank_1,
        rank_2,
        rank_3,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let dir_arg = args.iter().position(|a| a == "--dir").and_then(|i| args.get(i + 1));
    let out_arg = args.iter().position(|a| a == "--output").and_then(|i| args.get(i + 1));
    let log_interval_arg = args.iter().position(|a| a == "--log-interval").and_then(|i| args.get(i + 1)).and_then(|s| s.parse::<usize>().ok());

    let out_path = out_arg
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("nested_q4_raw_calibration_snapshots.json"));

    let dir_path = dir_arg
        .map(PathBuf::from)
        .expect("--dir <directory> is required");

    println!("================================================================================");
    println!("  NESTED Q4 STRATEGY CONFORMAL CALIBRATION DATA GATHERER");
    println!("  Input Directory: {}", dir_path.display());
    println!("  Output Destination: {}", out_path.display());
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

    println!("Found {} calibration dataset files to process.", files.len());
    println!("Progress Logging Interval: every {} seeds (5% steps)", log_interval);
    let t_start = Instant::now();

    let mut records: Vec<SeedCalibrationRecord> = Vec::new();

    for (idx, file_path) in files.iter().enumerate() {
        let (config_name, seed) = infer_config_name(file_path);
        let (dataset, tuples) = load_dataset(file_path);
        let gt_winner = dataset.ground_truth.naive_outer_winner.clone();

        let cli_non_exclusive = args.iter().any(|a| a == "--non-exclusive");
        let is_non_exclusive = cli_non_exclusive || dataset.is_non_exclusive;

        let strategies: Vec<(&str, AggregationMode, ItemBudgetSemantics, ItemBudgetSemantics)> = if is_non_exclusive {
            vec![
                ("Naive", AggregationMode::Naive, ItemBudgetSemantics::Independent, ItemBudgetSemantics::Independent),
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
            vec![
                ("Naive", AggregationMode::Naive, ItemBudgetSemantics::Independent, ItemBudgetSemantics::Independent),
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
            ]
        };

        let mut strategy_results = Vec::with_capacity(strategies.len());
        for (name, mode, s1_b, s2_b) in strategies {
            let res = run_single_calibration_strategy(name, mode, s1_b, s2_b, &tuples, &dataset);
            strategy_results.push(res);
        }

        records.push(SeedCalibrationRecord {
            dataset_file: file_path.to_string_lossy().to_string(),
            config_name: config_name.clone(),
            seed,
            ground_truth_winner: gt_winner,
            strategy_results,
        });

        if (idx + 1) % log_interval == 0 || (idx + 1) == files.len() {
            let elapsed_sec = t_start.elapsed().as_secs_f64();
            let rate = (idx + 1) as f64 / elapsed_sec.max(0.001);
            println!(
                "  [Calibration Progress] Processed {:4}/{} seeds ({:5.1}%) | Elapsed: {:.1}s | Rate: {:.1} seeds/s (Config: {})",
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
    serde_json::to_writer_pretty(out_file, &records).expect("Failed to write calibration JSON output");

    println!("================================================================================");
    println!("✓ Calibration completed in {:.2}s across {} seeds.", t_start.elapsed().as_secs_f64(), records.len());
    println!("  Output saved to: {}", out_path.display());
    println!("================================================================================");
}
