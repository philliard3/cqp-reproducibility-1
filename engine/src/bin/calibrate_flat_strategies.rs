/// calibrate_flat_strategies.rs
///
/// Runs calibration datasets across flat Composable ArgMax-over-Count strategies:
/// 1. QN1 (MutEx) [for exclusive datasets like COCO]
/// 2. QN1 (Indep) [for non-exclusive datasets like Jigsaw, and ablation on COCO]
///
/// Executes single pass up to top-3 per calibration seed and extracts snapshots
/// of candidate lower bounds at rank 1, rank 2, and rank 3.
/// Outputs calibration snapshot JSON directly compatible with generate_strategy_conformal_thresholds.py.

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

fn compute_flat_ground_truth(dataset: &PreparedDatasetFile) -> String {
    if let Some(fgt) = &dataset.flat_ground_truth {
        if let Some(w) = &fgt.naive_winner {
            return w.clone();
        }
    }

    let mut scores: FxHashMap<String, f64> = FxHashMap::default();
    for t in &dataset.tuples {
        *scores.entry(t.animal.clone()).or_insert(0.0) += t.score;
    }

    let mut sorted: Vec<(String, f64)> = scores.iter().map(|(k, v)| (k.clone(), *v)).collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    sorted.first().map(|(k, _)| k.clone()).unwrap_or_else(|| "none".to_string())
}

fn run_flat_calibration_strategy(
    strategy_name: &str,
    mode: AggregationMode,
    budget: ItemBudgetSemantics,
    tuples: &[Arc<Tuple>],
    dataset: &PreparedDatasetFile,
) -> StrategyRankSnapshots {
    let mut card = FxHashMap::default();
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
        k_target: 3,
    };

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

    let mut rank_1 = Vec::new();
    let mut rank_2 = Vec::new();
    let mut rank_3 = Vec::new();

    for (i, (c, lo, hi)) in emitted.iter().enumerate() {
        let snap = CandidateSnapshot {
            candidate: c.clone(),
            score_lo: *lo,
            score_hi: *hi,
        };
        if i == 0 {
            rank_1.push(snap.clone());
        }
        if i <= 1 {
            rank_2.push(snap.clone());
        }
        if i <= 2 {
            rank_3.push(snap);
        }
    }

    StrategyRankSnapshots {
        strategy: strategy_name.to_string(),
        rank_1,
        rank_2,
        rank_3,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut dir_path = PathBuf::from("mode_u_engine/real_data_prepared/flat_q2/calib_100_seeds");
    let mut out_path = PathBuf::from("flat_raw_calibration_snapshots.json");
    let mut log_interval = 25;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--dir" => { dir_path = PathBuf::from(&args[i + 1]); i += 2; }
            "--output" => { out_path = PathBuf::from(&args[i + 1]); i += 2; }
            "--log-interval" => { log_interval = args[i + 1].parse().unwrap_or(25); i += 2; }
            _ => { i += 1; }
        }
    }

    println!("================================================================================");
    println!("  CALIBRATE FLAT ARGMAX-OVER-COUNT STRATEGIES                                   ");
    println!("  Calibration Directory: {}", dir_path.display());
    println!("  Output Snapshot JSON:  {}", out_path.display());
    println!("================================================================================");

    let mut files: Vec<PathBuf> = fs::read_dir(&dir_path)
        .unwrap_or_else(|e| panic!("Failed to read dir {}: {}", dir_path.display(), e))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|ext| ext == "json").unwrap_or(false))
        .collect();
    files.sort();

    println!("Found {} calibration dataset files.", files.len());
    let t_start = Instant::now();
    let mut records: Vec<SeedCalibrationRecord> = Vec::with_capacity(files.len());

    for (idx, file_path) in files.iter().enumerate() {
        let (config_name, seed) = infer_config_name(file_path);
        let (dataset, tuples) = load_dataset(file_path);
        let gt_winner = compute_flat_ground_truth(&dataset);

        let strategies: Vec<(&str, AggregationMode, ItemBudgetSemantics)> = if dataset.is_non_exclusive {
            vec![
                ("QN1 (Indep)", AggregationMode::QN1, ItemBudgetSemantics::Independent),
            ]
        } else {
            vec![
                ("QN1 (MutEx)", AggregationMode::QN1, ItemBudgetSemantics::MutuallyExclusive),
                ("QN1 (Indep)", AggregationMode::QN1, ItemBudgetSemantics::Independent),
            ]
        };

        let mut strategy_results = Vec::with_capacity(strategies.len());
        for (name, mode, budget) in strategies {
            let res = run_flat_calibration_strategy(name, mode, budget, &tuples, &dataset);
            strategy_results.push(res);
        }

        records.push(SeedCalibrationRecord {
            dataset_file: file_path.file_name().unwrap().to_str().unwrap().to_string(),
            config_name,
            seed,
            ground_truth_winner: gt_winner,
            strategy_results,
        });

        if (idx + 1) % log_interval == 0 || (idx + 1) == files.len() {
            println!("  [Calibration Progress] Processed {}/{} seeds ({:.1}s elapsed)",
                idx + 1, files.len(), t_start.elapsed().as_secs_f64());
        }
    }

    if let Some(parent) = out_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let out_file = File::create(&out_path).unwrap_or_else(|e| panic!("Failed to create {}: {}", out_path.display(), e));
    serde_json::to_writer_pretty(out_file, &records).expect("Failed to write calibration JSON");

    println!("================================================================================");
    println!("✓ Flat Calibration Complete in {:.2}s! ({} records)", t_start.elapsed().as_secs_f64(), records.len());
    println!("  Output saved to: {}", out_path.display());
    println!("================================================================================");
}
