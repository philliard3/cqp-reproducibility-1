/// unified_argmax.rs
///
/// Composable, schema-agnostic ArgMaxOverCount operator for Probabilistic Database Queries.
/// Operates directly on standard `Tuple` throughout the pipeline.
///
/// Features:
///   - Arc<Tuple> zero-copy streaming: Eliminates deep cloning of IndexMaps and strings.
///   - CompactKey string interning: Uses 32-bit interned IDs for single-cycle integer hashing.
///   - O(1) incremental bounds: Maintains running budget scalars to eliminate historical rescans.
///   - Correct baseline semantics: Naive and MCDB strictly run to full stream exhaustion.
///   - Sound majority early-exit: Correctly uses remaining unvisited count (N - seen) for bounds.

use std::sync::Arc;
use rustc_hash::{FxHashMap, FxHashSet};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::tuple::{Tuple, FieldValue, ProbInterval};
use crate::statistical_estimation::{
    QPConfig, QPPartitionTracker, QPEngineType, QPUpdateScheme, DisambiguationCriterion
};

/// 16-byte zero-allocation compact key for up to 3 interned string/integer fields.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CompactKey {
    pub len: u8,
    pub ids: [u32; 3],
}

impl CompactKey {
    #[inline(always)]
    pub fn empty() -> Self {
        CompactKey { len: 0, ids: [0; 3] }
    }

    #[inline(always)]
    pub fn single(id: u32) -> Self {
        CompactKey { len: 1, ids: [id, 0, 0] }
    }

    #[inline(always)]
    pub fn pair(id0: u32, id1: u32) -> Self {
        CompactKey { len: 2, ids: [id0, id1, 0] }
    }

    pub fn from_fields(fields: &[FieldValue]) -> Self {
        let mut ids = [0u32; 3];
        let len = fields.len().min(3) as u8;
        for (i, f) in fields.iter().take(3).enumerate() {
            ids[i] = f.interned_id();
        }
        CompactKey { len, ids }
    }

    #[inline(always)]
    pub fn extract(data: &IndexMap<String, FieldValue>, keys: &[String]) -> Self {
        let mut ids = [0u32; 3];
        let len = keys.len().min(3) as u8;
        for (i, k) in keys.iter().take(3).enumerate() {
            if let Some(v) = data.get(k) {
                ids[i] = v.interned_id();
            }
        }
        CompactKey { len, ids }
    }
}

/// Universal trait for Volcano iterator streaming of immutable `Tuple`s wrapped in `Arc`.
pub trait TupleStream: Send {
    fn next_tuple(&mut self) -> Option<Arc<Tuple>>;
    fn total_items_read(&self) -> usize { 0 }
}

/// In-memory stream adapter for `Vec<Arc<Tuple>>`.
pub struct MemoryTupleStream {
    pub tuples: Vec<Arc<Tuple>>,
    pub cursor: usize,
}

impl MemoryTupleStream {
    pub fn new(mut tuples: Vec<Arc<Tuple>>) -> Self {
        // Ensure stream is sorted by upper bound score descending for optimal pruning
        tuples.sort_by(|a, b| {
            b.score.to_prob_hi()
                .partial_cmp(&a.score.to_prob_hi())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        MemoryTupleStream { tuples, cursor: 0 }
    }

    pub fn new_presorted(tuples: Vec<Arc<Tuple>>) -> Self {
        MemoryTupleStream { tuples, cursor: 0 }
    }
}

impl TupleStream for MemoryTupleStream {
    fn next_tuple(&mut self) -> Option<Arc<Tuple>> {
        if self.cursor < self.tuples.len() {
            let t = Arc::clone(&self.tuples[self.cursor]);
            self.cursor += 1;
            Some(t)
        } else {
            None
        }
    }

    fn total_items_read(&self) -> usize {
        self.cursor
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ItemBudgetSemantics {
    MutuallyExclusive,
    Independent,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum AggregationMode {
    Naive,
    MCDBBitvec { n_worlds: usize },
    R1V2,
    QN1,
    QPM {
        engine: QPEngineType,
        update_scheme: QPUpdateScheme,
        criterion: DisambiguationCriterion,
        alpha: f64,
    },
    QPI {
        engine: QPEngineType,
        update_scheme: QPUpdateScheme,
        criterion: DisambiguationCriterion,
        alpha: f64,
    },
    ConformalCountPruning { tau: f64 },
}

#[derive(Clone, Debug)]
pub struct ArgMaxConfig {
    /// Column names identifying distinct items being counted:
    /// e.g. Stage 1: ["image_id"], Stage 2: ["day"]
    pub item_key: Vec<String>,

    /// Column names identifying the candidate groups being aggregated:
    /// e.g. Stage 1: ["animal", "day"], Stage 2: ["animal"]
    pub count_group_by: Vec<String>,

    /// Column names defining the comparison partition scope:
    /// e.g. Stage 1: ["day"], Stage 2: [] (empty = global comparison)
    pub comparison_key: Vec<String>,

    /// Expected total items per comparison key (for unseen future mass bounds)
    pub expected_cardinality: FxHashMap<CompactKey, usize>,

    /// Independent axis 1: item exclusivity / budget semantics
    pub budget_semantics: ItemBudgetSemantics,

    /// Independent axis 2: execution strategy / score estimation mode
    pub mode: AggregationMode,

    /// Target number of top ranks to emit per comparison partition (default 1)
    pub k_target: usize,
}

#[derive(Debug, Clone)]
pub struct CandidateStats {
    pub sum_lo: f64,
    pub sum_hi: f64,
    pub seen_items: FxHashSet<CompactKey>,
    pub world_counts: Vec<u32>, // Used by MCDB
}

impl CandidateStats {
    pub fn new(n_worlds: usize) -> Self {
        CandidateStats {
            sum_lo: 0.0,
            sum_hi: 0.0,
            seen_items: FxHashSet::default(),
            world_counts: vec![0; n_worlds],
        }
    }
}

#[derive(Debug)]
pub struct ComparisonPartitionState {
    pub total_items: usize,
    pub seen_items: FxHashSet<CompactKey>,
    pub candidates: FxHashMap<CompactKey, CandidateStats>,

    // Incremental O(1) budget tracking
    pub total_seen_budget: f64,
    pub cand_consumed_budget: FxHashMap<CompactKey, f64>,
    pub item_budget: FxHashMap<CompactKey, f64>,

    pub progress: FxHashMap<CompactKey, f64>,
    pub p_max_next: f64,
    pub p_max_next_per_candidate: FxHashMap<CompactKey, f64>,
    pub emitted_candidates: FxHashSet<CompactKey>,
    pub last_emitted_interval: Option<(f64, f64)>,
    pub qp_tracker: Option<QPPartitionTracker>,
    pub is_resolved: bool,
    pub cand_bounds: Vec<(CompactKey, f64, f64)>,
}

impl ComparisonPartitionState {
    pub fn new(
        total_items: usize,
        mode: &AggregationMode,
        budget_semantics: ItemBudgetSemantics,
        all_candidate_names: &[String],
    ) -> Self {
        let qp_tracker = match mode {
            AggregationMode::QPM { engine, update_scheme, criterion, .. } => {
                let cfg = QPConfig {
                    engine_type: *engine,
                    update_scheme: *update_scheme,
                    disambiguation_criterion: *criterion,
                    candidate_k: all_candidate_names.len(),
                    threshold_delta: 0.05,
                    c_truncation: 1e-4,
                };
                let use_mex = budget_semantics == ItemBudgetSemantics::MutuallyExclusive;
                Some(QPPartitionTracker::new(
                    all_candidate_names.to_vec(),
                    total_items,
                    use_mex,
                    Some(cfg),
                ))
            }
            AggregationMode::QPI { engine, update_scheme, criterion, .. } => {
                let cfg = QPConfig {
                    engine_type: *engine,
                    update_scheme: *update_scheme,
                    disambiguation_criterion: *criterion,
                    candidate_k: all_candidate_names.len(),
                    threshold_delta: 0.05,
                    c_truncation: 1e-4,
                };
                Some(QPPartitionTracker::new(
                    all_candidate_names.to_vec(),
                    total_items,
                    false, // Independent marginals
                    Some(cfg),
                ))
            }
            _ => None,
        };

        ComparisonPartitionState {
            total_items,
            seen_items: FxHashSet::default(),
            candidates: FxHashMap::default(),
            total_seen_budget: 0.0,
            cand_consumed_budget: FxHashMap::default(),
            item_budget: FxHashMap::default(),
            progress: FxHashMap::default(),
            p_max_next: 1.0,
            p_max_next_per_candidate: FxHashMap::default(),
            emitted_candidates: FxHashSet::default(),
            last_emitted_interval: None,
            qp_tracker,
            is_resolved: false,
            cand_bounds: Vec::new(),
        }
    }
}

pub struct ComposableArgMaxOverCount {
    pub child: Box<dyn TupleStream>,
    pub config: ArgMaxConfig,
    pub partitions: FxHashMap<CompactKey, ComparisonPartitionState>,
    pub holding_queue: Vec<Arc<Tuple>>,
    pub items_read: usize,
    pub current_leaf_p_max: f64,
    pub is_child_exhausted: bool,
    pub all_candidate_names: Vec<String>,
}

impl ComposableArgMaxOverCount {
    pub fn new(
        child: Box<dyn TupleStream>,
        config: ArgMaxConfig,
        all_candidate_names: Vec<String>,
    ) -> Self {
        ComposableArgMaxOverCount {
            child,
            config,
            partitions: FxHashMap::default(),
            holding_queue: Vec::new(),
            items_read: 0,
            current_leaf_p_max: 1.0,
            is_child_exhausted: false,
            all_candidate_names,
        }
    }

    fn get_or_create_partition(&mut self, comp_key: &CompactKey) -> &mut ComparisonPartitionState {
        let default_expected = self.config.expected_cardinality.values().next().cloned().unwrap_or(1);
        let expected = self.config.expected_cardinality
            .get(comp_key)
            .cloned()
            .unwrap_or(default_expected);

        let mode = self.config.mode.clone();
        let budget_sem = self.config.budget_semantics;
        let candidate_names = self.all_candidate_names.clone();

        self.partitions
            .entry(*comp_key)
            .or_insert_with(|| ComparisonPartitionState::new(expected, &mode, budget_sem, &candidate_names))
    }

    fn process_tuple(&mut self, tuple: &Tuple) {
        self.items_read += 1;
        let comp_key = CompactKey::extract(&tuple.data, &self.config.comparison_key);
        let item_key = CompactKey::extract(&tuple.data, &self.config.item_key);
        let count_group = CompactKey::extract(&tuple.data, &self.config.count_group_by);

        let p_lo = tuple.score.to_prob_lo();
        let p_hi = tuple.score.to_prob_hi();
        self.current_leaf_p_max = p_hi;

        let n_worlds = match self.config.mode {
            AggregationMode::MCDBBitvec { n_worlds } => n_worlds,
            _ => 0,
        };

        let is_mut_ex = self.config.budget_semantics == ItemBudgetSemantics::MutuallyExclusive;
        let state = self.get_or_create_partition(&comp_key);

        let is_new_item = state.seen_items.insert(item_key);
        state.p_max_next = p_hi;
        state.p_max_next_per_candidate.insert(count_group, p_hi);
        state.progress.insert(item_key, p_hi);

        if is_mut_ex {
            if is_new_item {
                state.total_seen_budget += 1.0;
                state.item_budget.insert(item_key, 1.0);
            }

            // Mutual exclusion budget deduction
            let budget_entry = state.item_budget.entry(item_key).or_insert(1.0);
            let old_b = *budget_entry;
            let new_b = (old_b - p_lo).max(0.0);
            let delta = old_b - new_b;
            *budget_entry = new_b;
            state.total_seen_budget = (state.total_seen_budget - delta).max(0.0);

            let cand_stats = state.candidates
                .entry(count_group)
                .or_insert_with(|| CandidateStats::new(n_worlds));

            cand_stats.sum_lo += p_lo;
            cand_stats.sum_hi += p_hi;
            let is_new_for_cand = cand_stats.seen_items.insert(item_key);
            if is_new_for_cand {
                *state.cand_consumed_budget.entry(count_group).or_default() += new_b;
            } else if let Some(b) = state.cand_consumed_budget.get_mut(&count_group) {
                *b = (*b - delta).max(0.0);
            }
        } else {
            // Independent: no item budget tracking or deductions
            let cand_stats = state.candidates
                .entry(count_group)
                .or_insert_with(|| CandidateStats::new(n_worlds));

            cand_stats.sum_lo += p_lo;
            cand_stats.sum_hi += p_hi;
            cand_stats.seen_items.insert(item_key);
        }

        // Process MCDB Bitvector
        if n_worlds > 0 {
            let n_words = (n_worlds + 63) / 64;
            for word_idx in 0..n_words {
                let key = format!("__mc_bits_{}__", word_idx);
                let mask = tuple.data.get(&key)
                    .and_then(|v| v.as_f64().map(|f| f as i64).or_else(|| match v { FieldValue::Int(i) => Some(*i), _ => None }))
                    .unwrap_or(0);

                if mask != 0 {
                    for bit in 0..64 {
                        let w = word_idx * 64 + bit;
                        if w < n_worlds && ((mask >> bit) & 1) != 0 {
                            state.candidates.get_mut(&count_group).unwrap().world_counts[w] += 1;
                        }
                    }
                }
            }
        }

        // Process QPM / QPI tracker
        if let Some(tracker) = &mut state.qp_tracker {
            let img_str = if item_key.len > 0 {
                crate::ids::lookup(item_key.ids[0])
            } else {
                "?".to_string()
            };
            let class_str = if count_group.len > 0 {
                crate::ids::lookup(count_group.ids[0])
            } else {
                "?".to_string()
            };

            tracker.ingest_tuple(&img_str, &class_str, p_hi);
        }

        // Check for early disambiguation
        self.check_partition_convergence(&comp_key);
    }

    fn check_partition_convergence(&mut self, comp_key: &CompactKey) {
        let state = match self.partitions.get_mut(comp_key) {
            Some(s) => s,
            None => return,
        };

        if state.is_resolved {
            return;
        }

        match &self.config.mode {
            AggregationMode::Naive => {
                // Naive never terminates early: strictly consumes until stream exhaustion.
            }
            AggregationMode::MCDBBitvec { .. } => {
                // MCDB runs to full stream exhaustion.
            }
            AggregationMode::R1V2 => {
                self.check_r1v2_convergence(comp_key);
            }
            AggregationMode::QN1 => {
                self.check_qn1_convergence(comp_key);
            }
            AggregationMode::ConformalCountPruning { tau } => {
                self.check_conformal_count_convergence(comp_key, *tau);
            }
            AggregationMode::QPM { .. } | AggregationMode::QPI { .. } => {
                self.check_qpm_convergence(comp_key);
            }
        }
    }

    fn check_r1v2_convergence(&mut self, comp_key: &CompactKey) {
        let is_mut_ex = self.config.budget_semantics == ItemBudgetSemantics::MutuallyExclusive;
        let state = self.partitions.get_mut(comp_key).unwrap();
        let unvisited = state.total_items.saturating_sub(state.seen_items.len()) as f64;

        state.cand_bounds.clear();
        for (&cand, stats) in &state.candidates {
            if state.emitted_candidates.contains(&cand) {
                continue;
            }
            let lo = stats.sum_lo;
            let p_max = state.p_max_next_per_candidate.get(&cand).cloned().unwrap_or(state.p_max_next);
            
            let n_seen_unvisited = state.seen_items.len().saturating_sub(stats.seen_items.len()) as f64;
            let mass_by_pmax = n_seen_unvisited * p_max;

            let fm_known = if is_mut_ex {
                let cand_seen_b = state.cand_consumed_budget.get(&cand).cloned().unwrap_or(0.0);
                let avail_budget = (state.total_seen_budget - cand_seen_b).max(0.0);
                mass_by_pmax.min(avail_budget)
            } else {
                mass_by_pmax
            };

            let fm_unknown = unvisited * p_max;
            let hi = lo + fm_known + fm_unknown;
            state.cand_bounds.push((cand, lo, hi));
        }

        if state.cand_bounds.is_empty() {
            return;
        }

        // Sort by lower bound descending
        state.cand_bounds.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let (leader_key, leader_lo, _) = state.cand_bounds[0];

        let mut max_comp_upper = 0.0f64;
        for (_, _, hi) in &state.cand_bounds[1..] {
            if *hi > max_comp_upper {
                max_comp_upper = *hi;
            }
        }
        let unseen_upper = state.p_max_next * unvisited;
        let effective_comp_upper = max_comp_upper.max(unseen_upper);

        let mut emit_keys = Vec::new();
        if leader_lo > effective_comp_upper {
            state.emitted_candidates.insert(leader_key);
            if state.emitted_candidates.len() >= self.config.k_target {
                state.is_resolved = true;
            }
            emit_keys.push(leader_key);
        } else if state.seen_items.len() >= state.total_items && leader_lo >= max_comp_upper {
            // Stream completed: emit remaining leaders up to k_target plus ties
            let needed = self.config.k_target.saturating_sub(state.emitted_candidates.len());
            let mut remaining: Vec<(CompactKey, f64)> = state.cand_bounds.iter()
                .filter(|(k, _, _)| !state.emitted_candidates.contains(k))
                .map(|(k, lo, _)| (*k, *lo))
                .collect();
            remaining.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            if !remaining.is_empty() {
                let limit = needed.min(remaining.len());
                let threshold_lo = remaining[limit - 1].1;
                for (cand, lo) in remaining {
                    if state.emitted_candidates.len() < self.config.k_target || (lo - threshold_lo).abs() < 1e-9 {
                        state.emitted_candidates.insert(cand);
                        emit_keys.push(cand);
                    } else {
                        break;
                    }
                }
            }
            state.is_resolved = true;
        }

        for cand in emit_keys {
            let tuple = Arc::new(self.build_output_tuple(cand, *comp_key, (1.0, 1.0), None));
            self.holding_queue.push(tuple);
        }
    }

    fn check_qn1_convergence(&mut self, comp_key: &CompactKey) {
        let is_mut_ex = self.config.budget_semantics == ItemBudgetSemantics::MutuallyExclusive;
        let state = self.partitions.get_mut(comp_key).unwrap();
        let n_tot = state.total_items.max(1) as f64;
        let unvisited = state.total_items.saturating_sub(state.seen_items.len()) as f64;

        state.cand_bounds.clear();
        for (&cand, stats) in &state.candidates {
            if state.emitted_candidates.contains(&cand) {
                continue;
            }
            let lo = stats.sum_lo;
            let p_max = state.p_max_next_per_candidate.get(&cand).cloned().unwrap_or(state.p_max_next);
            
            let n_seen_unvisited = state.seen_items.len().saturating_sub(stats.seen_items.len()) as f64;
            let mass_by_pmax = n_seen_unvisited * p_max;

            let fm_known = if is_mut_ex {
                let cand_seen_b = state.cand_consumed_budget.get(&cand).cloned().unwrap_or(0.0);
                let avail_budget = (state.total_seen_budget - cand_seen_b).max(0.0);
                mass_by_pmax.min(avail_budget)
            } else {
                mass_by_pmax
            };

            let fm_unknown = unvisited * p_max;
            let hi = lo + fm_known + fm_unknown;
            state.cand_bounds.push((cand, lo, hi));
        }

        if state.cand_bounds.is_empty() {
            return;
        }

        state.cand_bounds.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let (leader_key, leader_lo, leader_hi) = state.cand_bounds[0];

        let mut max_comp_upper = 0.0f64;
        for (_, _, hi) in &state.cand_bounds[1..] {
            if *hi > max_comp_upper {
                max_comp_upper = *hi;
            }
        }
        let unseen_upper = state.p_max_next * unvisited;
        let effective_comp_upper = max_comp_upper.max(unseen_upper);

        let mut tuples_to_emit = Vec::new();
        if leader_lo > effective_comp_upper {
            state.emitted_candidates.insert(leader_key);

            let mut norm_lo = leader_lo / n_tot;
            let mut norm_hi = leader_hi / n_tot;

            // Enforce strict monotonicity with previous rank
            if let Some(prev) = state.last_emitted_interval {
                norm_hi = norm_hi.min(prev.0 - 1e-6);
                norm_lo = norm_lo.min(norm_hi);
            }
            state.last_emitted_interval = Some((norm_lo, norm_hi));
            if state.emitted_candidates.len() >= self.config.k_target {
                state.is_resolved = true;
            }
            tuples_to_emit.push((leader_key, (norm_lo, norm_hi)));
        } else if state.seen_items.len() >= state.total_items && leader_lo >= max_comp_upper {
            // Stream completed: emit remaining leaders up to k_target plus ties
            let needed = self.config.k_target.saturating_sub(state.emitted_candidates.len());
            let mut remaining: Vec<(CompactKey, f64, f64)> = state.cand_bounds.iter()
                .filter(|(k, _, _)| !state.emitted_candidates.contains(k))
                .map(|(k, lo, hi)| (*k, *lo, *hi))
                .collect();
            remaining.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            if !remaining.is_empty() {
                let limit = needed.min(remaining.len());
                let threshold_lo = remaining[limit - 1].1;
                for (cand, lo, hi) in remaining {
                    if state.emitted_candidates.len() < self.config.k_target || (lo - threshold_lo).abs() < 1e-9 {
                        state.emitted_candidates.insert(cand);
                        let mut norm_lo = lo / n_tot;
                        let mut norm_hi = hi / n_tot;
                        if let Some(prev) = state.last_emitted_interval {
                            norm_hi = norm_hi.min(prev.0 - 1e-6);
                            norm_lo = norm_lo.min(norm_hi);
                        }
                        state.last_emitted_interval = Some((norm_lo, norm_hi));
                        tuples_to_emit.push((cand, (norm_lo, norm_hi)));
                    } else {
                        break;
                    }
                }
            }
            state.is_resolved = true;
        }

        for (cand, interval) in tuples_to_emit {
            let tuple = Arc::new(self.build_output_tuple(cand, *comp_key, interval, None));
            self.holding_queue.push(tuple);
        }
    }

    fn check_conformal_count_convergence(&mut self, comp_key: &CompactKey, tau: f64) {
        let is_mut_ex = self.config.budget_semantics == ItemBudgetSemantics::MutuallyExclusive;
        let state = self.partitions.get_mut(comp_key).unwrap();
        let n_tot = state.total_items.max(1) as f64;
        let unvisited = state.total_items.saturating_sub(state.seen_items.len()) as f64;

        state.cand_bounds.clear();
        for (&cand, stats) in &state.candidates {
            if state.emitted_candidates.contains(&cand) {
                continue;
            }
            let lo = stats.sum_lo;
            let p_max = state.p_max_next_per_candidate.get(&cand).cloned().unwrap_or(state.p_max_next);
            
            let n_seen_unvisited = state.seen_items.len().saturating_sub(stats.seen_items.len()) as f64;
            let mass_by_pmax = n_seen_unvisited * p_max;

            let fm_known = if is_mut_ex {
                let cand_seen_b = state.cand_consumed_budget.get(&cand).cloned().unwrap_or(0.0);
                let avail_budget = (state.total_seen_budget - cand_seen_b).max(0.0);
                mass_by_pmax.min(avail_budget)
            } else {
                mass_by_pmax
            };

            let fm_unknown = unvisited * p_max;
            let hi = lo + fm_known + fm_unknown;
            state.cand_bounds.push((cand, lo, hi));
        }

        if state.cand_bounds.is_empty() {
            return;
        }

        // Active contenders: Any candidate whose normalized upper bound (hi / n_tot) can reach tau
        let mut contenders: Vec<(CompactKey, f64, f64)> = Vec::new();
        let mut max_non_contender_hi = 0.0f64;

        for &(cand, lo, hi) in &state.cand_bounds {
            let norm_hi = hi / n_tot;
            if norm_hi >= tau - 1e-9 {
                contenders.push((cand, lo / n_tot, norm_hi));
            } else if norm_hi > max_non_contender_hi {
                max_non_contender_hi = norm_hi;
            }
        }

        let unseen_upper = (state.p_max_next * unvisited) / n_tot;
        let can_non_contender_reach_tau = max_non_contender_hi >= tau - 1e-9 || unseen_upper >= tau - 1e-9;
        let all_contenders_safe = !contenders.is_empty() && contenders.iter().all(|(_, lo, _)| *lo >= tau - 1e-9);

        if (!can_non_contender_reach_tau && all_contenders_safe)
            || state.seen_items.len() >= state.total_items
        {
            state.is_resolved = true;
            let mut tuples_to_emit = Vec::new();
            for (cand, lo, hi) in contenders {
                if !state.emitted_candidates.contains(&cand) {
                    state.emitted_candidates.insert(cand);
                    tuples_to_emit.push((cand, (lo, hi)));
                }
            }
            for (cand, interval) in tuples_to_emit {
                let tuple = Arc::new(self.build_output_tuple(cand, *comp_key, interval, None));
                self.holding_queue.push(tuple);
            }
        }
    }

    fn emit_final_conformal_count(&mut self, comp_key: &CompactKey, tau: f64) {
        let state = self.partitions.get_mut(comp_key).unwrap();
        state.is_resolved = true;
        let n_tot = state.total_items.max(1) as f64;

        let mut sorted: Vec<(CompactKey, f64)> = state.candidates
            .iter()
            .map(|(k, v)| (*k, v.sum_lo / n_tot))
            .collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut tuples_to_emit = Vec::new();
        for (cand, score) in sorted {
            if score >= tau - 1e-9 || tuples_to_emit.is_empty() {
                if !state.emitted_candidates.contains(&cand) {
                    state.emitted_candidates.insert(cand);
                    tuples_to_emit.push((cand, (score, score)));
                }
            }
        }

        for (cand, interval) in tuples_to_emit {
            let tuple = Arc::new(self.build_output_tuple(cand, *comp_key, interval, None));
            self.holding_queue.push(tuple);
        }
    }

    fn check_qpm_convergence(&mut self, comp_key: &CompactKey) {
        let is_mut_ex = self.config.budget_semantics == ItemBudgetSemantics::MutuallyExclusive;
        let state = self.partitions.get_mut(comp_key).unwrap();
        let tracker = match &mut state.qp_tracker {
            Some(t) => t,
            None => return,
        };

        let emitted_std_set: std::collections::HashSet<String> = state.emitted_candidates
            .iter()
            .map(|c| crate::ids::lookup(c.ids[0]))
            .collect();

        let prob_winner = tracker.check_probabilistic_convergence(Some(&emitted_std_set));

        // Deterministic count bound separation check
        let unvisited = state.total_items.saturating_sub(state.seen_items.len()) as f64;
        state.cand_bounds.clear();
        for (&cand, stats) in &state.candidates {
            if state.emitted_candidates.contains(&cand) {
                continue;
            }
            let lo = stats.sum_lo;
            let p_max = state.p_max_next_per_candidate.get(&cand).cloned().unwrap_or(state.p_max_next);
            let n_seen_unvisited = state.seen_items.len().saturating_sub(stats.seen_items.len()) as f64;
            let mass_by_pmax = n_seen_unvisited * p_max;
            let fm_known = if is_mut_ex {
                let cand_seen_b = state.cand_consumed_budget.get(&cand).cloned().unwrap_or(0.0);
                let avail_budget = (state.total_seen_budget - cand_seen_b).max(0.0);
                mass_by_pmax.min(avail_budget)
            } else {
                mass_by_pmax
            };
            let fm_unknown = unvisited * p_max;
            let hi = lo + fm_known + fm_unknown;
            state.cand_bounds.push((cand, lo, hi));
        }

        state.cand_bounds.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let count_winner = if !state.cand_bounds.is_empty() {
            let (leader_cand, leader_lo, _) = state.cand_bounds[0];
            let mut max_comp_upper = 0.0f64;
            for (_, _, hi) in &state.cand_bounds[1..] {
                if *hi > max_comp_upper {
                    max_comp_upper = *hi;
                }
            }
            let unseen_upper = state.p_max_next * unvisited;
            let effective_comp_upper = max_comp_upper.max(unseen_upper);
            if leader_lo > effective_comp_upper || (state.seen_items.len() >= state.total_items && leader_lo >= max_comp_upper) {
                let cand_str = crate::ids::lookup(leader_cand.ids[0]);
                let p_bounds = tracker.get_bounds(&cand_str);
                Some((leader_cand, p_bounds.0, p_bounds.1))
            } else {
                None
            }
        } else {
            None
        };

        let criterion = match &self.config.mode {
            AggregationMode::QPM { criterion, .. } | AggregationMode::QPI { criterion, .. } => *criterion,
            _ => DisambiguationCriterion::ProbabilitySeparation,
        };

        let mut resolved_candidates: Vec<(CompactKey, f64, f64)> = match criterion {
            DisambiguationCriterion::ProbabilitySeparation => {
                if let Some((w_str, p_lo, p_hi)) = prob_winner {
                    let w_key = CompactKey::single(crate::ids::intern(&w_str));
                    vec![(w_key, p_lo, p_hi)]
                } else {
                    Vec::new()
                }
            }
            DisambiguationCriterion::ExpectedValueSeparation => {
                if let Some((w_key, p_lo, p_hi)) = count_winner {
                    vec![(w_key, p_lo, p_hi)]
                } else {
                    Vec::new()
                }
            }
            DisambiguationCriterion::Both => {
                if let (Some((pw_str, p_lo, p_hi)), Some((cw_key, _, _))) = (&prob_winner, &count_winner) {
                    let pw_key = CompactKey::single(crate::ids::intern(pw_str));
                    if pw_key == *cw_key {
                        vec![(pw_key, *p_lo, *p_hi)]
                    } else {
                        Vec::new()
                    }
                } else {
                    Vec::new()
                }
            }
            DisambiguationCriterion::Either => {
                if let Some((pw_str, p_lo, p_hi)) = prob_winner {
                    let pw_key = CompactKey::single(crate::ids::intern(&pw_str));
                    vec![(pw_key, p_lo, p_hi)]
                } else if let Some((cw_key, p_lo, p_hi)) = count_winner {
                    vec![(cw_key, p_lo, p_hi)]
                } else {
                    Vec::new()
                }
            }
        };

        // If partition stream is complete, emit all tied top candidates
        if state.seen_items.len() >= state.total_items && !state.cand_bounds.is_empty() {
            let top_lo = state.cand_bounds[0].1;
            let tied: Vec<(CompactKey, f64, f64)> = state.cand_bounds.iter()
                .filter(|(_, lo, _)| (*lo - top_lo).abs() < 1e-9)
                .map(|(k, _, _)| {
                    let cand_str = crate::ids::lookup(k.ids[0]);
                    let p_bounds = tracker.get_bounds(&cand_str);
                    (*k, p_bounds.0, p_bounds.1)
                })
                .collect();
            if tied.len() > 1 {
                resolved_candidates = tied;
            }
        }

        let mut tuples_to_emit = Vec::new();
        if !resolved_candidates.is_empty() {
            for (cand_key, p_lo, p_hi) in resolved_candidates {
                state.emitted_candidates.insert(cand_key);
                let mut final_hi = p_hi;
                let mut final_lo = p_lo;
                if let Some(prev) = state.last_emitted_interval {
                    final_hi = final_hi.min(prev.0 - 1e-6);
                    final_lo = final_lo.min(final_hi);
                }
                state.last_emitted_interval = Some((final_lo, final_hi));
                tuples_to_emit.push((cand_key, (final_lo, final_hi)));
            }
            if state.emitted_candidates.len() >= self.config.k_target || state.seen_items.len() >= state.total_items {
                state.is_resolved = true;
            }
        } else if state.seen_items.len() >= state.total_items && !state.cand_bounds.is_empty() {
            // Stream completed: emit remaining candidates up to k_target plus ties
            let needed = self.config.k_target.saturating_sub(state.emitted_candidates.len());
            let mut remaining: Vec<(CompactKey, f64, f64)> = state.cand_bounds.iter()
                .filter(|(k, _, _)| !state.emitted_candidates.contains(k))
                .map(|(k, lo, hi)| (*k, *lo, *hi))
                .collect();
            remaining.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            if !remaining.is_empty() {
                let limit = needed.min(remaining.len());
                let threshold_lo = remaining[limit - 1].1;
                for (cand, _, _) in remaining {
                    if state.emitted_candidates.len() < self.config.k_target || (state.candidates[&cand].sum_lo - threshold_lo).abs() < 1e-9 {
                        state.emitted_candidates.insert(cand);
                        let cand_str = crate::ids::lookup(cand.ids[0]);
                        let p_bounds = tracker.get_bounds(&cand_str);
                        let mut final_hi = p_bounds.1;
                        let mut final_lo = p_bounds.0;
                        if let Some(prev) = state.last_emitted_interval {
                            final_hi = final_hi.min(prev.0 - 1e-6);
                            final_lo = final_lo.min(final_hi);
                        }
                        state.last_emitted_interval = Some((final_lo, final_hi));
                        tuples_to_emit.push((cand, (final_lo, final_hi)));
                    } else {
                        break;
                    }
                }
            }
            state.is_resolved = true;
        }

        for (cand_key, interval) in tuples_to_emit {
            let tuple = Arc::new(self.build_output_tuple(cand_key, *comp_key, interval, None));
            self.holding_queue.push(tuple);
        }
    }

    fn emit_final_naive(&mut self, comp_key: &CompactKey) {
        let state = self.partitions.get_mut(comp_key).unwrap();
        state.is_resolved = true;
        let n_tot = state.total_items.max(1) as f64;

        let mut sorted_cands: Vec<(CompactKey, f64)> = state.candidates
            .iter()
            .filter(|(k, _)| !state.emitted_candidates.contains(k))
            .map(|(k, v)| (*k, v.sum_hi))
            .collect();
        sorted_cands.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let needed = self.config.k_target.saturating_sub(state.emitted_candidates.len());
        let mut tuples_to_emit = Vec::new();
        if needed > 0 && !sorted_cands.is_empty() {
            let limit = needed.min(sorted_cands.len());
            let threshold_sum = sorted_cands[limit - 1].1;
            for (winner_key, sum) in sorted_cands {
                if state.emitted_candidates.len() < self.config.k_target || (sum - threshold_sum).abs() < 1e-9 {
                    state.emitted_candidates.insert(winner_key);
                    let norm = sum / n_tot;
                    tuples_to_emit.push((winner_key, (norm, norm)));
                } else {
                    break;
                }
            }
        }

        for (winner_key, interval) in tuples_to_emit {
            let tuple = Arc::new(self.build_output_tuple(winner_key, *comp_key, interval, None));
            self.holding_queue.push(tuple);
        }
    }

    fn emit_final_mcdb(&mut self, comp_key: &CompactKey, n_worlds: usize) {
        let state = self.partitions.get_mut(comp_key).unwrap();
        state.is_resolved = true;

        let mut world_winners = vec![None; n_worlds];
        for w in 0..n_worlds {
            let mut best_cand = None;
            let mut best_count = 0u32;
            for (cand, stats) in &state.candidates {
                let cnt = stats.world_counts[w];
                if cnt > best_count {
                    best_count = cnt;
                    best_cand = Some(*cand);
                }
            }
            world_winners[w] = best_cand;
        }

        // Pack victory bitvector per candidate
        let n_words = (n_worlds + 63) / 64;
        let mut victory_masks: FxHashMap<CompactKey, Vec<u64>> = FxHashMap::default();
        let mut victory_counts: FxHashMap<CompactKey, usize> = FxHashMap::default();

        for w in 0..n_worlds {
            if let Some(cand) = world_winners[w] {
                let mask = victory_masks.entry(cand).or_insert_with(|| vec![0u64; n_words]);
                let word_idx = w / 64;
                let bit_idx = w % 64;
                mask[word_idx] |= 1u64 << bit_idx;
                *victory_counts.entry(cand).or_insert(0) += 1;
            }
        }

        // Sort candidates by worlds won descending
        let mut cand_order: Vec<(CompactKey, usize)> = victory_counts.into_iter()
            .filter(|(k, _)| !state.emitted_candidates.contains(k))
            .collect();
        cand_order.sort_by(|a, b| b.1.cmp(&a.1));

        let needed = self.config.k_target.saturating_sub(state.emitted_candidates.len());
        let mut tuples_to_emit = Vec::new();
        if needed > 0 && !cand_order.is_empty() {
            let limit = needed.min(cand_order.len());
            let threshold_won = cand_order[limit - 1].1;
            for (cand, won_count) in cand_order {
                if state.emitted_candidates.len() < self.config.k_target || won_count == threshold_won {
                    state.emitted_candidates.insert(cand);
                    let p_win = won_count as f64 / n_worlds as f64;
                    let masks = victory_masks.get(&cand).cloned();
                    tuples_to_emit.push((cand, (p_win, p_win), masks));
                } else {
                    break;
                }
            }
        }

        for (cand, interval, masks) in tuples_to_emit {
            let tuple = Arc::new(self.build_output_tuple(cand, *comp_key, interval, masks));
            self.holding_queue.push(tuple);
        }
    }

    fn build_output_tuple(
        &self,
        count_group: CompactKey,
        comp_key: CompactKey,
        interval: (f64, f64),
        mc_masks: Option<Vec<u64>>,
    ) -> Tuple {
        let mut data = IndexMap::new();

        // Project the count_group fields
        for i in 0..count_group.len as usize {
            if i < self.config.count_group_by.len() {
                let col_name = &self.config.count_group_by[i];
                let val_str = crate::ids::lookup(count_group.ids[i]);
                data.insert(col_name.clone(), FieldValue::Str(val_str));
            }
        }

        // Project the comparison_key fields if not already present
        for i in 0..comp_key.len as usize {
            if i < self.config.comparison_key.len() {
                let col_name = &self.config.comparison_key[i];
                if !data.contains_key(col_name) {
                    let val_str = crate::ids::lookup(comp_key.ids[i]);
                    data.insert(col_name.clone(), FieldValue::Str(val_str));
                }
            }
        }

        // Add MCDB bitvectors if present
        if let Some(masks) = mc_masks {
            for (word_idx, mask) in masks.iter().enumerate() {
                let key = format!("__mc_bits_{}__", word_idx);
                data.insert(key, FieldValue::Int(*mask as i64));
            }
        }

        let score = ProbInterval::from_probs(interval.0, interval.1);
        Tuple::new_interval(data, score)
    }
}

impl TupleStream for ComposableArgMaxOverCount {
    fn next_tuple(&mut self) -> Option<Arc<Tuple>> {
        loop {
            // 1. Check holding queue for ready emission in non-increasing score order
            if !self.holding_queue.is_empty() {
                self.holding_queue.sort_by(|a, b| {
                    b.score.to_prob_hi()
                        .partial_cmp(&a.score.to_prob_hi())
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| {
                            b.score.to_prob_lo()
                                .partial_cmp(&a.score.to_prob_lo())
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                });

                if self.is_child_exhausted {
                    return Some(self.holding_queue.remove(0));
                }

                let mut s_future_max = f64::NEG_INFINITY;
                for (_, s) in &self.partitions {
                    if !s.is_resolved && s.p_max_next > s_future_max {
                        s_future_max = s.p_max_next;
                    }
                }
                if self.partitions.len() < self.config.expected_cardinality.len() {
                    if self.current_leaf_p_max > s_future_max {
                        s_future_max = self.current_leaf_p_max;
                    }
                }

                let top = &self.holding_queue[0];
                let top_hi = top.score.to_prob_hi();
                let top_lo = top.score.to_prob_lo();

                // Ready to emit if top item is guaranteed to dominate any future score,
                // or if it represents an exact indicator [1.0, 1.0] (R1V2)
                if top_lo >= s_future_max || (top_hi >= 1.0 && s_future_max <= 1.0) {
                    return Some(self.holding_queue.remove(0));
                }
            }

            // Early termination: If all expected comparison partitions have reached resolution
            // and the holding queue has been drained of all emitted items, the operator is done.
            if !self.partitions.is_empty()
                && self.partitions.len() >= self.config.expected_cardinality.len()
                && self.partitions.values().all(|s| s.is_resolved)
                && self.holding_queue.is_empty()
            {
                return None;
            }

            // 2. If child is exhausted and holding queue is empty, terminate
            if self.is_child_exhausted {
                return None;
            }

            // 3. Pull next item from child
            match self.child.next_tuple() {
                Some(tuple) => {
                    self.process_tuple(&tuple);
                }
                None => {
                    self.is_child_exhausted = true;
                    // Force final resolution on remaining partitions
                    let pending_keys: Vec<CompactKey> = self.partitions
                        .iter()
                        .filter(|(_, s)| !s.is_resolved)
                        .map(|(k, _)| *k)
                        .collect();

                    for k in pending_keys {
                        match self.config.mode {
                            AggregationMode::Naive => self.emit_final_naive(&k),
                            AggregationMode::MCDBBitvec { n_worlds } => self.emit_final_mcdb(&k, n_worlds),
                            AggregationMode::ConformalCountPruning { tau } => self.emit_final_conformal_count(&k, tau),
                            _ => {
                                // For bounding modes, emit best current candidate
                                self.emit_final_naive(&k);
                            }
                        }
                    }
                }
            }
        }
    }

    fn total_items_read(&self) -> usize {
        let child_read = self.child.total_items_read();
        if child_read > 0 {
            child_read
        } else {
            self.items_read
        }
    }
}
