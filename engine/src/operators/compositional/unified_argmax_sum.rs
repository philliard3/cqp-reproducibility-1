/// unified_argmax_sum.rs
///
/// Composable, schema-agnostic ArgMaxOverSum operator for continuous probabilistic revenue aggregation.
/// Specifically designed for continuous aggregate queries like TPC-H Q3.
///
/// Distinct from ArgMaxOverCount:
/// - Accumulates continuous floating-point values (e.g., filtered order revenue) rather than integer unit counts.
/// - Supports dedicated continuous MCDB simulation where each world w tracks accumulated float revenue.
/// - Supports in-operator conformal set pruning with calibrated threshold tau, halting the stream early
///   without requiring pairwise candidate-by-candidate disambiguation.

use std::sync::Arc;
use rustc_hash::{FxHashMap, FxHashSet};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::tuple::{Tuple, FieldValue, ProbInterval};
use crate::operators::compositional::unified_argmax::{CompactKey, TupleStream};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum AggregationModeSum {
    /// In-engine ground truth baseline: processes 100% of stream, sums expected revenue, emits top-k.
    NaiveSum,
    /// Continuous MCDB: accumulates float revenue per world for tuples whose existential bit is 1.
    MCDBSum { n_worlds: usize },
    /// Demand-driven QN1: ranks candidates one-by-one via strict upper/lower bound separation.
    QN1DemandRanked,
    /// In-operator Conformal Early Stopping: prunes candidates with hi < tau and stops the stream
    /// as soon as the contender set {c: hi_c >= tau} is guaranteed.
    ConformalSetPruning { tau: f64 },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CandidateSnapshot {
    pub candidate_key: String,
    pub sum_lo: f64,
    pub sum_hi: f64,
    pub is_contender: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LifetimeSnapshot {
    pub quantile: f64,
    pub items_in: usize,
    pub items_out: usize,
    pub total_expected: usize,
    pub v_max_next: f64,
    pub unseen_upper: f64,
    pub tau: Option<f64>,
    pub n_contenders: usize,
    pub n_safe_contenders: usize,
    pub max_non_contender_hi: f64,
    pub can_non_contender_reach_tau: bool,
    pub is_resolved: bool,
    pub active_contender_names: Vec<String>,
    pub top_candidates: Vec<CandidateSnapshot>,
}

#[derive(Clone, Debug)]
pub struct ArgMaxSumConfig {
    pub item_key: Vec<String>,
    pub count_group_by: Vec<String>,
    pub comparison_key: Vec<String>,
    pub expected_cardinality: FxHashMap<CompactKey, usize>,
    pub mode: AggregationModeSum,
    pub k_target: usize,
    pub value_field: String,
    pub v_max: f64,
    pub monitor_lifetime: bool,
}

#[derive(Debug, Clone)]
pub struct CandidateStatsSum {
    pub sum_lo: f64,
    pub sum_hi: f64,
    pub revenue: f64,
    pub seen_items: FxHashSet<CompactKey>,
    pub world_sums: Vec<f64>,
}

impl CandidateStatsSum {
    pub fn new(n_worlds: usize) -> Self {
        CandidateStatsSum {
            sum_lo: 0.0,
            sum_hi: 0.0,
            revenue: 0.0,
            seen_items: FxHashSet::default(),
            world_sums: vec![0.0; n_worlds],
        }
    }
}

#[derive(Debug)]
pub struct SumPartitionState {
    pub total_items: usize,
    pub seen_items: FxHashSet<CompactKey>,
    pub candidates: FxHashMap<CompactKey, CandidateStatsSum>,
    pub v_max_next: f64,
    pub max_cand_lo: f64,
    pub emitted_candidates: FxHashSet<CompactKey>,
    pub last_emitted_score: Option<f64>,
    pub is_resolved: bool,
    pub cand_bounds: Vec<(CompactKey, f64, f64)>, // (key, lo, hi)
}

impl SumPartitionState {
    pub fn new(total_items: usize, all_candidates: &[String]) -> Self {
        let mut candidates = FxHashMap::default();
        for name in all_candidates {
            let id = crate::ids::intern(name.as_str());
            let key = CompactKey::single(id);
            candidates.insert(key, CandidateStatsSum::new(0));
        }

        SumPartitionState {
            total_items,
            seen_items: FxHashSet::default(),
            candidates,
            v_max_next: 0.0,
            max_cand_lo: 0.0,
            emitted_candidates: FxHashSet::default(),
            last_emitted_score: None,
            is_resolved: false,
            cand_bounds: Vec::new(),
        }
    }
}

pub struct ComposableArgMaxOverSum {
    pub child: Box<dyn TupleStream>,
    pub config: ArgMaxSumConfig,
    pub partitions: FxHashMap<CompactKey, SumPartitionState>,
    pub holding_queue: Vec<Arc<Tuple>>,
    pub items_read: usize,
    pub items_emitted: usize,
    pub current_leaf_val_max: f64,
    pub is_child_exhausted: bool,
    pub all_candidate_names: Vec<String>,
    pub lifetime_trace: Vec<LifetimeSnapshot>,
    pub next_quantile_target: f64,
}

impl ComposableArgMaxOverSum {
    pub fn new(
        child: Box<dyn TupleStream>,
        config: ArgMaxSumConfig,
        all_candidate_names: Vec<String>,
    ) -> Self {
        ComposableArgMaxOverSum {
            child,
            config,
            partitions: FxHashMap::default(),
            holding_queue: Vec::new(),
            items_read: 0,
            items_emitted: 0,
            current_leaf_val_max: 0.0,
            is_child_exhausted: false,
            all_candidate_names,
            lifetime_trace: Vec::new(),
            next_quantile_target: 0.10,
        }
    }

    pub fn get_lifetime_trace(&self) -> Vec<LifetimeSnapshot> {
        self.lifetime_trace.clone()
    }

    pub fn record_lifetime_snapshot(&mut self, quantile: f64, total_expected: usize, comp_key: &CompactKey) {
        let state = match self.partitions.get(comp_key) {
            Some(s) => s,
            None => return,
        };
        let unvisited = state.total_items.saturating_sub(state.seen_items.len()) as f64;
        let unseen_upper = state.v_max_next * unvisited;

        let tau_opt = match self.config.mode {
            AggregationModeSum::ConformalSetPruning { tau } => Some(tau),
            _ => None,
        };
        let tau_val = tau_opt.unwrap_or(0.0);

        let mut contenders = Vec::new();
        let mut safe_contenders = 0;
        let mut max_non_contender_hi = 0.0f64;
        let mut details = Vec::new();

        for (&cand, stats) in &state.candidates {
            let lo = stats.sum_lo;
            let hi = lo + unseen_upper;
            let is_contender = hi >= tau_val;
            let cand_label = if cand.len > 0 {
                crate::ids::lookup(cand.ids[0]).to_string()
            } else {
                "Unknown".to_string()
            };

            if is_contender {
                contenders.push(cand_label.clone());
                if lo >= tau_val {
                    safe_contenders += 1;
                }
            } else if hi > max_non_contender_hi {
                max_non_contender_hi = hi;
            }

            details.push(CandidateSnapshot {
                candidate_key: cand_label,
                sum_lo: lo,
                sum_hi: hi,
                is_contender,
            });
        }

        details.sort_by(|a, b| b.sum_hi.partial_cmp(&a.sum_hi).unwrap_or(std::cmp::Ordering::Equal));
        if details.len() > 25 {
            details.truncate(25);
        }

        let can_non_contender_reach_tau = max_non_contender_hi >= tau_val || unseen_upper >= tau_val;

        self.lifetime_trace.push(LifetimeSnapshot {
            quantile,
            items_in: self.items_read,
            items_out: self.items_emitted + self.holding_queue.len(),
            total_expected,
            v_max_next: state.v_max_next,
            unseen_upper,
            tau: tau_opt,
            n_contenders: contenders.len(),
            n_safe_contenders: safe_contenders,
            max_non_contender_hi,
            can_non_contender_reach_tau,
            is_resolved: state.is_resolved,
            active_contender_names: contenders,
            top_candidates: details,
        });
    }

    fn get_or_create_partition(&mut self, comp_key: &CompactKey) -> &mut SumPartitionState {
        let expected = self.config.expected_cardinality
            .get(comp_key)
            .cloned()
            .unwrap_or(1);
        let cand_names = self.all_candidate_names.clone();

        self.partitions
            .entry(*comp_key)
            .or_insert_with(|| SumPartitionState::new(expected, &cand_names))
    }

    fn process_tuple(&mut self, tuple: &Tuple) {
        self.items_read += 1;
        let comp_key = CompactKey::extract(&tuple.data, &self.config.comparison_key);
        let item_key = CompactKey::extract(&tuple.data, &self.config.item_key);
        let count_group = CompactKey::extract(&tuple.data, &self.config.count_group_by);

        // Continuous revenue value
        let raw_val = tuple.data.get(&self.config.value_field)
            .and_then(|v| v.as_f64())
            .unwrap_or_else(|| {
                tuple.score.to_prob_lo() * self.config.v_max
            });

        let p_lo = tuple.score.to_prob_lo();
        let p_hi = tuple.score.to_prob_hi();

        // Expected revenue contributions
        let exp_val_lo = p_lo * raw_val;
        let _exp_val_hi = p_hi * raw_val;

        // Score space contribution (normalized to [0, 1])
        let proxy_val = tuple.data.get("proxy_score")
            .and_then(|v| v.as_f64().map(|f| (f / self.config.v_max).max(0.0).min(1.0)))
            .unwrap_or_else(|| tuple.score.to_prob_hi());

        self.current_leaf_val_max = proxy_val;

        let n_worlds = match self.config.mode {
            AggregationModeSum::MCDBSum { n_worlds } => n_worlds,
            _ => 0,
        };

        let state = self.get_or_create_partition(&comp_key);
        state.seen_items.insert(item_key);
        state.v_max_next = proxy_val;

        let cand_stats = state.candidates
            .entry(count_group)
            .or_insert_with(|| CandidateStatsSum::new(n_worlds));

        cand_stats.sum_lo += proxy_val;
        cand_stats.sum_hi += proxy_val;
        if cand_stats.sum_lo > state.max_cand_lo {
            state.max_cand_lo = cand_stats.sum_lo;
        }
        cand_stats.revenue += exp_val_lo;
        cand_stats.seen_items.insert(item_key);

        // Continuous MCDB: Accumulate float revenue into world_sums
        if n_worlds > 0 {
            if cand_stats.world_sums.len() < n_worlds {
                cand_stats.world_sums.resize(n_worlds, 0.0);
            }
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
                            cand_stats.world_sums[w] += raw_val;
                        }
                    }
                }
            }
        }

        // Check for early convergence / pruning
        self.check_partition_convergence(&comp_key);

        if self.config.monitor_lifetime {
            let total_expected = self.config.expected_cardinality
                .get(&comp_key)
                .cloned()
                .unwrap_or(1);
            let progress = (self.items_read as f64 / total_expected as f64).min(1.0);
            let is_resolved = self.partitions.get(&comp_key).map(|s| s.is_resolved).unwrap_or(false);
            if progress >= self.next_quantile_target || is_resolved {
                let q_rec = self.next_quantile_target;
                self.record_lifetime_snapshot(q_rec, total_expected, &comp_key);
                while self.next_quantile_target <= progress && self.next_quantile_target < 1.0 {
                    self.next_quantile_target += 0.10;
                }
            }
        }
    }

    fn check_partition_convergence(&mut self, comp_key: &CompactKey) {
        let state = match self.partitions.get_mut(comp_key) {
            Some(s) => s,
            None => return,
        };

        if state.is_resolved {
            return;
        }

        match self.config.mode {
            AggregationModeSum::NaiveSum | AggregationModeSum::MCDBSum { .. } => {
                // Naive and MCDB strictly run to full stream exhaustion.
            }
            AggregationModeSum::QN1DemandRanked => {
                self.check_qn1_convergence(comp_key);
            }
            AggregationModeSum::ConformalSetPruning { tau } => {
                self.check_conformal_convergence(comp_key, tau);
            }
        }
    }

    fn check_qn1_convergence(&mut self, comp_key: &CompactKey) {
        let state = self.partitions.get_mut(comp_key).unwrap();
        let unvisited = state.total_items.saturating_sub(state.seen_items.len()) as f64;
        let unseen_upper = state.v_max_next * unvisited;

        // O(1) lazy exit: if max seen lower bound cannot beat unseen upper,
        // leader cannot be disambiguated yet.
        if state.max_cand_lo <= unseen_upper && state.seen_items.len() < state.total_items {
            return;
        }

        state.cand_bounds.clear();
        for (&cand, stats) in &state.candidates {
            if state.emitted_candidates.contains(&cand) {
                continue;
            }
            let lo = stats.sum_lo;
            let hi = lo + unvisited * state.v_max_next;
            state.cand_bounds.push((cand, lo, hi));
        }

        if state.cand_bounds.is_empty() {
            return;
        }

        // Sort by lower bound descending
        state.cand_bounds.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let (leader_key, leader_lo, leader_hi) = state.cand_bounds[0];

        let mut max_comp_upper = 0.0f64;
        for (_, _, hi) in &state.cand_bounds[1..] {
            if *hi > max_comp_upper {
                max_comp_upper = *hi;
            }
        }
        let effective_comp_upper = max_comp_upper.max(unseen_upper);

        let mut tuples_to_emit = Vec::new();
        if leader_lo > effective_comp_upper {
            state.emitted_candidates.insert(leader_key);
            state.last_emitted_score = Some(leader_lo);
            if state.emitted_candidates.len() >= self.config.k_target {
                state.is_resolved = true;
            }
            tuples_to_emit.push((leader_key, (leader_lo, leader_hi)));
        } else if state.seen_items.len() >= state.total_items && leader_lo >= max_comp_upper {
            // Stream exhausted: emit remaining leaders
            let needed = self.config.k_target.saturating_sub(state.emitted_candidates.len());
            let mut remaining: Vec<(CompactKey, f64, f64)> = state.cand_bounds.iter()
                .filter(|(k, _, _)| !state.emitted_candidates.contains(k))
                .map(|(k, lo, hi)| (*k, *lo, *hi))
                .collect();
            remaining.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            if !remaining.is_empty() {
                let limit = needed.min(remaining.len());
                for (cand, lo, hi) in remaining.into_iter().take(limit) {
                    state.emitted_candidates.insert(cand);
                    tuples_to_emit.push((cand, (lo, hi)));
                }
            }
            state.is_resolved = true;
        }

        for (cand, interval) in tuples_to_emit {
            let tuple = Arc::new(self.build_output_tuple(cand, *comp_key, interval, None));
            self.holding_queue.push(tuple);
        }
    }

    fn check_conformal_convergence(&mut self, comp_key: &CompactKey, tau: f64) {
        let state = self.partitions.get_mut(comp_key).unwrap();
        let tau_norm = if tau > state.total_items as f64 && self.config.v_max > 0.0 {
            tau / self.config.v_max
        } else {
            tau
        };
        let unvisited = state.total_items.saturating_sub(state.seen_items.len()) as f64;
        let unseen_upper = state.v_max_next * unvisited;

        // O(1) lazy exit: while unseen upper >= tau_norm, unseen groups can theoretically reach tau_norm.
        // Therefore can_non_contender_reach_tau is true, and early termination is impossible.
        if unseen_upper >= tau_norm && state.seen_items.len() < state.total_items {
            return;
        }

        state.cand_bounds.clear();
        for (&cand, stats) in &state.candidates {
            let lo = stats.sum_lo;
            let hi = lo + unseen_upper;
            state.cand_bounds.push((cand, lo, hi));
        }

        // Active contenders: Any candidate whose upper bound can reach tau_norm
        let mut contenders: Vec<(CompactKey, f64, f64)> = Vec::new();
        let mut max_non_contender_hi = 0.0f64;

        for &(cand, lo, hi) in &state.cand_bounds {
            if hi >= tau_norm {
                contenders.push((cand, lo, hi));
            } else if hi > max_non_contender_hi {
                max_non_contender_hi = hi;
            }
        }

        // Early stopping condition:
        // 1. No non-contender can ever reach tau_norm: max_non_contender_hi < tau_norm AND unseen_upper < tau_norm
        // 2. All contenders have lower bounds >= tau_norm OR stream completed OR
        //    the set of contenders is fully resolved from non-contenders
        let can_non_contender_reach_tau = max_non_contender_hi >= tau_norm || unseen_upper >= tau_norm;
        let all_contenders_safe = !contenders.is_empty() && contenders.iter().all(|(_, lo, _)| *lo >= tau_norm);

        let mut tuples_to_emit = Vec::new();
        if (!can_non_contender_reach_tau && all_contenders_safe)
            || state.seen_items.len() >= state.total_items
        {
            state.is_resolved = true;
            // Emit all active contenders as the conformal prediction set
            for (cand, lo, hi) in contenders {
                if !state.emitted_candidates.contains(&cand) {
                    state.emitted_candidates.insert(cand);
                    tuples_to_emit.push((cand, (lo, hi), None));
                }
            }
        }

        for (cand, interval, masks) in tuples_to_emit {
            let tuple = Arc::new(self.build_output_tuple(cand, *comp_key, interval, masks));
            self.holding_queue.push(tuple);
        }
    }

    fn emit_final_naive(&mut self, comp_key: &CompactKey) {
        let state = self.partitions.get_mut(comp_key).unwrap();
        state.is_resolved = true;

        let mut sorted: Vec<(CompactKey, f64)> = state.candidates
            .iter()
            .map(|(k, v)| (*k, v.sum_lo))
            .collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut tuples_to_emit = Vec::new();
        for (cand, sum_rev) in sorted.into_iter().take(self.config.k_target) {
            state.emitted_candidates.insert(cand);
            tuples_to_emit.push((cand, (sum_rev, sum_rev), None));
        }

        for (cand, interval, masks) in tuples_to_emit {
            let tuple = Arc::new(self.build_output_tuple(cand, *comp_key, interval, masks));
            self.holding_queue.push(tuple);
        }
    }

    fn emit_final_mcdb(&mut self, comp_key: &CompactKey, n_worlds: usize) {
        let state = self.partitions.get_mut(comp_key).unwrap();
        state.is_resolved = true;

        // In each world w, find candidate with maximum accumulated continuous revenue
        let mut world_winners = vec![None; n_worlds];
        for w in 0..n_worlds {
            let mut best_cand = None;
            let mut best_rev = -1.0f64;
            for (cand, stats) in &state.candidates {
                let rev = stats.world_sums.get(w).copied().unwrap_or(0.0);
                if rev > best_rev {
                    best_rev = rev;
                    best_cand = Some(*cand);
                }
            }
            world_winners[w] = best_cand;
        }

        // Tally world wins
        let mut victory_counts: FxHashMap<CompactKey, usize> = FxHashMap::default();
        let n_words = (n_worlds + 63) / 64;
        let mut victory_masks: FxHashMap<CompactKey, Vec<u64>> = FxHashMap::default();

        for w in 0..n_worlds {
            if let Some(cand) = world_winners[w] {
                *victory_counts.entry(cand).or_insert(0) += 1;
                let mask = victory_masks.entry(cand).or_insert_with(|| vec![0u64; n_words]);
                mask[w / 64] |= 1u64 << (w % 64);
            }
        }

        // Rank candidates: primary by world wins descending, secondary by mean world revenue descending
        let mut all_cands: Vec<(CompactKey, usize, f64)> = state.candidates
            .iter()
            .map(|(k, stats)| {
                let wins = victory_counts.get(k).copied().unwrap_or(0);
                let avg_world_rev = stats.world_sums.iter().sum::<f64>() / n_worlds.max(1) as f64;
                (*k, wins, avg_world_rev)
            })
            .collect();
        all_cands.sort_by(|a, b| {
            b.1.cmp(&a.1)
                .then_with(|| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal))
        });

        let mut tuples_to_emit = Vec::new();
        for (cand, won_count, avg_rev) in all_cands.into_iter().take(self.config.k_target) {
            state.emitted_candidates.insert(cand);
            let p_win = won_count as f64 / n_worlds as f64;
            let masks = victory_masks.get(&cand).cloned();
            tuples_to_emit.push((cand, (avg_rev, p_win), masks));
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

        for i in 0..count_group.len as usize {
            if i < self.config.count_group_by.len() {
                let col_name = &self.config.count_group_by[i];
                let val_str = crate::ids::lookup(count_group.ids[i]);
                data.insert(col_name.clone(), FieldValue::Str(val_str));
            }
        }

        for i in 0..comp_key.len as usize {
            if i < self.config.comparison_key.len() {
                let col_name = &self.config.comparison_key[i];
                if !data.contains_key(col_name) {
                    let val_str = crate::ids::lookup(comp_key.ids[i]);
                    data.insert(col_name.clone(), FieldValue::Str(val_str));
                }
            }
        }

        if let Some(masks) = mc_masks {
            for (word_idx, mask) in masks.iter().enumerate() {
                let key = format!("__mc_bits_{}__", word_idx);
                data.insert(key, FieldValue::Int(*mask as i64));
            }
        }

        let rev = self.partitions.get(&comp_key)
            .and_then(|p| p.candidates.get(&count_group))
            .map(|c| c.revenue)
            .unwrap_or(0.0);

        let total_items = self.partitions.get(&comp_key).map(|p| p.total_items).unwrap_or(1).max(1) as f64;
        let rev_lo = interval.0 * self.config.v_max;
        let rev_hi = interval.1 * self.config.v_max;

        data.insert("revenue".to_string(), FieldValue::Float(rev));
        data.insert("revenue_lo".to_string(), FieldValue::Float(rev_lo));
        data.insert("revenue_hi".to_string(), FieldValue::Float(rev_hi));
        data.insert("score_lo".to_string(), FieldValue::Float(interval.0));
        data.insert("score_hi".to_string(), FieldValue::Float(interval.1));

        let norm_lo = (interval.0 / total_items).max(0.0).min(1.0);
        let norm_hi = (interval.1 / total_items).max(0.0).min(1.0);
        let score = ProbInterval::from_probs(norm_lo, norm_hi);
        Tuple::new_interval(data, score)
    }

    pub fn total_items_read(&self) -> usize {
        self.items_read
    }
}

impl TupleStream for ComposableArgMaxOverSum {
    fn next_tuple(&mut self) -> Option<Arc<Tuple>> {
        loop {
            if !self.holding_queue.is_empty() {
                let t = self.holding_queue.remove(0);
                self.items_emitted += 1;
                return Some(t);
            }

            if !self.partitions.is_empty()
                && self.partitions.len() >= self.config.expected_cardinality.len()
                && self.partitions.values().all(|s| s.is_resolved)
                && self.holding_queue.is_empty()
            {
                return None;
            }

            let next_tup = self.child.next_tuple();
            match next_tup {
                Some(t) => {
                    self.process_tuple(&t);
                }
                None => {
                    self.is_child_exhausted = true;
                    let keys: Vec<CompactKey> = self.partitions.keys().cloned().collect();
                    for k in &keys {
                        if !self.partitions[k].is_resolved {
                            match self.config.mode {
                                AggregationModeSum::NaiveSum => self.emit_final_naive(k),
                                AggregationModeSum::MCDBSum { n_worlds } => self.emit_final_mcdb(k, n_worlds),
                                AggregationModeSum::QN1DemandRanked => self.check_qn1_convergence(k),
                                AggregationModeSum::ConformalSetPruning { tau } => self.check_conformal_convergence(k, tau),
                            }
                        }
                        if self.config.monitor_lifetime {
                            let total_expected = self.config.expected_cardinality
                                .get(k)
                                .cloned()
                                .unwrap_or(1);
                            self.record_lifetime_snapshot(1.0, total_expected, k);
                        }
                    }
                    if !self.holding_queue.is_empty() {
                        let t = self.holding_queue.remove(0);
                        self.items_emitted += 1;
                        return Some(t);
                    }
                    return None;
                }
            }
        }
    }

    fn total_items_read(&self) -> usize {
        self.items_read
    }
}
