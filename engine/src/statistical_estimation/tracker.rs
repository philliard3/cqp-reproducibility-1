/// Streaming state tracking per partition for statistical bounds.

use std::collections::HashSet;
use rustc_hash::{FxHashMap, FxHashSet};
use indexmap::IndexMap;
use super::{QPEngineType, QPUpdateScheme, QPConfig, QPWinnerBounds};
use super::bounds::pairwise_defeat_upper_bound;
use super::local_dp::compute_pairwise_defeat_local_dp;

#[derive(Clone, Debug)]
pub struct QPPartitionTracker {
    pub class_ids: Vec<String>,
    pub class_to_idx: FxHashMap<String, usize>,
    pub m_classes: usize,
    pub n_tot: usize,
    pub use_mutual_exclusion: bool,
    pub config: QPConfig,

    // Cumulative statistics
    pub m_images_seen: usize,
    pub s: Vec<f64>,                  // Sum of expectations
    pub v: Vec<f64>,                  // Sum of marginal variances p*(1-p)
    pub c: Vec<Vec<f64>>,             // Cross terms sum p_g * p_j
    pub p_last: Vec<f64>,

    // Ingestion buffers
    pub observed_p_per_image: IndexMap<String, Vec<f64>>,

    // Update scheme bookkeeping
    pub cached_bounds: FxHashMap<String, QPWinnerBounds>,
    pub last_update_means: Vec<f64>,
    pub active_candidates: FxHashSet<usize>,
    pub dirty: bool,
}

impl QPPartitionTracker {
    pub fn new(
        class_ids: Vec<String>,
        n_images_total: usize,
        use_mutual_exclusion: bool,
        config: Option<QPConfig>,
    ) -> Self {
        let m = class_ids.len();
        let mut class_to_idx = FxHashMap::default();
        for (i, cid) in class_ids.iter().enumerate() {
            class_to_idx.insert(cid.clone(), i);
        }

        let cfg = config.unwrap_or_default();
        let p_init = if m > 0 { 1.0 / (m as f64) } else { 1.0 };

        let mut active_candidates = FxHashSet::default();
        let init_k = m.min(cfg.candidate_k);
        for i in 0..init_k {
            active_candidates.insert(i);
        }

        QPPartitionTracker {
            class_ids,
            class_to_idx,
            m_classes: m,
            n_tot: n_images_total,
            use_mutual_exclusion,
            config: cfg,
            m_images_seen: 0,
            s: vec![0.0; m],
            v: vec![0.0; m],
            c: vec![vec![0.0; m]; m],
            p_last: vec![p_init; m],
            observed_p_per_image: IndexMap::new(),
            cached_bounds: FxHashMap::default(),
            last_update_means: vec![0.0; m],
            active_candidates,
            dirty: true,
        }
    }

    pub fn ingest_tuple(&mut self, img_id: &str, class_id: &str, score: f64) {
        let g = match self.class_to_idx.get(class_id) {
            Some(&idx) => idx,
            None => return,
        };

        let p_val = score.max(0.0).min(1.0);

        if !self.observed_p_per_image.contains_key(img_id) {
            self.observed_p_per_image.insert(img_id.to_string(), vec![0.0; self.m_classes]);
            self.m_images_seen += 1;
        }

        let p_vec = self.observed_p_per_image.get_mut(img_id).unwrap();
        let old_val = p_vec[g];
        p_vec[g] = p_val;

        let delta_p = p_val - old_val;
        self.s[g] += delta_p;
        self.v[g] += p_val * (1.0 - p_val) - old_val * (1.0 - old_val);
        self.p_last[g] = p_val;

        if self.use_mutual_exclusion {
            for j in 0..self.m_classes {
                if j != g {
                    self.c[g][j] += delta_p * p_vec[j];
                    self.c[j][g] = self.c[g][j];
                }
            }
        }

        self.dirty = true;

        match self.config.update_scheme {
            QPUpdateScheme::AllGroupsEager => {
                self.refresh_all_bounds();
            }
            QPUpdateScheme::CandidateSetK2 => {
                let mut idx_and_s: Vec<(usize, f64)> = self.s.iter().cloned().enumerate().collect();
                idx_and_s.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                self.active_candidates.clear();
                for (idx, _) in idx_and_s.into_iter().take(self.config.candidate_k) {
                    self.active_candidates.insert(idx);
                }
                if self.active_candidates.contains(&g) {
                    self.refresh_candidates();
                }
            }
            QPUpdateScheme::ThresholdTriggered => {
                if (self.s[g] - self.last_update_means[g]).abs() >= self.config.threshold_delta {
                    self.refresh_all_bounds();
                    self.last_update_means = self.s.clone();
                }
            }
            QPUpdateScheme::FocalLazy | QPUpdateScheme::QueryDriven => {
                // Deferred until queried
            }
        }
    }

    fn unseen_count(&self) -> usize {
        self.n_tot.saturating_sub(self.m_images_seen)
    }

    pub fn get_pairwise_diff_bounds(&self, g: usize, j: usize) -> (f64, f64, f64, f64) {
        let r_ub = self.unseen_count() as f64;
        let diff_s = self.s[g] - self.s[j];

        let (obs_var_d, rem_diff_lb, rem_diff_ub, rem_var_max) = if self.use_mutual_exclusion {
            let obs = self.v[g] + self.v[j] + 2.0 * self.c[g][j];
            let r_lb = -r_ub * self.p_last[j];
            let r_ub_diff = r_ub * self.p_last[g];
            let r_var = r_ub * (self.p_last[g] + self.p_last[j]);
            (obs, r_lb, r_ub_diff, r_var)
        } else {
            let obs = self.v[g] + self.v[j];
            let r_lb = -r_ub * self.p_last[j];
            let r_ub_diff = r_ub * self.p_last[g];
            let r_var = r_ub * (self.p_last[g] * (1.0 - self.p_last[g]) + self.p_last[j] * (1.0 - self.p_last[j]));
            (obs, r_lb, r_ub_diff, r_var)
        };

        let mu_d_lb = diff_s + rem_diff_lb;
        let mu_d_ub = diff_s + rem_diff_ub;
        let var_d_lb = obs_var_d.max(1e-12);
        let var_d_ub = (obs_var_d + rem_var_max).max(1e-12);

        (mu_d_lb, mu_d_ub, var_d_lb, var_d_ub)
    }

    pub fn evaluate_group_bounds(&mut self, g: usize) -> QPWinnerBounds {
        let cid = self.class_ids[g].clone();
        let engine = self.config.engine_type;

        let mut pairwise_defeat_ubs = Vec::with_capacity(self.m_classes.saturating_sub(1));
        let mut pairwise_victory_ubs = Vec::with_capacity(self.m_classes.saturating_sub(1));

        for j in 0..self.m_classes {
            if j == g {
                continue;
            }
            let (mu_lb, _mu_ub, _var_lb, var_ub) = self.get_pairwise_diff_bounds(g, j);

            let p_def_ub = pairwise_defeat_upper_bound(mu_lb, var_ub, engine);
            pairwise_defeat_ubs.push((j, p_def_ub));

            let (mu_rev_lb, _, _, var_rev_ub) = self.get_pairwise_diff_bounds(j, g);
            let p_def_rev_ub = pairwise_defeat_upper_bound(mu_rev_lb, var_rev_ub, engine);
            let p_vic_ub = if mu_rev_lb > 0.0 { p_def_rev_ub } else { 1.0 };
            pairwise_victory_ubs.push(p_vic_ub);
        }

        let mut w_ub = if !pairwise_victory_ubs.is_empty() {
            pairwise_victory_ubs.into_iter().fold(1.0f64, f64::min)
        } else {
            1.0
        };

        let sum_defeat_ub: f64 = pairwise_defeat_ubs.iter().map(|(_, p)| *p).sum();
        let mut w_lb = (1.0 - sum_defeat_ub).max(0.0);

        // Tier 3: Local DP refinement for active contenders when adaptive or local_dp requested
        // Note: Local DP over exact observed distribution is valid when all items in partition are observed.
        if matches!(engine, QPEngineType::LocalDP | QPEngineType::AdaptiveMultiTier)
            && w_lb < 0.5
            && w_ub > 0.05
            && !self.observed_p_per_image.is_empty()
            && self.unseen_count() == 0
        {
            let m_obs = self.observed_p_per_image.len();
            let mut p_mat_g = Vec::with_capacity(m_obs);
            for p_vec in self.observed_p_per_image.values() {
                p_mat_g.push(p_vec[g]);
            }

            let mut refined_defeats = Vec::with_capacity(pairwise_defeat_ubs.len());
            for (j, _) in &pairwise_defeat_ubs {
                if self.use_mutual_exclusion {
                    let mut p_mat_j = Vec::with_capacity(m_obs);
                    for p_vec in self.observed_p_per_image.values() {
                        p_mat_j.push(p_vec[*j]);
                    }
                    let p_def_dp = compute_pairwise_defeat_local_dp(
                        &p_mat_g,
                        &p_mat_j,
                        self.config.c_truncation,
                    );
                    refined_defeats.push(p_def_dp);
                } else {
                    let (mu_lb, _, _, var_ub) = self.get_pairwise_diff_bounds(g, *j);
                    let p_def_dp = pairwise_defeat_upper_bound(mu_lb, var_ub, QPEngineType::Bennett);
                    refined_defeats.push(p_def_dp);
                }
            }

            let sum_refined: f64 = refined_defeats.iter().sum();
            let w_lb_refined = (1.0 - sum_refined).max(0.0);
            if w_lb_refined > w_lb {
                w_lb = w_lb_refined;
            }
        }

        w_lb = w_lb.max(0.0).min(1.0);
        w_ub = w_ub.max(0.0).min(1.0);
        if w_lb > w_ub {
            std::mem::swap(&mut w_lb, &mut w_ub);
        }

        let gap = (w_ub - w_lb).max(0.0);
        let res = QPWinnerBounds {
            group_id: cid.clone(),
            w_lb,
            w_ub,
            bound_gap: gap,
        };
        self.cached_bounds.insert(cid, res.clone());
        res
    }

    pub fn refresh_all_bounds(&mut self) {
        for g in 0..self.m_classes {
            self.evaluate_group_bounds(g);
        }
        self.dirty = false;
    }

    pub fn refresh_candidates(&mut self) {
        let candidates: Vec<usize> = self.active_candidates.iter().cloned().collect();
        for g in candidates {
            self.evaluate_group_bounds(g);
        }
        self.dirty = false;
    }

    pub fn get_bounds(&mut self, class_id: &str) -> (f64, f64) {
        if self.dirty || !self.cached_bounds.contains_key(class_id) {
            match self.config.update_scheme {
                QPUpdateScheme::QueryDriven | QPUpdateScheme::FocalLazy => {
                    self.refresh_all_bounds();
                }
                _ => {
                    if let Some(&idx) = self.class_to_idx.get(class_id) {
                        self.evaluate_group_bounds(idx);
                    } else {
                        return (0.0, 1.0);
                    }
                }
            }
        }

        if let Some(bounds) = self.cached_bounds.get(class_id) {
            (bounds.w_lb, bounds.w_ub)
        } else {
            (0.0, 1.0)
        }
    }

    pub fn check_probabilistic_convergence(
        &mut self,
        excluded_classes: Option<&HashSet<String>>,
    ) -> Option<(String, f64, f64)> {
        self.refresh_all_bounds();

        let mut active_bounds = Vec::new();
        for cid in &self.class_ids {
            if let Some(ex) = excluded_classes {
                if ex.contains(cid) {
                    continue;
                }
            }
            if let Some(b) = self.cached_bounds.get(cid) {
                active_bounds.push(b.clone());
            }
        }

        if active_bounds.is_empty() {
            return None;
        }

        // Sort by lower bound descending, breaking ties by running mean s[g]
        active_bounds.sort_by(|a, b| {
            let idx_a = self.class_to_idx[&a.group_id];
            let idx_b = self.class_to_idx[&b.group_id];
            let s_a = self.s[idx_a];
            let s_b = self.s[idx_b];

            b.w_lb
                .partial_cmp(&a.w_lb)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| s_b.partial_cmp(&s_a).unwrap_or(std::cmp::Ordering::Equal))
                .then_with(|| a.group_id.cmp(&b.group_id))
        });

        let top = &active_bounds[0];
        let max_other_ub = if active_bounds.len() > 1 {
            active_bounds[1..]
                .iter()
                .map(|b| b.w_ub)
                .fold(0.0f64, f64::max)
        } else {
            0.0
        };

        // Condition 1: Strict separation w_lb > max competitor w_ub
        // Condition 2: Absolute majority w_lb > 0.50 under mutual exclusion
        if top.w_lb > max_other_ub || (self.use_mutual_exclusion && top.w_lb > 0.50) {
            return Some((top.group_id.clone(), top.w_lb, top.w_ub));
        }

        // Terminal state: partition fully processed, disambiguate by highest point expectation
        if self.m_images_seen >= self.n_tot {
            return Some((top.group_id.clone(), top.w_lb, top.w_ub));
        }

        None
    }
}
