/// Statistical estimation module for Mode U.
///
/// Implements 1D LocalDP recurrences, closed-form concentration bounds
/// (Bennett, Bernstein, Hoeffding, Normal CLT), and partition-level state tracking.

pub mod bounds;
pub mod local_dp;
pub mod tracker;

pub use bounds::{bennett_h, norm_cdf, pairwise_defeat_upper_bound};
pub use local_dp::{compute_pairwise_pmf_dp, compute_pairwise_defeat_local_dp, compute_poisson_binomial_pmf};
pub use tracker::QPPartitionTracker;

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QPEngineType {
    NormalCLT,
    Bennett,
    Bernstein,
    Hoeffding,
    LocalDP,
    AdaptiveMultiTier,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QPUpdateScheme {
    AllGroupsEager,
    FocalLazy,
    CandidateSetK2,
    ThresholdTriggered,
    QueryDriven,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DisambiguationCriterion {
    ProbabilitySeparation,     // Option 1
    ExpectedValueSeparation,   // Option 2
    Both,                      // Option 3
    Either,                    // Option 4
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QPConfig {
    pub engine_type: QPEngineType,
    pub update_scheme: QPUpdateScheme,
    pub disambiguation_criterion: DisambiguationCriterion,
    pub candidate_k: usize,
    pub threshold_delta: f64,
    pub c_truncation: f64,
}

impl Default for QPConfig {
    fn default() -> Self {
        QPConfig {
            engine_type: QPEngineType::AdaptiveMultiTier,
            update_scheme: QPUpdateScheme::FocalLazy,
            disambiguation_criterion: DisambiguationCriterion::Either,
            candidate_k: 2,
            threshold_delta: 1.0,
            c_truncation: 3.0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QPWinnerBounds {
    pub group_id: String,
    pub w_lb: f64,
    pub w_ub: f64,
    pub bound_gap: f64,
}
