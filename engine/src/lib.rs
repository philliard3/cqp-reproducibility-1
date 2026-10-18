#![forbid(unsafe_code)]
/// mode_u_engine — Clean, minimal Canonical V3 implementation of the Mode U / CQP query engine.
///
/// Implements composable streaming Volcano operators with:
/// - Finite-sample conformal early stopping guarantees.
/// - Exact and proxy-sorted relative bound separation (QN1).
/// - Discrete rank-over-count (Q1, Q2) and continuous revenue aggregation (TPC-H Q3, Q10).
/// - 1000-world bitvector Monte Carlo simulation baseline (MCDB).

pub mod ids;
pub mod tuple;
pub mod operators;
pub mod statistical_estimation;

pub use tuple::{Tuple, FieldValue, SharedTuple, ProbInterval, LogProb};
pub use ids::{ImageId, ClassId};
pub use operators::compositional;
pub use operators::{
    ComposableArgMaxOverCount, ArgMaxConfig, AggregationMode, ItemBudgetSemantics, TupleStream,
    MemoryTupleStream, CompactKey,
    ComposableArgMaxOverSum, ArgMaxSumConfig, AggregationModeSum, CandidateStatsSum, LifetimeSnapshot,
    StreamTuple, EmittedTuple,
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use indexmap::IndexMap;
    use rustc_hash::FxHashMap;

    fn make_test_tuple(item: &str, candidate: &str, score: f64) -> Arc<Tuple> {
        let mut data = IndexMap::new();
        data.insert("item_id".to_string(), FieldValue::Str(item.to_string()));
        data.insert("candidate_id".to_string(), FieldValue::Str(candidate.to_string()));
        Arc::new(Tuple::new(data, score))
    }

    #[test]
    fn test_compact_key_interning() {
        use crate::ids::intern;
        let id1 = intern("dog");
        let id2 = intern("dog");
        assert_eq!(id1, id2, "String interning must produce identical IDs for equal strings");

        let k1 = CompactKey::single(id1);
        let k2 = CompactKey::single(id2);
        assert_eq!(k1, k2, "CompactKeys with identical interned IDs must be equal");
    }

    #[test]
    fn test_qn1_early_exit_soundness() {
        let n_items = 50;
        let mut tuples = Vec::new();
        for i in 0..n_items {
            let item = format!("item_{:02}", i);
            tuples.push(make_test_tuple(&item, "leader", 0.90));
            tuples.push(make_test_tuple(&item, "runner_up", 0.10));
        }

        let stream = Box::new(MemoryTupleStream::new(tuples));
        let mut card = FxHashMap::default();
        card.insert(CompactKey::empty(), n_items);

        let config = ArgMaxConfig {
            item_key: vec!["item_id".to_string()],
            count_group_by: vec!["candidate_id".to_string()],
            comparison_key: vec![],
            expected_cardinality: card,
            budget_semantics: ItemBudgetSemantics::MutuallyExclusive,
            mode: AggregationMode::QN1,
            k_target: 1,
        };

        let candidates = vec!["leader".to_string(), "runner_up".to_string()];
        let mut op = ComposableArgMaxOverCount::new(stream, config, candidates);

        let mut emitted = Vec::new();
        while let Some(t) = op.next_tuple() {
            emitted.push(t);
        }

        assert_eq!(emitted.len(), 1);
        assert_eq!(emitted[0].get_str("candidate_id").unwrap(), "leader");
        assert!(op.items_read < n_items * 2, "QN1 must terminate early on clear separation");
    }

    #[test]
    fn test_naive_exhaustion_baseline() {
        let n_items = 20;
        let mut tuples = Vec::new();
        for i in 0..n_items {
            let item = format!("item_{:02}", i);
            tuples.push(make_test_tuple(&item, "cand_a", 0.70));
            tuples.push(make_test_tuple(&item, "cand_b", 0.30));
        }

        let total_tuples = tuples.len();
        let stream = Box::new(MemoryTupleStream::new(tuples));
        let mut card = FxHashMap::default();
        card.insert(CompactKey::empty(), n_items);

        let config = ArgMaxConfig {
            item_key: vec!["item_id".to_string()],
            count_group_by: vec!["candidate_id".to_string()],
            comparison_key: vec![],
            expected_cardinality: card,
            budget_semantics: ItemBudgetSemantics::MutuallyExclusive,
            mode: AggregationMode::Naive,
            k_target: 1,
        };

        let candidates = vec!["cand_a".to_string(), "cand_b".to_string()];
        let mut op = ComposableArgMaxOverCount::new(stream, config, candidates);

        let mut emitted = Vec::new();
        while let Some(t) = op.next_tuple() {
            emitted.push(t);
        }

        assert_eq!(emitted.len(), 1);
        assert_eq!(emitted[0].get_str("candidate_id").unwrap(), "cand_a");
        assert_eq!(op.items_read, total_tuples, "Naive baseline must strictly scan full stream to exhaustion");
    }
}
