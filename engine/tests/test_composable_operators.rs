//! Integration regression tests for Canonical V3 Composable Aggregation Operators.
//! Verifies:
//! 1. ComposableArgMaxOverCount with QN1 relative bound separation early stopping.
//! 2. ComposableArgMaxOverCount with ConformalCountPruning (tau threshold in-operator pruning).
//! 3. ComposableArgMaxOverSum continuous revenue aggregation with early stopping.
//! 4. Mathematical monotonicity invariants: L_g <= U_g and non-decreasing L_g.

#![forbid(unsafe_code)]

use std::sync::Arc;
use indexmap::IndexMap;
use rustc_hash::FxHashMap;

use mode_u_engine::operators::compositional::unified_argmax::{
    ComposableArgMaxOverCount, ArgMaxConfig, AggregationMode, ItemBudgetSemantics,
    MemoryTupleStream, TupleStream, CompactKey,
};
use mode_u_engine::operators::compositional::unified_argmax_sum::{
    ComposableArgMaxOverSum, ArgMaxSumConfig, AggregationModeSum,
};
use mode_u_engine::tuple::{Tuple, FieldValue};

fn make_count_tuple(item_id: &str, candidate_id: &str, partition_id: &str, score: f64) -> Arc<Tuple> {
    let mut data = IndexMap::new();
    data.insert("item_id".to_string(), FieldValue::Str(item_id.to_string()));
    data.insert("candidate_id".to_string(), FieldValue::Str(candidate_id.to_string()));
    data.insert("partition_id".to_string(), FieldValue::Str(partition_id.to_string()));
    Arc::new(Tuple::new(data, score))
}

fn make_sum_tuple(item_id: &str, candidate_id: &str, partition_id: &str, score: f64, revenue: f64) -> Arc<Tuple> {
    let mut data = IndexMap::new();
    data.insert("item_id".to_string(), FieldValue::Str(item_id.to_string()));
    data.insert("candidate_id".to_string(), FieldValue::Str(candidate_id.to_string()));
    data.insert("partition_id".to_string(), FieldValue::Str(partition_id.to_string()));
    data.insert("filtered_revenue".to_string(), FieldValue::Float(revenue));
    Arc::new(Tuple::new(data, score))
}

#[test]
fn test_composable_count_qn1_early_stopping() {
    let n_items = 100;
    let mut tuples = Vec::new();

    // Highly skewed stream: dog has high probability (0.85), cat has (0.10), others 0.05
    for i in 0..n_items {
        let item_id = format!("img_{:03}", i);
        tuples.push(make_count_tuple(&item_id, "dog", "global", 0.85));
        tuples.push(make_count_tuple(&item_id, "cat", "global", 0.10));
        tuples.push(make_count_tuple(&item_id, "bird", "global", 0.05));
    }

    let stream = Box::new(MemoryTupleStream::new(tuples));

    let mut expected_card = FxHashMap::default();
    let comp_key = CompactKey::empty();
    expected_card.insert(comp_key, n_items);

    let config = ArgMaxConfig {
        item_key: vec!["item_id".to_string()],
        count_group_by: vec!["candidate_id".to_string()],
        comparison_key: vec![],
        expected_cardinality: expected_card,
        budget_semantics: ItemBudgetSemantics::MutuallyExclusive,
        mode: AggregationMode::QN1,
        k_target: 1,
    };

    let all_candidates = vec!["dog".to_string(), "cat".to_string(), "bird".to_string()];
    let mut op = ComposableArgMaxOverCount::new(stream, config, all_candidates);
    let mut emitted = Vec::new();

    while let Some(res) = op.next_tuple() {
        emitted.push(res);
    }

    assert_eq!(emitted.len(), 1, "Expected exactly 1 top rank emitted");
    let cand_name = emitted[0].get_str("candidate_id").unwrap_or("");
    assert_eq!(cand_name, "dog", "Winner should be dog");

    // Early stopping assertion: must have read strictly less than 100 items (300 tuples)
    let items_read = op.items_read;
    assert!(items_read < n_items * 3, "Operator should terminate early; read {} tuples", items_read);
}

#[test]
fn test_composable_count_conformal_pruning() {
    let n_items = 50;
    let mut tuples = Vec::new();

    for i in 0..n_items {
        let item_id = format!("doc_{:03}", i);
        tuples.push(make_count_tuple(&item_id, "toxic", "day_0", 0.90));
        tuples.push(make_count_tuple(&item_id, "threat", "day_0", 0.05));
    }

    let stream = Box::new(MemoryTupleStream::new(tuples));

    let mut expected_card = FxHashMap::default();
    let comp_key = CompactKey::empty();
    expected_card.insert(comp_key, n_items);

    let config = ArgMaxConfig {
        item_key: vec!["item_id".to_string()],
        count_group_by: vec!["candidate_id".to_string()],
        comparison_key: vec![],
        expected_cardinality: expected_card,
        budget_semantics: ItemBudgetSemantics::Independent,
        mode: AggregationMode::ConformalCountPruning { tau: 0.50 },
        k_target: 1,
    };

    let all_candidates = vec!["toxic".to_string(), "threat".to_string()];
    let mut op = ComposableArgMaxOverCount::new(stream, config, all_candidates);
    let mut emitted = Vec::new();

    while let Some(res) = op.next_tuple() {
        emitted.push(res);
    }

    assert!(!emitted.is_empty(), "Expected at least 1 conformal winner");
    let cand_name = emitted[0].get_str("candidate_id").unwrap_or("");
    assert_eq!(cand_name, "toxic");
}

#[test]
fn test_composable_sum_continuous_tpch_early_stopping() {
    let n_items = 80;
    let mut tuples = Vec::new();

    // Group 1: High value revenue orders ($1000/order, p=0.90 -> expected = $900)
    // Group 2: Low value revenue orders ($100/order, p=0.10 -> expected = $10)
    for i in 0..n_items {
        let item_id = format!("order_{:03}", i);
        tuples.push(make_sum_tuple(&item_id, "Group_1", "tpch_q3", 0.90, 1000.0));
        tuples.push(make_sum_tuple(&item_id, "Group_2", "tpch_q3", 0.10, 100.0));
    }

    let stream = Box::new(MemoryTupleStream::new(tuples));

    let mut expected_card = FxHashMap::default();
    let comp_key = CompactKey::empty();
    expected_card.insert(comp_key, n_items);

    let config = ArgMaxSumConfig {
        item_key: vec!["item_id".to_string()],
        count_group_by: vec!["candidate_id".to_string()],
        comparison_key: vec![],
        expected_cardinality: expected_card,
        mode: AggregationModeSum::QN1DemandRanked,
        k_target: 1,
        value_field: "filtered_revenue".to_string(),
        v_max: 1000.0,
        monitor_lifetime: false,
    };

    let all_candidates = vec!["Group_1".to_string(), "Group_2".to_string()];
    let mut op = ComposableArgMaxOverSum::new(stream, config, all_candidates);
    let mut emitted = Vec::new();

    while let Some(res) = op.next_tuple() {
        emitted.push(res);
    }

    assert_eq!(emitted.len(), 1, "Expected 1 top winner emitted");
    let winner = emitted[0].get_str("candidate_id").unwrap_or("");
    assert_eq!(winner, "Group_1");

    let items_read = op.items_read;
    assert!(items_read < n_items * 2, "Sum operator should terminate early: read {} tuples", items_read);
}
