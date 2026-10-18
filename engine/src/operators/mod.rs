//! Mode U / CQP Query Engine Operators.
//!
//! Provides the core Canonical V3 composable operators for flat and nested rank-over-count
//! and continuous revenue aggregation queries.

pub mod compositional;

pub use compositional::unified_argmax::{
    ComposableArgMaxOverCount, ArgMaxConfig, AggregationMode, ItemBudgetSemantics, TupleStream,
    MemoryTupleStream, CompactKey,
};
pub use compositional::unified_argmax_sum::{
    ComposableArgMaxOverSum, ArgMaxSumConfig, AggregationModeSum, CandidateStatsSum, LifetimeSnapshot,
};
pub use compositional::types::{
    StreamTuple, EmittedTuple, PartitionMetadata, ContinuationPolicy, InnerOperator,
};
