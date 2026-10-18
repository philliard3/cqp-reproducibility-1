//! Compositional aggregation operators for Canonical V3 queries.

pub mod types;
pub mod unified_argmax;
pub mod unified_argmax_sum;

pub use unified_argmax::{
    ComposableArgMaxOverCount, ArgMaxConfig, AggregationMode, TupleStream, MemoryTupleStream, CompactKey
};
pub use unified_argmax_sum::{
    ComposableArgMaxOverSum, ArgMaxSumConfig, AggregationModeSum, CandidateStatsSum, LifetimeSnapshot
};
pub use types::{StreamTuple, EmittedTuple, PartitionMetadata, ContinuationPolicy, InnerOperator};
