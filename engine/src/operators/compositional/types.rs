/// Core types for compositional aggregation queries.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StreamTuple {
    pub partition_key: String,
    pub image_id: String,
    pub class_id: String,
    pub score: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmittedTuple {
    pub partition_key: String,
    pub class_id: String,
    pub score_interval: (f64, f64),
    pub score_hi: f64,
    pub rank: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PartitionMetadata {
    pub partition_key: String,
    pub n_images: usize,
    pub total_cardinality: usize,
    pub image_ids: Vec<String>,
    pub n_classes: usize,
    pub total_stream_cardinality: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ContinuationPolicy {
    StopAfterFirst,
    DemandDriven,
}

/// Trait implemented by all inner partitioned operators.
pub trait InnerOperator {
    /// Demand-driven pull: requests partition to produce up to `rank` results.
    fn request_rank(&mut self, partition_key: &str, rank: usize);

    /// Demand-driven pull: requests partition to produce the next ranked result.
    fn request_next(&mut self, partition_key: &str);

    /// Pull next available tuple from inner operator.
    fn next_tuple(&mut self) -> Option<EmittedTuple>;

    /// Number of raw input stream items read so far.
    fn items_read(&self) -> usize;

    /// Total number of items in the underlying child stream.
    fn total_stream_items(&self) -> usize;

    /// Which partition keys have been fully resolved.
    fn is_partition_resolved(&self, partition_key: &str) -> bool;

    /// Number of ranks emitted so far for partition.
    fn num_emitted_ranks(&self, partition_key: &str) -> usize;

    /// Partition metadata lookup.
    fn partition_meta(&self, partition_key: &str) -> Option<&PartitionMetadata>;
}
