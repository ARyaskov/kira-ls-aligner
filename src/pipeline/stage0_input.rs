use crate::types::ReadRecord;

/// Stage 0 input batch.
#[derive(Clone, Debug)]
pub struct InputBatch {
    pub reads: Vec<ReadRecord>,
}

pub fn run(reads: Vec<ReadRecord>) -> InputBatch {
    InputBatch { reads }
}
