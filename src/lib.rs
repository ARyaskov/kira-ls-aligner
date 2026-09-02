// Do not add a `#[global_allocator]` here: it is imposed on the whole process of
// every consumer. Installing mimalloc in this library once broke a downstream
// tool, which started writing empty output while still exiting 0. It lives in
// `src/bin/kira_ls_aligner.rs` instead.

pub mod aligner_core;
pub mod alignment;
pub mod chaining;
pub mod cli;
pub mod eval;
pub mod exec;
pub mod index;
pub mod io;
pub mod log;
pub mod mapq;
pub mod pipeline;
pub mod seeding;
pub mod seq;
pub mod simd;
pub mod sketch;
pub mod types;

#[cfg(feature = "cuda")]
pub mod cuda;
