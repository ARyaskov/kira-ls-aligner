//! Hybrid-aware execution: P-core / E-core topology detection and routing
//! of pipeline stages to two independent rayon thread pools.
//!
//! On Intel Alder Lake+ (i7-12700, i9-13900K, …) the OS sees a heterogeneous
//! set of cores: high-frequency Performance cores (AVX2 at full throughput)
//! and lower-power Efficient cores (AVX2 throughput is lower, no SMT).
//! Default rayon scheduling is hybrid-unaware, so the OS may park the
//! SIMD-heavy banded Smith-Waterman path on E-cores while leaving P-cores
//! idle on light bookkeeping work.
//!
//! [`DualPool`] solves this by maintaining two pools:
//!   * P-pool  — pinned to the P-cores, drives the SIMD-heavy stages
//!     (Aho-Corasick batch, minimizer sketching, banded SW alignment).
//!   * E-pool  — pinned to the E-cores, drives the lighter structural
//!     stages (seeding, chaining, MAPQ scoring, SAM emit).
//!
//! When the host is homogeneous (AMD, older Intel, unknown topology),
//! [`DualPool::homogeneous`] degrades gracefully to a single pool covering
//! every logical CPU — so non-hybrid runs see no regression.

pub mod affinity;
pub mod pool;

pub use affinity::{HybridTopology, detect_topology};
pub use pool::DualPool;
