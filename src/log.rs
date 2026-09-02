//! Process-wide verbosity, bwa-mem `-v` semantics.
//!
//! Levels follow bwa: `1` errors only, `2` warnings, `3` messages (default),
//! `4` debug. The value is a plain atomic read on every log call, so gating a
//! message costs nothing measurable; the formatting itself is skipped when the
//! message is below the threshold.

use std::sync::atomic::{AtomicU8, Ordering};

static VERBOSITY: AtomicU8 = AtomicU8::new(3);

/// Set the verbosity level (clamped to `1..=4`).
pub fn set_verbosity(level: u8) {
    VERBOSITY.store(level.clamp(1, 4), Ordering::Relaxed);
}

/// Current verbosity level.
#[inline]
pub fn verbosity() -> u8 {
    VERBOSITY.load(Ordering::Relaxed)
}

/// Informational message (`-v 3`, the default).
#[macro_export]
macro_rules! kira_info {
    ($($arg:tt)*) => {
        if $crate::log::verbosity() >= 3 {
            eprintln!($($arg)*);
        }
    };
}

/// Warning (`-v 2`).
#[macro_export]
macro_rules! kira_warn {
    ($($arg:tt)*) => {
        if $crate::log::verbosity() >= 2 {
            eprintln!($($arg)*);
        }
    };
}

/// Debug trace (`-v 4`).
#[macro_export]
macro_rules! kira_debug {
    ($($arg:tt)*) => {
        if $crate::log::verbosity() >= 4 {
            eprintln!($($arg)*);
        }
    };
}
