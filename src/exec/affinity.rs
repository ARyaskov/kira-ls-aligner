//! Hybrid-CPU topology detection.
//!
//! Returns the set of logical CPU IDs that map to P-cores (Performance)
//! and E-cores (Efficient) on Intel Alder Lake / Raptor Lake / Meteor Lake
//! and equivalent designs. On homogeneous hosts (AMD, older Intel, Apple
//! Silicon when run via Rosetta, unknown topology) `detect_topology()`
//! returns `None` and the caller should fall back to a single pool.
//!
//! ## Backends
//!
//! * **Windows** — `GetLogicalProcessorInformationEx(RelationProcessorCore, …)`
//!   exposes a `PROCESSOR_RELATIONSHIP::EfficiencyClass` byte per physical
//!   core. EfficiencyClass = 0 marks E-cores; any higher value marks P-cores.
//!   The same call also gives the `GROUP_AFFINITY::Mask` listing the logical
//!   CPUs that belong to the core (so we capture both SMT siblings on P-cores).
//!
//! * **Linux** — `/sys/devices/system/cpu/cpu*/cpu_capacity` is exported by
//!   `arch/x86` and `arch/arm64`. The numeric capacity is lower for E-cores
//!   than P-cores (≈ 446 vs 1024 on Alder Lake P+E topologies). When the
//!   sysfs node is missing — older kernels or VMs — we report homogeneous.
//!
//! * **Other** — homogeneous (no detection).
//!
//! Detection runs once per process; results are not cached here because the
//! caller (`DualPool::new`) caches them naturally.

#[derive(Clone, Debug)]
pub struct HybridTopology {
    /// One entry per Performance physical core. Each inner vector lists
    /// the logical-CPU IDs that belong to that physical core (≥ 2 for an
    /// SMT-enabled P-core on Alder Lake, 1 when SMT is off).
    pub p_physical: Vec<Vec<usize>>,
    /// One entry per Efficient physical core. E-cores are not
    /// hyper-threaded, so each inner vector has length 1 on every
    /// platform we currently target.
    pub e_physical: Vec<Vec<usize>>,
}

impl HybridTopology {
    pub fn is_split(&self) -> bool {
        !self.p_physical.is_empty() && !self.e_physical.is_empty()
    }

    /// All P-class logical CPU IDs, SMT siblings included. Order matches
    /// the OS enumeration so callers see logical IDs paired by physical
    /// core (e.g. `[0, 1, 2, 3, …]` where each consecutive pair shares a
    /// core on Windows-on-Alder-Lake).
    pub fn p_logical(&self) -> Vec<usize> {
        self.p_physical.iter().flatten().copied().collect()
    }

    /// All E-class logical CPU IDs. Always one per physical core.
    pub fn e_logical(&self) -> Vec<usize> {
        self.e_physical.iter().flatten().copied().collect()
    }

    /// One representative logical ID per P-physical core (first SMT
    /// sibling). Use these when scheduling SIMD-heavy workloads where
    /// HT siblings would only cause execution-unit contention.
    pub fn p_primary(&self) -> Vec<usize> {
        self.p_physical
            .iter()
            .filter_map(|sibs| sibs.first().copied())
            .collect()
    }

    /// One representative logical ID per E-physical core. Identical to
    /// [`HybridTopology::e_logical`] on every platform we currently
    /// target since E-cores don't expose SMT.
    pub fn e_primary(&self) -> Vec<usize> {
        self.e_physical
            .iter()
            .filter_map(|sibs| sibs.first().copied())
            .collect()
    }

    pub fn n_p_physical(&self) -> usize {
        self.p_physical.len()
    }

    pub fn n_e_physical(&self) -> usize {
        self.e_physical.len()
    }

    pub fn total_logical(&self) -> usize {
        self.p_logical().len() + self.e_logical().len()
    }
}

/// Detect the host's hybrid topology. Returns `None` when the host is
/// homogeneous, the detection backend is unavailable, or the OS reports
/// only a single core class.
pub fn detect_topology() -> Option<HybridTopology> {
    #[cfg(windows)]
    {
        if let Some(t) = windows::detect() {
            return t.is_split().then_some(t);
        }
    }
    #[cfg(target_os = "linux")]
    {
        if let Some(t) = linux::detect() {
            return t.is_split().then_some(t);
        }
    }
    None
}

#[cfg(windows)]
mod windows {
    use super::HybridTopology;
    use std::mem::size_of;
    use windows_sys::Win32::System::SystemInformation::{
        GROUP_AFFINITY, GetLogicalProcessorInformationEx, LOGICAL_PROCESSOR_RELATIONSHIP,
        PROCESSOR_RELATIONSHIP, RelationProcessorCore, SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
    };

    pub fn detect() -> Option<HybridTopology> {
        let mut length: u32 = 0;
        // First call discovers the required buffer size.
        unsafe {
            let _ = GetLogicalProcessorInformationEx(
                RelationProcessorCore,
                std::ptr::null_mut(),
                &mut length,
            );
        }
        if length == 0 {
            return None;
        }
        let mut buf: Vec<u8> = vec![0u8; length as usize];
        let ok = unsafe {
            GetLogicalProcessorInformationEx(
                RelationProcessorCore,
                buf.as_mut_ptr() as *mut SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
                &mut length,
            )
        };
        if ok == 0 {
            return None;
        }

        let mut p_physical: Vec<Vec<usize>> = Vec::new();
        let mut e_physical: Vec<Vec<usize>> = Vec::new();
        let mut seen_eff_classes: Vec<u8> = Vec::new();

        // SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX entries are variable-length;
        // walk the buffer manually using the `Size` field.
        let mut offset: usize = 0;
        while offset + size_of::<LOGICAL_PROCESSOR_RELATIONSHIP>() <= buf.len() {
            let ptr = unsafe { buf.as_ptr().add(offset) }
                as *const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX;
            let entry = unsafe { &*ptr };
            let entry_size = entry.Size as usize;
            if entry_size == 0 || offset + entry_size > buf.len() {
                break;
            }
            if entry.Relationship == RelationProcessorCore {
                let proc_rel: &PROCESSOR_RELATIONSHIP = unsafe { &entry.Anonymous.Processor };
                let eff = proc_rel.EfficiencyClass;
                if !seen_eff_classes.contains(&eff) {
                    seen_eff_classes.push(eff);
                }
                // Each RelationProcessorCore entry describes ONE physical
                // core; its GroupMask lists the logical CPUs that belong
                // to that physical core (the SMT siblings on a P-core,
                // the single thread on an E-core).
                let group_count = proc_rel.GroupCount as usize;
                let groups_ptr = proc_rel.GroupMask.as_ptr();
                let mut siblings: Vec<usize> = Vec::new();
                for g in 0..group_count {
                    let ga: &GROUP_AFFINITY = unsafe { &*groups_ptr.add(g) };
                    let group_base = (ga.Group as usize) * 64;
                    let mut mask = ga.Mask as u64;
                    while mask != 0 {
                        let bit = mask.trailing_zeros() as usize;
                        siblings.push(group_base + bit);
                        mask &= mask - 1;
                    }
                }
                siblings.sort_unstable();
                if eff == 0 {
                    e_physical.push(siblings);
                } else {
                    p_physical.push(siblings);
                }
            }
            offset += entry_size;
        }

        if std::env::var_os("KIRA_AFFINITY_DEBUG").is_some() {
            eprintln!(
                "[KIRA_AFFINITY] win32: efficiency_classes={:?} p_physical={:?} e_physical={:?}",
                seen_eff_classes, p_physical, e_physical
            );
        }

        // If Windows only reported a single efficiency class, the host is
        // not hybrid (e.g. all-P AMD, all-E mobile chip). Collapse to
        // homogeneous so the caller falls back to a single pool.
        if seen_eff_classes.len() < 2 {
            return None;
        }
        Some(HybridTopology {
            p_physical,
            e_physical,
        })
    }
}

#[cfg(target_os = "linux")]
mod linux {
    use super::HybridTopology;
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::PathBuf;

    /// Parse a Linux `thread_siblings_list` (e.g. "0,1" or "0-3").
    fn parse_siblings(s: &str) -> Vec<usize> {
        let mut out = Vec::new();
        for part in s.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            if let Some((a, b)) = part.split_once('-') {
                if let (Ok(lo), Ok(hi)) = (a.trim().parse::<usize>(), b.trim().parse::<usize>()) {
                    for v in lo..=hi {
                        out.push(v);
                    }
                }
            } else if let Ok(v) = part.parse::<usize>() {
                out.push(v);
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    pub fn detect() -> Option<HybridTopology> {
        let cpu_dir = PathBuf::from("/sys/devices/system/cpu");
        let entries = fs::read_dir(&cpu_dir).ok()?;
        // (cpu_id, capacity, siblings)
        let mut rows: Vec<(usize, u64, Vec<usize>)> = Vec::new();
        for ent in entries.flatten() {
            let name = ent.file_name();
            let name = name.to_string_lossy();
            if !name.starts_with("cpu") {
                continue;
            }
            let id_str = &name[3..];
            let Ok(cpu_id) = id_str.parse::<usize>() else {
                continue;
            };
            let cap_path = ent.path().join("cpu_capacity");
            let Ok(cap_str) = fs::read_to_string(&cap_path) else {
                continue;
            };
            let Ok(cap) = cap_str.trim().parse::<u64>() else {
                continue;
            };
            let sib_path = ent.path().join("topology/thread_siblings_list");
            let siblings = fs::read_to_string(&sib_path)
                .ok()
                .map(|s| parse_siblings(s.trim()))
                .unwrap_or_else(|| vec![cpu_id]);
            rows.push((cpu_id, cap, siblings));
        }
        if rows.is_empty() {
            return None;
        }
        // Two distinct capacities ⇒ hybrid. Highest = P, lower = E.
        let mut caps: Vec<u64> = rows.iter().map(|(_, c, _)| *c).collect();
        caps.sort_unstable();
        caps.dedup();
        if caps.len() < 2 {
            return None;
        }
        let p_cap = *caps.last().unwrap();

        // Group logical CPUs by their thread_siblings_list — each
        // distinct sibling set is one physical core.
        let mut by_physical: BTreeMap<Vec<usize>, (u64, Vec<usize>)> = BTreeMap::new();
        for (cpu_id, cap, siblings) in rows {
            let entry = by_physical
                .entry(siblings.clone())
                .or_insert_with(|| (cap, Vec::new()));
            entry.1.push(cpu_id);
        }

        let mut p_physical: Vec<Vec<usize>> = Vec::new();
        let mut e_physical: Vec<Vec<usize>> = Vec::new();
        for (_siblings, (cap, mut logicals)) in by_physical {
            logicals.sort_unstable();
            logicals.dedup();
            if cap == p_cap {
                p_physical.push(logicals);
            } else {
                e_physical.push(logicals);
            }
        }
        Some(HybridTopology {
            p_physical,
            e_physical,
        })
    }
}

#[cfg(test)]
#[path = "../../tests/unit/exec_affinity.rs"]
mod tests;
