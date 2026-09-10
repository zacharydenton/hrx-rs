use std::sync::atomic::{AtomicU64, Ordering};
/// Cumulative counters for one coordinated runtime. Native low-level calls are
/// outside this accounting domain. Reads are snapshots, not global barriers.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct Statistics {
    /// Successfully allocated tracked buffers.
    pub allocations: u64,
    /// Shared dma-buf imports created during allocation.
    pub imports: u64,
    /// Accepted graph submissions.
    pub submissions: u64,
    /// Terminal graph submissions, including cancellation and failure.
    pub completions: u64,
    /// Explicit copy bytes executed by graph nodes.
    pub copied_bytes: u64,
    /// Byte extents passed to cross-engine cache maintenance.
    pub cache_maintenance_bytes: u64,
    /// Bytes whose tracked allocations are still retained.
    pub live_bytes: u64,
    /// Maximum retained allocation bytes observed by this runtime.
    pub peak_bytes: u64,
}
#[derive(Default)]
pub(super) struct Counters {
    pub allocations: AtomicU64,
    pub imports: AtomicU64,
    pub submissions: AtomicU64,
    pub completions: AtomicU64,
    pub copied_bytes: AtomicU64,
    pub cache_maintenance_bytes: AtomicU64,
    pub live_bytes: AtomicU64,
    pub peak_bytes: AtomicU64,
}
impl Counters {
    pub fn snapshot(&self) -> Statistics {
        Statistics {
            allocations: self.allocations.load(Ordering::Relaxed),
            imports: self.imports.load(Ordering::Relaxed),
            submissions: self.submissions.load(Ordering::Relaxed),
            completions: self.completions.load(Ordering::Relaxed),
            copied_bytes: self.copied_bytes.load(Ordering::Relaxed),
            cache_maintenance_bytes: self.cache_maintenance_bytes.load(Ordering::Relaxed),
            live_bytes: self.live_bytes.load(Ordering::Relaxed),
            peak_bytes: self.peak_bytes.load(Ordering::Relaxed),
        }
    }
}
