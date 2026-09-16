//! Bounded, single-flight preparation with idle-only LRU eviction.

use crate::{Error, Result};
use std::{
    collections::HashMap,
    hash::Hash,
    sync::{Arc, Condvar, Mutex},
};

/// Complete identity for a cache shared across artifacts and compiler profiles.
/// Model-local caches may instead use shape keys when their owner fixes all
/// other fields for the cache's entire lifetime.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PlanKey {
    /// Digest of the model artifact and its packing format.
    pub artifact: String,
    /// Actual device architecture and required features.
    pub target: String,
    /// Compiler version and compilation options identity.
    pub compiler: String,
    /// Model-specific specialization identity, including scalar arguments.
    pub specialization: String,
    /// Input shapes, element types, strides and layouts.
    pub inputs: Vec<crate::tensor::TensorDesc>,
    /// Output shapes, element types, strides and layouts.
    pub outputs: Vec<crate::tensor::TensorDesc>,
}

type Outcome<V> = std::result::Result<Arc<V>, Arc<Error>>;
struct Flight<V> {
    result: Mutex<Option<Outcome<V>>>,
    changed: Condvar,
}
enum Entry<V> {
    Loading(Arc<Flight<V>>),
    Ready { value: Arc<V>, used: u64 },
}
struct State<K, V> {
    entries: HashMap<K, Entry<V>>,
    clock: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
}

/// Cache accounting. A miss is one elected preparation, not every waiter.
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct CacheStatistics {
    /// Current ready and in-progress entries.
    pub entries: usize,
    /// Requests reusing ready or in-progress preparation.
    pub hits: u64,
    /// Elected preparations, including failures.
    pub misses: u64,
    /// Idle entries removed to admit another shape.
    pub evictions: u64,
}

/// Concurrent bounded preparation. Returned `Arc`s pin values. The additional
/// idle predicate protects work whose output or completion outlives that Arc.
/// Failed or panicking loaders release capacity and wake waiters for retry.
pub struct PlanCache<K, V> {
    capacity: usize,
    idle: fn(&V) -> bool,
    state: Mutex<State<K, V>>,
}
impl<K: Clone + Eq + Hash, V> PlanCache<K, V> {
    /// Set the maximum number of resident/preparing plans, never zero.
    pub fn new(capacity: usize, idle: fn(&V) -> bool) -> Result<Self> {
        if capacity == 0 {
            return Err(Error::Message("plan cache capacity must be nonzero".into()));
        }
        Ok(Self {
            capacity,
            idle,
            state: Mutex::new(State {
                entries: HashMap::new(),
                clock: 0,
                hits: 0,
                misses: 0,
                evictions: 0,
            }),
        })
    }

    /// Reuse or prepare a value. Concurrent requests for the same key share one
    /// outcome; other keys may prepare concurrently. Returns `Busy` when all
    /// capacity is pinned or preparing. No loader executes under the cache lock.
    pub fn get_or_prepare(&self, key: K, load: impl FnOnce() -> Result<V>) -> Result<Arc<V>> {
        let (flight, elected) = {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            state.clock = state.clock.saturating_add(1);
            let used = state.clock;
            if state.entries.contains_key(&key) {
                state.hits += 1;
                match state.entries.get_mut(&key).expect("entry checked") {
                    Entry::Ready { value, used: last } => {
                        *last = used;
                        return Ok(value.clone());
                    }
                    Entry::Loading(flight) => (flight.clone(), false),
                }
            } else {
                if state.entries.len() == self.capacity {
                    let victim = state
                        .entries
                        .iter()
                        .filter_map(|(key, entry)| match entry {
                            Entry::Ready { value, used }
                                if Arc::strong_count(value) == 1 && (self.idle)(value) =>
                            {
                                Some((key.clone(), *used))
                            }
                            _ => None,
                        })
                        .min_by_key(|(_, used)| *used)
                        .map(|(key, _)| key);
                    let Some(victim) = victim else {
                        return Err(Error::Busy(
                            "all shape plans are pinned or preparing".into(),
                        ));
                    };
                    state.entries.remove(&victim);
                    state.evictions += 1;
                }
                let flight = Arc::new(Flight {
                    result: Mutex::new(None),
                    changed: Condvar::new(),
                });
                state
                    .entries
                    .insert(key.clone(), Entry::Loading(flight.clone()));
                state.misses += 1;
                (flight, true)
            }
        };
        if elected {
            let result = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(load)) {
                Ok(result) => result.map(Arc::new).map_err(Arc::new),
                Err(_) => Err(Arc::new(Error::Message(
                    "shape preparation panicked".into(),
                ))),
            };
            {
                let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
                match &result {
                    Ok(value) => {
                        let used = state.clock;
                        state.entries.insert(
                            key,
                            Entry::Ready {
                                value: value.clone(),
                                used,
                            },
                        );
                    }
                    Err(_) => {
                        state.entries.remove(&key);
                    }
                }
                *flight.result.lock().unwrap_or_else(|e| e.into_inner()) = Some(result.clone());
            }
            flight.changed.notify_all();
            result.map_err(|source| Error::Execution { source })
        } else {
            let mut result = flight.result.lock().unwrap_or_else(|e| e.into_inner());
            while result.is_none() {
                result = flight
                    .changed
                    .wait(result)
                    .unwrap_or_else(|e| e.into_inner());
            }
            result
                .as_ref()
                .expect("published above")
                .clone()
                .map_err(|source| Error::Execution { source })
        }
    }

    /// Snapshot cache counters without waiting for preparation.
    pub fn statistics(&self) -> CacheStatistics {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        CacheStatistics {
            entries: state.entries.len(),
            hits: state.hits,
            misses: state.misses,
            evictions: state.evictions,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pinned_values_apply_backpressure_and_idle_entries_are_lru() {
        let cache = PlanCache::new(2, |_: &usize| true).unwrap();
        let pinned = cache.get_or_prepare(1, || Ok(1)).unwrap();
        drop(cache.get_or_prepare(2, || Ok(2)).unwrap());
        drop(cache.get_or_prepare(3, || Ok(3)).unwrap());
        assert_eq!(cache.statistics().evictions, 1);
        let third = cache.get_or_prepare(3, || unreachable!()).unwrap();
        assert!(matches!(
            cache.get_or_prepare(4, || Ok(4)),
            Err(Error::Busy(_))
        ));
        drop((pinned, third));
        drop(cache.get_or_prepare(4, || Ok(4)).unwrap());
        assert_eq!(cache.statistics().entries, 2);
    }
    #[test]
    fn concurrent_preparation_is_single_flight_and_failures_retry() {
        let cache = PlanCache::new(1, |_: &usize| true).unwrap();
        let starts = std::sync::atomic::AtomicUsize::new(0);
        let barrier = std::sync::Barrier::new(8);
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    barrier.wait();
                    assert_eq!(
                        *cache
                            .get_or_prepare(1, || {
                                starts.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                Ok(7)
                            })
                            .unwrap(),
                        7
                    );
                });
            }
        });
        assert_eq!(starts.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert!(
            cache
                .get_or_prepare(2, || Err(Error::Message("retry".into())))
                .is_err()
        );
        assert!(
            cache
                .get_or_prepare(2, || panic!("injected loader panic"))
                .is_err()
        );
        assert_eq!(*cache.get_or_prepare(2, || Ok(9)).unwrap(), 9);
    }
}
