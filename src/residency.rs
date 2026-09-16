//! Opt-in, byte-budgeted model residency with transactional loader reservations.
//!
//! The budget covers declared resident bytes and peak loader reservations, not
//! process RSS, compiler memory, or unregistered allocations. Register shared
//! weights/corpora once; private scratch is a separate resource or part of its
//! model's reservation. Native work must be drained before a resource is idle.
//! Declared storage must not grow beyond its reservation. Allocation-budgeted
//! resources may grow while leased, charging every allocation before it occurs.

use crate::{Error, Result};
use std::{
    any::Any,
    collections::HashMap,
    ops::Deref,
    sync::{
        Arc, Condvar, Mutex, Weak,
        atomic::{AtomicUsize, Ordering},
    },
};

type Value = dyn Any + Send + Sync;
struct Resource {
    value: Arc<Value>,
    bytes: usize,
    idle: Box<dyn Fn(&Value) -> bool + Send + Sync>,
    pins: AtomicUsize,
    // Release the charge only after the native value's destructor finishes.
    _reservation: MemoryReservation,
}
type Outcome = std::result::Result<Arc<Resource>, Arc<Error>>;
struct Flight {
    result: Mutex<Option<Outcome>>,
    changed: Condvar,
}
enum Entry {
    Loading(Arc<Flight>),
    Ready { resource: Arc<Resource>, used: u64 },
}
struct State {
    entries: HashMap<String, Entry>,
    clock: u64,
    evictions: u64,
}
struct Inner {
    usage: Arc<Usage>,
    state: Mutex<State>,
}

struct Usage {
    limit: usize,
    bytes: AtomicUsize,
}

/// The allocation side of a residency manager. It shares the byte ceiling and
/// may evict idle cached resources, but does not strongly retain the cache.
/// Models can own this handle without forming a manager/model ownership cycle.
/// The ceiling remains enforced even after the manager itself is dropped.
#[derive(Clone)]
pub struct MemoryBudget {
    usage: Arc<Usage>,
    registry: Weak<Inner>,
}
impl std::fmt::Debug for MemoryBudget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryBudget")
            .field("limit", &self.usage.limit)
            .field("reserved_bytes", &self.reserved_bytes())
            .finish()
    }
}

/// A non-cloneable charge for owned memory or a pending allocation. Reserve
/// before allocating and retain this guard until native storage is released,
/// including every alias, queued use and failed-device quarantine.
pub struct MemoryReservation {
    usage: Arc<Usage>,
    bytes: usize,
}
impl MemoryReservation {
    /// Number of bytes held by this reservation.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    fn shrink(&mut self, bytes: usize) {
        assert!(bytes <= self.bytes);
        self.usage
            .bytes
            .fetch_sub(self.bytes - bytes, Ordering::AcqRel);
        self.bytes = bytes;
    }
}
impl Drop for MemoryReservation {
    fn drop(&mut self) {
        self.usage.bytes.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}
impl Usage {
    fn reserve(self: &Arc<Self>, bytes: usize) -> Option<MemoryReservation> {
        self.bytes
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                used.checked_add(bytes).filter(|&total| total <= self.limit)
            })
            .ok()
            .map(|_| MemoryReservation {
                usage: self.clone(),
                bytes,
            })
    }
}
impl State {
    fn evict_idle(&mut self) -> Option<Arc<Resource>> {
        let victim = self
            .entries
            .iter()
            .filter_map(|(key, entry)| match entry {
                Entry::Ready { resource, used }
                    if Arc::strong_count(resource) == 1
                        && resource.pins.load(Ordering::Acquire) == 0
                        && (resource.idle)(resource.value.as_ref()) =>
                {
                    Some((key.clone(), *used))
                }
                _ => None,
            })
            .min_by_key(|(_, used)| *used)
            .map(|(key, _)| key)?;
        let Some(Entry::Ready { resource, .. }) = self.entries.remove(&victim) else {
            unreachable!("selected ready resource");
        };
        self.evictions += 1;
        Some(resource)
    }
}
impl MemoryBudget {
    /// Recover the live cache owning this ceiling, if it has not been dropped.
    /// Keep this owner outside cached values to avoid a cache/value cycle.
    pub fn manager(&self) -> Option<ResidencyManager> {
        self.registry
            .upgrade()
            .map(|inner| ResidencyManager { inner })
    }
    pub(crate) fn contains(&self, reservation: &MemoryReservation) -> bool {
        Arc::ptr_eq(&self.usage, &reservation.usage)
    }
    /// Reserve capacity before allocation. Idle model/corpus entries may be
    /// evicted, but live reservations are never reclaimed. A failed allocation
    /// rolls back by dropping the returned guard. Zero bytes are allowed.
    pub fn reserve(&self, bytes: usize) -> Result<MemoryReservation> {
        if bytes > self.usage.limit {
            return Err(Error::Message("allocation exceeds residency budget".into()));
        }
        loop {
            if let Some(reservation) = self.usage.reserve(bytes) {
                return Ok(reservation);
            }
            let victim = self.registry.upgrade().and_then(|registry| {
                registry
                    .state
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .evict_idle()
            });
            let Some(victim) = victim else {
                return Err(Error::Busy("residency budget is pinned or loading".into()));
            };
            // Native destruction may block. Keep the charge until it completes
            // without holding the registry lock or blocking unrelated lookups.
            drop(victim);
        }
    }

    /// Resident, loading and live allocation bytes sharing this ceiling.
    pub fn reserved_bytes(&self) -> usize {
        self.usage.bytes.load(Ordering::Acquire)
    }
}

/// One explicit budget shared by independently typed model units.
#[derive(Clone)]
pub struct ResidencyManager {
    inner: Arc<Inner>,
}

/// Accounting snapshot; loading reservations are included in `reserved_bytes`.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct ResidencyStatistics {
    /// Fixed byte ceiling selected by the caller.
    pub budget_bytes: usize,
    /// Resident resource bytes plus active peak loader reservations.
    pub reserved_bytes: usize,
    /// Number of ready and loading resources.
    pub resources: usize,
    /// Number of idle resources evicted for capacity.
    pub evictions: u64,
}

/// A strong resource lease. It prevents eviction and may outlive the manager.
pub struct ModelLease<T> {
    value: Arc<T>,
    resource: Arc<Resource>,
}
impl<T> Clone for ModelLease<T> {
    fn clone(&self) -> Self {
        Self {
            resource: self.resource.clone(),
            value: self.value.clone(),
        }
    }
}
impl<T> Deref for ModelLease<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.value
    }
}
impl<T> ModelLease<T> {
    /// Declared resident bytes. Zero for allocation-budgeted resources, whose
    /// changing storage is charged by their individual allocation owners.
    pub fn bytes(&self) -> usize {
        self.resource.bytes
    }
    /// Keep a corpus or model pinned after this request lease is dropped.
    pub fn pin(&self) -> ResidencyPin {
        self.resource.pins.fetch_add(1, Ordering::AcqRel);
        ResidencyPin {
            resource: self.resource.clone(),
        }
    }
    /// Retain this resource from an output or a native execution owner.
    pub fn keepalive(&self) -> Arc<dyn Send + Sync> {
        self.resource.clone()
    }
}
/// Explicit persistent residency pin; dropping it permits idle eviction.
pub struct ResidencyPin {
    resource: Arc<Resource>,
}
impl Drop for ResidencyPin {
    fn drop(&mut self) {
        self.resource.pins.fetch_sub(1, Ordering::AcqRel);
    }
}

impl ResidencyManager {
    /// Create a nonzero explicit storage budget. Nothing is allocated or loaded.
    pub fn new(budget_bytes: usize) -> Result<Self> {
        if budget_bytes == 0 {
            return Err(Error::Message("residency budget must be nonzero".into()));
        }
        Ok(Self {
            inner: Arc::new(Inner {
                usage: Arc::new(Usage {
                    limit: budget_bytes,
                    bytes: AtomicUsize::new(0),
                }),
                state: Mutex::new(State {
                    entries: HashMap::new(),
                    clock: 0,
                    evictions: 0,
                }),
            }),
        })
    }

    /// Use this manager's ceiling for separately owned runtime allocations.
    /// Do not also include those same bytes in a cached resource's declaration.
    pub fn budget(&self) -> MemoryBudget {
        MemoryBudget {
            usage: self.inner.usage.clone(),
            registry: Arc::downgrade(&self.inner),
        }
    }

    /// Load one unit or reuse its existing owner. Keys must include artifact,
    /// packing, target and specialization identity. `peak_bytes` reserves all
    /// declared storage that may coexist during loading; the loader returns
    /// `(value, resident_bytes)` with resident bytes no greater than the peak.
    ///
    /// The idle predicate additionally protects exported tensors and queued work
    /// after request leases disappear. Concurrent requests for one key share one
    /// load and outcome. Failure/panic rolls back the reservation and permits
    /// retry. Exhaustion with no idle unpinned victims returns `Busy`.
    pub fn load<T: Any + Send + Sync>(
        &self,
        key: impl Into<String>,
        peak_bytes: usize,
        idle: fn(&T) -> bool,
        loader: impl FnOnce() -> Result<(T, usize)>,
    ) -> Result<ModelLease<T>> {
        if peak_bytes == 0 || peak_bytes > self.inner.usage.limit {
            return Err(Error::Message(
                "loader reservation exceeds residency budget or is zero".into(),
            ));
        }
        self.load_inner(key.into(), peak_bytes, false, idle, loader)
    }

    /// Cache a resource whose allocations each hold their own budget charge.
    /// The loader receives this manager's budget; use it for every owned native
    /// allocation, including later workspace growth. Do not also declare these
    /// bytes through `load`. An active lease pins the resource while it grows.
    ///
    /// As with `load`, the idle predicate must exclude queued native work and
    /// exported owners. Only return an idle resource after draining its stream.
    pub fn load_budgeted<T: Any + Send + Sync>(
        &self,
        key: impl Into<String>,
        idle: fn(&T) -> bool,
        loader: impl FnOnce(MemoryBudget) -> Result<T>,
    ) -> Result<ModelLease<T>> {
        self.load_inner(key.into(), 0, true, idle, || {
            loader(self.budget()).map(|v| (v, 0))
        })
    }

    /// Remove an idle cached entry. Live leases, pins, native work and loaders
    /// return `Busy`; absent keys return false. Destruction is outside the lock.
    pub fn evict(&self, key: &str) -> Result<bool> {
        let victim = {
            let mut state = self.inner.state.lock().unwrap_or_else(|e| e.into_inner());
            match state.entries.get(key) {
                None => return Ok(false),
                Some(Entry::Ready { resource, .. })
                    if Arc::strong_count(resource) == 1
                        && resource.pins.load(Ordering::Acquire) == 0
                        && (resource.idle)(resource.value.as_ref()) => {}
                _ => return Err(Error::Busy("resident resource is in use".into())),
            }
            state.evictions += 1;
            state.entries.remove(key)
        };
        drop(victim);
        Ok(true)
    }

    fn load_inner<T: Any + Send + Sync>(
        &self,
        key: String,
        peak_bytes: usize,
        budgeted: bool,
        idle: fn(&T) -> bool,
        loader: impl FnOnce() -> Result<(T, usize)>,
    ) -> Result<ModelLease<T>> {
        let (flight, reservation) = loop {
            let mut state = self.inner.state.lock().unwrap_or_else(|e| e.into_inner());
            state.clock = state.clock.saturating_add(1);
            let used = state.clock;
            if let Some(entry) = state.entries.get_mut(&key) {
                match entry {
                    Entry::Ready {
                        resource,
                        used: last,
                    } => {
                        *last = used;
                        return lease(resource.clone());
                    }
                    Entry::Loading(flight) => break (flight.clone(), None),
                }
            } else {
                let Some(reservation) = self.inner.usage.reserve(peak_bytes) else {
                    let Some(victim) = state.evict_idle() else {
                        return Err(Error::Busy("residency budget is pinned or loading".into()));
                    };
                    drop(state);
                    drop(victim);
                    continue;
                };
                let flight = Arc::new(Flight {
                    result: Mutex::new(None),
                    changed: Condvar::new(),
                });
                state
                    .entries
                    .insert(key.clone(), Entry::Loading(flight.clone()));
                break (flight, Some(reservation));
            }
        };
        let resource = if let Some(mut reservation) = reservation {
            let result = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(loader)) {
                Ok(Ok((value, bytes))) if (budgeted || bytes > 0) && bytes <= peak_bytes => {
                    reservation.shrink(bytes);
                    Ok(Arc::new(Resource {
                        value: Arc::new(value),
                        bytes,
                        pins: AtomicUsize::new(0),
                        _reservation: reservation,
                        idle: Box::new(move |value| {
                            idle(value.downcast_ref::<T>().expect("registered resource type"))
                        }),
                    }))
                }
                other => {
                    let error = match other {
                        Ok(Ok(_)) => Error::Message(
                            "resident bytes exceed loader reservation or are zero".into(),
                        ),
                        Ok(Err(error)) => error,
                        Err(_) => Error::Message("resident model loader panicked".into()),
                    };
                    drop(reservation);
                    Err(Arc::new(error))
                }
            };
            {
                let mut state = self.inner.state.lock().unwrap_or_else(|e| e.into_inner());
                match &result {
                    Ok(resource) => {
                        let used = state.clock;
                        state.entries.insert(
                            key,
                            Entry::Ready {
                                resource: resource.clone(),
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
            result.map_err(|source| Error::Execution { source })?
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
                .unwrap()
                .clone()
                .map_err(|source| Error::Execution { source })?
        };
        lease(resource)
    }

    /// Snapshot budget accounting without waiting for active loads.
    pub fn statistics(&self) -> ResidencyStatistics {
        let state = self.inner.state.lock().unwrap_or_else(|e| e.into_inner());
        ResidencyStatistics {
            budget_bytes: self.inner.usage.limit,
            reserved_bytes: self.inner.usage.bytes.load(Ordering::Acquire),
            resources: state.entries.len(),
            evictions: state.evictions,
        }
    }
}
fn lease<T: Any + Send + Sync>(resource: Arc<Resource>) -> Result<ModelLease<T>> {
    let value = resource
        .value
        .clone()
        .downcast::<T>()
        .map_err(|_| Error::Message("residency key has another resource type".into()))?;
    Ok(ModelLease { resource, value })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn budgeted_units_grow_without_double_charging_and_evict_only_when_idle() {
        let manager = ResidencyManager::new(100).unwrap();
        let budget = manager.budget();
        let model = manager
            .load_budgeted(
                "model",
                |_: &Mutex<Vec<MemoryReservation>>| true,
                |budget| Ok(Mutex::new(vec![budget.reserve(40)?])),
            )
            .unwrap();
        assert_eq!(model.bytes(), 0);
        model.lock().unwrap().push(budget.reserve(50).unwrap());
        assert_eq!(budget.reserved_bytes(), 90);
        assert!(manager.evict("model").is_err());
        let pin = model.pin();
        drop(model);
        assert!(budget.reserve(11).is_err());
        drop(pin);
        let other = budget.reserve(100).unwrap();
        assert_eq!(manager.statistics().resources, 0);
        drop(other);
        assert_eq!(budget.reserved_bytes(), 0);
        assert!(!manager.evict("model").unwrap());
    }

    #[test]
    fn budgeted_loader_failure_and_panic_release_allocations_and_allow_retry() {
        let manager = ResidencyManager::new(64).unwrap();
        for panic in [false, true] {
            let result = manager.load_budgeted(
                "model",
                |_: &MemoryReservation| true,
                |budget| {
                    let _held = budget.reserve(64)?;
                    assert!(!panic, "injected loader panic");
                    Err(Error::Message("injected loader failure".into()))
                },
            );
            assert!(result.is_err());
            assert_eq!(manager.statistics().reserved_bytes, 0);
            assert_eq!(manager.statistics().resources, 0);
        }
        drop(
            manager
                .load_budgeted(
                    "model",
                    |_: &MemoryReservation| true,
                    |budget| budget.reserve(64),
                )
                .unwrap(),
        );
        assert!(manager.evict("model").unwrap());
        assert_eq!(manager.statistics().reserved_bytes, 0);
        assert!(manager.budget().manager().is_some());
        let budget = manager.budget();
        drop(manager);
        assert!(budget.manager().is_none());
    }
    #[test]
    fn allocations_and_loaders_share_a_ceiling_and_idle_eviction() {
        let manager = ResidencyManager::new(100).unwrap();
        let budget = manager.budget();
        let scratch = budget.reserve(40).unwrap();
        let model = manager
            .load("model", 60, |_: &u8| true, || Ok((1, 50)))
            .unwrap();
        assert_eq!(budget.reserved_bytes(), 90);
        assert!(matches!(budget.reserve(11), Err(Error::Busy(_))));
        drop(model);
        let staging = budget.reserve(60).unwrap();
        assert_eq!(manager.statistics().evictions, 1);
        assert_eq!(manager.statistics().reserved_bytes, 100);
        assert!(
            manager
                .load("other", 1, |_: &u8| true, || Ok((2, 1)))
                .is_err()
        );
        drop(staging);
        drop(scratch);
        assert_eq!(budget.reserved_bytes(), 0);
        assert!(budget.reserve(101).is_err());
        assert_eq!(budget.reserve(0).unwrap().bytes(), 0);
    }

    #[test]
    fn budget_owners_do_not_cycle_and_survive_registry_teardown() {
        let manager = ResidencyManager::new(64).unwrap();
        let registry = Arc::downgrade(&manager.inner);
        let budget = manager.budget();
        let scratch = budget.reserve(16).unwrap();
        drop(
            manager
                .load(
                    "owner",
                    32,
                    |_: &MemoryBudget| true,
                    || Ok((budget.clone(), 32)),
                )
                .unwrap(),
        );
        assert_eq!(budget.reserved_bytes(), 48);
        drop(manager);
        assert!(registry.upgrade().is_none());
        assert_eq!(budget.reserved_bytes(), 16);
        let tail = budget.reserve(48).unwrap();
        assert!(budget.reserve(1).is_err());
        drop((scratch, tail));
        assert_eq!(budget.reserved_bytes(), 0);
    }

    #[test]
    fn concurrent_reservations_cannot_oversubscribe() {
        let manager = ResidencyManager::new(64).unwrap();
        let budget = manager.budget();
        let barrier = std::sync::Barrier::new(8);
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    let held = budget.reserve(8).unwrap();
                    barrier.wait();
                    assert_eq!(budget.reserved_bytes(), 64);
                    assert!(budget.reserve(1).is_err());
                    barrier.wait();
                    drop(held);
                });
            }
        });
        assert_eq!(budget.reserved_bytes(), 0);
    }
    #[test]
    fn retiring_native_storage_remains_charged_until_its_destructor_finishes() {
        struct Retiring {
            started: std::sync::mpsc::Sender<()>,
            release: Mutex<std::sync::mpsc::Receiver<()>>,
        }
        impl Drop for Retiring {
            fn drop(&mut self) {
                self.started.send(()).unwrap();
                self.release.get_mut().unwrap().recv().unwrap();
            }
        }
        let manager = ResidencyManager::new(64).unwrap();
        let (started, seen) = std::sync::mpsc::channel();
        let (release, gate) = std::sync::mpsc::channel();
        drop(
            manager
                .load(
                    "old",
                    64,
                    |_: &Retiring| true,
                    || {
                        Ok((
                            Retiring {
                                started,
                                release: Mutex::new(gate),
                            },
                            64,
                        ))
                    },
                )
                .unwrap(),
        );
        std::thread::scope(|scope| {
            let loading = scope.spawn(|| {
                manager
                    .load("new", 64, |_: &u8| true, || Ok((1, 64)))
                    .unwrap()
            });
            seen.recv_timeout(std::time::Duration::from_secs(5))
                .unwrap();
            assert_eq!(manager.statistics().reserved_bytes, 64);
            assert!(matches!(
                manager.load("other", 1, |_: &u8| true, || Ok((2, 1))),
                Err(Error::Busy(_))
            ));
            release.send(()).unwrap();
            assert_eq!(*loading.join().unwrap(), 1);
        });
    }
    #[test]
    fn leases_pins_and_exported_work_prevent_eviction() {
        let manager = ResidencyManager::new(100).unwrap();
        let model = manager
            .load(
                "one",
                70,
                |v: &AtomicUsize| v.load(Ordering::Relaxed) == 0,
                || Ok((AtomicUsize::new(0), 60)),
            )
            .unwrap();
        let pin = model.pin();
        drop(model);
        assert!(matches!(
            manager.load("two", 50, |_: &u8| true, || Ok((2, 50))),
            Err(Error::Busy(_))
        ));
        drop(pin);
        let model = manager
            .load("one", 70, |_: &AtomicUsize| true, || unreachable!())
            .unwrap();
        model.store(1, Ordering::Relaxed);
        drop(model);
        assert!(matches!(
            manager.load("two", 50, |_: &u8| true, || Ok((2, 50))),
            Err(Error::Busy(_))
        ));
        manager
            .load("one", 70, |_: &AtomicUsize| true, || unreachable!())
            .unwrap()
            .store(0, Ordering::Relaxed);
        let second = manager
            .load("two", 50, |_: &u8| true, || Ok((2, 50)))
            .unwrap();
        assert_eq!(*second, 2);
        assert_eq!(manager.statistics().evictions, 1);
        assert_eq!(manager.statistics().reserved_bytes, 50);
    }
    #[test]
    fn failed_loads_rollback_and_concurrent_loads_share_one_reservation() {
        let manager = ResidencyManager::new(64).unwrap();
        assert!(
            manager
                .load(
                    "a",
                    64,
                    |_: &usize| true,
                    || Err(Error::Message("injected".into()))
                )
                .is_err()
        );
        assert_eq!(manager.statistics().reserved_bytes, 0);
        assert!(
            manager
                .load("a", 64, |_: &usize| true, || Ok((1, 65)))
                .is_err()
        );
        let barrier = std::sync::Barrier::new(8);
        let calls = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    barrier.wait();
                    let value = manager
                        .load(
                            "a",
                            64,
                            |_: &usize| true,
                            || {
                                calls.fetch_add(1, Ordering::Relaxed);
                                Ok((9, 32))
                            },
                        )
                        .unwrap();
                    assert_eq!(*value, 9);
                });
            }
        });
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(manager.statistics().reserved_bytes, 32);
        assert!(
            manager
                .load("a", 64, |_: &u8| true, || unreachable!())
                .is_err()
        );
    }
}
