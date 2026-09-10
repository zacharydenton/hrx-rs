use super::{Access, Buffer, Engine, GpuAccess, Runtime, scheduler::Core};
use crate::{Error, Result};
use std::{collections::BTreeMap, sync::Arc};

struct Lease {
    buffers: Vec<Buffer>,
    core: Arc<Core>,
    uncertain: bool,
}
impl Drop for Lease {
    fn drop(&mut self) {
        let _scheduler = self.core.state.lock().unwrap_or_else(|e| e.into_inner());
        for buffer in &self.buffers {
            let mut host = buffer
                .storage
                .host
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if self.uncertain {
                host.poison = Some("external GPU completion is uncertain".into());
            }
            host.external = false;
        }
        if self.uncertain {
            // Retain mapped wrappers and imported pages beneath possible DMA.
            std::mem::forget(std::mem::take(&mut self.buffers));
        }
        self.core.host_changed.notify_all();
    }
}

impl Runtime {
    /// Run existing Stream operations against coordinated allocations at a stage
    /// boundary. Reservations are allocation-wide; conflicting submitted work
    /// completes first. A live host mapping or another handoff returns `Busy`.
    /// The supplied stream is synchronized before releasing the reservations,
    /// including when the callback fails or panics.
    ///
    /// # Safety
    /// Declare every access to these allocations accurately. The callback must
    /// not retain pointers, buffers, or recorded uses, and must submit all such
    /// work on the supplied stream. No work may escape onto another stream.
    /// Kernel dispatch still carries the low-level GPU API's safety contract.
    pub unsafe fn with_gpu_access<T>(
        &self,
        stream: &mut crate::gpu::Stream,
        accesses: &[GpuAccess],
        callback: impl FnOnce(&mut crate::gpu::Stream, &[crate::gpu::View<'_>]) -> Result<T>,
    ) -> Result<T> {
        let mut roots = BTreeMap::new();
        for usage in accesses {
            let storage = &usage.view.buffer.storage;
            if !Arc::ptr_eq(&storage.runtime, &self.inner) {
                return Err(Error::Message("buffer belongs to another runtime".into()));
            }
            if usage.view.gpu()?.owner().device_id() != stream.device_id() {
                return Err(Error::Message("stream belongs to another device".into()));
            }
            roots.insert(Arc::as_ptr(storage) as usize, usage.view.buffer.clone());
        }
        let buffers: Vec<Buffer> = roots.into_values().collect();
        let core = &self.inner.core;
        let mut scheduler = core.state.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            for buffer in &buffers {
                buffer.storage.host_conflict(Access::ReadWrite)?;
            }
            if !buffers
                .iter()
                .any(|b| scheduler.conflicts(&b.view(), Access::ReadWrite))
            {
                break;
            }
            scheduler = core
                .host_changed
                .wait(scheduler)
                .unwrap_or_else(|e| e.into_inner());
        }
        for buffer in &buffers {
            buffer
                .storage
                .host
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .external = true;
        }
        let mut lease = Lease {
            buffers,
            core: core.clone(),
            uncertain: false,
        };
        drop(scheduler);
        for buffer in &lease.buffers {
            buffer.storage.make_visible(Engine::Gpu)?;
        }
        let views = accesses
            .iter()
            .map(|a| a.view.gpu())
            .collect::<Result<Vec<_>>>()?;
        let outcome =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| callback(stream, &views)));
        let completion = stream.synchronize();
        if completion.is_err() {
            lease.uncertain = true;
        } else {
            for usage in accesses.iter().filter(|a| a.access.writes()) {
                usage
                    .view
                    .buffer
                    .storage
                    .visibility
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .wrote(Engine::Gpu);
            }
        }
        drop(lease);
        match outcome {
            Err(panic) => std::panic::resume_unwind(panic),
            Ok(result) => {
                completion?;
                result
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn leases_release_reservations_on_unwind() {
        let runtime = Runtime::new().unwrap();
        let buffer = super::super::tests::buffer(&runtime);
        buffer.storage.host.lock().unwrap().external = true;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _lease = Lease {
                buffers: vec![buffer.clone()],
                core: runtime.inner.core.clone(),
                uncertain: false,
            };
            assert!(buffer.try_map_read().is_err());
            panic!("unwind");
        }));
        assert!(result.is_err());
        assert!(buffer.try_map_write().is_ok());
    }
    #[test]
    #[cfg_attr(miri, ignore = "deliberate quarantine leak")]
    fn uncertain_handoff_poison_reaches_every_alias() {
        let runtime = Runtime::new().unwrap();
        let buffer = super::super::tests::buffer(&runtime);
        let alias = buffer.clone();
        drop(Lease {
            buffers: vec![buffer],
            core: runtime.inner.core.clone(),
            uncertain: true,
        });
        assert!(matches!(alias.try_map_read(), Err(Error::DeviceLost(_))));
        assert!(matches!(alias.try_map_write(), Err(Error::DeviceLost(_))));
    }
}
