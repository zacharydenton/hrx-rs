//! Bounded automatic reuse of temporary device buffers on one ordered stream.

use crate::{Buffer, Error, Result, Stream, View};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

const DEFAULT_LIMIT: usize = 512 << 20;

#[derive(Default)]
struct FreeList {
    cached: usize,
    blocks: BTreeMap<usize, Vec<Buffer>>,
    stream: Option<usize>,
}

/// A bounded free list whose allocations are safe to reuse on one ordered stream.
pub struct BufferPool {
    limit: usize,
    free: Mutex<FreeList>,
}

impl Default for BufferPool {
    fn default() -> Self {
        Self::with_limit(DEFAULT_LIMIT)
    }
}

impl BufferPool {
    /// Create a pool with the default 512 MiB retained-byte limit.
    #[must_use]
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Create a pool with an explicit retained-byte limit.
    #[must_use]
    pub fn with_limit(limit: usize) -> Self {
        Self {
            limit,
            free: Mutex::new(FreeList::default()),
        }
    }

    /// Acquire at least `bytes`, reusing a block no larger than twice the request.
    pub fn acquire(self: &Arc<Self>, stream: &Stream, bytes: usize) -> Result<PooledBuffer> {
        if bytes == 0 {
            return Err(Error::Message("empty pooled allocation".into()));
        }
        let mut free = self.free.lock().unwrap_or_else(|error| error.into_inner());
        match free.stream {
            Some(owner) if owner != stream.id() => {
                return Err(Error::Message(
                    "a buffer pool serves one stream because reuse follows that queue's order"
                        .into(),
                ));
            }
            Some(_) => {}
            None => free.stream = Some(stream.id()),
        }
        let reusable = free
            .blocks
            .range_mut(bytes..=bytes.saturating_mul(2))
            .next()
            .map(|(&size, blocks)| (size, blocks.pop()));
        if let Some((size, Some(buffer))) = reusable {
            free.cached -= size;
            if free.blocks[&size].is_empty() {
                free.blocks.remove(&size);
            }
            return Ok(PooledBuffer {
                buffer: Some(buffer),
                pool: self.clone(),
            });
        }
        drop(free);
        Ok(PooledBuffer {
            buffer: Some(stream.allocate(bytes)?),
            pool: self.clone(),
        })
    }

    /// Bytes currently retained for reuse.
    #[must_use]
    pub fn cached_bytes(&self) -> usize {
        self.free
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .cached
    }

    fn release(&self, buffer: Buffer) {
        let mut free = self.free.lock().unwrap_or_else(|error| error.into_inner());
        if buffer.bytes() > self.limit.saturating_sub(free.cached) {
            return;
        }
        free.cached += buffer.bytes();
        free.blocks.entry(buffer.bytes()).or_default().push(buffer);
    }
}

/// Device storage returned to its [`BufferPool`] when the lease drops.
pub struct PooledBuffer {
    buffer: Option<Buffer>,
    pool: Arc<BufferPool>,
}

impl PooledBuffer {
    /// Borrow the allocation.
    #[must_use]
    pub fn buffer(&self) -> &Buffer {
        self.buffer.as_ref().expect("live pooled buffer")
    }

    /// Borrow the whole allocation as a binding.
    #[must_use]
    pub fn binding(&self) -> View<'_> {
        self.buffer().binding()
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.pool.release(buffer);
        }
    }
}
