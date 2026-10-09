//! SDMA v1 packets follow libamdf/include/amdf/gpu.h, using the reported features.
use super::*;

// Keep each publication below half the ring so its wrap padding also fits.
// Reserve two cache packets, a fence, and two unaligned fill-tail copy packets.
fn transfer_word_budget(ring_bytes: usize) -> usize {
    (ring_bytes / 8).saturating_sub(1) & !7
}
pub(super) fn transfer_limit(ring_bytes: usize) -> Result<usize> {
    let packets = transfer_word_budget(ring_bytes).saturating_sub(10 + 4 + 14) / 7;
    if packets == 0 {
        return Err(Error::Unsupported(
            "SDMA ring cannot hold a transfer".into(),
        ));
    }
    Ok(packets.min((u32::MAX as usize) >> 20) << 20)
}

impl Queue {
    fn sdma_scoped(&self) -> bool {
        self.0.info.format_features & AMDF_GPU_SDMA_FORMAT_FEATURE_MEMORY_SCOPE as u64 != 0
    }
    fn sdma_cache(&self, words: &mut Vec<u32>, release: bool) {
        if self.0.info.format_features & AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR as u64 != 0 {
            words.extend([
                17 | (1 << 8),
                0,
                (if release { 0x8040 } else { 0xc3c0 }) << 16,
                0,
                0,
            ]);
        }
    }
    fn sdma_visibility(&self, buffer: &Buffer) -> Result<()> {
        let device = MemorySite::Device(self.device(), self.family_ordinal());
        for plan in [
            buffer.visibility(MemorySite::Host, device)?,
            buffer.visibility(device, MemorySite::Host)?,
        ] {
            if !plan.shared_backing_reachable {
                return Err(Error::Unsupported(
                    "SDMA cannot reach shared backing".into(),
                ));
            }
            for transition in [plan.release, plan.acquire] {
                if transition.kind == TransitionKind::Unknown {
                    return Err(Error::Unsupported(
                        "SDMA visibility is not qualified".into(),
                    ));
                }
                if transition.executor == CacheExecutor::Queue
                    && transition.kind != TransitionKind::None
                    && (transition.kind != TransitionKind::Global
                        || self.0.info.format_features
                            & AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR as u64
                            == 0)
                {
                    return Err(Error::Unsupported(
                        "SDMA cache transition is not encodable".into(),
                    ));
                }
            }
        }
        Ok(())
    }
    fn sdma_prepared(
        &self,
        mut words: Vec<u32>,
        mut buffers: Vec<Buffer>,
        dependencies: Vec<(Arc<Work>, u32)>,
    ) -> Result<PreparedGpu> {
        for buffer in &buffers {
            self.sdma_visibility(buffer)?;
        }
        let fence = self
            .device()
            .fabric()
            .allocate_shared(64, std::slice::from_ref(self.device()))?;
        self.sdma_visibility(&fence)?;
        let address = fence.device_address(self.device())?;
        self.sdma_cache(&mut words, true);
        let end = words.len();
        let features = self.0.info.format_features;
        let mut header = 5;
        if features
            & (AMDF_GPU_SDMA_FORMAT_FEATURE_FENCE_MEMORY_TYPE
                | AMDF_GPU_SDMA_FORMAT_FEATURE_FENCE_SYSTEM) as u64
            != 0
        {
            header |= 3 << 16;
        }
        if features & AMDF_GPU_SDMA_FORMAT_FEATURE_FENCE_SYSTEM as u64 != 0 {
            header |= 1 << 20;
        }
        if self.sdma_scoped() {
            header |= 3 << 24;
        }
        words.extend([header, address as u32, (address >> 32) as u32, 1]);
        let fence_word = words.len() - 1;
        words.resize(words.len().next_multiple_of(8), 0);
        if words.len() * 4 >= self.0.info.ring_byte_length as usize {
            return Err(Error::Unsupported(
                "prepared SDMA command exceeds queue capacity".into(),
            ));
        }
        buffers.sort_by_key(Buffer::identity);
        buffers.dedup_by(|a, b| a.same_backing(b));
        let leases = Mutex::new(Vec::with_capacity(buffers.len()));
        Ok(PreparedGpu {
            prefix: Default::default(),
            queue: self.clone(),
            inner: Arc::new(PreparedInner {
                storage_bytes: fence.len(),
                work: Arc::new(Work {
                    fence,
                    buffers,
                    leases,
                    _kernels: Vec::new(),
                    retired: AtomicU32::new(0),
                    active: AtomicU32::new(0),
                    dependencies,
                }),
                words,
                dispatch_range: 0..end,
                indirect_words: None,
                fence_word,
                previous: Mutex::new(0),
            }),
        })
    }
    pub(super) fn prepare_sdma_copy(
        &self,
        destination: &Buffer,
        destination_offset: usize,
        source: &Buffer,
        source_offset: usize,
        length: usize,
    ) -> Result<PreparedGpu> {
        let mut src = source
            .device_address(self.device())?
            .checked_add(source_offset as u64)
            .ok_or_else(|| Error::Message("SDMA source address overflow".into()))?;
        let mut dst = destination
            .device_address(self.device())?
            .checked_add(destination_offset as u64)
            .ok_or_else(|| Error::Message("SDMA destination address overflow".into()))?;
        // Also reject aliased imported ranges, whose Rust owners differ.
        if src < dst.saturating_add(length as u64) && dst < src.saturating_add(length as u64) {
            return Err(Error::Message("SDMA copy ranges overlap".into()));
        }
        let mut words = Vec::new();
        self.sdma_cache(&mut words, false);
        let mut remaining = length;
        while remaining != 0 {
            let count = remaining.min(1 << 20);
            words.extend([
                1,
                count as u32 - 1,
                if self.sdma_scoped() {
                    (3 << 18) | (3 << 26)
                } else {
                    0
                },
                src as u32,
                (src >> 32) as u32,
                dst as u32,
                (dst >> 32) as u32,
            ]);
            remaining -= count;
            src += count as u64;
            dst += count as u64;
        }
        self.sdma_prepared(words, vec![source.clone(), destination.clone()], Vec::new())
    }
    pub(super) fn prepare_sdma_fill(
        &self,
        destination: &Buffer,
        offset: usize,
        length: usize,
        value: u8,
    ) -> Result<PreparedGpu> {
        let address = destination
            .device_address(self.device())?
            .checked_add(offset as u64)
            .ok_or_else(|| Error::Message("SDMA fill address overflow".into()))?;
        let prefix = ((4 - (address & 3)) & 3).min(length as u64) as usize;
        let middle = (length - prefix) & !3;
        let suffix = length - prefix - middle;
        let mut words = Vec::new();
        let mut buffers = vec![destination.clone()];
        self.sdma_cache(&mut words, false);
        // Byte tails need COPY_LINEAR. Retain one word of prepared data,
        // independently of the destination length; the aligned body uses fill.
        if prefix != 0 || suffix != 0 {
            let pattern = self
                .device()
                .fabric()
                .allocate_shared(64, std::slice::from_ref(self.device()))?;
            pattern.write(0, &[value; 4])?;
            let source = pattern.device_address(self.device())?;
            for (dst, count) in [
                (address, prefix),
                (address + (prefix + middle) as u64, suffix),
            ] {
                if count == 0 {
                    continue;
                }
                words.extend([
                    1,
                    count as u32 - 1,
                    if self.sdma_scoped() {
                        (3 << 18) | (3 << 26)
                    } else {
                        0
                    },
                    source as u32,
                    (source >> 32) as u32,
                    dst as u32,
                    (dst >> 32) as u32,
                ]);
            }
            buffers.push(pattern);
        }
        let mut address = address + prefix as u64;
        let mut remaining = middle;
        while remaining != 0 {
            let count = remaining.min(1 << 20);
            words.extend([
                11 | (2 << 30) | if self.sdma_scoped() { 3 << 24 } else { 0 },
                address as u32,
                (address >> 32) as u32,
                u32::from_le_bytes([value; 4]),
                count as u32 - 1,
            ]);
            remaining -= count;
            address += count as u64;
        }
        self.sdma_prepared(words, buffers, Vec::new())
    }
    pub(super) fn prepare_sdma_wait(
        &self,
        completion: &Completion,
        dependencies: Vec<(Arc<Work>, u32)>,
    ) -> Result<PreparedGpu> {
        let address = completion.work.fence.device_address(self.device())?;
        let mut words = vec![
            8 | (5 << 28) | (1 << 31),
            address as u32,
            (address >> 32) as u32,
            completion.value,
            u32::MAX,
            (0xfff << 16) | 4 | if self.sdma_scoped() { 3 << 28 } else { 0 },
        ];
        self.sdma_cache(&mut words, false);
        self.sdma_prepared(words, Vec::new(), dependencies)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transfer_packets_and_wrap_fit_every_ring_cursor() {
        for ring_bytes in [4096, 8192, 65536, 1 << 20] {
            let limit = transfer_limit(ring_bytes).unwrap();
            assert_eq!(limit % (1 << 20), 0);
            let packets = limit.div_ceil(1 << 20);
            // Worst case includes both unaligned fill tails and both cache packets.
            let words = (packets * 7 + 10 + 4 + 14).next_multiple_of(8);
            let capacity = ring_bytes / 4;
            for cursor in (0..capacity).step_by(8) {
                let wrap = if words > capacity - cursor {
                    capacity - cursor
                } else {
                    0
                };
                assert!(
                    wrap + words < capacity,
                    "{ring_bytes}: cursor {cursor}, words {words}"
                );
            }
        }
        assert!(transfer_limit(0).is_err());
        assert!(transfer_limit(64).is_err());
    }
}
