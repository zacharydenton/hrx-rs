use super::*;

/// Access site participating in an ordered transfer of ownership.
#[derive(Clone, Copy)]
pub enum MemorySite<'a> {
    /// The buffer's actual CPU mapping.
    Host,
    /// An admitted device and an exact native queue family ordinal.
    Device(&'a Device, u32),
}
/// Granularity of a qualified cache action.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransitionKind {
    /// The provider cannot establish a transition.
    Unknown,
    /// No cache operation is necessary; ordering is still required.
    None,
    /// An explicit byte range must be transitioned.
    Range,
    /// The complete cache domain must be transitioned.
    Global,
}
/// Participant responsible for the transition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheExecutor {
    /// No operation or no qualified executor.
    None,
    /// Encode a queue command.
    Queue,
    /// Encode an operation in the running program.
    Program,
    /// Execute a host instruction and its prescribed fences.
    HostDirect,
    /// Call the native host cache API.
    HostApi,
}
/// Semantic cache action; independent of ordering and atomic reach.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheOperation {
    /// No semantic queue operation.
    None,
    /// Publish writes to system memory.
    ReleaseToSystem,
    /// Acquire writes from system memory.
    AcquireFromSystem,
}
/// Immutable facts for one half of a directional visibility relation.
#[derive(Clone, Copy, Debug)]
pub struct CacheTransition {
    /// Scope of maintenance.
    pub kind: TransitionKind,
    /// Responsible participant.
    pub executor: CacheExecutor,
    /// Queue/program action.
    pub operation: CacheOperation,
    /// Minimum independently transitionable range in bytes.
    pub range_granularity: u64,
    native: amdf_cache_transition_t,
}
impl CacheTransition {
    fn from_native(native: amdf_cache_transition_t) -> Result<Self> {
        Ok(Self {
            kind: match native.kind {
                AMDF_CACHE_TRANSITION_KIND_UNKNOWN => TransitionKind::Unknown,
                AMDF_CACHE_TRANSITION_KIND_NONE => TransitionKind::None,
                AMDF_CACHE_TRANSITION_KIND_RANGE => TransitionKind::Range,
                AMDF_CACHE_TRANSITION_KIND_GLOBAL => TransitionKind::Global,
                _ => return Err(missing("cache transition kind")),
            },
            executor: match native.executor {
                AMDF_CACHE_TRANSITION_EXECUTOR_NONE => CacheExecutor::None,
                AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE => CacheExecutor::Queue,
                AMDF_CACHE_TRANSITION_EXECUTOR_PROGRAM => CacheExecutor::Program,
                AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT => CacheExecutor::HostDirect,
                AMDF_CACHE_TRANSITION_EXECUTOR_HOST_API => CacheExecutor::HostApi,
                _ => return Err(missing("cache transition executor")),
            },
            operation: match native.operation {
                AMDF_CACHE_OPERATION_NONE => CacheOperation::None,
                AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM => CacheOperation::ReleaseToSystem,
                AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM => CacheOperation::AcquireFromSystem,
                _ => return Err(missing("cache operation")),
            },
            range_granularity: native.range_granularity,
            native,
        })
    }
}
/// Largest scope of mutually atomic access reported for a width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AtomicScope {
    /// Atomic reach is unavailable.
    None,
    /// Within one device.
    Device,
    /// Across the participating fabric.
    Fabric,
    /// Across the system.
    System,
}
impl AtomicScope {
    fn from_native(scope: amdf_atomic_scope_t) -> Result<Self> {
        match scope {
            AMDF_ATOMIC_SCOPE_NONE => Ok(Self::None),
            AMDF_ATOMIC_SCOPE_DEVICE => Ok(Self::Device),
            AMDF_ATOMIC_SCOPE_FABRIC => Ok(Self::Fabric),
            AMDF_ATOMIC_SCOPE_SYSTEM => Ok(Self::System),
            _ => Err(missing("atomic scope")),
        }
    }
}
/// Directional cache, reachability, atomic and provider cost facts.
#[derive(Clone, Copy, Debug)]
pub struct VisibilityFacts {
    /// Whether the consumer can directly reach the producer's backing.
    pub shared_backing_reachable: bool,
    /// Producer action after writes.
    pub release: CacheTransition,
    /// Consumer action before reads.
    pub acquire: CacheTransition,
    /// Atomic reach for naturally aligned 32-bit accesses; operation support is separate.
    pub atomic_scope_32: AtomicScope,
    /// Atomic reach for naturally aligned 64-bit accesses; operation support is separate.
    pub atomic_scope_64: AtomicScope,
    /// Provider-qualified fixed cost, when known; not a measured workload latency.
    pub estimated_fixed_cost_ns: Option<u64>,
}
impl VisibilityFacts {
    pub(super) fn from_native(info: amdf_memory_pair_info_t) -> Result<Self> {
        Ok(Self {
            shared_backing_reachable: info.flags
                & AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE as u64
                != 0,
            release: CacheTransition::from_native(info.release)?,
            acquire: CacheTransition::from_native(info.acquire)?,
            atomic_scope_32: AtomicScope::from_native(info.atomic_reach.scope_32)?,
            atomic_scope_64: AtomicScope::from_native(info.atomic_reach.scope_64)?,
            estimated_fixed_cost_ns: (info.flags & AMDF_MEMORY_PAIR_FLAG_FIXED_COST_KNOWN as u64
                != 0)
                .then_some(info.estimated_fixed_cost_nanoseconds),
        })
    }
}
/// Prepared directional visibility retaining its actual backing.
/// Cache actions do not establish execution ordering.
pub struct MemoryVisibility {
    buffer: Buffer,
    facts: VisibilityFacts,
}
impl std::ops::Deref for MemoryVisibility {
    type Target = VisibilityFacts;
    fn deref(&self) -> &Self::Target {
        &self.facts
    }
}
impl Buffer {
    fn site(&self, site: MemorySite<'_>) -> Result<amdf_memory_site_t> {
        let mut result = amdf_memory_site_t {
            type_: AMDF_STRUCTURE_TYPE_MEMORY_SITE,
            structure_size: size_of::<amdf_memory_site_t>() as u32,
            ..Default::default()
        };
        match site {
            MemorySite::Host => {
                if let Some(source) = &self.0.source {
                    return source.site(site);
                }
                result.kind = AMDF_MEMORY_SITE_KIND_HOST;
                result.value.host_mapping = self.0.mapping;
            }
            MemorySite::Device(device, family) => {
                let Some(index) = self
                    .0
                    .devices
                    .iter()
                    .position(|entry| Arc::ptr_eq(&entry.0, &device.0))
                else {
                    return self.0.source.as_ref().map_or_else(
                        || {
                            Err(Error::Message(
                                "visibility device has no access to backing".into(),
                            ))
                        },
                        |source| source.site(site),
                    );
                };
                result.kind = AMDF_MEMORY_SITE_KIND_DEVICE;
                result.value.device = amdf_memory_site_t__bindgen_ty_1__bindgen_ty_1 {
                    memory: self.0.raw,
                    access_ordinal: index as u32,
                    queue_family_ordinal: family,
                };
            }
        }
        Ok(result)
    }
    /// Query exact producer-to-consumer visibility once during preparation.
    /// Host aliases resolve to their original mapping; device sites resolve to
    /// their actual native attachments rather than equating virtual addresses.
    pub fn visibility(
        &self,
        producer: MemorySite<'_>,
        consumer: MemorySite<'_>,
    ) -> Result<MemoryVisibility> {
        let mut info = amdf_memory_pair_info_t {
            type_: AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
            structure_size: size_of::<amdf_memory_pair_info_t>() as u32,
            ..Default::default()
        };
        check("memory_query_pair_info", unsafe {
            entry!(self.0.instance.api.core, memory_query_pair_info)(
                &self.site(producer)?,
                &self.site(consumer)?,
                &mut info,
            )
        })?;
        Ok(MemoryVisibility {
            buffer: self.clone(),
            facts: VisibilityFacts::from_native(info)?,
        })
    }
}
impl MemoryVisibility {
    /// Execute the prepared host producer action.
    /// # Safety
    /// Caller owns the range and its boundary cache lines, excludes conflicting
    /// access, and supplies the ordering edge to the consumer.
    pub unsafe fn release_host(&self, offset: usize, length: usize) -> Result<()> {
        unsafe { self.host_transition(self.release, offset, length) }
    }
    /// Execute the prepared host consumer action after producer completion.
    /// # Safety
    /// Caller owns the range and its boundary cache lines and has established
    /// producer completion before acquiring visibility.
    pub unsafe fn acquire_host(&self, offset: usize, length: usize) -> Result<()> {
        unsafe { self.host_transition(self.acquire, offset, length) }
    }
    unsafe fn host_transition(
        &self,
        transition: CacheTransition,
        offset: usize,
        length: usize,
    ) -> Result<()> {
        if offset
            .checked_add(length)
            .is_none_or(|end| end > self.buffer.len())
        {
            return Err(Error::Message("visibility range out of bounds".into()));
        }
        if !self.shared_backing_reachable {
            return Err(Error::Unsupported("shared backing is unreachable".into()));
        }
        if transition.kind == TransitionKind::None {
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            return Ok(());
        }
        if !matches!(
            transition.executor,
            CacheExecutor::HostApi | CacheExecutor::HostDirect
        ) {
            return Err(Error::Unsupported(
                "transition is not a qualified host action".into(),
            ));
        }
        let mut buffer = &self.buffer;
        while let Some(source) = &buffer.0.source {
            buffer = source;
        }
        check("prepared host visibility", unsafe {
            entry!(buffer.0.instance.api.core, host_mapping_cache_control)(
                buffer.0.mapping,
                transition.native.host_operation,
                offset as u64,
                length as u64,
            )
        })
    }
}

/// Coalesced host cache work prepared against one retained backing and direction.
/// Preparation performs all range allocation and validation; execution does not
/// query visibility or allocate storage.
pub struct PreparedHostVisibility {
    visibility: MemoryVisibility,
    transition: CacheTransition,
    ranges: Vec<std::ops::Range<usize>>,
}
impl MemoryVisibility {
    /// Prepare producer cache work, merging overlapping and adjacent ranges.
    pub fn prepare_release_host(
        &self,
        ranges: &[std::ops::Range<usize>],
    ) -> Result<PreparedHostVisibility> {
        self.prepare_host(self.release, ranges)
    }
    /// Prepare consumer cache work, merging overlapping and adjacent ranges.
    pub fn prepare_acquire_host(
        &self,
        ranges: &[std::ops::Range<usize>],
    ) -> Result<PreparedHostVisibility> {
        self.prepare_host(self.acquire, ranges)
    }
    fn prepare_host(
        &self,
        transition: CacheTransition,
        ranges: &[std::ops::Range<usize>],
    ) -> Result<PreparedHostVisibility> {
        if !self.shared_backing_reachable || transition.kind == TransitionKind::Unknown {
            return Err(Error::Unsupported(
                "unqualified host visibility relation".into(),
            ));
        }
        if transition.kind != TransitionKind::None
            && !matches!(
                transition.executor,
                CacheExecutor::HostDirect | CacheExecutor::HostApi
            )
        {
            return Err(Error::Unsupported(
                "transition is not a qualified host action".into(),
            ));
        }
        let mut ranges = coalesce_ranges(ranges, self.buffer.len())?;
        if !ranges.is_empty()
            && matches!(
                transition.kind,
                TransitionKind::None | TransitionKind::Global
            )
        {
            ranges.clear();
            ranges.push(0..self.buffer.len());
        }
        Ok(PreparedHostVisibility {
            visibility: MemoryVisibility {
                buffer: self.buffer.clone(),
                facts: self.facts,
            },
            transition,
            ranges,
        })
    }
}
fn coalesce_ranges(
    ranges: &[std::ops::Range<usize>],
    length: usize,
) -> Result<Vec<std::ops::Range<usize>>> {
    if ranges.iter().any(|r| r.start > r.end || r.end > length) {
        return Err(Error::Message("visibility range out of bounds".into()));
    }
    let mut sorted: Vec<_> = ranges.iter().filter(|r| !r.is_empty()).cloned().collect();
    sorted.sort_unstable_by_key(|r| r.start);
    let mut result: Vec<std::ops::Range<usize>> = Vec::with_capacity(sorted.len());
    for range in sorted {
        if let Some(last) = result.last_mut()
            && last.end >= range.start
        {
            last.end = last.end.max(range.end);
        } else {
            result.push(range);
        }
    }
    Ok(result)
}
impl PreparedHostVisibility {
    /// Number of coalesced cache calls (or ordering fences for a no-op recipe).
    pub fn operation_count(&self) -> usize {
        self.ranges.len()
    }
    /// Execute the prepared host side of the ownership transfer.
    /// # Safety
    /// Caller owns all covered ranges and their boundary cache lines, excludes
    /// conflicting accesses, and establishes the directional execution edge.
    /// A global recipe additionally requires ownership of the entire cache domain.
    pub unsafe fn execute(&self) -> Result<()> {
        for range in &self.ranges {
            unsafe {
                self.visibility
                    .host_transition(self.transition, range.start, range.len())
            }?;
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn host_ranges_preserve_holes_and_reject_out_of_bounds() {
        assert_eq!(
            coalesce_ranges(&[8..16, 0..4, 3..10, 24..32, 16..16], 32).unwrap(),
            vec![0..16, 24..32]
        );
        assert!(coalesce_ranges(std::slice::from_ref(&(0..33)), 32).is_err());
        assert!(coalesce_ranges(&[std::ops::Range { start: 9, end: 8 }], 32).is_err());
        assert!(
            coalesce_ranges(std::slice::from_ref(&(0..0)), 0)
                .unwrap()
                .is_empty()
        );
    }
}
