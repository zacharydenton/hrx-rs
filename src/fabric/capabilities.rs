use super::*;

/// Native packet representation accepted by one queue family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueueCommand {
    /// GPU compute/control packets.
    Pm4,
    /// GPU DMA packets.
    Sdma,
    /// HSA kernel dispatch packets.
    Aql,
    /// HSA packets with native metadata.
    AqlMetadata,
    /// XDNA establishing commands.
    Xdna,
    /// A representation not currently encoded by this crate.
    Unknown(u32),
}
/// Passive, immutable capabilities copied from the current native provider.
#[derive(Clone, Debug)]
pub struct QueueCapabilities {
    /// Dense family ordinal accepted by memory visibility queries.
    pub ordinal: u32,
    /// Native command representation.
    pub command: QueueCommand,
    /// Command-format version.
    pub format_version: u32,
    /// Representation-specific feature bits from the native ABI.
    pub format_features: u64,
    /// Host/device direct publication is available.
    pub user_publication: bool,
    /// Kernel-mediated publication is available.
    pub kernel_publication: bool,
    /// Minimum ring length in bytes.
    pub minimum_ring_bytes: u64,
    /// Maximum ring length in bytes.
    pub maximum_ring_bytes: u64,
    /// Required ring-length alignment.
    pub ring_alignment: u64,
    /// Queue can release writes to system memory.
    pub system_release: bool,
    /// Queue can acquire writes from system memory.
    pub system_acquire: bool,
}
impl Endpoint {
    /// Inspect all native queue families without opening a device or queue.
    pub fn queue_capabilities(&self) -> Result<Vec<QueueCapabilities>> {
        let api = self.0.instance.api.core;
        let mut info = amdf_endpoint_info_t {
            type_: AMDF_STRUCTURE_TYPE_ENDPOINT_INFO,
            structure_size: size_of::<amdf_endpoint_info_t>() as u32,
            ..Default::default()
        };
        check("endpoint_query_info", unsafe {
            entry!(api, endpoint_query_info)(self.0.raw, &mut info)
        })?;
        (0..info.queue_family_count)
            .map(|ordinal| {
                let mut family = amdf_queue_family_info_t {
                    type_: AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO,
                    structure_size: size_of::<amdf_queue_family_info_t>() as u32,
                    ..Default::default()
                };
                check("endpoint_query_queue_family_info", unsafe {
                    entry!(api, endpoint_query_queue_family_info)(self.0.raw, ordinal, &mut family)
                })?;
                Ok(QueueCapabilities {
                    ordinal,
                    command: match family.command_type {
                        AMDF_QUEUE_COMMAND_TYPE_GPU_PM4 => QueueCommand::Pm4,
                        AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA => QueueCommand::Sdma,
                        AMDF_QUEUE_COMMAND_TYPE_GPU_AQL => QueueCommand::Aql,
                        AMDF_QUEUE_COMMAND_TYPE_GPU_AQL_METADATA => QueueCommand::AqlMetadata,
                        AMDF_QUEUE_COMMAND_TYPE_XDNA => QueueCommand::Xdna,
                        unknown => QueueCommand::Unknown(unknown),
                    },
                    format_version: family.format_version,
                    format_features: family.format_features,
                    user_publication: family.publication_modes & AMDF_QUEUE_PUBLICATION_MODE_USER
                        != 0,
                    kernel_publication: family.publication_modes
                        & AMDF_QUEUE_PUBLICATION_MODE_KERNEL
                        != 0,
                    minimum_ring_bytes: family.minimum_ring_byte_length,
                    maximum_ring_bytes: family.maximum_ring_byte_length,
                    ring_alignment: family.ring_byte_length_alignment,
                    system_release: family.cache_operations
                        & (1 << AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM)
                        != 0,
                    system_acquire: family.cache_operations
                        & (1 << AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM)
                        != 0,
                })
            })
            .collect()
    }
}
