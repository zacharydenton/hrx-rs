use super::*;

/// Timestamp units for one source. Equal rates do not establish a shared epoch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClockInfo {
    /// Ticks per second; zero means unknown.
    pub frequency_hz: u64,
    /// Meaningful bits; zero means unspecified.
    pub counter_bits: u32,
}
impl From<amdf_observation_clock_info_t> for ClockInfo {
    fn from(info: amdf_observation_clock_info_t) -> Self {
        Self {
            frequency_hz: info.frequency_hz,
            counter_bits: info.counter_bit_count,
        }
    }
}
/// One native driver clock sample, without retained calibration.
#[derive(Clone, Copy, Debug)]
pub struct ClockSample {
    /// Units of the native device query, not necessarily the queue's epoch.
    pub clock: ClockInfo,
    /// Device counter value.
    pub timestamp: u64,
    /// Common profiling host clock before, paired with, and after the query.
    pub host: [u64; 3],
    /// Auxiliary system timestamp and ticks/second; absent when rate is zero.
    pub system: [u64; 2],
}
impl Fabric {
    /// Units of the monotonic, boot-wide profiling host clock.
    pub fn host_clock_info(&self) -> Result<ClockInfo> {
        let api = self.native_api();
        let mut info = Default::default();
        check("query_host_clock_info", unsafe {
            entry!(api.observation()?, query_host_clock_info)(&mut info)
        })?;
        Ok(ClockInfo::from(info))
    }
    /// Read the common profiling host clock. This adds no data synchronization.
    pub fn host_clock(&self) -> Result<u64> {
        let api = self.native_api();
        let mut tick = 0;
        check("query_host_clock", unsafe {
            entry!(api.observation()?, query_host_clock)(&mut tick)
        })?;
        Ok(tick)
    }
}
impl Device {
    /// Nominal GPU queue timestamp units, without assuming a common host epoch.
    pub fn queue_clock_info(&self, family: u32) -> Result<ClockInfo> {
        let mut info = Default::default();
        check("device_query_queue_clock_info", unsafe {
            entry!(
                self.0.endpoint.0.instance.api.gpu,
                device_query_queue_clock_info
            )(self.0.raw, family, &mut info)
        })?;
        Ok(ClockInfo::from(info))
    }
    /// Sample the native GPU and host clocks. Callers own correlation policy.
    pub fn clock_sample(&self, family: u32) -> Result<ClockSample> {
        let mut sample = amdf_observation_clock_sample_t::default();
        check("device_query_clock_sample", unsafe {
            entry!(
                self.0.endpoint.0.instance.api.gpu,
                device_query_clock_sample
            )(self.0.raw, family, &mut sample)
        })?;
        Ok(ClockSample {
            clock: sample.clock.into(),
            timestamp: sample.timestamp,
            host: [sample.host.before, sample.host.timestamp, sample.host.after],
            system: [sample.system.timestamp, sample.system.frequency_hz],
        })
    }
    /// Current XDNA RECORD_TIMER rate. Admission/power changes may change it;
    /// equal rates before/after a workload do not prove a stable interval.
    pub fn timer_clock_info(&self) -> Result<ClockInfo> {
        let mut info = Default::default();
        check("device_query_timer_clock_info", unsafe {
            entry!(
                self.0.endpoint.0.instance.api.xdna,
                device_query_timer_clock_info
            )(self.0.raw, &mut info)
        })?;
        Ok(ClockInfo::from(info))
    }
    /// Query the device-wide NPU firmware recorder without changing it.
    pub fn firmware_trace_info(&self) -> Result<FirmwareTraceInfo> {
        let mut info = amdf_xdna_firmware_trace_info_t {
            type_: HRX_AMDF_STRUCTURE_TYPE_XDNA_FIRMWARE_TRACE_INFO,
            structure_size: size_of::<amdf_xdna_firmware_trace_info_t>() as u32,
            ..Default::default()
        };
        check("device_query_firmware_trace_info", unsafe {
            entry!(
                self.0.endpoint.0.instance.api.xdna,
                device_query_firmware_trace_info
            )(self.0.raw, &mut info)
        })?;
        Ok(FirmwareTraceInfo {
            enabled: info.enabled != 0,
            categories: info.categories,
            format: info.format_version,
            buffer_alignment: info.buffer_alignment,
            minimum_buffer_bytes: info.minimum_buffer_byte_length,
            recommended_buffer_bytes: info.recommended_buffer_byte_length,
            clock: info.clock.into(),
        })
    }
    /// Configure the global NPU recorder. Category bits are in [`native`].
    /// Arrange ownership across processes and serialize configuration and reads.
    /// Dropping the device does not disable recording. On error, query live state
    /// before deciding cleanup because configuration failure may have effects.
    pub fn configure_firmware_trace(&self, enabled: bool, categories: u32) -> Result<()> {
        check("device_configure_firmware_trace", unsafe {
            entry!(
                self.0.endpoint.0.instance.api.xdna,
                device_configure_firmware_trace
            )(self.0.raw, u32::from(enabled), categories)
        })
    }
    /// Read one lossy batch, allocating aligned temporary storage. Start at
    /// position zero for a new capture and advance only after accepting a batch.
    /// Exclude external reconfiguration; an empty batch is not completion.
    pub fn read_firmware_trace(
        &self,
        info: &FirmwareTraceInfo,
        position: u64,
        capacity: u32,
    ) -> Result<FirmwareTraceBatch> {
        let capacity = capacity.max(info.minimum_buffer_bytes);
        let layout =
            std::alloc::Layout::from_size_align(capacity as usize, info.buffer_alignment as usize)
                .map_err(|_| Error::Message("invalid firmware trace storage geometry".into()))?;
        if capacity == 0 {
            return Err(Error::Message("empty firmware trace storage".into()));
        }
        struct Storage(*mut u8, std::alloc::Layout);
        impl Drop for Storage {
            fn drop(&mut self) {
                unsafe { std::alloc::dealloc(self.0, self.1) }
            }
        }
        let storage = Storage(unsafe { std::alloc::alloc_zeroed(layout) }, layout);
        if storage.0.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        let mut result = amdf_observation_read_result_t::default();
        check("device_read_firmware_trace", unsafe {
            entry!(
                self.0.endpoint.0.instance.api.xdna,
                device_read_firmware_trace
            )(
                self.0.raw,
                info.format,
                position,
                storage.0.cast(),
                capacity,
                &mut result,
            )
        })?;
        let size = size_of::<amdf_xdna_firmware_trace_event_t>() as u64;
        if result.byte_offset > u64::from(capacity)
            || result.byte_length > u64::from(capacity) - result.byte_offset
            || !result.byte_length.is_multiple_of(size)
        {
            return Err(Error::Message(
                "invalid firmware trace result bounds".into(),
            ));
        }
        let events = (0..result.byte_length / size)
            .map(|i| {
                let event = unsafe {
                    ptr::read_unaligned(
                        storage
                            .0
                            .add((result.byte_offset + i * size) as usize)
                            .cast::<amdf_xdna_firmware_trace_event_t>(),
                    )
                };
                FirmwareTraceEvent {
                    timestamp: event.timestamp,
                    arguments: event.arguments,
                    event_id: event.event_id,
                    sequence: event.sequence,
                }
            })
            .collect();
        Ok(FirmwareTraceBatch {
            next_position: result.next_position,
            lost_bytes: result.lost_byte_length,
            flags: result.flags,
            events,
        })
    }
}
/// Live firmware recorder state and read geometry.
#[derive(Clone, Debug)]
pub struct FirmwareTraceInfo {
    /// Whether recording is currently enabled.
    pub enabled: bool,
    /// Active category bits.
    pub categories: u32,
    /// Active native schema.
    pub format: u32,
    /// Required read storage alignment.
    pub buffer_alignment: u32,
    /// Minimum accepted capacity, including native overhead.
    pub minimum_buffer_bytes: u32,
    /// Ordinary full-window capacity.
    pub recommended_buffer_bytes: u32,
    /// Timestamp representation for this capture.
    pub clock: ClockInfo,
}
/// Decoded firmware event; unknown identifiers/arguments are preserved.
#[derive(Clone, Copy, Debug)]
pub struct FirmwareTraceEvent {
    /// Capture-local timestamp.
    pub timestamp: u64,
    /// Native arguments, in the low 48 bits.
    pub arguments: u64,
    /// Native schema event identifier.
    pub event_id: u32,
    /// Wrapping entry sequence.
    pub sequence: u32,
}
impl FirmwareTraceEvent {
    /// Physical context for a format-2 COMMAND_BEGIN/COMMAND_END event.
    pub fn command_context(&self) -> u32 {
        (self.arguments & 15) as u32
    }
    /// Virtual context for a format-2 COMMAND_BEGIN/COMMAND_END event.
    pub fn command_virtual_context(&self) -> u32 {
        ((self.arguments >> 20) & 255) as u32
    }
    /// Firmware opcode for a format-2 COMMAND_BEGIN/COMMAND_END event.
    pub fn command_opcode(&self) -> u32 {
        ((self.arguments >> 4) & 65535) as u32
    }
}
/// One copied batch. Process discontinuities before pairing any intervals.
#[derive(Clone, Debug)]
pub struct FirmwareTraceBatch {
    /// Position for the next accepted read.
    pub next_position: u64,
    /// Lost native bytes; u64::MAX means unquantifiable after restart.
    pub lost_bytes: u64,
    /// GAP, RESTART and DISCARDED bits from [`native`].
    pub flags: u32,
    /// Owned decoded records.
    pub events: Vec<FirmwareTraceEvent>,
}
