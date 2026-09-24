//! Optional device-clock diagnostics. No hooks run during ordinary submission.
use super::*;
use serde::Serialize;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_RECORDER: AtomicU64 = AtomicU64::new(1);

type Marker = unsafe extern "C" fn(u64, *mut u32, u32, *mut u32) -> i32;
type Clock = unsafe extern "C" fn(u32, u32, *mut u64) -> i32;

pub(super) struct MarkerApi {
    _library: libloading::Library,
    marker: Marker,
    frequency: u64,
    device: [u32; 2],
}
impl MarkerApi {
    pub(super) fn load(device: &Device) -> Result<Self> {
        let api = &device.0.endpoint.0.instance.api;
        let mut info = amdf_endpoint_info_t {
            type_: AMDF_STRUCTURE_TYPE_ENDPOINT_INFO,
            structure_size: size_of::<amdf_endpoint_info_t>() as u32,
            ..Default::default()
        };
        check("profile endpoint", unsafe {
            entry!(api.core, endpoint_query_info)(device.0.endpoint.0.raw, &mut info)
        })?;
        if info.native_identity.type_ != AMDF_ENDPOINT_NATIVE_IDENTITY_TYPE_LINUX_DEVICE {
            return Err(Error::Unsupported(
                "GPU timestamps require a Linux device identity".into(),
            ));
        }
        // The discriminator above validates this union member. Symbols are
        // optional so older runtimes retain ordinary inference compatibility.
        let (library, marker, clock, native) = unsafe {
            let library = libloading::Library::new(api.directory.join("libhrx_fabric.so"))?;
            let marker = *library
                .get::<Marker>(b"hrx_fabric_gpu_profile_marker\0")
                .map_err(|_| missing("GPU profile markers"))?;
            let clock = *library
                .get::<Clock>(b"hrx_fabric_gpu_profile_clock\0")
                .map_err(|_| missing("GPU profile clock"))?;
            (
                library,
                marker,
                clock,
                info.native_identity.value.linux_device,
            )
        };
        let mut frequency = 0;
        let status = unsafe { clock(native.major, native.minor, &mut frequency) };
        if status != 0 || frequency == 0 {
            return Err(Error::Unsupported(format!(
                "GPU profile clock unavailable: errno {status}"
            )));
        }
        Ok(Self {
            _library: library,
            marker,
            frequency,
            device: [native.major, native.minor],
        })
    }
    pub(super) fn emit(&self, address: u64, words: &mut Vec<u32>) -> Result<()> {
        let mut packet = [0u32; 32];
        let mut count = 0;
        let status = unsafe {
            (self.marker)(
                address,
                packet.as_mut_ptr(),
                packet.len() as u32,
                &mut count,
            )
        };
        if status != 0 || count == 0 || count as usize > packet.len() {
            return Err(Error::Unsupported(format!(
                "GPU timestamp marker unavailable: errno {status}"
            )));
        }
        words.extend_from_slice(&packet[..count as usize]);
        Ok(())
    }
}

/// One instrumented command interval in the device's clock domain.
#[derive(Debug, Clone, Serialize)]
pub struct DeviceInterval {
    /// Caller-assigned operation name.
    pub label: String,
    /// Start counter value.
    pub start_tick: u64,
    /// End counter value, after completion ordering.
    pub end_tick: u64,
}
/// Serialized diagnostic execution, including marker and barrier overhead.
/// These intervals measure an instrumented timeline, not hardware utilization.
#[derive(Debug, Clone, Serialize)]
pub struct DeviceProfile {
    /// Process owning the queue and recorder identifiers.
    pub process: u32,
    /// Linux device major/minor identifying the counter's device domain.
    pub device: [u32; 2],
    /// Process-local identity of the originating queue.
    pub queue: usize,
    /// Process-local recorder identity, disambiguating graphs on the same queue.
    pub recorder: u64,
    /// Monotonically increasing sample identity within this recorder.
    pub execution: u64,
    /// GPU counter frequency obtained from the native device.
    pub frequency_hz: u64,
    /// Ordered intervals in one device-clock domain.
    pub intervals: Vec<DeviceInterval>,
    /// Union of measured intervals, converted to milliseconds.
    pub interval_union_ms: f64,
    /// Time between the first marker and final marker.
    pub span_ms: f64,
    /// Gaps in that span outside measured intervals.
    pub gaps_ms: f64,
}

pub(crate) struct ProfileCapture {
    pub(super) buffer: Buffer,
    pub(super) api: MarkerApi,
    labels: Vec<String>,
    queue: usize,
    recorder: u64,
    execution: u64,
    previous_end: u64,
}
impl ProfileCapture {
    pub(super) fn new(queue: &Queue, labels: &[String]) -> Result<Self> {
        if labels.is_empty() {
            return Err(Error::Message("empty profile".into()));
        }
        let bytes = labels
            .len()
            .checked_mul(16)
            .ok_or_else(|| Error::Message("profile size overflow".into()))?;
        let api = MarkerApi::load(queue.device())?;
        let buffer = queue
            .device()
            .fabric()
            .allocate_shared(bytes, std::slice::from_ref(queue.device()))?;
        Ok(Self {
            buffer,
            api,
            labels: labels.to_vec(),
            queue: queue.identity(),
            recorder: NEXT_RECORDER
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| v.checked_add(1))
                .map_err(|_| Error::Message("profile recorder identity exhausted".into()))?,
            execution: 0,
            previous_end: 0,
        })
    }
    pub(crate) fn read(&mut self) -> Result<DeviceProfile> {
        let mut bytes = vec![0; self.labels.len() * 16];
        // Buffer ownership refuses reads until all device leases have retired.
        self.buffer.read(0, &mut bytes)?;
        let intervals = bytes
            .as_chunks::<16>()
            .0
            .iter()
            .zip(&self.labels)
            .map(|(bytes, label)| DeviceInterval {
                label: label.clone(),
                start_tick: u64::from_le_bytes(bytes[..8].try_into().unwrap()),
                end_tick: u64::from_le_bytes(bytes[8..].try_into().unwrap()),
            })
            .collect::<Vec<_>>();
        let (union, span) = interval_totals(&intervals)?;
        if intervals.iter().any(|i| i.start_tick <= self.previous_end) {
            return Err(Error::Message("stale or reset GPU profile counter".into()));
        }
        self.previous_end = intervals.iter().map(|i| i.end_tick).max().unwrap_or(0);
        self.execution = self
            .execution
            .checked_add(1)
            .ok_or_else(|| Error::Message("profile sample identity exhausted".into()))?;
        let ms = 1000. / self.api.frequency as f64;
        Ok(DeviceProfile {
            process: std::process::id(),
            device: self.api.device,
            queue: self.queue,
            recorder: self.recorder,
            execution: self.execution,
            frequency_hz: self.api.frequency,
            intervals,
            interval_union_ms: union as f64 * ms,
            span_ms: span as f64 * ms,
            gaps_ms: (span - union) as f64 * ms,
        })
    }
}
fn interval_totals(intervals: &[DeviceInterval]) -> Result<(u64, u64)> {
    if intervals.is_empty() {
        return Ok((0, 0));
    }
    let mut ordered = intervals.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|i| i.start_tick);
    if ordered
        .iter()
        .any(|i| i.start_tick == 0 || i.end_tick < i.start_tick)
    {
        return Err(Error::Message(
            "unwritten, reversed, or wrapped GPU timestamps".into(),
        ));
    }
    let first = ordered[0].start_tick;
    let mut end = first;
    let mut union = 0;
    for i in ordered {
        let start = end.max(i.start_tick);
        if i.end_tick > start {
            union += i.end_tick - start;
        }
        end = end.max(i.end_tick);
    }
    Ok((union, end - first))
}

/// An explicitly instrumented reusable native batch.
/// Samples synchronize before reading and cannot overwrite an in-flight sample.
pub struct ProfiledGpu {
    pub(crate) commands: PreparedGpu,
    pub(crate) capture: ProfileCapture,
}
impl ProfiledGpu {
    /// Execute and wait for one sample.
    ///
    /// # Safety
    /// The prepared kernels' contracts and cross-queue ordering must still hold.
    pub unsafe fn run(&mut self) -> Result<DeviceProfile> {
        unsafe { self.commands.dispatch()? }.wait()?;
        self.capture.read()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn union_does_not_double_count_overlap_and_detects_invalid_ticks() {
        let interval = |start_tick, end_tick| DeviceInterval {
            label: String::new(),
            start_tick,
            end_tick,
        };
        assert_eq!(
            interval_totals(&[interval(20, 30), interval(10, 25), interval(40, 45)]).unwrap(),
            (25, 35)
        );
        assert!(interval_totals(&[interval(0, 1)]).is_err());
        assert!(interval_totals(&[interval(2, 1)]).is_err());
    }
}
