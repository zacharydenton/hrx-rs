use std::{
    collections::VecDeque,
    sync::{
        Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

/// Bounded trace of host-observed synchronized region latencies, including
/// driver and scheduling overhead. These are never GPU timestamps.
#[derive(Clone, Debug, serde::Serialize)]
pub struct ExecutionTrace {
    /// Events in completion order; timestamps retain actual start order.
    pub events: Vec<TraceEvent>,
    /// Oldest events discarded when capacity was exceeded.
    pub dropped: u64,
}
/// A synchronized native region as observed by its execution worker.
#[derive(Clone, Debug, serde::Serialize)]
pub struct TraceEvent {
    /// Start relative to capture start, in microseconds.
    pub start_us: f64,
    /// Host-observed execution latency, in microseconds.
    pub duration_us: f64,
    /// Upload=0, compute=1, download=2, NPU=3, host=4.
    pub lane: usize,
    /// Explicit copied bytes in the region.
    pub copied_bytes: usize,
    /// Whether execution failed.
    pub failed: bool,
}
impl ExecutionTrace {
    /// Chrome/Perfetto trace JSON with an explicit measurement-scope label.
    pub fn to_json(&self) -> crate::Result<String> {
        let events = self
            .events
            .iter()
            .map(|event| {
                serde_json::json!({
                    "name": "native region (host-observed)", "cat": "hrx.host", "ph": "X",
                    "ts": event.start_us, "dur": event.duration_us, "pid": 0, "tid": event.lane,
                    "args": {"copied_bytes": event.copied_bytes, "failed": event.failed},
                })
            })
            .collect::<Vec<_>>();
        serde_json::to_string(&serde_json::json!({
            "traceEvents": events, "displayTimeUnit": "ms", "dropped_events": self.dropped,
            "measurement": "host-observed synchronized region latency; not device timestamps",
        }))
        .map_err(|error| crate::Error::Message(error.to_string()))
    }
}
struct Capture {
    epoch: Instant,
    capacity: usize,
    events: VecDeque<TraceEvent>,
    dropped: u64,
}
#[derive(Default)]
pub(super) struct Tracer {
    enabled: AtomicBool,
    capture: Mutex<Option<Capture>>,
}
impl Tracer {
    pub fn start(&self, capacity: usize) -> crate::Result<()> {
        if capacity == 0 {
            return Err(crate::Error::Message(
                "trace capacity must be nonzero".into(),
            ));
        }
        let mut capture = self.capture.lock().unwrap_or_else(|e| e.into_inner());
        if capture.is_some() {
            return Err(crate::Error::Busy("a trace is already running".into()));
        }
        *capture = Some(Capture {
            epoch: Instant::now(),
            capacity,
            events: VecDeque::new(),
            dropped: 0,
        });
        self.enabled.store(true, Ordering::Release);
        Ok(())
    }
    pub fn finish(&self) -> Option<ExecutionTrace> {
        let mut capture = self.capture.lock().unwrap_or_else(|e| e.into_inner());
        self.enabled.store(false, Ordering::Release);
        capture.take().map(|capture| ExecutionTrace {
            events: capture.events.into(),
            dropped: capture.dropped,
        })
    }
    pub fn record(
        &self,
        start: Instant,
        elapsed: Duration,
        lane: usize,
        copied_bytes: usize,
        failed: bool,
    ) {
        if !self.enabled.load(Ordering::Acquire) {
            return;
        }
        let mut capture = self.capture.lock().unwrap_or_else(|e| e.into_inner());
        let Some(capture) = capture.as_mut() else {
            return;
        };
        let Some(offset) = start.checked_duration_since(capture.epoch) else {
            return;
        };
        if capture.events.len() == capture.capacity {
            capture.events.pop_front();
            capture.dropped += 1;
        }
        capture.events.push_back(TraceEvent {
            start_us: offset.as_secs_f64() * 1e6,
            duration_us: elapsed.as_secs_f64() * 1e6,
            lane,
            copied_bytes,
            failed,
        });
    }
}
