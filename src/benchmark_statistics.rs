//! Reusable benchmark distributions and synchronized stage timing.

use crate::{Error, Result, Stream};
use std::{collections::BTreeMap, time::Instant};

/// Select a nearest-rank percentile from sorted, nonempty samples.
///
/// `percent` must be in `1..=100`. This helper intentionally asserts its
/// preconditions: benchmark collection owns both sorting and sample counts.
#[must_use]
pub fn percentile(sorted: &[f64], percent: usize) -> f64 {
    assert!(!sorted.is_empty() && (1..=100).contains(&percent));
    sorted[(sorted.len() * percent).div_ceil(100) - 1]
}

/// Millisecond distribution measured around submission and completion.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct Distribution {
    /// Nearest-rank 50th percentile.
    pub median_ms: f64,
    /// Nearest-rank 95th percentile.
    pub p95_ms: f64,
}

impl Distribution {
    /// Summarize finite, nonempty millisecond samples.
    pub fn from_samples(mut samples: Vec<f64>) -> Result<Self> {
        if samples.is_empty()
            || samples
                .iter()
                .any(|sample| !sample.is_finite() || *sample < 0.0)
        {
            return Err(Error::Message(
                "benchmark samples must be finite, nonnegative, and nonempty".into(),
            ));
        }
        samples.sort_by(f64::total_cmp);
        Ok(Self {
            median_ms: percentile(&samples, 50),
            p95_ms: percentile(&samples, 95),
        })
    }
}

/// Named synchronized GPU stage samples collected by a benchmark.
#[derive(Default)]
pub struct StageTimings {
    samples: BTreeMap<String, Vec<f64>>,
}

impl StageTimings {
    /// Create an empty collector.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Synchronize, run one stage, synchronize again, and record elapsed milliseconds.
    pub fn measure<T>(
        &mut self,
        stream: &mut Stream,
        name: impl Into<String>,
        stage: impl FnOnce(&mut Stream) -> Result<T>,
    ) -> Result<T> {
        stream.synchronize()?;
        let start = Instant::now();
        let value = stage(stream)?;
        stream.synchronize()?;
        self.samples
            .entry(name.into())
            .or_default()
            .push(start.elapsed().as_secs_f64() * 1_000.0);
        Ok(value)
    }

    /// Add an already measured nonnegative millisecond sample.
    pub fn push(&mut self, name: impl Into<String>, milliseconds: f64) -> Result<()> {
        if !milliseconds.is_finite() || milliseconds < 0.0 {
            return Err(Error::Message("invalid benchmark duration".into()));
        }
        self.samples
            .entry(name.into())
            .or_default()
            .push(milliseconds);
        Ok(())
    }

    /// Summarize each named stage in lexical order.
    pub fn distributions(&self) -> Result<BTreeMap<String, Distribution>> {
        self.samples
            .iter()
            .map(|(name, samples)| Ok((name.clone(), Distribution::from_samples(samples.clone())?)))
            .collect()
    }

    /// Sum the samples for every stage, in milliseconds.
    #[must_use]
    pub fn totals_ms(&self) -> BTreeMap<String, f64> {
        self.samples
            .iter()
            .map(|(name, samples)| (name.clone(), samples.iter().sum()))
            .collect()
    }

    /// Drain the samples and return each stage's total milliseconds.
    pub fn take_totals_ms(&mut self) -> BTreeMap<String, f64> {
        std::mem::take(&mut self.samples)
            .into_iter()
            .map(|(name, samples)| (name, samples.into_iter().sum()))
            .collect()
    }

    /// Whether no stage samples have been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nearest_rank_includes_the_correct_order_statistic() {
        let values: Vec<_> = (1..=100).map(f64::from).collect();
        assert_eq!(percentile(&values, 50), 50.0);
        assert_eq!(percentile(&values, 95), 95.0);
        assert_eq!(percentile(&values[..50], 50), 25.0);
        assert_eq!(percentile(&values[..50], 95), 48.0);
        assert_eq!(percentile(&[7.0], 95), 7.0);
    }

    #[test]
    fn distributions_reject_bad_samples() {
        assert!(Distribution::from_samples(vec![]).is_err());
        assert!(Distribution::from_samples(vec![f64::NAN]).is_err());
        let value = Distribution::from_samples((1..=100).map(f64::from).collect()).unwrap();
        assert_eq!(value.median_ms, 50.0);
        assert_eq!(value.p95_ms, 95.0);
    }

    #[test]
    fn stage_totals_can_be_drained() {
        let mut timings = StageTimings::new();
        timings.push("load", 1.25).unwrap();
        timings.push("load", 2.75).unwrap();
        timings.push("store", 3.0).unwrap();
        assert_eq!(timings.totals_ms()["load"], 4.0);
        assert_eq!(timings.take_totals_ms()["store"], 3.0);
        assert!(timings.is_empty());
    }
}
