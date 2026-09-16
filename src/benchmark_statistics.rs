//! Nearest-rank quantiles for sorted, nonempty benchmark samples.

/// Select a nearest-rank percentile from sorted, nonempty samples.
///
/// `percent` must be in `1..=100`. This helper intentionally asserts its
/// preconditions: benchmark collection owns both sorting and sample counts.
#[must_use]
pub fn percentile(sorted: &[f64], percent: usize) -> f64 {
    assert!(!sorted.is_empty() && (1..=100).contains(&percent));
    sorted[(sorted.len() * percent).div_ceil(100) - 1]
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
}
