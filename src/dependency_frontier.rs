//! Allocation interval frontiers shared by execution schedulers and model graphs.

use crate::execution::Access;
use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Range,
};

#[derive(Clone)]
pub(crate) struct Use {
    pub allocation: usize,
    pub range: Range<usize>,
    pub access: Access,
}

#[derive(Clone, Default)]
struct Segment {
    range: Range<usize>,
    writer: Option<usize>,
    readers: BTreeSet<usize>,
}

/// Per-allocation interval frontiers retain only dependencies relevant to
/// future overlapping accesses instead of producing a dense graph.
#[derive(Default)]
pub(crate) struct Frontier(BTreeMap<usize, Vec<Segment>>);

impl Frontier {
    pub(crate) fn dependencies(
        &mut self,
        node: usize,
        uses: impl IntoIterator<Item = Use>,
    ) -> Vec<usize> {
        let mut dependencies = BTreeSet::new();
        for usage in uses {
            if usage.range.is_empty() {
                continue;
            }
            let segments = self.0.entry(usage.allocation).or_default();
            for point in [usage.range.start, usage.range.end] {
                if let Some(index) = segments
                    .iter()
                    .position(|segment| segment.range.start < point && point < segment.range.end)
                {
                    let mut right = segments[index].clone();
                    right.range.start = point;
                    segments[index].range.end = point;
                    segments.insert(index + 1, right);
                }
            }
            let mut position = usage.range.start;
            let mut index = segments.partition_point(|segment| segment.range.end <= position);
            while position < usage.range.end {
                if index == segments.len() || segments[index].range.start > position {
                    let end = segments.get(index).map_or(usage.range.end, |segment| {
                        segment.range.start.min(usage.range.end)
                    });
                    segments.insert(
                        index,
                        Segment {
                            range: position..end,
                            ..Segment::default()
                        },
                    );
                }
                let segment = &mut segments[index];
                dependencies.extend(segment.writer);
                if usage.access != Access::Read {
                    dependencies.append(&mut segment.readers);
                    segment.writer = Some(node);
                } else {
                    segment.readers.insert(node);
                }
                position = segment.range.end;
                index += 1;
            }
        }
        // Multiple arguments in one operation may overlap. Those uses update
        // the frontier in declaration order but cannot make the node depend on itself.
        dependencies.remove(&node);
        dependencies.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn usage(allocation: usize, range: Range<usize>, access: Access) -> Use {
        Use {
            allocation,
            range,
            access,
        }
    }

    #[test]
    fn preserves_only_overlapping_hazards() {
        let mut frontier = Frontier::default();
        assert!(
            frontier
                .dependencies(0, [usage(0, 0..64, Access::Write)])
                .is_empty()
        );
        assert_eq!(
            frontier.dependencies(1, [usage(0, 0..32, Access::Read)]),
            [0]
        );
        assert!(
            frontier
                .dependencies(2, [usage(0, 64..96, Access::Read)])
                .is_empty()
        );
        assert_eq!(
            frontier.dependencies(3, [usage(0, 16..80, Access::Write)]),
            [0, 1, 2]
        );
        assert_eq!(
            frontier.dependencies(4, [usage(0, 24..32, Access::Read)]),
            [3]
        );
        assert!(
            frontier
                .dependencies(5, [usage(1, 0..32, Access::Write)])
                .is_empty()
        );
    }

    #[test]
    fn one_node_can_read_and_write_the_same_region() {
        let mut frontier = Frontier::default();
        assert!(
            frontier
                .dependencies(
                    0,
                    [
                        usage(0, 0..32, Access::Read),
                        usage(0, 0..32, Access::Write),
                    ],
                )
                .is_empty()
        );
        assert_eq!(
            frontier.dependencies(1, [usage(0, 0..32, Access::Read)]),
            [0]
        );
    }
}
