//! Interval frontiers keep only the ordering needed by subsequent accesses.
use super::{Access, BufferView, graph::Use};
use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Range,
    sync::Arc,
};

fn region(view: &BufferView) -> (usize, Range<usize>) {
    (
        Arc::as_ptr(&view.buffer.storage) as usize,
        if view.buffer.storage.shared {
            0..view.buffer.len()
        } else {
            view.range.clone()
        },
    )
}

#[derive(Clone, Default)]
struct Segment {
    range: Range<usize>,
    writer: Option<usize>,
    readers: BTreeSet<usize>,
}

#[derive(Default)]
pub(super) struct Frontier(BTreeMap<usize, Vec<Segment>>);
impl Frontier {
    pub fn dependencies(&mut self, node: usize, uses: &[Use]) -> Vec<usize> {
        let mut dependencies = BTreeSet::new();
        for usage in uses {
            let (allocation, range) = region(&usage.view);
            if range.is_empty() {
                continue;
            }
            let segments = self.0.entry(allocation).or_default();
            for point in [range.start, range.end] {
                if let Some(i) = segments
                    .iter()
                    .position(|s| s.range.start < point && point < s.range.end)
                {
                    let mut right = segments[i].clone();
                    right.range.start = point;
                    segments[i].range.end = point;
                    segments.insert(i + 1, right);
                }
            }
            let mut position = range.start;
            let mut i = segments.partition_point(|s| s.range.end <= position);
            while position < range.end {
                if i == segments.len() || segments[i].range.start > position {
                    let end = segments
                        .get(i)
                        .map_or(range.end, |s| s.range.start.min(range.end));
                    segments.insert(
                        i,
                        Segment {
                            range: position..end,
                            ..Default::default()
                        },
                    );
                }
                let segment = &mut segments[i];
                dependencies.extend(segment.writer);
                if usage.access.writes() {
                    dependencies.append(&mut segment.readers);
                    segment.writer = Some(node);
                } else {
                    segment.readers.insert(node);
                }
                position = segment.range.end;
                i += 1;
            }
        }
        dependencies.remove(&node);
        dependencies.into_iter().collect()
    }
}

/// Union accesses into disjoint intervals, sorted by allocation and offset.
pub(super) fn summarize(uses: impl IntoIterator<Item = Use>) -> Vec<Use> {
    let mut allocations = BTreeMap::<usize, (BufferView, BTreeMap<usize, (i64, i64)>)>::new();
    for usage in uses {
        let (id, range) = region(&usage.view);
        let (_, events) = allocations
            .entry(id)
            .or_insert_with(|| (usage.view.clone(), BTreeMap::new()));
        let read = i64::from(usage.access != Access::Write);
        let write = i64::from(usage.access.writes());
        let start = events.entry(range.start).or_default();
        start.0 += read;
        start.1 += write;
        let end = events.entry(range.end).or_default();
        end.0 -= read;
        end.1 -= write;
    }
    let mut output: Vec<Use> = Vec::new();
    for (_, (root, events)) in allocations {
        let (mut reads, mut writes, mut previous) = (0, 0, 0);
        for (position, (read, write)) in events {
            if previous < position && (reads != 0 || writes != 0) {
                let access = match (reads != 0, writes != 0) {
                    (true, true) => Access::ReadWrite,
                    (false, true) => Access::Write,
                    _ => Access::Read,
                };
                if let Some(last) = output.last_mut().filter(|last| {
                    Arc::ptr_eq(&last.view.buffer.storage, &root.buffer.storage)
                        && last.access == access
                        && last.view.range.end == previous
                }) {
                    last.view.range.end = position;
                } else {
                    output.push(Use {
                        view: BufferView {
                            buffer: root.buffer.clone(),
                            range: previous..position,
                        },
                        access,
                    });
                }
            }
            reads += read;
            writes += write;
            previous = position;
        }
    }
    output
}

/// Both arguments are canonical summaries; sweep rather than cross-multiplying.
pub(super) fn conflicts(a: &[Use], b: &[Use]) -> bool {
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        let (ai, ar) = region(&a[i].view);
        let (bi, br) = region(&b[j].view);
        if ai < bi || (ai == bi && ar.end <= br.start) {
            i += 1;
        } else if bi < ai || br.end <= ar.start {
            j += 1;
        } else {
            if a[i].access.writes() || b[j].access.writes() {
                return true;
            }
            if ar.end <= br.end {
                i += 1;
            } else {
                j += 1;
            }
        }
    }
    false
}
