//! Interval frontiers keep only the ordering needed by subsequent accesses.
use super::{Access, BufferView, graph::Use};
use crate::dependency_frontier::{Frontier as IntervalFrontier, Use as IntervalUse};
use std::{collections::BTreeMap, ops::Range, sync::Arc};

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

#[derive(Default)]
pub(super) struct Frontier(IntervalFrontier);
impl Frontier {
    pub fn dependencies(&mut self, node: usize, uses: &[Use]) -> Vec<usize> {
        self.0.dependencies(
            node,
            uses.iter().map(|usage| {
                let (allocation, range) = region(&usage.view);
                IntervalUse {
                    allocation,
                    range,
                    access: usage.access,
                }
            }),
        )
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
