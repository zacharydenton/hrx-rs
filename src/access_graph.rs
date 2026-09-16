//! Raw GPU graphs with dependencies inferred from byte-range access.

use crate::{
    Constants, Graph, GraphExec, Kernel, Node, Result, Stream, View,
    dependency_frontier::{Frontier, Use},
    execution::Access,
};

/// One graph binding and the complete access performed through it.
#[derive(Clone, Copy, Debug)]
pub struct AccessView<'a> {
    /// Bound allocation range.
    pub view: View<'a>,
    /// Reads and writes performed through the binding.
    pub access: Access,
}

/// A raw GPU graph whose edges are inferred from declared buffer access.
pub struct AccessGraph<'a> {
    graph: Graph<'a>,
    frontier: Frontier,
    nodes: Vec<Node>,
}

impl Stream {
    /// Begin an access-aware graph recording on this stream.
    pub fn access_graph(&self) -> Result<AccessGraph<'_>> {
        Ok(AccessGraph {
            graph: self.graph()?,
            frontier: Frontier::default(),
            nodes: Vec::new(),
        })
    }
}

impl<'a> View<'a> {
    /// Declare how a graph operation accesses this view.
    #[must_use]
    pub fn access(self, access: Access) -> AccessView<'a> {
        AccessView { view: self, access }
    }

    /// Declare read-only graph access.
    #[must_use]
    pub fn read(self) -> AccessView<'a> {
        self.access(Access::Read)
    }

    /// Declare write-only graph access.
    #[must_use]
    pub fn write(self) -> AccessView<'a> {
        self.access(Access::Write)
    }

    /// Declare graph access that reads and writes existing bytes.
    #[must_use]
    pub fn read_write(self) -> AccessView<'a> {
        self.access(Access::ReadWrite)
    }
}

impl<'a> AccessGraph<'a> {
    fn dependencies(&mut self, uses: impl IntoIterator<Item = AccessView<'a>>) -> Vec<Node> {
        let index = self.nodes.len();
        self.frontier
            .dependencies(
                index,
                uses.into_iter().map(|binding| Use {
                    allocation: binding.view.owner() as *const crate::Buffer as usize,
                    range: binding.view.offset()
                        ..binding.view.offset().saturating_add(binding.view.len()),
                    access: binding.access,
                }),
            )
            .into_iter()
            .map(|dependency| self.nodes[dependency])
            .collect()
    }

    fn record(&mut self, node: Node) -> Node {
        self.nodes.push(node);
        node
    }

    /// Record a byte-pattern fill.
    pub fn fill(&mut self, destination: View<'a>, pattern: u8) -> Result<Node> {
        let after = self.dependencies([destination.write()]);
        let node = self.graph.fill(&after, destination, pattern)?;
        Ok(self.record(node))
    }

    /// Record a copy between equal spans.
    pub fn copy(&mut self, destination: View<'a>, source: View<'a>) -> Result<Node> {
        let after = self.dependencies([source.read(), destination.write()]);
        let node = self.graph.copy(&after, destination, source)?;
        Ok(self.record(node))
    }

    /// Record a kernel dispatch.
    ///
    /// # Safety
    ///
    /// In addition to [`Graph::dispatch`]'s contract, every binding must declare
    /// all bytes and the complete access performed by the kernel.
    pub unsafe fn dispatch(
        &mut self,
        kernel: &'a Kernel,
        grid: [u32; 3],
        block: [u32; 3],
        constants: &Constants,
        bindings: &[AccessView<'a>],
    ) -> Result<Node> {
        let after = self.dependencies(bindings.iter().copied());
        let views = bindings
            .iter()
            .map(|binding| binding.view)
            .collect::<Vec<_>>();
        let node = unsafe {
            self.graph
                .dispatch(&after, kernel, grid, block, constants, &views)?
        };
        Ok(self.record(node))
    }

    /// Instantiate the recorded graph.
    pub fn finish(self) -> Result<GraphExec> {
        self.graph.finish()
    }

    /// Number of recorded operations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether no operations have been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}
