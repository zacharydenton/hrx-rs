//! Private retirement-gated kernarg/fence reuse for synchronous scratch-free GFX9 dispatch.
use super::*;

/// A bounded synchronous dispatcher with private argument/fence storage.
/// Each successful call retires and releases all operand owners before returning.
/// No prepared command or completion escapes, so retired argument bytes and the
/// completion word may be overwritten for the next call. An execution error
/// permanently disables reuse and retains its command until this owner drops.
pub struct SerialGpuDispatcher {
    queue: Queue,
    arena: GraphArena,
    maximum_argument_bytes: usize,
    failed: bool,
    uncertain: Option<PreparedGpu>,
    last_kernel: Option<Kernel>,
    last_argument_bytes: usize,
    executions: usize,
    allocations: usize,
}
impl Queue {
    /// Reuse one private argument page and fence for awaited scratch-free calls.
    /// The qualified GFX9 protocol invalidates the full scalar cache on every
    /// dispatch; immutable prepared commands elsewhere remain unchanged.
    pub fn serial_dispatcher(&self, maximum_argument_bytes: usize) -> Result<SerialGpuDispatcher> {
        if self.0.info.command_type != AMDF_QUEUE_COMMAND_TYPE_GPU_PM4
            || !matches!(self.0.cache_control, CacheControl::Gfx9)
        {
            return Err(Error::Unsupported(
                "serial argument reuse requires the qualified GFX9 PM4 cache protocol".into(),
            ));
        }
        if !(1..=4096).contains(&maximum_argument_bytes) {
            return Err(Error::Message(
                "serial argument bound must be 1..4096 bytes".into(),
            ));
        }
        Ok(SerialGpuDispatcher {
            queue: self.clone(),
            arena: GraphArena::default(),
            maximum_argument_bytes,
            failed: false,
            uncertain: None,
            last_kernel: None,
            last_argument_bytes: 0,
            executions: 0,
            allocations: 0,
        })
    }
}
impl SerialGpuDispatcher {
    /// Successful awaited executions, native storage allocations, retained bytes.
    /// Storage is at most one 64 KiB argument page and one 64-byte fence.
    pub fn stats(&self) -> (usize, usize, usize) {
        (
            self.executions,
            self.allocations,
            self.arena.page.as_ref().map_or(0, Buffer::len)
                + self.arena.fence.as_ref().map_or(0, Buffer::len),
        )
    }

    /// Pack current arguments, submit once, await retirement and release operands.
    /// New argument owners are retained through completion. Scratch and instrumented
    /// kernels are rejected by the same preparation contracts as ordinary PM4.
    ///
    /// # Safety
    /// Geometry, bindings and initialized extents must satisfy the kernel contract.
    /// Callers order conflicts on other queues. No invocation may access these
    /// private argument/fence allocations through externally managed raw addresses.
    pub unsafe fn execute(
        &mut self,
        kernel: &Kernel,
        grid: [u32; 3],
        block: [u16; 3],
        arguments: &[Argument<'_>],
    ) -> Result<()> {
        if self.failed {
            return Err(Error::DeviceLost(
                "serial dispatcher has uncertain prior execution".into(),
            ));
        }
        if !Arc::ptr_eq(&kernel.device().0, &self.queue.0.device.0) {
            return Err(Error::Message(
                "kernel belongs to another native device".into(),
            ));
        }
        if kernel.0.info.private_bytes != 0 {
            return Err(Error::Unsupported(
                "serial dispatcher requires scratch-free kernels".into(),
            ));
        }
        let (packed, _argument_owners) = kernel.arguments(arguments)?;
        if packed.len() > self.maximum_argument_bytes {
            return Err(Error::Message(
                "serial arguments exceed configured bound".into(),
            ));
        }
        // No completion/prepared handle escapes. The previous successful call
        // retired and dropped its Work, including every argument/fence lease.
        if self
            .arena
            .page
            .as_ref()
            .is_some_and(|b| !b.exclusively_owned())
            || self
                .arena
                .fence
                .as_ref()
                .is_some_and(|b| !b.exclusively_owned())
        {
            return Err(Error::Busy(
                "serial dispatch backing is still retained".into(),
            ));
        }
        self.last_kernel = None;
        self.last_argument_bytes = 0;
        self.arena.used = 0;
        let had_page = self.arena.page.is_some();
        let had_fence = self.arena.fence.is_some();
        // SAFETY: exact ordinary prepare contract; this arena has one private
        // invocation at a time and is reset only after its completed retirement.
        let prepared = unsafe {
            self.queue
                .prepare_graph(kernel, grid, block, arguments, &mut self.arena)
        };
        self.allocations += usize::from(!had_page && self.arena.page.is_some())
            + usize::from(!had_fence && self.arena.fence.is_some());
        let prepared = prepared?;
        // A new Work starts at sequence one. Also initialize the very first
        // fence explicitly; native allocation contents are not assumed zero.
        // The command is still private and has not acquired any device leases.
        self.arena
            .fence
            .as_ref()
            .unwrap()
            .write(0, &0u32.to_le_bytes())?;
        self.uncertain = Some(prepared);
        // SAFETY: validated private command owns all current arguments; no
        // storage reset or second invocation occurs until this wait completes.
        let result =
            unsafe { self.uncertain.as_ref().unwrap().dispatch() }.and_then(|done| done.wait());
        if let Err(error) = result {
            self.failed = true;
            return Err(error);
        }
        // Do not pin operand allocations between calls. Only code metadata is
        // retained to support a read-only snapshot of the latest packed bytes.
        self.uncertain = None;
        self.last_kernel = Some(kernel.clone());
        self.last_argument_bytes = packed.len();
        self.executions += 1;
        Ok(())
    }

    /// Inspect the latest successful call's current packed bytes without GPU work.
    /// A later execute overwrites this storage; this is not historical consumption evidence.
    pub fn argument_snapshot(
        &self,
        arguments: &[Argument<'_>],
    ) -> Result<PreparedArgumentSnapshot> {
        if self.failed {
            return Err(Error::DeviceLost(
                "serial dispatcher execution is uncertain".into(),
            ));
        }
        let kernel = self
            .last_kernel
            .as_ref()
            .ok_or_else(|| Error::Message("no successful serial dispatch".into()))?;
        let (expected, _owners) = kernel.arguments(arguments)?;
        if expected.len() != self.last_argument_bytes {
            return Err(Error::Message(
                "serial snapshot argument extent differs".into(),
            ));
        }
        let page = self
            .arena
            .page
            .as_ref()
            .ok_or_else(|| Error::Message("missing serial argument storage".into()))?;
        let mut observed = vec![0; self.last_argument_bytes];
        page.read(0, &mut observed)?;
        Ok(PreparedArgumentSnapshot {
            device_address: page.device_address(self.queue.device())?,
            expected,
            observed,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const SOURCE: &str = r#"
amdgpu.target<gfx906> @serial_target {subgroup_size = 64}
kernel.def target(@serial_target) export("serial_add") @serial_add() {
 %one = index.constant 1 : index
 %groups = index.constant 16 : index
 %threads = index.constant 64 : index
 kernel.launch.config workgroups(%groups,%one,%one) workgroup_size(%threads,%one,%one) : index
} launch(%input: buffer, %output: buffer, %bias: i32) {
 %off = index.constant 0 : offset
 %iv = buffer.view %input[%off] : buffer -> view<1024xi32>
 %ov = buffer.view %output[%off] : buffer -> view<1024xi32>
 %group = kernel.workgroup.id<x> : index
 %lane = kernel.workitem.id<x> : index
 %width = index.constant 64 : index
 %base = index.mul %group, %width : index
 %at = index.add %base, %lane : index
 %safe = index.assume %at [range(%at, 0, 1023)] : index
 %value = view.load %iv[%safe] : view<1024xi32> -> i32
 %result = scalar.addi %value, %bias : i32
 view.store %result, %ov[%safe] : i32, view<1024xi32>
 kernel.return
}
"#;
    #[test]
    #[ignore = "requires gfx906 and HRX_TEST_LOOM"]
    fn changing_arguments_owners_snapshots_and_rejection() -> Result<()> {
        let library = std::env::var("HRX_TEST_LOOM")
            .map_err(|_| Error::Message("HRX_TEST_LOOM required".into()))?;
        let device = Device::open(Engine::Gpu, 0)?;
        let queue = device.queue()?;
        let compiler =
            crate::loom::Compiler::for_target(Some(Path::new(&library)), device.target())?;
        let spec = crate::loom::Specialization::new("serial_add");
        let first = compiler.module(SOURCE).compile(&spec)?;
        let second = compiler
            .module(&SOURCE.replace("scalar.addi", "scalar.subi"))
            .compile(&spec)?;
        // SAFETY: both authored sources have the fixed checked three-argument ABI.
        let kernels = unsafe {
            [
                device.load_bytes(first.bytes(), "serial_add")?,
                device.load_bytes(second.bytes(), "serial_add")?,
            ]
        };
        let fabric = device.fabric();
        let inputs = [
            fabric.allocate_shared(4096, std::slice::from_ref(&device))?,
            fabric.allocate_shared(4096, std::slice::from_ref(&device))?,
        ];
        let outputs = [
            fabric.allocate_shared(4224, std::slice::from_ref(&device))?,
            fabric.allocate_shared(4224, std::slice::from_ref(&device))?,
        ];
        assert!(queue.serial_dispatcher(0).is_err());
        assert!(queue.serial_dispatcher(4097).is_err());
        let mut slot = queue.serial_dispatcher(64)?;
        assert_eq!(slot.stats(), (0, 0, 0));
        let mut address = None;
        for call in 0..32i32 {
            let which = call as usize % 2;
            let values: Vec<i32> = (0..1024).map(|x| x * 31 - call * 17).collect();
            inputs[which].write(
                0,
                &values
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            outputs[which].write(0, &vec![0xa5; 4224])?;
            let output = outputs[which].view(64, 4096)?;
            let bias = (call * 113 - 421).to_le_bytes();
            let args = [
                Argument::Buffer(&inputs[which], 0),
                Argument::Buffer(&output, 0),
                Argument::Value(&bias),
            ];
            // SAFETY:1024 initialized i32 inputs, uniquely covered output, initialized guards.
            unsafe { slot.execute(&kernels[which], [16, 1, 1], [64, 1, 1], &args) }?;
            let snapshot = slot.argument_snapshot(&args)?;
            assert_eq!(snapshot.expected, snapshot.observed);
            if let Some(prior) = address {
                assert_eq!(prior, snapshot.device_address);
            } else {
                address = Some(snapshot.device_address);
            }
            let mut actual = vec![0; 4224];
            outputs[which].read(0, &mut actual)?;
            assert!(
                actual[..64]
                    .iter()
                    .chain(actual[4160..].iter())
                    .all(|x| *x == 0xa5)
            );
            let delta = i32::from_le_bytes(bias) * if which == 0 { 1 } else { -1 };
            assert_eq!(
                &actual[64..4160],
                values
                    .iter()
                    .flat_map(|x| (x + delta).to_le_bytes())
                    .collect::<Vec<_>>()
            );
            assert!(
                inputs[which].exclusively_owned(),
                "dispatcher pinned an operand after retirement"
            );
            let wrong_bias = 991i32.to_le_bytes();
            let wrong = [
                Argument::Buffer(&inputs[which], 0),
                Argument::Buffer(&output, 0),
                Argument::Value(&wrong_bias),
            ];
            let wrong_snapshot = slot.argument_snapshot(&wrong)?;
            assert_ne!(wrong_snapshot.expected, wrong_snapshot.observed);
            // SAFETY: intentionally incomplete arguments must reject before dispatch.
            assert!(
                unsafe { slot.execute(&kernels[which], [16, 1, 1], [64, 1, 1], &args[..2]) }
                    .is_err()
            );
            assert_eq!(slot.stats(), (call as usize + 1, 2, 65600));
            drop(output);
            assert!(outputs[which].exclusively_owned());
        }
        let retained = slot.arena.page.as_ref().unwrap().clone();
        let output = outputs[0].view(64, 4096)?;
        let bias = 1i32.to_le_bytes();
        let args = [
            Argument::Buffer(&inputs[0], 0),
            Argument::Buffer(&output, 0),
            Argument::Value(&bias),
        ];
        // SAFETY: valid bindings; an injected extra arena owner must block reuse.
        assert!(matches!(
            unsafe { slot.execute(&kernels[0], [16, 1, 1], [64, 1, 1], &args) },
            Err(Error::Busy(_))
        ));
        drop(retained);
        // SAFETY: same fully bounded initialized bindings after external owner release.
        unsafe { slot.execute(&kernels[0], [16, 1, 1], [64, 1, 1], &args) }?;
        let mut small = queue.serial_dispatcher(8)?;
        // SAFETY: valid ABI, but configured storage bound is intentionally insufficient.
        assert!(unsafe { small.execute(&kernels[0], [16, 1, 1], [64, 1, 1], &args) }.is_err());
        assert_eq!(small.stats(), (0, 0, 0));
        slot.failed = true;
        // SAFETY: valid bindings; injected uncertain state must reject without publication.
        assert!(matches!(
            unsafe { slot.execute(&kernels[0], [16, 1, 1], [64, 1, 1], &args) },
            Err(Error::DeviceLost(_))
        ));
        assert_eq!(slot.stats(), (33, 2, 65600));
        Ok(())
    }
}
