//! Hardware qualification for owned asynchronous model slots.

use hrx::{
    Error,
    execution::RuntimeOptions,
    inference::{InferenceGraph, ModelContext, PreparedModel},
    tensor::{DType, TensorDesc},
};

#[test]
#[ignore = "requires GPU and Loom compiler"]
fn graph_reuses_private_scratch_without_aliasing_outputs() -> hrx::Result<()> {
    use hrx::{
        loom::Specialization,
        model::{Command, Dispatch, ModelSession},
    };
    let context = ModelContext::new(Default::default())?;
    let mut model = ModelSession::in_context(&context)?;
    let temporary = model.allocate(64)?;
    let output = model.allocate_shared(64)?;
    let source = r#"
amdgpu.target<gfx11-generic> @t {subgroup_size = 32}
kernel.def target(@t) export("copy") @copy(%unused: index) {
  %one = index.constant 1 : index
  %threads = index.constant 64 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%threads, %one, %one) : index
} launch(%unused: index, %input: buffer, %output: buffer) {
  %zero = index.constant 0 : offset
  %lane = kernel.workitem.id<x> : index
  %i = index.assume %lane [range(%lane, 0, 63)] : index
  %ig = buffer.assume.memory_space<global> %input : buffer
  %og = buffer.assume.memory_space<global> %output : buffer
  %iv = buffer.view %ig[%zero] : buffer -> view<64xi8>
  %ov = buffer.view %og[%zero] : buffer -> view<64xi8>
  %v = view.load %iv[%i] : view<64xi8> -> i8
  view.store %v, %ov[%i] : i8, view<64xi8>
  kernel.return
}
"#;
    let kernel = unsafe { model.compile(&[(source, Specialization::new("copy"))])? }[0];
    let engine = model.freeze(&context)?;
    let fragment = |value| -> hrx::Result<_> {
        // The full temporary is initialized by Fill before its only read.
        unsafe {
            Ok(engine
                .fragment(
                    &[
                        Command::Fill {
                            region: temporary,
                            value,
                        },
                        Command::Dispatch(Dispatch::indices(
                            kernel,
                            [0],
                            [1, 1, 1],
                            vec![temporary.read(), output.write()],
                        )),
                    ],
                    &[],
                    &[(output, TensorDesc::new(DType::U8, vec![64])?)],
                )?
                .reuse_private_scratch())
        }
    };
    let first = fragment(7)?;
    let second = fragment(11)?;
    let before = context.runtime().statistics();
    let plan = PreparedModel::prepare(&context, 2, |context| {
        let mut graph = context.runtime().graph();
        let mut outputs = first.record(&mut graph, &[])?;
        outputs.extend(second.record(&mut graph, &[])?);
        Ok(InferenceGraph {
            inputs: vec![],
            outputs,
            graph: graph.prepare()?,
        })
    })?;
    let after = context.runtime().statistics();
    // Two independent slots; each has one temporary and two distinct outputs.
    assert_eq!(after.allocations - before.allocations, 6);
    for _ in 0..3 {
        let a = plan.submit_host(&[])?;
        let b = plan.submit_host(&[])?;
        assert_eq!(b.download()?.wait()?, vec![vec![7; 64], vec![11; 64]]);
        assert_eq!(a.download()?.wait()?, vec![vec![7; 64], vec![11; 64]]);
    }
    Ok(())
}

#[test]
#[ignore = "requires GPU and Loom compiler"]
fn packed_affine_reuses_geometry_and_bounds_invalid_records() -> hrx::Result<()> {
    use hrx::{
        execution::MemoryPlacement,
        image::{ImageOps, RgbSampling},
    };
    let context = ModelContext::new(Default::default())?;
    let ops = ImageOps::new(&context, 1)?;
    let fragment = ops.affine_packed_rgb_fragment(
        &TensorDesc::new(DType::U8, vec![48])?,
        4,
        2,
        2,
        RgbSampling::BlackTiesEven,
    )?;
    let plan = PreparedModel::prepare(&context, 1, |context| {
        let inputs = [
            TensorDesc::new(DType::U8, vec![48])?,
            TensorDesc::new(DType::F32, vec![4, 2, 3])?,
            TensorDesc::new(DType::U32, vec![4, 3])?,
        ]
        .into_iter()
        .map(|desc| context.allocate_with(desc, MemoryPlacement::HostVisible))
        .collect::<hrx::Result<Vec<_>>>()?;
        let mut graph = context.runtime().graph();
        let outputs = fragment.record(&mut graph, &inputs)?;
        Ok(InferenceGraph {
            inputs,
            outputs,
            graph: graph.prepare()?,
        })
    })?;
    let pixels: Vec<u8> = (0..48).collect();
    let maps: Vec<u8> = [1f32, 0., 0., 0., 1., 0.]
        .into_iter()
        .cycle()
        .take(24)
        .flat_map(f32::to_le_bytes)
        .collect();
    let geometry = |records: [[u32; 3]; 4]| {
        records
            .into_iter()
            .flatten()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>()
    };
    let meta = geometry([[12, 3, 2], [0, 2, 2], [12, 3, 2], [30, 2, 3]]);
    let expected: Vec<u8> = [12..18, 21..27, 0..12, 12..18, 21..27, 30..42]
        .into_iter()
        .flat_map(|range| pixels[range].iter().copied())
        .collect();
    assert_eq!(
        plan.submit_host(&[&pixels, &maps, &meta])?
            .download()?
            .wait()?[0],
        expected
    );
    let invalid = geometry([
        [u32::MAX, 2, 2],
        [0, 0, 1],
        [0, u32::MAX, u32::MAX],
        [47, 2, 2],
    ]);
    assert_eq!(
        plan.submit_host(&[&pixels, &maps, &invalid])?
            .download()?
            .wait()?[0],
        vec![0; 48]
    );
    let slot = plan.try_acquire()?;
    assert!(
        slot.submit_host_with(|_, _| Err(hrx::Error::Message("cancel publication".into())))
            .is_err()
    );
    assert!(plan.is_idle());
    assert_eq!(
        plan.try_acquire()?
            .submit_host_with(|i, dst| {
                dst.copy_from_slice([pixels.as_slice(), &maps, &meta][i]);
                Ok(())
            })?
            .download()?
            .wait()?[0],
        expected
    );
    Ok(())
}

#[test]
#[ignore = "requires GPU and Loom compiler"]
fn row_gather_fragment_binds_resident_source_and_reuses_mapped_io() -> hrx::Result<()> {
    use hrx::{execution::MemoryPlacement, tensor::TensorOps};
    let context = ModelContext::new(Default::default())?;
    let ops = TensorOps::new(&context, 1)?;
    let desc = TensorDesc::new(DType::U8, vec![4, 7])?;
    assert!(ops.gather_rows_fragment(&desc, 0).is_err());
    assert!(
        ops.gather_rows_fragment(&TensorDesc::new(DType::U8, vec![0, 7])?, 2)
            .is_err()
    );
    let gather = ops.gather_rows_fragment(&desc, 3)?;
    let source = context.allocate_with(desc, MemoryPlacement::HostVisible)?;
    let plan = PreparedModel::prepare(&context, 1, |context| {
        let ids = context.allocate_with(
            TensorDesc::new(DType::U32, vec![3])?,
            MemoryPlacement::HostVisible,
        )?;
        let mut graph = context.runtime().graph();
        let outputs = gather.record(&mut graph, &[source.clone(), ids.clone()])?;
        Ok(InferenceGraph {
            inputs: vec![source.clone(), ids],
            outputs,
            graph: graph.prepare()?,
        })
    })?;
    let bytes: Vec<u8> = (0..28).collect();
    let ids: Vec<u8> = [3u32, 0, 3]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect();
    let expected: Vec<u8> = [3, 0, 3]
        .into_iter()
        .flat_map(|i| bytes[i * 7..][..7].iter().copied())
        .collect();
    assert_eq!(
        plan.submit_host(&[&bytes, &ids])?.download()?.wait()?,
        vec![expected.clone()]
    );
    let before = context.runtime().statistics();
    for _ in 0..3 {
        assert_eq!(
            plan.submit_host(&[&bytes, &ids])?.download()?.wait()?,
            vec![expected.clone()]
        );
    }
    let after = context.runtime().statistics();
    assert_eq!(after.allocations, before.allocations);
    assert_eq!(after.copied_bytes, before.copied_bytes);
    assert_eq!(after.native_graphs_prepared, before.native_graphs_prepared);
    assert_eq!(after.submissions - before.submissions, 3);
    Ok(())
}

#[test]
#[ignore = "requires GPU and Loom compiler"]
fn fragments_share_one_graph_without_intermediate_copies_or_allocations() -> hrx::Result<()> {
    use hrx::{image::ImageOps, tensor::Layout};
    let context = ModelContext::new(Default::default())?;
    let ops = ImageOps::new(&context, 1)?;
    let desc = TensorDesc::new(DType::U8, vec![1, 4, 4, 3])?.with_layout(Layout::Nhwc)?;
    let normalized = TensorDesc::new(DType::F32, vec![1, 3, 4, 4])?.with_layout(Layout::Nchw)?;
    let normalize = ops.normalize_rgb_fragment(&desc, [0.; 3], [1.; 3])?;
    let patchify = ops.patchify_fragment(&normalized, 2)?;
    assert_eq!(context.runtime().statistics().native_graphs_prepared, 0);
    let plan = PreparedModel::prepare(&context, 2, |context| {
        let input = context.allocate(desc.clone())?;
        let mut graph = context.runtime().graph();
        let normalized = normalize.record(&mut graph, std::slice::from_ref(&input))?;
        let output = patchify.record(&mut graph, &normalized)?;
        Ok(InferenceGraph {
            inputs: vec![input],
            outputs: output,
            graph: graph.prepare()?,
        })
    })?;
    assert_eq!(context.runtime().statistics().native_graphs_prepared, 2);
    // Neither kernel depends on these builders or their definitions staying alive.
    drop((ops, normalize, patchify));
    let rgb: Vec<_> = (0..48).collect();
    let mut expected = Vec::new();
    for py in 0..2 {
        for px in 0..2 {
            for c in 0..3 {
                for dy in 0..2 {
                    for dx in 0..2 {
                        let value = rgb[((py * 2 + dy) * 4 + px * 2 + dx) * 3 + c] as f32 / 255.;
                        expected.extend_from_slice(&value.to_le_bytes());
                    }
                }
            }
        }
    }
    let a = plan.submit_host(&[&rgb])?;
    let b = plan.submit_host(&[&rgb])?;
    assert!(matches!(plan.try_acquire(), Err(Error::Busy(_))));
    assert_eq!(a.download()?.wait()?, vec![expected.clone()]);
    assert_eq!(b.download()?.wait()?, vec![expected.clone()]);
    let before = context.runtime().statistics();
    for _ in 0..8 {
        assert_eq!(
            plan.submit_host(&[&rgb])?.download()?.wait()?,
            vec![expected.clone()]
        );
    }
    let after = context.runtime().statistics();
    assert_eq!(after.allocations, before.allocations);
    assert_eq!(after.native_graphs_prepared, before.native_graphs_prepared);
    assert_eq!(after.device_copied_bytes, 0);
    assert_eq!(after.submissions - before.submissions, 8 * 3); // upload, single compute graph, readback
    Ok(())
}

#[test]
#[ignore = "requires the GPU runtime"]
fn fragment_binding_rejects_undersized_foreign_and_new_aliases() -> hrx::Result<()> {
    use hrx::model::{Command, ModelSession};
    let context = ModelContext::new(Default::default())?;
    let mut model = ModelSession::in_context(&context)?;
    let a = model.allocate(64)?;
    let b = model.allocate(64)?;
    let definition = model.freeze(&context)?;
    let desc = TensorDesc::new(DType::U8, vec![32])?;
    let fragment = unsafe {
        definition.fragment(
            &[Command::Fill {
                region: a,
                value: 7,
            }],
            &[(a, desc.clone())],
            &[(a, desc.clone())],
        )?
    };
    let input = context.allocate(desc.clone())?;
    let mut graph = context.runtime().graph();
    assert!(
        fragment
            .record(&mut graph, std::slice::from_ref(&input))
            .is_err()
    );
    let foreign = ModelContext::new(Default::default())?;
    assert!(
        fragment
            .record(&mut foreign.runtime().graph(), std::slice::from_ref(&input))
            .is_err()
    );
    let desc = TensorDesc::new(DType::U8, vec![64])?;
    let fragment = unsafe {
        definition.fragment(
            &[Command::Fill {
                region: a,
                value: 7,
            }],
            &[(a, desc.clone()), (b, desc.clone())],
            &[(a, desc.clone())],
        )?
    };
    let input = context.allocate(desc)?;
    assert!(
        fragment
            .record(&mut graph, &[input.clone(), input])
            .is_err()
    );
    // Rejected recording must not have appended even the otherwise-valid fill.
    assert!(graph.prepare().is_err());
    Ok(())
}

#[test]
#[ignore = "requires the GPU runtime"]
fn model_io_uses_sliced_scratch_without_hidden_copies() -> hrx::Result<()> {
    use hrx::model::{Command, ModelSession};
    let context = ModelContext::new(Default::default())?;
    let mut session = ModelSession::in_context(&context)?;
    let scratch = session.allocate(4096)?;
    let input = scratch.slice(8, 64)?;
    let output = scratch.slice(32, 64)?;
    let fill = scratch.slice(40, 8)?;
    let definition = session.freeze(&context)?;
    let desc = TensorDesc::new(DType::U8, vec![64])?;
    // Fill has no native kernel contract; overlapping IO is intentional.
    let plan = unsafe {
        definition.prepare(
            &[Command::Fill {
                region: fill,
                value: 7,
            }],
            &[(input, desc.clone())],
            &[(output, desc)],
            2,
        )?
    };
    let prepared = context.runtime().statistics();
    assert_eq!(prepared.allocations, 2);
    assert_eq!(prepared.live_bytes, 2 * 96);
    let a = plan.submit_host(&[&[9; 64]])?;
    let b = plan.submit_host(&[&[11; 64]])?;
    for (result, value) in [(a, 9), (b, 11)] {
        let mut expected = vec![0; 64];
        expected[..40].fill(value);
        expected[8..16].fill(7);
        assert_eq!(result.download()?.wait()?, vec![expected]);
    }
    let warm = context.runtime().statistics();
    assert_eq!(warm.device_copied_bytes, 0);
    assert_eq!(warm.uploaded_bytes, 128);
    assert_eq!(warm.downloaded_bytes, 128);
    for _ in 0..10 {
        plan.submit_host(&[&[5; 64]])?.download()?.wait()?;
    }
    let after = context.runtime().statistics();
    assert_eq!(after.allocations, warm.allocations);
    assert_eq!(after.native_graphs_prepared, warm.native_graphs_prepared);
    assert_eq!(after.device_copied_bytes, 0);
    Ok(())
}

#[test]
#[ignore = "requires the GPU runtime"]
fn host_visible_model_io_preserves_slices_slots_and_avoids_staging() -> hrx::Result<()> {
    use hrx::model::{Command, ModelSession};
    let context = ModelContext::new(Default::default())?;
    let mut session = ModelSession::in_context(&context)?;
    let root = session.allocate_shared(4096)?;
    let io = root.slice(16, 64)?;
    let desc = TensorDesc::new(DType::U8, vec![64])?;
    let plan = unsafe {
        session.freeze(&context)?.prepare(
            &[Command::Fill {
                region: root.slice(32, 8)?,
                value: 7,
            }],
            &[(io, desc.clone())],
            &[(io, desc)],
            2,
        )?
    };
    let before = context.runtime().statistics();
    assert_eq!(before.live_bytes, 2 * 80);
    let a = plan.submit_host(&[&[9; 64]])?;
    let b = plan.submit_host(&[&[11; 64]])?;
    assert!(a.outputs()[0].binding().unwrap().is_host_visible());
    assert!(matches!(plan.try_acquire(), Err(Error::Busy(_))));
    let a = a.download()?;
    assert!(matches!(plan.try_acquire(), Err(Error::Busy(_))));
    let mut expected = vec![9; 64];
    expected[16..24].fill(7);
    assert_eq!(a.wait()?, vec![expected]);
    let mut actual = [0; 64];
    b.download()?.read_into(&mut [&mut actual])?;
    let mut expected = [11; 64];
    expected[16..24].fill(7);
    assert_eq!(actual, expected);
    for value in 0..8 {
        let mut expected = vec![value; 64];
        expected[16..24].fill(7);
        assert_eq!(
            plan.submit_host(&[&[value; 64]])?.download()?.wait()?,
            vec![expected]
        );
    }
    let after = context.runtime().statistics();
    assert_eq!(after.allocations, before.allocations);
    assert_eq!(after.native_graphs_prepared, before.native_graphs_prepared);
    assert_eq!(after.submissions - before.submissions, 10);
    assert_eq!(after.uploaded_bytes, before.uploaded_bytes);
    assert_eq!(after.downloaded_bytes, before.downloaded_bytes);
    Ok(())
}

#[test]
#[ignore = "requires the GPU runtime"]
fn mixed_host_and_device_io_only_stages_device_bindings() -> hrx::Result<()> {
    use hrx::execution::MemoryPlacement;
    let context = ModelContext::new(Default::default())?;
    let desc = TensorDesc::new(DType::U8, vec![32])?;
    let plan = PreparedModel::prepare(&context, 1, |context| {
        let host = context.allocate_with(desc.clone(), MemoryPlacement::HostVisible)?;
        let device = context.allocate(desc.clone())?;
        let mut graph = context.runtime().graph();
        graph.fill(host.binding().unwrap().slice(0..1)?, 7)?;
        graph.fill(device.binding().unwrap().slice(0..1)?, 8)?;
        Ok(InferenceGraph {
            inputs: vec![host.clone(), device.clone()],
            outputs: vec![host, device],
            graph: graph.prepare()?,
        })
    })?;
    for value in 0..3 {
        let before = context.runtime().statistics();
        let output = plan
            .submit_host(&[&[value; 32], &[value + 10; 32]])?
            .download()?
            .wait()?;
        let mut a = vec![value; 32];
        a[0] = 7;
        let mut b = vec![value + 10; 32];
        b[0] = 8;
        assert_eq!(output, vec![a, b]);
        let after = context.runtime().statistics();
        assert_eq!(after.uploaded_bytes - before.uploaded_bytes, 32);
        assert_eq!(after.downloaded_bytes - before.downloaded_bytes, 32);
        if value > 0 {
            assert_eq!(after.allocations, before.allocations);
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires the GPU runtime"]
fn slot_factory_rejects_foreign_graphs_tensors_and_inconsistent_shapes() -> hrx::Result<()> {
    let context = ModelContext::new(Default::default())?;
    let foreign = ModelContext::new(Default::default())?;
    for foreign_graph in [false, true] {
        let result = PreparedModel::prepare(&context, 1, |context| {
            let tensor_context = if foreign_graph { context } else { &foreign };
            let graph_context = if foreign_graph { &foreign } else { context };
            let tensor = tensor_context.allocate(TensorDesc::new(DType::U8, vec![32])?)?;
            let buffer = graph_context.allocate(TensorDesc::new(DType::U8, vec![32])?)?;
            let mut graph = graph_context.runtime().graph();
            graph.fill(buffer.binding().unwrap(), 0)?;
            Ok(InferenceGraph {
                inputs: vec![],
                outputs: vec![tensor],
                graph: graph.prepare()?,
            })
        });
        assert!(result.is_err());
    }
    let mut count = 0;
    assert!(
        PreparedModel::prepare(&context, 2, |context| {
            count += 1;
            let tensor = context.allocate(TensorDesc::new(DType::U8, vec![count])?)?;
            let mut graph = context.runtime().graph();
            graph.fill(tensor.binding().unwrap(), 0)?;
            Ok(InferenceGraph {
                inputs: vec![],
                outputs: vec![tensor],
                graph: graph.prepare()?,
            })
        })
        .is_err()
    );
    let shared = context.allocate(TensorDesc::new(DType::U8, vec![32])?)?;
    assert!(
        PreparedModel::prepare(&context, 2, |context| {
            let mut graph = context.runtime().graph();
            graph.fill(shared.binding().unwrap(), 0)?;
            Ok(InferenceGraph {
                inputs: vec![],
                outputs: vec![shared.clone()],
                graph: graph.prepare()?,
            })
        })
        .is_err()
    );
    Ok(())
}

#[test]
#[ignore = "requires the GPU runtime"]
fn lazy_staging_budget_failure_releases_slot_and_can_retry() -> hrx::Result<()> {
    let manager = hrx::residency::ResidencyManager::new(128)?;
    let budget = manager.budget();
    let held = budget.reserve(64)?;
    let context = ModelContext::new(RuntimeOptions {
        memory_budget: Some(budget),
        ..Default::default()
    })?;
    let plan = PreparedModel::prepare(&context, 1, |context| {
        let input = context.allocate(TensorDesc::new(DType::U8, vec![32])?)?;
        let output = context.allocate(input.desc().clone())?;
        let mut graph = context.runtime().graph();
        graph.copy(output.binding().unwrap(), input.binding().unwrap())?;
        Ok(InferenceGraph {
            inputs: vec![input],
            outputs: vec![output],
            graph: graph.prepare()?,
        })
    })?;
    assert!(plan.submit_host(&[&[17; 32]]).is_err());
    assert!(plan.is_idle());
    assert_eq!(context.runtime().statistics().live_bytes, 64);
    drop(held);
    assert_eq!(
        plan.submit_host(&[&[17; 32]])?.download()?.wait()?,
        vec![vec![17; 32]]
    );
    assert_eq!(context.runtime().statistics().live_bytes, 128);
    Ok(())
}

#[test]
#[ignore = "requires GPU and Loom compiler"]
fn row_gather_preserves_bits_order_and_output_leases() -> hrx::Result<()> {
    use hrx::tensor::TensorOps;
    let context = ModelContext::new(Default::default())?;
    let ops = TensorOps::new(&context, 4)?;
    for dtype in [DType::U8, DType::F16, DType::F32, DType::I64] {
        let desc = TensorDesc::new(dtype, vec![5, 7])?;
        let bytes: Vec<_> = (0..desc.bytes()).map(|i| (i * 37) as u8).collect();
        let input = context.upload(desc, &bytes)?;
        let indices = [4, 0, 4];
        let rowbytes = 7 * dtype.bytes();
        let expected: Vec<_> = indices
            .iter()
            .flat_map(|&i| bytes[i * rowbytes..(i + 1) * rowbytes].iter().copied())
            .collect();
        let a = ops.gather_rows(&input, &indices)?;
        let b = ops.gather_rows(&input, &indices)?;
        let c = ops.gather_rows(&input, &indices)?;
        a.completion().wait()?;
        assert!(matches!(
            ops.gather_rows(&input, &indices),
            Err(Error::Busy(_))
        ));
        let retained = a.clone();
        drop(a);
        assert_eq!(context.download(&retained)?.wait()?, expected);
        assert!(matches!(
            ops.gather_rows(&input, &indices),
            Err(Error::Busy(_))
        ));
        drop(retained);
        let next = ops.gather_rows(&input, &[1, 2, 3])?;
        assert_eq!(
            context.download(&next)?.wait()?,
            bytes[rowbytes..4 * rowbytes]
        );
        assert!(ops.gather_rows(&input, &[5]).is_err());
        assert!(ops.gather_rows(&input, &[usize::MAX]).is_err());
        let empty = ops.gather_rows(&input, &[])?;
        assert_eq!(empty.desc().shape(), [0, 7]);
        assert!(context.download(&empty)?.wait()?.is_empty());
        drop((b, c, next));
    }
    let other =
        ModelContext::new(Default::default())?.allocate(TensorDesc::new(DType::U8, vec![1, 1])?)?;
    assert!(ops.gather_rows(&other, &[0]).is_err());
    let scalar = context.allocate(TensorDesc::new(DType::U8, vec![])?)?;
    assert!(ops.gather_rows(&scalar, &[]).is_err());
    Ok(())
}

#[test]
#[ignore = "requires GPU and Loom compiler"]
fn rgb_views_roundtrip_batches_and_reject_invalid_shapes() -> hrx::Result<()> {
    use hrx::{
        image::{ImageOps, RgbEncoding},
        tensor::Layout,
    };
    let context = ModelContext::new(Default::default())?;
    let ops = ImageOps::new(&context, 8)?;
    let desc = TensorDesc::new(DType::U8, vec![2, 6, 6, 3])?.with_layout(Layout::Nhwc)?;
    for encoding in [RgbEncoding::Unit, RgbEncoding::Symmetric] {
        for factor in [1, 2, 3] {
            let encode = ops.prepare_encode_rgb_views(&desc, factor, encoding)?;
            for seed in [0, 113] {
                let rgb: Vec<_> = (0..desc.bytes())
                    .map(|i| ((i * 37 + seed) % 256) as u8)
                    .collect();
                let encoded = encode.submit_host(&[&rgb])?;
                let views = &encoded.outputs()[0];
                let decode = ops.prepare_decode_rgb_views(views.desc(), factor, encoding)?;
                let before = context.runtime().statistics().downloaded_bytes;
                let restored = decode.submit(std::slice::from_ref(views))?;
                drop(encoded);
                let bytes = restored.download()?.wait()?.remove(0);
                assert_eq!(
                    context.runtime().statistics().downloaded_bytes - before,
                    desc.bytes() as u64 * 4
                );
                for (actual, expected) in bytes.as_chunks::<4>().0.iter().zip(&rgb) {
                    let value = f32::from_le_bytes(*actual);
                    if encoding == RgbEncoding::Symmetric {
                        assert_eq!(value, f32::from(*expected));
                    } else {
                        assert!((value - f32::from(*expected)).abs() < 2e-5);
                    }
                }
            }
        }
    }
    for factor in [0, 4, usize::MAX] {
        assert!(
            ops.prepare_encode_rgb_views(&desc, factor, RgbEncoding::Unit)
                .is_err()
        );
    }
    for invalid in [
        TensorDesc::new(DType::U8, vec![0, 6, 6, 3])?.with_layout(Layout::Nhwc)?,
        TensorDesc::new(DType::U8, vec![1, 6, 6, 4])?.with_layout(Layout::Nhwc)?,
        TensorDesc::new(DType::F32, vec![1, 6, 6, 3])?.with_layout(Layout::Nhwc)?,
        TensorDesc::strided(DType::U8, vec![1, 6, 6, 3], vec![144, 24, 4, 1])?
            .with_layout(Layout::Nhwc)?,
    ] {
        assert!(
            ops.prepare_encode_rgb_views(&invalid, 1, RgbEncoding::Unit)
                .is_err()
        );
    }
    let invalid_views = TensorDesc::new(DType::F32, vec![3, 3, 2, 2])?.with_layout(Layout::Nchw)?;
    assert!(
        ops.prepare_decode_rgb_views(&invalid_views, 2, RgbEncoding::Unit)
            .is_err()
    );
    Ok(())
}

#[test]
#[ignore = "requires GPU and Loom compiler"]
fn composite_masks_regions_and_nonfinite_coordinates_preserve_contracts() -> hrx::Result<()> {
    use hrx::{
        image::{ImageOps, RgbComposite},
        tensor::Layout,
    };
    let context = ModelContext::new(Default::default())?;
    let ops = ImageOps::new(&context, 2)?;
    let desc = TensorDesc::new(DType::U8, vec![1, 2, 3, 3])?.with_layout(Layout::Nhwc)?;
    let crop_desc = TensorDesc::new(DType::F32, vec![1, 2, 3, 3])?.with_layout(Layout::Nhwc)?;
    let frame = [21u8; 18];
    let floats = |values: &[f32]| {
        values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>()
    };
    let crop = floats(&[200.25; 18]);
    for mode in [RgbComposite::Truncate, RgbComposite::QuantizedBlend] {
        let plan = ops.prepare_composite_rgb(&desc, &crop_desc, mode)?;
        for alpha in [-1f32, 0., 0.5, 1., 2.] {
            for blend in [0f32, 0.5, 1.] {
                let mask = floats(&[alpha; 6]);
                let matrix = floats(&[1., 0., 0., 0., 1., 0.]);
                let region = floats(&[1., 0., 3., 1., blend]);
                let output = plan
                    .submit_host(&[&frame, &crop, &mask, &matrix, &region])?
                    .download()?
                    .wait()?
                    .remove(0);
                let a = alpha.clamp(0., 1.);
                let face = if mode == RgbComposite::Truncate {
                    200.25
                } else {
                    200.
                };
                let value = (21. * (1. - a) + face * a).clamp(0., 255.).floor();
                let value = if mode == RgbComposite::QuantizedBlend {
                    (21. * (1. - blend) + value * blend).round_ties_even()
                } else {
                    value
                } as u8;
                let mut expected = frame;
                expected[3..9].fill(value);
                assert_eq!(output, expected, "{mode:?}, alpha={alpha}, blend={blend}");
            }
        }
        for coordinate in [f32::NAN, f32::INFINITY, -1e30, 1e30] {
            let output = plan
                .submit_host(&[
                    &frame,
                    &crop,
                    &floats(&[1.; 6]),
                    &floats(&[1., 0., coordinate, 0., 1., 0.]),
                    &floats(&[0., 0., 3., 2., 1.]),
                ])?
                .download()?
                .wait()?
                .remove(0);
            assert_eq!(output, frame);
        }
    }
    let image = context.upload(desc, &frame)?;
    let crop = context.upload(crop_desc, &crop)?;
    let mask = context.upload(TensorDesc::new(DType::F32, vec![2, 3])?, &floats(&[1.; 6]))?;
    let matrix = context.upload(
        TensorDesc::new(DType::F32, vec![1, 2, 3])?,
        &floats(&[1., 0., 0., 0., 1., 0.]),
    )?;
    for (region, blend) in [
        ([0, 0, 4, 2], 1.),
        ([2, 0, 1, 2], 1.),
        ([0, 0, 3, 2], f32::NAN),
        ([0, 0, 3, 2], 2.),
    ] {
        assert!(
            ops.composite_rgb(
                &image,
                &crop,
                &mask,
                &matrix,
                region,
                blend,
                RgbComposite::Truncate
            )
            .is_err()
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires GPU and Loom compiler"]
fn finite_reduction_handles_tail_nonfinite_values_and_replay() -> hrx::Result<()> {
    let context = ModelContext::new(Default::default())?;
    let ops = hrx::tensor::TensorOps::new(&context, 8)?;
    for count in [1, 255, 256, 257, 65537] {
        let desc = TensorDesc::new(DType::F32, vec![count])?;
        let plan = ops.prepare_finite(&desc)?;
        for value in [
            0.,
            f32::MAX,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -0.,
        ] {
            let mut data = vec![1f32; count];
            data[count - 1] = value;
            let bytes = data
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>();
            let before = context.runtime().statistics().downloaded_bytes;
            let flag = plan.submit_host(&[&bytes])?.download()?.wait()?.remove(0);
            assert_eq!(
                i32::from_le_bytes(flag.try_into().unwrap()),
                i32::from(!value.is_finite())
            );
            assert_eq!(context.runtime().statistics().downloaded_bytes - before, 4);
        }
    }
    assert!(
        ops.prepare_finite(&TensorDesc::new(DType::F32, vec![0])?)
            .is_err()
    );
    Ok(())
}

#[test]
#[ignore = "requires GPU and Loom compiler"]
fn similarity_fitting_checks_geometry_and_accepts_detection_row_views() -> hrx::Result<()> {
    use hrx::image::{ImageOps, fit_similarity_2d, invert_affine};
    let context = ModelContext::new(Default::default())?;
    let ops = ImageOps::new(&context, 2)?;
    let template = [[1., 2.], [5., 2.], [3., 4.], [1., 6.], [5., 6.]];
    let faces = [
        template,
        template.map(|[x, y]| [2. * y + 37., -2. * x + 91.]),
        template.map(|[x, y]| [x * 0.001 + 0.003, y * 0.001 - 0.007]),
        [[0., 0.]; 5],
        [[f32::NAN, 0.]; 5],
        [[f32::MAX, f32::MAX]; 5],
    ];
    // Layout of SCRFD landmark fields: five ignored fields, ten coordinates,
    // one ranking field per detection row. The view excludes the leading five.
    let mut rows = vec![0f32; faces.len() * 16];
    for (i, face) in faces.iter().enumerate() {
        for (j, value) in face.iter().flatten().enumerate() {
            rows[i * 16 + 5 + j] = *value;
        }
    }
    let bytes: Vec<_> = rows.iter().flat_map(|v| v.to_le_bytes()).collect();
    let source = context.upload(TensorDesc::new(DType::F32, vec![faces.len(), 16])?, &bytes)?;
    let landmarks = source.view(
        20,
        TensorDesc::strided(DType::F32, vec![faces.len(), 5, 2], vec![16, 2, 1])?,
    )?;
    let before = context.runtime().statistics().downloaded_bytes;
    let fitted = ops.similarity_2d(&landmarks, &template)?;
    fitted.completion().wait()?;
    assert_eq!(context.runtime().statistics().downloaded_bytes, before);
    let result = fitted.download()?.wait()?;
    for (i, face) in faces.iter().enumerate() {
        let status = i32::from_le_bytes(result[2][i * 4..][..4].try_into().unwrap());
        if i >= 3 {
            assert_eq!(status, 1);
            assert!(f32::from_le_bytes(result[0][i * 24..][..4].try_into().unwrap()).is_nan());
            continue;
        }
        assert_eq!(status, 0);
        let expected = fit_similarity_2d(face, &template)?;
        for (bytes, expected) in [&result[0], &result[1]]
            .into_iter()
            .zip([expected, invert_affine(expected)?])
        {
            for (actual, expected) in bytes[i * 24..][..24]
                .as_chunks::<4>()
                .0
                .iter()
                .zip(expected.iter().flatten())
            {
                let actual = f32::from_le_bytes(*actual);
                assert!(
                    (actual - expected).abs() <= 1e-5 * expected.abs().max(1.),
                    "face {i}: {actual} vs {expected}"
                );
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires GPU and Loom compiler"]
fn resize_padding_batch_copy_and_concatenation_preserve_owners() -> hrx::Result<()> {
    use hrx::{
        image::{ImageOps, RgbResize},
        tensor::Layout,
    };
    let context = ModelContext::new(Default::default())?;
    let ops = ImageOps::new(&context, 2)?;
    let desc = TensorDesc::new(DType::U8, vec![2, 2, 3, 3])?.with_layout(Layout::Nhwc)?;
    let resize = RgbResize {
        height: 5,
        width: 7,
        region: [1, 1, 4, 3],
    };
    let plan = ops.prepare_resize_rgb(&desc, resize)?;
    let rgb = [vec![37; 18], vec![191; 18]].concat();
    let first = plan.submit_host(&[&rgb])?;
    let second = plan.submit_host(&[&rgb])?;
    let third = plan.submit_host(&[&rgb])?;
    assert!(matches!(plan.try_acquire(), Err(Error::Busy(_))));
    let output = first.outputs()[0].clone();
    drop(first);
    let one = TensorDesc::new(DType::U8, vec![1, 5, 7, 3])?.with_layout(Layout::Nhwc)?;
    let empty = context
        .allocate(TensorDesc::new(DType::U8, vec![0, 5, 7, 3])?.with_layout(Layout::Nhwc)?)?;
    let reversed = context.concatenate(&[
        output.view(one.bytes(), one.clone())?,
        empty,
        output.view(0, one.clone())?,
    ])?;
    drop(output);
    let mut copied = context.allocate(reversed.desc().clone())?;
    context.copy_into(&mut copied, 0, &reversed)?;
    assert!(context.copy_into(&mut copied, 1, &reversed).is_err());
    let actual = context.download(&copied)?.wait()?;
    let mut expected = vec![0; 2 * 5 * 7 * 3];
    for (batch, value) in [191, 37].into_iter().enumerate() {
        for y in 1..4 {
            for x in 1..5 {
                expected[(batch * 5 * 7 + y * 7 + x) * 3..][..3].fill(value);
            }
        }
    }
    assert_eq!(actual, expected);
    drop(plan.acquire_blocking()?);
    drop((second, third));
    assert!(
        ops.prepare_resize_rgb(
            &desc,
            RgbResize {
                region: [usize::MAX, 0, 1, 1],
                ..resize
            }
        )
        .is_err()
    );
    assert!(context.concatenate(&[]).is_err());
    let foreign = ModelContext::new(Default::default())?.allocate(one)?;
    assert!(context.concatenate(&[foreign]).is_err());
    Ok(())
}

#[test]
#[ignore = "requires GPU and Loom compiler"]
fn affine_rgb_preserves_sampling_and_replays_changed_matrices() -> hrx::Result<()> {
    use hrx::{image::ImageOps, tensor::Layout};
    let context = ModelContext::new(RuntimeOptions::default())?;
    let ops = ImageOps::new(&context, 2)?;
    let desc = TensorDesc::new(DType::U8, vec![1, 4, 6, 3])?.with_layout(Layout::Nhwc)?;
    let rgb: Vec<u8> = (0..72).map(|i| ((i * 37) % 256) as u8).collect();
    let maps = TensorDesc::new(DType::F32, vec![3, 2, 3])?;
    let plan = ops.prepare_affine_rgb(&desc, 3, 4, 6, hrx::image::RgbSampling::BlackTiesEven)?;
    for tx in [0., 0.5, -0.25, f32::INFINITY, 1e30] {
        let matrices: [[[f32; 3]; 2]; 3] = [
            [[1., 0., tx], [0., 1., 0.]],
            [[1., 0., -0.5], [0., 1., 0.75]],
            [[1., 0., f32::NAN], [0., 1., 0.]],
        ];
        let bytes: Vec<u8> = matrices
            .iter()
            .flatten()
            .flatten()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let image = context.upload(desc.clone(), &rgb)?;
        let inverse = context.upload(maps.clone(), &bytes)?;
        let output = ops
            .affine_rgb(
                &image,
                &inverse,
                4,
                6,
                hrx::image::RgbSampling::BlackTiesEven,
            )?
            .download()?
            .wait()?
            .remove(0);
        let mut expected = vec![0u8; 3 * 72];
        for (crop, t) in matrices.iter().enumerate() {
            for y in 0..4 {
                for x in 0..6 {
                    let xx = t[0][0] * x as f32 + t[0][1] * y as f32 + t[0][2];
                    let yy = t[1][0] * x as f32 + t[1][1] * y as f32 + t[1][2];
                    if !(xx > -1. && yy > -1. && xx < 6. && yy < 4.) {
                        continue;
                    }
                    let fx = xx - xx.floor();
                    let fy = yy - yy.floor();
                    for c in 0..3 {
                        let mut value = 0f32;
                        for dy in 0..2 {
                            for dx in 0..2 {
                                let px = xx.floor() as i32 + dx;
                                let py = yy.floor() as i32 + dy;
                                if (0..6).contains(&px) && (0..4).contains(&py) {
                                    value += rgb[(py as usize * 6 + px as usize) * 3 + c] as f32
                                        * if dx == 0 { 1. - fx } else { fx }
                                        * if dy == 0 { 1. - fy } else { fy };
                                }
                            }
                        }
                        expected[crop * 72 + (y * 6 + x) * 3 + c] =
                            value.round_ties_even().clamp(0., 255.) as u8;
                    }
                }
            }
        }
        assert_eq!(output, expected);
        assert!(std::sync::Arc::ptr_eq(
            &plan,
            &ops.prepare_affine_rgb(&desc, 3, 4, 6, hrx::image::RgbSampling::BlackTiesEven)?
        ));
    }
    assert!(
        ops.prepare_affine_rgb(&desc, 0, 4, 6, hrx::image::RgbSampling::BlackTiesEven)
            .is_err()
    );
    assert!(
        ops.prepare_affine_rgb(&desc, 1, 0, 6, hrx::image::RgbSampling::BlackTiesEven)
            .is_err()
    );
    assert!(
        ops.prepare_affine_rgb(&maps, 1, 4, 6, hrx::image::RgbSampling::BlackTiesEven)
            .is_err()
    );
    let image = context.upload(desc.clone(), &rgb)?;
    let inverse = context.upload(maps.clone(), &vec![0; maps.bytes()])?;
    let a = ops.affine_rgb(
        &image,
        &inverse,
        4,
        6,
        hrx::image::RgbSampling::BlackTiesEven,
    )?;
    let b = ops.affine_rgb(
        &image,
        &inverse,
        4,
        6,
        hrx::image::RgbSampling::BlackTiesEven,
    )?;
    let c = ops.affine_rgb(
        &image,
        &inverse,
        4,
        6,
        hrx::image::RgbSampling::BlackTiesEven,
    )?;
    assert!(matches!(
        ops.affine_rgb(
            &image,
            &inverse,
            4,
            6,
            hrx::image::RgbSampling::BlackTiesEven
        ),
        Err(Error::Busy(_))
    ));
    let retained = a.outputs()[0].clone();
    a.completion().wait()?;
    drop(a);
    assert!(matches!(
        ops.affine_rgb(
            &image,
            &inverse,
            4,
            6,
            hrx::image::RgbSampling::BlackTiesEven
        ),
        Err(Error::Busy(_))
    ));
    drop(retained);
    ops.affine_rgb(
        &image,
        &inverse,
        4,
        6,
        hrx::image::RgbSampling::BlackTiesEven,
    )?
    .completion()
    .wait()?;
    drop((b, c));
    let foreign = ModelContext::new(RuntimeOptions::default())?;
    assert!(
        ops.affine_rgb(
            &foreign.allocate(desc)?,
            &inverse,
            4,
            6,
            hrx::image::RgbSampling::BlackTiesEven
        )
        .is_err()
    );
    let wrong = context.allocate(TensorDesc::new(DType::U8, vec![3, 2, 3])?)?;
    assert!(
        ops.affine_rgb(&image, &wrong, 4, 6, hrx::image::RgbSampling::BlackTiesEven)
            .is_err()
    );
    Ok(())
}

#[test]
#[ignore = "requires the GPU runtime"]
fn inference_chaining_retains_slots_and_propagates_values() -> hrx::Result<()> {
    let context = ModelContext::new(RuntimeOptions::default())?;
    let desc = TensorDesc::new(DType::U8, vec![128])?;
    let model = PreparedModel::prepare(&context, 1, |context| {
        let input = vec![context.allocate(desc.clone())?];
        let output = vec![context.allocate(desc.clone())?];
        let mut graph = context.runtime().graph();
        graph.copy(output[0].binding().unwrap(), input[0].binding().unwrap())?;
        Ok(InferenceGraph {
            inputs: input,
            outputs: output,
            graph: graph.prepare()?,
        })
    })?;
    let bytes = (0..128).map(|n| n as u8).collect::<Vec<_>>();
    let input = context.upload(desc.clone(), &bytes)?;
    let first = model.submit(&[input])?;
    let output = first.outputs()[0].clone();
    first.completion().wait()?;
    drop(first);
    assert!(matches!(model.try_acquire(), Err(Error::Busy(_))));

    // A downstream graph must retain the slot even after all tensor handles drop.
    let destination = context.allocate(desc)?;
    let mut downstream = context.runtime().graph();
    downstream.copy(destination.binding().unwrap(), output.binding().unwrap())?;
    let downstream = downstream.prepare()?;
    drop(output);
    assert!(matches!(model.try_acquire(), Err(Error::Busy(_))));
    downstream.submit()?.wait()?;
    assert_eq!(context.download(&destination)?.wait()?, bytes);
    drop(downstream);
    drop(model.try_acquire()?);

    // Early abandonment still drains all submitted reads/writes before reuse.
    let next = context.upload(TensorDesc::new(DType::U8, vec![128])?, &[9; 128])?;
    let submitted = model.submit(&[next])?;
    drop(submitted);
    drop(model.acquire_blocking()?);
    // Host readback storage is lazy; warm it before checking replay resources.
    model.submit_host(&[&[0; 128]])?.download()?.wait()?;
    let before = context.runtime().statistics();
    for value in 0..8 {
        let input = context.upload(TensorDesc::new(DType::U8, vec![128])?, &[value; 128])?;
        let result = model.submit(&[input])?.download()?.wait()?;
        assert_eq!(result, vec![vec![value; 128]]);
    }
    let after = context.runtime().statistics();
    assert_eq!(after.native_graphs_prepared, before.native_graphs_prepared);
    assert_eq!(after.copy_streams_created, before.copy_streams_created);
    assert_eq!(after.copy_streams_created, 3);
    Ok(())
}

#[test]
#[ignore = "requires the GPU runtime"]
fn slots_are_independent_and_reject_foreign_tensors() -> hrx::Result<()> {
    let context = ModelContext::new(RuntimeOptions::default())?;
    let foreign = ModelContext::new(RuntimeOptions::default())?;
    let desc = TensorDesc::new(DType::U8, vec![32])?;
    let model = PreparedModel::prepare(&context, 3, |context| {
        let input = vec![context.allocate(desc.clone())?];
        let output = vec![context.allocate(desc.clone())?];
        let mut graph = context.runtime().graph();
        graph.copy(output[0].binding().unwrap(), input[0].binding().unwrap())?;
        Ok(InferenceGraph {
            inputs: input,
            outputs: output,
            graph: graph.prepare()?,
        })
    })?;
    assert!(model.submit(&[foreign.allocate(desc.clone())?]).is_err());
    let mut pending = Vec::new();
    for value in 1..=3u8 {
        let input = context.upload(desc.clone(), &[value; 32])?;
        pending.push(model.submit(&[input])?);
    }
    assert!(matches!(model.try_acquire(), Err(Error::Busy(_))));
    for (index, result) in pending.into_iter().enumerate() {
        let output = result.wait()?;
        assert_eq!(
            context.download(&output[0])?.wait()?,
            vec![index as u8 + 1; 32]
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires the GPU runtime"]
fn host_staging_reuses_allocations_and_wakes_capacity_waiters() -> hrx::Result<()> {
    use std::{
        future::Future,
        pin::pin,
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
        task::{Context, Wake, Waker},
    };
    struct Counter(AtomicUsize);
    impl Wake for Counter {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }
    let context = ModelContext::new(RuntimeOptions::default())?;
    let desc = TensorDesc::new(DType::U8, vec![32])?;
    let model = PreparedModel::prepare(&context, 1, |context| {
        let input = vec![context.allocate(desc.clone())?];
        let output = vec![context.allocate(desc.clone())?];
        let mut graph = context.runtime().graph();
        graph.copy(output[0].binding().unwrap(), input[0].binding().unwrap())?;
        Ok(InferenceGraph {
            inputs: input,
            outputs: output,
            graph: graph.prepare()?,
        })
    })?;
    assert_eq!(context.runtime().statistics().allocations, 2);
    model.submit_host(&[&[0; 32]])?.download()?.wait()?;
    let allocations = context.runtime().statistics().allocations;
    assert_eq!(allocations, 4);
    context.runtime().start_trace(5)?;
    for value in 0..10 {
        let result = model.submit_host(&[&[value; 32]])?;
        let counter = Arc::new(Counter(AtomicUsize::new(0)));
        let waker = Waker::from(counter.clone());
        let mut cx = Context::from_waker(&waker);
        let mut acquire = pin!(model.acquire());
        assert!(acquire.as_mut().poll(&mut cx).is_pending());
        assert_eq!(result.download()?.wait()?, vec![vec![value; 32]]);
        assert!(counter.0.load(Ordering::Relaxed) > 0);
        assert!(acquire.as_mut().poll(&mut cx).is_ready());
    }
    assert_eq!(context.runtime().statistics().allocations, allocations);
    let trace = context.runtime().finish_trace().unwrap();
    assert_eq!(trace.events.len(), 5);
    assert!(trace.dropped > 0);
    let json: serde_json::Value = serde_json::from_str(&trace.to_json()?).unwrap();
    assert_eq!(json["traceEvents"].as_array().unwrap().len(), 5);
    assert!(
        json["measurement"]
            .as_str()
            .unwrap()
            .contains("not device timestamps")
    );
    Ok(())
}

#[test]
#[ignore = "requires GPU and Loom compiler"]
fn image_patchification_preserves_values_and_channel_order() -> hrx::Result<()> {
    use hrx::{image::ImageOps, tensor::Layout};
    let context = ModelContext::new(RuntimeOptions::default())?;
    let ops = ImageOps::new(&context, 2)?;
    for (batch, height, width, patch) in [(1, 4, 6, 2), (2, 16, 24, 8)] {
        let desc = TensorDesc::new(DType::F32, vec![batch, 3, height, width])?
            .with_layout(Layout::Nchw)?;
        let bytes = (0..desc.elements())
            .flat_map(|i| (i as f32).to_le_bytes())
            .collect::<Vec<_>>();
        let input = context.upload(desc.clone(), &bytes)?;
        let output = ops.patchify(&input, patch)?.download()?.wait()?.remove(0);
        let mut expected = Vec::new();
        for b in 0..batch {
            for y in 0..height / patch {
                for x in 0..width / patch {
                    for c in 0..3 {
                        for dy in 0..patch {
                            for dx in 0..patch {
                                let i = ((b * 3 + c) * height + y * patch + dy) * width
                                    + x * patch
                                    + dx;
                                expected.extend_from_slice(&(i as f32).to_le_bytes());
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(output, expected);
        assert!(ops.prepare_patchify(&desc, 0).is_err());
    }
    Ok(())
}

#[test]
#[ignore = "requires GPU and Loom compiler"]
fn normalization_matches_rgb_channel_and_arithmetic_contract() -> hrx::Result<()> {
    use hrx::{image::ImageOps, tensor::Layout};
    let context = ModelContext::new(Default::default())?;
    let ops = ImageOps::new(&context, 2)?;
    let (mean, std) = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]);
    let input = TensorDesc::new(DType::U8, vec![2, 4, 5, 3])?.with_layout(Layout::Nhwc)?;
    let rgb = (0..120).map(|i| (i * 47) as u8).collect::<Vec<_>>();
    let plan = ops.prepare_normalize_rgb(&input, mean, std)?;
    let output = plan.submit_host(&[&rgb])?.download()?.wait()?.remove(0);
    for b in 0..2 {
        for c in 0..3 {
            for p in 0..20 {
                let expected = (rgb[b * 60 + p * 3 + c] as f32 / 255. - mean[c]) / std[c];
                let at = (b * 60 + c * 20 + p) * 4;
                let actual = f32::from_le_bytes(output[at..at + 4].try_into().unwrap());
                assert!((actual - expected).abs() < 5e-7, "{actual} vs {expected}");
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires the GPU runtime"]
fn cancelled_native_work_drains_before_slot_reuse() -> hrx::Result<()> {
    use hrx::{Access, execution::GpuAccess};
    let context = ModelContext::new(Default::default())?;
    let desc = TensorDesc::new(DType::U8, vec![32])?;
    let (started, seen) = std::sync::mpsc::channel();
    let (release, gate) = std::sync::mpsc::channel();
    let mut resources = Some((started, gate, hrx::Device::open(0)?.stream()?));
    let model = PreparedModel::prepare(&context, 1, |context| {
        let outputs = vec![context.allocate(desc.clone())?];
        let (started, gate, mut stream) = resources.take().unwrap();
        let mut first = true;
        let mut graph = context.runtime().graph();
        // This callback owns its stream, only writes the declared output and
        // drains it before returning, including the cancellation test gate.
        unsafe {
            graph.gpu_scoped(
                &[GpuAccess {
                    view: outputs[0].binding().unwrap(),
                    access: Access::Write,
                }],
                move |views| {
                    if first {
                        started
                            .send(())
                            .map_err(|e| Error::Message(e.to_string()))?;
                        gate.recv().map_err(|e| Error::Message(e.to_string()))?;
                        first = false;
                    }
                    stream.fill(views[0], 7)?;
                    stream.synchronize()
                },
            )?;
        }
        Ok(InferenceGraph {
            inputs: vec![],
            outputs,
            graph: graph.prepare()?,
        })
    })?;
    let result = model.submit_host(&[])?;
    seen.recv_timeout(std::time::Duration::from_secs(5))
        .unwrap();
    let completion = result.completion().clone();
    completion.cancel();
    drop(result);
    assert!(matches!(model.try_acquire(), Err(Error::Busy(_))));
    release.send(()).unwrap();
    assert!(matches!(completion.wait(), Err(Error::Cancelled)));
    drop(completion);
    let result = model.acquire_blocking()?.submit_host(&[])?;
    assert_eq!(result.download()?.wait()?, vec![vec![7; 32]]);
    Ok(())
}
#[test]
#[ignore = "requires a provisioned GPU runtime"]
fn shared_budget_charges_storage_aliases_and_transfer_staging() {
    use hrx::inference::ModelContext;
    use hrx::residency::ResidencyManager;
    use hrx::tensor::{DType, TensorDesc};
    let manager = ResidencyManager::new(8192).unwrap();
    let context = ModelContext::new(hrx::execution::RuntimeOptions {
        memory_budget: Some(manager.budget()),
        ..Default::default()
    })
    .unwrap();
    let desc = TensorDesc::new(DType::U8, vec![4096]).unwrap();
    let tensor = context.allocate(desc.clone()).unwrap();
    let alias = tensor.clone();
    assert_eq!(manager.statistics().reserved_bytes, 4096);
    // Destination fits, but its upload staging does not. The failed upload
    // must release its destination before returning to the caller.
    assert!(context.upload(desc.clone(), &[3; 4096]).is_err());
    assert_eq!(manager.statistics().reserved_bytes, 4096);
    drop(tensor);
    assert_eq!(manager.statistics().reserved_bytes, 4096);
    drop(alias);
    assert_eq!(manager.statistics().reserved_bytes, 0);
    let tensor = context.upload(desc, &[7; 4096]).unwrap();
    tensor.completion().wait().unwrap();
    let bytes = context.download(&tensor).unwrap().wait().unwrap();
    assert_eq!(bytes, vec![7; 4096]);
    drop(tensor);
    assert_eq!(manager.statistics().reserved_bytes, 0);
}
