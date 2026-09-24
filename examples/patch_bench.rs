//! Paired compiler qualification on resident attention, GEMM and convolution kernels.
//! Usage: patch_bench BASELINE_LIB CANDIDATE_LIB [pairs=30] [default|cu|wgp]
use hrx::{
    Buffer, Constants, GraphExec, Result, Stream,
    loom::{Compiler, CompilerOptions, ProcessorMode, ReportMode, Specialization},
};
use std::{path::Path, time::Instant};

fn half(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 255) as i32 - 127 + 15;
    if exponent <= 0 {
        return sign;
    }
    assert!(exponent < 31, "fixture value overflows fp16");
    let mantissa = bits & 0x7fffff;
    let rounded = mantissa + 0xfff + ((mantissa >> 13) & 1);
    sign | (((exponent as u32) << 10) + (rounded >> 13)) as u16
}
fn float(value: u16) -> f64 {
    let sign = if value & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = (value >> 10) & 31;
    let mantissa = f64::from(value & 1023);
    match exponent {
        0 => sign * mantissa * 2f64.powi(-24),
        31 => f64::NAN,
        _ => sign * (1.0 + mantissa / 1024.0) * 2f64.powi(i32::from(exponent) - 15),
    }
}
fn bf16(value: f32) -> f32 {
    let bits = value.to_bits();
    f32::from_bits((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000)
}
fn halves(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values
        .into_iter()
        .flat_map(|value| half(value).to_le_bytes())
        .collect()
}
fn floats(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values.into_iter().flat_map(f32::to_le_bytes).collect()
}
struct Case {
    name: String,
    source: &'static str,
    spec: Specialization,
    grid: [u32; 3],
    block: [u32; 3],
    indices: Vec<u32>,
    data: Vec<Vec<u8>>,
    output: usize,
    expected: Vec<f64>,
    tolerance: f64,
}
fn attention(tokens: usize, repack: bool) -> Case {
    let capacity = tokens + 32;
    let symbol = if repack {
        "krea2_attention_query32"
    } else {
        "krea2_attention_gqa_lds_f16_wmma"
    };
    let prefix = if repack {
        "krea2.attention_query32"
    } else {
        "krea2.attention_gqa_lds_f16_wmma"
    };
    let mut spec = Specialization::new(symbol).with_report(ReportMode::Summary);
    for (key, value) in [
        ("q_stride", 512),
        ("kv_stride", 128),
        ("out_stride", 512),
        ("tokens", tokens),
        ("token_capacity", capacity),
    ] {
        spec.set_config(format!("{prefix}.{key}"), value.to_string());
    }
    spec.set_config(format!("{prefix}.scale"), "0.08838834764831845");
    let query = |row: usize, head: usize| ((row + head * 3) % 9) as f32 * 0.0625 - 0.25;
    let key = |row: usize| (row % 7) as f32 * 0.0625 - 0.1875;
    let value = |row: usize, channel: usize| {
        (row % 7) as f32 * 0.125 - 0.375 + (channel % 8) as f32 * 0.0625
    };
    let q = halves((0..capacity * 512).map(|i| {
        if i / 512 < tokens {
            query(i / 512, (i % 512) / 128)
        } else {
            0.0
        }
    }));
    let k = halves((0..capacity * 128).map(|i| if i / 128 < tokens { key(i / 128) } else { 0.0 }));
    let v = halves((0..capacity * 128).map(|i| {
        let (row, channel) = if repack {
            (i % capacity, i / capacity)
        } else {
            (i / 128, i % 128)
        };
        if row < tokens {
            value(row, channel)
        } else {
            0.0
        }
    }));
    // Independent f64 softmax oracle. Inputs are constant across the 128 Q/K
    // channels, so their exact dot product is 128 times the scalar product.
    let mut expected = Vec::with_capacity(tokens * 512);
    for row in 0..tokens {
        for head in 0..4 {
            let mut numerator = 0.0;
            let mut denominator = 0.0;
            for token in 0..tokens {
                let score = (128.0 * f64::from(query(row, head)) * f64::from(key(token))
                    / 128f64.sqrt())
                .exp();
                numerator += score * f64::from(value(token, 0));
                denominator += score;
            }
            expected.extend(
                (0..128).map(|channel| numerator / denominator + (channel % 8) as f64 * 0.0625),
            );
        }
    }
    Case {
        name: format!("{symbol}-{tokens}"),
        source: if repack {
            include_str!("../native/qualification/attention_query32.loom")
        } else {
            include_str!("../native/qualification/attention_gqa_lds_f16_wmma.loom")
        },
        spec,
        grid: [(tokens / if repack { 32 } else { 16 }) as u32, 1, 1],
        block: [if repack { 256 } else { 128 }, 1, 1],
        indices: vec![tokens as u32],
        data: vec![q, k, v, vec![0; tokens * 512 * 2]],
        output: 3,
        expected,
        tolerance: 0.001,
    }
}
fn gemm(large: bool, integer: bool) -> Case {
    let (m, n, k) = if large {
        (256usize, 1024usize, 1024usize)
    } else {
        (256, 512, 256)
    };
    let symbol = if integer {
        "krea2_gemm_i8_256"
    } else {
        "h3_gemm_f16_fast_256b"
    };
    let prefix = if integer {
        "krea2.gemm_i8_256"
    } else {
        "h3.gemm_f16_fast_256b"
    };
    let mut spec = Specialization::new(symbol).with_report(ReportMode::Summary);
    for (name, value) in [
        ("k_size", k),
        ("n_size", n),
        ("k_stride", k),
        ("m_group", 1),
    ] {
        spec.set_config(format!("{prefix}.{name}"), value.to_string());
    }
    let a = |row: usize, col: usize| ((row * 3 + col * 5 + row * col) % 11) as i32 - 5;
    let w = |row: usize, col: usize| ((row * 7 + col * 3 + row * col) % 13) as i32 - 6;
    let bias = |col: usize| (col % 7) as f32 * 0.03125 - 0.09375;
    // Evaluate the full dot product once per distinct row-pattern pair. Pattern
    // repetition in the authored inputs is independent of the compiled program.
    let mut products = [[0f32; 13]; 11];
    for (row, values) in products.iter_mut().enumerate() {
        for (col, product) in values.iter_mut().enumerate() {
            let sum: i32 = (0..k).map(|i| a(row, i) * w(col, i)).sum();
            *product = sum as f32 / 64.0;
        }
    }
    let expected = (0..m * n)
        .map(|i| {
            let dot = products[(i / n) % 11][(i % n) % 13];
            float(half(if integer {
                bf16(dot)
            } else {
                dot + bias(i % n)
            }))
        })
        .collect();
    let data = if integer {
        vec![
            (0..m * k).map(|i| a(i / k, i % k) as i8 as u8).collect(),
            (0..n * k).map(|i| w(i / k, i % k) as i8 as u8).collect(),
            floats(vec![0.125; n]),
            floats(vec![0.125; m]),
            vec![0; m * n * 2],
        ]
    } else {
        vec![
            halves((0..m * k).map(|i| a(i / k, i % k) as f32 * 0.125)),
            halves((0..n * k).map(|i| w(i / k, i % k) as f32 * 0.125)),
            vec![0; m * n * 2],
            floats((0..n).map(bias)),
        ]
    };
    Case {
        name: format!("{symbol}-{m}x{n}x{k}"),
        source: if integer {
            include_str!("../native/qualification/gemm_i8_256.loom")
        } else {
            include_str!("../native/qualification/gemm_f16_fast_256b.loom")
        },
        spec,
        grid: [
            (n / if integer { 128 } else { 256 }) as u32,
            (m / if integer { 256 } else { 128 }) as u32,
            1,
        ],
        block: [256, 1, 1],
        indices: vec![m as u32],
        data,
        output: if integer { 4 } else { 2 },
        expected,
        // Permit subnormal residuals from the floating-point matrix accumulator.
        // Integer GEMM remains exact; the absolute floor is one fp16 minimum normal.
        tolerance: if integer { 0.0 } else { 2f64.powi(-14) },
    }
}
fn conv(height: usize, cin: usize, n: usize) -> Case {
    let symbol = "arcface_conv3x3_f16_wmma_bnprelu";
    let mut spec = Specialization::new(symbol).with_report(ReportMode::Details);
    for (key, value) in [
        ("height", height),
        ("width", height),
        ("stride", 1),
        ("cin_pad", cin),
        ("cin_stride", cin),
        ("k_size", 9 * cin),
        ("n_size", n),
    ] {
        spec.set_config(
            format!("arcface.conv3x3_f16_wmma_bnprelu.{key}"),
            value.to_string(),
        );
    }
    let m = height * height;
    // Constant power-of-two operands: exact mathematical convolution, including
    // zero padding at borders. Bias is zero and PReLU slope is one.
    let expected = (0..m)
        .flat_map(|p| {
            let y = p / height;
            let x = p % height;
            let rows = if y == 0 || y + 1 == height { 2 } else { 3 };
            let cols = if x == 0 || x + 1 == height { 2 } else { 3 };
            std::iter::repeat_n((rows * cols * cin) as f64 / 64.0, n)
        })
        .collect();
    Case {
        name: format!("arcface_conv-{height}x{height}x{cin}x{n}"),
        source: include_str!("../native/qualification/arcface_conv3x3_f16_wmma_bnprelu.loom"),
        spec,
        grid: [(n / 64) as u32, m.div_ceil(64) as u32, 1],
        block: [256, 1, 1],
        indices: vec![m as u32],
        data: vec![
            halves(std::iter::repeat_n(0.125, m * cin)),
            halves(std::iter::repeat_n(0.125, n * 9 * cin)),
            floats(std::iter::repeat_n(0.0, 9 * n)),
            vec![0; m * n * 2],
            floats(std::iter::repeat_n(1.0, n)),
        ],
        output: 3,
        expected,
        tolerance: 0.0,
    }
}
struct Arm {
    graph: GraphExec,
    buffers: std::sync::Arc<Vec<Buffer>>,
    identity: serde_json::Value,
}
fn prepare(
    stream: &mut Stream,
    compiler: &Compiler,
    case: &Case,
    shared: Option<std::sync::Arc<Vec<Buffer>>>,
) -> Result<Arm> {
    let artifact = compiler.module(case.source).compile(&case.spec)?;
    let kernel = unsafe { stream.load_artifact(&artifact) }?;
    let constants = Constants::indices(&kernel, &case.indices)?;
    let buffers = if let Some(buffers) = shared {
        buffers
    } else {
        std::sync::Arc::new(
            case.data
                .iter()
                .map(|data| {
                    let buffer = stream.allocate(data.len())?;
                    stream.upload_blocking(buffer.binding(), data)?;
                    Ok(buffer)
                })
                .collect::<Result<Vec<_>>>()?,
        )
    };
    let bindings: Vec<_> = buffers.iter().map(Buffer::binding).collect();
    let mut graph = stream.graph()?;
    let mut prior = None;
    for _ in 0..32 {
        prior = Some(unsafe {
            graph.dispatch(
                prior.as_slice(),
                &kernel,
                case.grid,
                case.block,
                &constants,
                &bindings,
            )
        }?);
    }
    let identity = serde_json::json!({"compiler":artifact.compiler_identity(), "artifact":hrx::bundle::digest(artifact.bytes()),
        "report":artifact.report().map(|report| report.json())});
    Ok(Arm {
        graph: graph.finish()?,
        buffers,
        identity,
    })
}
fn check(stream: &mut Stream, arm: &Arm, case: &Case) -> Result<f64> {
    let mut bytes = vec![0; case.expected.len() * 2];
    stream.read_blocking(arm.buffers[case.output].binding(), &mut bytes)?;
    let mut maximum = 0f64;
    for (index, (value, expected)) in bytes
        .as_chunks::<2>()
        .0
        .iter()
        .zip(&case.expected)
        .enumerate()
    {
        let value = float(u16::from_le_bytes(*value));
        let error = (value - expected).abs();
        if !value.is_finite() || error > case.tolerance {
            return Err(hrx::Error::Message(format!(
                "{} output {index}: {value} vs {expected}, error {error}, limit {}",
                case.name, case.tolerance
            )));
        }
        maximum = maximum.max(error);
    }
    Ok(maximum)
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 3 || args.len() > 5 {
        return Err(hrx::Error::Message(
            "usage: patch_bench BASELINE_LIB CANDIDATE_LIB [pairs=30] [default|cu|wgp]".into(),
        ));
    }
    let pairs = args
        .get(3)
        .map(|s| s.parse::<usize>())
        .transpose()
        .map_err(|e| hrx::Error::Message(e.to_string()))?
        .unwrap_or(30);
    if pairs < 30 {
        return Err(hrx::Error::Message("at least 30 pairs are required".into()));
    }
    let mode = match args.get(4).map(String::as_str).unwrap_or("default") {
        "default" => ProcessorMode::Default,
        "cu" => ProcessorMode::ComputeUnit,
        "wgp" => ProcessorMode::WorkgroupProcessor,
        _ => {
            return Err(hrx::Error::Message(
                "mode must be default, cu or wgp".into(),
            ));
        }
    };
    let mut stream = Stream::open()?;
    let mut options = CompilerOptions::for_target(stream.target());
    options.processor_mode = mode;
    let compilers = [
        Compiler::with_options(Some(Path::new(&args[1])), options.clone())?,
        Compiler::with_options(Some(Path::new(&args[2])), options)?,
    ];
    for case in [
        attention(128, false),
        attention(512, false),
        gemm(false, false),
        gemm(true, false),
        gemm(false, true),
        gemm(true, true),
        conv(56, 64, 64),
        conv(14, 256, 256),
        conv(7, 512, 512),
    ] {
        let baseline = prepare(&mut stream, &compilers[0], &case, None)?;
        let candidate = prepare(
            &mut stream,
            &compilers[1],
            &case,
            Some(baseline.buffers.clone()),
        )?;
        let mut arms = [baseline, candidate];
        for arm in &mut arms {
            for _ in 0..5 {
                stream.launch(&mut arm.graph)?;
                stream.synchronize()?;
            }
            check(&mut stream, arm, &case)?;
        }
        let errors = [
            check(&mut stream, &arms[0], &case)?,
            check(&mut stream, &arms[1], &case)?,
        ];
        println!(
            "{}",
            serde_json::json!({"case":case.name,"mode":format!("{mode:?}"),"baseline":arms[0].identity,"candidate":arms[1].identity,"max_absolute_error":errors,"tolerance":case.tolerance,"replays":32})
        );
        for pair in 0..pairs {
            let mut ns = [0.0; 2];
            for index in if pair % 2 == 0 { [0, 1] } else { [1, 0] } {
                let start = Instant::now();
                stream.launch(&mut arms[index].graph)?;
                stream.synchronize()?;
                ns[index] = start.elapsed().as_secs_f64() * 1e9 / 32.0;
            }
            println!(
                "{}",
                serde_json::json!({"case":case.name,"pair":pair,"baseline_ns":ns[0],"candidate_ns":ns[1]})
            );
        }
        for arm in &mut arms {
            stream.launch(&mut arm.graph)?;
            stream.synchronize()?;
            check(&mut stream, arm, &case)?;
        }
    }
    Ok(())
}
