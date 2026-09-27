//! Regenerates `src/artifacts/onnx_proto.rs` from `src/artifacts/onnx.proto`
//! with the pure-Rust protobuf parser, so no `protoc` is needed.
use std::path::Path;

fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let artifacts = root.join("src/artifacts");
    let out = scratch_dir(&root);
    protobuf_codegen::Codegen::new()
        .pure()
        .include(&artifacts)
        .input(artifacts.join("onnx.proto"))
        .out_dir(&out)
        .run_from_script();
    std::fs::rename(out.join("onnx.rs"), artifacts.join("onnx_proto.rs"))
        .expect("moving the generated module into place");
    std::fs::remove_dir_all(&out).expect("removing the generator's scratch directory");
}

/// A scratch directory under `target/`, which the generator writes into.
fn scratch_dir(root: &Path) -> std::path::PathBuf {
    let out = root.join("target/onnx-codegen");
    let _ = std::fs::remove_dir_all(&out);
    std::fs::create_dir_all(&out).expect("creating the generator's scratch directory");
    out
}
