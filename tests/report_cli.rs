//! Saved compiler evidence must keep its identity and unknown-resource semantics.
use serde_json::{Value, json};
use std::{
    path::Path,
    process::{Command, Output},
};
fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_hrx"))
        .args(args)
        .env("HRX_OFFLINE", "1")
        .output()
        .unwrap()
}
fn save(root: &Path, name: &str, value: &Value) -> String {
    let path = root.join(name);
    std::fs::write(&path, serde_json::to_vec(value).unwrap()).unwrap();
    path.to_str().unwrap().to_owned()
}
#[test]
fn report_cli_preserves_unknowns_and_rejects_incompatible_evidence() {
    let root = tempfile::tempdir().unwrap();
    let evidence = json!({"compiler":"fixture-compiler", "target":"gfx1151", "processor_mode":"default",
        "document":{"kind":"loom.compile_report", "schema_version":0, "backend":"amdgpu-hsaco",
        "target_key":"gfx1151", "mode":"summary", "entries":{"rows":[{"function":"worker","allocation_spill_count":0}]}}});
    let before = save(root.path(), "before.json", &evidence);
    let show = run(&["report", "show", &before]);
    assert!(show.status.success());
    let shown: Value = serde_json::from_slice(&show.stdout).unwrap();
    assert_eq!(shown["entries"][0]["spills"], 0);
    assert!(shown["entries"][0]["vector_registers"].is_null());
    let same = run(&["report", "diff", &before, &before]);
    assert!(same.status.success());
    for (field, replacement) in [
        ("compiler", json!("another")),
        ("target", json!("gfx1100")),
        ("processor_mode", json!("compute_unit")),
    ] {
        let mut other = evidence.clone();
        other[field] = replacement;
        let after = save(root.path(), "other.json", &other);
        assert!(!run(&["report", "diff", &before, &after]).status.success());
    }
    let mut future = evidence;
    future["document"]["schema_version"] = 1.into();
    let path = save(root.path(), "future.json", &future);
    assert!(!run(&["report", "show", &path]).status.success());
    std::fs::write(&path, b"{broken").unwrap();
    assert!(!run(&["report", "show", &path]).status.success());
}
