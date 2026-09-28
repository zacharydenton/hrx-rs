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

#[test]
fn wait_diffs_match_identities_and_preserve_missing_counts() {
    let root = tempfile::tempdir().unwrap();
    let row = |reason: &str, count: u64| {
        json!({"function":"worker", "counter":"lds",
        "reason":reason, "summary":{"action_count":count, "drained_count":count}})
    };
    let mut baseline = json!({"compiler":"fixture-compiler", "target":"gfx1151", "processor_mode":"default",
        "document":{"kind":"loom.compile_report", "schema_version":0, "backend":"amdgpu-hsaco",
        "target_key":"gfx1151", "mode":"details", "entries":{"count":1,"rows":[{"function":"worker"}]},
        "wait_reason_summary_rows":{"count":2,"rows":[row("ssa_use", u64::MAX), row("storage_reuse", 4)]}}});
    let before = save(root.path(), "before.json", &baseline);
    let mut candidate = baseline.clone();
    candidate["document"]["wait_reason_summary_rows"] = json!({"count":2,
        "rows":[row("new_reason", 3), row("ssa_use", 0)]});
    let after = save(root.path(), "after.json", &candidate);
    let output = run(&["report", "diff", &before, &after]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    // Inspect large signed deltas through the typed API: JSON consumers may use floats.
    let a: hrx::loom::CompileReport = serde_json::from_value(baseline.clone()).unwrap();
    let b: hrx::loom::CompileReport = serde_json::from_value(candidate.clone()).unwrap();
    let changes = a.wait_reason_changes(&b).unwrap().unwrap();
    assert_eq!(changes.len(), 3);
    assert_eq!(changes[0].reason, "new_reason");
    assert!(changes[0].before.is_none());
    assert_eq!(changes[0].delta["action_count"], None);
    assert_eq!(changes[1].reason, "ssa_use");
    assert_eq!(
        changes[1].delta["drained_count"],
        Some(-i128::from(u64::MAX))
    );
    assert_eq!(changes[1].delta["max_outstanding_before"], None);
    assert_eq!(changes[2].reason, "storage_reuse");
    assert!(changes[2].after.is_none());
    assert_eq!(changes[2].delta["action_count"], None);

    baseline["document"]
        .as_object_mut()
        .unwrap()
        .remove("wait_reason_summary_rows");
    let unavailable: hrx::loom::CompileReport = serde_json::from_value(baseline).unwrap();
    assert!(unavailable.wait_reason_changes(&b).unwrap().is_none());
    candidate["document"]["wait_reason_summary_rows"] = json!({"count":0});
    let empty: hrx::loom::CompileReport = serde_json::from_value(candidate).unwrap();
    assert!(
        empty
            .wait_reason_changes(&empty)
            .unwrap()
            .unwrap()
            .is_empty()
    );
}

#[test]
fn report_cli_rejects_incomplete_collections_and_ambiguous_wait_identities() {
    let root = tempfile::tempdir().unwrap();
    let row = json!({"function":"worker","counter":"lds","reason":"ssa_use","summary":{"action_count":1}});
    let evidence = json!({"compiler":"fixture-compiler", "target":"gfx1151", "processor_mode":"default",
        "document":{"kind":"loom.compile_report", "schema_version":0, "backend":"amdgpu-hsaco",
        "target_key":"gfx1151", "mode":"details", "entries":{"count":1,"rows":[{"function":"worker"}]},
        "wait_reason_summary_rows":{"count":1,"rows":[row.clone()]}}});
    for (section, value) in [
        ("entries", json!({"count":1})),
        ("entries", json!({"count":1,"rows":[null]})),
        ("entries", json!({"count":2,"rows":[{"function":"worker"}]})),
        (
            "wait_reason_summary_rows",
            json!({"count":2,"rows":[row.clone()]}),
        ),
        ("wait_reason_summary_rows", json!({"count":1,"rows":null})),
    ] {
        let mut malformed = evidence.clone();
        malformed["document"][section] = value;
        let path = save(root.path(), "malformed.json", &malformed);
        assert!(!run(&["report", "show", &path]).status.success());
    }
    for rows in [json!([row.clone(), row]), json!([{"summary":{}}])] {
        let mut ambiguous = evidence.clone();
        ambiguous["document"]["wait_reason_summary_rows"] =
            json!({"count":rows.as_array().unwrap().len(),"rows":rows});
        let path = save(root.path(), "ambiguous.json", &ambiguous);
        assert!(!run(&["report", "diff", &path, &path]).status.success());
    }
}
