//! Guidance must cite qualified evidence and preserve missing joint constraints.
use hrx::loom::CompileReport;
use serde_json::{Value, json};

fn report(document: Value) -> CompileReport {
    serde_json::from_value(json!({"compiler":"fixture", "target":"gfx1151",
        "processor_mode":"default", "document":document}))
    .unwrap()
}

fn evidence() -> Value {
    json!({"kind":"loom.compile_report", "schema_version":0, "target_family":"amdgpu",
        "status":{"code":0}, "entries":{"count":1,"rows":[{"function":"attention",
            "target_resources":{"residency":{"current_tier":5,"next_better_tier":6,"limiting_resource_count":2}}}]},
        "residency_constraints":{"count":2,"rows":[
            {"function":"attention","name":"amdgpu.lds","limiting":true,"units":21760,
             "reduction_units_to_next_better_tier":256,"unit":"bytes","allocation_scope":"workgroup"},
            {"function":"attention","name":"amdgpu.vgpr","limiting":true,"units":256,
             "reduction_units_to_next_better_tier":16,"unit":"registers","allocation_scope":"subgroup"}]}})
}

#[test]
fn residency_requires_complete_joint_thresholds_and_cites_original_evidence() {
    let document = evidence();
    let report = report(document.clone());
    let guidance = report.guidance().unwrap();
    assert!(guidance.residency_available);
    assert!(!guidance.bank_service_available);
    assert_eq!(guidance.suggestions.len(), 1);
    let suggestion = &guidance.suggestions[0];
    assert!(
        suggestion
            .action
            .contains("amdgpu.lds by 256 bytes/workgroup to at most 21504")
    );
    assert!(
        suggestion
            .action
            .contains("amdgpu.vgpr by 16 registers/subgroup to at most 240")
    );
    for (path, value) in &suggestion.evidence {
        assert_eq!(report.json().pointer(path), Some(value));
    }
    for field in ["units", "reduction_units_to_next_better_tier"] {
        let mut incomplete = document.clone();
        incomplete["residency_constraints"]["rows"][1]
            .as_object_mut()
            .unwrap()
            .remove(field);
        assert!(
            self::report(incomplete)
                .guidance()
                .unwrap()
                .suggestions
                .is_empty()
        );
    }
    let mut invalid = document;
    invalid["residency_constraints"]["rows"][1]["reduction_units_to_next_better_tier"] = 257.into();
    assert!(self::report(invalid).guidance().is_err());
}

#[test]
fn absent_or_failed_analysis_is_not_actionable() {
    let mut document = evidence();
    document
        .as_object_mut()
        .unwrap()
        .remove("residency_constraints");
    let guidance = report(document).guidance().unwrap();
    assert!(!guidance.residency_available);
    assert!(guidance.suggestions.is_empty());
    let mut failed = evidence();
    failed["status"]["code"] = 1.into();
    assert!(report(failed).guidance().unwrap().suggestions.is_empty());
    let mut foreign = evidence();
    foreign["target_family"] = "xdna".into();
    assert!(report(foreign).guidance().unwrap().suggestions.is_empty());
}

#[test]
fn bank_advice_requires_qualified_exact_service_evidence() {
    let mut document = evidence();
    document["source_low"] = json!({"memory":{"bank_service_group_count":1,"bank_service_groups":[{
        "function":"attention","source_root":"scratch","model":{"evidence":"public-vendor-documentation"},
        "summary":{"exact_packet_count":2,"unknown_packet_count":3,"unmodeled_packet_count":4,
            "structural":{"conflicted_packet_count":1,"extra_round_count":8,"required_round_count":24,"uncontended_round_count":16}}
    }]}});
    let guidance = report(document.clone()).guidance().unwrap();
    assert!(guidance.bank_service_available);
    assert_eq!(guidance.suggestions.len(), 2);
    let bank = &guidance.suggestions[1];
    assert_eq!(bank.kind, "amdgpu.lds_bank_service");
    assert!(bank.action.contains("8 extra bank-service rounds"));
    for (path, value) in &bank.evidence {
        assert_eq!(document.pointer(path), Some(value));
    }
    for model in [
        Value::Null,
        json!("vendor-software-model-unvalidated"),
        json!("future-model"),
    ] {
        let mut unqualified = document.clone();
        unqualified["source_low"]["memory"]["bank_service_groups"][0]["model"]["evidence"] = model;
        assert_eq!(report(unqualified).guidance().unwrap().suggestions.len(), 1);
    }
    let mut inconsistent = document.clone();
    inconsistent["source_low"]["memory"]["bank_service_groups"][0]["summary"]["structural"]["required_round_count"] =
        25.into();
    assert!(report(inconsistent).guidance().is_err());
    document["source_low"]["memory"]["bank_service_group_count"] = 2.into();
    assert!(report(document).guidance().is_err());
}
