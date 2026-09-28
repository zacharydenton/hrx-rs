use super::CompileReport;
use crate::{Error, Result};
use serde::Serialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

/// Experiments suggested by retained compiler evidence, not measured speedups.
#[derive(Clone, Debug, Serialize)]
pub struct ReportGuidance {
    /// Whether source-memory bank groups were captured.
    pub bank_service_available: bool,
    /// Whether per-resource residency constraints were captured.
    pub residency_available: bool,
    /// Suggestions supported by complete, qualified evidence.
    pub suggestions: Vec<ReportSuggestion>,
}

/// One compiler-supported experiment and the exact facts supporting it.
#[derive(Clone, Debug, Serialize)]
pub struct ReportSuggestion {
    /// Stable suggestion category.
    pub kind: &'static str,
    /// Compiled entry to which the experiment applies.
    pub function: String,
    /// Suggested experiment, including model limitations.
    pub action: String,
    /// JSON pointers into [`CompileReport::json`] and their original values.
    pub evidence: BTreeMap<String, Value>,
}

impl CompileReport {
    /// Suggest residency and LDS layout experiments using producer-supplied
    /// thresholds. Unavailable evidence and unvalidated bank models yield no
    /// recommendations. All joint residency limits must be captured.
    pub fn guidance(&self) -> Result<ReportGuidance> {
        self.validate()?;
        let constraints = self.rows("residency_constraints")?;
        let groups = self
            .document
            .pointer("/source_low/memory/bank_service_groups");
        let mut guidance = ReportGuidance {
            bank_service_available: groups.is_some_and(|v| !v.is_null()),
            residency_available: constraints.is_some(),
            suggestions: Vec::new(),
        };
        if self
            .document
            .pointer("/status/code")
            .and_then(Value::as_u64)
            != Some(0)
            || self.document["target_family"] != "amdgpu"
        {
            return Ok(guidance);
        }
        let entries = self.rows("entries")?.unwrap_or_default();
        let mut names = BTreeSet::new();
        for entry in entries {
            if let Some(name) = entry["function"].as_str()
                && (!names.insert(name) || name.is_empty())
            {
                return Err(Error::Message(
                    "report has ambiguous entry identities".into(),
                ));
            }
        }
        if let Some(constraints) = constraints {
            for (index, entry) in entries.iter().enumerate() {
                let Some(function) = entry["function"].as_str() else {
                    continue;
                };
                let Some(summary) = entry.pointer("/target_resources/residency") else {
                    continue;
                };
                let (Some(current), Some(next), Some(limiters)) = (
                    summary["current_tier"].as_u64(),
                    summary["next_better_tier"].as_u64(),
                    summary["limiting_resource_count"].as_u64(),
                ) else {
                    continue;
                };
                if next <= current || limiters == 0 {
                    continue;
                }
                let limiting: Vec<_> = constraints
                    .iter()
                    .enumerate()
                    .filter(|(_, row)| row["function"] == function && row["limiting"] == true)
                    .collect();
                if limiting.len() as u64 != limiters {
                    continue;
                }
                let mut reductions = Vec::new();
                let mut resources = BTreeSet::new();
                let mut evidence = BTreeMap::new();
                for (position, row) in limiting {
                    let (Some(name), Some(unit), Some(scope), Some(units), Some(reduction)) = (
                        row["name"].as_str(),
                        row["unit"].as_str(),
                        row["allocation_scope"].as_str(),
                        row["units"].as_u64(),
                        row["reduction_units_to_next_better_tier"].as_u64(),
                    ) else {
                        break;
                    };
                    let Some(maximum) = units.checked_sub(reduction) else {
                        return Err(Error::Message(
                            "residency reduction exceeds resource usage".into(),
                        ));
                    };
                    if !resources.insert(name) || name.is_empty() || reduction == 0 {
                        return Err(Error::Message(
                            "report has invalid residency requirements".into(),
                        ));
                    }
                    reductions.push(format!(
                        "{name} by {reduction} {unit}/{scope} to at most {maximum}"
                    ));
                    cite(
                        &mut evidence,
                        &format!("/residency_constraints/rows/{position}"),
                        row,
                    );
                }
                if reductions.len() as u64 != limiters {
                    continue;
                }
                cite(
                    &mut evidence,
                    &format!("/entries/rows/{index}/target_resources/residency"),
                    summary,
                );
                guidance.suggestions.push(ReportSuggestion {
                    kind: "amdgpu.residency_cliff", function: function.into(), evidence,
                    action: format!("Reduce {} together to target modeled residency {current} -> {next} subgroups/SIMD. Recompile and benchmark; higher residency does not guarantee higher throughput.", reductions.join(" and ")),
                });
            }
        }
        if guidance.bank_service_available {
            let groups = groups
                .and_then(Value::as_array)
                .ok_or_else(|| Error::Message("report bank groups must be an array".into()))?;
            if self
                .document
                .pointer("/source_low/memory/bank_service_group_count")
                .and_then(Value::as_u64)
                != Some(groups.len() as u64)
            {
                return Err(Error::Message(
                    "report bank group count does not match its rows".into(),
                ));
            }
            for (index, group) in groups.iter().enumerate() {
                let Some(function) = group["function"]
                    .as_str()
                    .filter(|name| names.contains(name))
                else {
                    continue;
                };
                if !matches!(
                    group.pointer("/model/evidence").and_then(Value::as_str),
                    Some("public-vendor-documentation" | "silicon-calibrated-vendor-model")
                ) {
                    continue;
                }
                let summary = &group["summary"];
                let (Some(exact), Some(conflicts), Some(extra), Some(required), Some(uncontended)) = (
                    summary["exact_packet_count"].as_u64(),
                    summary
                        .pointer("/structural/conflicted_packet_count")
                        .and_then(Value::as_u64),
                    summary
                        .pointer("/structural/extra_round_count")
                        .and_then(Value::as_u64),
                    summary
                        .pointer("/structural/required_round_count")
                        .and_then(Value::as_u64),
                    summary
                        .pointer("/structural/uncontended_round_count")
                        .and_then(Value::as_u64),
                ) else {
                    continue;
                };
                if exact == 0 || conflicts == 0 || extra == 0 {
                    continue;
                }
                if conflicts > exact
                    || uncontended == 0
                    || uncontended.checked_add(extra) != Some(required)
                {
                    return Err(Error::Message(
                        "report bank service counts are inconsistent".into(),
                    ));
                }
                let mut evidence = BTreeMap::new();
                cite(
                    &mut evidence,
                    &format!("/source_low/memory/bank_service_groups/{index}"),
                    group,
                );
                let location = group["source_root"].as_str().unwrap_or("this access group");
                guidance.suggestions.push(ReportSuggestion {
                    kind: "amdgpu.lds_bank_service", function: function.into(), evidence,
                    action: format!("Test a pitch or padding change for {location}: {conflicts} proven packet sites require {extra} extra bank-service rounds. These are structural counts, not measured cycles; unknown and unmodeled accesses remain unqualified. Check the resulting LDS residency and benchmark before adopting the layout."),
                });
            }
        }
        Ok(guidance)
    }
}

fn cite(evidence: &mut BTreeMap<String, Value>, pointer: &str, value: &Value) {
    evidence.insert(pointer.into(), value.clone());
}
