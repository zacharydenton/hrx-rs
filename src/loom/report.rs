//! Versioned compiler evidence. Missing analysis remains unknown.
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

/// Structured evidence tied to the exact compiler which produced it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompileReport {
    compiler: String,
    target: String,
    processor_mode: super::ProcessorMode,
    document: Value,
}

/// Resource facts for one compiled entry. These are not throughput estimates.
#[derive(Clone, Debug, Serialize)]
pub struct EntryResources {
    /// Compiled function name.
    pub function: Option<String>,
    /// Machine-code bytes, excluding container overhead.
    pub code_bytes: Option<u64>,
    /// Final scalar register count.
    pub scalar_registers: Option<u64>,
    /// Final vector register count.
    pub vector_registers: Option<u64>,
    /// Workgroup-local storage in bytes.
    pub local_bytes: Option<u64>,
    /// Per-workitem private storage in bytes.
    pub private_bytes: Option<u64>,
    /// Compiler-reported spill count.
    pub spills: Option<u64>,
    /// Target-model occupancy percentage, when known.
    pub occupancy_percent: Option<u64>,
    /// Packets covered by the compiler's LDS bank-service model.
    pub bank_modeled_packets: Option<u64>,
    /// Packets outside the model's coverage; these are not conflict-free claims.
    pub bank_unmodeled_packets: Option<u64>,
    /// Modeled packets with structural bank conflicts, not measured stalls.
    pub bank_conflicted_packets: Option<u64>,
    /// Additional structural bank-service rounds, not hardware cycles.
    pub bank_extra_rounds: Option<u64>,
}

/// Compiler-planned waits grouped by function, hardware counter and reason.
/// Counts describe the compiled program, not measured device latency.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WaitReason {
    /// Compiled function, when identified by the compiler.
    pub function: Option<String>,
    /// Target counter name.
    pub counter: Option<String>,
    /// Compiler explanation for this wait family.
    pub reason: Option<String>,
    /// Structural wait counts, preserving missing evidence.
    pub summary: WaitCounts,
}

/// Structural wait evidence. Missing facts remain unknown.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct WaitCounts {
    /// Total wait actions.
    pub action_count: Option<u64>,
    /// Waits explicitly authored in source.
    pub explicit_action_count: Option<u64>,
    /// Waits inserted by the compiler.
    pub planned_action_count: Option<u64>,
    /// Waits draining a counter completely.
    pub full_drain_count: Option<u64>,
    /// Waits permitting some requests to remain outstanding.
    pub partial_wait_count: Option<u64>,
    /// Total packets drained by these waits.
    pub drained_count: Option<u64>,
    /// Most packets drained by one wait.
    pub max_drained_count: Option<u64>,
    /// Maximum block-local outstanding packets before a wait.
    pub max_outstanding_before: Option<u64>,
    /// Maximum block-local outstanding packets before a full drain.
    pub max_full_drain_outstanding_before: Option<u64>,
}

/// Wait evidence paired by function, counter and reason, independent of row order.
#[derive(Clone, Debug, Serialize)]
pub struct WaitReasonChange {
    /// Compiled function shared by the compared rows.
    pub function: String,
    /// Target counter name.
    pub counter: String,
    /// Compiler explanation for this wait family.
    pub reason: String,
    /// Evidence before the change, or null for an added reason.
    pub before: Option<WaitCounts>,
    /// Evidence after the change, or null for a removed reason.
    pub after: Option<WaitCounts>,
    /// Signed after-minus-before changes, only for known facts on both sides.
    pub delta: BTreeMap<&'static str, Option<i128>>,
}

impl WaitCounts {
    fn fields(&self) -> [(&'static str, Option<u64>); 9] {
        [
            ("action_count", self.action_count),
            ("explicit_action_count", self.explicit_action_count),
            ("planned_action_count", self.planned_action_count),
            ("full_drain_count", self.full_drain_count),
            ("partial_wait_count", self.partial_wait_count),
            ("drained_count", self.drained_count),
            ("max_drained_count", self.max_drained_count),
            ("max_outstanding_before", self.max_outstanding_before),
            (
                "max_full_drain_outstanding_before",
                self.max_full_drain_outstanding_before,
            ),
        ]
    }
}

/// Resource change for one named function; missing or unknown facts stay null.
#[derive(Clone, Debug, Serialize)]
pub struct EntryChange {
    /// Function identity shared by the compared entries.
    pub function: String,
    /// Evidence before the change, or null if the entry was added.
    pub before: Option<EntryResources>,
    /// Evidence after the change, or null if the entry was removed.
    pub after: Option<EntryResources>,
    /// Signed after-minus-before changes, only for known facts on both sides.
    pub delta: BTreeMap<&'static str, Option<i128>>,
}

impl CompileReport {
    pub(super) fn new(
        compiler: String,
        target: String,
        processor_mode: super::ProcessorMode,
        document: Value,
    ) -> Result<Self> {
        let report = Self {
            compiler,
            target,
            processor_mode,
            document,
        };
        report.validate()?;
        Ok(report)
    }

    /// Reject unrecognized evidence instead of interpreting a different schema.
    pub fn validate(&self) -> Result<()> {
        if self.compiler.is_empty()
            || crate::Target::new(&self.target).is_err()
            || self.document["kind"] != "loom.compile_report"
            || self.document["schema_version"].as_u64() != Some(0)
        {
            return Err(Error::Unsupported(
                "compile report schema or compiler identity is unsupported; regenerate the report"
                    .into(),
            ));
        }
        Ok(())
    }

    /// Content digest of the producing compiler library.
    pub fn compiler_identity(&self) -> &str {
        &self.compiler
    }

    /// Exact device profile used to specialize the source.
    pub fn target(&self) -> &str {
        &self.target
    }

    /// Requested AMDGPU execution policy.
    pub fn processor_mode(&self) -> super::ProcessorMode {
        self.processor_mode
    }

    /// Original versioned JSON, including evidence not represented by accessors.
    pub fn json(&self) -> &Value {
        &self.document
    }

    /// Resource summaries, preserving unknown values rather than reporting zero.
    pub fn entries(&self) -> Result<Vec<EntryResources>> {
        self.validate()?;
        let rows = self
            .rows("entries")?
            .ok_or_else(|| Error::Message("report entries are missing".into()))?;
        Ok(rows
            .iter()
            .map(|row| {
                let number = |path| row.pointer(path).and_then(Value::as_u64);
                EntryResources {
                    function: row["function"].as_str().map(str::to_owned),
                    code_bytes: number("/code_byte_count"),
                    scalar_registers: number("/target_resources/scalar/final/register_count"),
                    vector_registers: number("/target_resources/vector/final/register_count"),
                    local_bytes: number("/local_memory_bytes"),
                    private_bytes: number("/private_memory_bytes"),
                    spills: number("/allocation_spill_count"),
                    occupancy_percent: number("/target_resources/occupancy_percent"),
                    bank_modeled_packets: number(
                        "/source_low_memory/bank_service/modeled_packet_count",
                    ),
                    bank_unmodeled_packets: number(
                        "/source_low_memory/bank_service/unmodeled_packet_count",
                    ),
                    bank_conflicted_packets: number(
                        "/source_low_memory/bank_service/structural/conflicted_packet_count",
                    ),
                    bank_extra_rounds: number(
                        "/source_low_memory/bank_service/structural/extra_round_count",
                    ),
                }
            })
            .collect())
    }

    /// Wait explanations from detailed reports. `None` means this evidence was
    /// not collected; `Some([])` means it was collected with no wait rows.
    pub fn wait_reasons(&self) -> Result<Option<Vec<WaitReason>>> {
        self.validate()?;
        self.rows("wait_reason_summary_rows")?
            .map(|rows| {
                rows.iter()
                    .cloned()
                    .map(serde_json::from_value)
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .map_err(Error::from)
            })
            .transpose()
    }

    fn rows(&self, name: &str) -> Result<Option<&[Value]>> {
        let Some(section) = self.document.get(name).filter(|value| !value.is_null()) else {
            return Ok(None);
        };
        let rows = match section.get("rows") {
            Some(Value::Array(rows)) => rows.as_slice(),
            None if section["count"].as_u64() == Some(0) => &[],
            _ => {
                return Err(Error::Message(format!(
                    "report {name} is missing valid rows"
                )));
            }
        };
        if let Some(count) = section.get("count")
            && count.as_u64() != Some(rows.len() as u64)
        {
            return Err(Error::Message(format!(
                "report {name} count does not match its rows"
            )));
        }
        if rows.iter().any(|row| !row.is_object()) {
            return Err(Error::Message(format!(
                "report {name} contains a non-object row"
            )));
        }
        Ok(Some(rows))
    }

    /// Compare wait reasons by semantic identity. `None` means at least one
    /// report lacks wait evidence; an empty vector means both collected no rows.
    /// Added/removed reasons have null deltas, never invented zero counts.
    pub fn wait_reason_changes(&self, after: &Self) -> Result<Option<Vec<WaitReasonChange>>> {
        self.ensure_comparable(after)?;
        let before_rows = self.wait_reasons()?;
        let after_rows = after.wait_reasons()?;
        let (Some(before_rows), Some(after_rows)) = (before_rows, after_rows) else {
            return Ok(None);
        };
        let index =
            |rows: Vec<WaitReason>| -> Result<BTreeMap<(String, String, String), WaitCounts>> {
                let mut indexed = BTreeMap::new();
                for row in rows {
                    let identity = row
                        .function
                        .zip(row.counter)
                        .zip(row.reason)
                        .filter(|((function, counter), reason)| {
                            !function.is_empty() && !counter.is_empty() && !reason.is_empty()
                        })
                        .map(|((function, counter), reason)| (function, counter, reason))
                        .ok_or_else(|| {
                            Error::Message("report wait reason has no complete identity".into())
                        })?;
                    if indexed.insert(identity, row.summary).is_some() {
                        return Err(Error::Message(
                            "report contains duplicate wait reason identities".into(),
                        ));
                    }
                }
                Ok(indexed)
            };
        let mut before = index(before_rows)?;
        let mut after = index(after_rows)?;
        let identities: std::collections::BTreeSet<_> =
            before.keys().chain(after.keys()).cloned().collect();
        Ok(Some(
            identities
                .into_iter()
                .map(|identity| {
                    let before = before.remove(&identity);
                    let after = after.remove(&identity);
                    let unknown = WaitCounts::default();
                    let delta = before
                        .as_ref()
                        .unwrap_or(&unknown)
                        .fields()
                        .into_iter()
                        .zip(after.as_ref().unwrap_or(&unknown).fields())
                        .map(|((field, before), (_, after))| {
                            (
                                field,
                                before
                                    .zip(after)
                                    .map(|(before, after)| i128::from(after) - i128::from(before)),
                            )
                        })
                        .collect();
                    WaitReasonChange {
                        function: identity.0,
                        counter: identity.1,
                        reason: identity.2,
                        before,
                        after,
                        delta,
                    }
                })
                .collect(),
        ))
    }

    /// Compare named entries after checking compiler, schema and target identity.
    /// Added/removed entries are explicit; unknown analysis is never treated as zero.
    pub fn changes(&self, after: &Self) -> Result<Vec<EntryChange>> {
        self.ensure_comparable(after)?;
        let index = |report: &Self| -> Result<BTreeMap<String, EntryResources>> {
            let mut rows = BTreeMap::new();
            for row in report.entries()? {
                let name = row
                    .function
                    .clone()
                    .filter(|name| !name.is_empty())
                    .ok_or_else(|| {
                        Error::Message("report entry has no function identity".into())
                    })?;
                if rows.insert(name, row).is_some() {
                    return Err(Error::Message(
                        "report contains duplicate function identities".into(),
                    ));
                }
            }
            Ok(rows)
        };
        let mut before = index(self)?;
        let mut after = index(after)?;
        let names: std::collections::BTreeSet<_> =
            before.keys().chain(after.keys()).cloned().collect();
        Ok(names
            .into_iter()
            .map(|function| {
                let before = before.remove(&function);
                let after = after.remove(&function);
                let fields = |row: Option<&EntryResources>| {
                    [
                        row.and_then(|r| r.code_bytes),
                        row.and_then(|r| r.scalar_registers),
                        row.and_then(|r| r.vector_registers),
                        row.and_then(|r| r.local_bytes),
                        row.and_then(|r| r.private_bytes),
                        row.and_then(|r| r.spills),
                        row.and_then(|r| r.occupancy_percent),
                        row.and_then(|r| r.bank_modeled_packets),
                        row.and_then(|r| r.bank_unmodeled_packets),
                        row.and_then(|r| r.bank_conflicted_packets),
                        row.and_then(|r| r.bank_extra_rounds),
                    ]
                };
                let names = [
                    "code_bytes",
                    "scalar_registers",
                    "vector_registers",
                    "local_bytes",
                    "private_bytes",
                    "spills",
                    "occupancy_percent",
                    "bank_modeled_packets",
                    "bank_unmodeled_packets",
                    "bank_conflicted_packets",
                    "bank_extra_rounds",
                ];
                let delta = names
                    .into_iter()
                    .zip(
                        fields(before.as_ref())
                            .into_iter()
                            .zip(fields(after.as_ref()))
                            .map(|(before, after)| {
                                before
                                    .zip(after)
                                    .map(|(before, after)| i128::from(after) - i128::from(before))
                            }),
                    )
                    .collect();
                EntryChange {
                    function,
                    before,
                    after,
                    delta,
                }
            })
            .collect())
    }

    /// Check the minimum identity contract for comparing compiler evidence.
    /// Source and configuration may differ: those are the experiment variables.
    pub fn ensure_comparable(&self, other: &Self) -> Result<()> {
        self.validate()?;
        other.validate()?;
        if self.compiler != other.compiler
            || self.target != other.target
            || self.processor_mode != other.processor_mode
            || ["backend", "target_key", "mode"].iter().any(|key| {
                self.document.get(key).is_none()
                    || self.document.get(key) != other.document.get(key)
            })
        {
            return Err(Error::Message(
                "reports must use the same compiler, target, backend, processor policy, and detail mode".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn evidence() -> Value {
        serde_json::json!({"kind":"loom.compile_report", "schema_version":0,
            "backend":"amdgpu-hsaco", "target_key":"gfx1151", "mode":"summary",
            "entries":{"rows":[{"function":"empty", "allocation_spill_count":0}]}})
    }
    #[test]
    fn bank_coverage_and_wait_reasons_preserve_missing_evidence() {
        let mut document = evidence();
        document["entries"]["rows"][0]["source_low_memory"] = serde_json::json!({
            "bank_service": {"modeled_packet_count": 3, "unmodeled_packet_count": 2,
                "structural": {"conflicted_packet_count": 1, "extra_round_count": 4}}
        });
        let mut report = CompileReport::new(
            "compiler-a".into(),
            "gfx1151".into(),
            super::super::ProcessorMode::Default,
            document,
        )
        .unwrap();
        let entry = report.entries().unwrap().remove(0);
        assert_eq!(entry.bank_modeled_packets, Some(3));
        assert_eq!(entry.bank_unmodeled_packets, Some(2));
        assert_eq!(entry.bank_conflicted_packets, Some(1));
        assert_eq!(entry.bank_extra_rounds, Some(4));
        assert!(report.wait_reasons().unwrap().is_none());
        report.document["wait_reason_summary_rows"] = serde_json::json!({"count": 0});
        assert!(report.wait_reasons().unwrap().unwrap().is_empty());
        report.document["wait_reason_summary_rows"] = serde_json::json!({"count": 1,
            "rows": [{"function": "empty", "counter": "smem", "reason": "storage_reuse",
                "summary": {"action_count": 2, "full_drain_count": 2}}]});
        let waits = report.wait_reasons().unwrap().unwrap();
        assert_eq!(waits[0].summary.full_drain_count, Some(2));
        assert_eq!(waits[0].summary.partial_wait_count, None);
        let mut after = report.clone();
        after.document["entries"]["rows"][0]["source_low_memory"]["bank_service"]["structural"]["extra_round_count"] =
            1.into();
        assert_eq!(
            report.changes(&after).unwrap()[0].delta["bank_extra_rounds"],
            Some(-3)
        );
        report.document["wait_reason_summary_rows"] = serde_json::json!({"count": 1});
        assert!(report.wait_reasons().is_err());
    }

    #[test]
    fn unknown_resources_are_not_zero_and_comparisons_check_identity() {
        let report = CompileReport::new(
            "compiler-a".into(),
            "gfx1151".into(),
            super::super::ProcessorMode::Default,
            evidence(),
        )
        .unwrap();
        let row = report.entries().unwrap().remove(0);
        assert_eq!(row.spills, Some(0));
        assert_eq!(row.vector_registers, None);
        report.ensure_comparable(&report).unwrap();
        let changes = report.changes(&report).unwrap();
        assert_eq!(changes[0].delta["spills"], Some(0));
        assert_eq!(changes[0].delta["vector_registers"], None);
        let other = CompileReport::new(
            "compiler-b".into(),
            "gfx1151".into(),
            super::super::ProcessorMode::Default,
            evidence(),
        )
        .unwrap();
        assert!(report.ensure_comparable(&other).is_err());
        let mut policy = report.clone();
        policy.processor_mode = super::super::ProcessorMode::ComputeUnit;
        assert!(report.ensure_comparable(&policy).is_err());
        let mut target = report.clone();
        target.target = "gfx1100".into();
        assert!(report.ensure_comparable(&target).is_err());
        let mut future = evidence();
        future["schema_version"] = 1.into();
        assert!(
            CompileReport::new(
                "compiler-a".into(),
                "gfx1151".into(),
                super::super::ProcessorMode::Default,
                future
            )
            .is_err()
        );
    }
}
