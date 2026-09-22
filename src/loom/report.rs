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
            .document
            .pointer("/entries/rows")
            .and_then(Value::as_array);
        Ok(rows
            .into_iter()
            .flatten()
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
                }
            })
            .collect())
    }

    /// Compare named entries after checking compiler, schema and target identity.
    /// Added/removed entries are explicit; unknown analysis is never treated as zero.
    pub fn changes(&self, after: &Self) -> Result<Vec<EntryChange>> {
        self.ensure_comparable(after)?;
        let index = |report: &Self| -> Result<BTreeMap<String, EntryResources>> {
            let mut rows = BTreeMap::new();
            for row in report.entries()? {
                let name = row.function.clone().ok_or_else(|| {
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
