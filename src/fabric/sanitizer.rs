//! Owned native sanitizer feedback, collected at checked execution boundaries.
use super::*;
use serde::Serialize;
use std::collections::BTreeMap;

/// Bounded runtime storage for sanitizer diagnostics, address shadow, and race state.
#[derive(Clone, Debug)]
pub struct SanitizerRuntimeOptions {
    /// Power-of-two packet capacity, 128 bytes through 16 MiB.
    /// Collection happens after retirement; excess reports increment a drop count.
    pub capacity_bytes: usize,
    /// Maximum combined address/race shadow bytes per prepared dispatch, excluding code and reports.
    /// Address shadow covers bound allocations and code; race shadow covers the grid and LDS.
    pub maximum_shadow_bytes: usize,
    /// Optional charge for reports, shadow storage, and private instrumented code.
    /// Reservations remain with their allocations through native use.
    pub memory_budget: Option<crate::residency::MemoryBudget>,
}
impl Default for SanitizerRuntimeOptions {
    fn default() -> Self {
        Self {
            capacity_bytes: 64 << 10,
            maximum_shadow_bytes: 64 << 20,
            memory_budget: None,
        }
    }
}
/// Native sanitizer failure classification.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum SanitizerCheck {
    /// Check kind not classified by the native producer.
    Unknown(u32),
    /// Integer arithmetic exceeded its declared domain.
    IntegerOverflow,
    /// An integer divisor was zero.
    DivideByZero,
    /// A memory address violated required alignment.
    Alignment,
    /// Floating-point data violated a not-NaN contract.
    FloatNanContract,
    /// Execution reached a declared unreachable location.
    Unreachable,
    /// An authored runtime assertion failed.
    Assertion,
    /// Conflicting unsynchronized accesses to workgroup memory.
    DataRace,
    /// A memory access touched inaccessible allocation bytes.
    InvalidAccess,
}
/// Owned compiler site metadata. Line and column numbers preserve native values.
#[derive(Clone, Debug, Serialize)]
pub struct SanitizerSite {
    /// Compiler-assigned operation kind.
    pub operation_kind: u32,
    /// Original source name, when the compiler retained a file location.
    pub source_name: Option<String>,
    /// Start line and column, when a source range exists.
    pub start: Option<[u32; 2]>,
    /// End line and column, when a source range exists.
    pub end: Option<[u32; 2]>,
    /// Compiler-owned encoded predicate payload.
    pub predicate_payload: Vec<u8>,
}
/// Native memory access classification for sanitizer diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum SanitizerAccess {
    /// Non-atomic read.
    Read,
    /// Non-atomic write.
    Write,
    /// Non-atomic read-modify-write.
    ReadWrite,
    /// Atomic memory operation.
    Atomic,
    /// Unknown native classification.
    Unknown(u32),
}
impl SanitizerAccess {
    fn from_native(value: u32) -> Self {
        match value {
            1 => Self::Read,
            2 => Self::Write,
            3 => Self::ReadWrite,
            4 => Self::Atomic,
            other => Self::Unknown(other),
        }
    }
}
/// Owned details of both accesses involved in a native race diagnostic.
#[derive(Clone, Debug, Serialize)]
pub struct SanitizerRace {
    /// Native memory-space code: global=1, workgroup=2, private=3.
    pub memory_space: u32,
    /// Current access classification.
    pub current_access: SanitizerAccess,
    /// Prior access classification.
    pub prior_access: SanitizerAccess,
    /// Whether the current access used an atomic memory operation.
    pub current_atomic: bool,
    /// Whether the prior access used an atomic memory operation.
    pub prior_atomic: bool,
    /// Access width in bytes.
    pub access_bytes: u32,
    /// Address, or byte offset within the reported memory space.
    pub memory_address: u64,
    /// Prior compiler instrumentation site.
    pub prior_site_id: u64,
    /// Owned source metadata for the prior access.
    pub prior_site: Option<SanitizerSite>,
    /// Device address of the detector entry.
    pub shadow_address: u64,
    /// Prior encoded detector entry.
    pub shadow_value: u64,
    /// Full current workgroup coordinates.
    pub current_workgroup: [u32; 3],
    /// Full current workitem coordinates, or linear id in X when flagged.
    pub current_workitem: [u32; 3],
    /// Full prior workgroup coordinates.
    pub prior_workgroup: [u32; 3],
    /// Full prior workitem coordinates, or linear id in X when flagged.
    pub prior_workitem: [u32; 3],
    /// Whether current workitem X contains a linear local id.
    pub current_workitem_linear: bool,
    /// Whether prior workitem X contains a linear local id.
    pub prior_workitem_linear: bool,
}

/// Owned details of an address sanitizer failure.
#[derive(Clone, Debug, Serialize)]
pub struct SanitizerAddress {
    /// Read, write, or atomic operation.
    pub access: SanitizerAccess,
    /// Original application address checked by the instrumentation.
    pub fault_address: u64,
    /// Access width in bytes.
    pub access_bytes: u64,
    /// Computed shadow address; out-of-window checks skip the load.
    pub shadow_address: u64,
    /// Shadow byte, or full poison when the lookup is out of bounds.
    pub shadow_value: u64,
}
/// One fully owned diagnostic from a completed native invocation.
#[derive(Clone, Debug, Serialize)]
pub struct SanitizerReport {
    /// Failure classification.
    pub check: SanitizerCheck,
    /// Instrumentation site identifier.
    pub site_id: u64,
    /// Raw bits of check-specific operands.
    pub operands: [u64; 2],
    /// Native dispatch packet address, if supplied by the execution backend.
    pub dispatch_address: u64,
    /// X workgroup coordinate reported by the native producer.
    pub workgroup_x: u32,
    /// X workitem coordinate reported by the native producer.
    pub workitem_x: u32,
    /// Source and predicate metadata from the retained executable.
    pub site: Option<SanitizerSite>,
    /// Both sides of a workgroup race, when this is a race report.
    pub race: Option<SanitizerRace>,
    /// Address details when this is an inaccessible memory report.
    pub address: Option<SanitizerAddress>,
}
/// Reports since the last successful collection across all uses of one kernel.
#[derive(Clone, Debug, Serialize)]
pub struct SanitizerReports {
    /// Successfully decoded reports in reservation order.
    pub reports: Vec<SanitizerReport>,
    /// Reports that did not fit or could not reserve channel capacity.
    /// Nonzero means the report set is incomplete.
    pub dropped: u64,
}

pub(super) struct Feedback {
    device: Device,
    buffer: Buffer,
    capacity: usize,
    sites: BTreeMap<u64, SanitizerSite>,
}
impl Feedback {
    pub(super) fn new(
        device: &Device,
        options: &SanitizerRuntimeOptions,
        sites: &[u8],
    ) -> Result<Self> {
        let capacity = options.capacity_bytes;
        if !capacity.is_power_of_two() || !(128..=16 << 20).contains(&capacity) {
            return Err(Error::Message(
                "sanitizer capacity must be a power of two between 128 bytes and 16 MiB".into(),
            ));
        }
        let sites = parse_sites(sites)?;
        let buffer = device.fabric().allocate_owned(
            64 + capacity,
            device,
            AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
            64,
            true,
            options.memory_budget.as_ref(),
        )?;
        let base = buffer.device_address(device)?;
        let mut header = [0; 64];
        put32(&mut header, 0, 64);
        put32(&mut header, 8, 1);
        put64(&mut header, 16, base.checked_add(64).ok_or_else(corrupt)?);
        put64(&mut header, 24, capacity as u64);
        buffer.write(0, &header)?;
        Ok(Self {
            device: device.clone(),
            buffer,
            capacity,
            sites,
        })
    }
    pub(super) fn buffer(&self) -> &Buffer {
        &self.buffer
    }
    pub(super) fn configure(&self, output: &mut [u8]) -> Result<()> {
        output.fill(0);
        put32(output, 0, 64);
        put32(output, 8, 1);
        put64(output, 16, self.buffer.device_address(&self.device)?);
        // Null notify_signal is the native ABI's polling-only contract.
        // Each private channel has exactly one retained executable context.
        put64(output, 32, 1);
        Ok(())
    }
    pub(super) fn drain(&self) -> Result<SanitizerReports> {
        self.buffer.with_host_bytes(|bytes| {
            let result = decode(bytes, self.capacity, &self.sites)?;
            // No consumer advances read_tail while a producer is live. Every
            // reservation therefore fits contiguously before capacity, without
            // needing a doubly mapped ring. Reset only under the same lease lock
            // that excludes new submissions, after all prior uses have retired.
            bytes[32..56].fill(0);
            bytes[64..].fill(0);
            Ok(result)
        })
    }
}

pub(super) struct AddressState {
    pub(super) shadow: Buffer,
    config: [u8; 96],
}
fn address_shadow(mut ranges: Vec<(u64, usize)>, maximum: usize) -> Result<(u64, u64, Vec<u8>)> {
    ranges.sort_unstable();
    let mut merged: Vec<(u64, u64)> = Vec::new();
    for (start, len) in ranges {
        if start % 8 != 0 || len == 0 {
            return Err(Error::Unsupported(
                "address sanitizer allocations must start on an 8-byte boundary and be nonempty"
                    .into(),
            ));
        }
        let end = start
            .checked_add(len as u64)
            .ok_or_else(|| Error::Message("address sanitizer allocation overflow".into()))?;
        if let Some(last) = merged.last_mut()
            && start <= last.1
        {
            last.1 = last.1.max(end);
            continue;
        }
        merged.push((start, end));
    }
    let base = merged
        .first()
        .ok_or_else(|| Error::Message("empty address sanitizer window".into()))?
        .0;
    let end = merged
        .last()
        .unwrap()
        .1
        .checked_add(7)
        .map(|n| n & !7)
        .ok_or_else(|| Error::Message("address sanitizer window overflow".into()))?;
    let size = usize::try_from((end - base) / 8)
        .ok()
        .filter(|n| *n <= maximum)
        .ok_or_else(|| Error::Message("address shadow exceeds configured byte limit".into()))?;
    let mut bytes = vec![0xff; size];
    for (start, end) in merged {
        let first = ((start - base) / 8) as usize;
        let last = ((end - base) / 8) as usize;
        bytes[first..last].fill(0);
        if end % 8 != 0 {
            bytes[last] = (end % 8) as u8;
        }
    }
    Ok((base, end, bytes))
}
impl AddressState {
    pub(super) fn new(
        device: &Device,
        options: &SanitizerRuntimeOptions,
        ranges: Vec<(u64, usize)>,
    ) -> Result<Self> {
        let (base, end, bytes) = address_shadow(ranges, options.maximum_shadow_bytes)?;
        let shadow = device.fabric().allocate_owned(
            bytes.len(),
            device,
            AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
            64,
            false,
            options.memory_budget.as_ref(),
        )?;
        shadow.write(0, &bytes)?;
        let mut config = [0; 96];
        put32(&mut config, 0, 96);
        put32(&mut config, 8, 1);
        put32(&mut config, 12, 3);
        put64(
            &mut config,
            16,
            shadow.device_address(device)?.wrapping_sub(base >> 3),
        );
        put64(&mut config, 24, base);
        put64(&mut config, 32, end - base);
        put64(&mut config, 40, bytes.len() as u64);
        put64(&mut config, 48, bytes.len() as u64);
        Ok(Self { shadow, config })
    }
    pub(super) fn configure(&self, output: &mut [u8]) -> Result<()> {
        if output.len() != 96 {
            return Err(Error::Unsupported(
                "unexpected address configuration ABI".into(),
            ));
        }
        output.copy_from_slice(&self.config);
        Ok(())
    }
}

// One immutable configuration per prepared dispatch. AQL barrier packets order
// a GPU clear before every invocation, so a single shadow slot is sufficient
// even across ring wrap and native dispatch-generation rollover.
pub(super) struct RaceState {
    pub(super) shadow: Buffer,
    pub(super) queue_state: Buffer,
    config: [u8; 96],
}
fn race_layout(local_bytes: u32, grid: [u32; 3], maximum: usize) -> Result<(u32, usize, usize)> {
    let capacity = grid
        .into_iter()
        .try_fold(1u32, |n, v| n.checked_mul(v))
        .filter(|n| *n != 0)
        .ok_or_else(|| Error::Message("race shadow workgroup count overflow or zero".into()))?;
    // Four-byte granules match the native workgroup race ABI.
    let stride = (local_bytes as usize)
        .div_ceil(4)
        .checked_mul(8)
        .and_then(|n| n.checked_add(8))
        .ok_or_else(corrupt)?;
    let size = stride
        .checked_mul(capacity as usize)
        .filter(|n| *n <= maximum && *n / 4 <= u32::MAX as usize - 63)
        .ok_or_else(|| Error::Message("race shadow exceeds configured byte limit".into()))?;
    Ok((capacity, stride, size))
}
impl RaceState {
    pub(super) fn new(
        device: &Device,
        options: &SanitizerRuntimeOptions,
        local_bytes: u32,
        grid: [u32; 3],
        ring: u64,
        mask: u64,
    ) -> Result<Self> {
        let (capacity, stride, size) =
            race_layout(local_bytes, grid, options.maximum_shadow_bytes)?;
        let allocate = |bytes| {
            device.fabric().allocate_owned(
                bytes,
                device,
                AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                64,
                false,
                options.memory_budget.as_ref(),
            )
        };
        let shadow = allocate(size)?;
        let queue_state = allocate(112)?;
        let base = shadow.device_address(device)?;
        let mut header = [0; 112];
        for (at, value) in [(0, 112), (28, 1), (88, capacity), (92, 8), (96, 2)] {
            put32(&mut header, at, value);
        }
        for (at, value) in [
            (32, ring),
            (40, mask),
            (56, base),
            (64, size as u64),
            (72, size as u64),
            (80, stride as u64),
        ] {
            put64(&mut header, at, value);
        }
        queue_state.write(0, &header)?;
        let mut config = [0; 96];
        for (at, value) in [(0, 96), (8, 3), (12, 2), (48, capacity), (52, 8), (80, 1)] {
            put32(&mut config, at, value);
        }
        for (at, value) in [
            (16, base),
            (24, size as u64),
            (32, size as u64),
            (40, stride as u64),
            (56, ring),
            (64, mask),
            (72, queue_state.device_address(device)?),
        ] {
            put64(&mut config, at, value);
        }
        Ok(Self {
            shadow,
            queue_state,
            config,
        })
    }
    pub(super) fn configure(&self, output: &mut [u8]) -> Result<()> {
        if output.len() != self.config.len() {
            return Err(Error::Unsupported(
                "unexpected race configuration ABI".into(),
            ));
        }
        output.copy_from_slice(&self.config);
        Ok(())
    }
    pub(super) fn clear_kernel(
        &self,
        device: &Device,
        budget: Option<&crate::residency::MemoryBudget>,
    ) -> Result<(Kernel, [u32; 3])> {
        let words = self.shadow.len() / 4;
        let groups = words.div_ceil(64);
        let source = format!(
            r#"
kernel.def @clear_race_shadow() {{
  %one = index.constant 1 : index
  %threads = index.constant 64 : index
  %groups = index.constant {groups} : index
  kernel.launch.config workgroups(%groups, %one, %one) workgroup_size(%threads, %one, %one) : index
}} launch(%output: buffer) {{
  %threads = index.constant 64 : index
  %count = index.constant {words} : index
  %zero = index.constant 0 : offset
  %value = scalar.constant 0 : i32
  %lane = kernel.workitem.id<x> : index
  %group = kernel.workgroup.id<x> : index
  %base = index.mul %group, %threads : index
  %index = index.add %base, %lane : index
  %memory = buffer.assume.memory_space<global> %output : buffer
  %view = buffer.view %memory[%zero] : buffer -> view<{words}xi32>
  %inside = index.cmp ult, %index, %count : index
  scf.if %inside {{
    view.store %value, %view[%index] : i32, view<{words}xi32>
  }}
  kernel.return
}}
"#
        );
        let artifact = crate::loom::Compiler::for_target(None, device.target())?
            .module(&source)
            .compile(&crate::loom::Specialization::new("clear_race_shadow"))?;
        // This private utility writes only the checked shadow allocation.
        let kernel =
            unsafe { device.load_image(artifact.bytes(), artifact.symbol(), None, budget) }?;
        Ok((kernel, [groups as u32, 1, 1]))
    }
}
fn corrupt() -> Error {
    Error::Message("invalid sanitizer runtime record".into())
}
fn u16_at(bytes: &[u8], offset: usize) -> Result<u16> {
    Ok(u16::from_le_bytes(
        bytes
            .get(offset..offset + 2)
            .ok_or_else(corrupt)?
            .try_into()
            .unwrap(),
    ))
}
fn u32_at(bytes: &[u8], offset: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(
        bytes
            .get(offset..offset + 4)
            .ok_or_else(corrupt)?
            .try_into()
            .unwrap(),
    ))
}
fn u64_at(bytes: &[u8], offset: usize) -> Result<u64> {
    Ok(u64::from_le_bytes(
        bytes
            .get(offset..offset + 8)
            .ok_or_else(corrupt)?
            .try_into()
            .unwrap(),
    ))
}
fn put32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}
fn put64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}
fn region(bytes: &[u8], offset: u32, length: u32) -> Result<&[u8]> {
    let start = offset as usize;
    bytes
        .get(start..start.checked_add(length as usize).ok_or_else(corrupt)?)
        .ok_or_else(corrupt)
}
fn parse_sites(bytes: &[u8]) -> Result<BTreeMap<u64, SanitizerSite>> {
    let mut sites = BTreeMap::new();
    if bytes.is_empty() {
        return Ok(sites);
    }
    if bytes.len() < 32
        || u32_at(bytes, 0)? != 0x5449534c
        || bytes[4] != 1
        || bytes[5] != 32
        || u16_at(bytes, 6)? != 48
    {
        return Err(Error::Unsupported(
            "unexpected sanitizer site table ABI".into(),
        ));
    }
    let rows = u32_at(bytes, 8)? as usize;
    let table_end = 32usize
        .checked_add(rows.checked_mul(48).ok_or_else(corrupt)?)
        .ok_or_else(corrupt)?;
    if table_end > bytes.len() {
        return Err(corrupt());
    }
    let strings = region(bytes, u32_at(bytes, 12)?, u32_at(bytes, 16)?)?;
    let payload = region(bytes, u32_at(bytes, 20)?, u32_at(bytes, 24)?)?;
    for row in bytes[32..table_end].as_chunks::<48>().0 {
        let flags = u32_at(row, 8)?;
        let source = flags & 2 != 0;
        let source_name = source
            .then(|| {
                let text = region(strings, u32_at(row, 20)?, u32_at(row, 24)?)?;
                String::from_utf8(text.to_vec()).map_err(|_| corrupt())
            })
            .transpose()?;
        let record = SanitizerSite {
            operation_kind: u32_at(row, 4)?,
            source_name,
            start: source
                .then(|| Ok::<_, Error>([u32_at(row, 28)?, u32_at(row, 32)?]))
                .transpose()?,
            end: source
                .then(|| Ok::<_, Error>([u32_at(row, 36)?, u32_at(row, 40)?]))
                .transpose()?,
            predicate_payload: if flags & 1 != 0 {
                region(payload, u32_at(row, 12)?, u32_at(row, 16)?)?.to_vec()
            } else {
                Vec::new()
            },
        };
        if sites.insert(u64::from(u32_at(row, 0)?), record).is_some() {
            return Err(corrupt());
        }
    }
    Ok(sites)
}
fn decode(
    bytes: &[u8],
    capacity: usize,
    sites: &BTreeMap<u64, SanitizerSite>,
) -> Result<SanitizerReports> {
    if bytes.len() != 64 + capacity
        || u32_at(bytes, 0)? != 64
        || u32_at(bytes, 4)? != 0
        || u64_at(bytes, 24)? != capacity as u64
        || u64_at(bytes, 32)? != 0
    {
        return Err(corrupt());
    }
    let head = usize::try_from(u64_at(bytes, 40)?).map_err(|_| corrupt())?;
    if head > capacity || !head.is_multiple_of(64) {
        return Err(corrupt());
    }
    let mut reports = Vec::new();
    let mut offset = 0;
    while offset < head {
        let packet = &bytes[64 + offset..64 + head];
        let length = u32_at(packet, 0)? as usize;
        let kind = u16_at(packet, 6)?;
        let expected = match kind {
            1 | 5 => (128, 64),
            4 => (192, 120),
            _ => return Err(corrupt()),
        };
        if length != expected.0
            || length > packet.len()
            || u16_at(packet, 4)? != 64
            || u32_at(packet, 12)? != 1
            || u64_at(packet, 16)? != offset as u64
            || u64_at(packet, 40)? != 1
            || u32_at(packet, 64)? != expected.1
            || u32_at(packet, 68)? != 0
        {
            return Err(corrupt());
        }
        let site_id = u64_at(packet, if kind == 5 { 80 } else { 96 })?;
        let check = if kind == 1 {
            SanitizerCheck::InvalidAccess
        } else if kind == 4 {
            match u32_at(packet, 72)? {
                1 => SanitizerCheck::DataRace,
                other => SanitizerCheck::Unknown(other),
            }
        } else {
            match u32_at(packet, 72)? {
                1 => SanitizerCheck::IntegerOverflow,
                2 => SanitizerCheck::DivideByZero,
                3 => SanitizerCheck::Alignment,
                4 => SanitizerCheck::FloatNanContract,
                5 => SanitizerCheck::Unreachable,
                6 => SanitizerCheck::Assertion,
                other => SanitizerCheck::Unknown(other),
            }
        };
        let race = if kind == 4 {
            let coordinates = |offset| -> Result<[u32; 3]> {
                Ok([
                    u32_at(packet, offset)?,
                    u32_at(packet, offset + 4)?,
                    u32_at(packet, offset + 8)?,
                ])
            };
            let flags = u32_at(packet, 76)?;
            let prior_site_id = u64_at(packet, 104)?;
            Some(SanitizerRace {
                memory_space: u32_at(packet, 80)?,
                current_access: SanitizerAccess::from_native(u32_at(packet, 84)?),
                prior_access: SanitizerAccess::from_native(u32_at(packet, 88)?),
                current_atomic: flags & 1 != 0,
                prior_atomic: flags & 2 != 0,
                access_bytes: u32_at(packet, 92)?,
                memory_address: u64_at(packet, 112)?,
                prior_site_id,
                prior_site: sites.get(&prior_site_id).cloned(),
                shadow_address: u64_at(packet, 120)?,
                shadow_value: u64_at(packet, 128)?,
                current_workgroup: coordinates(136)?,
                current_workitem: coordinates(148)?,
                prior_workgroup: coordinates(160)?,
                prior_workitem: coordinates(172)?,
                current_workitem_linear: flags & 8 != 0,
                prior_workitem_linear: flags & 4 != 0,
            })
        } else {
            None
        };
        let address = if kind == 1 {
            Some(SanitizerAddress {
                access: match u32_at(packet, 72)? {
                    1 => SanitizerAccess::Read,
                    2 => SanitizerAccess::Write,
                    3 => SanitizerAccess::Atomic,
                    other => SanitizerAccess::Unknown(other),
                },
                fault_address: u64_at(packet, 80)?,
                access_bytes: u64_at(packet, 88)?,
                shadow_address: u64_at(packet, 104)?,
                shadow_value: u64_at(packet, 112)?,
            })
        } else {
            None
        };
        reports.push(SanitizerReport {
            check,
            site_id,
            operands: if kind == 5 {
                [u64_at(packet, 88)?, u64_at(packet, 96)?]
            } else {
                [0; 2]
            },
            dispatch_address: u64_at(packet, 24)?,
            workgroup_x: u32_at(packet, 32)?,
            workitem_x: u32_at(packet, 36)?,
            site: sites.get(&site_id).cloned(),
            race,
            address,
        });
        offset += length;
    }
    Ok(SanitizerReports {
        reports,
        dropped: u64_at(bytes, 48)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn address_shadow_merges_aliases_and_preserves_tails_and_gaps() {
        let (base, end, bytes) =
            address_shadow(vec![(0x1020, 5), (0x1000, 16), (0x1008, 12)], 5).unwrap();
        assert_eq!((base, end), (0x1000, 0x1028));
        assert_eq!(bytes, [0, 0, 4, 0xff, 5]);
        assert!(address_shadow(vec![(0x1000, 1), (0x1020, 1)], 4).is_err());
        assert!(address_shadow(vec![(u64::MAX - 7, 8)], 8).is_err());
        assert!(address_shadow(vec![(u64::MAX - 7, 7)], 8).is_err());
        assert!(address_shadow(vec![(3, 8)], 8).is_err());
        assert!(address_shadow(vec![], 8).is_err());
    }

    #[test]
    fn address_report_uses_its_own_access_kind_and_site_layout() {
        let mut bytes = packet();
        let record = &mut bytes[64..];
        record[6..8].copy_from_slice(&1u16.to_le_bytes());
        put32(record, 72, 3);
        put64(record, 80, u64::MAX - 1);
        put64(record, 88, 16);
        put64(record, 96, 9);
        put64(record, 104, 0x1234);
        put64(record, 112, 0xff);
        let report = decode(&bytes, 128, &BTreeMap::new())
            .unwrap()
            .reports
            .remove(0);
        assert_eq!(report.check, SanitizerCheck::InvalidAccess);
        assert_eq!(report.site_id, 9);
        assert!(report.race.is_none());
        let address = report.address.unwrap();
        assert_eq!(address.access, SanitizerAccess::Atomic);
        assert_eq!(address.fault_address, u64::MAX - 1);
        assert_eq!(address.access_bytes, 16);
        assert_eq!(address.shadow_address, 0x1234);
        assert_eq!(address.shadow_value, 0xff);
    }

    #[test]
    fn shadow_geometry_rejects_overflow_and_respects_the_limit() {
        assert_eq!(race_layout(8, [2, 3, 4], 576).unwrap(), (24, 24, 576));
        assert_eq!(race_layout(5, [1; 3], 24).unwrap(), (1, 24, 24));
        assert!(race_layout(8, [2, 3, 4], 575).is_err());
        assert!(race_layout(8, [0, 1, 1], usize::MAX).is_err());
        assert!(race_layout(8, [u32::MAX, 2, 1], usize::MAX).is_err());
        assert!(race_layout(u32::MAX, [u32::MAX, 1, 1], usize::MAX).is_err());
    }

    #[test]
    fn mixed_reports_preserve_both_race_sites_and_validate_packet_extents() {
        let mut bytes = vec![0; 64 + 512];
        bytes[..192].copy_from_slice(&packet());
        put64(&mut bytes, 24, 512);
        put64(&mut bytes, 40, 320);
        let record = &mut bytes[192..384];
        put32(record, 0, 192);
        record[4..6].copy_from_slice(&64u16.to_le_bytes());
        record[6..8].copy_from_slice(&4u16.to_le_bytes());
        put32(record, 12, 1);
        put64(record, 16, 128);
        put64(record, 40, 1);
        put32(record, 64, 120);
        put32(record, 72, 1);
        put32(record, 76, 15);
        put32(record, 80, 2);
        put32(record, 84, 1);
        put32(record, 88, 2);
        put32(record, 92, 8);
        put64(record, 96, 7);
        put64(record, 104, 9);
        put64(record, 112, 4);
        put32(record, 148, 31);
        put32(record, 172, 63);
        let site = SanitizerSite {
            operation_kind: 17,
            source_name: Some("race.loom".into()),
            start: Some([2, 3]),
            end: Some([2, 8]),
            predicate_payload: vec![],
        };
        let sites = BTreeMap::from([(7, site.clone()), (9, site)]);
        let reports = decode(&bytes, 512, &sites).unwrap();
        drop(sites);
        assert_eq!(reports.reports.len(), 2);
        assert!(reports.reports[0].race.is_none());
        let report = &reports.reports[1];
        assert_eq!(report.check, SanitizerCheck::DataRace);
        assert!(report.site.is_some());
        let race = report.race.as_ref().unwrap();
        assert_eq!(race.prior_site_id, 9);
        assert!(race.prior_site.is_some());
        assert_eq!(race.current_access, SanitizerAccess::Read);
        assert_eq!(race.prior_access, SanitizerAccess::Write);
        assert!(race.current_atomic && race.prior_atomic);
        assert_eq!(race.current_workitem, [31, 0, 0]);
        assert_eq!(race.prior_workitem, [63, 0, 0]);
        assert!(race.current_workitem_linear && race.prior_workitem_linear);
        for (offset, value) in [(192, 128), (192 + 64, 64), (192 + 68, 1), (40, 256)] {
            let mut invalid = bytes.clone();
            put32(&mut invalid, offset, value);
            assert!(decode(&invalid, 512, &BTreeMap::new()).is_err(), "{offset}");
        }
    }

    fn packet() -> Vec<u8> {
        let mut bytes = vec![0; 192];
        put32(&mut bytes, 0, 64);
        put64(&mut bytes, 24, 128);
        put64(&mut bytes, 40, 128);
        put64(&mut bytes, 48, 3);
        let record = &mut bytes[64..];
        put32(record, 0, 128);
        record[4..6].copy_from_slice(&64u16.to_le_bytes());
        record[6..8].copy_from_slice(&5u16.to_le_bytes());
        put32(record, 12, 1);
        put64(record, 40, 1);
        put32(record, 64, 64);
        put32(record, 72, 6);
        put64(record, 80, 7);
        put64(record, 88, 42);
        bytes
    }
    #[test]
    fn report_decoding_preserves_owned_metadata_and_drop_count() {
        let sites = BTreeMap::from([(
            7,
            SanitizerSite {
                operation_kind: 9,
                source_name: Some("check.loom".into()),
                start: Some([12, 3]),
                end: Some([12, 8]),
                predicate_payload: vec![4, 2],
            },
        )]);
        let reports = decode(&packet(), 128, &sites).unwrap();
        drop(sites);
        assert_eq!(reports.dropped, 3);
        assert_eq!(reports.reports[0].check, SanitizerCheck::Assertion);
        assert_eq!(reports.reports[0].operands, [42, 0]);
        assert_eq!(
            reports.reports[0]
                .site
                .as_ref()
                .unwrap()
                .source_name
                .as_deref(),
            Some("check.loom")
        );
    }
    #[test]
    fn rejects_truncated_unpublished_and_overwritten_records() {
        let good = packet();
        for length in 0..good.len() {
            assert!(decode(&good[..length], 128, &BTreeMap::new()).is_err());
        }
        for (offset, value) in [(40, 256), (64, 64), (76, 0), (80, 128), (104, 2), (132, 1)] {
            let mut invalid = good.clone();
            put64(&mut invalid, offset, value);
            assert!(
                decode(&invalid, 128, &BTreeMap::new()).is_err(),
                "offset {offset}"
            );
        }
    }
    #[test]
    fn site_table_rejects_invalid_ranges_and_owns_source() {
        let mut bytes = vec![0; 88];
        put32(&mut bytes, 0, 0x5449534c);
        bytes[4] = 1;
        bytes[5] = 32;
        bytes[6..8].copy_from_slice(&48u16.to_le_bytes());
        put32(&mut bytes, 8, 1);
        put32(&mut bytes, 12, 80);
        put32(&mut bytes, 16, 6);
        put32(&mut bytes, 20, 86);
        put32(&mut bytes, 24, 2);
        put32(&mut bytes, 32, 7);
        put32(&mut bytes, 40, 3);
        put32(&mut bytes, 48, 2);
        put32(&mut bytes, 56, 6);
        bytes[80..].copy_from_slice(b"a.loom\x04\x02");
        let sites = parse_sites(&bytes).unwrap();
        assert_eq!(sites[&7].source_name.as_deref(), Some("a.loom"));
        assert_eq!(sites[&7].predicate_payload, [4, 2]);
        for length in 1..bytes.len() {
            assert!(parse_sites(&bytes[..length]).is_err());
        }
        put32(&mut bytes, 44, u32::MAX);
        assert!(parse_sites(&bytes).is_err());
    }
}
