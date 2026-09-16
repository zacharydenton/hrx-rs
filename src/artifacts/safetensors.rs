//! Indexed owned and memory-mapped SafeTensors files.

use crate::{Error, Result};
use memmap2::Mmap;
use safetensors::tensor::{Dtype as NativeDType, Metadata, SafeTensors};
use std::{
    collections::BTreeMap,
    fs::File,
    path::{Path, PathBuf},
};

/// SafeTensors element type, independent of the parser crate's public API.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[allow(missing_docs)]
#[allow(non_camel_case_types)]
pub enum DType {
    BOOL,
    F4,
    F6_E2M3,
    F6_E3M2,
    U8,
    I8,
    F8_E5M2,
    F8_E4M3,
    F8_E8M0,
    F8_E4M3FNUZ,
    F8_E5M2FNUZ,
    I16,
    U16,
    F16,
    BF16,
    I32,
    U32,
    F32,
    C64,
    F64,
    I64,
    U64,
    /// A type introduced by a newer `safetensors` parser.
    Unknown,
}

impl From<NativeDType> for DType {
    fn from(value: NativeDType) -> Self {
        match value {
            NativeDType::BOOL => Self::BOOL,
            NativeDType::F4 => Self::F4,
            NativeDType::F6_E2M3 => Self::F6_E2M3,
            NativeDType::F6_E3M2 => Self::F6_E3M2,
            NativeDType::U8 => Self::U8,
            NativeDType::I8 => Self::I8,
            NativeDType::F8_E5M2 => Self::F8_E5M2,
            NativeDType::F8_E4M3 => Self::F8_E4M3,
            NativeDType::F8_E8M0 => Self::F8_E8M0,
            NativeDType::F8_E4M3FNUZ => Self::F8_E4M3FNUZ,
            NativeDType::F8_E5M2FNUZ => Self::F8_E5M2FNUZ,
            NativeDType::I16 => Self::I16,
            NativeDType::U16 => Self::U16,
            NativeDType::F16 => Self::F16,
            NativeDType::BF16 => Self::BF16,
            NativeDType::I32 => Self::I32,
            NativeDType::U32 => Self::U32,
            NativeDType::F32 => Self::F32,
            NativeDType::C64 => Self::C64,
            NativeDType::F64 => Self::F64,
            NativeDType::I64 => Self::I64,
            NativeDType::U64 => Self::U64,
            _ => Self::Unknown,
        }
    }
}

impl DType {
    /// Bytes per element for byte-addressable formats. Sub-byte formats return `None`.
    #[must_use]
    pub fn bytes(self) -> Option<usize> {
        match self {
            Self::F4 | Self::F6_E2M3 | Self::F6_E3M2 | Self::Unknown => None,
            Self::BOOL
            | Self::U8
            | Self::I8
            | Self::F8_E5M2
            | Self::F8_E4M3
            | Self::F8_E8M0
            | Self::F8_E4M3FNUZ
            | Self::F8_E5M2FNUZ => Some(1),
            Self::I16 | Self::U16 | Self::F16 | Self::BF16 => Some(2),
            Self::I32 | Self::U32 | Self::F32 => Some(4),
            Self::C64 | Self::F64 | Self::I64 | Self::U64 => Some(8),
        }
    }
}

/// One tensor's metadata and placement in the SafeTensors data block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    /// Element type.
    pub dtype: DType,
    /// Tensor dimensions.
    pub shape: Vec<usize>,
    /// Byte offset from the start of the data block.
    pub offset: usize,
    /// Tensor byte length.
    pub bytes: usize,
}

impl Entry {
    /// Number of elements, treating a scalar as one element.
    pub fn elements(&self) -> Result<usize> {
        self.shape.iter().try_fold(1usize, |count, &dimension| {
            count
                .checked_mul(dimension)
                .ok_or_else(|| Error::Message("SafeTensors element count overflow".into()))
        })
    }

    /// First dimension, treating a scalar as one row.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.shape.first().copied().unwrap_or(1)
    }

    /// Bytes per first-dimension row.
    pub fn row_bytes(&self) -> Result<usize> {
        let rows = self.rows();
        if rows == 0 || !self.bytes.is_multiple_of(rows) {
            return Err(Error::Message(
                "tensor bytes are not a whole number of rows".into(),
            ));
        }
        Ok(self.bytes / rows)
    }
}

enum Storage {
    Owned(Vec<u8>),
    Mapped(Mmap),
}

impl Storage {
    fn bytes(&self) -> &[u8] {
        match self {
            Self::Owned(bytes) => bytes,
            Self::Mapped(map) => map,
        }
    }
}

/// An indexed SafeTensors file.
pub struct FileView {
    path: PathBuf,
    storage: Storage,
    data: usize,
    entries: BTreeMap<String, Entry>,
}

/// A tensor borrowing its file storage.
#[derive(Clone, Copy, Debug)]
pub struct Tensor<'a> {
    /// Element type.
    pub dtype: DType,
    /// Dimensions.
    pub shape: &'a [usize],
    /// Original encoded bytes.
    pub bytes: &'a [u8],
}

impl FileView {
    /// Read a file into owned memory and index its tensors.
    pub fn read(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let bytes = std::fs::read(&path)
            .map_err(|error| Error::Io(error).context(format!("reading {}", path.display())))?;
        Self::from_storage(path, Storage::Owned(bytes))
    }

    /// Map and index a file without copying its tensor data.
    ///
    /// # Safety
    ///
    /// The file must not be modified or truncated until the returned value is dropped.
    pub unsafe fn map(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)
            .map_err(|error| Error::Io(error).context(format!("opening {}", path.display())))?;
        let map = unsafe { Mmap::map(&file) }
            .map_err(|error| Error::Io(error).context(format!("mapping {}", path.display())))?;
        Self::from_storage(path, Storage::Mapped(map))
    }

    fn from_storage(path: PathBuf, storage: Storage) -> Result<Self> {
        let bytes = storage.bytes();
        let (header, metadata): (usize, Metadata) =
            SafeTensors::read_metadata(bytes).map_err(|error| {
                Error::Message(format!("{} is not SafeTensors: {error}", path.display()))
            })?;
        let data = 8usize
            .checked_add(header)
            .ok_or_else(|| Error::Message("SafeTensors header size overflow".into()))?;
        let data_bytes = bytes
            .len()
            .checked_sub(data)
            .ok_or_else(|| Error::Message("SafeTensors header exceeds the file".into()))?;
        let mut entries = BTreeMap::new();
        for (name, info) in metadata.tensors() {
            let (start, end) = info.data_offsets;
            if end < start || end > data_bytes {
                return Err(Error::Message(format!(
                    "tensor {name} exceeds the SafeTensors data block"
                )));
            }
            let entry = Entry {
                dtype: info.dtype.into(),
                shape: info.shape.clone(),
                offset: start,
                bytes: end - start,
            };
            if entries.insert(name.clone(), entry).is_some() {
                return Err(Error::Message(format!("duplicate tensor {name}")));
            }
        }
        Ok(Self {
            path,
            storage,
            data,
            entries,
        })
    }

    /// Source path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Indexed entries in lexical order.
    #[must_use]
    pub fn entries(&self) -> &BTreeMap<String, Entry> {
        &self.entries
    }

    /// Whether a tensor exists.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    /// Tensor names in lexical order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    /// Look up a tensor.
    pub fn get(&self, name: &str) -> Result<Tensor<'_>> {
        let entry = self.entries.get(name).ok_or_else(|| {
            Error::Message(format!("missing tensor {name} in {}", self.path.display()))
        })?;
        let start = self.data + entry.offset;
        let end = start + entry.bytes;
        Ok(Tensor {
            dtype: entry.dtype,
            shape: &entry.shape,
            bytes: &self.storage.bytes()[start..end],
        })
    }

    /// Borrow the data range described by an indexed entry.
    pub fn bytes(&self, entry: &Entry) -> Result<&[u8]> {
        let start = self
            .data
            .checked_add(entry.offset)
            .ok_or_else(|| Error::Message("SafeTensors offset overflow".into()))?;
        let end = start
            .checked_add(entry.bytes)
            .ok_or_else(|| Error::Message("SafeTensors range overflow".into()))?;
        self.storage
            .bytes()
            .get(start..end)
            .ok_or_else(|| Error::Message("SafeTensors entry exceeds the file".into()))
    }

    /// Look up and validate a tensor. Negative expected dimensions are wildcards.
    pub fn get_checked(&self, name: &str, dtype: DType, shape: &[i64]) -> Result<Tensor<'_>> {
        let tensor = self.get(name)?;
        let shape_matches = tensor.shape.len() == shape.len()
            && tensor.shape.iter().zip(shape).all(|(&actual, &expected)| {
                expected < 0 || usize::try_from(expected) == Ok(actual)
            });
        if tensor.dtype != dtype || !shape_matches {
            return Err(Error::Message(format!(
                "{name} is {:?} {:?}, expected {dtype:?} {shape:?} in {}",
                tensor.dtype,
                tensor.shape,
                self.path.display()
            )));
        }
        Ok(tensor)
    }

    /// Ask the operating system to prefetch the mapped pages backing a tensor.
    pub fn will_need(&self, tensor: Tensor<'_>) {
        self.advise(tensor.bytes, libc::MADV_WILLNEED, true);
    }

    /// Release complete mapped pages after a tensor has been consumed.
    pub fn done_with(&self, tensor: Tensor<'_>) {
        self.advise(tensor.bytes, libc::MADV_DONTNEED, false);
    }

    /// Ask the operating system to prefetch a checked range borrowed from this file.
    pub fn will_need_bytes(&self, bytes: &[u8]) {
        self.advise(bytes, libc::MADV_WILLNEED, true);
    }

    /// Release complete mapped pages in a checked range borrowed from this file.
    pub fn done_with_bytes(&self, bytes: &[u8]) {
        self.advise(bytes, libc::MADV_DONTNEED, false);
    }

    fn advise(&self, range: &[u8], how: i32, outward: bool) {
        let Storage::Mapped(map) = &self.storage else {
            return;
        };
        if range.is_empty() {
            return;
        }
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if page <= 0 {
            return;
        }
        let Some((begin, end)) = advise_pages(
            (map.as_ptr() as usize, map.len()),
            (range.as_ptr() as usize, range.len()),
            page as usize,
            outward,
        ) else {
            return;
        };
        let _ = unsafe { libc::madvise(begin as *mut libc::c_void, end - begin, how) };
    }
}

fn advise_pages(
    map: (usize, usize),
    range: (usize, usize),
    page: usize,
    outward: bool,
) -> Option<(usize, usize)> {
    let map_end = map.0.checked_add(map.1)?;
    let range_end = range.0.checked_add(range.1)?;
    if page == 0 || !page.is_power_of_two() || range.0 < map.0 || range_end > map_end {
        return None;
    }
    let mask = page - 1;
    let (begin, end) = if outward {
        (range.0 & !mask, range_end.checked_add(mask)? & !mask)
    } else {
        (range.0.checked_add(mask)? & !mask, range_end & !mask)
    };
    (end > begin).then_some((begin, end))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_advice_rounding_stays_inside_for_discard() {
        assert_eq!(
            advise_pages((0x1000, 0x4000), (0x1800, 0x2100), 0x1000, false),
            Some((0x2000, 0x3000))
        );
        assert!(advise_pages((0x1000, 0x1000), (0, 16), 0x1000, false).is_none());
    }
}
