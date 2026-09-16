//! Checked tensor metadata and owned views in one coordinated runtime.

use crate::{Completion, Error, Result, Runtime, execution::BufferView};
use std::sync::Arc;
mod ops;
pub use ops::TensorOps;

/// Tensor element representation. Strides are measured in elements.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum DType {
    /// Unsigned byte.
    U8,
    /// Signed 32-bit integer.
    I32,
    /// Unsigned 32-bit integer.
    U32,
    /// Signed 64-bit integer.
    I64,
    /// IEEE half precision.
    F16,
    /// Brain floating point.
    BF16,
    /// IEEE single precision.
    F32,
}
impl DType {
    /// Bytes per element.
    pub const fn bytes(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::F16 | Self::BF16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 => 8,
        }
    }
}

/// Logical axis convention, independent of element strides.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize)]
pub enum Layout {
    /// Arbitrary-rank tensor with axes in caller-defined order.
    #[default]
    General,
    /// Batch, channels, height, width.
    Nchw,
    /// Batch, height, width, channels.
    Nhwc,
    /// Rows and columns.
    Rows,
}

/// Validated tensor shape and non-overlapping element strides.
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub struct TensorDesc {
    dtype: DType,
    shape: Vec<usize>,
    strides: Vec<usize>,
    layout: Layout,
    elements: usize,
    bytes: usize,
}
impl TensorDesc {
    /// Create a contiguous row-major tensor. An empty shape is one scalar;
    /// any zero dimension makes the tensor empty.
    pub fn new(dtype: DType, shape: impl Into<Vec<usize>>) -> Result<Self> {
        let shape = shape.into();
        if shape.contains(&0) {
            let strides = vec![1; shape.len()];
            return Self::strided(dtype, shape, strides);
        }
        let mut stride = 1usize;
        let mut strides = vec![0; shape.len()];
        for (index, &dimension) in shape.iter().enumerate().rev() {
            strides[index] = stride;
            stride = stride
                .checked_mul(dimension.max(1))
                .ok_or_else(|| Error::Message("tensor stride overflow".into()))?;
        }
        Self::strided(dtype, shape, strides)
    }

    /// Create a tensor with positive element strides. Overlapping writable
    /// layouts are rejected; slicing and permutations may leave gaps.
    pub fn strided(dtype: DType, shape: Vec<usize>, strides: Vec<usize>) -> Result<Self> {
        if shape.len() != strides.len() || strides.contains(&0) {
            return Err(Error::Message("invalid tensor strides".into()));
        }
        let empty = shape.contains(&0);
        let elements = if empty {
            0
        } else {
            shape
                .iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d))
                .ok_or_else(|| Error::Message("tensor element count overflow".into()))?
        };
        let mut extent = usize::from(!empty);
        if !empty {
            let mut axes = shape
                .iter()
                .copied()
                .zip(strides.iter().copied())
                .filter(|&(d, _)| d > 1)
                .collect::<Vec<_>>();
            axes.sort_unstable_by_key(|&(_, stride)| stride);
            for (dimension, stride) in axes {
                if stride < extent {
                    return Err(Error::Message("overlapping tensor strides".into()));
                }
                extent = (dimension - 1)
                    .checked_mul(stride)
                    .and_then(|n| n.checked_add(extent))
                    .ok_or_else(|| Error::Message("tensor extent overflow".into()))?;
            }
        }
        let bytes = extent
            .checked_mul(dtype.bytes())
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or_else(|| Error::Message("tensor byte extent overflow".into()))?;
        Ok(Self {
            dtype,
            shape,
            strides,
            layout: Layout::General,
            elements,
            bytes,
        })
    }

    /// Set an axis convention after validating its rank.
    pub fn with_layout(mut self, layout: Layout) -> Result<Self> {
        let valid = match layout {
            Layout::General => true,
            Layout::Rows => self.shape.len() == 2,
            Layout::Nchw | Layout::Nhwc => self.shape.len() == 4,
        };
        if !valid {
            return Err(Error::Message("tensor layout rank mismatch".into()));
        }
        self.layout = layout;
        Ok(self)
    }
    /// Element representation.
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    /// Logical dimensions.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    /// Element strides.
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }
    /// Axis convention.
    pub fn layout(&self) -> Layout {
        self.layout
    }
    /// Number of logical elements.
    pub fn elements(&self) -> usize {
        self.elements
    }
    /// Accessible byte extent including stride gaps.
    pub fn bytes(&self) -> usize {
        self.bytes
    }
    /// Whether the tensor contains no elements.
    pub fn is_empty(&self) -> bool {
        self.elements == 0
    }
    /// Whether elements occupy contiguous row-major storage.
    pub fn is_contiguous(&self) -> bool {
        if self.is_empty() {
            return true;
        }
        let mut stride = 1usize;
        for (&dimension, &actual) in self.shape.iter().zip(&self.strides).rev() {
            if dimension > 1 && actual != stride {
                return false;
            }
            let Some(next) = stride.checked_mul(dimension.max(1)) else {
                return false;
            };
            stride = next;
        }
        true
    }
}

/// An owned tensor view. Clones retain storage, producer completion, and any
/// inference slot lease. Empty tensors carry metadata without allocating.
#[derive(Clone)]
pub struct DeviceTensor {
    pub(crate) runtime: Runtime,
    pub(crate) desc: TensorDesc,
    pub(crate) view: Option<BufferView>,
    pub(crate) producer: Completion,
    pub(crate) lease: Option<Arc<dyn Send + Sync>>,
}
impl DeviceTensor {
    /// Validated shape, dtype, layout, and extent.
    pub fn desc(&self) -> &TensorDesc {
        &self.desc
    }
    /// Completion of the operation that produced these values.
    pub fn completion(&self) -> &Completion {
        &self.producer
    }
    /// Checked owned binding. Its lifetime also retains the inference slot.
    /// Empty tensors have no native binding.
    pub fn binding(&self) -> Option<BufferView> {
        self.view.clone().map(|view| match &self.lease {
            Some(lease) => view.retain(lease.clone()),
            None => view,
        })
    }
    /// Reinterpret a checked subregion, retaining its producer and slot lease.
    pub fn view(&self, byte_offset: usize, desc: TensorDesc) -> Result<Self> {
        let end = byte_offset
            .checked_add(desc.bytes())
            .filter(|&end| end <= self.desc.bytes())
            .ok_or_else(|| Error::Message("tensor view exceeds storage".into()))?;
        let origin = self.view.as_ref().map_or(0, BufferView::offset);
        if !(origin + byte_offset).is_multiple_of(desc.dtype().bytes()) {
            return Err(Error::Message("unaligned tensor view".into()));
        }
        let view = if desc.is_empty() {
            None
        } else {
            Some(
                self.view
                    .as_ref()
                    .ok_or_else(|| Error::Message("empty tensor has no storage".into()))?
                    .slice(byte_offset..end)?,
            )
        };
        Ok(Self {
            runtime: self.runtime.clone(),
            desc,
            view,
            producer: self.producer.clone(),
            lease: self.lease.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shapes_distinguish_empty_scalar_and_strided_storage() {
        let scalar = TensorDesc::new(DType::F32, vec![]).unwrap();
        assert_eq!((scalar.elements(), scalar.bytes()), (1, 4));
        let empty = TensorDesc::new(DType::F32, vec![2, 0, 3]).unwrap();
        assert_eq!((empty.elements(), empty.bytes()), (0, 0));
        let huge_empty = TensorDesc::new(DType::F32, vec![usize::MAX, 0, usize::MAX]).unwrap();
        assert!(huge_empty.is_empty() && huge_empty.is_contiguous());
        let padded = TensorDesc::strided(DType::F32, vec![2, 3], vec![4, 1]).unwrap();
        assert_eq!((padded.elements(), padded.bytes()), (6, 28));
        assert!(!padded.is_contiguous());
        assert!(TensorDesc::strided(DType::F32, vec![2, 3], vec![1, 1]).is_err());
        assert!(TensorDesc::new(DType::F32, vec![usize::MAX]).is_err());
        assert!(
            TensorDesc::new(DType::U8, vec![2, 3])
                .unwrap()
                .with_layout(Layout::Nchw)
                .is_err()
        );
        let transpose = TensorDesc::strided(DType::U8, vec![3, 2], vec![1, 3]).unwrap();
        assert_eq!(transpose.bytes(), 6);
    }
}
