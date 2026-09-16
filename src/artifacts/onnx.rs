//! Strict ONNX protobuf loading and graph/tensor inspection.

use crate::{Error, Result};
use onnx_protobuf::{AttributeProto, GraphProto, Message, ModelProto, NodeProto, TensorProto};
use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

/// A parsed ONNX model with initializer and edge indexes.
pub struct Model {
    name: String,
    inputs: Vec<String>,
    input_shapes: HashMap<String, Vec<Option<usize>>>,
    outputs: Vec<String>,
    opsets: Vec<(String, i64)>,
    nodes: Vec<Node>,
    initializers: HashMap<String, Tensor>,
    initializer_order: Vec<String>,
    producers: HashMap<String, usize>,
    consumers: HashMap<String, Vec<usize>>,
}

impl Model {
    /// Read and parse an ONNX protobuf file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)
            .map_err(|error| Error::Io(error).context(format!("reading {}", path.display())))?;
        Self::from_bytes(&bytes)
            .map_err(|error| error.context(format!("parsing {}", path.display())))
    }

    /// Parse an ONNX protobuf document and build graph indexes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut protobuf = ModelProto::parse_from_bytes(bytes)
            .map_err(|error| Error::Message(format!("invalid ONNX protobuf: {error}")))?;
        let opsets = protobuf
            .opset_import
            .iter()
            .map(|opset| (opset.domain.clone(), opset.version))
            .collect();
        let graph = protobuf
            .graph
            .take()
            .ok_or_else(|| Error::Message("missing ONNX graph".into()))?;
        Self::from_graph(graph, opsets)
    }

    fn from_graph(mut graph: GraphProto, opsets: Vec<(String, i64)>) -> Result<Self> {
        let mut initializers = HashMap::new();
        let mut initializer_order = Vec::new();
        for protobuf in std::mem::take(&mut graph.initializer) {
            if protobuf.name.is_empty() {
                return Err(Error::Message("unnamed ONNX initializer".into()));
            }
            let name = protobuf.name.clone();
            if initializers
                .insert(name.clone(), Tensor { protobuf })
                .is_some()
            {
                return Err(Error::Message(format!("duplicate ONNX initializer {name}")));
            }
            initializer_order.push(name);
        }
        let nodes = std::mem::take(&mut graph.node)
            .into_iter()
            .map(|protobuf| Node { protobuf })
            .collect::<Vec<_>>();
        let mut producers = HashMap::new();
        let mut consumers = HashMap::<String, Vec<usize>>::new();
        for (index, node) in nodes.iter().enumerate() {
            for output in node.outputs().iter().filter(|name| !name.is_empty()) {
                if producers.insert(output.clone(), index).is_some() {
                    return Err(Error::Message(format!("duplicate ONNX value {output}")));
                }
            }
            for input in node.inputs().iter().filter(|name| !name.is_empty()) {
                consumers.entry(input.clone()).or_default().push(index);
            }
        }
        let mut inputs = Vec::with_capacity(graph.input.len());
        let mut input_shapes = HashMap::new();
        for value in graph.input {
            let shape = value
                .type_
                .as_ref()
                .map(|kind| kind.tensor_type())
                .and_then(|tensor| tensor.shape.as_ref())
                .map(|shape| {
                    shape
                        .dim
                        .iter()
                        .map(|dimension| {
                            usize::try_from(dimension.dim_value())
                                .ok()
                                .filter(|&value| value > 0)
                        })
                        .collect()
                })
                .unwrap_or_default();
            input_shapes.insert(value.name.clone(), shape);
            inputs.push(value.name);
        }
        Ok(Self {
            name: graph.name,
            inputs,
            input_shapes,
            outputs: graph.output.into_iter().map(|value| value.name).collect(),
            opsets,
            nodes,
            initializers,
            initializer_order,
            producers,
            consumers,
        })
    }

    /// Graph name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Nodes in serialized order.
    #[must_use]
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// Model input names.
    #[must_use]
    pub fn inputs(&self) -> &[String] {
        &self.inputs
    }

    /// Declared dimensions for a model input. Symbolic dimensions are `None`.
    pub fn input_shape(&self, name: &str) -> Result<&[Option<usize>]> {
        self.input_shapes
            .get(name)
            .map(Vec::as_slice)
            .ok_or_else(|| Error::Message(format!("missing ONNX input {name}")))
    }

    /// Imported operator-set domains and versions in serialized order.
    #[must_use]
    pub fn opsets(&self) -> &[(String, i64)] {
        &self.opsets
    }

    /// Model output names.
    #[must_use]
    pub fn outputs(&self) -> &[String] {
        &self.outputs
    }

    /// Look up an initializer.
    pub fn initializer(&self, name: &str) -> Result<&Tensor> {
        self.initializers
            .get(name)
            .ok_or_else(|| Error::Message(format!("missing ONNX initializer {name}")))
    }

    /// Initializers in serialized order.
    pub fn initializers(&self) -> impl Iterator<Item = (&str, &Tensor)> {
        self.initializer_order.iter().map(|name| {
            (
                name.as_str(),
                self.initializers
                    .get(name)
                    .expect("initializer order and index agree"),
            )
        })
    }

    /// Node producing a value, when it is not a graph input or initializer.
    #[must_use]
    pub fn producer(&self, value: &str) -> Option<&Node> {
        self.producers.get(value).map(|&index| &self.nodes[index])
    }

    /// Nodes consuming a value.
    pub fn consumers<'a>(&'a self, value: &'a str) -> impl Iterator<Item = &'a Node> + 'a {
        self.consumers
            .get(value)
            .into_iter()
            .flatten()
            .map(|&index| &self.nodes[index])
    }
}

/// One ONNX graph node with checked attribute accessors.
pub struct Node {
    protobuf: NodeProto,
}

impl Node {
    /// Optional diagnostic name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.protobuf.name
    }

    /// Operator identifier.
    #[must_use]
    pub fn op_type(&self) -> &str {
        &self.protobuf.op_type
    }

    /// Operator domain; empty means the standard ONNX domain.
    #[must_use]
    pub fn domain(&self) -> &str {
        &self.protobuf.domain
    }

    /// Input value names.
    #[must_use]
    pub fn inputs(&self) -> &[String] {
        &self.protobuf.input
    }

    /// Output value names.
    #[must_use]
    pub fn outputs(&self) -> &[String] {
        &self.protobuf.output
    }

    /// Require exactly one nonempty output and return it.
    pub fn output(&self) -> Result<&str> {
        match self.outputs() {
            [output] if !output.is_empty() => Ok(output),
            _ => Err(Error::Message(format!(
                "{} must have one output",
                self.op_type()
            ))),
        }
    }

    /// Whether an attribute is present.
    #[must_use]
    pub fn has(&self, name: &str) -> bool {
        self.attribute(name).is_some()
    }

    /// Require that all attributes are allowed and unique.
    pub fn validate_attributes(&self, allowed: &[&str]) -> Result<()> {
        let mut names = HashSet::new();
        if self.protobuf.attribute.iter().any(|attribute| {
            !allowed.contains(&attribute.name.as_str()) || !names.insert(attribute.name.as_str())
        }) {
            return Err(Error::Message(format!(
                "{} has an unknown or duplicate attribute",
                self.op_type()
            )));
        }
        Ok(())
    }

    /// Integer attribute or default.
    #[must_use]
    pub fn integer(&self, name: &str, default: i64) -> i64 {
        self.attribute(name)
            .map_or(default, |attribute| attribute.i)
    }

    /// Float attribute or default.
    #[must_use]
    pub fn float(&self, name: &str, default: f64) -> f64 {
        self.attribute(name)
            .map_or(default, |attribute| f64::from(attribute.f))
    }

    /// Integer-list attribute or default.
    #[must_use]
    pub fn integers(&self, name: &str, default: &[i64]) -> Vec<i64> {
        self.attribute(name)
            .map_or_else(|| default.to_vec(), |attribute| attribute.ints.clone())
    }

    /// UTF-8 text attribute or default. Invalid bytes are rejected.
    pub fn text(&self, name: &str, default: &str) -> Result<String> {
        match self.attribute(name) {
            None => Ok(default.to_owned()),
            Some(attribute) => String::from_utf8(attribute.s.clone())
                .map_err(|_| Error::Message(format!("{name} is not UTF-8"))),
        }
    }

    /// Embedded tensor attribute.
    pub fn tensor(&self, name: &str) -> Result<TensorView<'_>> {
        self.attribute(name)
            .and_then(|attribute| attribute.t.as_ref())
            .map(|protobuf| TensorView { protobuf })
            .ok_or_else(|| Error::Message(format!("missing tensor attribute {name}")))
    }

    fn attribute(&self, name: &str) -> Option<&AttributeProto> {
        self.protobuf
            .attribute
            .iter()
            .find(|attribute| attribute.name == name)
    }
}

/// One embedded ONNX initializer.
pub struct Tensor {
    protobuf: TensorProto,
}

impl Tensor {
    /// Borrow this initializer as a tensor view.
    #[must_use]
    pub fn view(&self) -> TensorView<'_> {
        TensorView {
            protobuf: &self.protobuf,
        }
    }
    /// Tensor name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.protobuf.name
    }

    /// Positive dimensions converted to `usize`.
    pub fn shape(&self) -> Result<Vec<usize>> {
        self.protobuf
            .dims
            .iter()
            .map(|&dimension| {
                usize::try_from(dimension)
                    .ok()
                    .filter(|&dimension| dimension > 0)
                    .ok_or_else(|| {
                        Error::Message(format!("{} has an invalid dimension", self.name()))
                    })
            })
            .collect()
    }

    /// Checked element count.
    pub fn count(&self) -> Result<usize> {
        self.shape()?
            .into_iter()
            .try_fold(1usize, |count, dimension| {
                count.checked_mul(dimension).ok_or_else(|| {
                    Error::Message(format!("{} element count overflow", self.name()))
                })
            })
    }

    /// Decode an embedded float32 tensor.
    pub fn f32s(&self) -> Result<Vec<f32>> {
        if self.protobuf.data_type != 1 || self.protobuf.data_location.value() != 0 {
            return Err(Error::Message(format!(
                "{} is not embedded float32",
                self.name()
            )));
        }
        let count = self.count()?;
        let values = if self.protobuf.raw_data.is_empty() {
            self.protobuf.float_data.clone()
        } else {
            let bytes = count
                .checked_mul(4)
                .ok_or_else(|| Error::Message("ONNX tensor byte count overflow".into()))?;
            if self.protobuf.raw_data.len() != bytes {
                return Err(Error::Message(format!("{} is truncated", self.name())));
            }
            self.protobuf
                .raw_data
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four bytes")))
                .collect()
        };
        if values.len() != count || values.iter().any(|value| !value.is_finite()) {
            return Err(Error::Message(format!(
                "{} has invalid float values",
                self.name()
            )));
        }
        Ok(values)
    }

    /// Decode an embedded float32 tensor and widen its values to `f64`.
    pub fn f64s(&self) -> Result<Vec<f64>> {
        Ok(self.f32s()?.into_iter().map(f64::from).collect())
    }

    /// Decode an embedded int64 tensor.
    pub fn i64s(&self) -> Result<Vec<i64>> {
        if self.protobuf.data_type != 7 || self.protobuf.data_location.value() != 0 {
            return Err(Error::Message(format!(
                "{} is not embedded int64",
                self.name()
            )));
        }
        let count = self.count()?;
        let values = if self.protobuf.raw_data.is_empty() {
            self.protobuf.int64_data.clone()
        } else {
            let bytes = count
                .checked_mul(8)
                .ok_or_else(|| Error::Message("ONNX tensor byte count overflow".into()))?;
            if self.protobuf.raw_data.len() != bytes {
                return Err(Error::Message(format!("{} is truncated", self.name())));
            }
            self.protobuf
                .raw_data
                .chunks_exact(8)
                .map(|bytes| i64::from_le_bytes(bytes.try_into().expect("eight bytes")))
                .collect()
        };
        if values.len() != count {
            return Err(Error::Message(format!("{} is truncated", self.name())));
        }
        Ok(values)
    }
}

/// A borrowed ONNX tensor, including tensor-valued node attributes.
#[derive(Clone, Copy)]
pub struct TensorView<'a> {
    protobuf: &'a TensorProto,
}

impl TensorView<'_> {
    /// Tensor name, which may be empty for an attribute.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.protobuf.name
    }

    /// Positive dimensions converted to `usize`.
    pub fn shape(&self) -> Result<Vec<usize>> {
        self.protobuf
            .dims
            .iter()
            .map(|&dimension| {
                usize::try_from(dimension)
                    .ok()
                    .filter(|&dimension| dimension > 0)
                    .ok_or_else(|| {
                        Error::Message(format!("{} has an invalid dimension", self.name()))
                    })
            })
            .collect()
    }

    /// Checked element count.
    pub fn count(&self) -> Result<usize> {
        self.shape()?
            .into_iter()
            .try_fold(1usize, |count, dimension| {
                count.checked_mul(dimension).ok_or_else(|| {
                    Error::Message(format!("{} element count overflow", self.name()))
                })
            })
    }

    /// Decode an embedded float32 tensor.
    pub fn f32s(&self) -> Result<Vec<f32>> {
        if self.protobuf.data_type != 1 || self.protobuf.data_location.value() != 0 {
            return Err(Error::Message(format!(
                "{} is not embedded float32",
                self.name()
            )));
        }
        let count = self.count()?;
        let values = if self.protobuf.raw_data.is_empty() {
            self.protobuf.float_data.clone()
        } else {
            let bytes = count
                .checked_mul(4)
                .ok_or_else(|| Error::Message("ONNX tensor byte count overflow".into()))?;
            if self.protobuf.raw_data.len() != bytes {
                return Err(Error::Message(format!("{} is truncated", self.name())));
            }
            self.protobuf
                .raw_data
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four bytes")))
                .collect()
        };
        if values.len() != count || values.iter().any(|value| !value.is_finite()) {
            return Err(Error::Message(format!(
                "{} has invalid float values",
                self.name()
            )));
        }
        Ok(values)
    }

    /// Decode an embedded int64 tensor.
    pub fn i64s(&self) -> Result<Vec<i64>> {
        if self.protobuf.data_type != 7 || self.protobuf.data_location.value() != 0 {
            return Err(Error::Message(format!(
                "{} is not embedded int64",
                self.name()
            )));
        }
        let count = self.count()?;
        let values = if self.protobuf.raw_data.is_empty() {
            self.protobuf.int64_data.clone()
        } else {
            let bytes = count
                .checked_mul(8)
                .ok_or_else(|| Error::Message("ONNX tensor byte count overflow".into()))?;
            if self.protobuf.raw_data.len() != bytes {
                return Err(Error::Message(format!("{} is truncated", self.name())));
            }
            self.protobuf
                .raw_data
                .chunks_exact(8)
                .map(|bytes| i64::from_le_bytes(bytes.try_into().expect("eight bytes")))
                .collect()
        };
        if values.len() != count {
            return Err(Error::Message(format!("{} is truncated", self.name())));
        }
        Ok(values)
    }
}

/// Resolve a possibly negative axis against a rank.
pub fn axis(value: i64, rank: usize) -> Result<usize> {
    let rank_i64 = i64::try_from(rank).map_err(|_| Error::Message("rank overflow".into()))?;
    let resolved = if value < 0 { value + rank_i64 } else { value };
    usize::try_from(resolved)
        .ok()
        .filter(|&axis| axis < rank)
        .ok_or_else(|| Error::Message(format!("axis {value} is invalid for rank {rank}")))
}

/// Row-major element strides.
pub fn strides(shape: &[usize]) -> Result<Vec<usize>> {
    let mut stride = 1usize;
    let mut output = vec![0; shape.len()];
    for (index, &dimension) in shape.iter().enumerate().rev() {
        output[index] = stride;
        stride = stride
            .checked_mul(dimension)
            .ok_or_else(|| Error::Message("shape stride overflow".into()))?;
    }
    Ok(output)
}

/// Standard right-aligned multidirectional broadcast shape.
pub fn broadcast(left: &[usize], right: &[usize]) -> Result<Vec<usize>> {
    let rank = left.len().max(right.len());
    (0..rank)
        .map(|index| {
            let from_right = rank - index;
            let a = left
                .len()
                .checked_sub(from_right)
                .and_then(|position| left.get(position))
                .copied()
                .unwrap_or(1);
            let b = right
                .len()
                .checked_sub(from_right)
                .and_then(|position| right.get(position))
                .copied()
                .unwrap_or(1);
            if a == b || a == 1 || b == 1 {
                Ok(a.max(b))
            } else {
                Err(Error::Message(format!(
                    "cannot broadcast {left:?} and {right:?}"
                )))
            }
        })
        .collect()
}

/// Map a flat broadcast-output index into one input's flat row-major index.
pub fn broadcast_index(index: usize, output: &[usize], input: &[usize]) -> Result<usize> {
    if input.len() > output.len() {
        return Err(Error::Message(
            "broadcast input rank exceeds output rank".into(),
        ));
    }
    let count = output.iter().try_fold(1usize, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or_else(|| Error::Message("broadcast output size overflow".into()))
    })?;
    if index >= count {
        return Err(Error::Message("broadcast index exceeds output".into()));
    }
    let output_strides = strides(output)?;
    let input_strides = strides(input)?;
    let offset = output.len() - input.len();
    let mut result = 0usize;
    for dimension in 0..input.len() {
        let output_dimension = output[offset + dimension];
        let input_dimension = input[dimension];
        if input_dimension != 1 && input_dimension != output_dimension {
            return Err(Error::Message("input is not broadcast-compatible".into()));
        }
        let coordinate = (index / output_strides[offset + dimension]) % output_dimension;
        if input_dimension != 1 {
            result = result
                .checked_add(
                    coordinate
                        .checked_mul(input_strides[dimension])
                        .ok_or_else(|| Error::Message("broadcast index overflow".into()))?,
                )
                .ok_or_else(|| Error::Message("broadcast index overflow".into()))?;
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axes_and_broadcasts_are_checked() {
        assert_eq!(axis(-1, 4).unwrap(), 3);
        assert!(axis(4, 4).is_err());
        assert_eq!(broadcast(&[2, 1, 4], &[3, 4]).unwrap(), [2, 3, 4]);
        assert!(broadcast(&[2], &[3]).is_err());
        assert_eq!(broadcast_index(23, &[2, 3, 4], &[3, 1]).unwrap(), 2);
        assert!(broadcast_index(24, &[2, 3, 4], &[3, 1]).is_err());
    }
}
