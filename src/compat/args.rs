//! Loom's AMDGPU argument ABI: naturally aligned scalars in declaration order,
//! then one 8-byte device address per buffer operand. Padding bytes are zeroed.
use super::DevicePtr;

/// The dispatch limit is 256 bytes of direct arguments.
const CAPACITY: usize = 256;

#[derive(Clone, Debug)]
/// Naturally aligned direct arguments and tracked allocation addresses.
pub struct Args {
    bytes: [u8; CAPACITY],
    size: usize,
    pub(crate) pointers: [usize; 32],
    pub(crate) pointer_count: usize,
    pub(crate) opaque: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self::new()
    }
}

impl Args {
    /// Create empty direct arguments.
    pub const fn new() -> Self {
        Args {
            bytes: [0; CAPACITY],
            size: 0,
            pointers: [0; 32],
            pointer_count: 0,
            opaque: false,
        }
    }

    /// Reset bytes, pointer metadata and the raw-argument mode for reuse.
    pub fn clear(&mut self) {
        *self = Self::new();
    }

    fn push(&mut self, value: &[u8], align: usize) -> &mut Self {
        self.size = (self.size + align - 1) & !(align - 1);
        assert!(
            self.size + value.len() <= CAPACITY,
            "kernel argument overflow"
        );
        self.bytes[self.size..self.size + value.len()].copy_from_slice(value);
        self.size += value.len();
        self
    }

    /// Append a naturally aligned signed 32-bit argument.
    pub fn i32(&mut self, value: i32) -> &mut Self {
        self.push(&value.to_ne_bytes(), 4)
    }

    /// Append a naturally aligned unsigned 32-bit argument.
    pub fn u32(&mut self, value: u32) -> &mut Self {
        self.push(&value.to_ne_bytes(), 4)
    }

    /// Append a naturally aligned 32-bit floating-point argument.
    pub fn f32(&mut self, value: f32) -> &mut Self {
        self.push(&value.to_ne_bytes(), 4)
    }

    /// An 8-byte Loom index, which is what the attention kernels take for the
    /// token count.
    pub fn i64(&mut self, value: i64) -> &mut Self {
        self.push(&value.to_ne_bytes(), 8)
    }

    /// A device address, as the kernel's `buffer` operand.
    pub fn ptr(&mut self, value: DevicePtr) -> &mut Self {
        assert!(self.pointer_count < 32, "kernel argument overflow");
        self.pointers[self.pointer_count] = value.address();
        self.pointer_count += 1;
        self.push(&(value.address() as u64).to_ne_bytes(), 8)
    }

    /// A kernarg blob a caller already has, for the test bridge that passes
    /// one straight through from Python.
    pub fn raw(&mut self, bytes: &[u8]) -> Result<&mut Self, super::Error> {
        if bytes.len() > CAPACITY {
            return Err(super::Error::Message("kernel argument overflow".into()));
        }
        self.opaque = true;
        self.pointer_count = 0;
        self.bytes.fill(0);
        self.bytes[..bytes.len()].copy_from_slice(bytes);
        self.size = bytes.len();
        Ok(self)
    }

    /// The initialized direct-argument blob, including zero padding.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes[..self.size]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalars_pad_to_their_alignment_and_pointers_to_eight() {
        let mut args = Args::new();
        args.i32(1).ptr(DevicePtr::from_address(0x1000)).f32(2.0);
        // i32 at 0, four bytes of zero padding, the address at 8, the float at 16.
        let bytes = args.as_bytes();
        assert_eq!(bytes.len(), 20);
        assert_eq!(&bytes[0..4], &1i32.to_ne_bytes());
        assert_eq!(
            &bytes[4..8],
            &[0, 0, 0, 0],
            "padding must be zero, not indeterminate"
        );
        assert_eq!(&bytes[8..16], &0x1000u64.to_ne_bytes());
        assert_eq!(&bytes[16..20], &2.0f32.to_ne_bytes());
    }

    #[test]
    fn clearing_raw_arguments_restores_pointer_tracking() {
        let mut args = Args::new();
        args.raw(&[0xff; 16]).unwrap();
        args.clear();
        args.ptr(DevicePtr::from_address(0x1000));
        assert!(!args.opaque);
        assert_eq!(&args.pointers[..args.pointer_count], &[0x1000]);
    }

    #[test]
    fn an_empty_argument_list_is_empty() {
        assert!(Args::new().as_bytes().is_empty());
    }

    #[test]
    #[should_panic(expected = "kernel argument overflow")]
    fn past_the_dispatch_limit_is_a_bug_not_an_error() {
        let mut args = Args::new();
        for _ in 0..33 {
            args.ptr(DevicePtr::from_address(8));
        }
    }
}
