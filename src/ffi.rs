//! Reusable C ABI call bodies. Keep extern signatures and repr(C) declarations
//! explicit in each model so cbindgen can generate headers without expansion.
use crate::{Error, Result};
use std::{
    ffi::c_char,
    panic::{UnwindSafe, catch_unwind},
    sync::{
        Mutex, MutexGuard,
        atomic::{AtomicBool, Ordering},
    },
};

/// A model chooses its own stable status numbers, including cancellation.
#[derive(Debug, Clone)]
pub struct Failure {
    /// Application-defined nonzero ABI failure code.
    pub code: i32,
    /// Human-readable diagnostic copied to the caller's error buffer.
    pub message: String,
}
impl Failure {
    /// Create an ABI failure with an owned diagnostic.
    pub fn new(code: i32, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

/// Catch ordinary unwind panics. A panic=abort build cannot contain panics.
/// The panic payload is forgotten on failure: user-defined payload destructors
/// are allowed to panic too, and must not unwind across the ABI boundary.
pub fn catch<T>(
    body: impl FnOnce() -> std::result::Result<T, Failure> + UnwindSafe,
    panic_code: i32,
) -> std::result::Result<T, Failure> {
    match catch_unwind(body) {
        Ok(result) => result,
        Err(payload) => {
            let message = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_owned())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "native inference panicked".into());
            std::mem::forget(payload);
            Err(Failure::new(panic_code, message))
        }
    }
}

/// # Safety
/// Non-null `out` must point to `capacity` writable bytes, disjoint from message.
/// Interior NULs become `?`; truncation preserves UTF-8 and always terminates.
pub unsafe fn report(out: *mut c_char, capacity: usize, message: &str) {
    if out.is_null() || capacity == 0 {
        return;
    }
    let mut len = message.len().min(capacity - 1);
    while !message.is_char_boundary(len) {
        len -= 1;
    }
    for (i, b) in message.as_bytes()[..len].iter().enumerate() {
        unsafe {
            *out.add(i) = if *b == 0 { b'?' } else { *b } as c_char;
        }
    }
    unsafe {
        *out.add(len) = 0;
    }
}
/// # Safety
/// As [`report`]. The closure must not invalidate the error buffer.
pub unsafe fn boundary(
    out: *mut c_char,
    capacity: usize,
    panic_code: i32,
    body: impl FnOnce() -> std::result::Result<(), Failure> + UnwindSafe,
) -> i32 {
    unsafe {
        report(out, capacity, "");
    }
    match catch(body, panic_code) {
        Ok(()) => 0,
        Err(e) => {
            unsafe {
                report(out, capacity, &e.message);
            }
            e.code
        }
    }
}

/// Validate what can be checked without dereferencing foreign memory. Pointer
/// provenance, actual capacity and aliasing remain the foreign caller's contract.
pub fn check_span<T>(p: *const T, count: usize) -> Result<()> {
    if count == 0 {
        return Ok(());
    }
    let bytes = count
        .checked_mul(std::mem::size_of::<T>())
        .filter(|n| *n <= isize::MAX as usize)
        .ok_or_else(|| Error::Message("foreign buffer length overflows isize".into()))?;
    if p.is_null()
        || !(p as usize).is_multiple_of(std::mem::align_of::<T>())
        || (p as usize).checked_add(bytes).is_none()
    {
        return Err(Error::Message(
            "foreign buffer is null, misaligned or wraps the address space".into(),
        ));
    }
    Ok(())
}
/// # Safety
/// `p` names `count` initialized, readable elements in one allocation, valid for
/// `'a`. No mutation may overlap the returned borrow.
pub unsafe fn slice<'a, T>(p: *const T, count: usize) -> Result<&'a [T]> {
    check_span(p, count)?;
    if count == 0 {
        Ok(&[])
    } else {
        Ok(unsafe { std::slice::from_raw_parts(p, count) })
    }
}
/// # Safety
/// As [`slice`], with exclusive writable access for `'a`.
pub unsafe fn slice_mut<'a, T>(p: *mut T, count: usize) -> Result<&'a mut [T]> {
    check_span(p, count)?;
    if count == 0 {
        Ok(&mut [])
    } else {
        Ok(unsafe { std::slice::from_raw_parts_mut(p, count) })
    }
}

/// An opaque model handle's contents. Poisoning is permanent: a panic may leave
/// model state inconsistent, so subsequent calls must construct a fresh handle.
pub struct Handle<T>(Mutex<T>);
impl<T> Handle<T> {
    /// Wrap model state in a mutex that detects panics during mutation.
    pub fn new(value: T) -> Self {
        Self(Mutex::new(value))
    }
    /// Lock model state, reporting poison after a panicking operation.
    pub fn lock(&self) -> Result<MutexGuard<'_, T>> {
        self.0
            .lock()
            .map_err(|_| Error::Message("model session is poisoned; create a new session".into()))
    }
}
/// # Safety
/// The pointer must name a live object of this exact library's handle type for
/// `'a`, with no overlapping destruction. A handle must never cross model DSOs.
pub unsafe fn handle<'a, T>(pointer: *const T) -> Result<&'a T> {
    check_span(pointer, 1)?;
    Ok(unsafe { &*pointer })
}

#[derive(Default)]
/// A shared cancellation flag for cooperative model operations.
pub struct Cancellation(AtomicBool);
impl Cancellation {
    /// Request cancellation.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }
    /// Check whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

/// Only expands a call body; exported names and C declarations remain visible
/// to cbindgen. The caller supplies its ABI's status mapping.
#[macro_export]
macro_rules! ffi_call {
    ($out:expr, $capacity:expr, $panic_code:expr, $body:expr) => {
        $crate::ffi::boundary($out, $capacity, $panic_code, $body)
    };
}
