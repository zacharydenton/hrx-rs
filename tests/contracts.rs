//! CPU tests for public argument, error and FFI contracts.
#![cfg(feature = "ffi")]
use hrx::{Constants, ffi};
#[test]
fn foreign_spans_and_error_buffers() {
    assert!(ffi::check_span::<u32>(std::ptr::null(), 1).is_err());
    assert!(ffi::check_span::<u32>(std::ptr::dangling::<u8>().cast(), 1).is_err());
    assert!(ffi::check_span::<u64>(std::ptr::dangling::<u64>(), usize::MAX).is_err());
    assert!(ffi::check_span::<u8>((usize::MAX - 1) as *const _, 4).is_err());
    assert!(ffi::check_span::<u32>(std::ptr::null(), 0).is_ok());
    let mut error = [0i8; 5];
    unsafe {
        ffi::report(error.as_mut_ptr(), error.len(), "a\0éé");
    }
    assert_eq!(
        unsafe { std::ffi::CStr::from_ptr(error.as_ptr()) }
            .to_str()
            .unwrap(),
        "a?é"
    );
    let status = unsafe { ffi::boundary(error.as_mut_ptr(), error.len(), 1, || Ok(())) };
    assert_eq!(status, 0);
    assert_eq!(error[0], 0);
    let status =
        unsafe { hrx::ffi::boundary(error.as_mut_ptr(), error.len(), 7, || panic!("boom")) };
    assert_eq!(status, 7);
}
#[test]
fn mixed_constants_and_overflow() {
    let mut args = Constants::new();
    args.push(1u32)
        .unwrap()
        .push(2u64)
        .unwrap()
        .push(3f32)
        .unwrap();
    assert_eq!(args.as_bytes().len(), 16);
    assert_eq!(&args.as_bytes()[4..12], &2u64.to_le_bytes());
    for _ in 0..30 {
        args.push(0u64).unwrap();
    }
    assert!(args.push(0u32).is_err());
    assert_eq!(args.as_bytes().len(), 256);
}
#[test]
fn a_panicking_handle_stays_poisoned() {
    let handle = ffi::Handle::new(1);
    let result: Result<(), ffi::Failure> = ffi::catch(
        || {
            let mut value = handle.lock().unwrap();
            *value = 9;
            panic!("interrupted update");
        },
        1,
    );
    assert!(result.is_err());
    assert!(handle.lock().is_err());
}
#[cfg(feature = "compat")]
#[test]
fn reset_direct_arguments_clears_old_padding() {
    let mut args = hrx::compat::Args::new();
    args.raw(&[0xff; 32]).unwrap();
    args.raw(&[1; 4]).unwrap();
    args.i64(0);
    assert_eq!(&args.as_bytes()[4..8], &[0; 4]);
    args.clear();
    assert!(args.as_bytes().is_empty());
    args.i32(3).i64(4);
    assert_eq!(&args.as_bytes()[4..8], &[0; 4]);
}

#[test]
fn errors_preserve_io_kinds_and_sources() {
    use std::error::Error as _;
    let error = hrx::Error::from(std::io::Error::new(
        std::io::ErrorKind::PermissionDenied,
        "denied",
    ));
    assert!(
        matches!(&error, hrx::Error::Io(e) if e.kind() == std::io::ErrorKind::PermissionDenied)
    );
    let error = error.context("opening bundle");
    assert_eq!(error.to_string(), "opening bundle: denied");
    let source = error.source().unwrap().source().unwrap();
    assert_eq!(
        source.downcast_ref::<std::io::Error>().unwrap().kind(),
        std::io::ErrorKind::PermissionDenied
    );
}
