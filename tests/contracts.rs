//! CPU tests for scalar packing and error contracts.
use hrx::Constants;
#[test]
fn mixed_constants_and_overflow() {
    let mut args = Constants::new();
    args.push(1u32).unwrap();
    args.push(2u64).unwrap();
    args.push(3f32).unwrap();
    assert_eq!(args.as_bytes().len(), 16);
    assert_eq!(&args.as_bytes()[4..12], &2u64.to_le_bytes());
    for _ in 0..30 {
        args.push(0u64).unwrap();
    }
    assert!(args.push(0u32).is_err());
    assert_eq!(args.as_bytes().len(), 256);
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
