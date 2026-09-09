#![cfg(feature = "compat")]
//! Initialization errors must cross Result-returning APIs without unwinding.
#[test]
fn initialization_failure_returns_errors() {
    if std::env::var_os("HRX_TEST_FAILED_INIT_CHILD").is_some() {
        let result = std::panic::catch_unwind(|| unsafe {
            hrx::compat::Kernel::load(std::path::Path::new("unused.hsaco"), "kernel")
        });
        assert!(result.is_ok(), "Result-returning load panicked");
        assert!(result.unwrap().is_err());
        return;
    }
    let directory = tempfile::tempdir().unwrap();
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "initialization_failure_returns_errors",
            "--nocapture",
        ])
        .env("HRX_TEST_FAILED_INIT_CHILD", "1")
        .env("HRX_RUNTIME_DIR", directory.path().join("missing-runtime"))
        .env("HRX_OFFLINE", "1")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
