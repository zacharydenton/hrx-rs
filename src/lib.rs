//! GPU execution and Loom compilation without a build-time native dependency.
//!
//! Native libraries are verified and loaded on first use. See [`bundle`] for
//! offline setup.
pub mod bundle;
mod runtime;
#[allow(dead_code)]
mod sys;
mod target;
pub use runtime::*;
/// Coordinated GPU/NPU execution with inferred memory dependencies.
pub mod execution;
pub use execution::{Completion, Runtime};
/// Low-level GPU execution. Kernel dispatch and external memory access require
/// the caller to establish their safety and synchronization contracts.
pub mod gpu {
    pub use crate::runtime::*;
}
pub use target::{TARGET_FAMILY, TARGET_KEY, Target};
#[cfg(feature = "loom")]
pub mod loom;
#[cfg(feature = "npu")]
pub mod npu;

/// The README's example is compiled with the crate, so an API change that would
/// invalidate it fails the build instead of reaching a reader.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct Readme;

/// A runtime, validation, provisioning or compiler failure.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A mapped allocation or exhausted prepared run capacity prevents submission.
    #[error("resource busy: {0}")]
    Busy(String),
    /// Hardware or the selected native runtime cannot provide a requested capability.
    #[error("unsupported capability: {0}")]
    Unsupported(String),
    /// A device failure left memory contents or completion uncertain.
    #[error("device execution failed: {0}")]
    DeviceLost(String),
    /// A native backend failure with its original status code.
    #[error("{backend} {operation} failed ({code}): {message}")]
    Backend {
        /// Native backend name.
        backend: &'static str,
        /// Operation that failed.
        operation: &'static str,
        /// Native status or command state.
        code: i32,
        /// Native diagnostic text. A boxed slice keeps the success-path Result
        /// layout within the pre-existing native error size.
        message: Box<str>,
    },
    /// A retained asynchronous failure, shared by completion observers.
    #[error("{source}")]
    Execution {
        /// Original structured failure, including backend and node context.
        #[source]
        source: std::sync::Arc<Error>,
    },
    /// Work was cancelled before completion.
    #[error("execution cancelled")]
    Cancelled,
    /// Invalid input or a contextual diagnostic.
    #[error("{0}")]
    Message(String),
    /// Structured diagnostics from a failed Loom compiler invocation.
    #[cfg(feature = "loom")]
    #[error("Loom compilation failed: {message}")]
    Compile {
        /// Rendered diagnostic summary.
        message: String,
        /// Complete structured diagnostics.
        diagnostics: Vec<loom::Diagnostic>,
    },
    /// An operating-system failure, preserving its error kind and source.
    #[error("{0}")]
    Io(#[from] std::io::Error),
    /// Invalid JSON, preserving the parser's location and source.
    #[error("{0}")]
    Json(#[from] serde_json::Error),
    /// A dynamic library or symbol could not be loaded.
    #[error("{0}")]
    Library(#[from] libloading::Error),
    /// A network provisioning failure.
    #[cfg(feature = "download")]
    #[error("{0}")]
    Download(#[source] Box<ureq::Error>),
    /// A native HRX status, including its machine-readable code.
    #[error("{context}: {message}")]
    Runtime {
        /// Operation that failed.
        context: String,
        /// Native HRX status code.
        code: i32,
        /// Owned native status message.
        message: String,
    },
    /// Adds context while preserving an underlying failure.
    #[error("{context}: {source}")]
    Context {
        /// Operation that failed.
        context: String,
        #[source]
        /// Underlying failure.
        source: Box<Error>,
    },
}
impl Error {
    /// Attach context without flattening the underlying error.
    pub fn context(self, context: impl Into<String>) -> Self {
        Self::Context {
            context: context.into(),
            source: Box::new(self),
        }
    }
}
/// The result of an HRX operation.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(feature = "runner")]
pub mod runner;

// Initialization failures remain retryable; only a successful value is memoized.
pub(crate) fn cached_init<T: Clone>(
    cache: &std::sync::Mutex<Option<T>>,
    initialize: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let mut cached = cache
        .lock()
        .map_err(|_| Error::Message("initialization cache poisoned".into()))?;
    if let Some(value) = cached.as_ref() {
        return Ok(value.clone());
    }
    let value = initialize()?;
    *cached = Some(value.clone());
    Ok(value)
}

#[cfg(test)]
mod initialization_tests {
    #[test]
    fn failures_retry_and_concurrent_success_initializes_once() {
        use std::sync::{
            Mutex,
            atomic::{AtomicUsize, Ordering},
        };
        let cache = Mutex::new(None);
        assert!(
            super::cached_init(&cache, || Err::<usize, _>(super::Error::Message(
                "transient".into()
            )))
            .is_err()
        );
        let calls = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    assert_eq!(
                        super::cached_init(&cache, || {
                            calls.fetch_add(1, Ordering::Relaxed);
                            Ok(42)
                        })
                        .unwrap(),
                        42
                    );
                });
            }
        });
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }
}
