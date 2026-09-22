//! Explicit translation-unit inputs, independent of the host filesystem.
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Source language accepted by the native frontend.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub enum CxxStandard {
    /// ISO C23.
    C23,
    /// ISO C++26.
    #[default]
    Cpp26,
}
impl CxxStandard {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::C23 => "c23",
            Self::Cpp26 => "c++26",
        }
    }
}

/// A C/C++ translation unit and all of its explicit include dependencies.
/// Embedded Loom facade headers are supplied by the pinned compiler.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CxxSource {
    /// Main source identifier, also used for quoted include resolution.
    pub identifier: String,
    /// Main translation-unit contents.
    pub contents: String,
    /// Language standard.
    pub standard: CxxStandard,
    /// Virtual paths to header contents. No implicit filesystem reads occur.
    pub headers: BTreeMap<String, String>,
    /// Ordered include-search roots within the virtual source set.
    pub include_paths: Vec<String>,
    /// Ordered preprocessor definitions, as (name, replacement) pairs.
    pub defines: Vec<(String, String)>,
    /// Exported source functions; empty selects visible concrete definitions.
    pub roots: Vec<String>,
    /// Permit approximate math under Loom's AFN contract.
    pub approximate_functions: bool,
}
impl CxxSource {
    /// C++26, LP64, strict math, and the compiler's embedded facade headers.
    pub fn new(identifier: impl Into<String>, contents: impl Into<String>) -> Self {
        Self {
            identifier: identifier.into(),
            contents: contents.into(),
            standard: CxxStandard::default(),
            headers: BTreeMap::new(),
            include_paths: Vec::new(),
            defines: Vec::new(),
            roots: Vec::new(),
            approximate_functions: false,
        }
    }
    pub(super) fn validate(&self) -> Result<()> {
        if self.identifier.is_empty()
            || self.identifier.contains('\0')
            || self
                .headers
                .keys()
                .any(|key| key.is_empty() || key.contains('\0'))
            || self.include_paths.iter().any(|key| key.contains('\0'))
        {
            return Err(Error::Message(
                "invalid C/C++ source identifier or include path".into(),
            ));
        }
        Ok(())
    }
}

/// One input to a linked compilation module.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Source {
    /// Textual Loom IR.
    Loom {
        /// Diagnostic source identifier.
        identifier: String,
        /// Source contents.
        contents: String,
    },
    /// Native C/C++ import, producing ordinary Loom IR.
    Cxx(CxxSource),
}
impl Source {
    /// Construct a named Loom input.
    pub fn loom(identifier: impl Into<String>, contents: impl Into<String>) -> Self {
        Self::Loom {
            identifier: identifier.into(),
            contents: contents.into(),
        }
    }
    pub(super) fn validate(&self) -> Result<()> {
        match self {
            Self::Loom { identifier, .. } if identifier.is_empty() || identifier.contains('\0') => {
                Err(Error::Message("invalid Loom source identifier".into()))
            }
            Self::Cxx(source) => source.validate(),
            Self::Loom { .. } => Ok(()),
        }
    }
}

// Lexical virtual paths only: never resolves symlinks or touches host files.
// Absolute virtual names have their own root; traversal cannot escape either.
pub(super) fn virtual_path(path: &str) -> Result<String> {
    if path.contains('\0') || path.contains('\\') {
        return Err(Error::Message("invalid virtual include path".into()));
    }
    let mut parts = Vec::new();
    for part in path.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                if parts.pop().is_none() {
                    return Err(Error::Message(
                        "virtual include escapes its source root".into(),
                    ));
                }
            }
            part => parts.push(part),
        }
    }
    let prefix = if path.starts_with('/') { "/" } else { "" };
    Ok(format!("{prefix}{}", parts.join("/")))
}
impl CxxSource {
    pub(super) fn normalize(&mut self) -> Result<()> {
        self.validate()?;
        self.identifier = virtual_path(&self.identifier)?;
        if self.identifier.is_empty() || self.identifier == "/" {
            return Err(Error::Message("source identifier must name a file".into()));
        }
        let mut headers = BTreeMap::new();
        for (path, contents) in &self.headers {
            let path = virtual_path(path)?;
            if path.is_empty()
                || path == "/"
                || path == self.identifier
                || headers.insert(path, contents.clone()).is_some()
            {
                return Err(Error::Message(
                    "empty or aliased virtual header path".into(),
                ));
            }
        }
        self.headers = headers;
        for path in &mut self.include_paths {
            *path = virtual_path(path)?;
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn virtual_paths_are_lexical_and_ambiguous_aliases_are_rejected() {
        assert_eq!(
            virtual_path("/src/./nested/../factor.h").unwrap(),
            "/src/factor.h"
        );
        assert!(virtual_path("../secret.h").is_err());
        assert!(virtual_path("/../../secret.h").is_err());
        let mut source = CxxSource::new("src/kernel.cpp", "");
        source
            .headers
            .insert("src/nested/../factor.h".into(), "1".into());
        source.headers.insert("src/factor.h".into(), "2".into());
        assert!(source.normalize().is_err());
    }
}
