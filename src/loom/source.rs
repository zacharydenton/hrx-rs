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

/// Default floating-point transformations for a C/C++ translation unit.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub enum CxxFpModel {
    /// Preserve floating-point values and evaluation order.
    #[default]
    Precise,
    /// Permit Loom's complete fast-math contract, including reassociation.
    Fast,
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
    /// Default math policy; source attributes can override individual functions.
    #[serde(default)]
    pub fp_model: CxxFpModel,
    /// Physical virtual path to retained diagnostic/module identifier. Lookup
    /// still uses the physical path. Distinct sources need distinct identities.
    #[serde(default)]
    pub source_identifiers: BTreeMap<String, String>,
    /// Optional process-local reusable leading includes. Serialization omits
    /// this optimization; header/configuration changes use ordinary parsing.
    #[serde(skip)]
    pub preamble: Option<CxxPreamble>,
}
/// Immutable parsed C++ include prefix, shareable across concurrent imports.
/// Create with [`super::Compiler::cxx_preamble`]. The compiler checks the exact
/// prefix; this wrapper also checks virtual header/configuration identity so a
/// changed header cannot silently reuse an old semantic snapshot.
#[derive(Clone)]
pub struct CxxPreamble(pub(super) std::sync::Arc<super::native::Preamble>);
impl std::fmt::Debug for CxxPreamble {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CxxPreamble").finish_non_exhaustive()
    }
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
            fp_model: CxxFpModel::default(),
            source_identifiers: BTreeMap::new(),
            preamble: None,
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
    pub(super) fn preamble_key(&self) -> Result<String> {
        Ok(crate::bundle::digest(&serde_json::to_vec(&(
            &self.identifier,
            self.standard,
            &self.headers,
            &self.include_paths,
            &self.defines,
            &self.source_identifiers,
        ))?))
    }
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
        let mut identifiers = BTreeMap::new();
        for (path, identifier) in &self.source_identifiers {
            if identifier.is_empty()
                || identifier.contains('\0')
                || identifiers
                    .insert(virtual_path(path)?, identifier.clone())
                    .is_some()
            {
                return Err(Error::Message(
                    "invalid or aliased source identifier mapping".into(),
                ));
            }
        }
        let mut logical = std::collections::BTreeSet::new();
        for path in std::iter::once(&self.identifier).chain(self.headers.keys()) {
            if !logical.insert(identifiers.get(path).unwrap_or(path)) {
                return Err(Error::Message(
                    "source identifier mapping merges distinct sources".into(),
                ));
            }
        }
        self.source_identifiers = identifiers;
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
