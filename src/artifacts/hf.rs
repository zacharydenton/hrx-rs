//! Hugging Face model files resolved through the standard local cache.

use crate::{Error, Result, bundle};
use hf_hub::progress::{DownloadEvent, ProgressEvent, ProgressHandler};
use hf_hub::{HFClientSync, HFError};
use std::{
    io::{IsTerminal, Write},
    path::PathBuf,
    sync::{Mutex, OnceLock},
};

/// One model repository and optional immutable revision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Repository {
    /// Repository owner or organization.
    pub owner: String,
    /// Repository name.
    pub name: String,
    /// Commit, tag, or branch; `None` uses the Hub default.
    pub revision: Option<String>,
}

impl Repository {
    /// Select a repository at its default revision.
    pub fn new(owner: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            owner: owner.into(),
            name: name.into(),
            revision: None,
        }
    }

    /// Pin a revision.
    #[must_use]
    pub fn at(mut self, revision: impl Into<String>) -> Self {
        self.revision = Some(revision.into());
        self
    }
}

/// A repository-relative file and optional expected SHA-256 digest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HubFile {
    /// Repository-relative filename.
    pub name: String,
    /// Expected lowercase SHA-256, when the publisher provides one.
    pub sha256: Option<String>,
}

impl HubFile {
    /// Name a file without an independent checksum.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            sha256: None,
        }
    }

    /// Require an exact lowercase SHA-256 digest.
    #[must_use]
    pub fn sha256(mut self, digest: impl Into<String>) -> Self {
        self.sha256 = Some(digest.into());
        self
    }
}

/// Cache and download policy for one repository.
#[derive(Clone, Debug)]
pub struct Resolver {
    repository: Repository,
    offline: bool,
    progress: Option<bool>,
}

impl Resolver {
    /// Resolve files from a repository, downloading cache misses by default.
    pub fn new(repository: Repository) -> Self {
        Self {
            repository,
            offline: false,
            progress: None,
        }
    }

    /// Restrict resolution to local cache entries.
    #[must_use]
    pub fn offline(mut self, offline: bool) -> Self {
        self.offline = offline;
        self
    }

    /// Force progress on or off. The default uses an interactive terminal and
    /// `HF_HUB_DISABLE_PROGRESS_BARS`.
    #[must_use]
    pub fn progress(mut self, progress: bool) -> Self {
        self.progress = Some(progress);
        self
    }

    /// Resolve a required file, checking the cache before any network request.
    pub fn resolve(&self, file: &HubFile) -> Result<PathBuf> {
        if let Some(path) = self.local(file)? {
            return Ok(path);
        }
        if self.offline || environment_true("HF_HUB_OFFLINE") {
            return Err(Error::Message(format!(
                "{}/{}/{} is not in the Hugging Face cache, and downloading is off",
                self.repository.owner, self.repository.name, file.name
            )));
        }
        let repository = client()?.model(&self.repository.owner, &self.repository.name);
        let path = repository
            .download_file()
            .filename(&file.name)
            .maybe_revision(self.repository.revision.clone())
            .maybe_progress(self.show_progress().then(DownloadBar::default))
            .send()
            .map_err(|error| self.fetch_error(file, error))?;
        self.verify(file, path)
    }

    /// Return a cached file without contacting the network.
    pub fn local(&self, file: &HubFile) -> Result<Option<PathBuf>> {
        let result = client()?
            .model(&self.repository.owner, &self.repository.name)
            .download_file()
            .filename(&file.name)
            .maybe_revision(self.repository.revision.clone())
            .local_files_only(true)
            .send();
        match result {
            Ok(path) => self.verify(file, path).map(Some),
            Err(HFError::LocalEntryNotFound { .. }) => Ok(None),
            Err(error) => Err(self.fetch_error(file, error)),
        }
    }

    /// Resolve required files in request order.
    pub fn resolve_all(&self, files: &[HubFile]) -> Result<Vec<PathBuf>> {
        files.iter().map(|file| self.resolve(file)).collect()
    }

    fn verify(&self, file: &HubFile, path: PathBuf) -> Result<PathBuf> {
        if let Some(expected) = &file.sha256 {
            let actual = bundle::file_digest(&path)
                .map_err(|error| error.context(format!("hashing {}", path.display())))?;
            if &actual != expected {
                return Err(Error::Message(format!(
                    "{} has SHA-256 {actual}, expected {expected}",
                    path.display()
                )));
            }
        }
        Ok(path)
    }

    fn show_progress(&self) -> bool {
        self.progress.unwrap_or_else(|| {
            std::io::stderr().is_terminal() && !environment_true("HF_HUB_DISABLE_PROGRESS_BARS")
        })
    }

    fn fetch_error(&self, file: &HubFile, error: HFError) -> Error {
        Error::Message(format!(
            "cannot fetch {}/{}/{}{}: {error}",
            self.repository.owner,
            self.repository.name,
            file.name,
            self.repository
                .revision
                .as_deref()
                .map_or(String::new(), |revision| format!(" at {revision}"))
        ))
    }
}

fn client() -> Result<&'static HFClientSync> {
    static CLIENT: OnceLock<std::result::Result<HFClientSync, String>> = OnceLock::new();
    CLIENT
        .get_or_init(|| HFClientSync::new().map_err(|error| error.to_string()))
        .as_ref()
        .map_err(|error| Error::Message(format!("cannot initialize Hugging Face client: {error}")))
}

fn environment_true(name: &str) -> bool {
    std::env::var(name).is_ok_and(|value| {
        ["1", "ON", "YES", "TRUE"]
            .iter()
            .any(|truth| value.eq_ignore_ascii_case(truth))
    })
}

#[derive(Default)]
struct DownloadBar {
    total: Mutex<u64>,
}

impl ProgressHandler for DownloadBar {
    fn on_progress(&self, event: &ProgressEvent) {
        let ProgressEvent::Download(event) = event else {
            return;
        };
        let mut stderr = std::io::stderr().lock();
        match event {
            DownloadEvent::Start { total_bytes, .. } => {
                *self.total.lock().unwrap_or_else(|error| error.into_inner()) = *total_bytes;
            }
            DownloadEvent::AggregateProgress {
                bytes_completed,
                total_bytes,
                ..
            } => {
                let total = if *total_bytes == 0 {
                    *self.total.lock().unwrap_or_else(|error| error.into_inner())
                } else {
                    *total_bytes
                };
                if total > 0 {
                    let _ = write!(
                        stderr,
                        "\rfetching {:5.1}% of {:.2} GB",
                        100.0 * *bytes_completed as f64 / total as f64,
                        total as f64 / 1e9
                    );
                    let _ = stderr.flush();
                }
            }
            DownloadEvent::Complete => {
                let _ = writeln!(stderr, "\rfetched                    ");
            }
            DownloadEvent::Progress { .. } => {}
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn documented_boolean_values_are_recognized() {
        for value in ["1", "ON", "on", "YES", "yes", "TRUE", "true"] {
            assert!(
                ["1", "ON", "YES", "TRUE"]
                    .iter()
                    .any(|truth| value.eq_ignore_ascii_case(truth))
            );
        }
    }
}
