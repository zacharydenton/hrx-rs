//! Release-time packaging of an already staged runtime; no native build is run.
use hrx::{Error, Result, bundle};
use std::{collections::BTreeMap, fs, path::Path};

pub fn pack(
    source: &Path,
    output: &Path,
    url: &str,
    revision: &str,
    target: &hrx::Target,
) -> Result<()> {
    fs::create_dir_all(output)?;
    let source = fs::canonicalize(source)?;
    for name in ["provenance.json", "THIRD-PARTY.json", "NOTICE"] {
        if !source.join(name).is_file() || fs::metadata(source.join(name))?.len() == 0 {
            return Err(Error::Message(format!(
                "release bundle requires nonempty {name}; see THIRD-PARTY.md"
            )));
        }
    }
    // These checks establish file structure, not license compliance or build provenance.
    // Release maintainers must review the inventory against every shipped binary.
    for name in ["provenance.json", "THIRD-PARTY.json"] {
        let value: serde_json::Value = serde_json::from_slice(&fs::read(source.join(name))?)?;
        if value.as_object().is_none_or(|object| object.is_empty()) {
            return Err(Error::Message(format!(
                "{name} must be a nonempty JSON object"
            )));
        }
    }
    if fs::canonicalize(output)? == source {
        return Err(Error::Message(
            "bundle output must differ from its source directory".into(),
        ));
    }
    let mut files = BTreeMap::new();
    for entry in fs::read_dir(&source)? {
        let entry = entry?;
        if entry.path().is_file() && entry.file_name() != "manifest.json" {
            let name = entry
                .file_name()
                .into_string()
                .map_err(|_| Error::Message("non-UTF8 bundle filename".into()))?;
            files.insert(name, bundle::file_digest(&entry.path())?);
        }
    }
    let mut manifest = bundle::Manifest {
        schema: 1,
        target: target.manifest_key(),
        revision: revision.into(),
        url: url.into(),
        archive_sha256: "0".repeat(64),
        files,
    };
    manifest.validate()?;
    let temporary = tempfile::NamedTempFile::new_in(output)?;
    let compressed = flate2::GzBuilder::new()
        .mtime(0)
        .write(temporary.as_file(), flate2::Compression::default());
    let mut archive = tar::Builder::new(compressed);
    for name in manifest.files.keys() {
        // Materialize source symlinks. Installation accepts only flat regular files.
        let mut file = fs::File::open(source.join(name))?;
        let mut header = tar::Header::new_gnu();
        header.set_size(file.metadata()?.len());
        header.set_mode(0o755);
        header.set_mtime(0);
        header.set_cksum();
        archive.append_data(&mut header, name, &mut file)?;
    }
    archive.into_inner()?.finish()?.sync_all()?;
    manifest.archive_sha256 = bundle::file_digest(temporary.path())?;
    // Detect changes in staged input before publishing.
    for (name, digest) in &manifest.files {
        if bundle::file_digest(&source.join(name))? != *digest {
            return Err(Error::Message(format!(
                "staged file changed during packaging: {name}"
            )));
        }
    }
    let path = output.join(format!("hrx-linux-x86_64-{}.tar.gz", target.as_str()));
    temporary.persist(&path).map_err(|e| e.error)?;
    let mut text = serde_json::to_string_pretty(&manifest)?;
    text.push('\n');
    fs::write(output.join("bundle.json"), text)?;
    println!("{}", path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn deterministic_bundle_installs_without_python_or_a_native_build() {
        let root = tempfile::tempdir().unwrap();
        let source = root.path().join("source");
        fs::create_dir(&source).unwrap();
        for name in ["libhrx.so", "libloomc.so", "libhsa-runtime64.so.1"] {
            fs::write(source.join(name), name).unwrap();
        }
        let rejected = root.path().join("rejected");
        assert!(
            pack(
                &source,
                &rejected,
                "https://example.test/native.tar.gz",
                "test",
                &hrx::Target::default()
            )
            .unwrap_err()
            .to_string()
            .contains("provenance.json")
        );
        assert!(!rejected.join("bundle.json").exists());
        for name in ["provenance.json", "THIRD-PARTY.json"] {
            fs::write(source.join(name), r#"{"fixture":true}"#).unwrap();
        }
        fs::write(
            source.join("NOTICE"),
            "Synthetic test fixture, no native binaries",
        )
        .unwrap();
        let a = root.path().join("a");
        let b = root.path().join("b");
        for out in [&a, &b] {
            pack(
                &source,
                out,
                "https://example.test/native.tar.gz",
                "test",
                &hrx::Target::default(),
            )
            .unwrap();
        }
        let archive = "hrx-linux-x86_64-gfx1151.tar.gz";
        assert_eq!(
            fs::read(a.join(archive)).unwrap(),
            fs::read(b.join(archive)).unwrap()
        );
        let manifest: bundle::Manifest =
            serde_json::from_slice(&fs::read(a.join("bundle.json")).unwrap()).unwrap();
        manifest
            .install(&a.join(archive), &root.path().join("cache"))
            .unwrap();
    }
}
