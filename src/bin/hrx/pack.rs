//! Release-time packaging of an already staged runtime; no native build is run.
use hrx::{Error, Result, bundle};
use std::{collections::BTreeMap, fs, path::Path};

pub fn pack(source: &Path, output: &Path, url: &str, revision: &str) -> Result<()> {
    fs::create_dir_all(output)?;
    let source = fs::canonicalize(source)?;
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
        target: "x86_64-unknown-linux-gnu-gfx1151".into(),
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
    let path = output.join("hrx-linux-x86_64-gfx1151.tar.gz");
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
        for name in ["libhrx.so", "loom-compile", "libhsa-runtime64.so.1"] {
            fs::write(source.join(name), name).unwrap();
        }
        let a = root.path().join("a");
        let b = root.path().join("b");
        for out in [&a, &b] {
            pack(&source, out, "https://example.test/native.tar.gz", "test").unwrap();
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
