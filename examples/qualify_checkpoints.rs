//! Verify the pinned generative qualification fixtures without initializing a GPU.
//! Offline by default; `--fetch` also downloads missing files into the Hub cache.
use hrx::artifacts::{hf, safetensors::FileView};
use serde::Deserialize;

#[derive(Deserialize)]
struct Repository {
    owner: String,
    repository: String,
    revision: String,
    files: Vec<File>,
}

#[derive(Deserialize)]
struct File {
    name: String,
    sha256: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let fetch = match args.as_slice() {
        [] => false,
        [argument] if argument == "--fetch" => true,
        _ => return Err("usage: qualify_checkpoints [--fetch]".into()),
    };
    let repositories: Vec<Repository> =
        serde_json::from_str(include_str!("../native/qualification/checkpoints.json"))?;
    let mut failures = 0;
    for repository in repositories {
        let resolver = hf::Resolver::new(
            hf::Repository::new(&repository.owner, &repository.repository).at(&repository.revision),
        )
        .offline(!fetch);
        for file in repository.files {
            eprintln!(
                "Checking {}/{} at {}: {}",
                repository.owner, repository.repository, repository.revision, file.name
            );
            let result = (|| -> hrx::Result<()> {
                let path = resolver.resolve(&hf::HubFile::new(file.name).sha256(file.sha256))?;
                // SAFETY: qualification requires completed, immutable cache files.
                // The Hub publishes downloads atomically; do not mutate these files
                // while this command is running.
                let view = unsafe { FileView::map(&path)? };
                println!("OK: {} tensors, {}", view.entries().len(), path.display());
                Ok(())
            })();
            if let Err(error) = result {
                failures += 1;
                eprintln!("FAILED: {error}");
            }
        }
    }
    if failures != 0 {
        return Err(format!("{failures} checkpoint(s) failed qualification").into());
    }
    Ok(())
}
