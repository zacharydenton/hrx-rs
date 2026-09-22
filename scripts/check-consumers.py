#!/usr/bin/env python3
"""Check committed consumer snapshots against this HRX tree without editing their locks."""
import argparse
import io
import re
import tomllib
from pathlib import Path
import subprocess
import tarfile
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("consumers", nargs="+", type=Path)
    args = parser.parse_args()
    runtime = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="hrx-consumers-") as work:
        for index, consumer in enumerate(args.consumers):
            snapshot = Path(work) / str(index)
            snapshot.mkdir()
            archive = subprocess.check_output(["git", "-C", str(consumer), "archive", "HEAD"])
            with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
                tar.extractall(snapshot, filter="data")
            # This is a disposable compatibility probe for a breaking release.
            # Change only the copied dependency requirement, never the checkout.
            version = tomllib.loads((runtime / "Cargo.toml").read_text())["package"]["version"]
            for manifest in snapshot.rglob("Cargo.toml"):
                text = manifest.read_text()
                text = re.sub(r'(package\s*=\s*"hrx-rs"[^}]*?version\s*=\s*")[^"]+(" )',
                              lambda match: match[1] + version + match[2], text)
                manifest.write_text(text)
            config = snapshot / "candidate.toml"
            # JSON strings are valid TOML basic strings; this is data, not shell text.
            import json
            dependency = 'hrx-rs = { path = ' + json.dumps(str(runtime)) + ' }\n'
            config.write_text('[patch.crates-io]\n' + dependency +
                              '[patch."https://github.com/zacharydenton/hrx-rs"]\n' + dependency)
            for command in [["cargo", "test", "--all-features"],
                            ["cargo", "clippy", "--all-features", "--all-targets"]]:
                subprocess.run(command + ["--config", str(config)] +
                               (["--", "-D", "warnings"] if command[1] == "clippy" else []),
                               cwd=snapshot, check=True)


if __name__ == "__main__":
    main()
