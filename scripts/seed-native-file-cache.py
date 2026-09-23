#!/usr/bin/env python3
"""Seed CMake's pinned header cache from verified release downloads."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("build", type=Path)
    parser.add_argument("downloads", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    inputs = json.loads((repo / "native/release-inputs.json").read_text())
    lock = (args.source / "MODULE.cmake.lock").read_text()
    for name, spec in inputs["downloads"].items():
        key = name.upper()
        match = re.search(
            rf'set\(IREE_DEP_{key}_DOWNLOADED_FILE_PATH "([^"]+)"\)', lock
        )
        if match is None:
            continue
        relative = Path(match[1])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid locked file path: {relative}")
        source = args.downloads / spec["archive"]
        with source.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != spec["sha256"]:
            raise ValueError(f"Digest mismatch: {name}")
        target = args.build / "_deps" / f"{name}-src" / "file" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        print(f"Seeded {name}: {digest}")


if __name__ == "__main__":
    main()
