#!/usr/bin/env python3
"""Create a deterministic release archive and checksum manifest."""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path


EXCLUDED_PARTS = {"__pycache__"}
EXCLUDED_SUFFIXES = {".pyc"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def included_files(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*")
        if path.is_file()
        and not EXCLUDED_PARTS.intersection(path.relative_to(root).parts)
        and path.suffix not in EXCLUDED_SUFFIXES
        and path.name != "MANIFEST.sha256"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    files = included_files(root)
    manifest = "".join(f"{sha256(path)}  {path.relative_to(root).as_posix()}\n" for path in files)
    (root / "MANIFEST.sha256").write_text(manifest, encoding="utf-8")
    files = included_files(root) + [root / "MANIFEST.sha256"]
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(files):
            relative = Path(root.name) / path.relative_to(root)
            info = zipfile.ZipInfo(relative.as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
