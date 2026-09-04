#!/usr/bin/env python3
"""Verify the SHA-256 file manifest shipped with this package."""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "CHECKSUMS.sha256"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    failures: list[str] = []
    checked = 0
    for line_number, line in enumerate(MANIFEST.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            expected, relative_text = line.split("  ", 1)
        except ValueError:
            failures.append(f"line {line_number}: malformed entry")
            continue
        relative = PurePosixPath(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            failures.append(f"line {line_number}: unsafe path {relative_text!r}")
            continue
        path = ROOT.joinpath(*relative.parts)
        if not path.is_file():
            failures.append(f"missing: {relative_text}")
            continue
        observed = sha256(path)
        checked += 1
        if observed != expected:
            failures.append(f"mismatch: {relative_text}")
    if failures:
        raise SystemExit("CHECKSUM_FAILURE\n" + "\n".join(failures))
    print(f"CHECKSUMS_VERIFIED {checked} files")


if __name__ == "__main__":
    main()
