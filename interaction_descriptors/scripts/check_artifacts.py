#!/usr/bin/env python3
"""Verify caller-supplied private artifacts without copying them into the repo."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--directory", type=Path, required=True)
p.add_argument("--require-all", action="store_true")
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "manifests/private_artifacts.json").read_text())
rows = []
for item in manifest["files"]:
    path = a.directory / item["filename"]
    found = path.is_file()
    actual = hashlib.sha256(path.read_bytes()).hexdigest() if found else None
    rows.append(dict(filename=item["filename"], present=found,
                     match=actual == item["sha256"] if found else None))
print(json.dumps(rows, indent=2))
if (not any(r["present"] for r in rows) or any(r["present"] and not r["match"] for r in rows)
        or (a.require_all and not all(r["present"] for r in rows))):
    raise SystemExit(1)
