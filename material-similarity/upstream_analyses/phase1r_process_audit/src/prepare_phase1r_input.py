#!/usr/bin/env python3
"""Create a property-free HTEM process audit table from cached source files."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Any


PROCESS_FIELDS = [
    "deposition_base_pressure_mtorr",
    "deposition_compounds",
    "deposition_cycles",
    "deposition_energy",
    "deposition_gas_flow_sccm",
    "deposition_gases",
    "deposition_growth_pressure_mtorr",
    "deposition_initial_temp_c",
    "deposition_power",
    "deposition_rep_rate",
    "deposition_sample_time_min",
    "deposition_substrate_material",
    "deposition_target_pulses",
    "deposition_ts_distance",
]

FORBIDDEN_OUTPUT_KEYS = {
    "fpm_resistivity_ohmcm",
    "fpm_conductivity_spercm",
    "fpm_standard_deviation_ohmpersq",
    "fpm_sheet_resistance_ohmpersq",
    "opt_direct_bandgap_ev",
    "opt_average_vis_trans",
    "thickness_um",
    "xrd_peak_count",
    "absolute_temp_c",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_gzip(payload: bytes) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as handle:
        handle.write(payload)
    return buffer.getvalue()


def read_instruments(cache_dir: Path) -> tuple[dict[int, str | None], list[dict[str, Any]]]:
    instruments: dict[int, str | None] = {}
    source_files = []
    for path in sorted(cache_dir.glob("*.csv.gz")):
        source_files.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)})
        with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"sample_library_id", "deposition_instrument"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError(f"Missing required columns in {path}")
            for row in reader:
                raw_id = (row.get("sample_library_id") or "").strip()
                if not raw_id or raw_id == "sample_library_id":
                    continue
                library_id = int(raw_id)
                value = (row.get("deposition_instrument") or "").strip() or None
                previous = instruments.get(library_id)
                if previous is not None and value is not None and previous != value:
                    raise ValueError(f"Conflicting instrument codes for library {library_id}")
                if library_id not in instruments or instruments[library_id] is None:
                    instruments[library_id] = value
    return instruments, source_files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modeling-records", type=Path, required=True)
    parser.add_argument("--raw-cache", type=Path, required=True)
    parser.add_argument("--studies", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--study-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    instruments, cache_files = read_instruments(args.raw_cache)
    rows = []
    with gzip.open(args.modeling_records, "rt", encoding="utf-8") as handle:
        for line in handle:
            source = json.loads(line)
            process = {name: (source.get("process") or {}).get(name) for name in PROCESS_FIELDS}
            row = {
                "sample_library_id": int(source["sample_library_id"]),
                "elements": sorted(set(source.get("elements") or [])),
                "process": process,
                # Audit/provenance only. The downstream specification forbids this
                # field from entering a material similarity.
                "validation_only": {
                    "deposition_instrument": instruments.get(int(source["sample_library_id"]))
                },
            }
            if FORBIDDEN_OUTPUT_KEYS.intersection(row) or FORBIDDEN_OUTPUT_KEYS.intersection(process):
                raise AssertionError("A forbidden outcome/state key entered the audit table")
            rows.append(row)
    rows.sort(key=lambda row: row["sample_library_id"])
    payload = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for row in rows
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(deterministic_gzip(payload))
    source_studies = json.loads(args.studies.read_text(encoding="utf-8"))
    study_links = [
        {
            "study_id": int(study["id"]),
            "sample_library": sorted({int(value) for value in (study.get("sample_library") or [])}),
        }
        for study in source_studies
    ]
    args.study_output.parent.mkdir(parents=True, exist_ok=True)
    args.study_output.write_text(
        json.dumps(study_links, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "version": "1.0.0",
        "records": len(rows),
        "policy": {
            "retained": ["sample_library_id", "elements", "process", "validation_only.deposition_instrument"],
            "deposition_instrument_role": "reporting-pattern audit only; prohibited from similarity",
            "property_or_target_columns_retained": [],
        },
        "inputs": {
            "modeling_records": {
                "path": str(args.modeling_records),
                "bytes": args.modeling_records.stat().st_size,
                "sha256": sha256(args.modeling_records),
            },
            "raw_cache_files": cache_files,
            "studies": {
                "path": str(args.studies),
                "bytes": args.studies.stat().st_size,
                "sha256": sha256(args.studies),
            },
        },
        "output": {
            "path": str(args.output),
            "bytes": args.output.stat().st_size,
            "sha256": sha256(args.output),
        },
        "study_output": {
            "path": str(args.study_output),
            "bytes": args.study_output.stat().st_size,
            "sha256": sha256(args.study_output),
            "retained_fields": ["study_id", "sample_library"],
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
