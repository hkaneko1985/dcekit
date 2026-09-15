#!/usr/bin/env python3
"""Build a target-free, state-free HTEM table for revised Phase 2."""

from __future__ import annotations

import argparse
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
FORBIDDEN = {
    "thickness_um", "xrd_peak_count", "absolute_temp_c",
    "fpm_resistivity_ohmcm", "fpm_conductivity_spercm",
    "fpm_standard_deviation_ohmpersq", "fpm_sheet_resistance_ohmpersq",
    "opt_direct_bandgap_ev", "opt_average_vis_trans",
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


def load_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modeling-records", type=Path, required=True)
    parser.add_argument("--phase1-records", type=Path, required=True)
    parser.add_argument("--study-links", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--study-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    phase1 = {
        int(row["sample_library_id"]): row
        for row in load_jsonl_gz(args.phase1_records)
    }
    output_rows = []
    with gzip.open(args.modeling_records, "rt", encoding="utf-8") as handle:
        for line in handle:
            source = json.loads(line)
            library_id = int(source["sample_library_id"])
            compositions = []
            for position in source.get("positions") or []:
                values = position.get("xrf_components") or {}
                if values:
                    compositions.append({str(key): float(value) for key, value in values.items()})
            row = {
                "sample_library_id": library_id,
                "elements": sorted(set(source.get("elements") or [])),
                "composition_measurements": compositions,
                "process": {
                    field: (source.get("process") or {}).get(field)
                    for field in PROCESS_FIELDS
                },
                "validation_only": {
                    "deposition_instrument": (
                        phase1.get(library_id, {}).get("validation_only") or {}
                    ).get("deposition_instrument")
                },
            }
            keys = set(row) | set(row["process"])
            if keys.intersection(FORBIDDEN):
                raise AssertionError("Forbidden target/state field entered Phase 2 table")
            output_rows.append(row)
    output_rows.sort(key=lambda row: row["sample_library_id"])
    if len(output_rows) != len({row["sample_library_id"] for row in output_rows}):
        raise ValueError("Duplicate sample_library_id")

    payload = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for row in output_rows
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(deterministic_gzip(payload))

    studies = json.loads(args.study_links.read_text(encoding="utf-8"))
    clean_studies = [
        {
            "study_id": int(study.get("study_id", study.get("id"))),
            "sample_library": sorted({int(value) for value in study.get("sample_library") or []}),
        }
        for study in studies
    ]
    args.study_output.write_text(json.dumps(clean_studies, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "version": "1.0.0",
        "records": len(output_rows),
        "retained": [
            "sample_library_id", "elements", "composition_measurements",
            "process", "validation_only.deposition_instrument"
        ],
        "validation_only": ["deposition_instrument", "study_id"],
        "property_or_state_fields_retained": [],
        "inputs": {
            "modeling_records": {"path": str(args.modeling_records), "sha256": sha256(args.modeling_records)},
            "phase1_records": {"path": str(args.phase1_records), "sha256": sha256(args.phase1_records)},
            "study_links": {"path": str(args.study_links), "sha256": sha256(args.study_links)},
        },
        "outputs": {
            "records": {"path": str(args.output), "bytes": args.output.stat().st_size, "sha256": sha256(args.output)},
            "studies": {"path": str(args.study_output), "bytes": args.study_output.stat().st_size, "sha256": sha256(args.study_output)},
        },
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
