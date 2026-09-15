#!/usr/bin/env python3
"""Normalize NanoMine cached sample views without imputing missing values."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import subprocess
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SAMPLE_RE = re.compile(r"materialsmine\.org/sample/([^>\s}]+)", re.I)
STEP_RE = re.compile(r"_step_(\d+)$", re.I)
NUMBER_RE = re.compile(r"^\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*(.*?)\s*$")
EQUIPMENT_LABELS = {"twin screw extruder", "single screw extruder"}
COMPONENT_NUMERIC_ATTRS = {
    "MassFraction",
    "VolumeFraction",
    "Density",
    "Width",
    "AspectRatio",
    "SpecificSurfaceArea",
}


def text(binding: dict[str, Any], key: str, default: str = "") -> str:
    return str(binding.get(key, {}).get("value", default))


def norm_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = value.replace("μ", "u").replace("µ", "u")
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def local_name(uri: str) -> str:
    return uri.rsplit("/", 1)[-1].rsplit("#", 1)[-1]


def load_cached_view(path: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    hits = json.loads(path.read_text(encoding="utf-8"))["data"]["hits"]
    grouped: dict[str, list[str]] = defaultdict(list)
    payloads: dict[str, list[dict[str, Any]]] = {}
    for hit in hits:
        source = hit.get("_source", {})
        match = SAMPLE_RE.search(source.get("label", ""))
        if not match:
            continue
        sample_id = match.group(1).lower()
        response = source.get("response", {})
        canonical = json.dumps(response, sort_keys=True, separators=(",", ":"))
        grouped[sample_id].append(canonical)
        payloads[sample_id] = response.get("results", {}).get("bindings", [])
    conflicts = sum(len(set(values)) > 1 for values in grouped.values())
    if conflicts:
        raise ValueError(f"{path}: {conflicts} samples have conflicting cached responses")
    audit = {
        "cache_hits": len(hits),
        "unique_samples": len(grouped),
        "duplicate_cache_hits": len(hits) - len(grouped),
        "conflicting_samples": conflicts,
    }
    return payloads, audit


def normalize_numeric(key: str, raw: str, explicit_unit: str = "") -> dict[str, Any] | None:
    match = NUMBER_RE.match(raw)
    if not match:
        return None
    value = float(match.group(1))
    if not math.isfinite(value):
        return None
    unit_raw = (explicit_unit + " " + match.group(2)).strip()
    unit = norm_text(unit_raw)
    key_norm = norm_text(key)
    group = f"raw:{unit or 'unitless'}"

    if "temperature" in key_norm or key_norm == "melt temperature":
        group = "temperature_c"
        if "kelvin" in unit or unit == "k":
            value -= 273.15
        elif "fahrenheit" in unit or unit == "f":
            value = (value - 32.0) * 5.0 / 9.0
    elif "time" in key_norm:
        group = "time_min"
        if "second" in unit or unit in {"s", "sec"}:
            value /= 60.0
        elif "hour" in unit or unit in {"h", "hr"}:
            value *= 60.0
        elif "day" in unit:
            value *= 1440.0
    elif "pressure" in key_norm:
        group = "pressure_mpa"
        if "giga" in unit or "gpa" in unit:
            value *= 1000.0
        elif "kilo" in unit or "kpa" in unit:
            value /= 1000.0
        elif unit in {"pa", "pascal"} or " pascal" in f" {unit}":
            value /= 1_000_000.0
        elif "bar" in unit:
            value *= 0.1
        elif "psi" in unit:
            value *= 0.006894757
    elif "rotation" in key_norm or "rotational" in key_norm or "frequency" in key_norm:
        group = "rotation_rpm"
        if unit in {"hz", "hertz"}:
            value *= 60.0
    elif any(token in key_norm for token in ("diameter", "width", "length")):
        group = "length_nm"
        if "micrometer" in unit or unit in {"um", "micron"}:
            value *= 1000.0
        elif "millimeter" in unit or unit == "mm":
            value *= 1_000_000.0
        elif "meter" in unit and "nano" not in unit:
            value *= 1_000_000_000.0
    elif key_norm in {"mass fraction", "volume fraction"}:
        group = "fraction"
        if "percent" in unit or unit == "%":
            value /= 100.0
    elif key_norm == "density":
        group = "density_g_cm3"
        if "kilogram per cubic meter" in unit:
            value /= 1000.0
    elif key_norm == "specific surface area":
        group = "specific_surface_area_m2_g"
    elif key_norm == "aspect ratio":
        group = "aspect_ratio"

    return {
        "kind": "numeric",
        "value": value,
        "unit_group": group,
        "raw": raw,
        "raw_unit": explicit_unit,
    }


def normalize_setting(key: str, raw: str) -> dict[str, Any]:
    numeric = normalize_numeric(key, raw)
    if numeric is not None:
        return numeric
    return {"kind": "categorical", "value": norm_text(raw), "raw": raw}


def parse_component_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    components: dict[tuple[str, str], dict[str, Any]] = {}
    label = ""
    for row in rows:
        role = text(row, "role") or "Unknown"
        name = text(row, "std_name")
        label = text(row, "fullLabel", label)
        key = (role, name)
        component = components.setdefault(key, {"role": role, "name": name, "attributes": []})
        attr_type = local_name(text(row, "attrType"))
        raw_value = text(row, "attrValue")
        raw_unit = text(row, "attrUnits")
        attr: dict[str, Any] = {"type": attr_type, "raw_value": raw_value, "raw_unit": raw_unit}
        if attr_type in COMPONENT_NUMERIC_ATTRS:
            parsed = normalize_numeric(attr_type, raw_value, raw_unit)
            if parsed:
                attr.update(parsed)
        component["attributes"].append(attr)

    by_role: dict[str, list[str]] = defaultdict(list)
    for component in components.values():
        by_role[component["role"]].append(component["name"])
        component["attributes"] = sorted(
            {json.dumps(x, sort_keys=True): x for x in component["attributes"]}.values(),
            key=lambda x: (x.get("type", ""), x.get("raw_value", "")),
        )
    return {
        "sample_label_from_component": label,
        "components": sorted(components.values(), key=lambda x: (x["role"], x["name"])),
        "matrix_names": sorted(set(by_role.get("Matrix", []))),
        "filler_names": sorted(set(by_role.get("Filler", []))),
        "surface_names": sorted(set(by_role.get("Surface Treatment", []))),
        "reported_roles": sorted(by_role),
    }


def parse_description(description: str) -> list[dict[str, Any]]:
    output = []
    for segment in description.split("; "):
        if ": " not in segment:
            continue
        key, raw = segment.split(": ", 1)
        if norm_text(key) == "description" or not raw.strip():
            continue
        output.append({"key": norm_text(key), **normalize_setting(key, raw.strip())})
    return output


def canonical_step(labels: list[str]) -> tuple[str, list[dict[str, Any]]]:
    clean = sorted({norm_text(label) for label in labels if label})
    non_equipment = [label for label in clean if label not in EQUIPMENT_LABELS]
    extra = []
    for label in clean:
        if label in EQUIPMENT_LABELS:
            extra.append({"key": "equipment type", "kind": "categorical", "value": label, "raw": label})
    if non_equipment:
        return "|".join(non_equipment), extra
    if any(label in EQUIPMENT_LABELS for label in clean):
        return "extrusion", extra
    return "unlabeled step", extra


def parse_process_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[text(row, "step")].append(row)

    def order_key(uri: str) -> tuple[int, str]:
        match = STEP_RE.search(uri)
        return (int(match.group(1)) if match else 10**9, uri)

    occurrence_count: Counter[str] = Counter()
    steps = []
    flat_settings = []
    for uri in sorted(grouped, key=order_key):
        step_rows = grouped[uri]
        labels = [text(row, "param_label") for row in step_rows if text(row, "param_label")]
        token, extra_settings = canonical_step(labels)
        occurrence = occurrence_count[token]
        occurrence_count[token] += 1
        settings = list(extra_settings)
        for row in step_rows:
            settings.extend(parse_description(text(row, "Descr")))
        settings = list({json.dumps(x, sort_keys=True): x for x in settings}.values())
        for setting in settings:
            flat_settings.append({
                **setting,
                "step_token": token,
                "occurrence": occurrence,
                "comparison_key": f"{token}#{occurrence}:{setting['key']}",
            })
        steps.append({
            "uri": uri,
            "index": order_key(uri)[0],
            "token": token,
            "labels": sorted(set(labels)),
            "settings": settings,
        })
    return {
        "steps": steps,
        "step_sequence": [step["token"] for step in steps],
        "step_types": sorted({part for step in steps for part in step["token"].split("|")}),
        "settings": flat_settings,
    }


def load_article_metadata(directory: Path | None) -> dict[str, dict[str, Any]]:
    if directory is None or not directory.exists():
        return {}
    output = {}
    for path in directory.glob("*.json"):
        try:
            common = json.loads(path.read_text(encoding="utf-8"))["Citation"]["CommonFields"]
        except (KeyError, json.JSONDecodeError):
            continue
        output[path.stem.lower()] = {
            "authors": common.get("Author", []),
            "location": common.get("Location"),
            "title": common.get("Title"),
            "citation_doi": common.get("DOI"),
        }
    return output


def git_commit(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw_cache"))
    parser.add_argument("--output-root", type=Path, default=Path("."))
    parser.add_argument("--citation-dir", type=Path)
    parser.add_argument("--pnc-repo", type=Path)
    parser.add_argument("--materialsmine-repo", type=Path)
    args = parser.parse_args()

    component_rows, component_audit = load_cached_view(args.raw_dir / "components_knowledge.json")
    process_rows, process_audit = load_cached_view(args.raw_dir / "processing_knowledge.json")
    header_rows, header_audit = load_cached_view(args.raw_dir / "headers_knowledge.json")
    index_payload = json.loads((args.raw_dir / "sample_index.json").read_text(encoding="utf-8"))
    index_hits = index_payload["data"]["hits"]
    sample_index = {
        hit["_source"]["identifier"].rsplit("/", 1)[-1].lower(): hit["_source"] for hit in index_hits
    }
    article_metadata = load_article_metadata(args.citation_dir)

    eligible_ids = sorted(set(component_rows) & set(process_rows) & set(header_rows))
    records = []
    for sample_id in eligible_ids:
        header = header_rows[sample_id]
        dois = sorted({text(row, "DOI") for row in header if text(row, "DOI")})
        process_families = sorted({text(row, "process_label") for row in header if text(row, "process_label")})
        labels = sorted({text(row, "sample_label") for row in header if text(row, "sample_label")})
        article_id = sample_id.split("-", 1)[0]
        component = parse_component_rows(component_rows[sample_id])
        process = parse_process_rows(process_rows[sample_id])
        source = sample_index.get(sample_id, {})
        record = {
            "sample_id": sample_id,
            "sample_uri": source.get("identifier", f"http://materialsmine.org/sample/{sample_id}"),
            "article_id": article_id,
            "doi": dois[0] if dois else None,
            # The L/E prefix is the NanoMine article identifier.  It is more
            # reliable than DOI here because control records may carry an
            # unpublished placeholder while sibling samples carry the final DOI.
            "paper_group": "article:" + article_id,
            "sample_label": labels[0] if labels else source.get("label", component["sample_label_from_component"]),
            "process_families": process_families,
            **component,
            **process,
            "citation_metadata": article_metadata.get(article_id),
        }
        records.append(record)

    data_dir = args.output_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    record_path = data_dir / "nanomine_phase3_records.jsonl.gz"
    with gzip.open(record_path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    family_samples: Counter[str] = Counter()
    operation_samples: dict[tuple[str, str], set[str]] = defaultdict(set)
    operation_occurrences: Counter[tuple[str, str]] = Counter()
    setting_samples: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in records:
        families = record["process_families"] or ["Unreported"]
        for family in families:
            family_samples[family] += 1
            for operation in record["step_types"]:
                operation_samples[(family, operation)].add(record["sample_id"])
            for token in record["step_sequence"]:
                for operation in token.split("|"):
                    operation_occurrences[(family, operation)] += 1
            for setting in record["settings"]:
                setting_samples[(family, setting["key"])].add(record["sample_id"])

    inventory_path = data_dir / "process_family_inventory.csv"
    with inventory_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "process_family", "record_type", "name", "sample_count", "family_sample_count",
            "prevalence", "total_occurrences",
        ])
        writer.writeheader()
        for (family, operation), ids in sorted(operation_samples.items()):
            writer.writerow({
                "process_family": family,
                "record_type": "operation",
                "name": operation,
                "sample_count": len(ids),
                "family_sample_count": family_samples[family],
                "prevalence": len(ids) / family_samples[family],
                "total_occurrences": operation_occurrences[(family, operation)],
            })
        for (family, setting), ids in sorted(setting_samples.items()):
            writer.writerow({
                "process_family": family,
                "record_type": "setting_key",
                "name": setting,
                "sample_count": len(ids),
                "family_sample_count": family_samples[family],
                "prevalence": len(ids) / family_samples[family],
                "total_occurrences": "",
            })

    role_patterns = Counter(tuple(record["reported_roles"]) for record in records)
    paper_counts = Counter(record["paper_group"] for record in records)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target_or_property_values_used": False,
        "source": {
            "site": "https://materialsmine.org/",
            "public_search_route": "https://materialsmine.org/api/search/filter",
            "note": "Public cached sample-view responses were used because the live SPARQL backend returned HTTP 503 during acquisition.",
            "materialsmine_commit": git_commit(args.materialsmine_repo),
            "pncextract_commit": git_commit(args.pnc_repo),
        },
        "raw_files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(args.raw_dir.glob("*.json"))
        },
        "cache_audit": {
            "sample_index": len(sample_index),
            "components": component_audit,
            "processing": process_audit,
            "headers": header_audit,
        },
        "analysis_population": {
            "samples": len(records),
            "papers": len(paper_counts),
            "papers_with_at_least_two_samples": sum(count >= 2 for count in paper_counts.values()),
            "samples_with_paper_neighbor": sum(count for count in paper_counts.values() if count >= 2),
            "same_paper_pairs": sum(count * (count - 1) // 2 for count in paper_counts.values()),
            "process_families": family_samples,
            "role_patterns": {"|".join(pattern): count for pattern, count in role_patterns.items()},
            "citation_metadata_samples": sum(record["citation_metadata"] is not None for record in records),
        },
    }
    (data_dir / "input_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=dict), encoding="utf-8"
    )
    print(json.dumps(manifest["analysis_population"], ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    main()
