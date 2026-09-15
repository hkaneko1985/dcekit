#!/usr/bin/env python3
"""Compare recomputed result tables with the frozen reference results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual", type=Path, default=ROOT / "outputs" / "reproduced")
    parser.add_argument("--reference", type=Path, default=ROOT / "results")
    parser.add_argument("--rtol", type=float, default=1e-7)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    failures: list[dict] = []
    checked = 0

    def compare(expected, actual, location: str) -> None:
        if isinstance(expected, dict):
            if not isinstance(actual, dict) or set(expected) != set(actual):
                failures.append({"location": location, "reason": "different object keys"})
                return
            for key in expected:
                compare(expected[key], actual[key], f"{location}.{key}")
        elif isinstance(expected, list):
            if not isinstance(actual, list) or len(expected) != len(actual):
                failures.append({"location": location, "reason": "different list length"})
                return
            for index, (left, right) in enumerate(zip(expected, actual)):
                compare(left, right, f"{location}[{index}]")
        elif isinstance(expected, bool) or expected is None:
            if expected != actual:
                failures.append({"location": location, "expected": expected, "actual": actual})
        else:
            try:
                left, right = float(expected), float(actual)
            except (ValueError, TypeError):
                equal = expected == actual
            else:
                equal = (math.isnan(left) and math.isnan(right)) or math.isclose(
                    left, right, rel_tol=args.rtol, abs_tol=args.atol
                )
            if not equal:
                failures.append({"location": location, "expected": expected, "actual": actual})

    reference_files = sorted(p for p in args.reference.iterdir() if p.suffix in {".csv", ".json"})
    if not reference_files:
        raise SystemExit("No reference result tables found.")
    for reference in reference_files:
        actual = args.actual / reference.name
        before = len(failures)
        if not actual.is_file():
            failures.append({"location": reference.name, "reason": "missing output file"})
        else:
            if reference.suffix == ".json":
                left = json.loads(reference.read_text(encoding="utf-8"))
                right = json.loads(actual.read_text(encoding="utf-8"))
            else:
                with reference.open(encoding="utf-8", newline="") as handle:
                    left = list(csv.DictReader(handle))
                with actual.open(encoding="utf-8", newline="") as handle:
                    right = list(csv.DictReader(handle))
            compare(left, right, reference.name)
            checked += 1
        print(f"{'PASS' if len(failures) == before else 'DIFFER'} {reference.name}")
    report = {
        "files_checked": checked,
        "reference_files": len(reference_files),
        "rtol": args.rtol,
        "atol": args.atol,
        "passed": not failures,
        "differences": failures,
        "scope": "Common-interface examples only; earlier resource-specific grids are not recomputed.",
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if failures:
        print(json.dumps(failures[:10], ensure_ascii=False, indent=2))
        raise SystemExit(f"{len(failures)} differences found. Inspect the results and environment before interpreting them.")
    print(f"All {checked} result files match within the specified tolerance.")


if __name__ == "__main__":
    main()
