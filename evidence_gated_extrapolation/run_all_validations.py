#!/usr/bin/env python3
"""Run the paper validations without overwriting the archived reference results."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "physics_validation_results"


def invoke(arguments: list[str]) -> None:
    print("RUN", " ".join(arguments), flush=True)
    subprocess.run(arguments, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("reproduction_results"))
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.resolve() == REFERENCE.resolve():
        raise ValueError("Choose an output directory other than physics_validation_results")
    output.mkdir(parents=True, exist_ok=True)

    if not args.regenerate:
        source = REFERENCE / "datasets"
        if not source.is_dir():
            raise FileNotFoundError(f"Cached datasets not found: {source}")
        shutil.copytree(source, output / "datasets", dirs_exist_ok=True)

    physics_command = [
        sys.executable,
        "run_physics_validation.py",
        "--output",
        str(output),
        "--forward-reps",
        "1" if args.quick else "5",
        "--inverse-reps",
        "1" if args.quick else "3",
        "--inverse-targets",
        "2" if args.quick else "8",
    ]
    if args.regenerate:
        physics_command.append("--regenerate")
    invoke(physics_command)

    if not args.quick:
        invoke(
            [
                sys.executable,
                "run_prediction_variation_validation.py",
                "--result-dir",
                str(output),
            ]
        )

    invoke(
        [
            sys.executable,
            "run_strong_baseline_validation.py",
            "--input-dir",
            str(output),
            "--output-dir",
            str(output),
            "--reps",
            "1" if args.quick else "5",
        ]
    )

    invoke(
        [
            sys.executable,
            "run_bayesian_optimization_validation.py",
            "--input-dir",
            str(output),
            "--output-dir",
            str(output),
            "--reps",
            "1" if args.quick else "5",
            "--budget",
            "3" if args.quick else "9",
            "--batch-size",
            "3",
        ]
    )
    invoke(
        [
            sys.executable,
            "reanalyze_reference_results.py",
            "--results-dir",
            str(output),
        ]
    )
    print(f"VALIDATION_COMPLETE {output}", flush=True)


if __name__ == "__main__":
    main()
