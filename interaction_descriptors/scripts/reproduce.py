#!/usr/bin/env python3
"""Explicit reproduction entry points; see PIPELINE.md for inputs and scope.

All fitting/figure work writes into new output directories. Archived inputs are
never edited. Commands distinguish table/figure regeneration, saved-prediction
audit, representative source refitting, and new target-regressor fitting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(command, output=None):
    env = dict(os.environ)
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        env[variable] = "1"
    if output is not None:
        env["MPLCONFIGDIR"] = str(output / "matplotlib_cache")
    subprocess.run([str(x) for x in command], check=True, env=env)


def fresh(path):
    path = path.resolve()
    check_empty_output(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def check_empty_output(path):
    """Validate an output without creating it or touching earlier results."""
    path = path.resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"Use a new or empty output directory: {path}")


def require_files(paths, description):
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        names = "; ".join(missing[:5])
        if len(missing) > 5:
            names += f"; and {len(missing) - 5} more"
        raise ValueError(f"Missing {description}: {names}. See PIPELINE.md for the required external inputs.")


def separate_output(output, inputs):
    candidate = output.resolve()
    for source in inputs:
        source = source.resolve()
        if candidate == source or source in candidate.parents:
            raise ValueError("The output must be separate from the immutable input directory.")


def require_prepared_inputs(source):
    """Check the supplied manifest's files before making a working copy."""
    manifest_path = source / "input_hashes.json"
    require_files([manifest_path], "prepared-input manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = [source / relative for relative in manifest]
    require_files(paths, "prepared input files")
    for relative, expected in manifest.items():
        if sha(source / relative) != expected:
            raise ValueError(f"Prepared input identity mismatch: {relative}")


def read_assets(path):
    require_files([path], "asset configuration")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not all(isinstance(value, str) for value in data.values()):
        raise ValueError("The asset configuration must map archive names to path strings.")
    roots = {}
    for key, value in data.items():
        root = Path(value).expanduser()
        roots[key] = root.resolve() if root.is_absolute() else (path.resolve().parent / root).resolve()
    return roots


def check_assets(args):
    assets = read_assets(args.assets)
    separate_output(args.output, assets.values())
    if args.output.exists():
        raise ValueError(f"Use a new asset-check output file: {args.output}")
    checks = []
    def check(root, relative, expected):
        path = root / relative
        actual = sha(path) if path.is_file() else None
        checks.append({"file": str(path), "sha256": actual,
                       "expected_sha256": expected, "matches": actual == expected})
    figures = json.loads((ROOT / "manifests/historical_figure_assets.json").read_text())
    for name in ["v40", "v41", "v42", "v44"]:
        if name in assets:
            for relative, expected in figures[name]["assets"].items():
                check(assets[name], relative, expected)
    if "v38" in assets:
        source = json.loads((ROOT / "manifests/source_training_assets.json").read_text())
        check(assets["v38"], "data/source_catalog.json", source["source_catalog_sha256"])
        for name, item in source["sources"].items():
            check(assets["v38"], item["input_path"], item["input_sha256"])
            for kind, seeds in item["models"].items():
                for seed, expected in seeds.items():
                    check(assets["v38"], f"models/{name}/{kind}/seed{seed}.npz", expected)
    if not checks:
        raise ValueError("No supported archive roots were supplied.")
    result = {"status": "PASS_FOR_PROVIDED_ASSETS" if all(row["matches"] for row in checks) else "FAIL",
              "files_checked": len(checks), "checks": checks,
              "scope": "Exact identities for the supplied source-bank/figure assets; not upstream data re-acquisition."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "checks"}, indent=2))
    if result["status"] == "FAIL":
        raise SystemExit(1)


def legacy_figures(args):
    assets = read_assets(args.assets)
    manifest = json.loads((ROOT / "manifests/historical_figure_assets.json").read_text())
    provenance = json.loads((ROOT / "manifests/pipeline_code_provenance.json").read_text())
    code_hashes = {item["public_file"]: item["sha256"] for item in provenance}
    separate_output(args.output_dir, assets.values())
    check_empty_output(args.output_dir)
    # Validate every group before any working tree is created. A missing later
    # archive must not leave an earlier figure group half-generated.
    for group in ["v40", "v41", "v42", "v44"]:
        if group not in assets:
            raise ValueError(f"Missing {group} root in asset configuration; see PIPELINE.md.")
        spec = manifest[group]
        require_files([assets[group] / relative for relative in spec["assets"]], f"{group} figure assets")
        for relative, expected in spec["assets"].items():
            if sha(assets[group] / relative) != expected:
                raise ValueError(f"Figure input identity mismatch: {group}/{relative}")
        code_files = [spec["script"], *["reference/figures/" + name for name in spec["extra_code"]]]
        require_files([ROOT / relative for relative in code_files], "bundled historical figure code")
        for relative in code_files:
            if sha(ROOT / relative) != code_hashes[relative]:
                raise ValueError(f"Archived plotting code changed: {relative}")
    output = fresh(args.output_dir)
    start = time.perf_counter()
    copied = []
    for group in ["v40", "v41", "v42", "v44"]:
        if group not in assets:
            raise ValueError(f"Missing {group} root in asset configuration.")
        spec = manifest[group]
        work = output / "work" / group
        for directory in ["code", "evidence", "results", "figures"]:
            (work / directory).mkdir(parents=True, exist_ok=True)
        for relative, expected in spec["assets"].items():
            source = assets[group] / relative
            if sha(source) != expected:
                raise ValueError(f"Figure input identity mismatch: {group}/{relative}")
            destination = work / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            copied.append({"archive": group, "path": relative, "sha256": expected})
        code_files = [spec["script"], *["reference/figures/" + name for name in spec["extra_code"]]]
        for relative in code_files:
            source = ROOT / relative
            if sha(source) != code_hashes[relative]:
                raise ValueError(f"Archived plotting code changed: {relative}")
            shutil.copyfile(source, work / "code" / source.name)
        run([args.python, work / "code" / Path(spec["script"]).name], output)
    destination = output / "main_figures"
    destination.mkdir()
    products = []
    for entry in manifest["main_figures"]:
        for extension in ["png", "svg"]:
            source = output / "work" / entry["group"] / "figures" / f"{entry['generated']}.{extension}"
            filename = f"Figure_{entry['figure']}_historical.{extension}"
            shutil.copyfile(source, destination / filename)
            products.append({"figure": entry["figure"], "file": filename, "sha256": sha(source),
                             "original_script": manifest[entry["group"]]["script"]})
    run([args.python, ROOT / "scripts/plot_steel_results.py", "--summary",
         ROOT / "publication/inputs/steel_summary_v46.csv", "--output-dir", destination], output)
    unchanged = all(sha(assets[row["archive"]] / row["path"]) == row["sha256"] for row in copied)
    result = {"status": "PASS" if unchanged else "FAIL", "original_assets_unchanged": unchanged,
              "original_scripts_executed": 4, "main_figures_1_to_6_rebuilt": 6,
              "figure7": "data-equivalent redraw; original render script not recovered",
              "new_training_or_prediction_fits": 0, "input_assets": copied,
              "products": products, "elapsed_seconds": time.perf_counter() - start,
              "rendering_note": "Numerical inputs and original scripts are fixed; PNG/SVG byte identity across rendering libraries is not assumed."}
    (output / "legacy_figure_status.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in {"products", "input_assets"}}, indent=2))


def stage_comparison(args, stage):
    source = (args.stage51_root if stage == 52 else args.input_dir).resolve()
    separate_output(args.output_dir, [source])
    check_empty_output(args.output_dir)
    if stage == 52:
        fusion = source / "fusion"
        required = [fusion / name for name in ["protocol.json", "input_hashes.json"]]
        required += [fusion / "results" / name for name in
                     ["summary.csv", "configuration_metrics.csv", "status.json", "validation.json"]]
        required.append(source / "sources/v49_replay.py")
        for case in ["polymer_modulus", "metals_volume", "oxides_formation"]:
            for seed in [101, 202, 303]:
                for repeat in range(1 if case == "polymer_modulus" else 2):
                    required.append(fusion / "results/jobs" / f"{case}__fusion_boruta__{seed}__r{repeat}.json")
        require_files(required, "stage51 comparison assets")
        require_prepared_inputs(fusion)
    else:
        require_files([source / name for name in
                       ["protocol.json", "input_hashes.json", "source_hashes.json", "environment.json"]],
                      "prepared architecture-comparison metadata")
        require_prepared_inputs(source)
    output = fresh(args.output_dir)
    folder = "regressor_comparison" if stage == 52 else "architecture_comparison"
    template = ROOT / "experiments" / folder
    shutil.copytree(template, output, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    if stage == 52:
        command = [args.python, output / "run_regressors52.py", "--stage51-root", source,
                   "--workers", str(args.workers)]
        if args.prepare_only:
            command.append("--prepare-only")
    else:
        for name in ["inputs", "protocol.json", "input_hashes.json", "source_hashes.json", "environment.json"]:
            source_path = source / name
            if source_path.is_dir():
                shutil.copytree(source_path, output / name)
            else:
                shutil.copyfile(source_path, output / name)
        command = [args.python, output / "run_fusion51.py", "--workers", str(args.workers)]
        if args.prepare_only:
            command.append("--prepare-only")
    run(command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable,
                        help="Interpreter for the chosen numerical/plotting entry point.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="Print the scoped reproduction commands.")
    command = commands.add_parser("check-assets")
    command.add_argument("--assets", required=True, type=Path)
    command.add_argument("--output", required=True, type=Path)
    command = commands.add_parser("tables")
    command.add_argument("--output-dir", required=True, type=Path)
    command = commands.add_parser("regressor-tables")
    command.add_argument("--output-dir", required=True, type=Path)
    command.add_argument("--summary", type=Path)
    command.add_argument("--protocol", type=Path)
    command = commands.add_parser("source-fit")
    command.add_argument("--archive-root", required=True, type=Path)
    command.add_argument("--output-dir", required=True, type=Path)
    command.add_argument("--source", default="mp_decomp")
    command.add_argument("--kind", choices=["NN", "YAE", "YVAE"], default="NN")
    command.add_argument("--seed", type=int, choices=[101, 202, 303], default=101)
    command.add_argument("--target-features", choices=["metals", "mixed_oxides"])
    command = commands.add_parser("steel-audit")
    command.add_argument("--input-dir", required=True, type=Path)
    command.add_argument("--output-dir", required=True, type=Path)
    command.add_argument("--mode", choices=["audit", "representative", "full"], default="audit")
    command = commands.add_parser("historical-figures")
    command.add_argument("--assets", required=True, type=Path)
    command.add_argument("--output-dir", required=True, type=Path)
    for name in ["architecture", "comparison"]:
        command = commands.add_parser(name)
        command.add_argument("--output-dir", required=True, type=Path)
        command.add_argument("--workers", type=int, default=4)
        command.add_argument("--prepare-only", action="store_true")
        if name == "comparison":
            command.add_argument("--stage51-root", required=True, type=Path)
        else:
            command.add_argument("--input-dir", required=True, type=Path)
    command = commands.add_parser("plot-comparison")
    command.add_argument("--summary", required=True, type=Path)
    command.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "list":
        print(json.dumps({
            "tables": "Tables 1–8/S1–S3 from bundled aggregate data; zero model fits",
            "regressor-tables": "Table 9 and SI Tables S4/S5 from the complete six-regressor aggregate matrix and frozen grids; zero fits",
            "source-fit": "One exact 200-epoch final fit from archived source-only rows; optional target-X feature parity",
            "steel-audit": "Original v49 runner: audit, representative 18 fits, or full 540 fits; select mode explicitly",
            "historical-figures": "Original scripts for Figures 1–6 in isolated work copies; data-equivalent steel Figure 7 redraw",
            "architecture": "New copy of the stage51 KRR runner with external prepared inputs",
            "comparison": "New copy of the stage52 five-additional-regressor runner with external stage51 inputs/selectors",
            "plot-comparison": "Figure 8 and compact Table 9 from the complete 144-row six-regressor summary; zero fits",
            "check-assets": "Verify exact external asset identities, without modifying them",
        }, indent=2))
    elif args.command == "check-assets":
        check_assets(args)
    elif args.command == "tables":
        check_empty_output(args.output_dir)
        run([args.python, ROOT / "scripts/export_publication_tables.py", "--output-dir", args.output_dir])
    elif args.command == "regressor-tables":
        check_empty_output(args.output_dir)
        require_files([getattr(args, name) for name in ("summary", "protocol")
                       if getattr(args, name) is not None], "regressor-table inputs")
        command = [args.python, ROOT / "scripts/export_regressor_tables52.py", "--output-dir", args.output_dir]
        for name in ("summary", "protocol"):
            if getattr(args, name) is not None:
                command += ["--" + name, getattr(args, name)]
        run(command)
    elif args.command == "source-fit":
        command = [args.python, ROOT / "scripts/source_fit52.py", "--archive-root", args.archive_root,
                   "--output-dir", args.output_dir, "--source", args.source, "--kind", args.kind,
                   "--seed", str(args.seed)]
        if args.target_features:
            command += ["--target-features", args.target_features]
        run(command)
    elif args.command == "steel-audit":
        separate_output(args.output_dir, [args.input_dir])
        check_empty_output(args.output_dir)
        require_files([args.input_dir / name for name in
                       ["v49_frozen_bank.npz", "steel_v46_replay_inputs.npz",
                        "steel_v46_oof_predictions.csv", "steel_v46_summary.csv"]],
                      "steel replay inputs")
        run([args.python, ROOT / "scripts/replay_v49.py", "--input-dir", args.input_dir,
             "--output-dir", args.output_dir, "--mode", args.mode])
    elif args.command == "historical-figures":
        legacy_figures(args)
    elif args.command == "architecture":
        stage_comparison(args, 51)
    elif args.command == "comparison":
        stage_comparison(args, 52)
    elif args.command == "plot-comparison":
        require_files([args.summary], "complete comparison summary")
        output = fresh(args.output_dir)
        run([args.python, ROOT / "scripts/plot_regressor_comparison52.py", "--summary", args.summary,
             "--output-dir", output], output)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as exc:
        raise SystemExit(f"error: {exc}") from None
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"The selected command exited with code {exc.returncode}; see its message above and PIPELINE.md.") from None
