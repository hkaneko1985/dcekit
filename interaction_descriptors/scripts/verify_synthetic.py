#!/usr/bin/env python3
"""Run the source-free tests without replacing the shipped verification record."""
import argparse
import json
import platform
import sys
import unittest
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("verification_output/synthetic_validation.json"),
                    help="New JSON result path; defaults to verification_output/ in the current directory.")
args = parser.parse_args()
out = args.output.resolve()
if out.exists():
    raise SystemExit("Use a new output JSON path; existing verification results are not overwritten.")
if root / "verification" in out.parents:
    raise SystemExit("Choose a result path outside the shipped verification/ directory.")
# Test the code in this checkout even when another package version is installed.
sys.path.insert(0, str(root / "src"))
import interaction_descriptors

suite = unittest.defaultTestLoader.discover(str(root / "tests"))
def test_names(group):
    for item in group:
        if isinstance(item, unittest.TestSuite):
            yield from test_names(item)
        else:
            yield item.id()
names = list(test_names(suite))
result = unittest.TextTestRunner(verbosity=2).run(suite)
report = dict(scope="synthetic mathematical identities and input/missingness contracts",
              python=platform.python_version(), numpy=np.__version__, tests_run=result.testsRun,
              tested_package=str(Path(interaction_descriptors.__file__).resolve()),
              test_cases=names, failures=len(result.failures), errors=len(result.errors), passed=result.wasSuccessful())
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
if not result.wasSuccessful():
    raise SystemExit(1)
