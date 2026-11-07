#!/usr/bin/env python3
"""Compare 12 whole-latent matrices to an external private steel cache.

This regenerates descriptors only; it neither retrains source models nor refits
downstream regressors. It writes aggregate errors/hashes, never sample data.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

from interaction_descriptors import FrozenBank


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("bank", "cards", "cache", "output"):
        p.add_argument("--" + flag, type=Path, required=True)
    a = p.parse_args()
    checks = []
    with np.load(a.cache, allow_pickle=False) as data, FrozenBank(a.bank, a.cards) as bank:
        for kind in ("NN", "YAE", "YVAE", "random"):
            for seed in (101, 202, 303):
                Z, M, names = bank.whole_latent(data["X"], list(map(str, data["input_names"])),
                                                kind=kind, seed=seed, fill_unavailable=0.)
                error = float(np.max(np.abs(Z - data[f"Z_{kind}_{seed}"])))
                masks_equal = bool(np.array_equal(M, data["M"]))
                checks.append(dict(kind=kind, seed=seed, shape=list(Z.shape),
                                   max_absolute_difference=error, masks_identical=masks_equal,
                                   source_count=M.shape[1], coordinate_count=len(names),
                                   passed=bool(error < 1e-10 and masks_equal)))
    report = dict(scope="whole-latent descriptor regeneration only; not source training or downstream replay",
                  python=platform.python_version(), numpy=np.__version__,
                  hashes={p.name: sha(p) for p in (a.bank, a.cards, a.cache)},
                  tolerance=1e-10, checks=checks, passed=all(c["passed"] for c in checks))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"matrices": len(checks), "passed": report["passed"],
                      "max_error": max(c["max_absolute_difference"] for c in checks)}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
