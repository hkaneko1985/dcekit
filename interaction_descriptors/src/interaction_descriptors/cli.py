"""Small CLI for a fully synthetic demo and caller-supplied private weights."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .frozen import FrozenBank, FrozenModel


def synthetic_model():
    """Deterministic random toy network, not a trained scientific source model."""
    rng = np.random.default_rng(20261107)
    w = {}
    for layer, before, after in (("trunk.0", 3, 5), ("trunk.2", 5, 4),
                                  ("mu", 4, 2), ("head.0", 2, 4), ("head.2", 4, 1)):
        w[layer + ".weight"] = rng.normal(0, .3, (after, before))
        w[layer + ".bias"] = rng.normal(0, .1, after)
    w.update(x_mean=np.zeros(3), x_sd=np.ones(3), z_mean=np.zeros(2),
             z_sd=np.ones(2), y_mean=np.array(0.), y_sd=np.array(1.))
    card = dict(source_id="synthetic", element_vocabulary=["A", "B", "C"],
                latent_scale="synthetic dimensionless coordinates",
                modeled_target_scale="synthetic arbitrary units",
                physical_or_catalog_target_unit="synthetic arbitrary units")
    return FrozenModel(w, card, "NN", 101)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("synthetic", help="No private data or model download required")
    demo.add_argument("--output", type=Path, required=True)
    infer = sub.add_parser("infer", help="Whole and all-pair outputs for one frozen v49 source")
    infer.add_argument("--bank", type=Path, required=True)
    infer.add_argument("--cards", type=Path, required=True)
    infer.add_argument("--input", type=Path, required=True, help="CSV; component names in header, fractions only")
    infer.add_argument("--source", required=True)
    infer.add_argument("--kind", choices=["NN", "YAE", "YVAE", "random"], default="NN")
    infer.add_argument("--seed", type=int, choices=[101, 202, 303], default=101)
    infer.add_argument("--fraction-basis", choices=["atomic_fraction", "mass_fraction"], required=True)
    infer.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "synthetic":
        model = synthetic_model()
        names = ("A", "B", "C", "D")
        X = np.array([[1., 0, 0, 0], [.4, .6, 0, 0], [.2, .3, .5, 0], [.5, 0, 0, .5]])
        result = model.describe(X, names, observed_pairs=[("A", "B")])
        metadata = model.output_metadata
        metadata["synthetic_only"] = True
        metadata["example_rows"] = ["pure", "binary", "ternary", "unsupported D retained"]
    else:
        with args.input.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            names = tuple(next(reader))
            X = np.array([[float(v) for v in row] for row in reader], dtype=float)
        with FrozenBank(args.bank, args.cards) as bank:
            model = bank.model(args.source, args.kind, args.seed)
            result = model.describe(X, names, fraction_basis=args.fraction_basis)
            metadata = model.output_metadata
        metadata["empirical_pair_support"] = "not audited; all codes -1"
    metadata.update(pair_schema="all unordered pairs in input-header order; activity masks required",
                    unsupported_outputs="NaN; original complete compositions retained",
                    pair_support_codes={"-1": "unaudited", "0": "not in supplied observed set", "1": "observed"},
                    n_rows=len(X), n_pairs=len(result.pair_indices))
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "descriptors.npz", **result.arrays())
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "rows": len(X),
                      "supported_rows": int(result.row_vocabulary_supported.sum()),
                      "pair_blocks": len(result.pair_indices)}))


if __name__ == "__main__":
    main()
