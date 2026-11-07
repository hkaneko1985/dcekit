"""NumPy-only inference for the v49 bank, not legacy polymer checkpoints."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .composition import audit_and_align, common_component_schema
from .extension import boundary_extension


def _silu(a):
    # Stable sigmoid without SciPy or overflow for large negative activations.
    sigmoid = np.empty_like(a)
    positive = a >= 0
    sigmoid[positive] = 1 / (1 + np.exp(-a[positive]))
    exp_a = np.exp(a[~positive])
    sigmoid[~positive] = exp_a / (1 + exp_a)
    return a * sigmoid


class FrozenModel:
    """One source, encoder kind and initialization; all saved axes stay distinct."""

    def __init__(self, weights, card, kind, seed):
        self.card = card
        self.source_id = card["source_id"]
        self.kind, self.seed = kind, int(seed)
        self.elements = tuple(card["element_vocabulary"])
        self.weights = {k: np.array(v, dtype=float, copy=True) for k, v in weights.items()}
        for key, value in self.weights.items():
            if not np.isfinite(value).all():
                raise ValueError(f"Nonfinite saved parameter: {key}")
            value.setflags(write=False)
        for key in ("x_sd", "z_sd", "y_sd"):
            if np.any(self.weights[key] <= 0):
                raise ValueError(f"Saved {key} must be strictly positive")
        w = self.weights
        if w["x_mean"].shape != (len(self.elements),) or w["x_sd"].shape != (len(self.elements),):
            raise ValueError("Source preprocessing and element vocabulary differ")
        self.latent_size = len(w["mu.bias"])
        self.coordinate_names = tuple(
            f"{self.source_id}::{kind}::{seed}::latent_{j + 1:02d}"
            for j in range(self.latent_size)) + (f"{self.source_id}::{kind}::{seed}::property_modeled",)

    @property
    def output_metadata(self):
        return {
            "source_id": self.source_id, "encoder": self.kind, "initialization": self.seed,
            "latent_scale": self.card["latent_scale"],
            "property_scale": self.card["modeled_target_scale"],
            "physical_or_catalog_target_unit": self.card["physical_or_catalog_target_unit"],
            "output_order": [*self.coordinate_names], "property_inverse_transform_applied": False,
            "property_note": "Head output is affine-unstandardized only; log/log1p scales are preserved. "
                             "Random-control property outputs are not calibrated physical predictions.",
            "boundary_operator": "B[standardized latent], B[affine-unstandardized modeled property]; "
                                 "not head(B[latent])",
            "weights_and_preprocessing_frozen": True,
        }

    def _forward(self, X):
        w = self.weights
        a = (np.asarray(X, dtype=float) - w["x_mean"]) / w["x_sd"]
        for layer in ("trunk.0", "trunk.2"):
            a = _silu(a @ w[layer + ".weight"].T + w[layer + ".bias"])
        # Deterministic posterior mean for YVAE; no log-variance or sampling.
        mu = a @ w["mu.weight"].T + w["mu.bias"]
        h = _silu(mu @ w["head.0.weight"].T + w["head.0.bias"])
        prop = (h @ w["head.2.weight"].T + w["head.2.bias"]) * w["y_sd"] + w["y_mean"]
        z = (mu - w["z_mean"]) / w["z_sd"]
        return np.column_stack((z, prop))

    def infer(self, X, components: Sequence[str], *, fraction_basis="atomic_fraction"):
        """Return latent+modeled property, support mask and lossless input audit.

        An unavailable whole-source output is NaN; zero filling is an explicit
        later operation requiring its mask. No physical mass conversion, clipping,
        target fitting, normalization, or removal of present elements occurs.
        """
        audit = audit_and_align(
            X, input_components=components,
            common_components=common_component_schema(components, self.elements),
            source_components=self.elements, input_basis=fraction_basis,
            source_basis="atomic_fraction")
        ok = np.asarray(audit.supported_mask, dtype=bool)
        out = np.full((len(ok), self.latent_size + 1), np.nan)
        if ok.any():
            aligned = np.asarray([r for r in audit.source_inputs if r is not None])
            out[ok] = self._forward(aligned)
            if not np.isfinite(out[ok]).all():
                raise FloatingPointError("Frozen source produced nonfinite outputs")
        return out, ok, audit

    def describe(self, X, components: Sequence[str], *, fraction_basis="atomic_fraction",
                 observed_pairs=None):
        """Complete fixed-schema pair blocks and whole/boundary descriptors."""
        # Validate the declared basis even if every row is outside vocabulary.
        self.infer(X, components, fraction_basis=fraction_basis)
        return boundary_extension(
            X, lambda u: self.infer(u, components, fraction_basis=fraction_basis)[0],
            components=components, coordinate_names=self.coordinate_names,
            available_components=self.elements, observed_pairs=observed_pairs)


class FrozenBank:
    """Read v49 NPZ plus its explicit model cards using allow_pickle=False.

    Use as a context manager. Source ordering is catalogue order, never sorted by
    downstream performance. Calls address one kind/seed; axes are not averaged.
    """

    _PARAMETERS = ("trunk.0.weight", "trunk.0.bias", "trunk.2.weight", "trunk.2.bias",
                   "mu.weight", "mu.bias", "head.0.weight", "head.0.bias", "head.2.weight",
                   "head.2.bias", "x_mean", "x_sd", "y_mean", "y_sd", "z_mean", "z_sd")

    def __init__(self, bank_path, model_cards_path):
        self.path = Path(bank_path)
        self._npz = np.load(self.path, allow_pickle=False)
        self._cache = {}
        try:
            self.metadata = json.loads(str(self._npz["__metadata_json__"]))
            if (self.metadata.get("schema") != "interaction_descriptor_bank.npz.v1"
                    or self.metadata.get("release") != "v49"):
                raise ValueError("This loader accepts the documented v49 bank only")
            document = json.loads(Path(model_cards_path).read_text(encoding="utf-8"))
            if document.get("schema") != "source_model_cards.v1":
                raise ValueError("Unsupported model-card schema")
            self.cards = tuple(sorted(document["cards"], key=lambda c: c["source_index"]))
            catalog = self.metadata["catalogue_original"]
            if len(self.cards) != len(catalog) or len(catalog) != self.metadata["source_count"]:
                raise ValueError("Bank/card source count mismatch")
            if document.get("source_archive_sha256") != self.metadata.get("origin_archive_sha256"):
                raise ValueError("Model cards and bank refer to different source archives")
            for i, (card, entry) in enumerate(zip(self.cards, catalog)):
                if card["source_index"] != i or card["source_id"] != entry["source"]:
                    raise ValueError("Bank/card source ordering mismatch")
                if tuple(card["element_vocabulary"]) != tuple(map(str, self._npz[f"s{i:02d}__elements"])):
                    raise ValueError("Bank/card element ordering mismatch")
            self.source_ids = tuple(c["source_id"] for c in self.cards)
        except Exception:
            self._npz.close()
            raise

    def model(self, source_id, kind="NN", seed=101):
        if kind not in ("NN", "YAE", "YVAE", "random") or seed not in (101, 202, 303):
            raise ValueError("Use NN/YAE/YVAE/random and initialization 101/202/303")
        key = (source_id, kind, int(seed))
        if key not in self._cache:
            i = self.source_ids.index(source_id)
            tag = f"s{i:02d}__{kind}__{seed}__"
            weights = {k: self._npz[tag + k] for k in self._PARAMETERS}
            self._cache[key] = FrozenModel(weights, self.cards[i], kind, seed)
        return self._cache[key]

    def whole_latent(self, X, components, *, kind="NN", seed=101,
                     fraction_basis="atomic_fraction", fill_unavailable=None):
        """Return Z, M, names in source-major/coordinate-minor fixed order.

        Default unavailable blocks are NaN. ``fill_unavailable=0.0`` explicitly
        requests the steel archive convention, and M must accompany those blocks.
        """
        if fill_unavailable is not None and not np.isfinite(fill_unavailable):
            raise ValueError("fill_unavailable must be finite or None")
        blocks, masks, names = [], [], []
        for source in self.source_ids:
            model = self.model(source, kind, seed)
            output, ok, _ = model.infer(X, components, fraction_basis=fraction_basis)
            z = output[:, :model.latent_size].copy()
            if fill_unavailable is not None:
                z[~ok] = fill_unavailable
            blocks.append(z)
            masks.append(ok)
            names.extend(model.coordinate_names[:-1])
        return np.concatenate(blocks, axis=1), np.column_stack(masks), tuple(names)

    def close(self):
        self._npz.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
