# External model assets

No trained-weight file is distributed in this proposed public source tree.
The core loader expects two **caller-supplied** files:

| File | Purpose |
|---|---|
| `v49_frozen_bank.npz` | Frozen source weights, preprocessing and element vocabularies |
| `v49_model_cards.json` | Source scales, provenance, ordering and reuse restrictions |

1. Obtain the exact research assets from the author through an authorized route.
   A public download URL has not yet been assigned. The model cards' source links
   point to underlying datasets, not to a public release of these fitted weights.
2. Keep the files outside the repository, for example `/private/models/`.
3. Verify SHA256 against `manifests/private_artifacts.json`, using
   `python scripts/check_artifacts.py --directory /private/models`.
   The checker reports absent optional replay files but only fails on mismatches
   among files actually supplied; use `--require-all` for a full private bundle.
4. Pass explicit `--bank` and `--cards` paths. The reader refuses older polymer
   checkpoints or a different release/schema. Do not substitute weights with a
   matching shape and a different scientific provenance.

The v49 archive has 24 tasks × 4 model kinds × 3 initializations. It contains no
raw source composition–property rows. Lack of raw rows does not by itself
establish permission to redistribute trained weights. The proposed MIT license
applies to this code; a weight/data release requires its own source-specific terms.

The full model cards are retained externally rather than silently reissued under
the code license. `manifests/source_catalogue.json` contains a compact provenance
index copied from their completed metadata, without raw composition/property rows.
