# Data setup and acquisition

No raw third-party data, target labels, cached feature matrices or fitted weights
are included in this source tree. The synthetic CLI does not need any of them.
Keep authorized downloads and research preparations outside the repository.

## Exact private replay inputs

The archived numerical replay requires `steel_v46_replay_inputs.npz`,
`steel_v46_oof_predictions.csv`, `steel_v46_summary.csv` and the external v49 bank.
The feature-only checker also needs `v49_model_cards.json`. Their exact hashes are
in `manifests/private_artifacts.json`. Request these preparations from the author
through an authorized route; no public asset URL has yet been assigned.

The cache includes the recorded atomic-fraction inputs, splits, masks and latent
matrices. It is a study-specific prepared artifact. Downloading a current copy of
a parent dataset does not recreate that cache or justify silently changing hashes.

## Underlying source data

`manifests/source_catalogue.json` lists the 24 source task identifiers, properties,
units, transforms, parent datasets and provenance links recorded in the model
cards. For each source, follow that original provider's access and license process,
retain the original bytes and license/citation information, and document the exact
snapshot, subset, aggregation and exclusion procedure. Some source tasks use
restricted archived subsets rather than the full public database. The inference
package does not download changing live source tables. The separate
[source-fit entry point](../PIPELINE.md) reproduces the archived training recipe
from caller-supplied exact prepared inputs; full raw-source reconstruction has
not been executed in stage52.

For the steel parent data, the manuscript records the official Matbench/matminer
`matbench_steels` dataset, derived from Citrine steel-strength records. Official
metadata are at:

https://raw.githubusercontent.com/hackingmaterials/matminer/main/src/matminer/datasets/dataset_metadata.json

Use that provider's download location for `matbench_steels`, preserve its compressed
bytes, and validate the archived compressed SHA256:

`473bc4957b2ea5e6465aef84bc29bb48ac34db27d69ea4ec5f508745c6fae252`

A changed provider snapshot requires a separately identified analysis. Do not
rewrite the expected hash. The modeled steel response is yield strength in MPa.
Formula coefficients were interpreted as stoichiometric amounts and normalized
to atomic fractions; they were not divided by atomic weights again as if they
were weight percentages. All declared constituents were retained.

The UCI superconductivity parent is identified by DOI `10.24432/C53P47`; the
Petousis dielectric data by DOI `10.1038/sdata.2016.134`; the experimental-gap
parent by DOI `10.1021/acs.jpclett.8b00124`. Check each provider's current terms
before redistribution. Public download access does not grant a blanket license
to all derived data. The data and model directories here contain instructions,
not licensed copies of those datasets.

## Reuse for a new target

Record the fraction basis, complete component schema, target units, composition
duplicate keys, source-property exclusions, parent/study overlaps, ratio support,
train/validation/test partitions and source/target preprocessing. Freeze that
record before evaluating a confirmation task. Unknown source components must
remain in the complete composition and make the corresponding source unavailable.
