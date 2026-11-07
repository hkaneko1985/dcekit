# Interaction-aware descriptors

[日本語ガイド](README_JA.md) · [DCEKit integration](DCEKIT_INTEGRATION.md) ·
[Reproduction commands](PIPELINE.md) · [Release review](RELEASE_NOTES.md) ·
[Publication status](PUBLICATION_STATUS.md)

Reusable, frozen composition-to-descriptor functions for materials property
prediction. The accompanying manuscript is **Reusable interaction-aware descriptors from frozen property models for materials prediction**,
by **Hiromasa Kaneko** (ORCID [0000-0001-8367-6476](https://orcid.org/0000-0001-8367-6476)).

**Public code:** this research directory is hosted in
[DCEKit](https://github.com/hkaneko1985/dcekit/tree/master/interaction_descriptors).
The verified initial public baseline is
[commit 0d8d31e9d061208f85d3ef4d8de9dc587c43ca89](https://github.com/hkaneko1985/dcekit/tree/0d8d31e9d061208f85d3ef4d8de9dc587c43ca89/interaction_descriptors).
The public-file review and subsequent checksum metadata correction are documented
in [PUBLICATION_STATUS.md](PUBLICATION_STATUS.md). No public release tag,
software DOI or manuscript DOI is asserted.
The author code is supplied under MIT; see [LICENSE](LICENSE).
The code license does not license third-party source data or trained weights.
Those assets are intentionally external pending source-specific redistribution
review. The source-free synthetic example below works immediately.

## What this release implements

- Full-composition input alignment without deleting any present component.
- NumPy-only inference from the documented v49 bank: NN, YAE and deterministic
  YVAE latent means, plus archived untrained `random` controls.
- Whole-composition latent and modeled-property outputs; complete fixed-order
  normalized-pair blocks; endpoint-linear and Kohler-type boundary extensions.
- Explicit vocabulary-compatibility, absent-pair and empirical-support metadata.
- A standalone check against the 12 cached steel whole-latent matrices.
- An unchanged historical steel replay runner, with its original dependency and
  input requirements, in `scripts/replay_v49.py`.

The release also provides explicit [reproduction entry points](PIPELINE.md) for
archived source training, descriptor checks, downstream comparisons, and all main
table/figure exports. The unchanged v36/v38 source trainers and prepared-input
identities have been recovered. One preselected source fit has been rerun in an
isolated environment and matches all 27 parameter/scaling arrays exactly; its
143-row metal descriptor outputs also match exactly. The full source bank and
every historical target-training branch have not been rerun. Earlier polymer and
low-component checkpoints used different preprocessing and must not be loaded
through `FrozenBank`. This remains a scoped reproduction release, with verified
paths and unexecuted historical recipes distinguished in the pipeline status.

## Installation and a runnable example

Using your chosen Python environment, install this research package explicitly
from the DCEKit repository root:

```bash
python -m pip install -e ./interaction_descriptors
cd interaction_descriptors
```

The existing DCEKit package and this research package have separate installation
definitions. Installing `dcekit` alone does not install this directory. All
remaining commands below run **inside `interaction_descriptors/`**. If you
extracted this directory on its own, start there instead. A virtual environment
is recommended for its dependencies. For a standalone extracted folder, run
`python -m pip install -e .` from inside that folder instead.

Python 3.10 or newer and NumPy are sufficient for the core. No PyTorch, SciPy,
RDKit, downloaded materials data or model weights are needed for this example.

```bash
interaction-descriptors synthetic --output example_output
python scripts/verify_synthetic.py --output example_output/synthetic_validation.json
```

The example uses a deterministic toy network and four artificial compositions:
a pure component, a binary, a ternary, and a composition containing an unsupported
component `D`. It writes `descriptors.npz` and `metadata.json`. These values are
demonstrations, not material-property predictions. The fourth composition remains
intact and its unavailable whole/boundary outputs are NaN. The six pair positions
for the fixed four-component schema are present in every row.

For a source checkout without installation, prefix commands with `PYTHONPATH=src`
(or set the equivalent environment variable on your operating system):

```bash
PYTHONPATH=src python -m interaction_descriptors synthetic --output example_output
```

## Composition and source input contract

Each dataset has a declared, ordered component schema. A changing number of
components means a changing number of **nonzero fractions** in that fixed schema.
Use the same column order in training and evaluation; add genuine absent components
as zeros before generating features. Never infer chemical identity from column
position. The input auditor copies the existing project specification and checks:

1. Exact, unique component names and matching row width.
2. Finite, nonnegative fractions summing to one within absolute tolerance `1e-8`.
3. Explicit fraction basis. The v49 bank accepts **atomic fractions**. Mass
   fractions must be converted explicitly, with documented atomic weights, before
   this API is called. The API does not normalize, clip, or silently convert them.
4. Every positive component must occur in a source's saved vocabulary. Even a
   positive unsupported fraction of `1e-12` makes that whole-source block
   unavailable. A missing species is never discarded and the rest renormalized.

`audit_and_align` forms a lossless common composition and then maps compatible
rows to the original source order. Input compatibility alone establishes neither
empirical pair support, ratio coverage, appropriate phase/processing conditions,
nor predictive usefulness. In particular, retain C, N, Si and oxygen whenever they
are in the declared material composition.

When source training compositions are available, `inspect_source_observations`
and `audit_with_source_observations` additionally distinguish declared vocabulary
from components actually present with positive fractions in those supplied source
rows. An all-zero source column cannot establish observed component support.
These helpers preserve the original vocabulary mask as well as the stricter
observed-component mask. Neither mask proves observed pair or ratio support.

### Frozen v49 inference

The external NPZ contains 24 source tasks, 216 fitted networks and 72 untrained
controls. These are dataset–property tasks, not 24 independent databases. One
architecture/initialization has 24 independent eight-coordinate source blocks,
giving 192 nominal coordinates. The reader checks the bank schema/release,
source-archive identity, card/source ordering and element vocabulary.

For each source, the saved input mean and standard deviation are applied first.
Two SiLU layers feed the latent mean. The returned latent descriptor is
`(mu - z_mean) / z_sd` using **saved source** statistics. YVAE does not draw a
random latent sample. A separate saved head is evaluated on the unstandardized
latent mean and its output is affine-unstandardized using `y_mean` and `y_sd`.
All parameters remain frozen; no target labels enter descriptor generation.

The reader concatenates source blocks in the original catalogue order, then latent
coordinate order. It never aligns or averages independently fitted latent axes,
including axes from different initializations. An optional downstream prediction
ensemble is a different operation and is outside this inference API.

```python
import numpy as np
from interaction_descriptors import FrozenBank

# Illustrative atomic fractions; provide real, audited compositions for your task.
X = np.array([[0.7, 0.3], [1.0, 0.0]])
components = ["Fe", "Ni"]
with FrozenBank("/private/models/v49_frozen_bank.npz",
                "/private/models/v49_model_cards.json") as bank:
    Z, M, names = bank.whole_latent(X, components, kind="NN", seed=101)
    source = bank.model("supercon_all_Tc", kind="NN", seed=101)
    pairs = source.describe(X, components)
    units_and_order = source.output_metadata
```

By default, unavailable source blocks in `Z` are NaN. For an explicitly specified
mask-aware downstream design, `fill_unavailable=0.0` reproduces the historical
steel convention; retain the returned `M` with those blocks. This is a neutral
standardized feature sentinel, not an imputed physical property. Comparisons of
`X`, `X+D`, `X+D+M` and `X+D+M+Z` help separate applicability information from
source descriptors. Fit constant-column filtering and downstream scaling using
the applicable target training partition only.

## Whole, pair-indexed and boundary outputs

Let `g(x)` be one fixed source output on a declared scale, possibly vector-valued.
It can be the standardized latent vector or the affine-unstandardized source
property. For component fractions satisfying $x_i\geq0$ and $\sum_i x_i=1$,
define pure endpoints $e_i$ and, for a pair $(i,j)$,

$$s_{ij}=x_i+x_j,\qquad t_{ij}=x_i/s_{ij},\qquad
u_{ij}=t_{ij}e_i+(1-t_{ij})e_j.$$

The pair departure uses **model-predicted endpoints on the same output scale**:

$$\Delta_{ij}[g]=g(u_{ij})-t_{ij}g(e_i)-(1-t_{ij})g(e_j).$$

The endpoint-linear and boundary-extended outputs are

$$L[g](x)=\sum_i x_i g(e_i),\qquad
B[g](x)=L[g](x)+\sum_{i<j}s_{ij}^2\Delta_{ij}[g].$$

At `s=0`, the corresponding correction is defined as zero without dividing by
zero. When either endpoint fraction is zero, the correction is also zero.
`pair_raw` uses an exact zero sentinel for these absent pairs; it does not pretend
that the mathematical pure-endpoint value `g(u)` is zero. `pair_active` records
which pair blocks were actually evaluated.

The result object keeps all `p*(p-1)/2` unordered pair positions for the declared
input schema, ordered lexicographically by **column index**, not by alphabetical
component names. No pair is removed because it is absent in the current batch.
Both normalized pair fractions are computed directly as `x_i/s` and `x_j/s`;
the second fraction is not obtained by subtraction from one. This preserves a
strictly positive trace fraction when the dominant fraction rounds to one.

| Result | Shape | Meaning |
|---|---|---|
| `composition` | n × p | Original fractions, with every input component |
| `whole` | n × q | `g(x)` |
| `vertices` | p × q | Predicted `g(e_i)`; NaN if vocabulary-incompatible |
| `unary` | n × q | `L[g]` |
| `pair_raw` | n × pairs × q | Complete normalized active-pair outputs |
| `pair_excess` | n × pairs × q | `s**2 * Delta[g]` for each pair |
| `boundary` | n × q | `B[g]` |
| `pair_indices` | pairs × 2 | Fixed pair indices in `components` |
| `pair_active` | n × pairs | Both input fractions are strictly positive |
| `pair_vocabulary_supported` | pairs | Both names belong to the source vocabulary |
| `pair_empirical_support` | pairs | -1 unaudited, 0 unobserved, 1 observed |
| `row_vocabulary_supported` | n | Every positive component is source-compatible |

For a v49 source, `q=9`: eight standardized latent coordinates followed by one
modeled-property output. Slice `[..., :8]` for latent-only and `[..., 8:]` for
prediction-only features. Pair outputs may be flattened in the documented
pair–coordinate order and concatenated by fixed source order. Preserve their
masks and ordering metadata. `boundary` does not replace storage of pair blocks.

Empirical support is not available from the model cards alone, so the default is
`-1`, including for vocabulary-compatible pairs. To supply an audited complete
observed-pair set, use `model.describe(..., observed_pairs=[("Fe", "Ni"), ...])`.
Pairs absent from that supplied set become `0`. Predictions for vocabulary-valid
but source-unobserved pairs are retained as extrapolations. Occurrence does not
establish ratio coverage; audit source observation counts and ratio ranges
separately. Pure endpoints may themselves be extrapolations.

### What the extension guarantees

For a continuous `g` on a fixed compatible simplex, `B[g]` is continuous and
matches `g` on pure and binary compositions. If

$$g(x)=\sum_i a_i x_i+\sum_{i<j}b_{ij}x_i x_j,$$

then $s_{ij}^2\Delta_{ij}[g]=b_{ij}x_i x_j$, and the extension exactly preserves
that quadratic response, coordinate by coordinate. This is a representation
identity conditional on the learned function, not a guarantee that training
discovers the physical response. A ternary term such as `x1*x2*x3` vanishes on
every binary face and cannot be identified from binary observations alone.

This is an application of the classical Kohler-type construction, not a newly
invented thermodynamic law. The manuscript reference is Barron, L. M. (1976),
*Multicomponent calulations. I. General excess and correction free energy solution
models*, Geochemical Journal 10, 137–144,
[doi:10.2343/geochemj.10.137](https://doi.org/10.2343/geochemj.10.137).
The general frozen-feature/multiple-source concepts also predate this work.

The continuity statement is about the compatible descriptor map: source masks,
active-component minima/maxima or a downstream tree model can be discontinuous.
Raw pair zero sentinels are likewise not a claim of continuous pair-output columns.

## Output scales and operator order

Property outputs in this package stay on the source **modeled target scale** after
affine unstandardization. Always read `output_metadata` and the original cards.
Some sources use `log1p`; `mp_bulk` and `mp_shear` already have catalogue targets
in `log10(GPa)` even though their later transform field says `identity`. Latent
coordinates are dimensionless and do not have physical-property units.

No log/exponential inverse transform is performed before or after aggregation.
For a nonlinear inverse `T`, `T(B[g])` generally differs from `B[T(g)]`; choose and
document such a change explicitly if your scientific application requires it.
Similarly, prediction descriptors are `B[H(E(x))]`, evaluated on pure/pair inputs
before aggregation. They are **not** the head applied to `B[E]`. Affine changes
of output commute with the mathematical extension on a normalized simplex;
arbitrary nonlinear changes do not. Random-control property outputs have no
calibrated physical interpretation.

## Using private assets from the CLI

The CSV header must contain the exact component names; all remaining fields are
fractions. Do not include target labels or identifiers as composition columns.

```bash
interaction-descriptors infer \
  --bank /private/models/v49_frozen_bank.npz \
  --cards /private/models/v49_model_cards.json \
  --input /private/data/compositions.csv \
  --fraction-basis atomic_fraction \
  --source supercon_all_Tc --kind YVAE --seed 101 \
  --output example_output
```

See [models/README.md](models/README.md), [data/README.md](data/README.md) and
[manifests/private_artifacts.json](manifests/private_artifacts.json). No public
weight download URL is invented here. Source-specific terms must be checked
before distributing data or weights; public accessibility is not a common license.

## Source–target separation and evaluation

For every new target, audit original studies, composition duplicates, property
definitions, processing conditions, source-target property overlap, and target
leakage. The v49 bank inherited exclusion of the 865-composition archived JARVIS
parent set; that exclusion is **target-specific**. The expanded bank contains
many earlier gap and superconductivity targets and must not be applied to those
historical tasks without filtering and, where required, retraining. The inference
package cannot establish those exclusions from weights alone.

Freeze source choices, representation forms, target-label budget, comparison
conditions and splits before evaluating errors on a new confirmation dataset.
Select regressors and preprocessing within target training data. The study's six
downstream regressors are PLS, LASSO, KRR, SVR, LightGBM and ordinary GBDT. Source
selection, if used, belongs inside target validation. Do not select a source
initialization using target test errors or average latent axes across sources.

The archived steel evaluation uses 312 compositions, three outer folds repeated
twice, 64 fitting labels from each 208-record outer training pool, and three inner
folds. The 144 unused training-pool records are not added to a fit. Each test fold
contains 104 records. These custom splits are not official Matbench leaderboard
scores. The repeated predictions/initializations are not independent specimens.
Metrics averaged across repetitions are distinct from prediction-ensemble metrics.

The development comparisons show positive and negative transfer. The fixed
specification steel evaluation did not establish stable improvement over strong
composition-only models. Do not present these descriptors as universal accuracy
improvements, unique microscopic interaction coefficients or causal explanations.
Reusing development data after methodological changes does not create a new
independent confirmation dataset.

## Verification and its limits

```bash
python scripts/verify_synthetic.py --output example_output/synthetic_validation.json
python scripts/verify_steel_cache.py \
  --bank /private/models/v49_frozen_bank.npz \
  --cards /private/models/v49_model_cards.json \
  --cache /private/data/steel_v46_replay_inputs.npz \
  --output verification/steel_whole_latent_validation.json
```

The focused synthetic tests cover pure/binary agreement for a nonlinear frozen
network, quadratic exactness, complete pair indexing, unsupported and absent
components, source order and fraction-basis constraints. They also include the
eight existing C11 composition-input tests and two observed-component support
tests, with only their package imports adapted, and a trace-fraction regression
test and a name-validation test added during the DCEKit release review
(16 tests in total). The private-cache check
regenerates all 12 whole-latent matrices (4 kinds × 3 initializations), each of
shape 312 × 192, and checks source masks independently. JSON evidence records
versions, input hashes, tolerances and numerical discrepancies, with no sample
data or weights. The included run passed with maximum absolute latent difference
`2.0872192862952943e-14` and identical masks; the tolerance is `1e-10`.

The NumPy implementation uses a stable sigmoid and can differ at floating-point
roundoff from the archived SciPy `expit` path. These tests verify the specified
inference and extension contracts, not new predictive gains. They do not establish
full source-training reproduction. See [REPLAY.md](REPLAY.md) for the distinct
historical downstream replay and its stricter environment requirements.

## Provenance, dependencies and citation

The input auditor is reused from the project's C11 composition specification.
The boundary construction follows the project v33 `extension.py`/`descriptors33.py`
and v48 Methods; v49 inference follows v44 `bank43.py` and v49's frozen replay,
with explicit scale/missingness metadata added here. File hashes and provenance
are recorded in [manifests/implementation_provenance.json](manifests/implementation_provenance.json).
Historical and current inference paths remain separate.

Use [CITATION.cff](CITATION.cff) for software attribution. Its sole author is
Hiromasa Kaneko. Its repository field identifies the confirmed DCEKit host;
it contains no unassigned DOI or invented commit identifier. The citation file
applies to this research directory and remains inside it.
No third-party Boruta implementation is vendored. Any separate analysis using
Boruta should declare the external `Boruta` package and its own license/version;
it is not a core dependency of this inference package.


## Additional architecture comparison

`experiments/architecture_comparison` contains the complete executable KRR
comparison of individual NN/YAE/YVAE blocks, their concatenation, and nested
Boruta selection. It retains the original target-specific source exclusions and
splits. The original source archives or prepared inputs are external assets;
see `PUBLIC_RELEASE.md` in that directory. No weights or raw rows are present in
this proposed public tree.

The completed 100-job experiment performed 360 outer fits and 198 training-only
Boruta fits. Polymer modulus favored NN alone (327.74 MPa), metal volume favored
concatenation (3.11164 Å³/atom), and oxide formation favored concatenation with
Boruta (0.46022 eV/atom) within this added comparison. These are descriptive,
developmental results. All eight conditions and baseline definitions are in the
reference CSV; no universal improvement or significance is inferred.

## Six-regressor extension

`experiments/regressor_comparison` adds PLS, LASSO, SVR, LightGBM, and GBDT
to the KRR conditions, with the same source models, inputs, source seeds and
ordered target partitions. All 500 new jobs completed (1,800 outer fits and
62,700 prediction events); the 100 KRR jobs remain separately identified.
The 198 training-specific Boruta selectors were verified and reused, with no
new selector fits. All 144 target–regressor–representation aggregate cells are
in `experiments/regressor_comparison/frozen_reference/summary.csv`.

Metal-volume concatenation with SVR achieved 2.85801 Å³/atom, 33.24% below its
common basis and 1.75% below its best single architecture. Boruta improved
concatenation for all six oxide regressors, while the best original representation
remained stronger than the added conditions for each polymer regressor. These
are descriptive comparisons on the same development cohorts. The release review
corrected a trace-fraction boundary calculation in the reusable core, without
editing or rerunning the archived routines behind those numerical experiments.

## Fresh-environment replay completed in stage51

A new Python 3.13.5 environment was created without system-site inheritance.
The eight pinned scientific dependencies were absent before installation and
then installed at the recorded versions. The unchanged historical runner
reconstructed 12 latent matrices and the fixed 18 descriptors exactly. All 126
saved aggregate metrics matched. Representative NN/seed-101/repeat-0 refitting
for six regressors yielded 18 outer fits and 1,872 predictions with maximum
absolute difference zero. The full 540-fit historical replay was not rerun in
this fresh environment. Source networks were not retrained.

`verification/fresh_environment_replay.json` is the wrapper's attestation. The
unchanged runner's nested flag remains false because that runner does not
perform or attest installation; the wrapper records the missing evidence.
See `REPLAY.md` for exact commands and scope.

## Funding and computational assistance

This work was supported by Grants-in-Aid for Scientific Research (KAKENHI)
(Grant Numbers 24K08152 and 24K01234) from the Japan Society for the Promotion
of Science. ChatGPT (OpenAI) assisted with code preparation, computational
verification, and documentation. Executed tests, source hashes, and numerical
comparison evidence are retained; final authorship and review remain with
Hiromasa Kaneko.
