# Reproduction commands, evidence, and remaining scope

This release connects the recovered original source trainer, frozen descriptors,
archived steel replay, architecture comparison, six-regressor extension, and
publication exports. Every command names its input and writes to a separate
output. The status below distinguishes actual execution from recovered recipes.
No command automatically replaces the canonical source bank or downloads a
changing upstream dataset. This research directory is published in DCEKit;
the verified initial commit and review scope are recorded in
[PUBLICATION_STATUS.md](PUBLICATION_STATUS.md). No public model-download URL has
been assigned.

Run the commands below from the `interaction_descriptors/` project directory
inside the DCEKit checkout. Paths such as `/private/...` and `/work/...` are
examples of separate asset and output locations, not directories supplied by
DCEKit. Replace them with paths on your computer. The wrappers also work when
invoked by an absolute script path from another current directory.

## What was executed

| Path | Executed verification | Scope limit |
|---|---|---|
| Source training → target descriptors | Original `mp_decomp`, NN, source seed 101, actual seed 360101, 200 epochs; all 27 parameter/scaling arrays and all 143 metal-row descriptor/prediction outputs exactly equal | One final source fit; no source out-of-fold diagnostics or full-bank retraining |
| Frozen steel audit | All 126 saved metrics exact; 12 latent matrices and fixed 18 descriptors exact; original files unchanged | Zero new target fits in this stage52 audit; the separate stage51 representative replay performed 18 fits |
| Historical Figures 1–6 | Four unchanged plot scripts executed on hash-verified original inputs; all six PNGs byte-identical to the v51 manuscript images | Rendering is from recorded results; source/target estimation underlying every figure was not rerun |
| Figure 7 | New rendering of the 36 original aggregate values | Original rendering script was not recovered; pixel identity is not claimed |
| Tables 1–8, S1–S3 | Exported from recorded aggregates/metadata; all 237 numeric cells agree exactly with the displayed v51 tables | Tables 1/S2 retain authored scope/distribution statements; this is not a new data-license audit |
| Table 9, S4, S5a–c | Exported from the complete 144-cell summary and frozen candidate grids; all 343 displayed cells agree with v52 | The best individual architecture is a descriptive post-evaluation minimum |
| Figure 8 | Pipeline wrapper regenerated the final figure and comparison exports byte-identically to the manuscript build | No model fitting occurs during plotting |
| KRR and five additional regressors | Completed numerical experiments retained with their own validation; pipeline wrappers verified by fresh preparation only | The wrapper smoke did not repeat the 100 KRR or 500 new jobs |

Machine-readable evidence is in `pipeline_status.json`,
`manifests/manuscript_provenance.json`, and `verification/`. A figure or table
regeneration pass is not evidence that every model underlying it was retrained.

## Environments

Core inference still requires only Python >=3.10 and NumPy. Table exports and
asset identity checks use the standard library. For plots, use Python >=3.11:

```bash
python -m pip install -r requirements-plot.txt
```

The current replay and plotting requirements pin Pillow 13.0.0. This changes
the optional image-library dependency following the November 2026 security
review; it does not change the numerical methods or input data. The archived
stage51 replay used Pillow 12.3.0, and its evidence files are retained unchanged.
Historical image-byte comparisons above describe their recorded environments;
a changed image-library version requires a separate rendering check. Use the
current requirements for new installations rather than copying the historical
Pillow pin from an evidence file.

The verified source fit used Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0,
scikit-learn 1.8.0, and CPU PyTorch 2.10.0+cpu in an isolated environment without
system-site inheritance. Create a separate environment with that Python version:

```bash
python3.13 -m venv /work/source_env
/work/source_env/bin/python -m pip install -r requirements-source.txt
/work/source_env/bin/python -m pip install torch==2.10.0+cpu --index-url https://download.pytorch.org/whl/cpu
/work/source_env/bin/python -m pip check
```

The original environment records are retained as
`reference/source_bank/environment_v36.json` and `environment_v38.json`. The
new interpreter used the same Python release with a different recorded compiler
build; the exact numerical parity reported above is measured, rather than assumed
from a claim of an identical operating-system image.

Install `requirements-plot.txt` there as well to execute the original historical
plots. Figure 2 imports the archived PyTorch model definition even though its
saved-weight evaluation uses NumPy. The installation log records principal
scientific versions and the actual transitive packages; equality of every
transitive package to an unrecorded earlier environment is not asserted.

Downstream fitting uses each experiment's own `requirements.txt`. The maintained
steel replay retains the Python 3.13 and numerical-version contract described in
[REPLAY.md](REPLAY.md), with the current Pillow pin enforced by preflight.
Use the global `--python` option before the subcommand to select that interpreter.

The documented `reproduce.py` entry points check required external files before
creating a fitting or historical-figure working directory. An asset error must
be resolved by obtaining the identified inputs; the wrapper does not replace
them with toy data or change the archived estimator. Outputs must be new or
empty and separate from immutable input roots. Plot regeneration through
`plot-comparison` also rejects a nonempty destination. If a numerical dependency
or a computation fails after execution starts, keep the resulting diagnostic
files and use a new output directory for a fresh attempt.

For the core's mathematical and input-contract checks, use a new result file:

```bash
python scripts/verify_synthetic.py --output verification_output/synthetic_validation.json
```

This tests the source code in the current checkout and records its path. It
does not replace the shipped `verification/synthetic_validation.json`, which
remains evidence from its recorded earlier run. The default result path is the
same `verification_output/synthetic_validation.json` under the current directory;
choose another name for a subsequent run.

The separate source-fit diagnostic tests require only NumPy and do not train a
model or require external data:

```bash
python -m unittest discover -s tests_pipeline -v
```

## Obtain and identify external inputs

The public checkout contains code, source provenance, aggregate results and
verification reports. It contains no source/target observation arrays, trained
weights, or row-level prediction caches. Obtain the author's exact archived
preparations through an authorized route; no public download link has been
assigned. For a new upstream acquisition, follow the providers and snapshot
identities in [data/README.md](data/README.md) and
`manifests/source_catalogue.json`. A current upstream download is not a substitute
for an archived, filtered preparation with a different hash.

| External preparation | Required use |
|---|---|
| `Interaction_Descriptors_Expanded_Sources_v38` | Catalogue, 24 saved source inputs, corresponding original models and cached metal/oxide features for source-fit parity |
| `Interaction_Descriptors_MultiSource_Bank_v36` | Original 22-source raw-to-prepared procedure and complete original training recipe |
| v40, v41, v42, v44 roots | Minimal recorded inputs for the original Figures 1–6; exact relative files are in `historical_figure_assets.json` |
| Extracted stage51 working evidence with `fusion/` and `sources/` | Prepared inputs and nested selector caches for the new regressor comparison |
| v49 steel prepared files | Frozen steel feature/metric audit or explicit target refitting |

Set local roots in a JSON file based on `manifests/assets.example.json`.
Relative paths are resolved against that JSON file's directory. Include only
roots you possess; `check-assets` reports the scope actually checked. The full
v38 identity audit checks every source/model, whereas `source-fit` can run from
the minimal files for the selected source/seed/target.

```bash
python scripts/reproduce.py list
python scripts/reproduce.py check-assets --assets /private/assets.json --output /work/asset_check.json
```

## Exact original source fit and target feature parity

The executable fit uses unchanged `reference/source_bank/source_common.py` and
the original scaling, loss, optimizer, epoch and seed rules recovered in
`train36.py`/`train38.py`. The v38 catalogue retains the original task order.
The following is the preselected bounded check, now portable with explicit paths:

```bash
python scripts/reproduce.py --python /work/source_env/bin/python source-fit \
  --archive-root /private/Interaction_Descriptors_Expanded_Sources_v38 \
  --source mp_decomp --kind NN --seed 101 --target-features metals \
  --output-dir /work/source_fit_mp_decomp_NN_101
```

Required files are `data/source_catalog.json`, `sources/mp_decomp.npz`,
`models/mp_decomp/NN/seed101.npz`, `data/target_metals.npz`, and
`features/metals/NN/seed101/mp_decomp.npz`. The wrapper checks their original
hashes and writes a new model, regenerated features, protocol and per-array
verification. Target labels are not read or used. All target compositions stay
intact and source component compatibility is preserved.

Some source/target combinations have no compatible target rows. In that case,
the modeled-property comparison is recorded as
`not_applicable_no_compatible_rows`, with zero comparison rows and a `null`
maximum property difference. The applicability mask, latent feature blocks and
standardized feature sentinels are still checked. An absent property comparison
does not turn a mask or latent mismatch into a pass.

The recovered original `train36.py` trains 22 tasks and `train38.py` trains the
two additions, using three architectures and three source seeds. Together these
define 216 fitted final models and 72 untrained controls. Original three-fold
source diagnostics are additional fits. These scripts expect their original
archive layout and write into it. For an explicitly chosen full rerun, first
copy the relevant archive to a separate work directory and run its original
`code/train36.py` or `code/train38.py` with the pinned interpreter. Do not execute
the flat `reference/source_bank/` copies in place: they are immutable historical
code evidence, while `source-fit` is the supported isolated entry point.

`prepare36.py` and its `prepare35.py` dependency retain the original source
preparation. `verify_inputs38.py` reconstructs the two added sources from archived
raw tables and checks all 865 inherited JARVIS parent exclusions. Those raw-table
reconstruction procedures were recovered but were not rerun in this stage.
No replacement training data or substitute trainer is labeled a historical
reproduction. Earlier polymer encoders remain a separate historical route.

## Frozen steel descriptors, saved predictions, and refitting

```bash
python scripts/reproduce.py --python /work/steel_env/bin/python steel-audit \
  --input-dir /private/stage51/sources --output-dir /work/steel_audit --mode audit
```

`audit` regenerates features and recomputes recorded metrics without new target
fits. `representative` explicitly requests the original 18 outer fits;
`full` explicitly requests the 540-fit historical target replay. Neither trains
source networks. Use [REPLAY.md](REPLAY.md) for exact input/version requirements.

## KRR architecture and six-regressor comparisons

```bash
python scripts/reproduce.py --python /work/comparison_env/bin/python architecture \
  --input-dir /private/stage51/fusion --output-dir /work/krr_reproduction \
  --workers 4 --prepare-only

python scripts/reproduce.py --python /work/comparison_env/bin/python comparison \
  --stage51-root /private/stage51 --output-dir /work/regressor_reproduction \
  --workers 5 --prepare-only
```

`--prepare-only` verifies inputs and prepares an isolated work copy without
predictive fitting. Omit it to run the corresponding full experiment in a new
output directory. Once prepared, resume using that directory's runner directly.
The KRR route refits nested Boruta; the five-regressor extension reuses all 198
verified stage51 selector decisions and performs no new selector fits. It
retains the original preprocessing, source seeds, split indices and candidate
grids. Frozen result metadata in the public experiment is under
`frozen_reference/`, so it cannot masquerade as prepared row-level inputs.

The completed extension contains 500 new jobs, 1,800 outer fits, and 62,700
prediction events. It inherits the 100 KRR jobs separately. No new material
cohort was acquired. The polymer route uses boundary features and six region
holdouts; the inorganic route uses whole-composition features and pooled
held-out predictions per repeat. Their basis definitions remain explicit.
For polymer, `B = X + D9`; for the inorganic comparisons,
`B = X + D18 + M24`. Earlier Table 4's composition-plus-fixed baseline does not
include the same mask basis. Do not merge those values as if they were identical
conditions. Neither latent axes nor independently trained predictions are
averaged to create these descriptors or the reported configuration means.

## Rebuild publication tables and figures

The aggregate table commands run from the public checkout alone:

```bash
python scripts/reproduce.py tables --output-dir /work/tables_1_to_8
python scripts/reproduce.py regressor-tables --output-dir /work/tables_9_S4_S5
python scripts/reproduce.py plot-comparison \
  --summary experiments/regressor_comparison/frozen_reference/summary.csv \
  --output-dir /work/figure_8
```

The first export covers Tables 1–8 and S1–S3, including an unrounded numeric map
for the main experimental tables. Source catalogue counts and maximum component
counts were checked directly against the 24 saved source input matrices.
The second produces Table 9, the six grid definitions in S4, and all 144 values
in S5a–c. The Figure 8 exporter also writes an 18-row descriptive comparison CSV.
Both complete-matrix exporters reject duplicate/missing conditions and incomplete
seed/repeat aggregation. They perform zero model fits.

```bash
python scripts/reproduce.py --python /work/source_env/bin/python historical-figures \
  --assets /private/assets.json --output-dir /work/figures_1_to_7
```

The historical command requires v40/v41/v42/v44 assets, copies only the declared
files to isolated work layouts, verifies unchanged original code and data hashes,
runs the original four plotting scripts, and collects main Figures 1–6. It also
renders Figure 7 from the bundled aggregate CSV. Figure 2 uses three archived
stress-model weights for its held-out example; the other main historical plots
use recorded aggregate inputs. Original archives remain unchanged. The map in
`manifests/manuscript_provenance.json` gives each figure/table's dependency,
command, and exact verified or recipe-only status.

## Remaining work

Full raw-to-prepared source reconstruction, all 216 source fits plus controls and
source diagnostics, and every historical downstream estimation branch are still
outside the executed stage52 scope. The whole pipeline therefore is not certified
as a single complete raw-source-to-all-figures rerun. Original Figure 7 rendering
code remains unrecovered, although its numerical inputs and new rendering are
available. External data/weight redistribution decisions, confirmation of the
actual DCEKit commit containing this directory, and an eventual archived
software identifier remain release tasks.
