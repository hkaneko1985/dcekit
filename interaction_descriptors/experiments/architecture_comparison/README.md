# Frozen-descriptor concatenation experiment (stage 51)

This package is an **additional analysis of previously used development cohorts**.
It does not replace the original six-regressor benchmark or provide an independent
external confirmation. No source encoder is trained or updated here.

## Question

Does appending the latent coordinate blocks from NN, YAE, and YVAE improve the
saved target prediction problems, and does nested Boruta selection improve the
combined representation when its dimension becomes large?

The three blocks are concatenated in NN, YAE, YVAE order at each of the original
source seeds (101, 202, 303). Unrelated latent axes are never averaged. Seed
summaries are means of separately evaluated models, not a prediction ensemble.
The YVAE block uses the saved latent mean, without resampling the latent variable.

## Fixed scope

| Case | Frozen representation | Target rows | Original outer splits | Protected basis | Combined learned coordinates |
|---|---|---:|---|---|---:|
| Polymer Young's modulus | Boundary-consistent pair latent | 49 | Six composition-region holdouts, 30–33 training rows | Three composition fractions + nine fixed terms | 72 |
| Metal atomic volume | Whole-composition latent | 143 | Three element-system group holdouts × two repeats, 64 training rows | 60 composition fractions + 18 fixed terms + 24 applicability masks | 576 |
| Oxide formation energy | Whole-composition latent | 120 | Three element-system group holdouts × two repeats, 64 training rows | 39 composition fractions + 18 fixed terms + 24 applicability masks | 576 |

All source tasks and applicability masks are inherited without alteration. The
polymer target property is excluded from its three source-property blocks. For
inorganic cases the inherited 24-source bank carries the target-composition
exclusions documented in its source catalog. This stage verifies input hashes
and mask consistency; it does not re-audit all original source-training rows.

Both the composition-only comparator (C) and protected fixed-basis comparator
(B) are retained. NN, YAE, YVAE, random-encoder, concatenation, and concatenation
plus Boruta comparisons all append to the same task-specific B. The random
control uses the original untrained encoder, without an invented three-family
random concatenation.

For the two inorganic cases, **B = X + D + M**. This differs from historical
manuscript tables using **X + D**, which omit source applicability indicators.
The additional numerical audit `audit_historical_basis51.py` rechecks X + D
against the historical predictions, and compares saved X + D + M predictions.
That audit is recorded separately from the fixed 100 configuration-repeat jobs;
it does not change the comparisons or select a new baseline after seeing errors.
Percent reductions calculated against these two baselines must not be conflated.

The additional final regressor is **KRR**, with the unchanged 30-candidate saved
grid. The original PLS, LASSO, KRR, SVR, LightGBM, and GBDT benchmark remains a
separate result set. This bounded analysis does not claim evaluation of new
concatenation features across all six regressors.

## Nested selection and preprocessing

BorutaPy 0.4.3 uses a RandomForestRegressor with 128 trees, maximum depth 5,
`max_features="sqrt"`, and one worker per selector. The frozen Boruta parameters
are `max_iter=100`, `perc=100`, `alpha=0.05`, `two_step=True`, and no optional
early stopping. BorutaPy's loop may perform at most 99 testing iterations with
this parameter value; the actual count is recorded.

Boruta is fitted separately on every saved inner-training partition and once on
each outer-training set. Validation and test labels never enter the corresponding
selector. All numerical constant-column filtering and standardization also use
only that fit's training rows. The KRR kernel divides squared distance by the
number of retained, varying columns after selection, as in the saved routine.

The selector sees the full base-plus-latent candidate matrix. The original basis
is protected from supervised removal, and only **statistically confirmed learned
coordinates** are appended. If no learned coordinate is confirmed, that fit uses
the protected basis alone. Tentative and rejected learned coordinates are not
silently substituted. Numerical training-constant columns are omitted by the
unchanged KRR preprocessing; the complete composition schema remains in the
saved inputs and no positive input component is zeroed or renormalized.

`AuditedBoruta` overrides only the statistical-test hook to copy its returned
decisions. It does not change the tests. Statistical `confirmed`, `unresolved`,
and `rejected` decisions are recorded separately from the library's weak
rough-fix flag. The weak flag is not used for selection. This also avoids treating
a display/ranking label as proof of a statistical rejection.

## Aggregation

For the polymer case, the RMSE reported for each source seed is the unweighted
mean of the six saved outer-region RMSE values, matching the original benchmark.
The 49-row target dataset contributes 101 prediction events from 47 distinct test
rows; two target rows are never in the six region test sets. Samples can appear
in several different region test sets. These events, folds, and source seeds
are not independent samples. For inorganic cases,
held-out predictions are pooled once per row within each repeat; each
configuration's RMSE is then summarized across two repeats and three source
seeds. C and B are not artificially duplicated across source seeds. Between-run
standard deviations are descriptive variation, not confidence intervals.

The saved polymer aliases `component_0`, `component_1`, and `component_2` correspond
to **IDA, IBOA, and PEGDA**, respectively, in mass-fraction units. This mapping is
verified against the original composition table and saved application code.
`release_annotations.json` records the mapping and test-row coverage without
changing any frozen input arrays or their hashes.

## Reproduce

Use Python 3.12 and the pinned dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run_fusion51.py --workers 4
```

With the prepared `inputs/`, `input_hashes.json`, and `protocol.json` present,
the runner does not need the full historical archives. It checks hashes before
and after execution and reuses only existing complete job files. To rerun from
scratch, move the `results/jobs` directory aside first. This explicit step avoids
silently overwriting archived results.

To reconstruct the prepared inputs from the original archives, remove the
prepared inputs/protocol only in a separate working copy and pass the directory
containing the extracted v33 and v44 roots:

```bash
.venv/bin/python run_fusion51.py --source-root /path/to/extracted_archives --prepare-only
```

Focused verification against the original saved predictions requires those
historical archives:

```bash
.venv/bin/python verify_fusion51.py --source-root /path/to/extracted_archives
```

The working environment used for this stage was a new venv **with system packages
inherited**, followed by installation of Boruta and LightGBM. It is not claimed
as a clean reinstall of every dependency or as a rerun of source pretraining.

## Files and audit trail

- `protocol.json`: specification frozen before stage-51 target prediction errors.
- `environment.json`: actual Python and dependency versions.
- `source_hashes.json`: original files used to prepare the inputs.
- `input_hashes.json`: exact prepared-input identities.
- `job_manifest.json`: predefined configuration-repeat tasks.
- `results/summary.csv`: compact case/feature-set summary.
- `results/configuration_metrics.csv`: all source-seed/repeat metrics.
- `results/fold_metrics.csv`: every outer-fit metric and selected KRR setting.
- `results/selection_summary.csv`: counts for every inner and outer selector.
- `results/jobs/*.json`: exact row splits, all candidate losses, full per-coordinate
  statistical decisions, hit counts, and selected learned-coordinate names.
- `results/jobs/*.npz`: saved predictions, observations, row IDs, groups and folds.
- `results/validation.json`: training-only perturbation, numerical parity, split
  audits and historical prediction replay.
- `results/historical_baseline_recheck.csv` and `.json`: separate X + D versus
  X + D + M audit, with 12 additional outer fits and 526 prediction events.

## References

Kursa, M. B. and Rudnicki, W. R. (2010). Feature Selection with the Boruta Package.
*Journal of Statistical Software*, 36(11), 1–13.
https://doi.org/10.18637/jss.v036.i11

BorutaPy, official implementation and description of its deviations from the
original R implementation: https://github.com/scikit-learn-contrib/boruta_py

scikit-learn, Common pitfalls and recommended practices (training-only feature
selection and preprocessing):
https://scikit-learn.org/stable/common_pitfalls.html

## Publication and data scope

The experiment code is prepared for the author's GitHub release. Actual public
repository creation, repository URL assignment, code licensing and redistribution
of any third-party data must be handled in the main release package. A local
prepared-input bundle is not a statement that every upstream raw dataset can be
relicensed or redistributed. Source download/provenance instructions can be
published where raw files are not redistributed.
