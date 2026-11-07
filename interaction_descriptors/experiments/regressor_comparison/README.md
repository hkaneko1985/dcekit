# Stage52: regressor dependence of frozen-descriptor concatenation

This analysis extends the completed stage51 KRR comparison to PLS, LASSO,
SVR, LightGBM, and conventional GBDT. It uses the same three development
datasets, full-component inputs, source encoders, outer/inner row splits,
three source initializations, and eight feature conditions. It does not
retrain source encoders or add independent external materials.

## Comparison protocol

The eight conditions are composition alone (C), the common basis B, B plus
each of NN/YAE/YVAE, B plus the original single-family untrained control,
B plus concatenated NN/YAE/YVAE coordinates, and B plus concatenation with
the fixed nested Boruta selection. The last condition retains all original
basis columns and only statistically confirmed learned coordinates.

| Target | Rows | B | Descriptor route | Outer evaluation |
|---|---:|---|---|---|
| Acrylate Young's modulus | 49 | 3 mass fractions + 9 fixed terms | Boundary-consistent pair latent | Six region holdouts, 47 distinct test rows and 101 events per initialization |
| Metal atomic volume | 143 | 60 atomic fractions + 18 fixed terms + 24 source masks | Whole-composition latent | Three element-system holdouts × two repeats; 64 training labels |
| Oxide formation energy | 120 | 39 atomic fractions + 18 fixed terms + 24 source masks | Whole-composition latent | Three element-system holdouts × two repeats; 64 training labels |

The polymer metric is the unweighted mean of the six regional RMSEs, followed
by a mean over the three source initializations. The inorganic metric pools
test predictions within each repeat, followed by the mean over repeats and
initializations. C and B are not duplicated over source seeds. Individual
latent axes and predictions are not averaged into an ensemble.

The inorganic B includes source applicability masks and therefore differs
from the historical manuscript's X+D comparator. Stage51 separately audited
that distinction; those earlier baselines are not silently redefined here.

## Fixed regression implementations

`numerics/archived_regression.py` is byte-identical to the v44 numerical code
and to the numerical block embedded in the v49 replay runner. The v33 code's
function/assignment abstract syntax trees are identical; only imports/header
differ. No regressor, grid, random state, or numerical objective was selected
using the stage52 prediction errors.

| Regressor | Original grid |
|---|---|
| PLS | Components 1, 2, 4, 8, clipped to training rank/size |
| LASSO | Alpha 0.0001, 0.001, 0.01, 0.1, 1 |
| SVR | Length scale 0.5, 1, 2, 4 × C 0.1, 1, 10, 100 × epsilon 0.05, 0.1 |
| LightGBM | Leaves 3, 7, 15 × minimum child samples 3, 7; 100 trees |
| GBDT | Depth 1, 2, 3 × minimum leaf samples 2, 5; 100 trees |

The unchanged LASSO implementation first uses LARS and audits the primal-dual
gap. Its existing dual quadratic-program fallback solves the same LASSO
objective when needed; a gap above 1e-6 is a failure. Warnings, fallback use,
and gap values are retained in the job records. SVR retains the original
dimension-normalized RBF kernel. Training-only numerical constant filtering
and scaling are fitted independently for every inner and outer fit.

## Reusing the nested selectors

The 198 original Boruta fits depend on the target-training rows and candidate
features, not on the final regression family. Reuse is therefore deliberate:
the stage52 preparation checks complete input hashes, ordered feature names,
all train/validation/test indices, training matrices and targets, constant
columns, protected basis, and exact confirmed selections. Each worker checks
the relevant cached identity again before using it.

There are **zero new Boruta fits** in this stage. The 198 distinct selectors
are applied to each of the five new regressors (990 selection applications).
They are not described as 990 independent selector fits or an independent
replication of variable selection. The stage51 held-out-perturbation check is
retained as reference evidence.

`protocol.json` and `job_manifest.json` were frozen before stage52 prediction
errors were computed. `provenance.json`, `input_hashes.json`, and
`selector_reuse_audit.json` preserve the original assets and reuse evidence.

## Run

Use Python 3.12 and the pinned versions in a suitable environment:

```bash
python3.12 -m venv /path/to/regressor_env
/path/to/regressor_env/bin/python -m pip install -r requirements.txt
/path/to/regressor_env/bin/python run_regressors52.py --workers 5
```

The prepared working package is self-contained for these additional target
fits. It includes prepared input arrays, selector records, KRR reference
summaries, and the archived numerical module. The recorded stage52 execution
used the existing stage51 environment with inherited system dependencies;
it is not claimed to be a fresh installation.

The runner resumes only complete jobs whose protocol/input identities and
prediction-file hash match. To force all fits to run anew, use a separate
working copy and move `results/jobs/` aside. An unchanged cached job is not
counted as a new fit. A failed job is recorded under `results/failures/`;
results are never replaced with a favorable substitute.

To prepare a new copy from the original project assets, provide the stage51
root containing `fusion/` and `sources/`, as supplied in the private stage51
working-evidence ZIP. The bundled `numerics/archived_regression.py` is checked
against the exact numerical block in `sources/v49_replay.py`:

```bash
python run_regressors52.py --stage51-root /path/to/stage51 --prepare-only
```

If the v44 archive is available, its numerical source is used and the v33
numerical-body comparison is also repeated when that archive exists. Without
v33 the additional AST comparison is recorded as not checked for the new
preparation. If a numerical source file is elsewhere, supply
`--regression-source` with that exact file; it must still match the v49
embedded numerical block. The v33 archive is needed only for the optional
historical-polymer prediction comparison below.

After completion:

```bash
python analyze_regressors52.py
python verify_regressors52.py --archive-root /path/to/stage51/archives
# Without the historical v33 archive, integrity and nested-loop checks remain available:
python verify_regressors52.py --skip-historical
```

## Outputs and interpretation

- `results/status.json`: exact new-fit counts and reused KRR counts.
- `results/summary.csv`: all 144 target × regressor × representation means.
- `results/rmse_matrix_*.csv`: the full 8 × 6 tables for each target.
- `results/configuration_metrics.csv`: all 600 configuration-repeat records,
  with provenance identifying the 500 newly fitted and 100 reused KRR jobs.
- `results/jobs/*`: every new prediction, row split, candidate loss, selected
  hyperparameter, numerical diagnostic, and selector reference.
- `results/comparisons.csv`: descriptive changes against B, best single-family
  descriptors, and concatenation without selection.
- `results/added_vs_best_original.csv`: comparison against the strongest
  original feature condition within each regressor.
- `results/analysis_summary.json`: task-specific best conditions and counts.
- `results/validation.json`: full-result integrity, parity with the original
  nested-loop routine, and available historical polymer predictions.

The scheduled calculation is 500 new configuration-repeat jobs, 1,800 new
outer fits, 4,800 inner partitions, and 62,700 prediction events. KRR's 100
jobs, 360 outer fits, and 12,540 events are explicitly reused from stage51.
The numerical parity check performs five additional verification outer
refits, separate from the study-job counts.

All differences are descriptive development-cohort comparisons. Regressors,
seeds, and folds are correlated and do not constitute independent samples for
a significance test. A best row found in these tables is not a prospectively
validated selected model. The pre-existing single-family random control is
retained, but no dimension-matched three-family random concatenation is added.
The original six-regressor paper analyses and the negative steel evaluation
remain separate and are not replaced by this extension.

## Completed run and verification

All 500 study jobs completed without a failed or substituted job in 201.26
seconds at five single-threaded workers. They contain 1,800 outer fits,
4,800 inner partitions, 50,880 inner candidate fits, and 62,700 saved prediction
events. The separate verification used five original-loop outer refits and
215 predictions; it did not add study conditions.

Every saved prediction, test-row identity, target label, feature order, and
nested-selection application passed the result audit. The five original
`nested()` control-flow checks had a maximum prediction difference of
3.091e-13 and maximum candidate-MSE difference of 1.335e-11; chosen grid
settings matched. The 70 available historical polymer baseline/single-family
jobs (7,070 prediction events) matched with maximum absolute difference
5.805e-9, below the declared 1e-7 tolerance.

The 5,160 LASSO candidate/outer fits include 1,791 uses of the original
same-objective dual-QP fallback. There were 1,622 fits with 6,475 recorded
warning messages, all in LASSO. All fits satisfied the pre-existing 1e-6
primal-dual gap gate; the maximum gap was 7.768e-8. Among 588 fallback calls
whose SLSQP success flag was false, the largest accepted final primal-dual gap
was 3.209e-12. These optimizer flags and warnings are retained, not hidden or
interpreted as changed regression objectives.

The strongest added metal condition was SVR plus concatenation, with mean
RMSE 2.85800767 Å³/atom versus 2.90888403 for SVR plus the best single-family
YVAE block (1.75% lower). Across all six regressors, polymer composition-only
GBDT remained strongest at 315.463795 MPa; the best added condition was GBDT
plus concatenation at 328.701613 MPa. For oxides, the single YAE block plus SVR
gave 0.41879158 eV/atom, compared with 0.42662762 for the best added condition,
SVR plus concatenation and Boruta. These are minima over the fixed comparison
table, not independent prospective validation of a selected model.

Concatenation improved over B in 12 of the 18 task/regressor cells, but improved
over the best single-family block in only three. Boruta improved concatenation
in 11 cells. The better of the two added conditions beat the strongest original
representation within that regressor in six cells. These counts describe
heterogeneity; the 18 cells are not independent statistical trials.
