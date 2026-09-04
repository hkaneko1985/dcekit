# Evidence-gated extrapolation

Reference implementation and reproducibility package for:

> Evidence-gated anchor-delta models for extrapolative prediction,
> Bayesian optimization, and direct inverse analysis

The directory is designed to be copied directly into the root of
[`hkaneko1985/dcekit`](https://github.com/hkaneko1985/dcekit).

## Scope

The package contains three complementary methods selected by the engineering
question, not by an overall performance ranking.

1. **AD-ADE-GPR/SVR** predicts a response at an already specified external
   input. A five-nearest-neighbor (5NN) mean distance in the standardized input
   space defines the applicability domain (AD).
2. **EG-AD-ADE-GPR-BO** selects sequential or three-point-batch evaluations
   with a training-calibrated support gate, outer-shell discrepancy, and a
   risk-adjusted upper-confidence-bound acquisition function.
3. **EGC-GMR** performs forward conditioning and direct inverse analysis with
   competing absolute- and transition-coordinate Gaussian mixture regression
   models. It is useful when the latent input space is too large or multimodal
   for exhaustive forward prediction and a partial search may omit candidates.

The term *index of extrapolation* (IoE) is reserved here for the density-based,
GMR-specific reliability measure. GPR, SVR, and Bayesian optimization use the
5NN input-distance AD instead.

The numerical study uses radial-basis-function kernels. The limiting argument
also applies to local kernels whose similarity to every training sample decays
with distance, such as Matérn and rational-quadratic kernels, but those variants
were not evaluated numerically.

## Directory contents

- `run_numerical_validation.py`: core AD-ADE and GMR implementations plus
  synthetic diagnostics.
- `run_physics_validation.py`: forward and direct-inverse validation on D1-D7.
- `run_prediction_variation_validation.py`: fixed-AD-shell directional-collapse
  and GMR-support diagnostics.
- `run_bayesian_optimization_validation.py`: sequential and three-point-batch
  Bayesian-optimization validation.
- `run_strong_baseline_validation.py`: training-shell-tuned RBF-GPR, RBF-SVR,
  and ridge baselines.
- `reanalyze_reference_results.py`: dataset-clustered statistics, common
  no-clipping inverse evaluation, interval scores, and D5 audit.
- `run_all_validations.py`: orchestrates all engineering validations and the
  reviewer-driven reanalysis.
- `verify_reference_results.py`: fast integrity and headline-result checks.
- `test_revision_invariants.py`: regression tests for the corrected
  statistical unit, D7 bounds, common feasibility rule, and interval score.
- `physics_simulators.py`: engineering simulators and public-data loader.
- `data/raw/`: UCI Concrete Slump source data.
- `physics_validation_results/`: cached datasets and all reference CSV/PNG
  outputs used in the paper.
- `paper_figures/`: publication figures with English labels.

See `MANIFEST.md` for a more detailed inventory.

## Quick verification

Python 3.12 was used for the archived calculation. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-core.txt
python verify_reference_results.py
python verify_checksums.py
```

The last command reads only the archived CSV files and should finish quickly.
Successful verification prints `REVISED_REFERENCE_RESULTS_VERIFIED` and
`CHECKSUMS_VERIFIED`. The first command is a consistency check of saved
outputs, not a model refit; the second detects accidental file changes.

Run the correction-specific regression tests with:

```bash
python -m unittest -v test_revision_invariants.py
```

## Reproduce the analyses from cached engineering data

The cached D1-D7 datasets permit rerunning the learning steps, forward
validation, directional diagnostics, and Bayesian optimization without
regenerating the physical datasets:

```bash
python run_all_validations.py --output reproduction_results
```

The default settings use five forward repetitions, three direct-inverse
repetitions with eight targets per system, five Bayesian-optimization
repetitions, and a budget of nine external evaluations. Simulator verification
of newly generated inverse candidates, especially D4, still requires the
optional physical-simulator stack. Without it, the runner reports a warning and
retains the candidate-generation results; the exact archived verified metrics
remain available in `physics_validation_results/`. The run can take a
substantial amount of time. A smoke test is available:

```bash
python run_all_validations.py --output smoke_results --quick
```

## Regenerate the physical datasets

Install the optional simulator stack and invoke the orchestrator with
`--regenerate`:

```bash
python -m pip install -r requirements-physics.txt
python run_all_validations.py --output regenerated_results --regenerate
```

Pycalphad, PyBaMM, and IDAES/Pyomo may require platform-specific solvers and
their own setup steps. The archived CSV datasets are therefore the recommended
starting point for model-method verification.

## Revised headline results

- D5 is exploratory because 21/125 training simulations and 23/65 external
  simulations failed and were removed. D1-D4, D6, and D7 form the primary
  forward and Bayesian-optimization comparison.
- Using mean response-wise RMSE normalized by the external-test response SD,
  AD-ADE-GPR improved 5/6 fixed RBF-GPR controls and AD-ADE-SVR improved 6/6.
  Against training-shell-tuned RBF models, the counts were 5/6 and 4/6.
  Tuned ridge regression was stronger on several systems, so the results do
  not establish universal predictive dominance.
- On all 120 primary inverse targets (D1-D4 and D6), EGC-GMR returned a
  simulator-verified feasible answer for 87.5% and achieved top-five
  NRMSE <= 1 for 81.7%. Transition-GMR obtained 88.3% and 80.8%, respectively.
- Bayesian-optimization inference uses one median difference per dataset.
  Sequential EG-AD-ADE versus RBF produced 3 wins, 2 ties, and 1 loss across
  six independent systems (Wilcoxon p=0.375); batch comparison produced
  3 wins, 1 tie, and 2 losses (p=0.438).
- Proposed 90% intervals reached median conditional coverage 0.953 across the
  six primary systems, with median width 5.141 standardized objective units.
  Coverage and sharpness are reported together.
- D4 is an explicit failure case: a carbon-dioxide-purity mechanism change was
  absent from the training data and could not be predicted.

## Reproducibility notes

- Every learning representation uses component-wise standardization fitted to
  the corresponding training split. No PCA, PLS, nonlinear transform, or
  external-test-driven tuning is used.
- The base random seed is `20260823`; deterministic scenario and repetition
  offsets are recorded in the scripts and run manifest.
- The reference DCEKit repository state is commit
  `5f84d5b4bc87ed00a58875a2de3fdfdf98dbc1a2`.
- D7 inverse bounds are derived from training inputs only. Held-out test inputs
  do not define feasibility.
- The same feasibility rule is applied to all inverse methods: out-of-bounds
  candidates are discarded, never clipped to a boundary.
- Repetitions within an engineering system are descriptive replicates.
  Inferential tests use the dataset as the independent unit.
- Regenerated datasets write every nonfinite simulator input to a separate
  failure log before excluding it from regression modeling.
- Support gates are empirical warning/rejection rules, not prediction-error
  bounds or safety certificates.

## Data and license

Code in this directory is distributed under the DCEKit MIT License. The
Concrete Slump Test data are from the UCI Machine Learning Repository and are
distributed under CC BY 4.0:

I.-C. Yeh, *Concrete Slump Test* [Dataset], 2007,
<https://doi.org/10.24432/C5FG7D>.
