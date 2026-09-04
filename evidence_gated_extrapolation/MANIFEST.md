# Package manifest

## Executable Python files

| File | Purpose |
|---|---|
| `physics_simulators.py` | D1-D7 simulator definitions, physical bounds, and data loader |
| `run_numerical_validation.py` | AD-ADE, applicability-domain, Absolute-GMR, Transition-GMR, and EGC-GMR implementations |
| `run_physics_validation.py` | Forward and direct-inverse engineering validation |
| `run_prediction_variation_validation.py` | Fixed-shell directional-variation and support diagnostics |
| `run_bayesian_optimization_validation.py` | Sequential and three-point-batch optimization validation |
| `run_strong_baseline_validation.py` | Training-only tuning and evaluation of RBF-GPR, RBF-SVR, and ridge baselines |
| `reanalyze_reference_results.py` | Dataset-clustered inference, fair inverse evaluation, interval scores, and data-generation audit |
| `run_all_validations.py` | Reproduction orchestrator |
| `verify_reference_results.py` | Fast archived-result assertions |
| `verify_checksums.py` | SHA-256 integrity verification for packaged files |
| `test_revision_invariants.py` | Regression tests for the reviewer-driven corrections |

## Data and numerical outputs

- `data/raw/slump_test.data` and `slump_test.names`: public UCI source files.
- `physics_validation_results/datasets/D1_data.csv` through `D7_data.csv`:
  cached training and external-test data.
- `physics_validation_results/*_raw.csv`: repetition- or candidate-level data.
- `physics_validation_results/*_summary.csv`: manuscript-level aggregations.
- `physics_validation_results/forward_all_model_summary.csv` and
  `forward_dataset_level_comparisons.csv`: strong-baseline comparison.
- `physics_validation_results/inverse_metrics_fair_raw.csv` and
  `inverse_decision_utility_*.csv`: common no-clipping, all-target evaluation.
- `physics_validation_results/bo_pairwise_summary.csv`: dataset-level
  inferential analysis; repetition-level counts are explicitly descriptive.
- `physics_validation_results/bo_risk_performance_tradeoff.csv`: primary
  six-dataset performance/support summary; the all-dataset version is marked
  exploratory in a separate file.
- `physics_validation_results/dataset_generation_audit.csv`: requested and
  retained simulation counts, including D5 failures.
- `physics_validation_results/run_manifest.json`: seeds, versions, methods,
  support definitions, and limitations.
- `paper_figures/fig1_overview.png` through
  `paper_figures/fig6_bo_uncertainty_batch.png`: English publication figures.

## Environment files

- `requirements-core.txt`: dependencies needed for cached-data validation.
- `requirements-physics.txt`: optional simulator dependencies.
- `requirements.txt`: convenience alias for the core environment.
- `Dockerfile`: pinned core environment for archived-data verification.
- `PHYSICS_ENVIRONMENT.md`: optional solver requirements and numerical-
  convergence limitations for D2-D6.
- `DATA_LICENSE.md`: UCI attribution and third-party data/software licensing
  boundaries.
- `CITATION.cff`: citation metadata.
- `REVISION_NOTES.md`: reviewer-driven analytical and implementation changes.
- `LICENSE`: MIT license inherited from DCEKit.
- `.gitignore`: generated-runtime exclusions.
- `CHECKSUMS.sha256`: SHA-256 digest for every packaged file except the
  checksum list itself.
