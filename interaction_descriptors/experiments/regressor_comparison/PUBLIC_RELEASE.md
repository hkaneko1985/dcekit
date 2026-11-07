# Public checkout and private prepared assets

This is an unprepared public checkout. `frozen_reference/` contains the completed
stage52 aggregate evidence and metadata; its presence is not a prepared-input or
prediction-job cache. No row-level target arrays, source weights, selector cache,
or per-row prediction NPZ files are included here.

Prepare and run in a new output directory through the top-level orchestrator:

```bash
python scripts/reproduce.py comparison --stage51-root /private/stage51 --output-dir /work/regressor_replay --workers 5
```

Or use this directory directly (it will create inputs/results beside the runner):

```bash
python run_regressors52.py --stage51-root /private/stage51 --workers 5
```

The stage51 working archive must expose `fusion/` and `sources/`. Preparation
verifies its exact inputs and 198 nested-selector identities. All 500 target jobs
are new fitting operations if no validated result cache exists. The reference
KRR 100 jobs are inherited; no new Boruta fit is implied. Original v33/v44 archives
are needed only for the optional historical-prediction verification. See README.md
for the scientific protocol and the original environment.
