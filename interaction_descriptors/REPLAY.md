# Historical steel replay

`scripts/replay_v49.py` is an unchanged copy of the project's v49 replay runner,
attributed to the Hiromasa Kaneko research project. Its hash and source are listed
in `manifests/implementation_provenance.json`. Its numerical block calls external
NumPy/SciPy/scikit-learn/LightGBM implementations; those libraries are not vendored.
The core package does not import this runner or its additional dependencies.

This archival runner has its **own** original contract: Python 3.13, exact
versions in `requirements-replay.txt`, exact input hashes, and an empty output
directory. Its checks are deliberately unchanged. A different environment that
passes the core inference tests does not automatically pass this archival replay.

Run these commands from the `interaction_descriptors/` project directory inside
DCEKit. The example `/private/...` paths are user-supplied external locations.
For missing-input checks before a replay output directory is created, prefer
the supported wrapper:

```bash
python scripts/reproduce.py --python /path/to/replay_env/python steel-audit \
  --input-dir /private/replay_inputs --output-dir /private/replay_audit --mode audit
```

Replace the interpreter path with the Python 3.13 executable in your replay
environment (`bin/python` on Unix or `Scripts/python.exe` on Windows). The
direct archived commands below retain their original behavior: they create a
diagnostic output directory before performing dependency and asset preflight.

```bash
# Use a separate Python 3.13 virtual environment for this historical runner.
python -m pip install -r requirements-replay.txt
python scripts/replay_v49.py --input-dir /private/replay_inputs \
  --mode audit --output-dir /private/replay_audit
python scripts/replay_v49.py --input-dir /private/replay_inputs \
  --mode representative --output-dir /private/replay_representative
# Full historical repeat only, when required:
python scripts/replay_v49.py --input-dir /private/replay_inputs \
  --mode full --output-dir /private/replay_full
```

`preflight` reports dependencies and asset identities. `audit` regenerates the
features, verifies metric aggregation and checks stored splits. `representative`
refits the six regressors for the designated NN/seed/repetition condition. `full`
replays the existing 180 configuration–repeat jobs, 540 outer fits and 56,160
prediction events on the same 312 steels. It does not add materials or retrain
source encoders. The isolated-environment flag is a check, not a dependency
installer, and isolation alone does not prove a fresh dependency installation.

The current package's included `verification/steel_whole_latent_validation.json`
is narrower: it compares 12 regenerated latent matrices and applicability masks.
It is not evidence that this entire historical runner has been executed in a
freshly installed environment. Separate fresh-environment/downstream-replay
evidence should be added under its own scope and environment report.


## Stage51 installation and representative replay

`scripts/bootstrap_clean51.py` creates a new venv, records absence of the eight
scientific packages, installs the exact requirements, checks imports and package
consistency, and invokes the unmodified v49 runner with `--require-isolated`.
Use a Python 3.13 interpreter. Put the original runner under the name
`v49_replay.py` and `v49_requirements.txt` alongside the four required private
input assets in a separate replay-input directory. The supplied
`scripts/replay_v49.py` and `requirements-replay.txt` are byte-identical copies
of those two files and may be copied there under their original names.

```bash
python scripts/bootstrap_clean51.py \
  --python-executable /path/to/python3.13 \
  --input-dir /private/replay_inputs \
  --output-dir /private/new_clean_run \
  --mode representative
```

The successful stage51 installation used Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0,
scikit-learn 1.8.0, LightGBM 4.6.0, RDKit 2025.9.4, joblib 1.5.3,
threadpoolctl 3.6.0, and Pillow 12.3.0. The wrapper's
`fresh_environment_replay_completed=true` is supported by before-install,
installation, import-location, and numerical logs. The embedded unmodified
runner does not own that attestation and its own field remains false.

The representative run performed 18 outer fits and reproduced 1,872 saved
predictions with maximum absolute difference zero. It also checked all 126
saved aggregate metrics and regenerated 12 latent matrices exactly. This is
distinct from the NumPy-only public core check (maximum difference 2.09e-14),
the original same-environment full replay, and the Python 3.12 architecture
comparison. It is not complete source retraining or all-study reproduction.
