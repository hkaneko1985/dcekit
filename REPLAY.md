# Steel replay: maintained environment and historical evidence

`scripts/replay_v49.py` retains the project's v49 replay numerical routines and
input identities, attributed to the Hiromasa Kaneko research project. The
stage55 dependency update changes its exact Pillow requirement from 12.3.0 to
13.0.0. The script is therefore no longer a byte-identical copy of the original
v49 runner. Its ancestry and the limited adaptation are recorded in
`manifests/implementation_provenance.json`. Its numerical block calls external
NumPy/SciPy/scikit-learn/LightGBM implementations; those libraries are not
vendored. The core package does not import this runner or its additional
dependencies.

The current runner requires Python 3.13, the exact versions in
`requirements-replay.txt`, exact input hashes, and an empty output directory.
The Pillow pin and the runner's preflight check must be updated together;
changing the requirements file alone would fail preflight. Its seven numerical
dependency pins, all input hashes, and the archived numerical block are
unchanged. A different environment that passes the core inference tests does
not automatically pass this replay.

Pillow is installed as a dependency of RDKit and is also used by the optional
Matplotlib plotting environment. Removing the direct requirement would not
remove those transitive dependencies. Both current replay and plot requirements
therefore pin Pillow 13.0.0. The stage51 results below document a completed run
with Pillow 12.3.0; that version is retained in historical evidence, not
recommended for a new installation. The security update does not establish a
new full-study reproduction.

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
direct commands below retain their original behavior: they create a
diagnostic output directory before performing dependency and asset preflight.

```bash
# Use a separate Python 3.13 virtual environment with the current requirements.
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


## Fresh installation of the current replay environment

`scripts/bootstrap_clean51.py` creates a new venv, records absence of the eight
required packages, installs the exact requirements, checks imports and package
consistency, and invokes the supplied runner with `--require-isolated`. The
filename records the utility's origin; current output is labelled stage55.
Use a Python 3.13 interpreter. Copy the current `scripts/replay_v49.py` as
`v49_replay.py` and the current `requirements-replay.txt` as
`v49_requirements.txt` alongside the four required private input assets in a
separate replay-input directory. Do not substitute the old Pillow 12.3.0
requirements from the historical archive. The wrapper records the supplied
files' hashes, reports whether the runner differs from the original v49 hash,
and verifies that it does not change the runner during execution.

```bash
python scripts/bootstrap_clean51.py \
  --python-executable /path/to/python3.13 \
  --input-dir /private/replay_inputs \
  --output-dir /private/new_clean_run \
  --mode representative
```

## Historical stage51 installation and representative replay

The successful stage51 installation used Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0,
scikit-learn 1.8.0, LightGBM 4.6.0, RDKit 2025.9.4, joblib 1.5.3,
threadpoolctl 3.6.0, and Pillow 12.3.0. The wrapper's
`fresh_environment_replay_completed=true` is supported by before-install,
installation, import-location, and numerical logs. The original runner used
in that recorded run does not own that attestation and its own field remains
false. `verification/fresh_environment_replay.json` is preserved unchanged;
its runner and requirements hashes refer to the historical files, not the
current maintained environment. They must not be relabelled as evidence of a
Pillow 13.0.0 run.

The representative run performed 18 outer fits and reproduced 1,872 saved
predictions with maximum absolute difference zero. It also checked all 126
saved aggregate metrics and regenerated 12 latent matrices exactly. This is
distinct from the NumPy-only public core check (maximum difference 2.09e-14),
the original same-environment full replay, and the Python 3.12 architecture
comparison. It is not complete source retraining or all-study reproduction.

## Stage55 dependency-update compatibility check

Pillow 13.0.0 was checked separately with the maintained runner in a copy of the
recorded isolated Python 3.13.5 environment. Its strict preflight passed, and
the representative 18-fit replay reproduced all 1,872 predictions exactly.
The 12 latent arrays, fixed 18 descriptors and 126 aggregate metrics also
matched exactly. This was a Pillow-only update in a copied environment, not
a new fresh installation or a complete historical replay. See
[SECURITY_UPDATE.md](SECURITY_UPDATE.md) and
[verification/security_update55.json](verification/security_update55.json).
