# Verification of version 0.2.2

A complete `python run_all.py` run in the pinned Python 3.12 environment passed all 39 tests, regenerated 13 common-interface outputs and 18 additional outputs, and generated Figures 2–7 and S1–S3. Numerical comparisons matched at relative tolerance 1e-7 and absolute tolerance 1e-8. The additional outputs comprise 15 unchanged R2 reference files and three unmasked-baseline files. No original 25/50/75% mask configuration, random stream, view weight, or reference result was changed.

The exact verification files included in both archives are:
- `docs/full_verification.txt`: complete stdout/stderr of the full R3 numerical run.
- `docs/common_reproduction_check.json`: all 13 common-output checks.
- `docs/revision_reproduction_check.json`: all 18 additional-output checks.
- `docs/baseline_reference_checks.json`: all 180 original conditions have reference-close and evaluable counts three times the corresponding one-pass baseline.
- `docs/RELEASE_AUDIT.json`: environment, scope, counts and package checks.
- `docs/public_tables_check.txt`: isolated public saved-table execution.

The `.txt` log names avoid the repository's `*.log` ignore pattern. Our retained R2 archive did contain `docs/full_verification.log`; that older evidence is now explicitly historical in `provenance/v021/docs/`. Current commands use the files above.

The unmasked baseline supplies 60 conditions and 7,200 query/rule rows. Both baseline and masked rules concern documented pairwise reference distances, not chemical truth, cluster-membership guarantees, or superiority of a similarity. The single record without reported settings contributes nine empty masks to the existing experiment.

Fresh NanoMine analysis requires the author-held 832-record input, including full-source scale information. The public archive excludes nine input/raw-value paths and independently supports saved-table checks and figure regeneration only. No independent, permission-backed acquisition or publication route has been established. Hash matching validates existing bytes; it does not obtain permission or initial access. Details are in `DATA_ACCESS_STATUS.md` and `data/input_dependency_scope.json`.

The final ZIPs are checked for CRC errors and against every MANIFEST.sha256 entry. These checks establish package and numerical consistency within the retained inputs. They do not validate live acquisition, restore missing original Starrydata/HTEM acquisition archives, or recheck every source article.

The isolated public stage passed 35 available tests, with the source-regression class explicitly skipped, and redrew all saved figures. The complete-analysis command refused to proceed without NanoMine input before doing any analysis. No new acquisition is implied.
