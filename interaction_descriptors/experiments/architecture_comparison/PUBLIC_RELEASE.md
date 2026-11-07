# Reproducing the architecture comparison

This directory contains executable code and aggregate reference results from the
stage51 developmental comparison. The source archives and prepared row-level
inputs are external. The `reference` files document the completed run and do not
act as mutable run-state files.

1. Obtain and extract the original v33 and v44 project archives described in
   `README.md` into an external source directory.
2. Install `requirements.txt` in your analysis environment.
3. Run `python run_fusion51.py --source-root /path/to/extracted_archives --workers 4`.
4. Run `python verify_fusion51.py --source-root /path/to/extracted_archives`;
   consult its `--help` for the archived path arguments required by your layout.

The run creates `inputs`, `results`, and a new `protocol.json` in this directory.
The checked-in reference protocol remains separate. Source and input hashes
must match before comparing numeric outputs. Do not upload those generated
row-level directories without separately resolving their distribution status.

The common inorganic baseline is X+D+M, where M holds 24 source applicability
indicators. The separate historical X+D audit is retained. The reference results
are KRR-only additions, not a replacement for the original six-regressor study.
The original untrained single-architecture control is retained; no new
dimension-matched 576-coordinate triple-random control was run.
