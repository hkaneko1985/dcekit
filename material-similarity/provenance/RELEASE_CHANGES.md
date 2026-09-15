# Changes from the supplied Phase 5 package

Prepared 13 September 2026 for a GitHub directory named `material-similarity`.

- The five Python source files in `material_similarity/` are byte-for-byte unchanged.
- The eight frozen upstream inputs and thirteen reference result files are byte-for-byte unchanged.
- The runner resolves its project path from its own file, removing dependence on the original directory name.
- Reruns write into `outputs/reproduced/` by default; an explicit `--output-dir` is supported. Progress messages were added; calculations and selection rules were not changed.
- The minimal example can be run directly. English plotting code for Figures 2–7 accepts separate result/output directories.
- One legacy artifact test checks the public result summary instead of an unpublished manuscript. The original 14 unit checks and the other artifact checks remain; the public suite contains 21 checks.
- A result comparator and `run_all.py` provide an execution path.
- README, data dictionary, source notices, citation metadata, environment records, and package metadata were added.
- The original source notice and manifest are historical records; their original paths refer to the earlier package. The current root `MANIFEST.sha256` identifies this distribution.
- The MaterialsMine software license's exclusion of API data is stated in the current notice. No data-license grant is inferred from it.
- Manuscript Word files, ACS templates, manuscript build scripts, and illustrative artwork are separate manuscript assets and are not included.

The common-interface analyses were rerun. All 13 CSV/JSON outputs matched the frozen references at relative tolerance 1e-7 and absolute tolerance 1e-8. Earlier phase-specific artifacts were subsequently recovered: the Starrydata 5,001-record selected feature/assignment table and historical code, revised process audit, full Phase 2/3/4 retained packages, and PNCExtract normalized cohort. The original full Starrydata source CSV and HTEM acquisition cache are still absent.

The only computational change in recovered phase scripts is an optional PNCExtract `--records` loader and output-directory creation; the scientific calculations are unchanged. Source hashes before packaging are recorded in `recovered_files.json`. The top-level README, data inventory, and notices describe the expanded scope. Earlier reports are historical records, not newly written scientific claims.
