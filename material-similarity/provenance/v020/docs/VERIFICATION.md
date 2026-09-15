# Verification of version 0.2.0

Run date: 2026-09-14. The pinned Python 3.12 environment completed 35 checks (28 schema, distance, normalization, source-regression or analysis checks, plus seven artifact consistency checks). All 18 affected source records were exercised as regression subtests.

`python run_all.py` regenerated all 13 common-interface CSV/JSON files and nine additional evaluation files in separate output directories. Both reference comparisons passed with relative tolerance 1e-7 and absolute tolerance 1e-8. The JSON reports are retained in this directory. This establishes reproducibility of the included numerical analysis, not physical validity or algorithmic superiority.

The historical Phase 4 aggregation was rerun after excluding Starrydata F3 from the cross-view aggregate. Historical source-specific calculations remain distinct from the current API. The complete raw-source Starrydata and HTEM acquisition pipelines were not rerun. New source registry lookups are optional network audits; reproducibility uses the stored DOI-resolution snapshot rather than repeating live lookups.

Public-code mode intentionally omits NanoMine source data. It runs available tests and redraws saved figures with --tables-only; it cannot claim a fresh NanoMine analysis until an authorized hash-matching input is restored. Public archive exclusion and restoration checks are documented in the release audit.

Release checks: the isolated public package completed its saved-table mode (30 executable checks; one source-regression class skipped), and correctly refused a fresh analysis without the NanoMine input. Restoration rejected an incorrect pre-existing snapshot, then restored six source files from the supplied author-held archive with matching hashes. The restored copy passed all 35 tests. Eight restricted/unconfirmed source or raw-audit files are absent from the public stage and retained in the author stage. ZIP integrity and every release-manifest hash are checked during packaging. See `RELEASE_AUDIT.json`.
