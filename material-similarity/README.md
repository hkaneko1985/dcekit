# Material similarity

Version 0.2.2 accompanies **Material Similarity from Incomplete Composition and Synthesis Metadata** by Hiromasa Kaneko.

This target-free comparison interface separates composition, constituent identity, synthesis method, operation types, sequence order/repetition, and settings. Fixed weights express different comparison questions. No response properties, expert similarity labels, optimized weights, or imputed settings are used. The revised primary NanoMine features exclude density, size, and other characterization descriptors.

## Run the analyses

Use Python 3.12 and install the pinned environment:

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python run_all.py
```

Run from `material-similarity`. The full command requires the frozen NanoMine input. It runs 39 software/artifact checks, regenerates 13 common-interface result files and 18 additional evaluation files (15 unchanged R2 references and three unmasked-baseline files), compares them with the reference tables, and regenerates main Figures 2–7 and SI Figures S1–S3. Current outputs go to `outputs/`. Current reference results are in `results/`; earlier results are explicitly historical in `provenance/results_v010/` and `provenance/v020/`.

The **public-code archive excludes NanoMine API data**, whose public redistribution basis was not confirmed. To inspect supplied results and redraw figures without those inputs:

```bash
python run_all.py --tables-only
```

This does not rerun the NanoMine analyses. Source regression tests are explicitly skipped if the source input is absent. To restore authorized, hash-matching files from an author-held ZIP:

```bash
python src/restore_external_data.py --archive /path/to/author_archive.zip
python run_all.py
```

The external file manifest and exact cohort selection are in `data/`. The upstream public acquisition script is `upstream_analyses/phase3_nanomine/src/acquire_phase3_cache.py`. The search and SPARQL routes returned HTTP 503 on 2026-09-15; a changed live database is not silently substituted for the frozen input. The author reproduction archive retains the supplied snapshots; it is not the public redistribution package.

## What was evaluated

| Resource | Historical records | Current common-API role |
|---|---:|---|
| NanoMine | 832 records / 119 article groups | Overlapping 183-, 72-, and 120-record examples; retrieval/masking tasks |
| HTEM | 1,891 libraries | 183-library composition/settings example |
| Starrydata | 5,001 selected records | Historical F0–F2 assignment reaggregation |
| PNCExtract | 1,103 samples | Historical constituent/loading analysis |

The sum 8,827 is not a deduplicated or common-API corpus. Original Starrydata all-sample data and the HTEM acquisition cache are absent. Historical Phase 1–4 scripts and available tables remain in `upstream_analyses/`; their algorithms and normalizers are not the revised API.

## Revision findings

- All 18 NanoMine records with duplicate setting names retain every value; unlabelled numeric multivalues have a minimum/maximum correspondence range.
- Explicit quantity/unit validation quarantines 288 numeric entries (279 process, nine nonprimary component entries). No units or conditions are inferred.
- Same-information comparisons have similar masked-record recovery; the task is near a ceiling. Held-out method agreement supplies modest cross-group support without establishing that balanced weights outperform simple averaging.
- Fixed-reference settings bounds retain the original interval under all tested masks, at the cost of greater width. Pair-active bounds can contract.
- Operation correspondence, scale reference, and descriptor inclusion remain sensitivity choices.

See `docs/METHODS.md`, `docs/RESULTS_SUMMARY.md`, `docs/FIGURE_REPRODUCTION.csv`, and `docs/REVISION_NOTES.md`. Tables and source provenance, rather than a single optimal score, support interpretation. Transfer learning is an untested prospective application.

## Source notices

Original project code is MIT licensed. Third-party data retain source-specific terms. `data/SOURCES_AND_LICENSES.md`, `NOTICE.md`, and `LICENSES/` distinguish MaterialsMine software from API data. A public GitHub URL/commit has not been invented; update `CITATION.cff` after an actual release exists. Public hosting and contacting data owners are not performed by this archive.

## Re-review changes (0.2.1)

The bilateral mask now uses the intersection of jointly reported operation-type/within-type-occurrence/setting keys, shared with the distance engine. `mask_key_audit.csv` records the actual keys for every pair and scenario. `config/interval_utility.json` fixes three existing views, a threshold grid, query-only setting masks, and different-article-group galleries. `src/run_interval_utility.py` compares point-only, coverage-assisted and interval-assisted interpretation against documented pre-mask distances. This is a conservative-interpretation/retention trade-off, not chemical truth or a search for optimal weights.

The data-access issue remains open: the author-held input is retained only in the author reproduction archive for local reproduction, but a permission-backed reviewer/public route has not been established. The archive does not claim that `restore_external_data.py` obtains data independently. See `docs/DATA_ACCESS_STATUS.md`.

## R3 clarification and unmasked baseline (0.2.2)

`src/run_baseline_documentation.py` evaluates the unchanged Figure 7 rules before additional masking: 120 queries, 93 article groups, one pass, 14,222 directed comparisons and 60 conditions. Table S15 supplies all baseline acceptance counts. The original 25/50/75% experiments, weights, random streams and 180-condition reference results remain unchanged. Three new `baseline_documentation_*` result files separate baseline documentation gaps from added withholding. The constituent view assigns zero weight to settings, so its point distances are unchanged. These are pairwise decisions; cluster membership and stability are not guaranteed. A nonsignificant paired comparison does not establish equivalence.

Complete execution evidence is explicitly named in `docs/full_verification.txt`, `docs/common_reproduction_check.json`, `docs/revision_reproduction_check.json`, and `docs/RELEASE_AUDIT.json`. These paths are included in the release manifest. Public saved-table execution is logged in `docs/public_tables_check.txt`. The `.txt` suffix avoids the repository's `*.log` ignore rule.

The full 832-record input is required for numeric scale fitting, beyond displayed subsets. `data/input_dependency_scope.json` records the dependency and points to all source IDs and hashes. The initial data-access issue remains unresolved. `docs/DATA_PERMISSION_REQUEST_DRAFT.md` is an unsent, concrete provider inquiry addressing reviewer access, publication access, and the complete scale-source input; it is not permission or an acquisition route.
