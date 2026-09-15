# NanoMine input access status for R3

The author reproduction archive retains the supplied input for local computation. A new reviewer or reader still has no verified independent route to the exact frozen input. A saved result table, checksum or local restoration command does not resolve initial acquisition. This remains an unresolved issue for the principal NanoMine results.

## Actual required input

Current numerical analyses depend on `data/upstream/nanomine_phase3_records.jsonl.gz`: 832 records in 119 article groups, 138,028 bytes, SHA-256 `209168324fcbbcf9d308a2d6358d0caacf7a885b19b2c8e3de951bbff704833a`. Its recorded normalization timestamp is 2026-09-10T00:59:08.885129+00:00, not an immutable upstream database version. All IDs and semantic record hashes are already in `data/nanomine_selection_manifest.json`.

The 183-record API example, 72-record constituent display and 120-record method display overlap. Numeric scale fitting uses the full 832-record input, excluding the query article group for retrieval and relation-retention tasks. The missingness tests use full-source scales, with separate labelled scale sensitivities. The displayed subset alone is insufficient for identical reruns. `data/input_dependency_scope.json` maps scripts, comparison cohorts and scale references.

## Checks on 15 September 2026

Both the public search API and SPARQL endpoint returned HTTP 503 in the checks recorded in `data_access_checks_r3.json`. This is a point-in-time result, not a claim of permanent unavailability. The search endpoint's original requests and four retained cache names remain in `upstream_analyses/phase3_nanomine/src/acquire_phase3_cache.py`.

The LICENSE at MaterialsMine code commit `5a48d01d21b112aa36c62be22e470cc00620ef74` expressly excludes API data/services unless another statement applies. No separate terms covering reviewer/public redistribution of this exact snapshot, or matching independently downloadable data release, were verified. The software commit is not a live knowledge-graph version. Public project descriptions, an ontology license, and an article license do not by themselves identify terms for these input bytes.

## Journal policy and remaining action

JCIM's current Author Guidelines point to the 2021 data/software policy, including provision of data or an explicit extraction route from public sources. The 2026 joint JCTC/JCIM editorial states that inaccessible components should not be indispensable to reproducing principal claims. Level 2 disclosure alone therefore does not resolve the present access gap. These policies do not establish a redistribution license for third-party inputs.

Sources checked 2026-09-15:
- https://researcher-resources.acs.org/publish/author_guidelines?coden=jcisd8 (updated 2026-08-27)
- https://doi.org/10.1021/acs.jcim.0c01389
- https://doi.org/10.1021/acs.jctc.6c00733 (joint editorial, effective 2026-05-01; author-hosted text also checked at https://files.batistalab.com/publications/advancing-reproducibility-and-open-data-in-theoretical-and-computational-chemistry.pdf)

A working route must cover the full scale-source input and identify applicable terms, version and selection. It could be a permitted deposit or a verifiable official acquisition. If access is confidential during review, the post-publication route and editorial suitability also need clarification. If no route can be established, principal claims need accessible supporting inputs; a saved-table rerender is insufficient. No exception or editorial approval is asserted.

`DATA_PERMISSION_REQUEST_DRAFT.md` specifies the file and scope in an unsent inquiry. No provider or editor has been contacted, no permission obtained, and no repository published by this revision. Possession of the author archive is not represented as permission for reviewer distribution or public deposition.

## Package scope

`material-similarity_github_v0.2.2.zip` contains code, configurations, permitted retained inputs, IDs, hashes, result tables and figures; nine NanoMine input/raw-value paths are excluded by `data/external_data_manifest.json`. It independently runs saved-table checks and plotting only. `material-similarity_author_reproduction_v0.2.2.zip` retains the previously supplied inputs for author-local reruns. The hash-checked restore utility imports an already-held authorized archive, not a newly acquired input.
