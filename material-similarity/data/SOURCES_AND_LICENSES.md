# Sources, snapshots, and data reuse

Revised 14 September 2026. The input bytes are frozen artifacts supplied with the research. Current pages were checked for provenance and notices; the input data were not replaced by current database downloads.

## HTEM

- Original service URL recorded upstream: https://htem.nrel.gov/
- Catalog: https://data.nlr.gov/submissions/75
- Data DOI: https://doi.org/10.7799/1407128
- Database paper: https://doi.org/10.1038/sdata.2018.53
- License: https://data.nlr.gov/node/75/license
- Upstream retrieval: 2026-09-07.
- File: `upstream/htem_phase2_records.jsonl.gz`, 1,891 normalized libraries.
- Upstream changes: composition and deposition metadata normalized into JSONL; target-property tables not retained. The complete live-to-normalized acquisition pipeline is not included.

The catalog notice grants use/copy subject to retaining the entire notice and crediting DOE/NREL/ALLIANCE in resulting publications. The full notice is in `../LICENSES/htem_LICENSE_full.txt`; retain it with copies of the HTEM data. Other conditions in that notice remain applicable. No endorsement is implied.

Dataset authors in the catalog: Andriy Zakutayev, John Perkins, Marcus Schwarting, Robert White, Kristin Munch, William Tumas, Nick Wunder, and Caleb Phillips.

## MaterialsMine / NanoMine

- Service: https://materialsmine.org/
- Project: https://github.com/Duke-MatSci/materialsmine
- Source-code snapshot recorded upstream: `5a48d01d21b112aa36c62be22e470cc00620ef74`.
- Inspected license: https://raw.githubusercontent.com/Duke-MatSci/materialsmine/5a48d01d21b112aa36c62be22e470cc00620ef74/LICENSE
- Schema paper: https://doi.org/10.1063/1.5046839
- Knowledge graph paper: https://doi.org/10.1007/978-3-030-62466-8_10
- Author archive only: `upstream/nanomine_phase3_records.jsonl.gz`, 832 samples from 119 article groups. This file is absent from the public-code archive.
- Upstream changes: constituents, documented operations/settings, and citation context normalized into JSONL; sample URIs and source identifiers retained. The extraction and acquisition scripts are included in both archives under `../upstream_analyses/phase3_nanomine/`; normalized NanoMine records and fixed caches are present only in the author archive. Rebuilding citation enrichment from raw sources additionally requires the recorded PNCExtract repository snapshot.

**Code and data licenses differ.** The inspected MaterialsMine LICENSE describes CC BY-NC-SA 4.0 for its codebase and explicitly excludes API data/services unless otherwise stated. Its supplied text is retained in `../LICENSES/materialsmine_LICENSE.txt` as source-project evidence, not a license assigned to this JSONL snapshot. A separate redistribution license covering the supplied snapshot has not been established in this preparation. Its data license is therefore recorded as `NOASSERTION`; confirm the data redistribution basis before public redistribution.

The code commit is a code-version pointer, not a version identifier for the live knowledge graph. The JSONL hash identifies the exact normalized snapshot analyzed here. Fifty-three records across 20 article groups contain `unpublished/doi-...`, `unpublished-`, or `unpublished-initial-create` DOI placeholders; these must not be represented as registered DOIs.

Credit remains due to the MaterialsMine/NanoMine team, curators, and original studies identified by source URIs and citation metadata. No MaterialsMine application code is incorporated in the independent `material_similarity` implementation.

## Starrydata

- Project: https://starrydata.org/
- Project paper: https://doi.org/10.1080/27660400.2025.2506976
- Use rules: https://starrydata.org/utility/terms.html
- Dataset manifest recorded upstream: https://starrydata.github.io/starrydata_datasets/manifest.json
- Upstream snapshot: 2026-09-06.
- Earlier analysis: 5,001 records from 947 source groups.

The selected 5,001-record feature/assignment table is included at `cohorts/starrydata_phase1_cohort.jsonl.gz`; historical preprocessing, clustering, and summary code is under `../upstream_analyses/phase1_starrydata/`. The full original `all_samples.csv.gz` and property curves are absent. The expected source SHA-256 is recorded in the Phase 1 configuration. No current download is substituted for that historical snapshot.

The project use rules require citation of the Starrydata paper when publishing results or derived datasets, and original experimental papers where practicable; they also prohibit misleading copies that could be mistaken for the service. Preserve the SID/sample provenance and identify this as a derived research cohort. These are source-specific terms, not an MIT grant. The official dataset repository documents identifier issues in snapshots from this period; this cohort and its code retain the composite key `(SID, sample_id)`.

## PNCExtract

- Project: https://github.com/ghazalkhalighinejad/PNCExtract
- Source snapshot: `46a8175f1371048ae3d381bbc6e01e5922321794`.
- Paper: https://aclanthology.org/2024.findings-acl.779/
- Inspected license: https://raw.githubusercontent.com/ghazalkhalighinejad/PNCExtract/46a8175f1371048ae3d381bbc6e01e5922321794/LICENSE
- Earlier analysis: 1,103 records from 217 paper groups.

The normalized 1,103-record sample table is included at `cohorts/pncextract_breadth_records.jsonl.gz`, alongside the project-specific breadth analysis in `../upstream_analyses/phase3_nanomine/`. The source-repository Apache 2.0 notice is retained in `../LICENSES/pncextract_LICENSE.txt`. Normalization extracts constituent names, fractions and citation metadata from `sample_data` and article metadata; the original full corpus and article texts are not bundled. Credit Ghazal Khalighinejad, Defne Circi, L. Catherine Brinson, and Bhuwan Dhingra. Earlier source notes linked a different 2025 paper; the 2024 Findings of ACL reference above is the verified PNCExtract paper.

## Original project materials

The independent `material_similarity` code, project-specific scripts, and original documentation are covered by the root MIT license. Source data and rights in data-derived artifacts are not relicensed by that software license. Dataset-specific notices and attribution should accompany copies of data. `../provenance/ORIGINAL_SOURCES_AND_LICENSES.md` preserves the earlier notice for history; this current document explicitly distinguishes MaterialsMine software licensing from API data.

## Distribution in version 0.2.2

The author reproduction archive retains the supplied NanoMine snapshots for local analysis. The public-code archive omits those snapshots, raw caches, audited interchange rows and detailed raw-value audit extracts. The exact omitted files and their hashes are listed in `external_data_manifest.json`. The independent code, source selection IDs, summary results and read-only acquisition/restoration scripts remain. The public search route returned HTTP 503 on 2026-09-14; current acquisition cannot be claimed to reproduce the frozen bytes. A dataset-specific redistribution basis remains unconfirmed.

Registry lookup resolved 109 of 110 distinct DOI-formatted source strings to matching identifiers (772 records). One source identifier returned different canonical metadata and remains excluded from the strict resolved subset; placeholders remain separately identified. Registry resolution is not confirmation of a sample transcription. See `../results/revision/doi_resolution.json`.

The original input is retained for the author’s local analysis. No permission-backed distribution route to reviewers or the public has been verified. The local restoration script requires a copy already held with appropriate authorization; a checksum and a software commit do not supply access. The public package supports saved-result inspection and regeneration of figures, not an independent fresh NanoMine analysis. See `../docs/DATA_ACCESS_STATUS.md`.
