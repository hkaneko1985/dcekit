# Sources and licenses

## Starrydata

- Project and citation: https://starrydata.nims.go.jp/links/cite/
- Dataset manifest used upstream: https://starrydata.github.io/starrydata_datasets/manifest.json
- Frozen upstream snapshot: 2026-09-06
- Phase 5 contains aggregate Phase 4 results only, not Starrydata raw samples or property curves.

## HTEM

- Database: https://htem.nrel.gov/
- Data catalog: https://data.nlr.gov/submissions/75
- Frozen upstream retrieval: 2026-09-07
- Phase 5 contains normalized composition and deposition-setting records. Property and characterization targets are absent.

## MaterialsMine and NanoMine

- Website: https://materialsmine.org/
- Source repository: https://github.com/Duke-MatSci/materialsmine
- Source snapshot used upstream: `5a48d01d21b112aa36c62be22e470cc00620ef74`
- Inspected repository license: CC BY-NC-SA 4.0

The bundled normalized records are derived from the public MaterialsMine service. Reuse must retain attribution and comply with upstream terms.

## PNCExtract

- Repository: https://github.com/ghazalkhalighinejad/PNCExtract
- Frozen source commit: `46a8175f1371048ae3d381bbc6e01e5922321794`
- Repository license at the inspected snapshot: Apache License 2.0
- Related paper: https://aclanthology.org/2024.findings-acl.779/

Phase 5 contains aggregate PNCExtract results from Phase 4, not the original sample table.

## Scope

All inputs are frozen artifacts from earlier phases. No target-property table is bundled or used by the Phase 5 API validation.
