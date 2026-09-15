# Sources and attribution

## Starrydata

- Dataset manifest: https://starrydata.github.io/starrydata_datasets/manifest.json
- Frozen upstream source: 2026-09-06 snapshot
- Phase 4 includes only the compact Phase 1 summary and clustering assignments
  needed for synthesis, not the full raw samples or curves.

## HTEM

- Public REST API used upstream: https://htem-api.nlr.gov
- Frozen upstream retrieval: 2026-09-07
- Phase 4 includes only Phase 2 aggregate results, selected assignments, and
  candidate-pair diagnostics. Property and characterization targets are absent.

## MaterialsMine / NanoMine

- Website: https://materialsmine.org/
- Source repository: https://github.com/Duke-MatSci/materialsmine
- Source snapshot used upstream: 5a48d01d21b112aa36c62be22e470cc00620ef74
- Repository license at the inspected snapshot: CC BY-NC-SA 4.0

The compact Phase 3 artifacts included here are derived from the public
MaterialsMine service. Reuse must retain attribution and respect upstream terms.

## PNCExtract

- Repository: https://github.com/ghazalkhalighinejad/PNCExtract
- Source snapshot used upstream: 46a8175f1371048ae3d381bbc6e01e5922321794
- Repository license at the inspected snapshot: Apache License 2.0
- Paper: https://aclanthology.org/2025.naacl-long.185/

PNCExtract is used only for the composition-breadth check and auxiliary
author/location context. Neither author nor location is a similarity feature.

## Scope of the bundled inputs

All files in data/upstream are frozen intermediate outputs from the previous
phases. They contain no target-property table. Paper/study/source information,
where present, serves only as weak post-clustering support and never as a
distance feature.
