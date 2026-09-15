# Data sources and attribution

## MaterialsMine / NanoMine

- Website: https://materialsmine.org/
- Source repository: https://github.com/Duke-MatSci/materialsmine
- Source snapshot inspected: `5a48d01d21b112aa36c62be22e470cc00620ef74`
- Public search route used for the fixed cache: `https://materialsmine.org/api/search/filter`
- Repository license at the inspected snapshot: CC BY-NC-SA 4.0

The raw cache files and normalized records in this package are derived from the public MaterialsMine service. Reuse must respect the source terms and attribution requirements.

## NanoMine schema

- Repository: https://github.com/Duke-MatSci/nanomine-schema
- Snapshot inspected: `4a8c0dc1d56581500ed36f282f8ac9a44a463e2b`
- Paper: https://doi.org/10.1063/1.5046839

## PNCExtract

- Repository: https://github.com/ghazalkhalighinejad/PNCExtract
- Snapshot used: `46a8175f1371048ae3d381bbc6e01e5922321794`
- Repository license at the inspected snapshot: Apache License 2.0
- Paper: https://aclanthology.org/2025.naacl-long.185/

PNCExtract `sample_data` was used only for the composition-only breadth check. Its article metadata was used as secondary author/location context; neither field was included in a similarity feature.

## Acquisition note

Acquisition date: 2026-09-10 UTC. The live MaterialsMine SPARQL wrapper returned HTTP 503 during acquisition. The analysis therefore uses public, website-generated cached query responses. `data/input_manifest.json` records SHA-256 hashes for every raw cache file.

