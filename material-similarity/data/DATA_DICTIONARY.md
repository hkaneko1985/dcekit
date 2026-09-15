# Data dictionary

The record inputs are UTF-8 JSON Lines compressed with gzip: one object per line. Null, empty containers, absent keys, unreported settings, and structural nonapplicability must not be collapsed into numeric zero. Source adapters in `src/run_validation.py` assign the core reporting states.

## HTEM: `upstream/htem_phase2_records.jsonl.gz`

1,891 records; unit = sample library. A library can contain several spatial composition measurements.

| Field | Meaning |
| --- | --- |
| `sample_library_id` | Source identifier; the adapter prefixes it with `htem-` |
| `elements` | Element names recorded for the library |
| `composition_measurements` | Element-to-amount dictionaries; usable measurements are normalized, then averaged for each library |
| `process` | Deposition fields; values can be numbers, lists, strings encoding lists, or null |
| `validation_only` | Context dictionary including deposition instrument; excluded by the distance adapter |

Retained process keys: `deposition_base_pressure_mtorr`, `deposition_compounds`, `deposition_cycles`, `deposition_energy`, `deposition_gas_flow_sccm`, `deposition_gases`, `deposition_growth_pressure_mtorr`, `deposition_initial_temp_c`, `deposition_power`, `deposition_rep_rate`, `deposition_sample_time_min`, `deposition_substrate_material`, `deposition_target_pulses`, and `deposition_ts_distance`.

Unit suffixes are retained exactly. The adapter uses field-specific unit keys rather than assuming unspecified units. Compounds, gases, and substrates become categorical sets; usable numeric fields become global settings. The adapter does not invent a method or sequence. Averaging reported spatial compositions is a library-level summary, not filling an absent setting.

## NanoMine: `upstream/nanomine_phase3_records.jsonl.gz`

832 records; 119 paper groups; unit = documented sample.

| Field | Meaning |
| --- | --- |
| `sample_id`, `sample_uri` | Sample identity and original URI |
| `article_id`, `paper_group` | Article/group identifiers used as context |
| `doi` | DOI string or source placeholder; `unpublished/…` is not a registered DOI |
| `sample_label`, `sample_label_from_component` | Source/descriptive labels |
| `matrix_names`, `filler_names`, `surface_names` | Constituent names, not chemically inferred identities |
| `reported_roles` | Roles explicitly represented in the normalized source |
| `components` | Entries with role, name, and attributes |
| `process_families` | Documented method categories |
| `steps` | Operation entries with URI, index, token, labels, and settings |
| `step_sequence`, `step_types`, `settings` | Retained source summaries; the adapter constructs core steps from `steps` |
| `citation_metadata` | Nullable context object: title, authors, location, citation DOI |

Attributes retain original type/value/unit text and, where available, normalized kind, numeric value, and unit group. They may describe density, loading, dimensions, or source/trade names. Such constituent descriptors are distinct from a selected prediction target. The adapter uses names and reported numeric constituent attributes; it does not infer elemental composition from polymer names.

Settings retain the source key, kind, value, and available units. Step order and occurrences are preserved. Mass fractions and volume fractions remain different keys; no density-based conversion is invented. The full live-service-to-JSONL extraction history is not reconstructed by these scripts.

## Frozen aggregates

| File | Content |
| --- | --- |
| `upstream/phase4_summary.json` | Earlier counts, diagnostics, and historical project decisions |
| `upstream/cross_dataset_diagnostics.csv` | Resource-specific contextual diagnostics |
| `upstream/objective_disagreement.csv` | Global-clustering versus local-retrieval diagnostics |
| `upstream/uncertainty_sensitivity.csv` | Sensitivity across missingness bounds |
| `upstream/view_disagreement_summary.csv` | ARI summary across principal views |
| `upstream/weight_response.csv` | Process-weight response summaries |

Metrics differ across resources. `global_gain` and `local_gain` denote contextual diagnostic changes, not improvements in material performance. The retained phase-specific grids are included under `../upstream_analyses/`.

## Common-interface outputs

`../results/` contains reference API summaries, pair comparisons, consensus candidates, subset inventories, PCoA coordinates, and original-distance separation/neighbor metrics. IDs link rows to normalized inputs. The runner regenerates these 13 files separately from the frozen results. Publication labels are not independent truth labels for material equivalence.

## Recovered Starrydata cohort

`cohorts/starrydata_phase1_cohort.jsonl.gz` contains 5,001 selected records. Identify a sample by **(SID, sample_id)**, not sample_id alone. `composition` is the original formula string; `form_categories`, `fabrication_categories`, and `purity_categories` are category lists. `relative_density` and `grain_size_log10_um` are nullable legacy descriptors. They are not process setpoints. `partition` and `F0_cluster` through `F3_cluster` are historical analysis outputs and must not enter a distance calculation.

The historical pilot explored form, purity, density, and grain descriptors and selected cluster count by an internal criterion. It predates the later process-focused design and is retained for provenance; those choices are not silently imposed on the final common API. No target-property curves are included. The original full-source selection cannot be rerun without its exact missing input snapshot; the SHA-256 is retained in the Phase 1 run configuration.

## Recovered PNCExtract cohort

`cohorts/pncextract_breadth_records.jsonl.gz` contains 1,103 normalized records. `sample_id`, `article_id`, `paper_group`, and `doi` identify provenance. `matrix_names`, `filler_names`, and `loading` supply the composition/loading comparison. `loading` has optional `MassFraction` and `VolumeFraction` values. `citation_metadata` is secondary paper/author/location context, not a similarity feature. Empty process lists reflect this composition-only extraction scope; they do not prove that a physical process was absent.

The optional `--records` argument to the bundled PNCExtract breadth runner loads this exact normalized cohort without reacquiring the source repository. It does not infer missing values.

## Version 0.2.0 audited representation

`config/numeric_dictionary.json` defines quantities and units. Numeric values without supported units are retained as quarantined source evidence and excluded from numerical comparison. Duplicate settings retain all entries, source step URI and a multiset comparison. Primary constituent inputs use only names and mass/volume fractions. `source_identifiers.csv` distinguishes DOI-formatted strings from unpublished placeholders; legacy column names containing `paper` refer to article groups. Registry validation has a separate status, with source spelling retained.
