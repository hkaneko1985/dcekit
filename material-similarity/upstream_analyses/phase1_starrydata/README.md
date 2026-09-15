# Historical Starrydata pilot

The original parser, clustering runner, summary/plotting scripts, configurations, tests and retained outputs are collected here. `results/phase1_cluster_assignments.jsonl.gz` contains all 5,001 selected records and their documented features. A byte-identical copy is available at `../../data/cohorts/starrydata_phase1_cohort.jsonl.gz`.

`src/run_phase1.py` requires the original full `data/all_samples.csv.gz`, whose expected SHA-256 is recorded in `config/phase1_run_config.json`. That source file is not included, so full-source cohort selection cannot be rerun from this directory alone. The selected metadata and existing clusters are available for inspection and new analysis using `src/phase1_core.py`. Target curves were not used.

The pilot's historical descriptor and internal cluster-count choices are documented in its configuration/report and differ from the later process-focused framework. It is preserved as provenance rather than presented as a newly executed validation. Current source terms and attribution are in the repository-level `data/SOURCES_AND_LICENSES.md`.
