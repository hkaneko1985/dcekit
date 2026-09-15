# Revised Phase 1 — HTEM process-data audit

This package implements the revised Phase 1 of the property-free materials
similarity study. It audits synthesis/deposition metadata before clustering.
It does not use a property/target variable, expert label, missing-value
imputation, study identity, owner identity, or instrument identity as a
similarity feature.

The central distinction is between `REPORTED` and `NOT_REPORTED`. Blank values
are never converted into `NOT_PERFORMED` or `NOT_APPLICABLE` without an explicit
source assertion. Parallel target/gas arrays are not treated as process order.

## Reproduce

From this directory:

```bash
python src/prepare_phase1r_input.py \
  --modeling-records ../phase2r_material_similarity/data/htem_modeling_records.jsonl.gz \
  --raw-cache ../phase2_material_similarity/data/.fetch_cache \
  --studies ../phase2r_material_similarity/data/htem_studies.json \
  --output data/htem_process_audit_records.jsonl.gz \
  --study-output data/htem_study_links.json \
  --manifest data/input_manifest.json

python src/run_phase1r_audit.py \
  --input data/htem_process_audit_records.jsonl.gz \
  --studies data/htem_study_links.json \
  --spec config/phase1r_spec.json \
  --output-dir .

python -m unittest discover -s tests -v

python src/make_release.py \
  --root . \
  --archive ../phase1r_process_audit_bundle.zip
```

Read `reports/phase1r_report_ja.md` for the results. The machine-readable audit,
method-evidence templates, reporting matrices, and library-level inventory are
under `results/`.
