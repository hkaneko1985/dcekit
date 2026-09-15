# Earlier analyses and frozen cohorts

These directories recover the earlier project source code, configurations, data and results. Scientific source files and retained outputs are preserved; the PNCExtract runner has one packaging change: `--records` can read the bundled normalized cohort, and its output data directory is created when needed. `../provenance/recovered_files.json` records the original hashes before that change.

Historical README/report/license notes are retained for provenance. Use the current top-level `NOTICE.md` and `data/SOURCES_AND_LICENSES.md` for the distinction between MaterialsMine **software** licensing and API data, and for the corrected PNCExtract reference. Historical milestone decisions are not new evidence of predictive transfer.

| Directory | Scope | Input status |
| --- | --- | --- |
| `phase1_starrydata` | Original Starrydata pilot, summary code and results | 5,001 selected features/assignments retained; original full `all_samples.csv.gz` missing |
| `phase1r_process_audit` | Revised process-family and reporting audit on HTEM | Fixed audit inputs, study links and specification retained |
| `phase2_htem` | Thin-film composition/settings weight grid and controls | 1,891 normalized libraries and study links retained |
| `phase3_nanomine` | NanoMine process/constituent grids and controls; PNCExtract breadth check | 832 NanoMine and 1,103 PNCExtract normalized records; NanoMine caches retained |
| `phase4_synthesis` | Cross-resource synthesis of frozen earlier results | All 17 required upstream artifacts retained |

## Commands

Install the top-level requirements first. Run phase-specific tests from the top-level repository directory, for example:

```bash
python -m unittest discover -s upstream_analyses/phase2_htem/tests -v
```

A PNCExtract rerun using the bundled cohort, without network acquisition:

```bash
python upstream_analyses/phase3_nanomine/src/run_pncextract_breadth.py --records data/cohorts/pncextract_breadth_records.jsonl.gz --output-root outputs/pncextract
```

Phase-specific scripts ordinarily write to their own `results/`. To preserve the shipped references, work in a copy of that phase directory. A cross-platform Python example for Phase 4 is:

```python
import shutil
import subprocess
import sys
from pathlib import Path

work = Path("outputs/phase4_synthesis")
shutil.copytree("upstream_analyses/phase4_synthesis", work, dirs_exist_ok=True)
subprocess.run([sys.executable, str(work / "src/run_phase4.py")], check=True)
```

In a working copy of Phase 2, change into that directory and run:

```bash
python src/run_phase2.py --config config/phase2_run_config.json
python src/run_posthoc_sensitivity.py --root .
```

In a working copy of Phase 3, change into that directory and run:

```bash
python src/run_phase3.py
python src/run_controls.py
python src/run_duplicate_sensitivity.py
```

These full grids can take longer than the common-interface examples. They are not launched by top-level `run_all.py`.

The revised process audit can be rerun from the top-level directory:

```bash
python upstream_analyses/phase1r_process_audit/src/run_phase1r_audit.py --input upstream_analyses/phase1r_process_audit/data/htem_process_audit_records.jsonl.gz --studies upstream_analyses/phase1r_process_audit/data/htem_study_links.json --spec upstream_analyses/phase1r_process_audit/config/phase1r_spec.json --output-dir outputs/phase1r
```

## Reconstruction limits

The Starrydata full-source runner checks the expected SHA-256 of `data/all_samples.csv.gz`. That historical file was not recovered. The selected records and all fields stored in their assignment artifact remain inspectable; do not rename this cohort to impersonate the missing raw CSV. No new source download is claimed to reproduce its cohort-selection step.

The original HTEM acquisition cache is not included, although normalized inputs, preparation code, and the complete retained clustering grid are present. NanoMine rebuilding from raw caches additionally uses citation/source directories outside the phase package. The normalized cohort is the fixed input for the included analysis.

The original Starrydata pilot included morphology/quality descriptors and an internal cluster-count selection. It is historical context, not a redefinition of the final process-focused, fixed-weight comparison framework. The current study does not infer physical process absence from unreported information, impute process setpoints, or optimize weights using source labels.
