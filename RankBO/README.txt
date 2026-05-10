# Demo ranking-based Bayesian Optimization

This repository provides a reproducible notebook workflow for demo ranking-based Bayesian optimization (RankBO) on the Direct Arylation benchmark.

The workflow is designed for a candidate-condition table without yield values, together with a fixed initial observation file and an update template for newly observed experimental results.

---

## Overview

The notebook supports the following workflow:

1. Load candidate conditions from `Demo_Direct_Arylation_conditions.csv`
2. Load fixed initial observations from `Demo_Direct_Arylation_initial_observations.csv`
3. Select a kernel once from the initial observations (or use a fixed kernel)
4. Propose the next experimental batch
5. Enter newly observed yields into `Demo_update_template.csv`
6. Update the campaign state and continue the optimization
7. Save progress so that the campaign can be resumed later

This public workflow uses:

- **Thompson sampling** as the acquisition rule
- **z-score latent y** as the latent representation for ranking

---

## Repository structure

A typical repository structure is:

```text
.
├── README.md
├── Demo_ranking_based_BO.ipynb
├── Demo_Direct_Arylation_conditions.csv
├── Demo_Direct_Arylation_initial_observations.csv
├── Demo_update_template.csv
└── Direct_Arylation_result.xlsx   # reference / answer key, not used as BO input
```

> `Direct_Arylation_result.xlsx` is provided only as reference data and is not used as input to the BO workflow.

---

## Input files

### 1. `Demo_Direct_Arylation_conditions.csv`

Candidate-condition file **without yield values**.

Expected structure:

- **Column A**: `index`
- **Columns B–D**: human-readable condition labels
  - `base`
  - `ligand`
  - `solvent`
- **Columns E onward**: model input columns
  - `concentration`
  - `temperature`
  - descriptor variables used internally by the BO model

When the notebook proposes experiments, it displays only:

- `index`
- `base`
- `ligand`
- `solvent`
- `concentration`
- `temperature`

---

### 2. `Demo_Direct_Arylation_initial_observations.csv`

Fixed initial observation file.

Required columns:

```csv
index,yield
790,13.88
1707,4.87
...
```

This file is used as the initial observed dataset at the start of the campaign.

---

### 3. `Demo_update_template.csv`

Template for entering newly observed yields after each experimental round.

Required columns:

```csv
round,batch_order,index,yield
1,1,1706,
1,2,1708,
...
```

After performing the proposed experiments, fill in the `yield` column and rerun the update cell.

---

## Notebook file

The main notebook is:

Demo_ranking_based_BO.ipynb



---

## User-configurable settings

The notebook exposes the following user settings:

- `SEED`
- `BATCH_SIZE`
- `FOLD_NUMBER`
- `KERNEL_SELECTION_MODE` (`"cv"` or `"fixed"`)

These settings are intended to be the only user-facing controls in the public workflow.

---

## Fixed settings in this public workflow

The notebook fixes the following choices in order to keep the workflow simple and reproducible:

- **Acquisition rule**: Thompson sampling
- **Latent y representation**: z-score
- Kernel selection for proposal:
  - the kernel is selected once from the initial observations when `KERNEL_SELECTION_MODE="cv"`
  - the selected kernel is reused in later proposal rounds

---

## Reproducibility notes

To reproduce the same proposal sequence:

- use the same `SEED`
- use the same `Demo_Direct_Arylation_initial_observations.csv
.csv`
- start from a clean `OUTDIR`
- keep the same candidate table and model input columns

The notebook saves campaign state in the output directory, including proposal history and observed results, so that the optimization can be resumed later.

---

## How to run

### Initial run

1. Open `Demo_ranking_based_BO.ipynb`
2. Adjust the user settings if needed
3. Run the notebook cells from top to bottom
4. Execute the proposal cell
5. Review `current_proposal.csv` and `Demo_update_template.csv`

### After experiments are performed

1. Fill in the observed yields in `Demo_update_template.csv`
2. Rerun the **Update cell**
3. Rerun the **propose cell**
4. Continue the campaign as needed

---

## Outputs

The notebook writes the following files in `OUTDIR`:

- `run_state.json`
- `observed_results.csv`
- `proposal_history.csv`
- `campaign_trajectory.csv`
- `best_so_far_log.csv`
- `seedwise_results.csv`
- `current_proposal.csv`

These files allow the campaign to be resumed and analyzed later.

---

## Notes

- The candidate-condition file is used as the BO input.
- The result file (`Direct_Arylation_result.xlsx`) is not used by the notebook during optimization.


---

## Citation / acknowledgement

If you use this workflow in your own work, please cite the associated manuscript or repository as appropriate.
