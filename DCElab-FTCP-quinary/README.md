# FTCP-VAE for Five-Element (Quinary) Inorganic Crystals

A variational autoencoder (VAE) for crystal structure reconstruction and property prediction of ternary to quinary inorganic crystals, based on the Fourier-transformed crystal properties (FTCP) representation.

This repository contains the code used in:

> Onishi, I. et al. "Variational Autoencoder Extension to Multi-Component Inorganic Crystals: Architecture and Reconstruction." *Chemical Engineering Research and Design* (under revision, CHERD-D-26-01837).

---

## Attribution

This repository is built on top of the original FTCP-VAE implementation by Ren et al.:

> Ren, Z., Tian, S. I. P., Noh, J., Oviedo, F., Xing, G., Li, J., ... & Buonassisi, T. (2022).
> An invertible crystallographic representation for general inverse design of inorganic crystals with targeted properties.
> *Matter*, 5(1), 314–335.
> https://doi.org/10.1016/j.matt.2021.11.032

**Original repository:** https://github.com/PV-Lab/FTCP

The following files are taken from, or based on, the original repository:

| File | Source |
|------|--------|
| `data/atom_init.json` | PV-Lab/FTCP (unmodified) |
| `data/element.pkl` | PV-Lab/FTCP (unmodified) |
| `data/thermoelectric_prop.csv` | PV-Lab/FTCP (unmodified) |
| `module/data.py` | PV-Lab/FTCP (modified) |
| `module/sampling.py` | PV-Lab/FTCP (modified) |
| `module/utils.py` | PV-Lab/FTCP (modified) |
| `module/try_data_query.py` | Adapted from `data_query` in PV-Lab/FTCP |

Each modified or adapted file carries a notice at its top stating its origin in PV-Lab/FTCP.

The following files are newly written or substantially modified in this work:

| File | Description |
|------|-------------|
| `01_main.py` | Training of all 42 models and per-model evaluation (validation and test sets) |
| `02_evaluate_all_models.py` | Evaluation of all 42 models by subset (Tables S1/S2) and validation-based model selection |
| `03_compare_conventional_vs_best.py` | Comparison of the conventional and proposed models on the test set (Tables 4–6, Figures 4–6, S1, S5) |
| `04_narrow_inverse_design_claims.py` | Coordination number, space group, and unit cell volume analyses (Figures S2–S4) |
| `05_structure_match_rate.py` | StructureMatcher-based structure match rate (Section 3.4) |
| `module/FTCP.py` | VAE model with modular encoder/decoder and supervised/unsupervised switching |
| `module/vae/encoder.py` | New: CNN pattern × network pattern encoder builder |
| `module/vae/decoder.py` | New: CNN pattern × network pattern decoder builder |
| `module/result_analysis.py` | New: analysis utilities |

---

## Repository Structure

```
.
├── 01_main.py                          # Training and per-model evaluation
├── 02_evaluate_all_models.py           # 42-model evaluation and model selection (validation set)
├── 03_compare_conventional_vs_best.py  # Conventional vs. proposed model (test set)
├── 04_narrow_inverse_design_claims.py  # CN match rate, space group, unit cell volume
├── 05_structure_match_rate.py          # StructureMatcher match rate
├── module/
│   ├── data.py                         # FTCP representation (from PV-Lab/FTCP)
│   ├── utils.py                        # pad, min-max scaling (from PV-Lab/FTCP)
│   ├── sampling.py                     # CIF generation (from PV-Lab/FTCP)
│   ├── try_data_query.py               # Data query utility (from PV-Lab/FTCP)
│   ├── FTCP.py                         # FTCP_VAE model and LossHistoryCallback
│   ├── result_analysis.py              # Analysis utilities used by 02–05
│   └── vae/
│       ├── encoder.py                  # EncoderPattern class
│       └── decoder.py                  # DecoderPattern class
└── data/
    ├── element.pkl                     # Element list (103 elements) for FTCP encoding (from PV-Lab/FTCP)
    ├── atom_init.json                  # Elemental property vectors from CGCNN (from PV-Lab/FTCP)
    └── thermoelectric_prop.csv         # Optional thermoelectric property labels (not used in the paper)
```

> **Note:** The dataset CSV (`data_query_3_5_elements_property_nsites_below_112.csv`, ~180 MB) is not included in this repository. See [Data](#data).

---

## Environment Setup

The results in the paper were obtained with the following environment (`ftcp_env`):

| Package | Version |
|---------|---------|
| Python | 3.7.16 |
| TensorFlow | 1.15.5 (`tf.keras`) |
| NumPy | 1.21.6 |
| pandas | 1.3.5 |
| scikit-learn | 1.0.2 |
| pymatgen | 2022.0.4 |
| matplotlib | 3.5.3 |
| seaborn | 0.12.2 |

```bash
conda create -n ftcp_env python=3.7
conda activate ftcp_env
pip install tensorflow==1.15.5 numpy==1.21.6 pandas==1.3.5 scikit-learn==1.0.2 \
            pymatgen==2022.0.4 matplotlib==3.5.3 seaborn==0.12.2 joblib tqdm
```

> **Windows users:** Some packages require a C++ compiler. Install **Desktop development with C++** from [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) before running `pip install`, then restart your terminal.

To export the active environment for archiving:

```bash
conda env export > ftcp_env.yml
```

---

## Data

The dataset CSV (`data_query_3_5_elements_property_nsites_below_112.csv`, ~180 MB) is hosted on Meiji University's data storage because of its file size:

**Download:** [data_query_3_5_elements_property_nsites_below_112.csv](https://meijiuniversity-my.sharepoint.com/:x:/g/personal/hkaneko_meiji_ac_jp/IQDiczhDRo9vSqn_fKLKkPUDAc9aHesI5rKEIECbN2d8j3w?e=CIp2pR)

The dataset was retrieved from the Materials Project with the following criteria:

- three to five constituent elements
- at most 112 sites per unit cell
- energy above the convex hull of 0.08 eV/atom or less
- formation energy (`formation_energy_per_atom`) and band gap (`band_gap`) available

It contains 77,284 structures (43,582 ternary, 25,748 quaternary, and 7,954 quinary) composed of 87 element species.

After downloading, place the file in the project root:

```
DCElab-FTCP-quinary/
├── data_query_3_5_elements_property_nsites_below_112.csv   <- here
├── 01_main.py
└── ...
```

> **Note:** The query utilities (`module/data.py: data_query`, `module/try_data_query.py`) are not needed to reproduce the results. Use the provided CSV as is.

---

## Usage

Run the scripts in order from the project root. Each script reads the outputs of the previous ones.

```bash
python 01_main.py                            # training (all 42 models)
python 02_evaluate_all_models.py             # Tables S1/S2 and model selection
python 03_compare_conventional_vs_best.py    # Tables 4–6, Figures 4–6, S1, S5
python 04_narrow_inverse_design_claims.py    # Figures S2–S4
python 05_structure_match_rate.py            # structure match rate
```

### Step 1: Training (`01_main.py`)

Key settings at the top of `01_main.py`:

```python
data_name               = 'data_query_3_5_elements_property_nsites_below_112'
network_pattern_numbers = ["0", "1", "2", "3", "4", "5", "6"]
cnn_patterns            = ['0', '1', '2']
supervised_modes        = [True, False]   # SVAE and USVAE
epochs                  = 200
batch_size              = 256
learning_rate           = 5e-4
coeff_KL                = 2               # beta
coeff_prop              = 10              # lambda (0 for USVAE)
restart_flag            = True            # resume from saved weights if available
code_test               = False           # True: quick 100-sample smoke test
```

The data are split into training (70%, 54,098), validation (10%, 7,729), and test (20%, 15,457) sets. The min-max scalers for the FTCP tensors and the target properties are fitted on the training set only and applied to all three sets without refitting. The split indices and scalers are saved and reused by scripts 02–05.

Outputs:

```
data_query_3_5_elements_property_nsites_below_112_result/
├── FTCP_representation_full.npz     # cached FTCP tensors
├── split_indices.pkl                # train/validation/test indices
├── scaler_X.pkl, scaler_y.pkl       # scalers fitted on the training set
└── CNN_{cnn_pattern}/
    ├── SVAE_result/
    │   ├── recon_result/            # FTCP_evaluation_pattern*.csv (validation),
    │   │                            # FTCP_evaluation_TEST_pattern*.csv (test, not used for selection)
    │   └── learning_log/            # loss CSVs and model weights (VAE_weight_pattern*)
    └── USVAE_result/
        └── ...
```

### Step 2: Evaluation of all models and model selection (`02_evaluate_all_models.py`)

Evaluates all 42 models on the validation and test sets, separately for the 3–4-element and 5-element subsets, and selects the best SVAE and the best USVAE using **validation-set metrics only** (sum of the ranks of element accuracy, lattice constant MAPE, lattice angle MAPE, and atomic coordinate MAE). The test set is never used for selection.

Outputs (`result_for_paper/`):

| File | Content |
|------|---------|
| `TableS1_val_3to4elements.csv`, `TableS2_val_5elements.csv` | Validation results of all 42 models (manuscript Tables S1, S2; Table 3 is taken from S2) |
| `TableS3_test_3to4elements.csv`, `TableS4_test_5elements.csv` | Test results of all 42 models (source of manuscript Tables 4–6 for the selected models) |
| `*_element_acc_valid_slots.csv` | Element accuracy counted over occupied slots only |
| `model_ranking_svae.csv`, `model_ranking_usvae.csv` | Ranking used for model selection |
| `best_model.json` | Selected models (conventional, best SVAE, best USVAE) |

> The file names `TableS3`/`TableS4` are internal names; these test-set tables are not reproduced as supplementary tables in the manuscript.

### Step 3: Conventional vs. proposed model (`03_compare_conventional_vs_best.py`)

Compares the conventional model (CNN 0, pattern 0, SVAE) with the proposed model (CNN 1, pattern 2, SVAE) on the test set.

| Output (`result_for_paper/`) | Manuscript |
|------|------------|
| `fig_breakdown/FigS1_metric_comparison.png` | Figure S1 |
| `fig_slot_accuracy/slot_accuracy.png` | Figure 4 |
| `fig_slot_accuracy/FigS5_element_worst10_*.png` | Figure S5 |
| `fig_bond_error/bond_error_comparison_5el.png` | Figure 5 |
| `fig_latent/Ef_Eg_*.png` | Figure 6 |
| `cif_samples_100/` | 100 test structures (35 ternary, 35 quaternary, 30 quinary) used in scripts 04 and 05 |
| `best_worst_reconstructed/` | Per-structure errors and best/worst reconstructed structures |
| `summary_main_table.csv`, `summary_table4_table5_source.csv` | Summary tables |

### Step 4: Local structure, symmetry, and volume (`04_narrow_inverse_design_claims.py`)

For the 100 test structures in `cif_samples_100/`:

| Output (`result_for_paper/fig_additional/`) | Manuscript |
|------|------------|
| `volume_error_box_violin.png`, `volume_*.csv` | Figure S2 |
| `cn_match_rate_100.png`, `cn_*.csv` | Figure S3 |
| `spacegroup_conservation_100.png`, `spacegroup_*.csv` | Figure S4 |

### Step 5: Structure match rate (`05_structure_match_rate.py`)

Computes the match rate and normalized RMS displacement with pymatgen `StructureMatcher` (ltol = 0.3, stol = 0.5, angle_tol = 10; default volume scaling) for the 100 test structures. Outputs: `result_for_paper/fig_structure_match/*.csv` (no figure).

---

## Network Architecture Patterns

Training and evaluation cover combinations of **CNN patterns** and **network patterns**, which are selected independently in `01_main.py`.

### CNN Patterns

Define the convolutional structure of the encoder (Conv1D) and decoder (Conv2DTranspose).

| Pattern | Encoder Conv1D layers (filters, kernel, stride) | Notes |
|---------|------------------------------------------------|-------|
| 0 | (32, 5, 2) → (64, 3, 2) → (128, 3, 1) | Conventional (same as PV-Lab/FTCP) |
| 1 | (32, 3, 1) → (64, 3, 2) → (128, 3, 2) | Resolution-preserving first layer (proposed) |
| 2 | (16, 3, 1) → (32, 3, 2) → (64, 3, 2) → (128, 3, 1) | Additional Conv1D layer |

### Network Patterns

Define the dense layers between the convolutional output and the latent space z. Encoder dense layers use sigmoid activation; decoder dense layers use ReLU activation.

| Pattern | Encoder FC → z | z dim | Decoder z → FC |
|---------|---------------|-------|----------------|
| 0 | 1024 → z | 256 | z (no dense layer) — conventional (Ren et al.) |
| 1 | z | 256 | z |
| 2 | z | 512 | z — proposed |
| 3 | 1024 → z | 256 | z → 1024 |
| 4 | 2048 → z | 1024 | z → 2048 |
| 5 | 1024 → 512 → z | 256 | z → 512 → 1024 |
| 6 | 2048 → 1024 → z | 512 | z → 1024 → 2048 |

---

## Reproducibility

| Item | Value |
|------|-------|
| Train/validation/test split | 70/10/20 (`train_test_split`, `random_state=21`, test split first) |
| Scaling | Min-max scalers fitted on the training set only |
| Random seed | 42 (Python, NumPy, TensorFlow) |
| Optimizer | Adam, learning rate 5e-4 |
| Learning rate schedule | ReduceLROnPlateau (monitor='loss', factor=0.3, patience=4, min_lr=1e-6); set to 1e-4 at epoch 50 and 5e-5 at epoch 100 |
| Epochs / batch size | 200 / 256 |
| Loss weights | beta = 2; lambda = 10 (SVAE), 0 (USVAE) |
| Model selection | Validation set only (see Step 2) |

---

## License

This repository is licensed under the Apache License 2.0 (see `LICENSE`). It contains files derived from [PV-Lab/FTCP](https://github.com/PV-Lab/FTCP), which is also distributed under the Apache License 2.0; files modified from or adapted from PV-Lab/FTCP are marked as such at the top of each file.
