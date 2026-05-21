# FTCP-VAE for Five-Element Inorganic Crystals

A Variational Autoencoder (VAE) for crystal structure reconstruction and property prediction, based on the Fourier-Transformed Crystal Properties (FTCP) representation.

---

## Attribution

This repository is built on top of the original FTCP-VAE implementation by Ren et al.:

> Ren, Z., Tian, S. I. P., Noh, J., Oviedo, F., Xing, G., Li, J., ... & Buonassisi, T. (2022).
> An invertible crystallographic representation for general inverse design of inorganic crystals with targeted properties.
> *Matter*, 5(1), 314–335.
> https://doi.org/10.1016/j.matt.2021.11.032

**Original repository:** https://github.com/PV-Lab/FTCP

The following files are taken directly from the original repository with minimal modification:

| File | Source |
|------|--------|
| `data/atom_init.json` | PV-Lab/FTCP |
| `data/element.pkl` | PV-Lab/FTCP |
| `module/data.py` | PV-Lab/FTCP |
| `module/sampling.py` | PV-Lab/FTCP |
| `module/utils.py` | PV-Lab/FTCP |
| `module/try_data_query.py` | PV-Lab/FTCP |

The following files are newly written or substantially modified in this work:

| File | Description |
|------|-------------|
| `module/FTCP.py` | Rewritten VAE model with modular encoder/decoder and supervised/unsupervised switching |
| `module/vae/encoder.py` | New: CNN Pattern × Network Pattern encoder builder |
| `module/vae/decoder.py` | New: CNN Pattern × Network Pattern decoder builder |
| `module/result_analysis.py` | New: analysis utilities |
| `main.py` | Modified: multi-pattern training loop |
| `result_all.py` | New: cross-model comparison and figure generation |

---

## Repository Structure

```
.
├── main.py                        # Training and per-model evaluation
├── result_all.py                  # Cross-model comparison and figure generation
├── module/
│   ├── data.py                    # Data loading and FTCP representation (from PV-Lab/FTCP)
│   ├── utils.py                   # pad, minmax, inv_minmax (from PV-Lab/FTCP)
│   ├── sampling.py                # CIF generation from latent vectors (from PV-Lab/FTCP)
│   ├── try_data_query.py          # Data query utility (from PV-Lab/FTCP)
│   ├── FTCP.py                    # FTCP_VAE model and LossHistoryCallback
│   ├── result_analysis.py         # Analysis utilities imported by result_all.py
│   └── vae/
│       ├── encoder.py             # EncoderPattern class
│       └── decoder.py             # DecoderPattern class
├── data/
│   ├── element.pkl                # Element list for FTCP encoding (from PV-Lab/FTCP)
│   ├── atom_init.json             # Elemental property vectors from CGCNN (from PV-Lab/FTCP)
│   └── thermoelectric_prop.csv    # Optional thermoelectric property labels
└── environment/
    └── ftcp_env.yml               # Conda environment file
```

> **Note:** The dataset CSV (`data_query_3_5_elements_property_nsites_below_112.csv`, ~180 MB) is not included in this repository. See [Data](#data).

---

## Environment Setup

**Python 3.7 is required.** The following packages are incompatible with Python 3.8 or later:

| Package | Reason |
|---------|--------|
| `tensorflow==1.15.5` | Not available on PyPI for Python 3.8+ |
| `numpy==1.18.5` | Build fails on Python 3.9+ |
| `pymatgen==2021.3.9` | Compilation errors on Python 3.9+ |

```bash
conda create -n ftcp_env python=3.7   # 3.7 required; do not use 3.8 or later
conda activate ftcp_env

pip install numpy==1.18.5 pandas matplotlib scikit-learn joblib tqdm
pip install tensorflow==1.15.5 keras==2.3.1 protobuf==3.20.3  # or tensorflow-gpu==1.15.5 for GPU
pip install matminer==0.6.2 pymatgen==2021.3.9 monty==3.0.2 ase
pip install ruamel.yaml==0.17.21 ruamel.yaml.clib==0.2.7

# Optional: Spyder IDE
conda install -c conda-forge spyder=4.2.5
```

> **Windows users:** Some packages require a C++ compiler. Install **Desktop development with C++** from [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) before running `pip install`, then restart your terminal.

To export the active environment for archiving:

```bash
conda list --explicit > ftcp_env_snapshot.txt
```

---

## Data

The dataset CSV (`data_query_3_5_elements_property_nsites_below_112.csv`, ~180 MB) is hosted on Meiji University's data storage due to its file size:

**Download:** [data_query_3_5_elements_property_nsites_below_112.csv](https://meijiuniversity-my.sharepoint.com/:x:/g/personal/hkaneko_meiji_ac_jp/IQDiczhDRo9vSqn_fKLKkPUDAc9aHesI5rKEIECbN2d8j3w?e=CIp2pR)

After downloading, place the file in the project root:

```
DCElab-FTCP-quinary/
├── data_query_3_5_elements_property_nsites_below_112.csv   <- here
├── main.py
└── ...
```

> **Note:** `module/data.py` contains a `data_query` function for re-querying from the Materials Project API, but it depends on a legacy API client (`matminer==0.6.2`) that is no longer compatible with the current Materials Project API. Use the provided CSV as-is.

---

## Usage

### Step 1: Training

Configure the settings at the top of `main.py`, then run:

```bash
python main.py
```

Key settings:

```python
data_name               = 'data_query_3_5_elements_property_nsites_below_112'
network_pattern_numbers = ["0", "1", "2", "3", "4", "5", "6"]
cnn_patterns            = ['0', '1', '2']
supervised_modes        = [True, False]   # SVAE and USVAE
epochs                  = 200
batch_size              = 256
code_test               = False           # Set True for a quick 100-sample smoke test
```

Outputs are saved under:

```
data_query_3_5_elements_property_nsites_below_112_result/
└── CNN_{cnn_pattern}/
    ├── SVAE_result/
    │   ├── recon_result/       # Evaluation CSVs and latent space plots
    │   └── learning_log/       # Loss CSVs, loss plots, and model weights (.keras)
    └── USVAE_result/
        └── ...
```

### Step 2: Cross-model comparison and figure generation

After training, run:

```bash
python result_all.py
```

This script reads all evaluation CSVs produced by `main.py`, automatically detects the best-performing model via ranking, and generates comparison figures. Outputs are saved to `result_for_paper/`.

---

## Network Architecture Patterns

Training and evaluation cover combinations of **CNN Patterns** and **Network Patterns**, which are independently selectable in `main.py`.

### CNN Patterns

Define the convolutional structure of the encoder (Conv1D) and decoder (Conv2DTranspose).

| Pattern | Encoder first layer | Total Conv layers | Notes |
|---------|--------------------|--------------------|-------|
| 0 | kernel=5, stride=2 | 3 | Baseline (same as PV-Lab/FTCP) |
| 1 | kernel=3, stride=1 | 3 | Resolution-preserving in first stage |
| 2 | kernel=3, stride=1 | 4 | Additional Conv layer |

### Network Patterns

Define the fully connected layer structure between the convolutional output and the latent space z.

| Pattern | Encoder FC → z | z dim | Decoder z → FC |
|---------|---------------|-------|----------------|
| 0 | 1024 → z | 256 | z → 1024 (baseline) |
| 1 | z | 256 | z |
| 2 | z | 512 | z |
| 3 | 1024 → z | 256 | z → 1024 |
| 4 | 2048 → z | 1024 | z → 2048 |
| 5 | 1024 → 512 → z | 256 | z → 512 → 1024 |
| 6 | 2048 → 1024 → z | 512 | z → 1024 → 2048 |

---

## Reproducibility

| Item | Value |
|------|-------|
| Python | 3.7 (3.8+ not supported; see [Environment Setup](#environment-setup)) |
| TensorFlow | 1.15.5 (CPU) |
| Keras | 2.3.1 |
| Train/test split | 80/20, `random_state=21` |
| Random seed | 42 |
| Full environment | `environment/ftcp_env.yml` |

---

## License

The files originating from [PV-Lab/FTCP](https://github.com/PV-Lab/FTCP) are subject to their original license. All other files in this repository are released under the MIT License. See `LICENSE` for details.
