# LUFNet-CPC2026

This repository provides **reproducible workflows for training and evaluating neural-network-driven molecular dynamics models** on **2D and 3D Lennard–Jones (LJ) systems**, as described in the accompanying *Computer Physics Communications* paper:

**Scalable Neural-Network–Driven Molecular Dynamics Simulation**  
*Computer Physics Communications* (2026)  
DOI: https://doi.org/10.1016/j.cpc.2026.110036

---

## Repository overview

- **2D Lennard–Jones (LUFNet)**  
  The 2D LJ workflow and associated thumbnail data are not included in this release and will be added in a future version (expected March 2026).

- **3D Lennard–Jones (LUFNet)**  
  This release contains the complete 3D LJ workflow, including:
  - training and evaluation code
  - example datasets
  - reproducibility and checkpoint comparison utilities

---

## Requirements

- Python ≥ 3.9  
- PyTorch ≥ 2.0  
- Linux (tested on Ubuntu)  
- CUDA-enabled GPU (optional but recommended)

---

## Typical workflow

1. Inspect or test the pipeline using example data (`thumbnail_data/`)
2. Configure training parameters in `main_config.yaml`
3. Train a model using `train_main.py`
4. Evaluate the trained model using `maintest_combined.py`
5. (Optional) Compare checkpoints against a baseline for reproducibility

---

## Directory structure (3D Lennard–Jones)

> Note: the directory name `3D_LJ_Potential` is kept for backward compatibility.

```
3D_LJ_Potential/
├── code/
│   ├── train_main.py
│   ├── maintest_combined.py
│   ├── main_config.yaml
│   ├── maintest_config.yaml
│   ├── compare_ckpt.py
│   ├── compare_with_baseline.py
│   ├── utils/
│   ├── data_loader/
│   ├── ML/
│   ├── hamiltonian/
│   ├── MD/
│   └── MC/
├── thumbnail_data/
```

---

## Quick start

```bash
cd 3D_LJ_Potential/code
python train_main.py
python maintest_combined.py
```

- Training parameters are read from `main_config.yaml`
- Evaluation parameters are read from `maintest_config.yaml`
- Outputs are written to `results/`

---

## Configuration files

- **`main_config.yaml`**  
  Training configuration: model architecture, optimizer, dataset paths, output directories.

- **`maintest_config.yaml`**  
  Evaluation and post-processing configuration.

All relative paths in YAML files are interpreted relative to  
`3D_LJ_Potiential/code`.

---

## Data

- **`thumbnail_data/`**  
  Small example datasets sufficient for pipeline testing and debugging.

- **Full dataset**  
  The full dataset (MD/MC trajectories, energies, and forces) is deposited on Zenodo.  
  DOI: (to be added)

Update dataset paths in `main_config.yaml` to use the full dataset.

---

## Reproducibility and deterministic execution (CUDA)

When using deterministic algorithms in PyTorch, CuBLAS requires an explicit workspace configuration:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python train_main.py
```

This enables deterministic GEMM behavior on supported GPUs.

---

## Checkpoint comparison (optional)

### Direct comparison

```bash
python compare_ckpt.py \
  --ckpt-a /path/to/baseline.pth \
  --ckpt-b /path/to/new.pth
```

### Baseline vs new run

```bash
python compare_with_baseline.py
```

This script:
- runs training once
- compares the checkpoint at a fixed epoch (default: 20)
- uses `results/baseline_run/` as the reference

---

## Outputs

All outputs are written under:

```
3D_LJ_Potiential/code/results/
```

This includes logs, checkpoints, and evaluation summaries.

---

## Dataset availability (2D LJ)

The 2D Lennard–Jones workflow and datasets are not included in this release and will be added in a future update.

---

## Citation

```bibtex
@article{LUFNetCPC2026,
  title   = {Scalable Neural-Network--Driven Molecular Dynamics Simulation},
  journal = {Computer Physics Communications},
  year    = {2026},
  doi     = {10.1016/j.cpc.2026.110036}
}
```

---

## Contact

For questions or issues related to this repository, please contact the authors listed in the corresponding paper.