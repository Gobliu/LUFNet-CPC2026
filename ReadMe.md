# LUFNet-CPC2026

This repository contains 2D and 3D Lennard-Jones (LJ) workflows and related assets.
## Corresponding Paper


## Dataset


## 2D Lennard-Jones Potential (LUFNet)
The code and thumbnail data will be released in the next version (expected in March 2026).

## 3D Lennard-Jones Potential (LUFNet)
This folder contains the 3D Lennard-Jones (LJ) workflow used in this project, including training/evaluation code and example datasets.

### Structure
- `3D_LJ_Potiential/code/`: training, evaluation, and analysis scripts.
  - Entry points: `train_main.py` (training) and `maintest_combined.py` (evaluation).
  - Configs: `main_config.yaml`, `maintest_config.yaml`.
  - Utilities: `utils/`, `data_loader/`, `ML/`, `hamiltonian/`, `MD/`, `MC/`.
- `3D_LJ_Potiential/thumbnail_data/`: sample datasets for MD/MC and training/validation.

### Quick start
From `3D_LJ_Potiential/code`:

```bash
python train_main.py
python maintest_combined.py
```

Training uses parameters in `main_config.yaml`. Evaluation uses `maintest_config.yaml`.

### Reproducibility / deterministic runs (CUDA)
If you enable deterministic algorithms in PyTorch and see a CuBLAS warning, set the workspace config before running:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python train_main.py
```

Checkpoint comparison (recommended if you might change stdout formatting later):

```bash
python compare_ckpt.py --ckpt-a /path/to/baseline.pth --ckpt-b /path/to/new.pth
```

`compare_ckpt.py` only supports checkpoint comparison via `--ckpt-a` and `--ckpt-b`.

For convenience, `3D_LJ_Potiential/code/compare_with_baseline.py` will run training once, then compare the checkpoint
at a fixed epoch (default 20) in `3D_LJ_Potiential/code/results/baseline_run/` with the new run's checkpoint under
`3D_LJ_Potiential/code/results/` (excluding the baseline folder) using `compare_ckpt.py`.
From `3D_LJ_Potiential/code`:

```bash
python compare_with_baseline.py
```

### Baseline + refactor comparison
Use `3D_LJ_Potiential/code/compare_with_baseline.py` to run training once and compare the baseline checkpoint
against the new run's checkpoint with `compare_ckpt.py`.

### Outputs
Training and evaluation logs and checkpoints are written under `3D_LJ_Potiential/code/results/` at runtime. Post-processing helpers live in:
- `3D_LJ_Potiential/code/run.sh`
- `3D_LJ_Potiential/code/show_results.sh`

### Data notes
Paths in configs may assume external datasets. If you relocate data, update the paths in the config files accordingly.

### Tips
- Start with smaller datasets (e.g., lower `dpt_train` / `dpt_valid`) when iterating.
- Keep `ngrid: 12` for 3D as expected by the current implementation.
