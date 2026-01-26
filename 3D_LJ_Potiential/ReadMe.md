# 3D Lennard-Jones Potential (LUFNet)

This folder contains the 3D Lennard-Jones (LJ) workflow used in this project, including training/evaluation code and example datasets.

## Structure
- `code/`: training, evaluation, and analysis scripts.
  - Entry points: `train_main.py` (training) and `maintest_combined.py` (evaluation).
  - Configs: `main_config.yaml`, `maintest_config.yaml`.
  - Utilities: `utils/`, `data_loader/`, `ML/`, `hamiltonian/`, `MD/`, `MC/`.
- `data_sets/`: sample datasets for MD/MC and training/validation.

## Quick start
From `3D_LJ_Potiential/code`:

```bash
python train_main.py
python maintest_combined.py
```

Training uses parameters in `main_config.yaml`. Evaluation uses `maintest_config.yaml`.

## Outputs
Training and evaluation logs and checkpoints are written under `code/results/` at runtime. Post-processing helpers live in:
- `code/run.sh`
- `code/show_results.sh`

## Data notes
Paths in configs may assume external datasets. If you relocate data, update the paths in the config files accordingly.

## Tips
- Start with smaller datasets (e.g., lower `dpt_train` / `dpt_valid`) when iterating.
- Keep `ngrid: 12` for 3D as expected by the current implementation.
