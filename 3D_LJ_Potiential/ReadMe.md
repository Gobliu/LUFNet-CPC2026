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

## Reproducibility / deterministic runs (CUDA)
If you enable deterministic algorithms in PyTorch and see a CuBLAS warning, set the workspace config before running:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python train_main.py
```

To save a baseline log (stdout) for later comparison:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1 python -u train_main.py | tee results/baseline_run/baseline.log
```

To compare later, you can save a new log and run the comparer:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1 python -u train_main.py | tee /tmp/new_run.log
python compare_runs.py --log-a results/baseline_run/baseline.log --log-b /tmp/new_run.log
```

Checkpoint comparison (recommended if you might change stdout formatting later):

```bash
python compare_runs.py --ckpt-a /path/to/baseline.pth --ckpt-b /path/to/new.pth
```

For convenience, `code/compare_with_baseline.py` will run training once, then compare the checkpoint
at a fixed epoch (default 20) in `results/baseline_run/` with the new run's checkpoint under
`results/` (excluding the baseline folder).
From `3D_LJ_Potiential/code`:

```bash
python compare_with_baseline.py
```

Optional overrides:

```bash
python compare_with_baseline.py --epoch 20 --baseline-ckpt results/baseline_run/baseline000020.pth
python compare_with_baseline.py --train-cmd "CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1 python -u train_main.py"
```

## Baseline + refactor comparison
Use `code/run_compare_baseline.sh` to run two trainings and compare logs (and optionally checkpoints).

One-liner from `3D_LJ_Potiential/code`:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 ENV_NAME=pytorch RUN_CMD="python train_main.py" OUTPUT_DIR=/tmp/lufnet_baseline bash run_compare_baseline.sh
```

After the run:
- Baseline log: `/tmp/lufnet_baseline/run1.log`
- Refactor log: `/tmp/lufnet_baseline/run2.log`
- Compare helper: `code/compare_runs.py`

## Outputs
Training and evaluation logs and checkpoints are written under `code/results/` at runtime. Post-processing helpers live in:
- `code/run.sh`
- `code/show_results.sh`

## Data notes
Paths in configs may assume external datasets. If you relocate data, update the paths in the config files accordingly.

## Tips
- Start with smaller datasets (e.g., lower `dpt_train` / `dpt_valid`) when iterating.
- Keep `ngrid: 12` for 3D as expected by the current implementation.
