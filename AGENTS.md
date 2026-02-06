# !!!!!  This file is for Codex use only. You can safely ignore it.  !!!!!

# Repository Guidelines

## Project Structure & Module Organization
- `ML/`: model, trainer, and predictor components.
- `data_loader/`: dataset loading utilities and data wrappers.
- `hamiltonian/`, `MD/`, `MC/`: physics/MD/MC-related modules.
- `utils/`: shared helpers (logging, device setup, parameter checks).
- Top-level entrypoints: `train_main.py` (training) and `test_main.py` (evaluation).
- Configuration files: `train_config.yaml`, `test_config.yaml`.
- Scripts and analysis: `utils/show_results.sh`, `plot_output_pwnet.py`, `plot_tau.py`.
- Outputs typically land under `results/` (created at runtime).

## Build, Test, and Development Commands
- `python train_main.py`: run training; uses parameters loaded from `train_config.yaml` (edit this for experiments).
- `python test_main.py`: run evaluation/inference using checkpoints referenced in `test_config.yaml`.
- `python compare_with_baseline.py`: run one training pass and compare epoch-20 checkpoint vs baseline.
- `bash utils/show_results.sh <logfile>`: summarize a single log file.
- `python plot_output_pwnet.py` or `python plot_tau.py`: optional plotting utilities.

## Coding Style & Naming Conventions
- Python code uses 4-space indentation and `snake_case` for variables/functions.
- Keep additions consistent with existing module structure (`ML/`, `utils/`, etc.).
- No formatter/linter is configured in-repo; avoid reformatting unrelated code.

## Testing Guidelines
- There is no dedicated test framework; validation is performed via `test_main.py`.
- Use small data slices when iterating, then rerun full evaluation for reporting.
- Check outputs under `results/` for metrics and logs.

## Commit & Pull Request Guidelines
- Git history shows short, descriptive commit messages (no formal convention).
- Use concise, imperative messages (e.g., “Fix data loader shape handling”).
- PRs should include: purpose, key config changes, data requirements, and any new result plots/log summaries.

## Configuration & Data Notes
- Many paths point to external datasets (e.g., `../thumbnail_data/...`). Verify local data availability.
- Checkpoint paths are embedded in scripts; update configs when moving or renaming runs.
