# Repository Guidelines

## Project Structure & Module Organization
- `ML/`: model, trainer, and predictor components.
- `data_loader/`: dataset loading utilities and data wrappers.
- `hamiltonian/`, `MD/`, `MC/`: physics/MD/MC-related modules.
- `utils/`: shared helpers (logging, device setup, parameter checks).
- Top-level entrypoints: `train_main.py` (training) and `maintest_combined.py` (evaluation).
- Configuration files: `main_config.yaml`, `maintest_config.yaml`.
- Scripts and analysis: `run.sh`, `show_results.sh`, `plot_output_pwnet.py`, `plot_tau.py`.
- Outputs typically land under `results/` (created at runtime).

## Build, Test, and Development Commands
- `python train_main.py`: run training; uses parameters loaded from `main_config.yaml` (edit this for experiments).
- `python maintest_combined.py`: run evaluation/inference using checkpoints referenced in `maintest_config.yaml`.
- `python compare_with_baseline.py`: run one training pass and compare epoch-20 checkpoint vs baseline.
- `bash run.sh`: batch post-processing of logs in `results/` and summary generation via `show_results.sh`.
- `bash show_results.sh <logfile>`: summarize a single log file.
- `python plot_output_pwnet.py` or `python plot_tau.py`: optional plotting utilities.

## Coding Style & Naming Conventions
- Python code uses 4-space indentation and `snake_case` for variables/functions.
- Keep additions consistent with existing module structure (`ML/`, `utils/`, etc.).
- No formatter/linter is configured in-repo; avoid reformatting unrelated code.

## Testing Guidelines
- There is no dedicated test framework; validation is performed via `maintest_combined.py`.
- Use small data slices when iterating, then rerun full evaluation for reporting.
- Check outputs under `results/` for metrics and logs.

## Commit & Pull Request Guidelines
- Git history shows short, descriptive commit messages (no formal convention).
- Use concise, imperative messages (e.g., “Fix data loader shape handling”).
- PRs should include: purpose, key config changes, data requirements, and any new result plots/log summaries.

## Configuration & Data Notes
- Many paths point to external datasets (e.g., `../thumbnail_data/...`). Verify local data availability.
- Checkpoint paths are embedded in scripts; update configs when moving or renaming runs.
