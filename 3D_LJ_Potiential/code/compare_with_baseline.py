#!/usr/bin/env python3
import argparse
import subprocess
import time
from pathlib import Path


def find_epoch_ckpts(root: Path, epoch: int, exclude: Path | None = None) -> set[Path]:
    """
    Search for checkpoints matching the mbpwXXXXXX.pth pattern.
    """
    pattern = f"mbpw{epoch:06d}.pth"
    results = set()
    for path in root.rglob(pattern):
        if exclude is not None and exclude in path.parents:
            continue
        results.add(path)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train once, then compare epoch checkpoint against baseline."
    )
    parser.add_argument(
        "--baseline-dir",
        default="results/baseline_run",
        help="Baseline directory to exclude from new-run search.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=0.0,
        help="Relative tolerance for checkpoint comparison.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=0.0,
        help="Absolute tolerance for checkpoint comparison.",
    )
    args = parser.parse_args()

    # Hardcoded configuration values
    epoch = 20
    baseline_ckpt_str = "results/baseline_run/baseline000020.pth"
    results_root_str = "results/baseline_run"
    train_cmd = "CUBLAS_WORKSPACE_CONFIG=:4096:8 python train_main.py"

    results_root = Path(results_root_str)
    baseline_dir = Path(args.baseline_dir)
    baseline_ckpt = Path(baseline_ckpt_str)

    print(f"Baseline checkpoint: {baseline_ckpt}")

    if not baseline_ckpt.exists():
        print(f"ERROR: baseline checkpoint not found: {baseline_ckpt}")
        return 1

    start_time = time.time()
    # Remove any existing epoch checkpoint to ensure the run creates a fresh file.
    existing = find_epoch_ckpts(results_root, epoch, exclude=baseline_dir)
    if existing:
        for path in existing:
            print(f"Removing existing checkpoint: {path}")
            path.unlink()

    print(f"Running training: {train_cmd}")
    proc = subprocess.run(train_cmd, shell=True)
    if proc.returncode != 0:
        print(f"ERROR: training command failed with exit code {proc.returncode}")
        return proc.returncode

    # Directly check for the existence of the expected checkpoint file with 6-digit padding
    # For epoch 20, this expects mbpw000020.pth
    new_ckpt = results_root / f"mbpw{epoch:06d}.pth"
    if not new_ckpt.exists():
        print(f"ERROR: expected checkpoint not found at {new_ckpt}")
        return 1

    print(f"Baseline checkpoint: {baseline_ckpt}")
    print(f"New checkpoint:      {new_ckpt}")

    compare_cmd = [
        "python",
        "compare_runs.py",
        "--ckpt-a",
        str(baseline_ckpt),
        "--ckpt-b",
        str(new_ckpt),
        "--rtol",
        str(args.rtol),
        "--atol",
        str(args.atol),
    ]
    print("Comparing checkpoints...")
    return subprocess.run(compare_cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())