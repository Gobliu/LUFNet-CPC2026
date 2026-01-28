#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import List, Tuple, Any, Dict

import torch


def read_text(path: str | Path) -> str:
    """Reads text from a file, ignoring encoding errors."""
    return Path(path).read_text(errors="ignore")


def compare_logs(log_a: str, log_b: str, max_diffs: int = 50) -> Tuple[List[Tuple[int, str, str]], int, int]:
    """
    Compares two log files line by line, ignoring timestamps to reduce noise.
    """
    a = read_text(log_a).splitlines()
    b = read_text(log_b).splitlines()

    # Ignore timestamps to reduce noise (e.g., 20231027, 12:00:00)
    time_re = re.compile(r"\b\d{8},\s\d{2}:\d{2}:\d{2}\b")
    a = [time_re.sub("<TIME>", line) for line in a]
    b = [time_re.sub("<TIME>", line) for line in b]

    diffs = []
    for i, (la, lb) in enumerate(zip(a, b), start=1):
        if la != lb:
            diffs.append((i, la, lb))
            if len(diffs) >= max_diffs:
                break

    extra_a = len(a) - len(b) if len(a) > len(b) else 0
    extra_b = len(b) - len(a) if len(b) > len(a) else 0

    return diffs, extra_a, extra_b


def is_equal(va: Any, vb: Any, rtol: float, atol: float) -> bool:
    """
    Recursively checks for equality between two objects, handling nested 
    dictionaries, lists, and PyTorch tensors.
    """
    if type(va) != type(vb):
        return False

    if torch.is_tensor(va):
        if va.shape != vb.shape:
            return False
        return torch.allclose(va, vb, rtol=rtol, atol=atol)

    if isinstance(va, dict):
        if va.keys() != vb.keys():
            return False
        return all(is_equal(va[k], vb[k], rtol, atol) for k in va)

    if isinstance(va, (list, tuple)):
        if len(va) != len(vb):
            return False
        return all(is_equal(ai, bi, rtol, atol) for ai, bi in zip(va, vb))

    # Default fallback for scalars
    return va == vb


def compare_checkpoints(ckpt_a: str, ckpt_b: str, rtol: float = 0.0, atol: float = 0.0) -> List[str]:
    """
    Compares two PyTorch checkpoints for parity using recursive matching.
    """
    # map_location='cpu' ensures we don't run out of VRAM during comparison
    a = torch.load(ckpt_a, map_location="cpu", weights_only=False)
    b = torch.load(ckpt_b, map_location="cpu", weights_only=False)

    if not isinstance(a, dict) or not isinstance(b, dict):
        return ["checkpoint is not a state dict"]

    keys = sorted(set(a.keys()) | set(b.keys()))
    mismatches = []
    for k in keys:
        if k not in a or k not in b:
            mismatches.append(f"{k} (missing key)")
            continue
            
        va = a[k]
        vb = b[k]
        
        try:
            if not is_equal(va, vb, rtol, atol):
                mismatches.append(k)
        except Exception as e:
            mismatches.append(f"{k} (comparison error: {str(e)})")
                
    return mismatches


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline vs refactor runs.")
    parser.add_argument("--log-a", help="baseline log file")
    parser.add_argument("--log-b", help="new log file")
    parser.add_argument("--ckpt-a", help="baseline checkpoint (.pth)")
    parser.add_argument("--ckpt-b", help="new checkpoint (.pth)")
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--atol", type=float, default=0.0)
    args = parser.parse_args()

    if args.log_a and args.log_b:
        diffs, extra_a, extra_b = compare_logs(args.log_a, args.log_b)
        if diffs:
            print("LOG_DIFFS")
            for line_no, la, lb in diffs:
                print(f"line {line_no}:")
                print(f"  A: {la}")
                print(f"  B: {lb}")
        else:
            print("LOGS_MATCH")
        if extra_a or extra_b:
            print(f"LOG_LENGTH_MISMATCH extra_a={extra_a} extra_b={extra_b}")

    if args.ckpt_a and args.ckpt_b:
        mismatches = compare_checkpoints(args.ckpt_a, args.ckpt_b, rtol=args.rtol, atol=args.atol)
        if mismatches:
            print(f"CKPT_MISMATCH keys={len(mismatches)}")
            print(mismatches[:20])
        else:
            print("CHECKPOINTS_MATCH")


if __name__ == "__main__":
    main()