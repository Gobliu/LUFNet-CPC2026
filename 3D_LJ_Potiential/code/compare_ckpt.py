#!/usr/bin/env python3
import argparse
from typing import List, Tuple, Any, Dict

import torch


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
    """Run CLI checkpoint comparison and print a summary."""
    parser = argparse.ArgumentParser(description="Compare baseline vs refactor checkpoints.")
    parser.add_argument("--ckpt-a", required=True, help="baseline checkpoint (.pth)")
    parser.add_argument("--ckpt-b", required=True, help="new checkpoint (.pth)")
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--atol", type=float, default=0.0)
    args = parser.parse_args()

    mismatches = compare_checkpoints(args.ckpt_a, args.ckpt_b, rtol=args.rtol, atol=args.atol)
    if mismatches:
        print(f"CKPT_MISMATCH keys={len(mismatches)}")
        print(mismatches[:20])
    else:
        print("CHECKPOINTS_MATCH")


if __name__ == "__main__":
    main()
