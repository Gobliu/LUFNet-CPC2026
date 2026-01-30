import os
import random

import numpy as np
import torch


def set_global_seed(seed, deterministic=False):
    """Seed Python, NumPy, and PyTorch RNGs.

    Args:
    seed (int | None): Seed value; if None, do nothing.
    deterministic (bool): Enable deterministic algorithms where possible.

    Returns:
    int | None: The seed value if set, else None.
    """
    if seed is None:
        return None

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                torch.use_deterministic_algorithms(True)

    return seed


def seed_worker(_worker_id):
    # Use the base seed set by the DataLoader's generator for per-worker RNG.
    """Seed NumPy and random for a DataLoader worker.

    Args:
    _worker_id (int): Worker index (unused).
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
