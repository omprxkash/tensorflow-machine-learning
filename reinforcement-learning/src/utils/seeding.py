"""Utilities for setting global random seeds across numpy, Python, and torch."""

import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set all relevant RNG seeds for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
