"""Seed utilities for reproducibility."""

import random
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"Set random seed to {seed}")


def get_generator(seed: int = 42) -> torch.Generator:
    """
    Get a torch Generator with the specified seed.

    Args:
        seed: The seed value to use.

    Returns:
        A torch.Generator initialized with the seed.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g
