# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

import random
import warnings
from typing import Any

import numpy as np
import torch
from torch.optim.lr_scheduler import LRScheduler

from ._lazy import LazyCall


def seed_all(seed: int) -> None:
    """Seeds all random number generators.

    This function seeds the random number generators of the following libraries:
    - `random`
    - `numpy`
    - `torch` (both CPU and CUDA random number generators)

    Args:
        seed: The seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def check_precision(precision: int | str) -> int | str:
    """Checks if the specified precision is supported by the current device.

    Args:
        precision: The precision to check. The accepted values are those supported by
            the argument precision of `lightning.Trainer`.

    Returns:
        If the precision is supported, it is returned as is. Otherwise, the function
        returns the next best supported precision.
    """
    if (
        isinstance(precision, str)
        and precision.startswith("bf16")
        and not torch.cuda.is_bf16_supported()
    ):
        msg = (
            "Mixed precision with bfloat16 is not supported on this device. "
            "Falling back to mixed precision with float16."
        )
        warnings.warn(msg, UserWarning, stacklevel=1)
        precision = precision.replace("bf16", "16")

    return precision


def get_scheduler_hyperparameters(
    cfg: LazyCall[LRScheduler] | dict[str, Any] | None,
) -> None | dict[str, Any]:
    """Gets the hyperparameters of the scheduler from the configuration."""
    if cfg is None:
        return None
    if isinstance(cfg, LazyCall):
        return cfg.to_dict()

    cfg = cfg.copy()
    scheduler = cfg.pop("scheduler")
    return {"scheduler": scheduler.to_dict(), **cfg}
