# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

"""Utility functions."""

from ._config import load_config
from ._lazy import LazyCall
from ._misc import (
    check_precision,
    get_scheduler_hyperparameters,
    seed_all,
)

__all__ = [
    # _config
    "load_config",
    # _lazy
    "LazyCall",
    # _misc
    "check_precision",
    "get_scheduler_hyperparameters",
    "seed_all",
]
