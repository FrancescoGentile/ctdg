# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

"""Utility functions."""

from ._config import load_config
from ._lazy import LazyCall
from ._lightning import configure_optimizers, get_scheduler_hyperparameters
from ._misc import check_precision, seed_all

__all__ = [
    # _config
    "load_config",
    # _lazy
    "LazyCall",
    # _lightning
    "configure_optimizers",
    "get_scheduler_hyperparameters",
    # _misc
    "check_precision",
    "seed_all",
]
