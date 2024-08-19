# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

import entmax
from torch import Tensor


def entmax15(x: Tensor, dim: int) -> Tensor:
    return entmax.entmax15(x, dim=dim)  # type: ignore


def sparsemax(x: Tensor, dim: int) -> Tensor:
    return entmax.sparsemax(x, dim=dim)  # type: ignore
