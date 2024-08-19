# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

"""Operations for coalescing and scattering sparse tensors."""

from ._coalesce import coalesce
from ._entmax import entmax15, sparsemax
from ._misc import cosine_similarity
from ._scatter import (
    scatter,
    scatter_argmax,
    scatter_argmin,
    scatter_max,
    scatter_mean,
    scatter_min,
    scatter_mul,
    scatter_softmax,
    scatter_sum,
)

__all__ = [
    # _coalesce
    "coalesce",
    # _entmax
    "entmax15",
    "sparsemax",
    # _misc
    "cosine_similarity",
    # _scatter
    "scatter",
    "scatter_argmax",
    "scatter_argmin",
    "scatter_max",
    "scatter_mean",
    "scatter_min",
    "scatter_mul",
    "scatter_softmax",
    "scatter_sum",
]
