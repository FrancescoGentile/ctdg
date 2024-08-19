# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

import torch_scatter
from torch import Tensor


def scatter_sum(
    src: Tensor,
    index: Tensor,
    dim: int,
    output_dim_size: int | None = None,
) -> Tensor:
    """Scatter sum operation."""
    return torch_scatter.scatter_add(src, index, dim, dim_size=output_dim_size)


def scatter_mul(
    src: Tensor,
    index: Tensor,
    dim: int,
    output_dim_size: int | None = None,
) -> Tensor:
    """Scatter multiply operation."""
    return torch_scatter.scatter_mul(src, index, dim, dim_size=output_dim_size)


def scatter_mean(
    src: Tensor,
    index: Tensor,
    dim: int,
    output_dim_size: int | None = None,
) -> Tensor:
    """Scatter mean operation."""
    return torch_scatter.scatter_mean(src, index, dim, dim_size=output_dim_size)


def scatter_min(
    src: Tensor,
    index: Tensor,
    dim: int,
    output_dim_size: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Scatter min operation."""
    return torch_scatter.scatter_min(src, index, dim, dim_size=output_dim_size)


def scatter_max(
    src: Tensor,
    index: Tensor,
    dim: int,
    output_dim_size: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Scatter max operation."""
    return torch_scatter.scatter_max(src, index, dim, dim_size=output_dim_size)


def scatter_argmin(
    src: Tensor,
    index: Tensor,
    dim: int,
    output_dim_size: int | None = None,
) -> Tensor:
    """Scatter argmin operation."""
    return torch_scatter.scatter_min(src, index, dim, dim_size=output_dim_size)[1]


def scatter_argmax(
    src: Tensor,
    index: Tensor,
    dim: int,
    output_dim_size: int | None = None,
) -> Tensor:
    """Scatter argmax operation."""
    return torch_scatter.scatter_max(src, index, dim, dim_size=output_dim_size)[1]


def scatter(
    src: Tensor,
    index: Tensor,
    dim: int,
    output_dim_size: int | None = None,
    reduce: Literal["sum", "mul", "mean", "min", "max"] = "sum",
) -> Tensor:
    """Scatter operation."""
    match reduce:
        case "sum":
            return scatter_sum(src, index, dim, output_dim_size)
        case "mul":
            return scatter_mul(src, index, dim, output_dim_size)
        case "mean":
            return scatter_mean(src, index, dim, output_dim_size)
        case "min":
            return scatter_min(src, index, dim, output_dim_size)[0]
        case "max":
            return scatter_max(src, index, dim, output_dim_size)[0]


def scatter_softmax(
    src: Tensor,
    index: Tensor,
    dim: int,
    output_dim_size: int | None = None,
) -> Tensor:
    """Scatter softmax operation."""
    return torch_scatter.scatter_softmax(src, index, dim, dim_size=output_dim_size)
