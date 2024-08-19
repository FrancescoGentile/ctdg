# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0


from typing import Annotated, Literal, overload

import torch
from torch import Tensor

from ._scatter import scatter


@overload
def coalesce(
    indices: Annotated[Tensor, "I N", int],
    values: Annotated[Tensor, "N ..."],
    *,
    reduce: Literal["sum", "mul", "mean", "min", "max"] = "sum",
    is_sorted: bool = False,
) -> tuple[Annotated[Tensor, "I M", int], Annotated[Tensor, "M ..."]]: ...


@overload
def coalesce(
    indices: Annotated[Tensor, "I N", int],
    values: None = None,
    *,
    reduce: Literal["sum", "mul", "mean", "min", "max"] = "sum",
    is_sorted: bool = False,
) -> Annotated[Tensor, "I M", int]: ...


def coalesce(
    indices: Annotated[Tensor, "I N", int],
    values: Annotated[Tensor, "N ..."] | None = None,
    *,
    reduce: Literal["sum", "mul", "mean", "min", "max"] = "sum",
    is_sorted: bool = False,
) -> (
    Annotated[Tensor, "I M", int]
    | tuple[Annotated[Tensor, "I M", int], Annotated[Tensor, "M ..."]]
):
    """Coalesces the indices and values (if given) of a sparse tensor.

    Args:
        indices: The indices of the non-zero values in the original dense tensor. This
            must be a two-dimensional tensor of shape `(I, N)` where `I` is the number
            of sparse dimensions and `N` is the number of non-zero values.
        values: The non-zero values in the original dense tensor. This must be a
            tensor of shape `(N, ...)`. Note that `...` denotes an arbitrary number of
            dense dimensions (i.e. they do not contribute to the sparsity).
        reduce: The reduce operation to use for coalescing.
        is_sorted: If set to `True`, the indices will be assumed to be sorted.

    Returns:
        If `values` is `None`, returns the coalesced indices of the sparse tensor.
        Otherwise, returns a tuple containing the coalesced indices and values of the
        sparse tensor.
    """
    match (indices, values, reduce):
        case (indices, None, _):
            return indices.unique(sorted=is_sorted, dim=1)
        case (indices, values, "sum"):
            # Using torch.sparse_coo_tensor.coalesce() is 2x faster than removing
            # duplicates and then using scatter. However, it can only be used for
            # sum reduction.
            tmp = torch.sparse_coo_tensor(indices, values)
            tmp = tmp.coalesce()
            return tmp.indices(), tmp.values()
        case (_, _, _):
            indices, pos = indices.unique(sorted=is_sorted, return_inverse=True, dim=1)
            values = scatter(values, pos, dim=0, reduce=reduce)
            return indices, values
