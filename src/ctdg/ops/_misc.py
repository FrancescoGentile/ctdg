# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

from torch import Tensor


def cosine_similarity(x: Tensor, y: Tensor | None = None) -> Tensor:
    """Computes the pairwise cosine similarity between two tensors.

    Args:
        x: The first tensor. This should be a tensor of shape `(N, D)`.
        y: The second tensor. This should be a tensor of shape `(M, D)`.
            If `None`, `y` is set to `x`.

    Returns:
        A tensor of shape `(N, M)` containing the cosine similarity between
        each pair of vectors in `x` and `y`.
    """
    if y is None:
        y = x

    x_norm = x / x.norm(dim=1, keepdim=True)
    y_norm = y / y.norm(dim=1, keepdim=True)

    return x_norm @ y_norm.t()


def normalize(x: Tensor, dim: int = -1) -> Tensor:
    """Normalizes a tensor to have unit norm along a given dimension.

    Args:
        x: The tensor to normalize.
        dim: The dimension along which to normalize the tensor.

    Returns:
        The normalized tensor.
    """
    return x / x.norm(dim=dim, keepdim=True)
