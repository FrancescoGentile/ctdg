# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

from torch import Tensor


def cosine_similarity(x: Tensor, y: Tensor) -> Tensor:
    """Computes the pairwise cosine similarity between two tensors.

    Args:
        x: The first tensor. This should be a tensor of shape `(N, D)`.
        y: The second tensor. This should be a tensor of shape `(M, D)`.

    Returns:
        A tensor of shape `(N, M)` containing the cosine similarity between
        each pair of vectors in `x` and `y`.
    """
    x_norm = x / x.norm(dim=1, keepdim=True)
    y_norm = y / y.norm(dim=1, keepdim=True)

    return x_norm @ y_norm.t()
