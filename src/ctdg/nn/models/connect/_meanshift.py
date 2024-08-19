# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor, nn

from ctdg import ops
from ctdg.nn import Module


class MeanShift(Module):
    """Mean shift clustering algorithm."""

    def __init__(
        self,
        sigma: float,
        max_iterations: int = 100,
        shift_threshold: float = 1e-5,
        *,
        learnable: bool = True,
    ) -> None:
        """Initializes the mean shift clustering algorithm.

        Args:
            sigma: The bandwidth of the gaussian kernel.
            max_iterations: The maximum number of iterations to perform.
            shift_threshold: The threshold to stop the iterations. If the shift
                of the centroids is smaller than this value, the algorithm stops.
            learnable: Whether to learn the bandwidth of the gaussian kernel.
        """
        super().__init__()

        self.sigma = nn.Parameter(torch.tensor(float(sigma)), requires_grad=learnable)
        self.max_iterations = max_iterations
        self.shift_threshold = shift_threshold

    def __call__(self, x: Tensor) -> Tensor:
        """Computes the cluster centroids using the mean shift algorithm.

        Args:
            x: The input tensor to cluster. This should be a tensor of shape
                `(N, D)` where `N` is the number of samples and `D` is the
                number of features.

        Returns:
            The cluster centroids. This is a tensor of shape `(M, D)` where `M`
            is the number of clusters and `D` is the number of features.
        """
        centroids = x.clone()  # (M, D)
        inv_gamma = 1 / (2 * self.sigma**2)
        for _ in range(self.max_iterations):
            # the gaussian kernel is defined as: exp(-||x - y||^2 / (2 * sigma^2))
            # if we normalize the inputs x and y, then ||x - y||^2 = 2 * (1 - sim(x, y))
            sim = ops.cosine_similarity(centroids, x)  # (M, N)
            neg_dist = 2 * (sim - 1)  # (M, N)
            weights = torch.exp(neg_dist * inv_gamma)  # (M, N)

            nominator = weights @ x  # (M, D)
            denominator = weights.sum(dim=-1, keepdim=True)  # (M, 1)
            new_centroids = nominator / denominator

            with torch.no_grad():
                shift = new_centroids - centroids
                if torch.norm(shift, dim=-1).max() < self.shift_threshold:
                    break

            centroids = new_centroids

        keep_idx = _nms(x, centroids, self.sigma.item())
        return centroids[keep_idx]


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #


@torch.no_grad()  # type: ignore
def _nms(x: Tensor, centroids: Tensor, bandwidth: float) -> Tensor:
    """Performs non-maximum suppression on the computed cluster centroids.

    Args:
        x: The points to cluster. This should be a tensor of shape `(N, D)` where
            `N` is the number of samples and `D` is the number of features.
        centroids: The computed cluster centroids. This should be a tensor of
            shape `(M, D)` where `M` is the number of clusters and `D` is the
            number of features.
        bandwidth: The bandwidth of the mean shift algorithm.

    Returns:
        The indices of the cluster centroids that are not suppressed.
    """
    # For each cluster, find to how many points it is the closest.
    cp_sim = ops.cosine_similarity(centroids, x)  # (M, N)
    cp_dist = 2 * (1 - cp_sim)  # (M, N)
    p_to_closest_c = cp_dist.argmin(dim=0)  # (N,)
    closest_c, num_points = p_to_closest_c.unique(return_counts=True)
    counts = torch.zeros(centroids.size(0), device=centroids.device, dtype=torch.long)
    counts[closest_c] = num_points

    # Mark as close the clusters whose distance is smaller than the bandwidth.
    cc_sim = ops.cosine_similarity(centroids, centroids)  # (M, M)
    cc_dist = 2 * (1 - cc_sim)
    cc_close = (cc_dist < bandwidth).float()  # (M, M)

    # For each non empty cluster (a cluster that is the closest to at least one point)
    # find the larger cluster whose distance is smaller than the bandwidth.
    # Such clusters will be the ones that are not suppressed.
    c_to_close_larger_c = torch.argmax(cc_close[closest_c] * counts[None], dim=1)
    return torch.unique(c_to_close_larger_c)
