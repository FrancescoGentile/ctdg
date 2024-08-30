# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, Protocol

import torch
from torch import Tensor

from ctdg import ops

from ._records import GraphState


class Rewirer(Protocol):
    """Interface for rewiring modules."""

    def __call__(self, state: GraphState) -> Tensor:
        """Computes the new incidence matrix of the network.

        Args:
            state: The current state of the network.

        Returns:
            The new incidence matrix of the network. This should be a tensor of shape
            `(N, C)` where `N` is the number of nodes and `C` is the number of
            communities.
        """
        ...


# --------------------------------------------------------------------------- #
# Implementations
# --------------------------------------------------------------------------- #


class HDBSCANRewirer(Rewirer):
    """A rewiring module that uses the HDBSCAN clustering algorithm."""

    def __init__(
        self,
        num_neighbors: int,
        metric: Literal["euclidean", "l2", "cosine"] = "euclidean",
        min_cluster_size: int = 5,
        min_samples: int | None = None,
    ) -> None:
        """Initializes the HDBSCAN rewirer.

        Args:
            num_neighbors: The maximum number of neighbors to set in the adjacency
                matrix.
            metric: The metric to use to compute the distance between the nodes.
            min_cluster_size: The minimum number of nodes that a cluster should have.
            min_samples: The number of samples in a neighborhood for a point to be
                considered as a core point (this includes the point itself). If `None`,
                it is set to `min_cluster_size`.
        """
        # import cuml inside the constructor instead of at the top level
        # to avoid importing cuml when this rewirer is not used (installing cuml
        # is heavy, thus users should not be forced to install it if they don't
        # use this rewirer)
        from cuml.cluster import hdbscan

        self._num_neighbors = num_neighbors
        self._metric = metric
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric if metric != "cosine" else "euclidean",
            prediction_data=True,
        )

    def __call__(self, state: GraphState) -> Tensor:
        import cupy as cp
        from cuml.cluster import hdbscan

        mask = state.nodes_mask  # the mask of observed nodes
        features = state.nodes_memory.memory
        if state.nodes_features is not None:
            features = features + state.nodes_features
        features = features[mask]
        if features.size(0) == 0:
            return torch.zeros(state.num_nodes, state.num_nodes, device=features.device)

        if self._metric == "cosine":
            features = ops.normalize(features, dim=-1)

        self.clusterer.fit(cp.asarray(features.detach()))
        soft_clusters = hdbscan.all_points_membership_vectors(self.clusterer)  # (N', C)
        if soft_clusters.ndim == 1:
            # no clusters have been found
            return torch.zeros(state.num_nodes, state.num_nodes, device=features.device)

        soft_clusters = torch.as_tensor(soft_clusters, device=features.device)
        N, C = state.num_nodes, soft_clusters.size(1)  # noqa: N806
        boundary_matrix = torch.zeros(
            N, C, dtype=soft_clusters.dtype, device=soft_clusters.device
        )
        boundary_matrix[mask] = soft_clusters

        if self._num_neighbors < C:
            scores, indices = boundary_matrix.topk(self._num_neighbors, dim=1)
            boundary_matrix.fill_(0)
            boundary_matrix.scatter_(1, indices, scores)

        return boundary_matrix
