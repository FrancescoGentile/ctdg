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
        """Computes the new adjacency matrix of the network.

        Args:
            state: The current state of the network.

        Returns:
            The new adjacency matrix of the network. This should be a tensor of shape
            `(N, N)`, where `N` is the number of nodes in the network.
        """
        ...


# --------------------------------------------------------------------------- #
# Implementations
# --------------------------------------------------------------------------- #


class HDBSCANRewirer(Rewirer):
    """A rewiring module that uses the HDBSCAN clustering algorithm.

    This module uses the soft-HDBSCAN clustering algorithm to identify the communities
    in the network. The resulting community assignments are then used to compute the
    strength of the connections between the nodes as the dot product of their membership
    vectors. Finally, the strength matrix is sparsified using the sparsemax function to
    obtain the adjacency matrix of the network.
    """

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
        features = state.memory.memory
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

        adj = boundary_matrix @ boundary_matrix.t()  # (N', N')
        scores, indices = adj.topk(self._num_neighbors, dim=1)
        adj.fill_(0)
        adj.scatter_(1, indices, scores)

        return adj


class SimilarityRewirer(Rewirer):
    """A rewiring module that computes the similarity between the nodes.

    This module first computes the strength of the connections between the nodes as the
    cosine similarity between their features. The resulting similarity matrix is then
    sparsified using the sparsemax function to obtain the adjacency matrix of the
    network.
    """

    def __call__(self, state: GraphState) -> Tensor:
        features = state.memory.memory + state.nodes_features
        adj = ops.cosine_similarity(features)
        adj = ops.sparsemax(adj, dim=1)

        # set the connections between the nodes that have not been observed to 0
        mask = state.nodes_mask
        idx = torch.nonzero(~mask).squeeze()
        adj[idx, idx[:, None]] = 0
        adj.fill_diagonal_(0)

        return adj
