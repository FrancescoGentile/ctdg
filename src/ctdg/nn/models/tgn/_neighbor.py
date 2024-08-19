# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

from typing import Protocol

import torch
from torch import Tensor

from ctdg.structures import Events


class NeighborSampler(Protocol):
    """Interface for neighbor samplers.

    A neighbor sampler is responsible for sampling from the temporal neigborhood of a
    node at a given time.
    """

    def sample_before(
        self,
        idx: Tensor,
        t: Tensor,
        n_neighbors: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Samples the temporal neighborhood of the given nodes up to the given time.

        Args:
            idx: The indices of the nodes for which to sample the neighborhood.
                This is a tensor of shape `(N,)` where `N` is the number of nodes
                for which to sample the neighborhood.
            t: The time up to which to sample the neighborhood. This should be a tensor
                with the same shape as `idx`.
            n_neighbors: The number of neighbors to sample.

        Returns:
            A tuple of tensors with shape `(N, n_neighbors)` containing the indices of
            the neighbors, the timestamps of the interactions with the neighbors, the
            indices of the interactions with the neighbors, and a binary mask indicating
            which neighbors are valid (not all nodes may have `n_neighbors` neighbors
            at the given time).
        """
        ...


class LastNeighborSampler(NeighborSampler):
    """A neighbor sampler that samples the last nodes a node interacted with."""

    def __init__(self, events: Events, num_nodes: int) -> None:
        """Initializes the sampler.

        Args:
            events: The events to sample from.
            num_nodes: The number of nodes in the graph.
        """

        def get_adjacency(i: int) -> tuple[Tensor, Tensor, Tensor]:
            s_mask = events.src_nodes == i
            s_nodes = events.dst_nodes[s_mask]
            s_t = events.timestamps[s_mask]
            s_idx = events.indices[s_mask]

            d_mask = events.dst_nodes == i
            d_nodes = events.src_nodes[d_mask]
            d_t = events.timestamps[d_mask]
            d_idx = events.indices[d_mask]

            nodes = torch.cat([s_nodes, d_nodes])
            t = torch.cat([s_t, d_t])
            idx = torch.cat([s_idx, d_idx])

            order = t.argsort()
            return nodes[order], t[order], idx[order]

        self.adj_list = [get_adjacency(i) for i in range(num_nodes)]

    def sample_before(
        self,
        idx: Tensor,
        t: Tensor,
        n_neighbors: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        neighbors = torch.zeros(len(idx), n_neighbors, dtype=torch.long)
        timestamps = torch.zeros(len(idx), n_neighbors, dtype=t.dtype)
        e_idx = torch.zeros(len(idx), n_neighbors, dtype=torch.long)
        mask = torch.zeros(len(idx), n_neighbors, dtype=torch.bool)

        for i, (src, time) in enumerate(zip(idx.tolist(), t.tolist(), strict=True)):
            n, n_t, n_i = self.adj_list[src]

            pos = torch.searchsorted(n_t, time, right=False).item()
            num_neighbors = min(n_neighbors, pos)

            neighbors[i, :num_neighbors] = n[pos - num_neighbors : pos]
            timestamps[i, :num_neighbors] = n_t[pos - num_neighbors : pos]
            e_idx[i, :num_neighbors] = n_i[pos - num_neighbors : pos]
            mask[i, :num_neighbors] = True

        return (
            neighbors.to(idx.device, non_blocking=True),
            timestamps.to(t.device, non_blocking=True),
            e_idx.to(idx.device, non_blocking=True),
            mask.to(idx.device, non_blocking=True),
        )
