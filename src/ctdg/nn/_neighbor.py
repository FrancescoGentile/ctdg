# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

from typing import Protocol, TypeAlias

import torch
from torch import Tensor
from typing_extensions import override

from ctdg.data import Events

Neighborhood: TypeAlias = tuple[Tensor, Tensor, Tensor, Tensor]
"""A neighborhood of a node at a given time.

This is a tuple of tensors with shape `(N, n_neighbors)` where `N` is the number of
nodes for which the neighborhood was sampled and `n_neighbors` is the number of
neighbors to sample. The tuple contains the indices of the neighbors, the timestamps
of the interactions with the neighbors, the indices of the interactions with the
neighbors, and a binary mask indicating which neighbors are valid (not all nodes may
have `n_neighbors` neighbors at the given time).
"""


class NeighborSampler(Protocol):
    """Interface for neighbor samplers.

    A neighbor sampler is responsible for sampling from the temporal neigborhood of a
    node at a given time.
    """

    def sample(self, idx: Tensor, t: Tensor, n_neighbors: int) -> Neighborhood:
        """Samples the temporal neighborhood of the given nodes up to the given time.

        Args:
            idx: The indices of the nodes for which to sample the neighborhood.
                This is a tensor of shape `(N,)` where `N` is the number of nodes
                for which to sample the neighborhood.
            t: The time up to which to sample the neighborhood. This should be a tensor
                with the same shape as `idx`.
            n_neighbors: The number of neighbors to sample.

        Returns:
            The sampled neighborhood of the nodes up to the given time.
        """
        ...

    def last_interaction(
        self, i: Tensor, j: Tensor, t: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Finds the last interaction between two nodes up to the given time.

        Args:
            i: The indices of the first nodes.
            j: The indices of the second nodes.
            t: The time up to which to search for the last interaction.

        Returns:
            The timestamps and indices of the last interactions between the nodes. If
            no interaction is found, the timestamp is set to `-1.0` and the index is set
            to `-1`.
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

    @override
    def sample(self, idx: Tensor, t: Tensor, n_neighbors: int) -> Neighborhood:
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

    @override
    def last_interaction(
        self, i: Tensor, j: Tensor, t: Tensor
    ) -> tuple[Tensor, Tensor]:
        timestamps = torch.full_like(i, -1.0)
        indices = torch.full_like(i, -1)

        for k, (src, dst, time) in enumerate(
            zip(i.tolist(), j.tolist(), t.tolist(), strict=True)
        ):
            n, n_t, n_i = self.adj_list[src]

            mask = (n == dst) & (n_t < time)
            interactions = torch.nonzero(mask, as_tuple=True)[0]
            if interactions.numel() > 0:
                last = interactions[-1]
                timestamps[k] = n_t[last]
                indices[k] = n_i[last]

        return timestamps.to(i.device, non_blocking=True), indices.to(
            i.device, non_blocking=True
        )
