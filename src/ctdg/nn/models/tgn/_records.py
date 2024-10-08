# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from ctdg.data import Events
from ctdg.nn import Memory, Module, NeighborSampler, StreamStore


class GraphState(Module):
    """The state of the graph."""

    def __init__(
        self,
        num_nodes: int,
        nodes_features: Tensor | None,
        events_features: Tensor | None,
        memory_dim: int,
        neighbor_sampler: NeighborSampler,
    ) -> None:
        super().__init__()

        self.num_nodes = num_nodes

        self.register_buffer("nodes_features", None)
        self.nodes_features = nodes_features

        self.register_buffer("events_features", None)
        self.events_features = events_features

        self.memory = Memory(num_nodes, memory_dim)

        self.neighbor_sampler = neighbor_sampler

        self.src_store = StreamStore[Events](num_nodes)
        self.dst_store = StreamStore[Events](num_nodes)

    # ----------------------------------------------------------------------- #
    # Properties
    # ----------------------------------------------------------------------- #

    @property
    def device(self) -> torch.device:
        """The device on which the state is stored."""
        return self.memory.memory.device

    # ----------------------------------------------------------------------- #
    # Methods
    # ----------------------------------------------------------------------- #

    def reset(self) -> None:
        """Resets the state of the graph to its initial state."""
        self.memory.reset()
        self.src_store.clear()
        self.dst_store.clear()

    def detach(self) -> None:
        """Detaches the state of the graph from the computation graph."""
        self.memory.detach()
