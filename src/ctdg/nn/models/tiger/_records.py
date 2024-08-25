# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor
from typing_extensions import Self

from ctdg.data import Events, Stream
from ctdg.nn import Memory, Module, NeighborSampler, StreamStore


class RawMessages(Stream):
    """Raw messages."""

    src_embeds: Tensor
    dst_embeds: Tensor
    src_nodes: Tensor
    dst_nodes: Tensor
    timestamps: Tensor
    indices: Tensor

    @classmethod
    def from_events(cls, events: Events, s_embeds: Tensor, d_embeds: Tensor) -> Self:
        if len(events) != len(s_embeds) or len(events) != len(d_embeds):
            msg = "The number of events must match the number of embeddings."
            raise ValueError(msg)

        return cls(
            src_embeds=s_embeds,
            dst_embeds=d_embeds,
            src_nodes=events.src_nodes,
            dst_nodes=events.dst_nodes,
            timestamps=events.timestamps,
            indices=events.indices,
        )


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
        """Initialize the graph state."""
        super().__init__()

        self.num_nodes = num_nodes

        self.register_buffer("nodes_features", None)
        self.nodes_features = nodes_features

        self.register_buffer("events_features", None)
        self.events_features = events_features

        self.memory = Memory(num_nodes, memory_dim)

        self.neighbor_sampler = neighbor_sampler

        self.src_store = StreamStore[RawMessages](num_nodes)
        self.dst_store = StreamStore[RawMessages](num_nodes)

    # ----------------------------------------------------------------------- #
    # Properties
    # ----------------------------------------------------------------------- #

    @property
    def device(self) -> torch.device:
        """Returns the device of the graph state."""
        return self.memory.memory.device

    # ----------------------------------------------------------------------- #
    # Public methods
    # ----------------------------------------------------------------------- #

    def reset(self) -> None:
        """Resets the graph to its initial state."""
        self.memory.reset()
        self.src_store.clear()
        self.dst_store.clear()

    def detach(self) -> None:
        """Detaches the graph state from the computational graph."""
        self.memory.detach()
