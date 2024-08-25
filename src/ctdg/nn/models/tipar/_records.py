# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor
from typing_extensions import Self

from ctdg.data import Events, Stream
from ctdg.nn import Memory, Module, NeighborSampler, StreamStore


class RawMessages(Stream):
    """Raw messages to be stored in the message store."""

    src_embeds: Tensor
    dst_embeds: Tensor
    src_nodes: Tensor
    dst_nodes: Tensor
    timestamps: Tensor
    indices: Tensor

    def __post_init__(self) -> None:
        """Checks that the tensors have the same number of elements."""
        if (
            len(self.src_embeds)
            != len(self.dst_embeds)
            != len(self.timestamps)
            != len(self.indices)
        ):
            msg = "All tensors must have the same number of elements."
            raise ValueError(msg)

    @classmethod
    def from_events(cls, events: Events, s_embeds: Tensor, d_embeds: Tensor) -> Self:
        """Creates a new instance from the given events and embeddings."""
        return cls(
            src_embeds=s_embeds,
            dst_embeds=d_embeds,
            src_nodes=events.src_nodes,
            dst_nodes=events.dst_nodes,
            timestamps=events.timestamps,
            indices=events.indices,
        )


class GraphState(Module):
    """The state of the network."""

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

        self.register_buffer("adjacency", None)
        self.adjacency = torch.zeros(num_nodes, num_nodes)

        self.neighbor_sampler = neighbor_sampler

        self.src_store = StreamStore[RawMessages](num_nodes)
        self.dst_store = StreamStore[RawMessages](num_nodes)

    # ----------------------------------------------------------------------- #
    # Properties
    # ----------------------------------------------------------------------- #

    @property
    def nodes_mask(self) -> Tensor:
        """Returns the mask of the nodes that have been observed.

        For efficiency reasons, during training all nodes that will be observed
        (even at validation and test time) are stored in the nodes features and
        in the memory. However, nodes that have yet to be observed should not
        be taken into account when performing computations.

        This property returns a mask that can be used to filter out these nodes.
        It is a boolean tensor of shape `(num_nodes,)` where `True` indicates that
        the node has been observed and `False` indicates that the node has yet to
        be observed.
        """
        return self.memory.last_update >= 0

    @property
    def device(self) -> torch.device:
        """The device on which the graph is stored."""
        return self.memory.memory.device

    # ----------------------------------------------------------------------- #
    # Methods
    # ----------------------------------------------------------------------- #

    def reset(self) -> None:
        """Resets the graph to its initial state."""
        self.memory.reset()
        self.src_store.clear()
        self.dst_store.clear()
        self.adjacency.fill_(0)

    def detach(self) -> None:
        """Detaches the graph from the computational graph."""
        self.memory.detach()
        self.adjacency = self.adjacency.detach()
