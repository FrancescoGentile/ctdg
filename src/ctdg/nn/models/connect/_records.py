# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

import enum

import torch
from torch import Tensor
from typing_extensions import Self

from ctdg.data import Events, Stream
from ctdg.nn import Memory, Module, StreamStore


class Rank(enum.Enum):
    NODES = 0
    COMMUNITIES = 1


class RawMessages(Stream):
    """A record of raw messages."""

    src_embeds: Tensor
    dst_embeds: Tensor
    src_nodes: Tensor
    dst_nodes: Tensor
    timestamps: Tensor
    indices: Tensor

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
    """The state of the graph."""

    def __init__(
        self,
        num_nodes: int,
        nodes_features: Tensor | None,
        events_features: Tensor | None,
        memory_dim: int,
    ) -> None:
        super().__init__()

        self.num_nodes = num_nodes

        self.register_buffer("nodes_features", None)
        self.nodes_features = nodes_features

        self.register_buffer("events_features", None)
        self.events_features = events_features

        self.register_buffer("incidence_matrix", None)
        self.incidence_matrix = torch.zeros(num_nodes, 0)

        self.nodes_memory = Memory(num_nodes, memory_dim)
        self.comms_memory = Memory(0, memory_dim)

        self.src_store = StreamStore[RawMessages](num_nodes)
        self.dst_store = StreamStore[RawMessages](num_nodes)

        self.comms_store = StreamStore[RawMessages](0)

    # ----------------------------------------------------------------------- #
    # Properties
    # ----------------------------------------------------------------------- #

    @property
    def num_communities(self) -> int:
        """Returns the number of communities."""
        return self.incidence_matrix.size(1)

    @property
    def device(self) -> torch.device:
        """Returns the device of the model."""
        return self.nodes_memory.memory.device

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
        return self.nodes_memory.last_update >= 0

    # ----------------------------------------------------------------------- #
    # Public methods
    # ----------------------------------------------------------------------- #

    def update_communities(
        self,
        incidence: Tensor,
        memory: Tensor,
        last_update: Tensor,
    ) -> None:
        """Updates the communities in the graph state."""
        num_comms = incidence.size(1)
        self.incidence_matrix = incidence

        self.comms_memory = Memory(num_comms, self.comms_memory.dim, memory.device)
        self.comms_memory.memory = memory.float()
        self.comms_memory.last_update = last_update.float()

        self.comms_store = StreamStore[RawMessages](num_comms)

    def reset(self) -> None:
        """Resets the graph to its initial state."""
        self.nodes_memory.reset()
        self.comms_memory = Memory(0, self.comms_memory.dim, device=self.device)

        self.incidence_matrix = torch.zeros(self.num_nodes, 0, device=self.device)

        self.src_store.clear()
        self.dst_store.clear()
        self.comms_store = StreamStore[RawMessages](0)

    def detach(self) -> None:
        """Detaches the graph from the computational graph."""
        self.nodes_memory.detach()
        self.comms_memory.detach()
        self.incidence_matrix.detach_()
