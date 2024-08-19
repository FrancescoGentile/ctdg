# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor, nn

from ctdg import ops
from ctdg.nn import MLP, EventStore, Memory, MemoryUpdater, Module, StaticInfo
from ctdg.structures import Events

from ._embedder import Embedder
from ._message import MessageAggregator, MessageFunction
from ._neighbor import NeighborSampler
from ._records import GraphState


class TGN(Module):
    """The Temporal Graph Network (TGN) model."""

    def __init__(
        self,
        static_info: StaticInfo,
        memory_dim: int,
        memory_updater: MemoryUpdater,
        s_message_function: MessageFunction,
        d_message_function: MessageFunction,
        message_aggregator: MessageAggregator,
        embedder: Embedder,
        neighbor_sampler: NeighborSampler,
    ) -> None:
        super().__init__()

        self.static_info = static_info

        self.memory = Memory(static_info.num_nodes, memory_dim)
        self.memory_updater = memory_updater

        self.embedder = embedder
        self.s_message_function = s_message_function
        self.d_message_function = d_message_function
        self.message_aggregator = message_aggregator
        self.neighbor_sampler = neighbor_sampler

        self.link_predictor = MLP(
            input_dim=2 * embedder.output_dim,
            hidden_dim=embedder.output_dim,
            output_dim=1,
            num_layers=2,
            activation=nn.ReLU,
            dropout=0.0,
        )

        self.s_store = EventStore(static_info.num_nodes, conflict_mode="error")
        self.d_store = EventStore(static_info.num_nodes, conflict_mode="error")

    # ----------------------------------------------------------------------- #
    # Properties
    # ----------------------------------------------------------------------- #

    @property
    def graph_state(self) -> GraphState:
        """The state of the temporal graph."""
        return GraphState(
            static_info=self.static_info,
            memory=self.memory,
            neighbor_sampler=self.neighbor_sampler,
        )

    # ----------------------------------------------------------------------- #
    # Public methods
    # ----------------------------------------------------------------------- #

    @torch.no_grad()  # type: ignore
    def reset_graph_state(self) -> None:
        """Resets the state of the temporal graph.

        This method should be called at the beginning of each epoch to reset the state
        of the graph as if no events have happened yet.
        """
        self.memory.reset()
        self.s_store.clear()
        self.d_store.clear()

    def store_events(self, events: Events) -> None:
        """Stores the events in the event stores.

        !!! note

            This method does not update the state of the nodes. To do so, call
            `flush_events` after storing the events.
        """
        self.s_store.store(events.src_nodes, events)
        self.d_store.store(events.dst_nodes, events.flip_direction())

    def flush_events(self) -> None:
        """Updates the state of the nodes with the stored events."""
        device = self.memory.memory.device
        idx = torch.arange(self.static_info.num_nodes, device=device)
        self._update_memory(idx)

    def compute_events_probabilities(self, events: Events) -> Tensor:
        """Computes the probabilities of events happening.

        Args:
            events: The events for which to compute the probabilities.

        Returns:
            The log-probabilities of the events happening.
        """
        idx = torch.cat([events.src_nodes, events.dst_nodes])  # (2E,)
        t = events.timestamps.repeat(2)  # (2E,)

        # invoved nodes may have incoming messages that have not been processed yet
        # so we need to update the memory of all involved nodes
        involved_idx = torch.unique(idx)  # (N,)
        self._update_memory(involved_idx)

        embeds = self.embedder(self.graph_state, idx, t)
        src_embeds, dst_embeds = torch.chunk(embeds, 2, dim=0)
        embeds = torch.cat([src_embeds, dst_embeds], dim=1)

        return self.link_predictor(embeds).squeeze(-1)

    # ----------------------------------------------------------------------- #
    # Private methods
    # ----------------------------------------------------------------------- #

    def _update_memory(self, idx: Tensor) -> None:
        """Updates the memory of the given nodes."""
        # compute messages (src -> dst)
        _, s_events = self.s_store.retrieve(idx, clear=True)
        s_msg = self.s_message_function(self.graph_state, s_events)
        s_src, _, s_t, _ = s_events

        # compute messages (dst -> src)
        _, d_events = self.d_store.retrieve(idx, clear=True)
        d_msg = self.d_message_function(self.graph_state, d_events)
        d_src, _, d_t, _ = d_events

        # aggregate messages
        src = torch.cat([s_src, d_src], dim=0)
        t = torch.cat([s_t, d_t], dim=0)
        msg = torch.cat([s_msg, d_msg], dim=0)
        if len(src) == 0:
            return

        agg_msg, nodes = self.message_aggregator(msg, src, t)

        # Compute the time of the last message received for each node
        # that has received a message. This is set as the last update time
        # of the memory of the node.
        t, _ = ops.scatter_max(t, src, dim=0)
        t = t[nodes]

        old_memory, _ = self.memory[nodes]
        new_memory = self.memory_updater(agg_msg, old_memory)
        self.memory[nodes] = new_memory, t
