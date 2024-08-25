# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from ctdg import ops
from ctdg.data import Events
from ctdg.nn import MemoryUpdater, Module, NeighborSampler

from ._embedder import Embedder
from ._message import MessageAggregator, MessageFunction
from ._records import GraphState, RawMessages
from ._rewirer import Rewirer


class TIPAR(Module):
    """The Temporal Information Propagation through Adaptive Rewiring (TIPAR) model."""

    def __init__(
        self,
        num_nodes: int,
        nodes_features: Tensor | None,
        events_features: Tensor | None,
        memory_dim: int,
        memory_updater: MemoryUpdater,
        neighbor_sampler: NeighborSampler,
        src_message_function: MessageFunction,
        dst_message_function: MessageFunction,
        message_aggregator: MessageAggregator,
        embedder: Embedder,
        rewirer: Rewirer,
    ) -> None:
        super().__init__()

        self.graph = GraphState(
            num_nodes=num_nodes,
            nodes_features=nodes_features,
            events_features=events_features,
            memory_dim=memory_dim,
            neighbor_sampler=neighbor_sampler,
        )

        self.memory_updater = memory_updater
        self.src_msg_func = src_message_function
        self.dst_msg_func = dst_message_function
        self.msg_agg = message_aggregator
        self.embedder = embedder
        self.rewirer = rewirer

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def reset_graph_state(self) -> None:
        """Resets the state of the graph."""
        self.graph.reset()

    def detach_graph_state(self) -> None:
        """Detaches the state of the graph."""
        self.graph.detach()

    def change_neighbor_sampler(self, neighbor_sampler: NeighborSampler) -> None:
        """Changes the neighbor sampler of the graph.

        Args:
            neighbor_sampler: The new neighbor sampler.
        """
        self.graph.neighbor_sampler = neighbor_sampler

    def store_events(
        self,
        events: Events,
        src_embeds: Tensor,
        dst_embeds: Tensor,
    ) -> None:
        """Stores the events in the graph state.

        Args:
            events: The events to store.
            src_embeds: The embeddings of the source nodes.
            dst_embeds: The embeddings of the destination nodes.
        """
        src_embeds, dst_embeds = src_embeds.detach(), dst_embeds.detach()

        raw_msg = RawMessages.from_events(events, src_embeds, dst_embeds)
        self.graph.src_store.store(events.src_nodes, raw_msg)

        flipped_events = events.flip_direction()
        raw_msg = RawMessages.from_events(flipped_events, dst_embeds, src_embeds)
        self.graph.dst_store.store(events.dst_nodes, raw_msg)

    def flush_events(self) -> None:
        """Flushes the events from the graph state."""
        idx = torch.arange(self.graph.num_nodes, device=self.graph.device)
        self._update_memory(idx)

    def rewire_graph(self) -> None:
        """Rewires the graph."""
        self.flush_events()

        adj = self.rewirer(self.graph)
        self.graph.adjacency = adj

    def compute_node_embeddings(self, idx: Tensor, t: Tensor) -> Tensor:
        """Computes the embeddings of the nodes at the given timestamps.

        Args:
            idx: The indices of the nodes for which to compute the embeddings.
                This should be a tensor of shape `(N,)`.
            t: The timestamps at which to compute the embeddings. This should
                be a tensor of shape `(N,)`.

        Returns:
            The embeddings of the nodes at the given timestamps. This is a tensor
            of shape `(N, D)`, where `D` is the dimension of the embeddings.
        """
        involved_nodes = [idx]
        neighborhoods = self.embedder.get_neighborhoods(self.graph, idx, t)
        for temp, rewired in neighborhoods:
            for n_idx, _, _, mask in [temp, rewired]:
                involved_nodes.append(n_idx[mask])

        involved_nodes = torch.cat(involved_nodes, dim=0)
        involved_nodes = torch.unique(involved_nodes)
        self._update_memory(involved_nodes)

        return self.embedder(self.graph, idx, t, neighborhoods)

    # ----------------------------------------------------------------------- #
    # Private Methods
    # ----------------------------------------------------------------------- #

    def _update_memory(self, idx: Tensor) -> None:
        """Updates the memory of the given nodes.

        Args:
            idx: The indices of the nodes for which to update the memory.
        """
        # compute messages (src -> dst)
        tmp = self.graph.src_store.retrieve(idx, clear=True)
        if tmp is None:
            return
        _, s_raw_msg = tmp
        s_msg = self.src_msg_func(self.graph, s_raw_msg)
        s_src, s_t = s_raw_msg.src_nodes, s_raw_msg.timestamps

        # compute messages (dst -> src)
        tmp = self.graph.dst_store.retrieve(idx, clear=True)
        if tmp is None:
            return
        _, d_raw_msg = tmp
        d_msg = self.dst_msg_func(self.graph, d_raw_msg)
        d_src, d_t = d_raw_msg.src_nodes, d_raw_msg.timestamps

        # aggregate messages
        src = torch.cat([s_src, d_src], dim=0)
        t = torch.cat([s_t, d_t], dim=0)
        msg = torch.cat([s_msg, d_msg], dim=0)
        if len(src) == 0:
            return

        idx, msg = self.msg_agg(src, t, msg)

        # Compute the time of the last message received for each node
        # that has received a message. This is set as the last update time
        # of the memory of the node.
        t, _ = ops.scatter_max(t, src, dim=0)
        t = t[idx]

        old_memory = self.graph.memory.memory[idx]
        new_memory = self.memory_updater(msg, old_memory)
        self.graph.memory[idx] = new_memory, t
