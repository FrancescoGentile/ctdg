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


class TIGER(Module):
    """The Temporal Interaction Graph Embedding with Restarts (TIGER) model."""

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
        self.msg_aggregator = message_aggregator
        self.embedder = embedder

    # ----------------------------------------------------------------------- #
    # Public methods
    # ----------------------------------------------------------------------- #

    def reset_graph_state(self) -> None:
        """Resets the graph state."""
        self.graph.reset()

    def detach_graph_state(self) -> None:
        """Detaches the graph state from the computational graph."""
        self.graph.detach()

    def change_neighbor_sampler(self, neighbor_sampler: NeighborSampler) -> None:
        """Changes the neighbor sampler used by the model."""
        self.graph.neighbor_sampler = neighbor_sampler

    def store_events(self, events: Events, s_embeds: Tensor, d_embeds: Tensor) -> None:
        """Stores the events in the graph state."""
        s_embeds, d_embeds = s_embeds.detach(), d_embeds.detach()

        raw_msg = RawMessages.from_events(events, s_embeds, d_embeds)
        self.graph.src_store.store(raw_msg.src_nodes, raw_msg)

        flip_events = events.flip_direction()
        raw_msg = RawMessages.from_events(flip_events, d_embeds, s_embeds)
        self.graph.dst_store.store(raw_msg.src_nodes, raw_msg)

    def flush_events(self) -> None:
        """Flushes the stored events."""
        idx = torch.arange(self.graph.num_nodes, device=self.graph.device)
        self._update_memory(idx)

    def compute_node_embeddings(self, idx: Tensor, t: Tensor) -> Tensor:
        """Computes the node embeddings for the given nodes and time."""
        involved_nodes = [idx]
        neighborhoods = self.embedder.get_neighborhoods(self.graph, idx, t)
        for n_idx, _, _, mask in neighborhoods:
            involved_nodes.append(n_idx[mask])
        involved_nodes = torch.cat(involved_nodes, dim=0)
        involved_nodes = torch.unique(involved_nodes)
        self._update_memory(involved_nodes)

        return self.embedder(self.graph, idx, t, neighborhoods)

    # ----------------------------------------------------------------------- #
    # Private methods
    # ----------------------------------------------------------------------- #

    def _update_memory(self, idx: Tensor) -> None:
        """Updates the memory of the given nodes."""
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
        idx, msg = self.msg_aggregator(src, t, msg)

        # Compute the time of the last message received for each node
        # that has received a message. This is set as the last update time
        # of the memory of the node.
        t, _ = ops.scatter_max(t, src, dim=0)
        t = t[idx]

        old_memory = self.graph.memory.memory[idx]
        new_memory = self.memory_updater(msg, old_memory)
        self.graph.memory[idx] = new_memory, t
