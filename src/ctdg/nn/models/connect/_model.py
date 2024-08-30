# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from ctdg import ops
from ctdg.data import Events
from ctdg.nn import MemoryUpdater, Module

from ._embedder import Embedder
from ._message import MessageAggregator, MessageFunction
from ._records import GraphState, Rank, RawMessages
from ._rewirer import Rewirer


class CONNECT(Module):
    """The Community Network for Continuous-Time Dynamic Graphs (CONNECT) model."""

    def __init__(
        self,
        num_nodes: int,
        nodes_features: Tensor | None,
        events_features: Tensor | None,
        memory_dim: int,
        rewirer: Rewirer,
        nodes_memory_updater: MemoryUpdater,
        comms_memory_updater: MemoryUpdater,
        src_message_function: MessageFunction,
        dst_message_function: MessageFunction,
        comms_message_function: MessageFunction,
        nodes_message_aggregator: MessageAggregator,
        comms_message_aggregator: MessageAggregator,
        embedder: Embedder,
    ) -> None:
        super().__init__()

        self.graph = GraphState(
            num_nodes=num_nodes,
            nodes_features=nodes_features,
            events_features=events_features,
            memory_dim=memory_dim,
        )

        self.rewirer = rewirer

        self.nodes_memory_updater = nodes_memory_updater
        self.comms_memory_updater = comms_memory_updater

        self.src_msg_func = src_message_function
        self.dst_msg_func = dst_message_function
        self.comms_msg_func = comms_message_function

        self.nodes_msg_agg = nodes_message_aggregator
        self.comms_msg_agg = comms_message_aggregator

        self.embedder = embedder

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def reset_graph_state(self) -> None:
        """Resets the state of the graph."""
        self.graph.reset()

    def detach_graph_state(self) -> None:
        """Detaches the state of the graph."""
        self.graph.detach()

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
        self.graph.src_store.store(raw_msg.src_nodes, raw_msg)

        flipped_events = events.flip_direction()
        flipped_msg = RawMessages.from_events(flipped_events, dst_embeds, src_embeds)
        self.graph.dst_store.store(flipped_msg.src_nodes, flipped_msg)

        # given a node i that is involved in an event at time t, we also dispatch
        # the event to all the communities to which i belongs
        all_msg = RawMessages.cat([raw_msg, flipped_msg])
        idx = torch.unique(all_msg.src_nodes)
        for i in idx:
            i_msg = all_msg[all_msg.src_nodes == i]
            c_idx = self.graph.incidence_matrix[i].nonzero(as_tuple=True)[0]
            C, E = len(c_idx), len(i_msg)  # noqa: N806
            if C == 0:
                continue

            # to each community in c_idx we need to send all i_events
            c_idx = torch.repeat_interleave(c_idx, E, dim=0)
            i_msg = i_msg.repeat(C)

            self.graph.comms_store.store(c_idx, i_msg)

    def flush_events(self) -> None:
        """Flushes the events from the graph state."""
        idx = torch.arange(self.graph.num_nodes, device=self.graph.device)
        self._update_nodes_memory(idx)

        idx = torch.arange(self.graph.num_communities, device=self.graph.device)
        self._update_communities_memory(idx)

    def rewire_graph(self) -> None:
        """Rewires the graph."""
        self.flush_events()

        incidence = self.rewirer(self.graph)

        memory = incidence.T @ self.graph.nodes_memory.memory
        c_idx, n_idx = incidence.T.nonzero(as_tuple=True)
        c_t, _ = ops.scatter_max(self.graph.nodes_memory.last_update[n_idx], c_idx, 0)

        self.graph.update_communities(incidence, memory, c_t)

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
        involved_communities = []

        neighborhoods = self.embedder.get_neighborhoods(self.graph, idx)
        for i, mask, rank in neighborhoods:
            if rank == Rank.NODES:
                involved_nodes.append(i[mask])
            else:
                involved_communities.append(i[mask])

        involved_nodes = torch.cat(involved_nodes, dim=0)
        involved_nodes = torch.unique(involved_nodes)
        self._update_nodes_memory(involved_nodes)

        if len(involved_communities) > 0:
            involved_communities = torch.cat(involved_communities, dim=0)
            involved_communities = torch.unique(involved_communities)
            self._update_communities_memory(involved_communities)

        return self.embedder(self.graph, idx, t, neighborhoods)

    # ----------------------------------------------------------------------- #
    # Private Methods
    # ----------------------------------------------------------------------- #

    def _update_nodes_memory(self, idx: Tensor) -> None:
        """Updates the memory of the given nodes.

        Args:
            idx: The indices of the nodes for which to update the memory.
        """
        # compute messages (src -> dst)
        tmp = self.graph.src_store.retrieve(idx, clear=True)
        if tmp is None:
            return
        _, s_raw_msg = tmp
        s_src, s_t = s_raw_msg.src_nodes, s_raw_msg.timestamps
        s_msg = self.src_msg_func(self.graph, s_raw_msg, s_src, Rank.NODES)

        # compute messages (dst -> src)
        tmp = self.graph.dst_store.retrieve(idx, clear=True)
        if tmp is None:
            return
        _, d_raw_msg = tmp
        d_src, d_t = d_raw_msg.src_nodes, d_raw_msg.timestamps
        d_msg = self.dst_msg_func(self.graph, d_raw_msg, d_src, Rank.NODES)

        # aggregate messages
        src = torch.cat([s_src, d_src], dim=0)
        t = torch.cat([s_t, d_t], dim=0)
        msg = torch.cat([s_msg, d_msg], dim=0)
        if len(src) == 0:
            return

        idx, msg = self.nodes_msg_agg(src, t, msg)

        # Compute the time of the last message received for each node
        # that has received a message. This is set as the last update time
        # of the memory of the node.
        t, _ = ops.scatter_max(t, src, dim=0)
        t = t[idx]

        old_memory = self.graph.nodes_memory.memory[idx]
        new_memory = self.nodes_memory_updater(msg, old_memory)
        self.graph.nodes_memory[idx] = new_memory, t

    def _update_communities_memory(self, idx: Tensor) -> None:
        """Updates the memory of the given communities.

        Args:
            idx: The indices of the communities for which to update the memory.
        """
        tmp = self.graph.comms_store.retrieve(idx, clear=True)
        if tmp is None:
            return
        idx, raw_msg = tmp
        msg = self.comms_msg_func(self.graph, raw_msg, idx, Rank.COMMUNITIES)

        agg_idx, msg = self.comms_msg_agg(idx, raw_msg.timestamps, msg)

        t, _ = ops.scatter_max(raw_msg.timestamps, idx, dim=0)
        t = t[agg_idx]

        old_memory = self.graph.comms_memory.memory[agg_idx]
        new_memory = self.comms_memory_updater(msg, old_memory)
        self.graph.comms_memory[agg_idx] = new_memory, t
