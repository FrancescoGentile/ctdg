# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import torch
from torch import Tensor, nn

from ctdg import ops
from ctdg.nn import EventStore, Memory, MemoryUpdater, Module
from ctdg.structures import Events

from ._embedder import Embedder
from ._meanshift import MeanShift
from ._message import MessageAggregator, MessageFunction
from ._records import NetworkState


class CONNECT(Module):
    """The Community Network for Continuous-Time Dynamic Graphs model."""

    def __init__(
        self,
        nodes_features: Tensor,
        events_features: Tensor,
        ms_projector: nn.Module,
        meanshift: MeanShift,
        memory_dim: int,
        memory_updater: MemoryUpdater,
        sn_message_function: MessageFunction,
        dn_message_function: MessageFunction,
        c_message_function: MessageFunction,
        n_message_aggregator: MessageAggregator,
        c_message_aggregator: MessageAggregator,
        embedder: Embedder,
    ) -> None:
        super().__init__()

        self.network = NetworkState(nodes_features, events_features, memory_dim)

        self.ms_projector = ms_projector
        self.meanshift = meanshift

        self.n_updater = memory_updater
        self.c_updater = deepcopy(memory_updater)

        self.sn_msg_func = sn_message_function
        self.dn_msg_func = dn_message_function
        self.c_msg_func = c_message_function

        self.n_msg_agg = n_message_aggregator
        self.c_msg_agg = c_message_aggregator

        self.embedder = embedder

        self.sn_store = EventStore(self.network.num_nodes, conflict_mode="error")
        self.dn_store = EventStore(self.network.num_nodes, conflict_mode="error")
        self.c_store = EventStore(0, conflict_mode="append")

    # ----------------------------------------------------------------------- #
    # Public methods
    # ----------------------------------------------------------------------- #

    @torch.no_grad()  # type: ignore
    def reset_graph_state(self) -> None:
        """Resets the state of the temporal graph.

        This method should be called at the beginning of each epoch to reset the state
        of the graph as if no events have happened yet.
        """
        self.network.reset()
        self.sn_store.clear()
        self.dn_store.clear()
        self.c_store = EventStore(0, conflict_mode="append")

    def detach_graph_state(self) -> None:
        """Detaches the graph state from the computation graph."""
        self.network.detach()

    def store_events(self, events: Events) -> None:
        """Stores the events in the event stores.

        !!! note

            This method does not update the state of the nodes or the communities.
            To do so, call `flush_events` after storing the events.
        """
        flip_events = events.flip_direction()
        self.sn_store.store(events.src_nodes, events)
        self.dn_store.store(flip_events.src_nodes, flip_events)

        # give a node i that receives an event at time t, we also distribute
        # the event to the communities to which i belongs
        all_events = Events.cat([events, flip_events])
        n_idx = torch.unique(all_events.src_nodes)  # (N',)

        for i in n_idx:
            i_events = all_events[all_events.src_nodes == i]
            (c_idx,) = torch.nonzero(self.network.nc_membership[i], as_tuple=True)
            C, E = len(c_idx), len(i_events)  # noqa: N806

            # to each community c in c_idx, we needto send all i_events
            c_idx = torch.repeat_interleave(c_idx, E, dim=0)
            i_events = i_events.repeat(C)

            self.c_store.store(c_idx, i_events)

    def flush_events(self) -> None:
        """Updates the state of the nodes and the communities with the stored events."""
        idx = torch.arange(self.network.num_nodes, device=self.network.device)
        self._update_memory(idx)

    def detect_communities(self) -> None:
        """Finds the new communities in the temporal graph."""
        self.flush_events()
        self._detect_communities()

    def compute_node_embeddings(self, idx: Tensor, t: Tensor) -> Tensor:
        """Computes the embeddings of the nodes at the given timestamps.

        Args:
            idx: The indices of the nodes for which to compute the embeddings.
                This should be a tensor of shape `(N,)`.
            t: The timestamps at which to compute the embeddings. This should
                have the same shape as `idx`.

        Returns:
            The embeddings of the nodes at the given timestamps. This is a tensor
            of shape `(N, D)` where `D` is the dimension of the embeddings.
        """
        unique_idx = torch.unique(idx)
        self._update_memory(unique_idx)

        return self.embedder(self.network, idx, rank=0, t=t)

    # ----------------------------------------------------------------------- #
    # Private methods
    # ----------------------------------------------------------------------- #

    def _update_memory(self, n_idx: Tensor) -> None:
        """Updates the memory of the nodes and the communities.

        !!! note

            This method assumes that the provided indices are unique.

        Args:
            n_idx: The indices of the nodes for which to update the memory.
                This should be a tensor of shape `(N,)`.
        """
        # Before updating the nodes, we need to update the communities
        # to which the nodes belong to ensure that nodes use the most
        # recent information about the communities.
        nc_membership = self.network.nc_membership[n_idx]  # (N, C)
        _, c_idx = torch.nonzero(nc_membership, as_tuple=True)
        c_idx = torch.unique(c_idx)

        self._update_communities_memory(c_idx)
        self._update_nodes_memory(n_idx)

    def _update_nodes_memory(self, idx: Tensor) -> None:
        """Updates the memory of the nodes in the temporal graph.

        !!! note

            This method assumes that the provided indices are unique.

        Args:
            idx: The indices of the nodes for which to update the memory.
                This should be a tensor of shape `(N,)`.
        """
        # compute messages (src -> dst)
        s_idx, s_events = self.sn_store.retrieve(idx, clear=True)
        s_msg = self.sn_msg_func(self.network, s_events, s_idx, rank=0)

        # compute messages (dst -> src)
        d_idx, d_events = self.dn_store.retrieve(idx, clear=True)
        d_msg = self.dn_msg_func(self.network, d_events, d_idx, rank=0)

        # aggregate messages
        idx = torch.cat([s_idx, d_idx], dim=0)
        events = Events.cat([s_events, d_events])
        msg = torch.cat([s_msg, d_msg], dim=0)
        if len(events) == 0:
            return

        upd_idx, upd_msg = self.n_msg_agg(self.network, events, msg, idx, rank=0)
        upd_t, _ = ops.scatter_max(events.timestamps, idx, dim=0)
        upd_t = upd_t[upd_idx]

        old_memory, _ = self.network.n_memory[upd_idx]
        upd_memory = self.n_updater(upd_msg, old_memory)

        self.network.n_memory[upd_idx] = upd_memory, upd_t

    def _update_communities_memory(self, idx: Tensor) -> None:
        """Updates the memory of the communities in the temporal graph.

        !!! note

            This method assumes that the provided indices are unique.

        Args:
            idx: The indices of the communities for which to update the memory.
                This should be a tensor of shape `(C,)`.
        """
        idx, events = self.c_store.retrieve(idx, clear=True)
        if len(events) == 0:
            return

        msg = self.c_msg_func(self.network, events, idx, rank=1)

        upd_idx, upd_msg = self.c_msg_agg(self.network, events, msg, idx, rank=1)
        upd_t, _ = ops.scatter_max(events.timestamps, idx, dim=0)
        upd_t = upd_t[upd_idx]

        old_memory, _ = self.network.c_memory[upd_idx]
        upd_memory = self.c_updater(upd_msg, old_memory)

        self.network.c_memory[upd_idx] = upd_memory, upd_t

    def _detect_communities(self) -> None:
        """Detects the communities in the temporal graph."""
        memory = self.network.n_memory.memory
        last_update = self.network.n_memory.last_update
        mask = self.network.nodes_mask

        n_features = memory[mask] + self.network.nodes_features[mask]
        if n_features.size(0) == 0:
            # the beginning of the training no nodes have been observed yet
            # so we can leave zero communities
            return

        n_features = self.ms_projector(n_features)  # (N', D)
        centroids = self.meanshift(n_features)  # (C, D)
        N, C = self.network.num_nodes, len(centroids)  # noqa: N806

        sim = ops.cosine_similarity(n_features, centroids)  # (N', C)

        # compute the membership of the nodes to the communities
        # use sparsemax instead of softmax to enforce sparsity
        nc_membership = ops.sparsemax(sim, dim=-1)  # (N', C)
        self.network.nc_membership = torch.zeros(N, C, device=self.network.device)
        self.network.nc_membership[mask] = nc_membership

        # compute the membership of the communities to the nodes
        cn_membership = ops.sparsemax(sim.T, dim=-1)  # (C, N')
        self.network.cn_membership = torch.zeros(C, N, device=self.network.device)
        self.network.cn_membership[:, mask] = cn_membership

        # compute the memory of the communities
        # to do this, we simply compute the weighted average of the nodes' memory
        self.network.c_memory = Memory(
            C, self.network.c_memory.dim, device=self.network.device
        )

        c_memory = cn_membership @ memory[mask]
        c_idx, n_idx = torch.nonzero(cn_membership, as_tuple=True)
        c_t, _ = ops.scatter_max(last_update[mask][n_idx], c_idx, dim=0)

        c_idx = torch.arange(C, device=self.network.device)
        self.network.c_memory[c_idx] = c_memory, c_t

        self.c_store = EventStore(C, conflict_mode="append")
