# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, Protocol

import torch
from torch import Tensor, nn
from typing_extensions import override

from ctdg import ops
from ctdg.nn import MLP, Module, TimeEncoder
from ctdg.structures import Events

from ._embedder import Embedder
from ._records import NetworkState, Rank

# --------------------------------------------------------------------------- #
# Message Function
# --------------------------------------------------------------------------- #


class MessageFunction(Protocol):
    """Interface for message functions."""

    def __call__(
        self,
        state: NetworkState,
        events: Events,
        idx: Tensor,
        rank: Rank,
    ) -> Tensor:
        """Computes the messages for the given events.

        Args:
            state: The current state of the network.
            events: The events for which to compute the messages.
            idx: The indices of the receiving cells.
            rank: The rank of the receiving cells.

        Returns:
            The messages for the given events.
        """
        ...


class IdentityMessageFunction(Module, MessageFunction):
    """A message function that returns the concatenation of the input features."""

    def __init__(self, time_encoder: TimeEncoder) -> None:
        super().__init__()

        self.time_encoder = time_encoder

    @override
    def __call__(
        self,
        state: NetworkState,
        events: Events,
        idx: Tensor,
        rank: Rank,
    ) -> Tensor:
        src_memory = state.n_memory.memory[events.src_nodes]
        dst_memory = state.n_memory.memory[events.dst_nodes]

        if rank == 0:
            t = state.n_memory.last_update[idx]
        else:
            t = state.c_memory.last_update[idx]
        delta_t = self.time_encoder(events.timestamps - t)

        features = state.events_features[events.indices]

        return torch.cat([src_memory, dst_memory, delta_t, features], dim=-1)


class MLPMessageFunction(Module, MessageFunction):
    """A message function that uses an MLP to compute the messages."""

    def __init__(
        self,
        mlp: MLP,
        time_encoder: TimeEncoder,
    ) -> None:
        super().__init__()

        self.mlp = mlp
        self.time_encoder = time_encoder

    @override
    def __call__(
        self,
        state: NetworkState,
        events: Events,
        idx: Tensor,
        rank: Rank,
    ) -> Tensor:
        src_memory = state.n_memory.memory[events.src_nodes]
        dst_memory = state.n_memory.memory[events.dst_nodes]

        if rank == 0:
            t = state.n_memory.last_update[idx]
        else:
            t = state.c_memory.last_update[idx]
        delta_t = self.time_encoder(events.timestamps - t)

        features = state.events_features[events.indices]

        raw_messages = torch.cat([src_memory, dst_memory, delta_t, features], dim=-1)
        return self.mlp(raw_messages)


class EmbedderMessageFunction(Module, MessageFunction):
    """A message function that uses an embedder to compute the messages."""

    def __init__(
        self,
        embedder: Embedder,
        time_encoder: TimeEncoder,
        mlp: MLP | None = None,
    ) -> None:
        super().__init__()

        self.embedder = embedder
        self.time_encoder = time_encoder
        self.mlp = mlp if mlp is not None else nn.Identity()

    @override
    def __call__(
        self,
        state: NetworkState,
        events: Events,
        idx: Tensor,
        rank: Literal[0, 1],
    ) -> Tensor:
        e_idx = torch.cat([events.src_nodes, events.dst_nodes])
        e_t = events.timestamps.repeat(2)
        embeds = self.embedder(state, e_idx, rank=0, t=e_t)
        e_src, e_dst = torch.chunk(embeds, 2, dim=0)

        if rank == 0:
            t = state.n_memory.last_update[idx]
        else:
            t = state.c_memory.last_update[idx]
        delta_t = self.time_encoder(events.timestamps - t)
        features = state.events_features[events.indices]

        msg = torch.cat([e_src, e_dst, delta_t, features], dim=-1)
        return self.mlp(msg)


# --------------------------------------------------------------------------- #
# Message Aggregator
# --------------------------------------------------------------------------- #


class MessageAggregator(Protocol):
    """Interface for message aggregators."""

    def __call__(
        self,
        state: NetworkState,
        events: Events,
        msg: Tensor,
        idx: Tensor,
        rank: Rank,
    ) -> tuple[Tensor, Tensor]:
        """Aggregates the messages received by the same cells.

        Args:
            state: The current state of the graph.
            events: The events for which the messages are computed.
            msg: The messages received by the cells.
            idx: The indices of the receiving cells.
            rank: The rank of the receiving cells.

        Returns:
            The indices of the receiving cells and the aggregated messages.
        """
        ...


class LastMessageAggregator(MessageAggregator):
    """A message aggregator that returns the last message received by the cells."""

    @override
    def __call__(
        self,
        state: NetworkState,
        events: Events,
        msg: Tensor,
        idx: Tensor,
        rank: Rank,
    ) -> tuple[Tensor, Tensor]:
        argmax = ops.scatter_argmax(events.timestamps, idx, dim=0)
        idx = torch.unique(idx)
        argmax = argmax[idx]

        return idx, msg[argmax]


class WeightedSumMessageAggregator(MessageAggregator):
    """A message aggregator that computes the weighted sum of the messages."""

    @override
    def __call__(
        self,
        state: NetworkState,
        events: Events,
        msg: Tensor,
        idx: Tensor,
        rank: Rank,
    ) -> tuple[Tensor, Tensor]:
        if rank == 0:
            msg = ops.scatter_mean(msg, idx, dim=0)
            idx = torch.unique(idx)
            return idx, msg[idx]

        weights = state.cn_membership[idx, events.src_nodes]
        msg = msg * weights.unsqueeze(-1)
        msg = ops.scatter_sum(msg, idx, dim=0)
        idx = torch.unique(idx)
        return idx, msg[idx]
