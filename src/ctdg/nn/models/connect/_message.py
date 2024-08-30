# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

from typing import Protocol

import torch
from torch import Tensor
from typing_extensions import override

from ctdg import ops
from ctdg.nn import Module, TimeEncoder

from ._records import GraphState, Rank, RawMessages

# --------------------------------------------------------------------------- #
# MessageFunction
# --------------------------------------------------------------------------- #


class MessageFunction(Protocol):
    """Interface for message functions."""

    def __call__(
        self,
        state: GraphState,
        msg: RawMessages,
        idx: Tensor,
        rank: Rank,
    ) -> Tensor:
        """Computes the messages for the source nodes.

        Args:
            state: The current state of the graph.
            msg: The raw messages to process.
            idx: The indices of the receiver nodes/communities.
            rank: The rank of the receiver nodes/communities.

        Returns:
            The processed messages.
        """
        ...


class IdentityMessageFunction(Module, MessageFunction):
    """Message function that concatenates the input features."""

    def __init__(self, time_encoder: TimeEncoder) -> None:
        super().__init__()

        self.time_encoder = time_encoder

    @override
    def __call__(
        self,
        state: GraphState,
        msg: RawMessages,
        idx: Tensor,
        rank: Rank,
    ) -> Tensor:
        if rank == Rank.NODES:
            last_update = state.nodes_memory.last_update[idx]
        else:
            last_update = state.comms_memory.last_update[idx]

        delta_t = self.time_encoder(msg.timestamps - last_update)
        if state.events_features is not None:
            e_features = state.events_features[msg.indices]
            features = [msg.src_embeds, msg.dst_embeds, delta_t, e_features]
        else:
            features = [msg.src_embeds, msg.dst_embeds, delta_t]

        return torch.cat(features, dim=-1)


# --------------------------------------------------------------------------- #
# MessageAggregator
# --------------------------------------------------------------------------- #


class MessageAggregator(Protocol):
    """Interface for message aggregators."""

    def __call__(self, idx: Tensor, t: Tensor, msg: Tensor) -> tuple[Tensor, Tensor]:
        """Aggregates the messages received by the same node.

        Args:
            idx: The indices of the receiving nodes.
            t: The time at which the messages are received.
            msg: The messages received by the nodes.

        Returns:
            The indices of the nodes and the aggregated messages.
        """
        ...


class LastMessageAggregator(MessageAggregator):
    """Aggregator that retains only the last message received."""

    @override
    def __call__(self, idx: Tensor, t: Tensor, msg: Tensor) -> tuple[Tensor, Tensor]:
        argmax = ops.scatter_argmax(t, idx, dim=0)
        idx = torch.unique(idx)
        argmax = argmax[idx]

        return idx, msg[argmax]


class MeanMessageAggregator(MessageAggregator):
    """Aggregator that computes the mean of the messages received."""

    @override
    def __call__(self, idx: Tensor, t: Tensor, msg: Tensor) -> tuple[Tensor, Tensor]:
        msg = ops.scatter_mean(msg, idx, dim=0)
        idx = torch.unique(idx)
        return idx, msg[idx]
