# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

from typing import Protocol

import torch
from torch import Tensor
from typing_extensions import override

from ctdg import ops
from ctdg.nn import Module, TimeEncoder
from ctdg.structures import Events

from ._records import GraphState

# --------------------------------------------------------------------------- #
# Message Function
# --------------------------------------------------------------------------- #


class MessageFunction(Protocol):
    """Interface for message functions."""

    def __call__(self, state: GraphState, events: Events) -> Tensor:
        """Computes the messages for the given events.

        This method should compute the messages to be sent to the source nodes
        of the given events.

        Args:
            state: The current state of the graph.
            events: The events for which to compute the messages.

        Returns:
            The messages for the given events.
        """
        ...


class IdentityMessageFunction(Module, MessageFunction):
    """Message function that concatenates the input features."""

    def __init__(self, time_encoder: TimeEncoder) -> None:
        super().__init__()

        self.time_encoder = time_encoder

    @override
    def __call__(self, state: GraphState, events: Events) -> Tensor:
        src_memory, src_last_update = state.memory[events.src_nodes]
        dst_memory, _ = state.memory[events.dst_nodes]
        delta_t = self.time_encoder(events.timestamps - src_last_update)
        features = state.static_info.events_features[events.indices]

        return torch.cat([src_memory, dst_memory, delta_t, features], dim=-1)


# --------------------------------------------------------------------------- #
# Message Aggregator
# --------------------------------------------------------------------------- #


class MessageAggregator(Protocol):
    """Interface for message aggregators."""

    def __call__(self, msg: Tensor, idx: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Aggregates the messages received by the same node.

        Args:
            msg: The messages received by the nodes.
            idx: The indices of the receiving nodes.
            t: The time at which the messages are received.

        Returns:
            The aggregated messages and the indices of the receiving nodes.
        """
        ...


class LastMessageAggregator(MessageAggregator):
    """Aggregator that retains only the last message received."""

    @override
    def __call__(self, msg: Tensor, idx: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        argmax = ops.scatter_argmax(t, idx, dim=0)
        idx = torch.unique(idx)
        argmax = argmax[idx]

        return msg[argmax], idx


class MeanMessageAggregator(MessageAggregator):
    """Aggregator that computes the mean of the messages received."""

    @override
    def __call__(self, msg: Tensor, idx: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        msg = ops.scatter_mean(msg, idx, dim=0)
        idx = torch.unique(idx)
        return msg[idx], idx
