# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

"""Module for the Temporal Interaction Graph Embedding with Restarts (TIGER) model."""

from ._embedder import (
    Embedder,
    GraphAttentionEmbedder,
    GraphAttentionLayer,
    IdentityEmbedder,
    TimeProjectionEmbedder,
)
from ._message import (
    IdentityMessageFunction,
    LastMessageAggregator,
    MeanMessageAggregator,
    MessageAggregator,
    MessageFunction,
)
from ._model import TIGER
from ._records import GraphState, RawMessages

__all__ = [
    # _embedder
    "Embedder",
    "GraphAttentionEmbedder",
    "GraphAttentionLayer",
    "IdentityEmbedder",
    "TimeProjectionEmbedder",
    # _message
    "IdentityMessageFunction",
    "LastMessageAggregator",
    "MeanMessageAggregator",
    "MessageAggregator",
    "MessageFunction",
    # _model
    "TIGER",
    # _records
    "GraphState",
    "RawMessages",
]
