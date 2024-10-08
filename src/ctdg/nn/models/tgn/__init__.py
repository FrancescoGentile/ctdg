# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

"""Module for the Temporal Graph Network (TGN) model."""

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
from ._model import TGN
from ._records import GraphState

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
    # _records
    "GraphState",
    # _tgn
    "TGN",
]
