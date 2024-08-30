# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

"""Module for the Community Network for Continuous-Time Dynamic Graphs."""

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
from ._model import CONNECT
from ._records import GraphState, Rank, RawMessages
from ._rewirer import HDBSCANRewirer, Rewirer

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
    "CONNECT",
    # _records
    "GraphState",
    "Rank",
    "RawMessages",
    # _rewirer
    "HDBSCANRewirer",
    "Rewirer",
]
