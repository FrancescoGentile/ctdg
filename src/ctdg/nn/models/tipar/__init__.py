# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

"""Module for the Temporal Information Propagation through Adaptive Rewiring (TIPAR) model."""  # noqa: E501

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
from ._model import TIPAR
from ._records import GraphState, RawMessages
from ._rewirer import HDBSCANRewirer, Rewirer, SimilarityRewirer

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
    "TIPAR",
    # _records
    "GraphState",
    "RawMessages",
    # _rewirer
    "HDBSCANRewirer",
    "Rewirer",
    "SimilarityRewirer",
]
