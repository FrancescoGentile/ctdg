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
from ._lightning import LightningTGN
from ._message import (
    IdentityMessageFunction,
    LastMessageAggregator,
    MeanMessageAggregator,
    MessageAggregator,
    MessageFunction,
)
from ._neighbor import LastNeighborSampler, NeighborSampler
from ._records import GraphState
from ._tgn import TGN

__all__ = [
    # _embedder
    "Embedder",
    "GraphAttentionEmbedder",
    "GraphAttentionLayer",
    "IdentityEmbedder",
    "TimeProjectionEmbedder",
    # _lightning
    "LightningTGN",
    # _message
    "IdentityMessageFunction",
    "LastMessageAggregator",
    "MeanMessageAggregator",
    "MessageAggregator",
    "MessageFunction",
    # _neighbor
    "LastNeighborSampler",
    "NeighborSampler",
    # _records
    "GraphState",
    # _tgn
    "TGN",
]
