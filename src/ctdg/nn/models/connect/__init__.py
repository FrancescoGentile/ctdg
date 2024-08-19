# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

"""Module for the Community Network for Continuous-Time Dynamic Graphs."""

from ._connect import CONNECT
from ._embedder import (
    Embedder,
    HyperConvEmbedder,
    IdentityEmbedder,
    TimeProjectionEmbedder,
)
from ._link_prediction import LinkPredictionCONNECT
from ._meanshift import MeanShift
from ._message import (
    EmbedderMessageFunction,
    IdentityMessageFunction,
    LastMessageAggregator,
    MessageAggregator,
    MessageFunction,
    MLPMessageFunction,
    WeightedSumMessageAggregator,
)
from ._records import NetworkState, Rank

__all__ = [
    # _connect
    "CONNECT",
    # _embedder
    "Embedder",
    "HyperConvEmbedder",
    "IdentityEmbedder",
    "TimeProjectionEmbedder",
    # _link_prediction
    "LinkPredictionCONNECT",
    # _meanshift
    "MeanShift",
    # _message
    "EmbedderMessageFunction",
    "IdentityMessageFunction",
    "LastMessageAggregator",
    "MessageAggregator",
    "MessageFunction",
    "MLPMessageFunction",
    "WeightedSumMessageAggregator",
    # _records
    "NetworkState",
    "Rank",
]
