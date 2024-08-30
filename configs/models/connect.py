# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

from torch import nn

from ctdg.nn import TimeEncoder
from ctdg.nn.models.connect import (
    CONNECT,
    GraphAttentionEmbedder,
    GraphAttentionLayer,
    HDBSCANRewirer,
    IdentityMessageFunction,
    LastMessageAggregator,
    MeanMessageAggregator,
)
from ctdg.utils import LazyCall as L


def get_model(nodes_dim: int, events_dim: int) -> L[CONNECT]:
    dim = nodes_dim if nodes_dim > 0 else 172
    message_dim = 3 * dim + events_dim

    time_encoder = L(TimeEncoder, cache=True)(dim)

    return L(CONNECT)(
        memory_dim=dim,
        rewirer=L(HDBSCANRewirer)(
            num_neighbors=10,
            metric="cosine",
            min_cluster_size=10,
        ),
        nodes_memory_updater=L(nn.GRUCell)(message_dim, dim),
        comms_memory_updater=L(nn.GRUCell)(message_dim, dim),
        src_message_function=L(IdentityMessageFunction)(time_encoder),
        dst_message_function=L(IdentityMessageFunction)(time_encoder),
        comms_message_function=L(IdentityMessageFunction)(time_encoder),
        nodes_message_aggregator=L(LastMessageAggregator)(),
        comms_message_aggregator=L(MeanMessageAggregator)(),
        embedder=L(GraphAttentionEmbedder)(
            time_encoder=time_encoder,
            num_layers=1,
            layer=L(GraphAttentionLayer)(
                dim=dim,
                time_dim=dim,
                output_dim=dim,
                num_heads=2,
                dropout=0.1,
            ),
        ),
    )
