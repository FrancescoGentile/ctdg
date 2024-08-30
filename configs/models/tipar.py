# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

from torch import nn

from ctdg.nn import TimeEncoder
from ctdg.nn.models.tipar import (
    TIPAR,
    GraphAttentionEmbedder,
    GraphAttentionLayer,
    HDBSCANRewirer,
    IdentityMessageFunction,
    LastMessageAggregator,
)
from ctdg.utils import LazyCall as L


def get_model(nodes_dim: int, events_dim: int) -> L[TIPAR]:
    dim = nodes_dim if nodes_dim > 0 else 172
    message_dim = 3 * dim + events_dim

    time_encoder = L(TimeEncoder, cache=True)(dim)

    layer = L(GraphAttentionLayer, cache=True)(
        node_dim=dim,
        event_dim=events_dim,
        time_dim=dim,
        output_dim=dim,
        num_heads=2,
        dropout=0.1,
    )

    return L(TIPAR)(
        memory_dim=dim,
        memory_updater=L(nn.GRUCell)(message_dim, dim),
        src_message_function=L(IdentityMessageFunction)(time_encoder),
        dst_message_function=L(IdentityMessageFunction)(time_encoder),
        message_aggregator=L(LastMessageAggregator)(),
        embedder=L(GraphAttentionEmbedder)(
            time_encoder=time_encoder,
            temporal_layer=layer,
            rewire_layer=layer,
            num_layers=1,
            num_neighbors=10,
            shared_layers=True,
        ),
        rewirer=L(HDBSCANRewirer)(
            num_neighbors=10,
            metric="cosine",
            min_cluster_size=10,
        ),
    )
