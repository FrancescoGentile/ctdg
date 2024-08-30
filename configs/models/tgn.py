# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

from torch import nn

from ctdg.nn import TimeEncoder
from ctdg.nn.models.tgn import (
    TGN,
    GraphAttentionEmbedder,
    GraphAttentionLayer,
    IdentityMessageFunction,
    LastMessageAggregator,
)
from ctdg.utils import LazyCall as L


def get_model(nodes_dim: int, events_dim: int) -> L[TGN]:
    """Returns the TGN model."""
    dim = nodes_dim if nodes_dim > 0 else 172
    # The message dimension is the sum of the source, destination, and time dimensions
    # plus the event dimension.
    message_dim = 3 * dim + events_dim

    time_encoder = L(TimeEncoder, cache=True)(dim)

    return L(TGN)(
        memory_dim=dim,
        memory_updater=L(nn.GRUCell)(message_dim, dim),
        s_message_function=L(IdentityMessageFunction)(time_encoder),
        d_message_function=L(IdentityMessageFunction)(time_encoder),
        message_aggregator=L(LastMessageAggregator)(),
        embedder=L(GraphAttentionEmbedder)(
            layer=L(GraphAttentionLayer)(
                node_dim=dim,
                event_dim=events_dim,
                time_dim=dim,
                output_dim=dim,
                num_heads=2,
                dropout=0.1,
            ),
            num_layers=1,
            num_neighbors=10,
            time_encoder=time_encoder,
        ),
    )
