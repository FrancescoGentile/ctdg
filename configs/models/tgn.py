##
##

from torch import nn

from ctdg.nn import TimeEncoder
from ctdg.nn.models.tgn import (
    TGN,
    GraphAttentionEmbedder,
    GraphAttentionLayer,
    IdentityMessageFunction,
    LastMessageAggregator,
    LastNeighborSampler,
)
from ctdg.utils import LazyCall as L

dim = 172
message_dim = dim * 4

time_encoder = L(TimeEncoder, cache=True)(dim)

model = L(TGN)(
    memory_dim=dim,
    memory_updater=L(nn.GRUCell)(message_dim, dim),
    s_message_function=L(IdentityMessageFunction)(time_encoder),
    d_message_function=L(IdentityMessageFunction)(time_encoder),
    message_aggregator=L(LastMessageAggregator)(),
    neighbor_sampler=L(LastNeighborSampler)(),
    embedder=L(GraphAttentionEmbedder)(
        layer=L(GraphAttentionLayer)(
            node_dim=dim,
            event_dim=dim,
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
