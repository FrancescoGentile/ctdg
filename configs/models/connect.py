##
##

from torch import nn

from ctdg.nn import TimeEncoder
from ctdg.nn.models.connect import (
    CONNECT,
    EmbedderMessageFunction,
    HyperConvEmbedder,
    IdentityMessageFunction,
    MeanShift,
    WeightedSumMessageAggregator,
)
from ctdg.utils import LazyCall as L

dim = 172

time_encoder = L(TimeEncoder, cache=True)(dim)

embedder = L(HyperConvEmbedder, cache=True)(dim, dim, time_encoder)

n_msg_func = L(EmbedderMessageFunction, cache=True)(embedder, time_encoder)

model = L(CONNECT)(
    meanshift=L(MeanShift)(
        sigma=0.3,
        max_iterations=50,
        shift_threshold=1e-5,
        learnable=True,
    ),
    ms_projector=L(nn.Linear)(dim, dim, bias=False),
    memory_dim=dim,
    memory_updater=L(nn.GRUCell)(dim * 4, dim),
    sn_message_function=n_msg_func,
    dn_message_function=n_msg_func,
    c_message_function=L(IdentityMessageFunction)(time_encoder),
    n_message_aggregator=L(WeightedSumMessageAggregator)(),
    c_message_aggregator=L(WeightedSumMessageAggregator)(),
    embedder=embedder,
)
