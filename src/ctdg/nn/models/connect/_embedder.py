# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Protocol

import torch
from torch import Tensor, nn
from typing_extensions import override

from ctdg import ops
from ctdg.nn import Module, TimeEncoder

from ._records import NetworkState, Rank


class Embedder(Protocol):
    """Interface for embedding modules."""

    @property
    def output_dim(self) -> int:
        """The dimension of the embeddings."""
        ...

    def __call__(
        self,
        state: NetworkState,
        idx: Tensor,
        rank: Rank,
        t: Tensor,
    ) -> Tensor:
        """Computes the embeddings of the nodes at the given time steps.

        Args:
            state: The current state of the temporal graph.
            idx: The indices of cells for which to compute the embeddings.
            rank: The rank of the cells to which the indices refer.
            t: The time steps at which to compute the embeddings. This should be a
                tensor with the same shape as `idx`.

        Returns:
            The embeddings of the nodes at the given time steps. This is a tensor of
            shape `(N, D)` where `D` is the dimension of the embeddings.
        """
        ...


# --------------------------------------------------------------------------- #
# Implementations
# --------------------------------------------------------------------------- #


class IdentityEmbedder(Embedder):
    """An embedding module that returns the node memory as the embeddings."""

    def __init__(self, memory_dim: int) -> None:
        self.dim = memory_dim

    @property
    def output_dim(self) -> int:
        return self.dim

    @override
    def __call__(
        self,
        state: NetworkState,
        idx: Tensor,
        rank: Rank,
        t: Tensor,
    ) -> Tensor:
        memory = state.n_memory if rank == 0 else state.c_memory
        embeds, _ = memory[idx]
        return embeds


class TimeProjectionEmbedder(Module, Embedder):
    """An embedding module that uses a time projection to compute the embeddings."""

    def __init__(self, memory_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(1, memory_dim)

        self.reset_parameters()

    @property
    def output_dim(self) -> int:
        return self.proj.out_features

    @torch.no_grad()  # type: ignore
    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.proj.weight.size(1))
        self.proj.weight.normal_(0, stdv)
        if self.proj.bias is not None:  # type: ignore
            self.proj.bias.data.normal_(0, stdv)

    @override
    def __call__(
        self,
        state: NetworkState,
        idx: Tensor,
        rank: Rank,
        t: Tensor,
    ) -> Tensor:
        memory = state.n_memory if rank == 0 else state.c_memory
        embeds, last_update = memory[idx]
        time_diff = t - last_update
        return (1 + self.proj(time_diff.unsqueeze(-1))) * embeds


class HyperConvEmbedder(Module, Embedder):
    """An embedding module that uses the community memory to compute the embeddings."""

    def __init__(
        self,
        memory_dim: int,
        embed_dim: int,
        time_encoder: TimeEncoder,
    ) -> None:
        super().__init__()

        self.time_encoder = time_encoder
        self.n_proj = nn.Linear(memory_dim + time_encoder.output_dim, embed_dim)
        self.c_proj = nn.Linear(memory_dim + time_encoder.output_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim * 2, embed_dim)

    @property
    def output_dim(self) -> int:
        return self.out_proj.out_features

    @override
    def __call__(
        self,
        state: NetworkState,
        idx: Tensor,
        rank: Rank,
        t: Tensor,
    ) -> Tensor:
        if rank != 0:
            msg = "The HyperConvEmbedder can only be used to embed nodes."
            raise ValueError(msg)

        n_embeds, n_last_update = state.n_memory[idx]  # 2 x (N, D)
        n_time_diff = t - n_last_update
        n_time_embed = self.time_encoder(n_time_diff)  # (N, Dt)
        n_embeds = self.n_proj(torch.cat([n_embeds, n_time_embed], dim=-1))  # (N, De)

        c_embeds, c_last_update = state.c_memory  # 2 x (C, D)
        nc_membership = state.nc_membership[idx]  # (N, C)
        nc_membership = nc_membership.to_sparse_coo()  # (N, C)
        n_idx, c_idx = nc_membership.indices().unbind()  # 2 x (M,)
        c_embeds = c_embeds[c_idx]  # (M, D)
        c_last_update = c_last_update[c_idx]  # (M,)
        c_time_diff = t[n_idx] - c_last_update  # (M,)
        c_time_embed = self.time_encoder(c_time_diff)
        c_embeds = self.c_proj(torch.cat([c_embeds, c_time_embed], dim=-1))
        c_embeds = c_embeds * nc_membership.values().unsqueeze(-1)
        c_embeds = ops.scatter_sum(
            c_embeds, n_idx, dim=0, output_dim_size=n_embeds.size(0)
        )

        embeds = torch.cat([n_embeds, c_embeds], dim=-1)
        return self.out_proj(embeds)
