# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

import math
from copy import deepcopy
from typing import Protocol

import torch
from torch import Tensor, nn
from typing_extensions import override

from ctdg.nn import MLP, Module, TimeEncoder

from ._records import GraphState, Rank


class Embedder(Protocol):
    """Interface for embedding modules."""

    @property
    def output_dim(self) -> int:
        """The output dimension of the embedding."""
        ...

    def get_neighborhoods(
        self, state: GraphState, n_idx: Tensor
    ) -> list[tuple[Tensor, Tensor, Rank]]:
        """Returns the neighborhoods required to compute the embeddings."""
        ...

    def __call__(
        self,
        state: GraphState,
        n_idx: Tensor,
        t: Tensor,
        neighborhoods: list[tuple[Tensor, Tensor, Rank]],
    ) -> Tensor:
        """Computes the embeddings of the given nodes."""
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
    def get_neighborhoods(
        self, state: GraphState, n_idx: Tensor
    ) -> list[tuple[Tensor, Tensor, Rank]]:
        return []

    @override
    def __call__(
        self,
        state: GraphState,
        idx: Tensor,
        t: Tensor,
        neighborhoods: list[tuple[Tensor, Tensor, Rank]],
    ) -> Tensor:
        return state.nodes_memory.memory[idx]


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
    def get_neighborhoods(
        self, state: GraphState, n_idx: Tensor
    ) -> list[tuple[Tensor, Tensor, Rank]]:
        return []

    @override
    def __call__(
        self,
        state: GraphState,
        idx: Tensor,
        t: Tensor,
        neighborhoods: list[tuple[Tensor, Tensor, Rank]],
    ) -> Tensor:
        embeds, last_update = state.nodes_memory[idx]
        time_diff = t - last_update
        return (1 + self.proj(time_diff.unsqueeze(-1))) * embeds


# --------------------------------------------------------------------------- #
# Graph Attention Embedding
# --------------------------------------------------------------------------- #


class GraphAttentionLayer(Module):
    """A single layer of the graph attention embedding."""

    def __init__(
        self,
        dim: int,
        time_dim: int,
        output_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()

        q_dim = dim + time_dim
        k_dim = dim + time_dim

        self.output_dim = output_dim

        self.attn = nn.MultiheadAttention(
            embed_dim=q_dim,
            kdim=k_dim,
            vdim=k_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.merger = MLP(
            input_dim=dim + q_dim,
            hidden_dim=dim,
            output_dim=output_dim,
            num_layers=2,
            activation=nn.ReLU,
            dropout=0.0,
        )

    def __call__(
        self,
        q: Tensor,
        q_time: Tensor,
        kv: Tensor,
        kv_time: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Computes the attention embeddings."""
        src_features = q

        q = torch.cat([q, q_time], dim=-1)  # (N, Dq)
        q = q.unsqueeze(1)  # (N, 1, Dq)

        kv = torch.cat([kv, kv_time], dim=-1)  # (N, K, Dk + Dt)

        if kv.shape[1] > 0:
            # PyTorch nn.MultiheadAttention follows the opposite convention for the
            # mask: a value of `True` indicates that the key-value pair is invalid
            # (and it will be ignored in the attention computation), while a value of
            # `False` indicates that the key-value pair is valid (and it will be used
            # in the attention computation). Therefore, we need to invert the mask.
            mask = ~mask

            # it may happen that a node has no neighbors, in this case the mask will
            # be all `True` and the attention computation will fail. To avoid this,
            # we set the mask to `False` for nodes with no neighbors, thus making the
            # node attend to all the fake neighbors with zero embeddings. Then, to
            # discard this fake attention, we will set the output embeddings to zero
            # for nodes with no neighbors.
            no_neighbors = mask.all(dim=-1)  # (N,)
            mask[no_neighbors] = False

            output, _ = self.attn(q, kv, kv, key_padding_mask=mask, need_weights=False)
            output = output.squeeze(1)  # (N, Dq)
            output.masked_fill_(no_neighbors.unsqueeze(-1), 0.0)
        else:
            output = torch.zeros_like(q).squeeze(1)

        output = torch.cat([src_features, output], dim=-1)
        return self.merger(output)


class GraphAttentionEmbedder(Module, Embedder):
    """An embedding module that uses graph attention to compute the embeddings."""

    def __init__(
        self,
        time_encoder: TimeEncoder,
        layer: GraphAttentionLayer,
        num_layers: int,
    ) -> None:
        super().__init__()

        self.time_encoder = time_encoder
        self.num_layers = num_layers
        self.layers = nn.ModuleList([deepcopy(layer) for _ in range(num_layers)])

    @property
    def output_dim(self) -> int:
        return self.layers[-1].output_dim

    @override
    def get_neighborhoods(
        self, state: GraphState, n_idx: Tensor
    ) -> list[tuple[Tensor, Tensor, Rank]]:
        neighborhoods = []
        rank = Rank.NODES
        idx = n_idx
        for _ in range(self.num_layers):
            if rank == Rank.NODES:
                incidence = state.incidence_matrix[idx]
                _, c_idx = incidence.nonzero(as_tuple=True)

                num_neighbors = incidence.count_nonzero(dim=-1)
                max_neighbors = int(num_neighbors.max().item())

                neighbors = torch.full(
                    (len(idx), max_neighbors), -1, dtype=torch.long, device=idx.device
                )
                mask = torch.arange(max_neighbors, device=idx.device)
                mask = mask.expand_as(neighbors) < num_neighbors.unsqueeze(-1)
                neighbors[mask] = c_idx

                neighborhoods.append((neighbors, mask, Rank.COMMUNITIES))

                idx = c_idx
                rank = Rank.COMMUNITIES
            else:
                incidence = state.incidence_matrix.T[idx]
                _, n_idx = incidence.nonzero(as_tuple=True)

                num_neighbors = incidence.count_nonzero(dim=-1)
                max_neighbors = int(num_neighbors.max().item())

                neighbors = torch.full(
                    (len(idx), max_neighbors), -1, dtype=torch.long, device=idx.device
                )
                mask = torch.arange(max_neighbors, device=idx.device)
                mask = mask.expand_as(neighbors) < num_neighbors.unsqueeze(-1)
                neighbors[mask] = n_idx

                neighborhoods.append((neighbors, mask, Rank.NODES))

                idx = n_idx
                rank = Rank.NODES

        return neighborhoods

    @override
    def __call__(
        self,
        state: GraphState,
        idx: Tensor,
        t: Tensor,
        neighborhoods: list[tuple[Tensor, Tensor, Rank]],
    ) -> Tensor:
        return self._compute_nodes(state, idx, t, neighborhoods, 0)

    def _compute_nodes(
        self,
        state: GraphState,
        idx: Tensor,
        t: Tensor,
        neighborhoods: list[tuple[Tensor, Tensor, Rank]],
        depth: int,
    ) -> Tensor:
        memory = state.nodes_memory.memory[idx]
        if state.nodes_features is not None:
            q = memory + state.nodes_features[idx]
        else:
            q = memory

        if depth == self.num_layers:
            return q

        comms, mask, _ = neighborhoods[depth]
        num_neighbors = mask.sum(dim=-1)

        c_embeds = self._compute_communities(
            state,
            idx=comms[mask],
            t=torch.repeat_interleave(t, num_neighbors),
            neighborhoods=neighborhoods,
            depth=depth + 1,
        )

        kv = torch.zeros(
            len(q),
            mask.shape[1],
            c_embeds.shape[-1],
            dtype=c_embeds.dtype,
            device=c_embeds.device,
        )
        kv[mask] = c_embeds

        q_time = self.time_encoder(torch.zeros_like(t))

        time_diff = t.unsqueeze(-1) - state.comms_memory.last_update[comms]
        kv_time = self.time_encoder(time_diff)
        kv_time[mask] = 0.0

        return self.layers[depth](q, q_time, kv, kv_time, mask)

    def _compute_communities(
        self,
        state: GraphState,
        idx: Tensor,
        t: Tensor,
        neighborhoods: list[tuple[Tensor, Tensor, Rank]],
        depth: int,
    ) -> Tensor:
        memory = state.comms_memory.memory[idx]

        if depth == self.num_layers:
            return memory

        nodes, mask, _ = neighborhoods[depth]
        num_neighbors = mask.sum(dim=-1)

        n_embeds = self._compute_nodes(
            state,
            idx=nodes[mask],
            t=torch.repeat_interleave(t, num_neighbors),
            neighborhoods=neighborhoods,
            depth=depth + 1,
        )

        kv = torch.zeros(
            len(memory),
            mask.shape[1],
            n_embeds.shape[-1],
            dtype=n_embeds.dtype,
            device=n_embeds.device,
        )
        kv[mask] = n_embeds

        q_time = self.time_encoder(torch.zeros_like(t))

        time_diff = t.unsqueeze(-1) - state.nodes_memory.last_update[nodes]
        kv_time = self.time_encoder(time_diff)
        kv_time[mask] = 0.0

        return self.layers[depth](memory, q_time, kv, kv_time, mask)
