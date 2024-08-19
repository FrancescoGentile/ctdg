# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

import math
from copy import deepcopy
from typing import Protocol

import torch
from torch import Tensor, nn
from typing_extensions import override

from ctdg.nn import MLP, Module, TimeEncoder

from ._records import GraphState


class Embedder(Protocol):
    """Interface for embedding modules."""

    @property
    def output_dim(self) -> int:
        """The dimension of the embeddings."""
        ...

    def __call__(self, state: GraphState, idx: Tensor, t: Tensor) -> Tensor:
        """Computes the embeddings of the nodes at the given times.

        Args:
            state: The current state of the model.
            idx: The indices of the nodes for which to compute the embeddings.
                This is a tensor of shape `(N,)` where `N` is the number of nodes
                for which to compute the embeddings.
            t: The times at which to compute the embeddings. This should be a tensor
                with the same shape as `idx`.

        Returns:
            The embeddings of the nodes at the given times. This sis a tensor of shape
            `(N, D)` where `D` is the dimension of the embeddings.
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
    def __call__(self, state: GraphState, idx: Tensor, t: Tensor) -> Tensor:
        return state.memory[idx][0]


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
    def __call__(self, state: GraphState, idx: Tensor, t: Tensor) -> Tensor:
        embeds, last_update = state.memory[idx]
        time_diff = t - last_update
        return (1 + self.proj(time_diff.unsqueeze(-1))) * embeds


# --------------------------------------------------------------------------- #
# Graph Attention Embedding
# --------------------------------------------------------------------------- #


class GraphAttentionLayer(Module):
    """A single layer of the graph attention embedding."""

    def __init__(
        self,
        node_dim: int,
        event_dim: int | None = None,
        time_dim: int | None = None,
        output_dim: int | None = None,
        num_heads: int = 1,
        dropout: float = 0.0,
    ) -> None:
        """Initializes the layer.

        Args:
            node_dim: The dimension of the node embeddings.
            event_dim: The dimension of the event embeddings. If `None`, it is set
                to `node_dim`.
            time_dim: The dimension of the time embeddings. If `None`, it is set
                to `node_dim`.
            output_dim: The dimension of the output embeddings. If `None`, it is set
                to `node_dim`.
            num_heads: The number of attention heads.
            dropout: The dropout probability applied to the attention scores.
        """
        super().__init__()

        event_dim = event_dim or node_dim
        time_dim = time_dim or node_dim
        output_dim = output_dim or node_dim

        self.output_dim = output_dim

        query_dim = node_dim + time_dim
        key_dim = node_dim + time_dim + event_dim

        self.attn = nn.MultiheadAttention(
            embed_dim=query_dim,
            kdim=key_dim,
            vdim=key_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.merger = MLP(
            input_dim=query_dim + node_dim,
            hidden_dim=node_dim,
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
        kv_event: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Updates the query embeddings using the key-value embeddings.

        Args:
            q: The query embeddings of shape `(N, Dq)` where `N` is the number of
                queries and `Dq` is the dimension of the embeddings.
            q_time: The time encoding of the queries of shape `(N, Dt)` where
                `Dt` is the dimension of the time embeddings.
            kv: The key-value embeddings of shape `(N, K, Dk)` where `K` is the number
                of key-value pairs and `Dk` is the dimension of the embeddings.
            kv_time: The time encoding of the key-value pairs of shape `(N, K, Dt)`.
            kv_event: The event embeddings of the key-value pairs of shape `(N, K, De)`.
            mask: The binary mask indicating which key-value pairs are valid of shape
                `(N, K)`. A value of `True` indicates that the key-value pair is valid
                (and it will be used in the attention computation), while a value of
                `False` indicates that the key-value pair is not valid (and it will be
                ignored in the attention computation).

        Returns:
            The updated query embeddings of shape `(N, D)`, where `D` is the output
            dimension of the layer.
        """
        src_features = q

        q = torch.cat([q, q_time], dim=-1)  # (N, Dq + Dt)
        q = q.unsqueeze(1)  # (N, 1, Dq + Dt)

        kv = torch.cat([kv, kv_time, kv_event], dim=-1)  # (N, K, Dk + Dt + De)

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

        output = torch.cat([output, src_features], dim=-1)  # (N, Dq + D)
        return self.merger(output)


class GraphAttentionEmbedder(Module, Embedder):
    """An embedding module that uses graph attention to compute the embeddings."""

    def __init__(
        self,
        layer: GraphAttentionLayer,
        num_layers: int,
        num_neighbors: int,
        time_encoder: TimeEncoder,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList([deepcopy(layer) for _ in range(num_layers)])
        self.num_neighbors = num_neighbors
        self.time_encoder = time_encoder

    @property
    def output_dim(self) -> int:
        return self.layers[-1].output_dim

    @override
    def __call__(self, state: GraphState, idx: Tensor, t: Tensor) -> Tensor:
        return self._compute_recursively(state, idx, t, len(self.layers))

    def _compute_recursively(
        self,
        state: GraphState,
        idx: Tensor,
        t: Tensor,
        depth: int,
    ) -> Tensor:
        memory, _ = state.memory[idx]
        q = state.static_info.nodes_features[idx] + memory

        if depth == 0:
            return q

        neighbors, e_t, e_id, mask = state.neighbor_sampler.sample_before(
            idx, t, self.num_neighbors
        )
        num_neighbors = mask.sum(dim=-1)  # (N,)

        n_embeds = self._compute_recursively(
            state,
            idx=neighbors[mask],
            t=torch.repeat_interleave(t, num_neighbors),
            depth=depth - 1,
        )

        kv = torch.zeros(
            len(q),
            self.num_neighbors,
            n_embeds.size(-1),
            dtype=n_embeds.dtype,
            device=n_embeds.device,
        )
        kv[mask] = n_embeds

        q_time = self.time_encoder(torch.zeros_like(t))

        time_diff = t.unsqueeze(-1) - e_t  # (N, K)
        kv_time = self.time_encoder(time_diff)  # (N, K, Dt)
        kv_event = state.static_info.events_features[e_id]

        return self.layers[depth - 1](q, q_time, kv, kv_time, kv_event, mask)
