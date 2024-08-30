# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

import math
from copy import deepcopy
from typing import Protocol

import torch
from torch import Tensor, nn
from typing_extensions import override

from ctdg.nn import MLP, Module, Neighborhood, TimeEncoder

from ._records import GraphState


class Embedder(Protocol):
    """Interface for embedding modules."""

    @property
    def output_dim(self) -> int:
        """The dimension of the embeddings."""
        ...

    def get_neighborhoods(
        self,
        state: GraphState,
        idx: Tensor,
        t: Tensor,
    ) -> list[Neighborhood]:
        """Returns the neighborhoods needed to compute the embeddings."""
        ...

    def __call__(
        self,
        state: GraphState,
        idx: Tensor,
        t: Tensor,
        neighborhoods: list[Neighborhood],
    ) -> Tensor:
        """Computes the embeddings of the nodes at the given times.

        Args:
            state: The current state of the model.
            idx: The indices of the nodes for which to compute the embeddings.
                This is a tensor of shape `(N,)` where `N` is the number of nodes
                for which to compute the embeddings.
            t: The times at which to compute the embeddings. This should be a tensor
                with the same shape as `idx`.
            neighborhoods: The neighborhoods needed to compute the embeddings.

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
    def get_neighborhoods(
        self,
        state: GraphState,
        idx: Tensor,
        t: Tensor,
    ) -> list[Neighborhood]:
        # no neighborhoods needed
        return []

    @override
    def __call__(
        self,
        state: GraphState,
        idx: Tensor,
        t: Tensor,
        neighborhoods: list[Neighborhood],
    ) -> Tensor:
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
    def get_neighborhoods(
        self,
        state: GraphState,
        idx: Tensor,
        t: Tensor,
    ) -> list[Neighborhood]:
        # no neighborhoods needed
        return []

    @override
    def __call__(
        self,
        state: GraphState,
        idx: Tensor,
        t: Tensor,
        neighborhoods: list[Neighborhood],
    ) -> Tensor:
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

        event_dim = event_dim if event_dim is not None else node_dim
        time_dim = time_dim if time_dim is not None else node_dim
        output_dim = output_dim if output_dim is not None else node_dim

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
        kv_event: Tensor | None,
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

        if kv_event is not None:
            kv = torch.cat([kv, kv_time, kv_event], dim=-1)  # (N, K, Dk + Dt + De)
        else:
            kv = torch.cat([kv, kv_time], dim=-1)

        # this check is necessary because at first the rewire neighborhood is empty
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

        output = torch.cat([output, src_features], dim=-1)  # (N, Dq + D)
        return self.merger(output)


class GraphAttentionEmbedder(Module, Embedder):
    """An embedding module that uses graph attention to compute the embeddings."""

    def __init__(
        self,
        time_encoder: TimeEncoder,
        temporal_layer: GraphAttentionLayer | None,
        rewire_layer: GraphAttentionLayer | None,
        num_layers: int,
        num_neighbors: int,
        *,
        shared_layers: bool = False,
    ) -> None:
        super().__init__()

        if temporal_layer is None and rewire_layer is None:
            msg = "At least one of the temporal and rewire layers must be provided."
            raise ValueError(msg)

        self.num_layers = num_layers
        self.num_neighbors = num_neighbors
        self.time_encoder = time_encoder

        if temporal_layer is not None:
            self.t_layers = nn.ModuleList([
                deepcopy(temporal_layer) for _ in range(num_layers)
            ])
        else:
            self.t_layers = None

        if rewire_layer is not None:
            if shared_layers and self.t_layers is not None:
                self.r_layers = self.t_layers
            else:
                self.r_layers = nn.ModuleList([
                    deepcopy(rewire_layer) for _ in range(num_layers)
                ])
        else:
            self.r_layers = None

    @property
    def output_dim(self) -> int:
        if self.t_layers is not None:
            return self.t_layers[-1].output_dim

        if self.r_layers is not None:
            return self.r_layers[-1].output_dim

        msg = "No layers defined."
        raise RuntimeError(msg)

    @override
    def get_neighborhoods(
        self,
        state: GraphState,
        idx: Tensor,
        t: Tensor,
    ) -> list[Neighborhood]:
        temporal = self._get_temporal_neighborhoods(state, idx, t)
        rewired = self._get_rewire_neighborhoods(state, idx, t)

        return temporal + rewired

    @override
    def __call__(
        self,
        state: GraphState,
        idx: Tensor,
        t: Tensor,
        neighborhoods: list[Neighborhood],
    ) -> Tensor:
        if self.t_layers is not None:
            t_neighborhoods = neighborhoods[: self.num_layers]
            t_embeds = self._compute_temporal(state, idx, t, t_neighborhoods, depth=0)
        else:
            t_embeds = 0.0

        if self.r_layers is not None:
            r_neighborhoods = neighborhoods[-self.num_layers :]
            r_embeds = self._compute_rewire(state, idx, t, r_neighborhoods, depth=0)
        else:
            r_embeds = 0.0

        return t_embeds + r_embeds  # type: ignore

    # ----------------------------------------------------------------------- #
    # Private Methods
    # ----------------------------------------------------------------------- #

    def _compute_temporal(
        self,
        state: GraphState,
        idx: Tensor,
        t: Tensor,
        neighborhoods: list[Neighborhood],
        depth: int,
    ) -> Tensor:
        memory, _ = state.memory[idx]
        if state.nodes_features is not None:
            q = state.nodes_features[idx] + memory
        else:
            q = memory

        if depth == self.num_layers:
            return q

        neighbors, e_t, e_id, mask = neighborhoods[depth]
        num_neighbors = mask.sum(dim=-1)  # (N,)

        n_embeds = self._compute_temporal(
            state,
            idx=neighbors[mask],
            t=torch.repeat_interleave(t, num_neighbors),
            neighborhoods=neighborhoods,
            depth=depth + 1,
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
        if state.events_features is not None:
            kv_event = state.events_features[e_id]
        else:
            kv_event = None

        return self.t_layers[depth](q, q_time, kv, kv_time, kv_event, mask)  # type: ignore

    def _compute_rewire(
        self,
        state: GraphState,
        idx: Tensor,
        t: Tensor,
        neighborhoods: list[Neighborhood],
        depth: int,
    ) -> Tensor:
        memory, _ = state.memory[idx]
        if state.nodes_features is not None:
            q = state.nodes_features[idx] + memory
        else:
            q = memory

        if depth == self.num_layers:
            return q

        neighbors, e_t, e_id, mask = neighborhoods[depth]
        num_neighbors = mask.sum(dim=-1)  # (N,)

        n_embeds = self._compute_rewire(
            state,
            idx=neighbors[mask],
            t=torch.repeat_interleave(t, num_neighbors),
            neighborhoods=neighborhoods,
            depth=depth + 1,
        )

        kv = torch.zeros(
            len(q),
            int(num_neighbors.max()),
            n_embeds.size(-1),
            dtype=n_embeds.dtype,
            device=n_embeds.device,
        )
        kv[mask] = n_embeds

        q_time = self.time_encoder(torch.zeros_like(t))

        time_diff = t.unsqueeze(-1) - e_t  # (N, K)
        kv_time = self.time_encoder(time_diff)  # (N, K, Dt)
        kv_time[e_t == -1] = 0.0

        if state.events_features is not None:
            kv_event = state.events_features[e_id]
            kv_event[e_id == -1] = 0.0
        else:
            kv_event = None

        return self.r_layers[depth](q, q_time, kv, kv_time, kv_event, mask)  # type: ignore

    def _get_temporal_neighborhoods(
        self, state: GraphState, idx: Tensor, t: Tensor
    ) -> list[Neighborhood]:
        neighborhoods = []
        for it in range(self.num_layers):
            neighborhood = state.neighbor_sampler.sample(idx, t, self.num_neighbors)
            neighborhoods.append(neighborhood)
            neighbors, _, _, mask = neighborhood

            if it < self.num_layers - 1:
                num_neighbors = mask.sum(dim=-1)  # (N,)
                idx = neighbors[mask]
                t = torch.repeat_interleave(t, num_neighbors)

        return neighborhoods

    def _get_rewire_neighborhoods(
        self, state: GraphState, idx: Tensor, t: Tensor
    ) -> list[Neighborhood]:
        neighborhoods = []
        for _ in range(self.num_layers):
            adj = state.adjacency[idx]  # (P, N)
            _, n_idx = adj.nonzero(as_tuple=True)  # 2 x (E,)

            num_neighbors = adj.count_nonzero(dim=1)
            max_neighbors = int(num_neighbors.max().item())

            neighbors = torch.full(
                (len(idx), max_neighbors), -1, dtype=torch.long, device=idx.device
            )
            mask = torch.arange(max_neighbors, device=idx.device).expand_as(neighbors)
            mask = mask < num_neighbors.unsqueeze(-1)
            neighbors[mask] = n_idx

            t = torch.repeat_interleave(t, num_neighbors, dim=0)
            src = torch.repeat_interleave(idx, num_neighbors, dim=0)
            dst = n_idx
            e_t, e_id = state.neighbor_sampler.last_interaction(src, dst, t)  # 2 x (E,)

            full_e_t = torch.full_like(neighbors, -1.0)
            full_e_t[mask] = e_t

            full_e_id = torch.full_like(neighbors, -1, dtype=torch.long)
            full_e_id[mask] = e_id

            neighborhoods.append((neighbors, full_e_t, full_e_id, mask))

            idx, t = n_idx, e_t

        return neighborhoods
