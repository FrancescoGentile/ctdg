# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, TypeAlias

import torch
from torch import Tensor

from ctdg.nn import Memory, Module

Rank: TypeAlias = Literal[0, 1]
"""The rank of the cells in the network.

A value of 0 indicates that the cells are nodes, while a value of 1 indicates
that the cells are communities.
"""


class NetworkState(Module):
    """The state of the network."""

    def __init__(
        self,
        nodes_features: Tensor,
        events_features: Tensor,
        memory_dim: int,
    ) -> None:
        super().__init__()

        device = nodes_features.device
        if events_features.device != device:
            msg = "The nodes and events features must be on the same device."
            raise ValueError(msg)

        self._num_nodes = nodes_features.size(0)

        self.register_buffer("nodes_features", None)
        self.nodes_features = nodes_features
        self.register_buffer("events_features", None)
        self.events_features = events_features

        self.n_memory = Memory(self._num_nodes, memory_dim, device=device)
        self.c_memory = Memory(0, memory_dim)

        self.register_buffer("nc_membership", None)
        self.nc_membership = torch.zeros(self._num_nodes, 0, device=device)
        self.register_buffer("cn_membership", None)
        self.cn_membership = torch.zeros(0, self._num_nodes, device=device)

    # ----------------------------------------------------------------------- #
    # Properties
    # ----------------------------------------------------------------------- #

    @property
    def num_nodes(self) -> int:
        """The number of nodes in the network."""
        return self._num_nodes

    @property
    def device(self) -> torch.device:
        """The device of the network."""
        return self.nodes_features.device

    @property
    def nodes_mask(self) -> Tensor:
        """Returns the mask of the nodes that have been observed.

        For efficiency reasons, during training all nodes that will be observed
        (even at validation and test time) are stored in the nodes features and
        in the memory. However, nodes that have yet to be observed should not
        be taken into account when performing computations.

        This property returns a mask that can be used to filter out these nodes.
        It is a boolean tensor of shape `(num_nodes,)` where `True` indicates that
        the node has been observed and `False` indicates that the node has yet to
        be observed.
        """
        return self.n_memory.last_update >= 0

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def reset(self) -> None:
        """Resets the network to its initial state."""
        self.n_memory.reset()
        self.c_memory = Memory(0, self.c_memory.dim, device=self.device)

        self.nc_membership = torch.zeros(self._num_nodes, 0, device=self.device)
        self.cn_membership = torch.zeros(0, self._num_nodes, device=self.device)

    def detach(self) -> None:
        """Detaches the network from the computation graph."""
        self.n_memory.detach()
        self.c_memory.detach()

        self.nc_membership.detach_()
        self.cn_membership.detach_()
