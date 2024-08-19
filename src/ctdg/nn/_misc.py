# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

from torch import Tensor

from ._module import Module


class StaticInfo(Module):
    """Information about the temporal graph that does not change over time."""

    def __init__(self, nodes_features: Tensor, events_features: Tensor) -> None:
        super().__init__()

        self.register_buffer("nodes_features", None, persistent=False)
        self.nodes_features = nodes_features

        self.register_buffer("events_features", None, persistent=False)
        self.events_features = events_features

    @property
    def num_nodes(self) -> int:
        return self.nodes_features.size(0)
