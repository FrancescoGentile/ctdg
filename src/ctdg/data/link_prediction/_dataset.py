# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

from typing import Protocol

from torch import Tensor

from ctdg.data import Events


class Dataset(Protocol):
    """Intraface for link prediction datasets."""

    @property
    def num_nodes(self) -> int:
        """The number of nodes in the dataset."""
        ...

    @property
    def nodes_features(self) -> Tensor | None:
        """Returns the nodes features."""
        ...

    @property
    def events_features(self) -> Tensor | None:
        """Returns the events features."""
        ...

    @property
    def events(self) -> Events:
        """Returns the full event stream."""
        ...

    @property
    def train_events(self) -> Events:
        """Returns the stream of training events."""
        ...

    @property
    def val_events(self) -> Events:
        """Returns the stream of validation events."""
        ...

    @property
    def test_events(self) -> Events:
        """Returns the stream of testing events."""
        ...
