# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

from typing import Protocol

from torch import Tensor

from ctdg.data import Events


class Dataset(Protocol):
    """Intraface for information cascade datasets."""

    @property
    def num_sources(self) -> int:
        """The number of sources in the dataset.

        The sources are the active users in the information cascade, i.e., the users
        that interact with the content.
        """
        ...

    @property
    def num_destinations(self) -> int:
        """The number of destinations in the dataset.

        The destinations are the content with which the sources interact.
        """
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
