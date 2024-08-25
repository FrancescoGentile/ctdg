# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.utils.data
from torch import Tensor

from ctdg.data import Events


class DataLoader(torch.utils.data.DataLoader[int]):
    """DataLoader for the information cascade task."""

    def __init__(
        self,
        events: Events,
        batch_size: int,
        *,
        drop_last: bool = False,
    ) -> None:
        """Initializes the data loader.

        Args:
            events: The stream of events.
            batch_size: The batch size.
            drop_last: Whether to drop the last incomplete batch.
        """
        self.events = events

        # count how mnay times each destination node appears as destination
        counts = torch.bincount(events.dst_nodes)
        counts = counts[events.dst_nodes]
        self.counts = counts.float()

        super().__init__(
            dataset=list(range(len(events))),  # type: ignore
            batch_size=batch_size,
            shuffle=False,
            drop_last=drop_last,
            collate_fn=self._collate_fn,
        )

    # ----------------------------------------------------------------------- #
    # Private methods
    # ----------------------------------------------------------------------- #

    def _collate_fn(self, batch: list[int]) -> tuple[Events, Tensor]:
        events = self.events[batch]
        counts = self.counts[batch]
        return events, counts
