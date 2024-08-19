# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.utils.data

from ctdg.structures import Events


class DataLoader(torch.utils.data.DataLoader[int]):
    """Data loader for link prediction."""

    def __init__(
        self,
        events: Events,
        *,
        batch_size: int = 1,
        seed: int = 0,
        fixed_negatives: bool = False,
    ) -> None:
        """Initializes the data loader.

        Args:
            events: The stream of events.
            batch_size: The batch size.
            drop_last: Whether to drop the last incomplete batch.
            num_workers: The number of workers for data loading.
            seed: The random seed used to generate negatives.
            fixed_negatives: Whether to sample negatives once and keep them fixed.
        """
        self.events = events
        self.generator = torch.Generator().manual_seed(seed)
        self.src_list = torch.unique(events.src_nodes)
        self.dst_list = torch.unique(events.dst_nodes)

        if fixed_negatives:
            dst_negatives = torch.randint(
                low=0,
                high=len(self.dst_list),
                size=(len(events),),
                generator=self.generator,
            )
            self.negatives = self.dst_list[dst_negatives]
        else:
            self.negatives = None

        super().__init__(
            dataset=list(range(len(events))),  # type: ignore
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=self._collate_fn,
        )

    def _collate_fn(self, batch: list[int]) -> tuple[Events, Events]:
        pos_events = self.events[batch]
        if self.negatives is not None:
            neg_events = Events(
                src_nodes=pos_events.src_nodes,
                dst_nodes=self.negatives[batch],
                timestamps=pos_events.timestamps,
                indices=torch.full_like(pos_events.indices, -1),
            )
        else:
            dst_negatives = torch.randint(
                low=0,
                high=len(self.dst_list),
                size=(len(batch),),
                generator=self.generator,
            )
            neg_events = Events(
                src_nodes=pos_events.src_nodes,
                dst_nodes=self.dst_list[dst_negatives],
                timestamps=pos_events.timestamps,
                indices=torch.full_like(pos_events.indices, -1),
            )

        return pos_events, neg_events
