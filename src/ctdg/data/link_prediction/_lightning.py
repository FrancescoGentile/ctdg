# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch
from lightning import LightningDataModule
from typing_extensions import override

from ctdg.data import Events
from ctdg.utils import LazyCall

from ._dataloader import DataLoader
from ._dataset import Dataset


class DataModule(LightningDataModule):
    """Data module for link prediction."""

    def __init__(
        self,
        dataset: LazyCall[Dataset],
        train_batch_size: int = 1,
        val_batch_size: int = 1,
        test_batch_size: int = 1,
    ) -> None:
        super().__init__()

        self.save_hyperparameters({
            "dataset": dataset.to_dict(),
            "train_batch_size": train_batch_size,
            "val_batch_size": val_batch_size,
            "test_batch_size": test_batch_size,
        })

        self.dataset = dataset.evaluate()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size

    # ----------------------------------------------------------------------- #
    # Public methods
    # ----------------------------------------------------------------------- #

    @override
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset.train_events,
            batch_size=self.train_batch_size,
            fixed_negatives=False,
        )

    @override
    def val_dataloader(self) -> Any:
        return DataLoader(
            self.dataset.val_events,
            batch_size=self.val_batch_size,
            fixed_negatives=True,
        )

    @override
    def test_dataloader(self) -> Any:
        return DataLoader(
            self.dataset.test_events,
            batch_size=self.test_batch_size,
            fixed_negatives=True,
        )

    @override
    def transfer_batch_to_device(
        self,
        batch: tuple[Events, Events],
        device: torch.device,
        dataloader_idx: int,
    ) -> tuple[Events, Events]:
        return (
            batch[0].to(device, non_blocking=True),
            batch[1].to(device, non_blocking=True),
        )
