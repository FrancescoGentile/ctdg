# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

"""Datasets and utilities for link prediction."""

from ._dataloader import DataLoader
from ._dataset import Dataset
from ._jodie import JODIEDataset
from ._lightning import DataModule
from ._records import Data

__all__ = [
    "DataLoader",
    "Dataset",
    "JODIEDataset",
    "DataModule",
    "Data",
]
