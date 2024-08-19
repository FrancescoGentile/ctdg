# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

"""Datasets and utilities for link prediction."""

from ._dataset import Dataset
from ._jodie import JODIEDataset
from ._lightning import DataModule
from ._loader import DataLoader
from ._records import Data

__all__ = [
    "Dataset",
    "JODIEDataset",
    "DataModule",
    "DataLoader",
    "Data",
]
