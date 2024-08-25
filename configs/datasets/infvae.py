# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

from ctdg.data.information_cascade import InfVAEDataset
from ctdg.utils import LazyCall as L


def get_dataset(name: str) -> tuple[L[InfVAEDataset], int, int]:
    """Returns the dataset and the dimensions of the node and event features."""
    if name not in ["android", "christianity"]:
        msg = f"Unknown dataset: {name}"
        raise ValueError(msg)

    return L(InfVAEDataset)(f"data/{name}", name, download=True), 0, 0
