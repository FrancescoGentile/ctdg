# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

from ctdg.data.information_cascade import FOSTERDataset
from ctdg.utils import LazyCall as L


def get_dataset(name: str) -> tuple[L[FOSTERDataset], int, int]:
    """Returns the dataset and the dimensions of the node and event features."""
    if name not in ["douban", "memetracker", "twitter"]:
        msg = f"Unknown dataset: {name}"
        raise ValueError(msg)

    return L(FOSTERDataset)(f"data/{name}", name, download=True), 0, 0
