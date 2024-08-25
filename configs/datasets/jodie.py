# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

from ctdg.data.link_prediction import JODIEDataset
from ctdg.utils import LazyCall as L


def get_dataset(name: str) -> tuple[L[JODIEDataset], int, int]:
    """Returns the dataset and the dimensions of the node and event features."""
    if name != "wikipedia":
        msg = f"Unknown dataset: {name}"
        raise ValueError(msg)

    dataset = L(JODIEDataset)(
        path="data/wikipedia",
        name="Wikipedia",
        val_ratio=0.15,
        test_ratio=0.15,
        inductive_ratio=0.1,
        seed=2020,
    )

    return dataset, 172, 172
