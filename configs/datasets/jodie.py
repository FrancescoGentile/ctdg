# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

from ctdg.data.link_prediction import JODIEDataset
from ctdg.utils import LazyCall as L

_DIMS = {
    "wikipedia": (0, 172),
    "reddit": (0, 172),
    "mooc": (0, 4),
    "lastfm": (0, 4),
}


def get_dataset(name: str) -> tuple[L[JODIEDataset], int, int]:
    """Returns the dataset and the dimensions of the node and event features."""
    if name not in _DIMS:
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

    nodes_dim, events_dim = _DIMS[name]

    return dataset, nodes_dim, events_dim
