# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

import csv
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import requests
import torch
from torch import Tensor

from ctdg.data import Events
from ctdg.typing import PathLike


class InfVAEDataset:
    """Datasets from the InfVAE paper."""

    def __init__(
        self,
        path: PathLike,
        name: Literal["android", "christianity"],
        *,
        download: bool = True,
    ) -> None:
        """Initializes the dataset.

        Args:
            path: The path to the dataset. This should be the directory where the
                dataset is stored or should be stored.
            name: The name of the dataset to load.
            download: Whether to download the dataset if it is not found at the given
                path.
        """
        path = Path(path)
        splits = ["train", "val", "test"]
        if not all((path / f"{split}.h5").exists() for split in splits):
            if not download:
                msg = f"Dataset not found at {path}."
                raise FileNotFoundError(msg)

            self.download(path, name)

        self._events: dict[str, Events] = {}
        for split in splits:
            with h5py.File(path / f"{split}.h5", "r") as f:
                data = {str(key): torch.from_numpy(np.array(f[key])) for key in f}
                data["timestamps"] = data["timestamps"].float()
                self._events[split] = Events(**data)

        with (path / "statistics.csv").open() as f:
            reader = csv.DictReader(f)
            self._statistics = next(reader)

    # ----------------------------------------------------------------------- #
    # Properties
    # ----------------------------------------------------------------------- #

    @property
    def num_nodes(self) -> int:
        """The number of nodes in the dataset."""
        return self.num_sources + self.num_destinations

    @property
    def num_sources(self) -> int:
        """The number of sources in the dataset."""
        return int(self._statistics["num_sources"])

    @property
    def num_destinations(self) -> int:
        """The number of destinations in the dataset."""
        return int(self._statistics["num_destinations"])

    @property
    def nodes_features(self) -> Tensor | None:
        """Returns the nodes features."""
        return None

    @property
    def events_features(self) -> Tensor | None:
        """Returns the events features."""
        return None

    @property
    def train_events(self) -> Events:
        """Returns the stream of training events."""
        return self._events["train"]

    @property
    def val_events(self) -> Events:
        """Returns the stream of validation events."""
        return self._events["val"]

    @property
    def test_events(self) -> Events:
        """Returns the stream of testing events."""
        return self._events["test"]

    # ----------------------------------------------------------------------- #
    # Public methods
    # ----------------------------------------------------------------------- #

    @staticmethod
    def download(path: PathLike, name: Literal["android", "christianity"]) -> None:
        """Downloads the dataset.

        Args:
            path: The path to the directory where the dataset should be stored.
            name: The name of the dataset.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        splits = ["train", "val", "test"]
        if all((path / f"{split}.h5").exists() for split in splits):
            return

        cascades = {split: _download_cascades(name, split) for split in splits}

        nodes_to_idx: dict[int, int] = {}
        for split in splits:
            for cascade in cascades[split]:
                for node, _ in cascade:
                    if node not in nodes_to_idx:
                        nodes_to_idx[node] = len(nodes_to_idx)

        statistics = {}

        start = 0
        destinations_start = len(nodes_to_idx)
        for split in splits:
            events: list[tuple[int, int, float]] = []
            for cascade in cascades[split]:
                # create the destination node
                dst_node = destinations_start
                destinations_start += 1

                for src_node, ts in cascade:
                    events.append((nodes_to_idx[src_node], dst_node, ts))

            statistics[f"{split}_events"] = len(events)

            events = sorted(events, key=lambda x: x[2])
            src_nodes = [src_node for src_node, _, _ in events]
            dst_nodes = [dst_node for _, dst_node, _ in events]
            timestamps = [ts for _, _, ts in events]

            with h5py.File(path / f"{split}.h5", "w") as f:
                f.create_dataset("src_nodes", data=src_nodes)
                f.create_dataset("dst_nodes", data=dst_nodes)
                f.create_dataset("timestamps", data=timestamps)
                f.create_dataset("indices", data=np.arange(start, start + len(events)))

        statistics["num_sources"] = len(nodes_to_idx)
        statistics["num_destinations"] = destinations_start - len(nodes_to_idx)

        with (path / "statistics.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=statistics.keys())
            writer.writeheader()
            writer.writerow(statistics)


# ----------------------------------------------------------------------- #
# Private API
# ----------------------------------------------------------------------- #

BASE = "https://raw.githubusercontent.com/aravindsankar28/Inf-VAE/master/data"


def _download_cascades(name: str, split: str) -> list[list[tuple[int, float]]]:
    """Downloads the cascades of the dataset."""
    url = f"{BASE}/{name}/{split}.txt"
    data = requests.get(url, timeout=10).text

    cascades: list[list[tuple[int, float]]] = []
    for line in data.splitlines():
        line = line.strip()  # noqa: PLW2901
        if not line:
            continue

        chunks = line.split()
        nodes = [int(node) for node in chunks[1::2]]
        timestamps = [float(ts) for ts in chunks[2::2]]

        cascade = list(zip(nodes, timestamps, strict=True))
        cascades.append(cascade)

    return cascades
