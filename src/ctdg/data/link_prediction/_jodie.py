# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

import json
import random
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
import numpy.typing as npt
import requests
import torch
from torch import Tensor

from ctdg.data import Events
from ctdg.typing import PathLike

from ._dataset import Dataset


class JODIEDataset(Dataset):
    """Temporal graph datasets from the JODIE paper."""

    def __init__(
        self,
        path: PathLike,
        name: Literal["reddit", "wikipedia", "mooc", "lastfm"],
        *,
        download: bool = True,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        inductive_ratio: float = 0.1,
        seed: int = 2020,
    ) -> None:
        path = Path(path)
        if not (path / "data.h5").exists():
            if not download:
                msg = f"Dataset not found at {path}."
                raise FileNotFoundError(msg)

            self.download(path, name)

        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.inductive_ratio = inductive_ratio
        self.seed = seed

        data = h5py.File(path / "data.h5", "r")
        data = {key: np.array(value) for key, value in data.items()}
        splits = _create_splits(
            data,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            inductive_ratio=inductive_ratio,
            seed=seed,
        )

        self.data = {key: torch.from_numpy(value) for key, value in data.items()}
        for key, value in self.data.items():
            if value.is_floating_point():
                self.data[key] = value.float()

        self.splits = {key: torch.from_numpy(value) for key, value in splits.items()}

        with (path / "statistics.json").open("r") as f:
            self.statistics = json.load(f)

    # ----------------------------------------------------------------------- #
    # Properties
    # ----------------------------------------------------------------------- #

    @property
    def num_nodes(self) -> int:
        """Gets the number of nodes."""
        return self.statistics["num_nodes"]

    @property
    def nodes_features(self) -> Tensor | None:
        """Gets the node features."""
        return None

    @property
    def events_features(self) -> Tensor | None:
        """Gets the event features."""
        return self.data.get("edge_features")

    @property
    def events(self) -> Events:
        """Gets the full event stream."""
        indices = torch.arange(len(self.data["src_nodes"]), dtype=torch.long)
        return Events(
            src_nodes=self.data["src_nodes"],
            dst_nodes=self.data["dst_nodes"],
            timestamps=self.data["timestamps"],
            indices=indices,
        )

    @property
    def train_events(self) -> Events:
        """Gets the training event stream."""
        return self.events[self.splits["train_mask"]]

    @property
    def val_events(self) -> Events:
        """Gets the validation event stream."""
        return self.events[self.splits["val_mask"]]

    @property
    def test_events(self) -> Events:
        """Gets the test event stream."""
        return self.events[self.splits["test_mask"]]

    # ----------------------------------------------------------------------- #
    # Public methods
    # ----------------------------------------------------------------------- #

    @staticmethod
    def download(
        path: PathLike,
        name: Literal["reddit", "wikipedia", "mooc", "lastfm"],
        *,
        force: bool = False,
    ) -> None:
        """Download the dataset."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if not force and (path / "data.h5").exists():
            return

        num_nodes, data = _download(name)
        with h5py.File(path / "data.h5", "w") as f:
            for key, value in data.items():
                f.create_dataset(key, data=value)

        statistics = {
            "num_nodes": num_nodes,
            "num_events": len(data["src_nodes"]),
        }

        with (path / "statistics.json").open("w") as f:
            json.dump(statistics, f)


# ----------------------------------------------------------------------- #
# Helper functions
# ----------------------------------------------------------------------- #


def _download(name: str) -> tuple[int, dict[str, npt.NDArray[Any]]]:
    with tempfile.NamedTemporaryFile(delete=True) as file:
        url = f"https://zenodo.org/record/7213796/files/{name}.zip"
        r = requests.get(url, allow_redirects=True, timeout=10)
        file.write(r.content)

        with tempfile.TemporaryDirectory() as zip_dir:
            zip_dir = Path(zip_dir)  # noqa: PLW2901
            data_file = f"{name}/{name}.csv"
            if name != "reddit":
                with zipfile.ZipFile(file.name, "r") as zip_ref:
                    zip_ref.extract(data_file, zip_dir)
            else:
                try:
                    # The reddit zip file is compressed with type 9, which is not
                    # supported by the zipfile module. So we try to extract it using
                    # the unzip command line tool.
                    out_dir = zip_dir / name
                    command = ["unzip", "-q", "-j", file.name, data_file, "-d", out_dir]
                    subprocess.run(command, check=True)  # noqa: S603
                except subprocess.CalledProcessError:
                    msg = "Failed to extract the Reddit dataset. Please install unzip."
                    raise RuntimeError(msg) from None

            data_file = zip_dir / name / f"{name}.csv"
            return _parse_file(data_file)


def _parse_file(file: Path) -> tuple[int, dict[str, npt.NDArray[Any]]]:
    sources = []
    destinations = []
    timestamps = []
    labels = []
    edge_features = []

    with file.open("r") as f:
        next(f)

        last_timestamp = -1
        for idx, line in enumerate(f):
            elems = line.strip().split(",")

            t = float(elems[2])
            if t < last_timestamp:
                msg = f"Timestamps are not sorted at line {idx + 2}."
                raise RuntimeError(msg)

            sources.append(int(elems[0]))
            destinations.append(int(elems[1]))
            timestamps.append(t)
            labels.append(int(elems[3]))
            edge_features.append(list(map(float, elems[4:])))

    sources = np.array(sources)
    destinations = np.array(destinations)
    timestamps = np.array(timestamps)
    labels = np.array(labels)
    edge_features = np.array(edge_features)
    if not edge_features.any():
        # lastfm and mooc have all featuresset to 0, so we can safely ignore them
        edge_features = None

    # The graph is bipartite between users and items (e.g., wikipedia pages,
    # subreddits). Both type of nodes start from 0 in the csv file, so we need to
    # shift the destination nodes by the number of source nodes to make sure that
    # each node has a unique identifier.
    max_src = sources.max()
    destinations += max_src + 1

    num_nodes = destinations.max() + 1

    data = {
        "src_nodes": sources,
        "dst_nodes": destinations,
        "timestamps": timestamps,
        "edge_labels": labels,
    }
    if edge_features is not None:
        data["edge_features"] = edge_features

    return int(num_nodes), data


def _create_splits(
    data: dict[str, npt.NDArray[Any]],
    val_ratio: float,
    test_ratio: float,
    inductive_ratio: float,
    seed: int,
) -> dict[str, npt.NDArray[Any]]:
    old_random_state = random.getstate()
    random.seed(seed)

    nodes = set(data["src_nodes"]) | set(data["dst_nodes"])
    num_nodes = len(nodes)

    val_time, test_time = np.quantile(
        data["timestamps"],
        [1 - val_ratio - test_ratio, 1 - test_ratio],
    )

    mask = data["timestamps"] > val_time
    val_test_nodes = set(data["src_nodes"][mask]) | set(data["dst_nodes"][mask])

    num_ind_nodes = int(inductive_ratio * num_nodes)
    ind_nodes = random.sample(list(val_test_nodes), num_ind_nodes)

    ind_src_nodes = np.isin(data["src_nodes"], ind_nodes)
    ind_dst_nodes = np.isin(data["dst_nodes"], ind_nodes)
    ind_edge_mask = ind_src_nodes | ind_dst_nodes

    train_mask = (data["timestamps"] <= val_time) & ~ind_edge_mask
    train_nodes = set(data["src_nodes"][train_mask])
    train_nodes |= set(data["dst_nodes"][train_mask])
    if train_nodes.intersection(ind_nodes):
        msg = "If you see this message, please report it as a bug."
        raise RuntimeError(msg)

    ind_nodes = list(nodes - train_nodes)

    val_mask = (data["timestamps"] > val_time) & (data["timestamps"] <= test_time)
    test_mask = data["timestamps"] > test_time

    ind_src_nodes = np.isin(data["src_nodes"], ind_nodes)
    ind_dst_nodes = np.isin(data["dst_nodes"], ind_nodes)
    ind_edge_mask = ind_src_nodes | ind_dst_nodes

    ind_val_mask = val_mask & ind_edge_mask
    ind_test_mask = test_mask & ind_edge_mask

    random.setstate(old_random_state)

    return {
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "inductive_val_mask": ind_val_mask,
        "inductive_test_mask": ind_test_mask,
    }
