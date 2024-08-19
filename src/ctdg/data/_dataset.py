# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

import abc
import logging
import random
import tempfile
from pathlib import Path
from typing import Annotated, Any

import h5py
import numpy as np
import numpy.typing as npt
import requests
import torch
from torch import Tensor

from ctdg.structures import Events
from ctdg.typing import PathLike

from ._records import Data, Splits

_logger = logging.getLogger(__name__)


class Dataset(abc.ABC):
    """Dataset for link prediction."""

    def __init__(
        self,
        path: PathLike,
        *,
        download: bool = True,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        inductive_ratio: float = 0.1,
        seed: int = 2020,
    ) -> None:
        """Initializes the dataset.

        Args:
            path: The path to the dataset.
            download: Whether to download the dataset if it is not already downloaded.
            val_ratio: The ratio of validation edges.
            test_ratio: The ratio of test edges.
            inductive_ratio: The ratio of inductive nodes.
            seed: The random seed used to create the splits.

        Raises:
            FileNotFoundError: If the dataset is not found at the specified path.
        """
        path = Path(path)
        if download:
            self._download(path)

        data_file = path / "data.h5"
        if not data_file.exists():
            msg = f"The dataset {self._get_name()} was not found at {path}."
            raise FileNotFoundError(msg)

        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.inductive_ratio = inductive_ratio
        self.seed = seed

        data = h5py.File(data_file, "r")
        data = Data(**{key: np.array(value) for key, value in data.items()})
        splits = self._create_splits(data)

        self._data = data.to_tensor()
        self._splits = splits.to_tensor()

    # ----------------------------------------------------------------------- #
    # Public properties
    # ----------------------------------------------------------------------- #

    @property
    def nodes_features(self) -> Annotated[Tensor, "N D", torch.float32]:
        """Gets the node features."""
        return self._data.node_features

    @property
    def events_features(self) -> Annotated[Tensor, "E D", torch.float32]:
        """Gets the event features."""
        return self._data.edge_features

    @property
    def events(self) -> Events:
        """Gets the full event stream."""
        indices = torch.arange(len(self._data.src_nodes), dtype=torch.long)
        return Events(
            src_nodes=self._data.src_nodes,
            dst_nodes=self._data.dst_nodes,
            timestamps=self._data.timestamps,
            indices=indices,
        )

    @property
    def train_events(self) -> Events:
        """Gets the training event stream."""
        return self.events[self._splits.train_mask]

    @property
    def val_events(self) -> Events:
        """Gets the validation event stream."""
        return self.events[self._splits.val_mask]

    @property
    def test_events(self) -> Events:
        """Gets the test event stream."""
        return self.events[self._splits.test_mask]

    # ----------------------------------------------------------------------- #
    # Private methods
    # ----------------------------------------------------------------------- #

    def _download(self, path: Path) -> None:
        """Downloads the dataset to the specified path."""
        path.mkdir(parents=True, exist_ok=True)

        output_file = path / "data.h5"

        if output_file.exists():
            _logger.info("The %s dataset is already downloaded.", self._get_name())
            return

        _logger.info("Downloading the %s dataset to %s...", self._get_name(), path)

        with tempfile.NamedTemporaryFile(delete=True) as file:
            r = requests.get(self._get_url(), allow_redirects=True, timeout=10)
            file.write(r.content)

            data = self._extract(Path(file.name))

        with h5py.File(output_file, "w") as f:
            for key, value in data.to_dict().items():
                f.create_dataset(key, data=value)

        _logger.info(
            "The %s dataset has been downloaded to %s.", self._get_name(), path
        )

    def _create_splits(self, data: Data[npt.NDArray[Any]]) -> Splits[npt.NDArray[Any]]:
        old_random_state = random.getstate()
        random.seed(self.seed)

        nodes = set(data.src_nodes) | set(data.dst_nodes)
        num_nodes = len(nodes)

        val_time, test_time = np.quantile(
            data.timestamps,
            [1 - self.val_ratio - self.test_ratio, 1 - self.test_ratio],
        )

        mask = data.timestamps > val_time
        val_test_nodes = set(data.src_nodes[mask]) | set(data.dst_nodes[mask])

        num_ind_nodes = int(self.inductive_ratio * num_nodes)
        ind_nodes = random.sample(list(val_test_nodes), num_ind_nodes)

        ind_src_nodes = np.isin(data.src_nodes, ind_nodes)
        ind_dst_nodes = np.isin(data.dst_nodes, ind_nodes)
        ind_edge_mask = ind_src_nodes | ind_dst_nodes

        train_mask = (data.timestamps <= val_time) & ~ind_edge_mask
        train_nodes = set(data.src_nodes[train_mask])
        train_nodes |= set(data.dst_nodes[train_mask])
        if train_nodes.intersection(ind_nodes):
            msg = "If you see this message, please report it as a bug."
            raise RuntimeError(msg)

        ind_nodes = list(nodes - train_nodes)

        val_mask = (data.timestamps > val_time) & (data.timestamps <= test_time)
        test_mask = data.timestamps > test_time

        ind_src_nodes = np.isin(data.src_nodes, ind_nodes)
        ind_dst_nodes = np.isin(data.dst_nodes, ind_nodes)
        ind_edge_mask = ind_src_nodes | ind_dst_nodes

        ind_val_mask = val_mask & ind_edge_mask
        ind_test_mask = test_mask & ind_edge_mask

        random.setstate(old_random_state)

        return Splits(
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            inductive_val_mask=ind_val_mask,
            inductive_test_mask=ind_test_mask,
        )

    # ----------------------------------------------------------------------- #
    # Protected abstract methods
    # ----------------------------------------------------------------------- #

    @abc.abstractmethod
    def _get_name(self) -> str:
        """Gets the name of the dataset."""
        ...

    @abc.abstractmethod
    def _get_url(self) -> str:
        """Gets the URL of the dataset."""
        ...

    @abc.abstractmethod
    def _extract(self, file: Path) -> Data[npt.NDArray[Any]]:
        """Extracts the dataset.

        Args:
            file: The path to the file downloaded from the URL.
        """
        ...
