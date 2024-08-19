# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from typing_extensions import override

from ctdg.typing import PathLike

from ._dataset import Dataset
from ._records import Data


class JODIEDataset(Dataset):
    """Temporal graph datasets from the JODIE paper."""

    def __init__(
        self,
        path: PathLike,
        name: Literal["Reddit", "Wikipedia", "MOOC", "LastFM"],
        *,
        download: bool = True,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        inductive_ratio: float = 0.1,
        seed: int = 2020,
    ) -> None:
        self.name = name.lower()
        if self.name not in ["reddit", "wikipedia", "mooc", "lastfm"]:
            msg = (
                f"Invalid dataset name '{name}'. "
                "Choose from 'Reddit', 'Wikipedia', 'MOOC', 'LastFM'."
            )
            raise ValueError(msg)

        super().__init__(
            path,
            download=download,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            inductive_ratio=inductive_ratio,
            seed=seed,
        )

    # ----------------------------------------------------------------------- #
    # Protected methods
    # ----------------------------------------------------------------------- #

    @override
    def _get_name(self) -> str:
        return self.name

    @override
    def _get_url(self) -> str:
        # These are the URLs to the datasets provided by the authors of the EdgeBank
        # paper ("Towards Better Evaluation for Dynamic Link Prediction").
        # These zip files contain both the original csv file and the preprocessed
        # files used in the EdgeBank paper. Here, we ignore the preprocessed files
        # and extract the data from the original csv file.
        # Wedo not download the csv file from the link provided by the authors of the
        # JODIE paper because the download is extremely slow (if you want to try, you
        # the link is: # return http://snap.stanford.edu/jodie/{self.name}.zip).
        return f"https://zenodo.org/records/7213796/files/{self.name}.zip"

    @override
    def _extract(self, file: Path) -> Data[npt.NDArray[Any]]:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)  # noqa: PLW2901
            csv_file = f"{self.name}/{self.name}.csv"
            if self.name != "reddit":
                with zipfile.ZipFile(file, "r") as zip_ref:
                    zip_ref.extract(csv_file, tmp_dir)
            else:
                try:
                    # The reddit zip file is compressed with type 9, which is not
                    # supported by the zipfile module. So we try to extract it using
                    # the unzip command line tool.
                    out_dir = tmp_dir / self.name
                    command = ["unzip", "-q", "-j", file, csv_file, "-d", out_dir]
                    subprocess.run(command, check=True)  # noqa: S603
                except subprocess.CalledProcessError:
                    msg = "Failed to extract the Reddit dataset. Please install unzip."
                    raise RuntimeError(msg) from None

            csv_file = Path(tmp_dir) / self.name / f"{self.name}.csv"
            with csv_file.open("r") as f:
                next(f)

                sources = []
                destinations = []
                timestamps = []
                labels = []
                edge_features = []

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

        # The graph is bipartite between users and items (e.g., wikipedia pages,
        # subreddits). Both type of nodes start from 0 in the csv file, so we need to
        # shift the destination nodes by the number of source nodes to make sure that
        # each node has a unique identifier.
        max_src = sources.max()
        destinations += max_src + 1

        num_nodes = destinations.max() + 1
        node_features = np.zeros((num_nodes, edge_features.shape[1]))

        return Data(
            node_features=node_features,
            src_nodes=sources,
            dst_nodes=destinations,
            timestamps=timestamps,
            edge_labels=labels,
            edge_features=edge_features,
        )
