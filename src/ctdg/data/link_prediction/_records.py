# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
from typing import Annotated, Any, Generic, TypeVar

import numpy.typing as npt
import torch
from torch import Tensor

_T = TypeVar("_T", npt.NDArray[Any], Tensor)


@dataclasses.dataclass(frozen=True)
class Data(Generic[_T]):
    """The extracted data."""

    node_features: Annotated[_T, "E", float]
    src_nodes: Annotated[_T, "E", int]
    dst_nodes: Annotated[_T, "E", int]
    timestamps: Annotated[_T, "E", float]
    edge_labels: Annotated[_T, "E", int]
    edge_features: Annotated[_T, "E", float]

    def to_dict(self) -> dict[str, _T]:
        """Converts the object to a dictionary."""
        return dataclasses.asdict(self)

    def to_tensor(
        self,
        *,
        float_dtype: torch.dtype = torch.float32,
        int_dtype: torch.dtype = torch.long,
    ) -> Data[Tensor]:
        """Converts the object to a tensor."""
        return Data(
            node_features=torch.as_tensor(self.node_features, dtype=float_dtype),
            src_nodes=torch.as_tensor(self.src_nodes, dtype=int_dtype),
            dst_nodes=torch.as_tensor(self.dst_nodes, dtype=int_dtype),
            timestamps=torch.as_tensor(self.timestamps, dtype=float_dtype),
            edge_labels=torch.as_tensor(self.edge_labels, dtype=int_dtype),
            edge_features=torch.as_tensor(self.edge_features, dtype=float_dtype),
        )


@dataclasses.dataclass(frozen=True)
class Splits(Generic[_T]):
    """The created splits."""

    train_mask: Annotated[_T, "E", bool]
    val_mask: Annotated[_T, "E", bool]
    test_mask: Annotated[_T, "E", bool]
    inductive_val_mask: Annotated[_T, "E", bool]
    inductive_test_mask: Annotated[_T, "E", bool]

    def to_dict(self) -> dict[str, _T]:
        """Converts the object to a dictionary."""
        return dataclasses.asdict(self)

    def to_tensor(self) -> Splits[Tensor]:
        """Converts the object to a tensor."""
        data = {k: torch.as_tensor(v) for k, v in self.to_dict().items()}
        return Splits(**data)
