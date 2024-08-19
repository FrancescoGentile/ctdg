# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Annotated

import torch
from torch import Tensor
from typing_extensions import Self

from ctdg.typing import Moveable

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


@dataclasses.dataclass(frozen=True)
class Events(Moveable):
    """A stream of events."""

    src_nodes: Annotated[Tensor, "E", torch.long]
    dst_nodes: Annotated[Tensor, "E", torch.long]
    timestamps: Annotated[Tensor, "E", torch.float32]
    indices: Annotated[Tensor, "E", torch.long]

    # ----------------------------------------------------------------------- #
    # Initialization
    # ----------------------------------------------------------------------- #

    def __post_init__(self) -> None:
        if any(
            len(self.src_nodes) != len(x)
            for x in (self.dst_nodes, self.timestamps, self.indices)
        ):
            msg = "The number of events must be the same for all fields."
            raise ValueError(msg)

    @classmethod
    def cat(cls, events: Sequence[Events]) -> Events:
        """Concatenates a sequence of events."""
        return cls(
            src_nodes=torch.cat([e.src_nodes for e in events], dim=0),
            dst_nodes=torch.cat([e.dst_nodes for e in events], dim=0),
            timestamps=torch.cat([e.timestamps for e in events], dim=0),
            indices=torch.cat([e.indices for e in events], dim=0),
        )

    @classmethod
    def empty(cls, device: str | torch.device | None = None) -> Events:
        """Creates an empty stream of events."""
        return cls(
            src_nodes=torch.empty(0, dtype=torch.long, device=device),
            dst_nodes=torch.empty(0, dtype=torch.long, device=device),
            timestamps=torch.empty(0, dtype=torch.long, device=device),
            indices=torch.empty(0, dtype=torch.long, device=device),
        )

    # ----------------------------------------------------------------------- #
    # Public methods
    # ----------------------------------------------------------------------- #

    def flip_direction(self) -> Events:
        """Swaps the source and destination nodes."""
        return self.__class__(
            src_nodes=self.dst_nodes,
            dst_nodes=self.src_nodes,
            timestamps=self.timestamps,
            indices=self.indices,
        )

    def sort(self, *, descending: bool = False) -> Self:
        """Sorts the events by timestamp."""
        order = self.timestamps.argsort(descending=descending)
        return self[order]

    def repeat(self, n: int) -> Events:
        """Repeats the stream of events."""
        return self.__class__(
            src_nodes=self.src_nodes.repeat(n),
            dst_nodes=self.dst_nodes.repeat(n),
            timestamps=self.timestamps.repeat(n),
            indices=self.indices.repeat(n),
        )

    def clone(self) -> Events:
        """Clones the stream of events."""
        return self.__class__(
            src_nodes=self.src_nodes.clone(),
            dst_nodes=self.dst_nodes.clone(),
            timestamps=self.timestamps.clone(),
            indices=self.indices.clone(),
        )

    # ----------------------------------------------------------------------- #
    # Magic methods
    # ----------------------------------------------------------------------- #

    def __len__(self) -> int:
        """Returns the number of events in the stream."""
        return len(self.src_nodes)

    def __getitem__(self, key: int | slice | Sequence[int] | Tensor) -> Self:
        """Returns a subset of the stream."""
        return self.__class__(
            src_nodes=self.src_nodes[key],
            dst_nodes=self.dst_nodes[key],
            timestamps=self.timestamps[key],
            indices=self.indices[key],
        )

    def __iter__(self) -> Iterator[Tensor]:
        """Returns an iterator over the fields."""
        return iter((self.src_nodes, self.dst_nodes, self.timestamps, self.indices))
