# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator

import torch
from torch import Tensor
from typing_extensions import Self

from ._stream import Stream


class Events(Stream):
    """A stream of events."""

    src_nodes: Tensor
    dst_nodes: Tensor
    timestamps: Tensor
    indices: Tensor

    # ----------------------------------------------------------------------- #
    # Initialization
    # ----------------------------------------------------------------------- #

    @classmethod
    def empty(cls, device: str | torch.device | None = None) -> Self:
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

    def flip_direction(self) -> Self:
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

    # ----------------------------------------------------------------------- #
    # Magic methods
    # ----------------------------------------------------------------------- #

    def __iter__(self) -> Iterator[Tensor]:
        """Returns an iterator over the fields."""
        return iter((self.src_nodes, self.dst_nodes, self.timestamps, self.indices))
