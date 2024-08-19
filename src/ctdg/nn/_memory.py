# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import torch
from torch import Tensor

from ._module import Module

if TYPE_CHECKING:
    from collections.abc import Iterator


class MemoryUpdater(Protocol):
    """Interface for memory updaters.

    Memory updaters are responsible for updating the memory of the nodes in the graph
    given a message. Examples of memory updaters include the `torch.nn.RNNCell` and
    `torch.nn.GRUCell` modules.
    """

    def __call__(self, msg: Tensor, memory: Tensor) -> Tensor: ...


class Memory(Module):
    """A module to store and update the memory of the nodes in the graph."""

    def __init__(
        self,
        num_nodes: int,
        memory_dim: int,
        device: torch.device | str | None = None,
    ) -> None:
        """Initializes the memory module.

        Args:
            num_nodes: The number of nodes in the graph.
            memory_dim: The dimension of the memory.
            device: The device on which to store the memory.
        """
        super().__init__()

        self.register_buffer("memory", None)
        self.memory = torch.zeros(num_nodes, memory_dim, device=device)

        self.register_buffer("last_update", None)
        self.last_update = torch.full((num_nodes,), -1.0, device=device)

    # ----------------------------------------------------------------------- #
    # Properties
    # ----------------------------------------------------------------------- #

    @property
    def dim(self) -> int:
        """The feature dimension of the memory."""
        return self.memory.shape[-1]

    # ----------------------------------------------------------------------- #
    # Public methods
    # ----------------------------------------------------------------------- #

    def reset(self) -> None:
        """Resets the memory and last update tensors.

        This method can be used at the beginning of each epoch to reset the memory
        and last update tensors to their initial values. The memory tensor is set to
        zeros, and the last update tensor is set to negative ones (thus, you can
        check whether a node has been updated by checking if the last update is
        non-negative).
        """
        self.memory.zero_()
        self.last_update.fill_(-1.0)

    def detach(self) -> None:
        """Detaches the memory tensor from the computation graph."""
        self.memory.detach_()

    # ----------------------------------------------------------------------- #
    # Magic methods
    # ----------------------------------------------------------------

    def __getitem__(self, idx: Tensor) -> tuple[Tensor, Tensor]:
        """Returns the memory and last update of the nodes with the given indices."""
        return self.memory[idx], self.last_update[idx]

    def __setitem__(self, idx: Tensor, value: tuple[Tensor, Tensor]) -> None:
        """Sets the memory and last update of the nodes with the given indices."""
        memory, last_update = value
        # memory is stored in float32, while the new memory may be in float16 if
        # half precision is used at training time
        self.memory[idx] = memory.to(self.memory)
        self.last_update[idx] = last_update

    def __iter__(self) -> Iterator[Tensor]:
        """Iterates over the memory and last update tensors."""
        return iter((self.memory, self.last_update))

    def __len__(self) -> int:
        """Returns the number of nodes in the memory."""
        return len(self.memory)
