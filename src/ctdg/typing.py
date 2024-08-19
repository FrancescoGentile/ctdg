# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

"""Typing utilities."""

from pathlib import Path
from typing import TypeAlias

import torch
from typing_extensions import Protocol, Self, runtime_checkable

__all__ = ["PathLike", "Moveable"]

PathLike: TypeAlias = str | Path
"""Type alias for a path-like object."""


@runtime_checkable
class Moveable(Protocol):
    """Protocol for objects that can be moved to a device.

    This protocol offers a default implementation for the `to` method that moves all
    the attributes of the object that implement themselves such protocol.
    """

    def to(self, device: str | torch.device, *, non_blocking: bool = False) -> Self:
        """Moves the object to the specified device."""
        for key, value in self.__dict__.items():
            if isinstance(value, Moveable):
                self.__dict__[key] = value.to(device, non_blocking=non_blocking)

        return self
