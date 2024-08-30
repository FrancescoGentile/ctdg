# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

import inspect
import typing
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor
from typing_extensions import Self, dataclass_transform


@dataclass_transform(eq_default=False)
class Stream:
    """A class to represent a stream of tensors."""

    # ----------------------------------------------------------------------- #
    # Constructor and Factory methods
    # ----------------------------------------------------------------------- #

    def __init_subclass__(cls, *, init: bool = True) -> None:
        annotations = inspect.get_annotations(cls)
        if len(annotations) == 0:
            msg = f"Class {cls} must have at least one attribute with type "
            raise TypeError(msg)

        # check that all class attributes are of type torch.Tensor
        for attr, type_ in annotations.items():
            # check if the attributes is typing.Annotated[Tensor, ...]
            if typing.get_origin(type_) is not None:
                type_ = typing.get_args(type_)[0]  # noqa: PLW2901

            if not issubclass(type_, Tensor):
                msg = f"Attribute {attr} must be of type {Tensor}, got {type_}."
                raise TypeError(msg)

        if init:
            cls.__init__ = _create_init(annotations)

    @classmethod
    def cat(cls, streams: Sequence[Self]) -> Self:
        """Concatenates a sequence of streams."""
        if len(streams) == 0:
            msg = "Expected at least one stream, got 0."
            raise ValueError(msg)
        if len(streams) == 1:
            return streams[0]

        annotations = inspect.get_annotations(cls)
        attributes = {attr: [] for attr in annotations}
        for stream in streams:
            for attr in annotations:
                attributes[attr].append(getattr(stream, attr))

        attributes = {attr: torch.cat(values) for attr, values in attributes.items()}
        return cls(**attributes)

    @classmethod
    def empty(cls, device: str | torch.device | None = None) -> Self:
        """Creates an empty stream.

        !!! warning

            The default implementation of this method creates all 1D tensors of size 0
            and the defeault torch.dtype. If each attribute has different requirements,
            you should override this method in the subclass.

        Args:
            device: The device where to create the tensors.

        Returns:
            A new instance of the stream with all attributes set to empty tensors
            of size 0.
        """
        annotations = inspect.get_annotations(cls)
        attributes = {attr: torch.empty(0, device=device) for attr in annotations}
        return cls(**attributes)

    # ----------------------------------------------------------------------- #
    # Public methods
    # ----------------------------------------------------------------------- #

    def repeat(self, repeats: int) -> Self:
        """Repeats the stream `repeats` times."""
        annotations = inspect.get_annotations(self.__class__)
        attributes = {attr: getattr(self, attr).repeat(repeats) for attr in annotations}
        return self.__class__(**attributes)

    def to(self, device: str | torch.device, *, non_blocking: bool = False) -> Self:
        """Moves the stream to the specified device."""
        annotations = inspect.get_annotations(self.__class__)
        attributes = {
            attr: getattr(self, attr).to(device, non_blocking=non_blocking)
            for attr in annotations
        }
        return self.__class__(**attributes)

    def clone(self) -> Self:
        """Clones the stream."""
        annotations = inspect.get_annotations(self.__class__)
        attributes = {attr: getattr(self, attr).clone() for attr in annotations}
        return self.__class__(**attributes)

    # ----------------------------------------------------------------------- #
    # Magic methods
    # ----------------------------------------------------------------------- #

    def __len__(self) -> int:
        """Returns the length of the stream."""
        annotations = inspect.get_annotations(self.__class__)
        return getattr(self, next(iter(annotations))).size(0)

    def __getitem__(self, index: slice | Sequence[int] | Tensor) -> Self:
        """Returns a subset of the stream."""
        annotations = inspect.get_annotations(self.__class__)
        attributes = {attr: getattr(self, attr) for attr in annotations}
        attributes = {attr: value[index] for attr, value in attributes.items()}
        return self.__class__(**attributes)

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        """Deepcopy implementation."""
        return self.clone()

    def __repr__(self) -> str:
        """String representation of the stream."""
        annotations = inspect.get_annotations(self.__class__)
        attributes = {attr: getattr(self, attr) for attr in annotations}
        return f"{self.__class__.__name__}({attributes})"


# ----------------------------------------------------------------------- #
# Public methods
# ----------------------------------------------------------------------- #

# ruff: noqa: ANN001, N807


def _create_init(annotations: dict[str, type[Tensor]]) -> Any:
    """Creates the __init__ method for a Stream subclass."""

    def __init__(self: Stream, *args: Tensor, **kwargs: Tensor) -> None:
        if len(args) > len(annotations):
            msg = f"Expected at most {len(annotations)} arguments, got {len(args)}."
            raise TypeError(msg)

        for attr, value in zip(annotations, args, strict=False):
            if attr in kwargs:
                msg = f"Multiple values for argument {attr}."
                raise TypeError(msg)

            kwargs[attr] = value

        if len(kwargs) != len(annotations):
            missing = set(annotations) - set(kwargs)
            msg = f"Missing required arguments: {missing}."
            raise TypeError(msg)

        stream_size: int | None = None
        for attr, value in kwargs.items():
            if stream_size is None:
                stream_size = value.size(0)
            elif value.size(0) != stream_size:
                msg = (
                    f"Expected all tensors to have the same size along the first "
                    f"dimension, got {value.size(0)} for attribute {attr}, "
                    f"expected {stream_size}."
                )
                raise ValueError(msg)

            setattr(self, attr, value)

    return __init__
