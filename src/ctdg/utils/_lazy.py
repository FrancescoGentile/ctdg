# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0
# --------------------------------------------------------------------------- #
# Copyright (c) Facebook, Inc. and its affiliates.
# SPDX-License-Identifier: Apache-2.0
# --------------------------------------------------------------------------- #
# Modified from:
# https://github.com/facebookresearch/detectron2/blob/main/detectron2/config/lazy.py
# --------------------------------------------------------------------------- #

from collections.abc import Callable
from typing import Any, Generic, TypeVar, overload

from typing_extensions import Self

_T = TypeVar("_T")


class LazyCall(Generic[_T]):
    """Wrapper for a callable that is lazily evaluated."""

    def __init__(self, target: Callable[..., _T], *, cache: bool = False) -> None:
        """Initializes the object.

        Args:
            target: The callable to evaluate.
            cache: Whether to cache the result of the evaluation. If set to `True`,
                the callable will be evaluated only once and the result will be reused
                in subsequent calls.
        """
        if not callable(target):
            msg = f"{target} is not callable."
            raise TypeError(msg)

        self._target = target
        self._args: tuple[Any, ...] | None = None
        self._kwargs: dict[str, Any] | None = None

        self._cache = cache

    # ----------------------------------------------------------------------- #
    # Public properties
    # ----------------------------------------------------------------------- #

    @property
    def target(self) -> Callable[..., _T]:
        """The callable to evaluate."""
        return self._target

    # ----------------------------------------------------------------------- #
    # Public methods
    # ----------------------------------------------------------------------- #

    def evaluate(self, *args: Any, **kwargs: Any) -> _T:
        """Evaluates the wrapped callable.

        Args:
            *args: Positional arguments to pass to the callable. These will be appended
                to the arguments passed when the object was created.
            **kwargs: Keyword arguments to pass to the callable. These will be merged
                with the keyword arguments passed when the object was created
                (overwriting any existing keys).

        Returns:
            The result of the callable.
        """
        if self._args is None or self._kwargs is None:
            msg = "You must call the object before evaluating it."
            raise RuntimeError(msg)

        if self._cache and hasattr(self, "_result"):
            return self._result  # type: ignore

        args = _evaluate_sequence((*self._args, *args))
        kwargs = _evaluate_mapping({**self._kwargs, **kwargs})

        result = self._target(*args, **kwargs)
        if self._cache:
            self._result = result

        return result

    def get(self, key: int | str, default: Any = None) -> Any | None:
        """Gets a positional or keyword argument.

        Args:
            key: The index or key of the argument to get.
            default: The default value to return if the key does not exist.

        Returns:
            The argument if it exists, otherwise the default value.
        """
        try:
            return self[key]
        except (IndexError, KeyError):
            return default

    def to_dict(self) -> dict[str, Any]:
        """Converts the object to a dictionary."""
        res: dict[str, Any] = {"_target_": self._target}
        if self._args:
            res["_args_"] = _to_dict_sequence(self._args)
        if self._kwargs:
            res["_kwargs_"] = _to_dict_mapping(self._kwargs)

        return res

    # ----------------------------------------------------------------------- #
    # Magic methods
    # ----------------------------------------------------------------------- #

    def __call__(self, *args: Any, **kwargs: Any) -> Self:
        if self._args is not None or self._kwargs is not None:
            msg = "Cannot call a LazyCall object more than once."
            raise RuntimeError(msg)

        self._args = args
        self._kwargs = kwargs
        return self

    def __getitem__(self, key: int | str) -> Any:
        if self._args is None or self._kwargs is None:
            msg = "You must call the object before accessing its arguments."
            raise RuntimeError(msg)

        match key:
            case int():
                return self._args[key]
            case str():
                return self._kwargs[key]
            case _:  # type: ignore
                msg = "Key must be an integer or a string."
                raise TypeError(msg)

    def __setitem__(self, key: str, value: Any) -> None:
        if self._args is None or self._kwargs is None:
            msg = "You must call the object before accessing its arguments."
            raise RuntimeError(msg)

        self._kwargs[key] = value

    def __repr__(self) -> str:
        res = f"{self._target.__name__}("
        if self._args:
            res += ", ".join(map(repr, self._args))
            if self._kwargs:
                res += ", "
        if self._kwargs:
            res += ", ".join(f"{k}={v!r}" for k, v in self._kwargs.items())

        return res + ")"


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #


def _evaluate(arg: Any) -> Any:
    match arg:
        case LazyCall():
            return arg.evaluate()
        case list() | tuple():
            return _evaluate_sequence(arg)
        case dict():
            return _evaluate_mapping(arg)
        case _:
            return arg


@overload
def _evaluate_sequence(args: list[_T]) -> list[_T]: ...


@overload
def _evaluate_sequence(args: tuple[_T]) -> tuple[_T]: ...


def _evaluate_sequence(args: list[_T] | tuple[_T]) -> list[_T] | tuple[_T]:
    return args.__class__(_evaluate(arg) for arg in args)


def _evaluate_mapping(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {k: _evaluate(v) for k, v in kwargs.items()}


# --------------------------------------------------------------------------- #
# Serialization helpers
# --------------------------------------------------------------------------- #


def _to_dict(arg: Any) -> Any:
    match arg:
        case LazyCall():
            return arg.to_dict()
        case list() | tuple():
            return _to_dict_sequence(arg)
        case dict():
            return _to_dict_mapping(arg)
        case _:
            return arg


@overload
def _to_dict_sequence(args: list[_T]) -> list[_T]: ...


@overload
def _to_dict_sequence(args: tuple[_T]) -> tuple[_T]: ...


def _to_dict_sequence(args: list[_T] | tuple[_T]) -> list[_T] | tuple[_T]:
    return args.__class__(_to_dict(arg) for arg in args)


def _to_dict_mapping(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {k: _to_dict(v) for k, v in kwargs.items()}
