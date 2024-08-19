# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0
# --------------------------------------------------------------------------- #
# Copyright (c) Facebook, Inc. and its affiliates.
# SPDX-License-Identifier: Apache-2.0
# --------------------------------------------------------------------------- #
# Modified from:
# https://github.com/facebookresearch/detectron2/blob/main/detectron2/config/lazy.py
# --------------------------------------------------------------------------- #

import builtins
import contextlib
import importlib.util
from collections.abc import Generator, Mapping, Sequence
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from typing import Any

from ctdg.typing import PathLike


def load_config(path: PathLike) -> dict[str, Any]:
    """Loads a configuration file."""
    path = Path(path)
    if path.suffix != ".py":
        msg = f"Only python files are supported, got {path.suffix}."
        raise ValueError(msg)

    with _relative_imports():
        cfg = {
            "__file__": str(path),
            "__package__": _package_name(path),
        }
        exec(compile(path.read_text(), path, "exec"), cfg)  # noqa: S102

    return {k: v for k, v in cfg.items() if not k.startswith("__")}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


_PKG_PREFIX = "ctdg._config"


@contextlib.contextmanager
def _relative_imports() -> Generator[None, None, None]:
    """Context manager to allow relative imports in configuration files."""
    old_import = builtins.__import__

    def new_import(
        name: str,
        globals: Mapping[str, Any] | None = None,  # noqa: A002
        locals: Mapping[str, Any] | None = None,  # noqa: A002
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> ModuleType:
        globals = globals or {}  # noqa: A001
        if level != 0 and globals.get("__package__", "").startswith(_PKG_PREFIX):
            file = _find_relative_path(globals["__file__"], name, level)
            spec = ModuleSpec(_package_name(file), None, origin=file)
            module = importlib.util.module_from_spec(spec)
            module.__file__ = file
            exec(compile(Path(file).read_text(), file, "exec"), module.__dict__)  # noqa: S102

            return module

        return old_import(name, globals, locals, fromlist, level)

    try:
        builtins.__import__ = new_import
        yield
    finally:
        builtins.__import__ = old_import


def _find_relative_path(base: PathLike, relative: str, level: int) -> str:
    current = Path(base)
    for _ in range(level):
        current = current.parent

    parts = relative.split(".")
    for part in parts:
        current = current / part

    if not current.exists():
        current = current.with_suffix(".py")

        if not current.exists():
            msg = f"Module {relative} not found at {current}."
            raise ModuleNotFoundError(msg)

    return str(current)


def _package_name(path: PathLike) -> str:
    return _PKG_PREFIX + "." + Path(path).stem
