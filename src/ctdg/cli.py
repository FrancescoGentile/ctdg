# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

"""Command line interface for the ctdg package."""

import argparse
import importlib
import pkgutil
import warnings

from ctdg import __version__


def main() -> None:
    """Entry point for the ctdg command line interface."""
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    parser = argparse.ArgumentParser(description="Continuous-Time Dynamic Graphs")
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(title="models", required=True)

    # Add subparsers for each model
    models = importlib.import_module("ctdg.nn.models")
    for _, name, _ in pkgutil.iter_modules(models.__path__):
        try:
            model = importlib.import_module(f"ctdg.nn.models.{name}.cli")
            model_parser = subparsers.add_parser(name, help=name)
            model.init_parser(model_parser)
        except (ModuleNotFoundError, AttributeError):  # noqa: PERF203
            continue

    args = parser.parse_args()
    args.func(args)
