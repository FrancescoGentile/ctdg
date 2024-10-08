# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

"""Command line interface for the TGN model."""

import argparse

from ctdg import utils

from . import information_cascade, link_prediction


def init_parser(parser: argparse.ArgumentParser) -> None:
    """Initializes the parser for the TGN model."""
    parser.description = "Temporal Graph Network (TGN) model"
    parser.add_argument(
        "--task",
        type=str,
        choices=["link-prediction", "information-cascade"],
        required=True,
        help="the task to perform",
    )

    parser.add_argument(
        "--phase",
        type=str,
        choices=["train", "test"],
        required=True,
        help="the phase to perform",
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="path to the configuration file",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["micro", "macro"],
        default="macro",
        help="the mode to use for the information cascade task",
    )

    parser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    """Main function for the TGN model."""
    config = utils.load_config(args.config)
    match args.phase, args.task:
        case "train", "link-prediction":
            link_prediction.train(config)
        case "test", "link-prediction":
            link_prediction.test(config)
        case "train", "information-cascade":
            information_cascade.train(config, args.mode)
        case "test", "information-cascade":
            information_cascade.test(config, args.mode)
        case _, _:
            msg = f"Unsupported task {args.task} or phase {args.phase}."
            raise ValueError(msg)
