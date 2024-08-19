# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

"""Command line interface for the TGN model."""

import argparse
from typing import TYPE_CHECKING

from ctdg import utils
from ctdg.data import DataModule

from ._lightning import LightningTGN

if TYPE_CHECKING:
    from lightning import Trainer


def init_parser(parser: argparse.ArgumentParser) -> None:
    """Initializes the parser for the TGN model."""
    parser.description = "Temporal Graph Network (TGN) model"
    subparsers = parser.add_subparsers(title="commands", required=True)

    train_parser = subparsers.add_parser("train", help="train the model")
    train_parser.set_defaults(func=train)
    train_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="path to the configuration file",
    )

    test_parser = subparsers.add_parser("test", help="test the model")
    test_parser.set_defaults(func=test)
    test_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="path to the configuration file",
    )


def train(args: argparse.Namespace) -> None:
    """Trains the TGN model."""
    config = utils.load_config(args.config)

    if "seed" in config:
        utils.seed_all(config["seed"])

    data = DataModule(**config["data"])
    data.prepare_data()
    data.setup("fit")

    model = LightningTGN(
        nodes_features=data.dataset.nodes_features,
        events_features=data.dataset.events_features,
        train_events=data.dataset.train_events,
        full_events=data.dataset.events,
        model=config["model"],
        optimizer=config["optimizer"],
        scheduler=config.get("scheduler"),
    )

    t_cfg = config["trainer"]
    t_cfg["precision"] = utils.check_precision(t_cfg.get("precision", "32-true"))
    trainer: Trainer = t_cfg.evaluate()

    trainer.fit(model, datamodule=data)


def test(args: argparse.Namespace) -> None:
    """Tests the TGN model."""
    config = utils.load_config(args.config)

    if "seed" in config:
        utils.seed_all(config["seed"])

    data = DataModule(**config["data"])
    data.prepare_data()
    data.setup("test")

    model = LightningTGN(
        nodes_features=data.dataset.nodes_features,
        events_features=data.dataset.events_features,
        train_events=data.dataset.train_events,
        full_events=data.dataset.events,
        model=config["model"],
    )

    t_cfg = config["trainer"]
    t_cfg["precision"] = utils.check_precision(t_cfg.get("precision", "32-true"))
    trainer: Trainer = t_cfg.evaluate()

    trainer.test(model, datamodule=data)
