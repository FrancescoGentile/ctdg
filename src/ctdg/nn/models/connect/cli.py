# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

"""Coomand line interface for the CONNECT model."""

import argparse
from typing import TYPE_CHECKING

from ctdg import utils
from ctdg.data import DataModule

from ._link_prediction import LinkPredictionCONNECT

if TYPE_CHECKING:
    from lightning import Trainer


def init_parser(parser: argparse.ArgumentParser) -> None:
    """Initializes the parser for the CONNECT model."""
    parser.description = "Community Network for Continuous-Time Dynamic Graphs"
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
    """Trains the CONNECT model."""
    config = utils.load_config(args.config)

    if "seed" in config:
        utils.seed_all(config["seed"])

    data = DataModule(**config["data"])
    data.prepare_data()
    data.setup("fit")

    model = LinkPredictionCONNECT(
        nodes_features=data.dataset.nodes_features,
        events_features=data.dataset.events_features,
        model=config["model"],
        community_detection_p=config["community_detection_p"],
        community_detection_every_n_steps=config["community_detection_every_n_steps"],
        optimizer=config["optimizer"],
        scheduler=config.get("scheduler"),
    )

    t_cfg = config["trainer"]
    t_cfg["precision"] = utils.check_precision(t_cfg.get("precision", "32-true"))
    trainer: Trainer = t_cfg.evaluate()

    trainer.fit(model, datamodule=data)


def test(_args: argparse.Namespace) -> None:
    """Tests the CONNECT model."""
    raise NotImplementedError
