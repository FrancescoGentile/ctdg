# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

"""Module for the TIGER model for the information cascade task."""

from typing import Any, Literal

import torch
from lightning import LightningModule, Trainer
from torch import Tensor, nn
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from typing_extensions import override

from ctdg import utils
from ctdg.data import Events
from ctdg.data.information_cascade import DataModule
from ctdg.metrics.information_cascade import get_macro_metrics, get_micro_metrics
from ctdg.nn import LastNeighborSampler
from ctdg.utils import LazyCall

from ._model import TIGER as _TIGER


class TIGER(LightningModule):
    """The TIGER model for the information cascade task."""

    def __init__(
        self,
        num_sources: int,
        num_targets: int,
        nodes_features: Tensor | None,
        events_features: Tensor | None,
        train_events: Events,
        val_events: Events,
        test_events: Events,
        model: LazyCall[_TIGER],
        optimizer: LazyCall[Optimizer] | None = None,
        scheduler: LazyCall[LRScheduler] | dict[str, Any] | None = None,
        mode: Literal["micro", "macro"] = "macro",
    ) -> None:
        """Initializes the TIGER model."""
        super().__init__()

        self.save_hyperparameters({
            "mode": mode,
            "model": model.to_dict(),
            "optimizer": optimizer.to_dict() if optimizer else None,
            "scheduler": utils.get_scheduler_hyperparameters(scheduler),
        })

        self._optimizer = optimizer
        self._scheduler = scheduler

        num_nodes = num_sources + num_targets
        if nodes_features is not None and nodes_features.size(0) != num_nodes:
            msg = f"Expected {num_nodes} nodes, got {nodes_features.size(0)}."
            raise ValueError(msg)

        self.train_sampler = LastNeighborSampler(train_events, num_nodes)
        self.val_sampler = LastNeighborSampler(val_events, num_nodes)
        self.test_sampler = LastNeighborSampler(test_events, num_nodes)

        self.model = model.evaluate(
            num_nodes=num_nodes,
            nodes_features=nodes_features,
            events_features=events_features,
            neighbor_sampler=self.train_sampler,
        )

        self.mode = mode
        if mode == "micro":
            self.predictor = nn.Linear(self.model.embedder.output_dim, num_sources)
            self.criterion = nn.CrossEntropyLoss()
            self.val_metrics = get_micro_metrics(prefix="val/")
            self.test_metrics = get_micro_metrics(prefix="test/")
        else:
            self.predictor = nn.Linear(self.model.embedder.output_dim, 1)
            self.criterion = nn.MSELoss()
            self.val_metrics = get_macro_metrics(prefix="val/")
            self.test_metrics = get_macro_metrics(prefix="test/")

    def forward(self, events: Events) -> Tensor:
        """Estimates the size of the information cascade."""
        idx = torch.cat([events.src_nodes, events.dst_nodes])
        t = events.timestamps.repeat(2)
        embeds = self.model.compute_node_embeddings(idx, t)
        src_embeds, dst_embeds = embeds.chunk(2, dim=0)

        pred = self.predictor(dst_embeds).squeeze(-1)

        self.model.store_events(events, src_embeds, dst_embeds)

        return pred

    @override
    def configure_optimizers(self) -> Any:
        return utils.configure_optimizers(self.model, self._optimizer, self._scheduler)

    @override
    def on_train_epoch_start(self) -> None:
        self.model.reset_graph_state()
        self.model.change_neighbor_sampler(self.train_sampler)

    @override
    def training_step(self, batch: tuple[Events, Tensor], batch_idx: int) -> Tensor:
        events = batch[0]
        pred = self(events)
        gold = batch[1] if self.mode == "macro" else events.src_nodes

        pred = self(events)
        loss = self.criterion(pred, gold)

        self.log("train/loss", loss, batch_size=len(events))

        return loss

    @override
    def on_after_backward(self) -> None:
        self.model.detach_graph_state()

    @override
    def on_validation_epoch_start(self) -> None:
        self.model.reset_graph_state()
        self.model.change_neighbor_sampler(self.val_sampler)

    @override
    def validation_step(self, batch: tuple[Events, Tensor], batch_idx: int) -> None:
        events = batch[0]
        pred = self(events)
        gold = batch[1] if self.mode == "macro" else events.src_nodes

        loss = self.criterion(pred, gold)
        self.log("val/loss", loss, batch_size=len(events))

        if self.mode == "macro":
            pred = pred.clamp(min=0.0)

        self.val_metrics.update(pred, gold)
        self.log_dict(self.val_metrics, batch_size=len(events))

    @override
    def on_test_epoch_start(self) -> None:
        self.model.reset_graph_state()
        self.model.change_neighbor_sampler(self.test_sampler)

    @override
    def test_step(self, batch: tuple[Events, Tensor], batch_idx: int) -> None:
        events = batch[0]
        pred = self(events)
        gold = batch[1] if self.mode == "macro" else events.src_nodes

        loss = self.criterion(pred, gold)
        self.log("test/loss", loss, batch_size=len(events))

        if self.mode == "macro":
            pred = pred.clamp(min=0.0)

        self.test_metrics.update(pred, gold)
        self.log_dict(self.test_metrics, batch_size=len(events))


# --------------------------------------------------------------------------- #
# Command-line interface
# --------------------------------------------------------------------------- #


def train(config: dict[str, Any], mode: Literal["micro", "macro"]) -> None:
    """Trains the TIGER model for the information cascade task."""
    if "seed" in config:
        utils.seed_all(config["seed"])

    data = DataModule(**config["data"])
    data.prepare_data()
    data.setup("fit")

    model = TIGER(
        mode=mode,
        num_sources=data.dataset.num_sources,
        num_targets=data.dataset.num_destinations,
        nodes_features=data.dataset.nodes_features,
        events_features=data.dataset.events_features,
        train_events=data.dataset.train_events,
        val_events=data.dataset.val_events,
        test_events=data.dataset.test_events,
        model=config["model"],
        optimizer=config["optimizer"],
        scheduler=config.get("scheduler"),
    )

    t_cfg = config["trainer"]
    t_cfg["precision"] = utils.check_precision(t_cfg.get("precision"))
    trainer: Trainer = t_cfg.evaluate()

    trainer.fit(model, datamodule=data)


def test(config: dict[str, Any], mode: Literal["micro", "macro"]) -> None:
    """Tests the TIGER model for the information cascade task."""
    if "seed" in config:
        utils.seed_all(config["seed"])

    data = DataModule(**config["data"])
    data.prepare_data()
    data.setup("test")

    model = TIGER(
        mode=mode,
        num_sources=data.dataset.num_sources,
        num_targets=data.dataset.num_destinations,
        nodes_features=data.dataset.nodes_features,
        events_features=data.dataset.events_features,
        train_events=data.dataset.train_events,
        val_events=data.dataset.val_events,
        test_events=data.dataset.test_events,
        model=config["model"],
    )

    t_cfg = config["trainer"]
    t_cfg["precision"] = utils.check_precision(t_cfg.get("precision"))
    trainer: Trainer = t_cfg.evaluate()

    trainer.test(model, datamodule=data, ckpt_path=config["ckpt_path"])
