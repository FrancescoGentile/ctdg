# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

"""Module for the CONNECT model for the link prediction task."""

from typing import Any

import torch
from lightning import LightningModule, Trainer
from torch import Tensor, nn
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from typing_extensions import override

from ctdg import utils
from ctdg.data import Events
from ctdg.data.link_prediction import DataModule
from ctdg.metrics.link_prediction import get_metrics
from ctdg.nn import MLP
from ctdg.utils import LazyCall

from ._model import CONNECT as _CONNECT


class CONNECT(LightningModule):
    """A lightning module for the link prediction task using the CONNECT model."""

    def __init__(
        self,
        num_nodes: int,
        nodes_features: Tensor | None,
        events_features: Tensor | None,
        model: LazyCall[_CONNECT],
        rewiring_p: float = 0.1,
        rewiring_every_n_steps: int = 1,
        optimizer: LazyCall[Optimizer] | None = None,
        scheduler: LazyCall[LRScheduler] | dict[str, Any] | None = None,
    ) -> None:
        """Initializes the link prediction module."""
        super().__init__()

        self.save_hyperparameters({
            "model": model.to_dict(),
            "optimizer": optimizer.to_dict() if optimizer else None,
            "scheduler": utils.get_scheduler_hyperparameters(scheduler),
        })

        self._optimizer = optimizer
        self._scheduler = scheduler

        self.model = model.evaluate(
            num_nodes=num_nodes,
            nodes_features=nodes_features,
            events_features=events_features,
        )
        self.link_predictor = MLP(
            input_dim=2 * self.model.embedder.output_dim,
            hidden_dim=self.model.embedder.output_dim,
            output_dim=1,
            num_layers=2,
            activation=nn.ReLU,
            dropout=0.0,
        )

        self.rewiring_p = rewiring_p
        self.rewiring_every_n_steps = rewiring_every_n_steps

        self.criterion = nn.BCEWithLogitsLoss()

        self.val_metrics = get_metrics(prefix="val/")
        self.test_metrics = get_metrics(prefix="test/")

    def forward(self, pos: Events, neg: Events) -> Tensor:
        """Performs link prediction on the given events.

        Args:
            pos: The positive events.
            neg: The negative events.

        Returns:
            The loss of the link prediction.
        """
        events = Events.cat([pos, neg])

        idx = torch.cat([events.src_nodes, events.dst_nodes])  # (2E,)
        t = events.timestamps.repeat(2)

        embeds = self.model.compute_node_embeddings(idx, t)  # (2E, D)
        src_embeds, dst_embeds = torch.chunk(embeds, 2, dim=0)
        link_embeds = torch.cat([src_embeds, dst_embeds], dim=1)
        logits = self.link_predictor(link_embeds).squeeze(-1)

        pos_src_embeds = src_embeds[: len(pos)]
        pos_dst_embeds = dst_embeds[: len(pos)]

        self.model.store_events(pos, pos_src_embeds, pos_dst_embeds)

        return logits

    @override
    def configure_optimizers(self) -> Any:
        return utils.configure_optimizers(self.model, self._optimizer, self._scheduler)

    @override
    def on_train_epoch_start(self) -> None:
        self.model.reset_graph_state()

    @override
    def training_step(self, batch: tuple[Events, Events], batch_idx: int) -> Tensor:
        if torch.rand(1) < self.rewiring_p:
            self.model.rewire_graph()

        logits = self(*batch)
        labels = _get_labels(*batch)
        loss = self.criterion(logits, labels)

        self.log("train/loss", loss, batch_size=len(labels))
        return loss

    @override
    def on_after_backward(self) -> None:
        self.model.detach_graph_state()

    @override
    def on_train_epoch_end(self) -> None:
        with torch.no_grad():
            self.model.eval()
            self.model.flush_events()
            self.model.train()

    @override
    def validation_step(self, batch: tuple[Events, Events], batch_idx: int) -> None:
        if batch_idx % self.rewiring_every_n_steps == 0:
            self.model.rewire_graph()

        logits = self(*batch)
        labels = _get_labels(*batch)
        loss = self.criterion(logits, labels)

        self.val_metrics.update(logits.sigmoid(), labels.to(torch.long))
        self.log("val/loss", loss, batch_size=len(labels))
        self.log_dict(self.val_metrics, batch_size=len(labels))

    @override
    def on_validation_epoch_end(self) -> None:
        self.model.flush_events()

    @override
    def test_step(self, batch: tuple[Events, Events], batch_idx: int) -> None:
        if batch_idx % self.rewiring_every_n_steps == 0:
            self.model.rewire_graph()

        logits = self(*batch)
        labels = _get_labels(*batch)
        loss = self.criterion(logits, labels)

        self.test_metrics.update(logits.sigmoid(), labels.to(torch.long))
        self.log("test/loss", loss, batch_size=len(labels))
        self.log_dict(self.test_metrics, batch_size=len(labels))

    @override
    def on_test_epoch_end(self) -> None:
        self.model.flush_events()


def _get_labels(pos: Events, neg: Events) -> Tensor:
    """Returns the labels for the positive and negative events."""
    return torch.cat([
        torch.ones_like(pos.timestamps),
        torch.zeros_like(neg.timestamps),
    ])


# --------------------------------------------------------------------------- #
# Command-line interface
# --------------------------------------------------------------------------- #


def train(config: dict[str, Any]) -> None:
    """Trains the CONNECT model for the link prediction task."""
    if "seed" in config:
        utils.seed_all(config["seed"])

    data = DataModule(**config["data"])
    data.prepare_data()
    data.setup("fit")

    model = CONNECT(
        num_nodes=data.dataset.num_nodes,
        nodes_features=data.dataset.nodes_features,
        events_features=data.dataset.events_features,
        model=config["model"],
        rewiring_p=config["rewiring_p"],
        rewiring_every_n_steps=config["rewiring_every_n_steps"],
        optimizer=config["optimizer"],
        scheduler=config.get("scheduler"),
    )

    t_cfg = config["trainer"]
    t_cfg["precision"] = utils.check_precision(t_cfg.get("precision"))
    trainer: Trainer = t_cfg.evaluate()

    trainer.fit(model, datamodule=data)


def test(config: dict[str, Any]) -> None:
    """Tests the CONNECT model for the link prediction task."""
    if "seed" in config:
        utils.seed_all(config["seed"])

    data = DataModule(**config["data"])
    data.prepare_data()
    data.setup("test")

    model = CONNECT(
        num_nodes=data.dataset.num_nodes,
        nodes_features=data.dataset.nodes_features,
        events_features=data.dataset.events_features,
        model=config["model"],
        rewiring_every_n_steps=config["rewiring_every_n_steps"],
    )

    t_cfg = config["trainer"]
    t_cfg["precision"] = utils.check_precision(t_cfg.get("precision"))
    trainer: Trainer = t_cfg.evaluate()

    trainer.test(model, datamodule=data, ckpt_path=config["ckpt_path"])
