# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch import Tensor, nn
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
)
from typing_extensions import override

from ctdg import utils
from ctdg.nn import StaticInfo
from ctdg.structures import Events
from ctdg.utils import LazyCall

from ._tgn import TGN


class LightningTGN(LightningModule):
    """A PyTorch Lightning module for the TGN model."""

    def __init__(
        self,
        nodes_features: Tensor,
        events_features: Tensor,
        train_events: Events,
        full_events: Events,
        model: LazyCall[TGN],
        optimizer: LazyCall[Optimizer] | None = None,
        scheduler: LazyCall[LRScheduler] | dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        self.save_hyperparameters({
            "model": model.to_dict(),
            "optimizer": optimizer.to_dict() if optimizer else None,
            "scheduler": utils.get_scheduler_hyperparameters(scheduler),
        })

        self._optimizer = optimizer
        self._scheduler = scheduler

        static_info = StaticInfo(nodes_features, events_features)
        self.train_sampler = model["neighbor_sampler"].evaluate(
            events=train_events,
            num_nodes=static_info.num_nodes,
        )
        self.full_sampler = model["neighbor_sampler"].evaluate(
            events=full_events,
            num_nodes=static_info.num_nodes,
        )

        self.model = model.evaluate(
            static_info=static_info,
            neighbor_sampler=self.train_sampler,
        )
        self.criterion = nn.BCEWithLogitsLoss()

        self.val_metrics = MetricCollection(
            {
                "accuracy": BinaryAccuracy(),
                "average_precision": BinaryAveragePrecision(),
                "auroc": BinaryAUROC(),
            },
            prefix="val/",
        )
        self.test_metrics = MetricCollection(
            {
                "accuracy": BinaryAccuracy(),
                "average_precision": BinaryAveragePrecision(),
                "auroc": BinaryAUROC(),
            },
            prefix="test/",
        )

    def forward(
        self,
        pos: Events,
        neg: Events,
        phase: Literal["train", "val", "test"],
    ) -> Tensor:
        events = Events.cat([pos, neg])
        logits = self.model.compute_events_probabilities(events)
        labels = torch.cat([
            torch.ones_like(pos.timestamps),
            torch.zeros_like(neg.timestamps),
        ])
        loss = self.criterion(logits, labels)

        self.model.store_events(pos)

        self.log(f"{phase}/loss", loss, batch_size=len(events))
        if phase != "train":
            metrics = self.val_metrics if phase == "val" else self.test_metrics
            metrics.update(logits.sigmoid(), labels.to(torch.long))
            self.log_dict(metrics, batch_size=len(events))

        return loss

    @override
    def configure_optimizers(self) -> None | Optimizer | OptimizerLRSchedulerConfig:
        if self._optimizer is None:
            return None

        optimizer = self._optimizer.evaluate(self.model.parameters())
        if self._scheduler is None:
            return optimizer

        if isinstance(self._scheduler, dict):
            scheduler = self._scheduler.pop("scheduler").evaluate(optimizer)
            extras = self._scheduler
        else:
            scheduler = self._scheduler.evaluate(optimizer)
            extras = {}

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                **extras,
            },  # type: ignore
        }

    @override
    def on_train_epoch_start(self) -> None:
        self.model.reset_graph_state()
        self.model.neighbor_sampler = self.train_sampler

    @override
    def training_step(self, *args: Any, **kwargs: Any) -> Tensor:
        return self(*args[0], phase="train")

    @override
    def on_after_backward(self) -> None:
        self.model.memory.detach()

    @override
    def on_train_epoch_end(self) -> None:
        with torch.no_grad():
            self.model.eval()
            self.model.flush_events()
            self.model.train()

    @override
    def on_validation_epoch_start(self) -> None:
        self.model.neighbor_sampler = self.full_sampler

    @override
    def validation_step(self, *args: Any, **kwargs: Any) -> Tensor:
        return self(*args[0], phase="val")

    @override
    def on_validation_epoch_end(self) -> None:
        self.model.flush_events()

    @override
    def on_test_epoch_start(self) -> None:
        self.model.neighbor_sampler = self.full_sampler

    @override
    def test_step(self, *args: Any, **kwargs: Any) -> Tensor:
        return self(*args[0], phase="test")
