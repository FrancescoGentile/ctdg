# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

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
from ctdg.nn import MLP
from ctdg.structures import Events
from ctdg.utils import LazyCall

from ._connect import CONNECT


class LinkPredictionCONNECT(LightningModule):
    """A Lightning module that performs link prediction with the CONNECT model."""

    def __init__(
        self,
        nodes_features: Tensor,
        events_features: Tensor,
        model: LazyCall[CONNECT],
        community_detection_p: float = 0.1,
        community_detection_every_n_steps: int = 1,
        optimizer: LazyCall[Optimizer] | None = None,
        scheduler: LazyCall[LRScheduler] | dict[str, Any] | None = None,
    ) -> None:
        """Initializes the Lightning module.

        Args:
            nodes_features: The static features of the nodes.
            events_features: The features of the events.
            model: The configuration of the CONNECT model.
            optimizer: The configuration of the optimizer. If this module is used
                only for evaluation, this can be set to `None`.
            scheduler: The configuration of the learning rate scheduler. This parameter
                is always optional. If passed, this can be either a lazy call to the
                scheduler or a dictionary with a "scheduler" key that contains the lazy
                call to the scheduler and other optional parameters (see lightning docs
                for the available options).
            community_detection_p: The probability of detecting communities at each
                training step.
            community_detection_every_n_steps: The number of steps after which to
                detect communities during evaluation.
        """
        super().__init__()

        self.save_hyperparameters({
            "model": model.to_dict(),
            "optimizer": optimizer.to_dict() if optimizer else None,
            "scheduler": utils.get_scheduler_hyperparameters(scheduler),
        })

        self._optimizer = optimizer
        self._scheduler = scheduler
        self.community_detection_p = community_detection_p
        self.community_detection_every_n_steps = community_detection_every_n_steps

        self.model = model.evaluate(
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

    def forward(self, events: Events) -> Tensor:
        """Performs link prediction on the given events.

        Args:
            events: The events for which to perform link prediction.

        Returns:
            The log-probabilities of the events happening.
        """
        idx = torch.cat([events.src_nodes, events.dst_nodes])  # (2E,)
        t = events.timestamps.repeat(2)  # (2E,)

        embeds = self.model.compute_node_embeddings(idx, t)  # (2E, D)
        src_embeds, dst_embeds = torch.chunk(embeds, 2, dim=0)  # 2 x (E, D)
        link_embeds = torch.cat([src_embeds, dst_embeds], dim=1)  # (E, 2D)

        return self.link_predictor(link_embeds).squeeze(-1)  # (E,)

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

    @override
    def training_step(self, batch: tuple[Events, Events], batch_idx: int) -> Tensor:
        pos, neg = batch
        events = Events.cat([pos, neg])

        if torch.rand(1) < self.community_detection_p:
            self.model.detect_communities()

        logits = self(events)
        labels = _get_labels(pos, neg)
        loss = self.criterion(logits, labels)

        self.model.store_events(pos)

        self.log("train/loss", loss, batch_size=len(events))

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
    def validation_step(self, batch: tuple[Events, Events], batch_idx: int) -> Tensor:
        pos, neg = batch
        events = Events.cat([pos, neg])

        if batch_idx % self.community_detection_every_n_steps == 0:
            self.model.detect_communities()

        logits = self(events)
        labels = _get_labels(pos, neg)
        loss = self.criterion(logits, labels)

        self.model.store_events(pos)

        self.log("val/loss", loss, batch_size=len(events))
        self.val_metrics.update(logits.sigmoid(), labels.to(torch.long))
        self.log_dict(self.val_metrics, batch_size=len(events))

        return loss

    @override
    def on_validation_epoch_end(self) -> None:
        # if there are some events left, we need to flush them
        # so that the state of the temporal graph is updated to the last event
        # otherwise, when saving the model, these last events will be lost
        self.model.flush_events()

    @override
    def test_step(self, batch: tuple[Events, Events], batch_idx: int) -> Tensor:
        pos, neg = batch
        events = Events.cat([pos, neg])

        if batch_idx % self.community_detection_every_n_steps == 0:
            self.model.detect_communities()

        logits = self(events)
        labels = _get_labels(pos, neg)
        loss = self.criterion(logits, labels)

        self.model.store_events(pos)

        self.log("test/loss", loss, batch_size=len(events))
        self.test_metrics.update(logits.sigmoid(), labels.to(torch.long))
        self.log_dict(self.test_metrics, batch_size=len(events))

        return loss

    @override
    def on_test_epoch_end(self) -> None:
        self.model.flush_events()


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #


def _get_labels(pos: Events, neg: Events) -> Tensor:
    return torch.cat([
        torch.ones_like(pos.timestamps),
        torch.zeros_like(neg.timestamps),
    ])
