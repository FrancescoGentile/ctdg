# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from ._lazy import LazyCall


def get_scheduler_hyperparameters(
    cfg: LazyCall[LRScheduler] | dict[str, Any] | None,
) -> None | dict[str, Any]:
    """Gets the hyperparameters of the scheduler from the configuration."""
    if cfg is None:
        return None
    if isinstance(cfg, LazyCall):
        return cfg.to_dict()

    cfg = cfg.copy()
    scheduler = cfg.pop("scheduler")
    return {"scheduler": scheduler.to_dict(), **cfg}


def configure_optimizers(
    model: nn.Module,
    optimizer_cfg: LazyCall[Optimizer] | None,
    scheduler_cfg: LazyCall[LRScheduler] | dict[str, Any] | None,
) -> None | Optimizer | OptimizerLRSchedulerConfig:
    """Configures the optimizer and the scheduler."""
    if optimizer_cfg is None:
        return None

    optimizer = optimizer_cfg.evaluate(model.parameters())
    if scheduler_cfg is None:
        return optimizer

    if isinstance(scheduler_cfg, dict):
        scheduler = scheduler_cfg.pop("scheduler").evaluate(optimizer)
        extras = scheduler
    else:
        scheduler = scheduler_cfg.evaluate(optimizer)
        extras = {}

    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            **extras,
        },  # type: ignore
    }
