# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

"""Metrics for the information cascade task."""

from collections.abc import Sequence

import torch
from torch import Tensor
from torchmetrics import MeanSquaredLogError, Metric, MetricCollection
from typing_extensions import override


class HitsAtK(Metric):
    """Hits@K metric."""

    def __init__(self, k: int = 10) -> None:
        """Initializes the metric.

        Args:
            k: The number of top elements to consider.
        """
        super().__init__()

        self.k = k
        self.add_state("hits", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

        # just for pyright
        self.hits: Tensor
        self.total: Tensor

    @override
    def update(self, pred: Tensor, gold: Tensor) -> None:
        """Updates the metric.

        Args:
            pred: The tensor of probabilities or logits. This should be a tensor of
                shape `(N, C)`, where `N` is the number of samples and `C` is the number
                of classes.
            gold: The tensor of ground truth labels. This should be a tensor of shape
                `(N,)` such that `gold[i]` is the true class of `pred[i]`.
        """
        _, topk = pred.topk(self.k, dim=1)  # (N, K)
        self.hits += topk.eq(gold.view(-1, 1)).sum().float()
        self.total += gold.size(0)

    @override
    def compute(self) -> Tensor:
        return self.hits / self.total


def get_macro_metrics(prefix: str | None = None) -> MetricCollection:
    """Returns the metrics for the macroscopic information cascade task.

    The metrics are:
    - mean squared log error (MSLE)
    """
    return MetricCollection({"msle": MeanSquaredLogError()}, prefix=prefix)


def get_micro_metrics(
    k: int | Sequence[int] = (10, 50, 100),
    prefix: str | None = None,
) -> MetricCollection:
    """Returns the metrics for the microscopic information cascade task.

    The metrics are:
    - Hits@K

    Args:
        k: The number of top elements to consider.
        prefix: The prefix to add to the names of the metrics.
    """
    k = k if isinstance(k, Sequence) else [k]
    return MetricCollection({f"hits@{k}": HitsAtK(k) for k in k}, prefix=prefix)
