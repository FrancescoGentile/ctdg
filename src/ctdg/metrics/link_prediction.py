# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

"""Metrics for the link prediction task."""

from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
)


def get_metrics(prefix: str | None = None) -> MetricCollection:
    """Returns the metrics for the link prediction task.

    The metrics are:
    - accuracy
    - average precision
    - area under the receiver operating characteristic curve (AUROC)

    Args:
        prefix: The prefix to add to the metrics.

    Returns:
        The metrics for the link prediction task as a `torchmetrics.MetricCollection`.
    """
    return MetricCollection(
        {
            "accuracy": BinaryAccuracy(),
            "average_precision": BinaryAveragePrecision(),
            "auroc": BinaryAUROC(),
        },
        prefix=prefix,
    )
