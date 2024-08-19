# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

import dataclasses

from ctdg.nn import Memory, StaticInfo

from ._neighbor import NeighborSampler


@dataclasses.dataclass(frozen=True)
class GraphState:
    """The state of the temporal graph."""

    static_info: StaticInfo
    memory: Memory
    neighbor_sampler: NeighborSampler
