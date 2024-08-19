# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor, nn

from ._module import Module


class TimeEncoder(Module):
    """The time encoder module proposed by TGAT."""

    def __init__(self, dim: int) -> None:
        super().__init__()

        self.freq = nn.Parameter(1 / 10 * torch.linspace(0, 9, dim))
        self.phase = nn.Parameter(torch.zeros(dim))

    @property
    def output_dim(self) -> int:
        return self.freq.size(0)

    def __call__(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(-1)
        return torch.cos(x * self.freq + self.phase)
