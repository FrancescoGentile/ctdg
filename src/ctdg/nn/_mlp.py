# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

from torch import nn


class MLP(nn.Sequential):
    """A multi-layer perceptron (MLP) module."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int | None = None,
        output_dim: int | None = None,
        activation: Callable[[], nn.Module] = nn.ReLU,
        num_layers: int = 1,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        """Initializes the MLP module.

        Args:
            input_dim: The dimensionality of the input tensor.
            hidden_dim: The dimensionality of the hidden layers. If not provided,
                defaults to the input dimension.
            output_dim: The dimensionality of the output tensor. If not provided,
                defaults to the input dimension.
            activation: The activation function to use between layers.
            num_layers: The number layers in the MLP, including the input and output
                layers.
            dropout: The dropout rate to apply between layers.
            bias: Whether to include bias in the linear layers.
        """
        if hidden_dim is not None and num_layers == 1:
            msg = "hidden_dim should be None when num_layers is 1."
            raise ValueError(msg)

        hidden_dim = hidden_dim or input_dim
        output_dim = output_dim or input_dim

        layers = []
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(input_dim, hidden_dim, bias=bias))
            layers.append(activation())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, output_dim, bias=bias))

        super().__init__(*layers)
