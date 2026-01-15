#!/usr/bin/env python3
"""Model architectures for implicit neural representations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class PositionalEncodingConfig:
    num_bands: int = 10
    include_input: bool = True
    scale: float = 1.0  # Multiplier applied before frequency bands.


class PositionalEncoding(nn.Module):
    """Applies sin/cos positional encoding to input features."""

    def __init__(self, in_dims: int, config: PositionalEncodingConfig) -> None:
        super().__init__()
        self.in_dims = in_dims
        self.config = config

        freq_bands = 2.0 ** torch.arange(config.num_bands, dtype=torch.float32)
        self.register_buffer("freq_bands", freq_bands * config.scale, persistent=False)

        out_dims = config.num_bands * 2 * in_dims
        if config.include_input:
            out_dims += in_dims
        self.out_dims = out_dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expand for broadcasting: (..., dims, 1) * (bands,) -> (..., dims, bands)
        xb = x.unsqueeze(-1) * self.freq_bands
        encoded_parts = []
        if self.config.include_input:
            encoded_parts.append(x.unsqueeze(-1))
        encoded_parts.append(torch.sin(xb))
        encoded_parts.append(torch.cos(xb))

        encoded = torch.cat(encoded_parts, dim=-1)
        # Flatten the frequency dimension into the feature axis.
        return encoded.reshape(*x.shape[:-1], -1)


class MLP(nn.Module):
    """ReLU MLP with optional positional encoding front-end."""

    def __init__(
        self,
        in_dims: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        pe_config: Optional[PositionalEncodingConfig] = None,
    ) -> None:
        super().__init__()
        self.pe: Optional[PositionalEncoding]
        if pe_config is not None:
            self.pe = PositionalEncoding(in_dims, pe_config)
            in_dims = self.pe.out_dims
        else:
            self.pe = None

        layers = []
        last_dim = in_dims
        for _ in range(num_layers):
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)
        self.final_activation = nn.Sigmoid()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pe is not None:
            x = self.pe(x)
        return self.final_activation(self.net(x))


class SineLayer(nn.Module):
    """SIREN layer with configurable omega_0."""

    def __init__(self, in_features: int, out_features: int, *, omega_0: float, is_first: bool) -> None:
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.linear.in_features
            else:
                bound = math.sqrt(6.0 / self.linear.in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(nn.Module):
    """SIREN network for implicit representations."""

    def __init__(
        self,
        in_dims: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        omega_0: float = 30.0,
        omega_0_hidden: float = 1.0,
    ) -> None:
        super().__init__()

        layers = []
        layers.append(SineLayer(in_dims, hidden_dim, omega_0=omega_0, is_first=True))
        for _ in range(num_layers - 1):
            layers.append(SineLayer(hidden_dim, hidden_dim, omega_0=omega_0_hidden, is_first=False))
        self.sine_layers = nn.ModuleList(layers)
        self.final_linear = nn.Linear(hidden_dim, 1)
        self.final_activation = nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(
            self.final_linear.weight,
            -math.sqrt(6.0 / self.final_linear.in_features) / 1.0,
            math.sqrt(6.0 / self.final_linear.in_features) / 1.0,
        )
        nn.init.zeros_(self.final_linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.sine_layers:
            x = layer(x)
        x = self.final_linear(x)
        return self.final_activation(x)
