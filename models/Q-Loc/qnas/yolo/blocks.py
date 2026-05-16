from __future__ import annotations

import torch
import torch.nn as nn


class QuantumFeatureBlock(nn.Module):
    """CNN feature -> QNN -> CNN feature adapter."""

    def __init__(self, channels: int, q_layer: nn.Module, q_in_dim: int, q_out_dim: int, residual_scale: float = 1.0):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.to_q = nn.Linear(channels, q_in_dim)
        self.q_layer = q_layer
        self.from_q = nn.Linear(q_out_dim, channels)
        self.norm = nn.BatchNorm2d(channels, eps=0.001, momentum=0.03)
        self.act = nn.SiLU(inplace=True)
        self.residual_scale = float(residual_scale)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = feat.shape
        q_input = self.pool(feat).flatten(1)
        q_input = torch.tanh(self.to_q(q_input)) * torch.pi
        q_output = self.q_layer(q_input)
        q_output = self.from_q(q_output).view(b, c, 1, 1)
        return self.act(self.norm(feat + self.residual_scale * q_output))

