"""Convolutional network used for MNIST classification."""

from __future__ import annotations

import torch
from torch import nn


class ConvNet(nn.Module):
    """A small CNN tailored for MNIST digit classification."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.relu3(self.fc1(x))
        return self.fc2(x)

    def forward_with_activations(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass with intermediate activations for visualization."""
        activations: dict[str, torch.Tensor] = {}

        x = self.pool1(self.relu1(self.conv1(x)))
        activations["conv1"] = x

        x = self.pool2(self.relu2(self.conv2(x)))
        activations["conv2"] = x

        x = torch.flatten(x, start_dim=1)
        x = self.relu3(self.fc1(x))
        activations["fc1"] = x

        logits = self.fc2(x)
        return logits, activations
