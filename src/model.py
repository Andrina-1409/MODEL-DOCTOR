from __future__ import annotations

import torch
from torch import nn


class TinyCNN(nn.Module):
    """
    Small CNN for MNIST.

    The architecture is intentionally simple so that the model can later
    be inspected by Model Doctor using internal feature representations.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=3,
        )

        self.relu1 = nn.ReLU()

        self.pool1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
        )

        self.relu2 = nn.ReLU()

        self.pool2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        # MNIST input:
        #   28 x 28
        #
        # After conv1:
        #   26 x 26
        #
        # After pool1:
        #   13 x 13
        #
        # After conv2:
        #   11 x 11
        #
        # After pool2:
        #   5 x 5
        #
        # 16 channels × 5 × 5 = 400 features.
        self.classifier = nn.Linear(
            16 * 5 * 5,
            num_classes,
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the final convolutional feature maps.

        Shape:
            [batch, 16, 5, 5]
        """

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        return x

    def forward_penultimate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the flattened feature representation before classification.

        Shape:
            [batch, 400]
        """

        features = self.forward_features(x)

        return torch.flatten(
            features,
            start_dim=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return class logits.

        Shape:
            [batch, num_classes]
        """

        features = self.forward_penultimate(x)

        return self.classifier(features)
