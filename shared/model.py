import torch
import torch.nn as nn


class IoTClassifier(nn.Module):
    """
    Multi-layer Perceptron for IoT Risk Classification
    Input: 7 sensor features
    Output: 3 classes (Normal, Warning, Critical)
    """

    def __init__(self, input_dim=7, num_classes=3):
        super(IoTClassifier, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.network(x)


def initialize_model(device="cpu"):
    """
    Initialize model and move to device
    """
    model = IoTClassifier()
    return model.to(device)
