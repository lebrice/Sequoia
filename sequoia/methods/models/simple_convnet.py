from torch import nn, Tensor
from torch.nn import Flatten

class SimpleConvNet(nn.Module):
    def __init__(self, in_channels=3, n_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 120), # NOTE: This '256' is what gets used as the
            # hidden size of the encoder.
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, n_classes),
        )
    def forward(self, x: Tensor) -> Tensor:
        return self.fc(self.features(x))
