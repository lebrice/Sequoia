
from sequoia.settings import  ClassIncrementalSetting
from typing import Callable, Dict, Optional, Tuple

import torch
import tqdm

from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
import gym

class SimpleConvNet(nn.Module):
    def __init__(self, in_channels: int = 3, n_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.AdaptiveAvgPool2d(output_size=(8, 8)),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 120),  # NOTE: This '512' is what gets used as the
            # hidden size of the encoder.
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, n_classes),
        )

    def forward(self, x: Tensor, task_labels: Optional[Tensor] = None) -> Tensor:
        return self.fc(self.features(x))


class Method:
    def __init__(self, n_epochs_per_task: int = 1, learning_rate: float = 1e-3):
        self.n_epochs_per_task = n_epochs_per_task
        self.learning_rate = learning_rate

        self.loss_function: Callable[Tuple[Tensor, Tensor], Tensor]
        self.model: SimpleConvNet
        self.optimizer: Optimizer

    def configure(self, setting: "Setting"):
        self.loss_function = F.cross_entropy
        self.model = SimpleConvNet(
            in_channels=setting.in_channels, n_classes=setting.nb_classes
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

    def fit(self, train_loader: DataLoader, valid_loader: DataLoader):
        self.model.train()
        torch.set_grad_enabled(True)
        for epoch in range(self.n_epochs_per_task):
            train_pbar = tqdm.tqdm(train_loader, desc=f"epoch #{epoch}")
            for i, (x, y, t) in enumerate(train_pbar):
                logits = self.model(x)
                loss = self.loss_function(logits, y)

                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                if i % 10 == 0:
                    y_pred = logits.detach().argmax(-1)
                    accuracy = (y_pred == y).int().sum().item() / len(y)
                    train_pbar.set_postfix(
                        {"loss": loss.item(), "accuracy": f"{accuracy:.2f}"}
                    )

            self.model.eval()
            validation_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            valid_pbar = tqdm.tqdm(valid_loader, desc="Validation")
            for i, (x, y, t) in enumerate(valid_pbar):
                with torch.set_grad_enabled(False):
                    logits = self.model(x)
                    validation_loss += self.loss_function(logits, y).item()
                y_pred = logits.argmax(-1)
                correct_predictions += (y_pred == y).int().sum().item()
                total_predictions += len(y)

                val_accuracy = correct_predictions / total_predictions
                valid_pbar.set_postfix(
                    {
                        "total loss": validation_loss,
                        "average accuracy": f"{val_accuracy:.2%}",
                    }
                )

    def get_actions(self, observations: Tuple[Tensor, Optional[Tensor]], action_space: gym.Space) -> Tensor:
        self.model.eval()
        with torch.set_grad_enabled(False):
            x, task_labels = observations
            logits = self.model(x, task_labels)
            y_pred = logits.argmax(-1)
            return y_pred


if __name__ == "__main__":
    setting = ClassIncrementalSetting()
    method = Method()

    results = setting.apply(method)
    print(f"Results: {results}")
