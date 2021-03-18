from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import tqdm
from continuum.datasets import MNIST
from continuum.scenarios import ClassIncremental
from continuum.tasks import split_train_val
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


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
            nn.AdaptiveAvgPool2d(output_size=(8, 8)),  # [16, 8, 8]
            nn.Conv2d(
                16, 32, kernel_size=3, stride=1, padding=0, bias=False
            ),  # [32, 6, 6]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                32, 32, kernel_size=3, stride=1, padding=0, bias=False
            ),  # [32, 4, 4]
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

    def predict(self, x: Tensor, task_labels: Optional[Tensor] = None) -> Tensor:
        self.model.eval()
        with torch.set_grad_enabled(False):
            logits = self.model(x, task_labels)
            y_pred = logits.argmax(-1)
            return y_pred


class Setting:
    def __init__(self, dataset_type=MNIST, data_dir="data", nb_tasks: int = 5):
        self.dataset_type = dataset_type
        self.data_dir = data_dir
        self.nb_tasks = nb_tasks

        self.train_dataset = dataset_type(self.data_dir, train=True)
        self.test_dataset = dataset_type(self.data_dir, train=False)
        self.train_cl_scenario = ClassIncremental(
            self.train_dataset, nb_tasks=self.nb_tasks
        )
        self.test_cl_scenario = ClassIncremental(
            self.test_dataset, nb_tasks=self.nb_tasks
        )

        self.in_channels = 1 if dataset_type is MNIST else 3
        self.nb_classes = self.train_cl_scenario.nb_classes

    def apply(self, method: Method) -> Dict:
        # Give the Method a chance to configure itself
        method.configure(self)

        transfer_matrix = np.zeros([self.nb_tasks, self.nb_tasks])

        for task_index, task_dataset in enumerate(self.train_cl_scenario):
            print(f"Starting task {task_index}.")
            train_task_dataset, valid_task_dataset = split_train_val(
                task_dataset, val_split=0.1
            )
            train_loader = DataLoader(train_task_dataset, batch_size=32, num_workers=4)
            valid_loader = DataLoader(valid_task_dataset, batch_size=32, num_workers=4)

            method.fit(
                train_loader, valid_loader,
            )

            # After training on task `task_index`, we would like to be able to *test* on all
            # tasks!
            # model.eval()
            for test_task_index, test_task_dataset in enumerate(self.test_cl_scenario):
                test_loader = DataLoader(
                    test_task_dataset, batch_size=32, num_workers=4
                )
                test_pbar = tqdm.tqdm(
                    test_loader, desc=f"Test on task {test_task_index}"
                )

                correct_predictions = 0
                total_predictions = 0
                with torch.set_grad_enabled(False):

                    for i, (x, y, t) in enumerate(test_pbar):
                        # NOTE: Here we handle both the predicitons and the test labels.
                        y_pred = method.predict(x, task_labels=t)
                        correct_predictions += (y_pred == y).int().sum().item()
                        total_predictions += len(y)

                        test_accuracy = correct_predictions / total_predictions
                        test_pbar.set_postfix(
                            {
                                "average accuracy": f"{test_accuracy:.2%}",
                            }
                        )

                transfer_matrix[task_index][test_task_index] = test_accuracy

        return Results(transfer_matrix=transfer_matrix)


class Results:
    def __init__(self, transfer_matrix: np.ndarray):
        self.transfer_matrix = transfer_matrix

    @property
    def objective(self) -> float:
        """ Returns the 'objective' for this Setting. """
        return self.transfer_matrix[-1].mean()

    def make_plots(self):
        ax: plt.Axes
        fig, ax = plt.subplots()
        ax.imshow(self.transfer_matrix)

        # We want to show all ticks...
        ax.set_xticks(np.arange(self.nb_tasks))
        ax.set_yticks(np.arange(self.nb_tasks))
        # ... and label them with the respective list entries
        ax.set_xticklabels([f"Task {i}" for i in range(self.nb_tasks)])
        ax.set_yticklabels([f"{i} Tasks learned" for i in range(1, self.nb_tasks + 1)])
        ax.set_ylabel("# of tasks learned")
        ax.set_xlabel("Task Performance")
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(self.nb_tasks):
            for j in range(self.nb_tasks):
                acc = self.transfer_matrix[i, j]
                ax.text(
                    j,
                    i,
                    f"{acc:.1%}",
                    ha="center",
                    va="center",
                    color="g" if acc > 0.8 else "orange" if acc > 0.4 else "r",
                )
        ax.set_title("Transfer Matrix")
        fig.tight_layout()
        return {"Transfer Matrix": fig}

    def show(self):
        self.make_plots()
        plt.show()


if __name__ == "__main__":
    setting = Setting(dataset_type=MNIST, data_dir="data", nb_tasks=5)
    method = Method(n_epochs_per_task=1, learning_rate=1e-3)

    results = setting.apply(method)
    print(f"Result: {results.objective}")
    results.show()
