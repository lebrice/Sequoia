from typing import Optional

from continuum import plot_samples
from continuum.datasets import MNIST, _ContinuumDataset
from torch.nn import functional as F
from continuum.scenarios import ClassIncremental
from continuum.tasks import split_train_val
from torch import Tensor, nn
from torch.utils.data import DataLoader
import torch
import numpy as np
import tqdm


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
            nn.Linear(512, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, n_classes),
        )

    def forward(self, x: Tensor, task_labels: Optional[Tensor] = None) -> Tensor:
        return self.fc(self.features(x))


def cl_experiment(
    dataset_type=MNIST,
    data_dir: str = "data",
    n_epochs_per_task: int = 1,
    nb_tasks: int = 5,
    loss_function=F.cross_entropy,
    optimizer_type=torch.optim.Adam,
    show: bool = True,
    learning_rate=1e-3,
):
    train_dataset = dataset_type(data_dir, train=True)
    test_dataset = dataset_type(data_dir, train=False)
    train_cl_scenario = ClassIncremental(train_dataset, nb_tasks=nb_tasks)
    test_cl_scenario = ClassIncremental(test_dataset, nb_tasks=nb_tasks)

    in_channels = 1 if dataset_type is MNIST else 3
    model = SimpleConvNet(
        in_channels=in_channels, n_classes=train_cl_scenario.nb_classes
    )
    optimizer = optimizer_type(model.parameters(), lr=learning_rate)

    transfer_matrix = np.zeros([nb_tasks, nb_tasks])

    for task_index, task_dataset in enumerate(train_cl_scenario):
        print(f"Starting task {task_index}.")
        train_task_dataset, valid_task_dataset = split_train_val(
            task_dataset, val_split=0.1
        )
        train_loader = DataLoader(train_task_dataset, batch_size=32, num_workers=4)
        valid_loader = DataLoader(valid_task_dataset, batch_size=32, num_workers=4)

        for epoch in range(n_epochs_per_task):
            model.train()
            train_pbar = tqdm.tqdm(train_loader, desc=f"epoch #{epoch}")
            for i, (x, y, t) in enumerate(train_pbar):
                logits = model(x)
                loss = loss_function(logits, y)

                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                if i % 10 == 0:
                    y_pred = logits.detach().argmax(-1)
                    accuracy = (y_pred == y).int().sum().item() / len(y)
                    train_pbar.set_postfix(
                        {"loss": loss.item(), "accuracy": f"{accuracy:.2f}"}
                    )

            model.eval()
            validation_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            valid_pbar = tqdm.tqdm(valid_loader, desc="Validation")
            for i, (x, y, t) in enumerate(valid_pbar):
                with torch.set_grad_enabled(False):
                    logits = model(x)
                    validation_loss += loss_function(logits, y).item()
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

        # After training on task `task_index`, we would like to be able to *test* on all
        # tasks!
        model.eval()
        for test_task_index, test_task_dataset in enumerate(test_cl_scenario):
            test_loader = DataLoader(test_task_dataset, batch_size=32, num_workers=4)
            test_pbar = tqdm.tqdm(test_loader, desc=f"Test on task {test_task_index}")

            test_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            with torch.set_grad_enabled(False):
                model.eval()
                for i, (x, y, t) in enumerate(test_pbar):
                    # NOTE: Here we handle both the predicitons and the test labels.
                    logits = model(x)
                    test_loss += loss_function(logits, y).item()
                    y_pred = logits.argmax(-1)

                    correct_predictions += (y_pred == y).int().sum().item()
                    total_predictions += len(y)

                    test_accuracy = correct_predictions / total_predictions
                    test_pbar.set_postfix(
                        {
                            "total loss": test_loss,
                            "average accuracy": f"{test_accuracy:.2%}",
                        }
                    )

            transfer_matrix[task_index][test_task_index] = test_accuracy

    return transfer_matrix

def plot_results(transfer_matrix: np.ndarray):
    import matplotlib.pyplot as plt

    ax: plt.Axes
    fig, ax = plt.subplots()
    img = ax.imshow(transfer_matrix)

    # We want to show all ticks...
    ax.set_xticks(np.arange(nb_tasks))
    ax.set_yticks(np.arange(nb_tasks))
    # ... and label them with the respective list entries
    ax.set_xticklabels([f"Task {i}" for i in range(nb_tasks)])
    ax.set_yticklabels([f"{i} Tasks learned" for i in range(1, nb_tasks + 1)])
    ax.set_ylabel("# of tasks learned")
    ax.set_xlabel("Task Performance")
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(nb_tasks):
        for j in range(nb_tasks):
            acc = transfer_matrix[i, j]
            text = ax.text(
                j,
                i,
                f"{acc:.1%}",
                ha="center",
                va="center",
                color="g" if acc > 0.8 else "orange" if acc > 0.4 else "r",
            )
    ax.set_title("Transfer Matrix")
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    nb_tasks = 5
    transfer_matrix = cl_experiment(
        dataset_type=MNIST,
        data_dir="data",
        n_epochs_per_task=1,
        nb_tasks=nb_tasks,
        loss_function=F.cross_entropy,
        optimizer_type=torch.optim.Adam,
        show=True,
        learning_rate=1e-3,
    )
    plot_results(transfer_matrix)

    objective = transfer_matrix[-1].mean()
    print(f"Average final accuracy across all tasks: {objective}")
