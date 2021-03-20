""" pseudocode_before.py """
# flake8: noqa


class SimpleConvNet(nn.Module):
    def __init__(self, in_channels: int = 3, n_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(...)
        self.fc = nn.Sequential(...,)

    def forward(self, x: Tensor, task_labels: Optional[Tensor] = None) -> Tensor:
        return self.fc(self.features(x))


def cl_experiment(dataset=MNIST, n_tasks: int = 5, lr: float = 1e-3, **etc):
    # Create the training / validation / testing datasets for each task:
    train_datasets: List[Dataset] = make_train_datasets(dataset, n_tasks)
    valid_datasets: List[Dataset] = make_valid_datasets(dataset, n_tasks)
    test_datasets: List[Dataset] = make_test_datasets(dataset, n_tasks)

    # Create the Model, optimizer, etc:
    model = SimpleConvNet(...)
    optimizer = ...

    # Matrix with test performance on all tasks after learning each task
    transfer_matrix = np.zeros([n_tasks, n_tasks])

    for i in range(n_tasks):
        train_loader = DataLoader(train_datasets[i], ...)
        valid_loader = DataLoader(valid_datasets[i], ...)

        # Train on task `i`:
        for epoch in range(epochs_per_task):
            for train_batch in train_loader:
                ...  # Training loop
            for valid_batch in valid_loader:
                ...  # Validation loop

        # Evaluate on all tasks:
        model.eval()
        for j in range(n_tasks):
            for test_batch in DataLoader(test_datasets[j], ...):
                ...  # Test loop
            task_accuracy = correct_predictions / total_predictions
            transfer_matrix[train_task][test_task] = task_accuracy
    
    plot_results(transfer_matrix)

    # Main objective: average performance on all tasks at the end of training.
    cl_objective = transfer_matrix[-1].mean()
    return cl_objective

if __name__ == "__main__":
    cl_objective = cl_experiment(dataset=MNIST, n_tasks=5, ...)
    print("Objective: ", cl_objective)
