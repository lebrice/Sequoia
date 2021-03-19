""" pseudocode_after.py """
# flake8: noqa


class SimpleConvNet(nn.Module):
    def __init__(self, input_shape: Tuple, n_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(...)
        self.fc = nn.Sequential(...,)

    def forward(self, x: Tensor, task_labels: Optional[Tensor] = None) -> Tensor:
        return self.fc(self.features(x))


class Method:
    def __init__(self, lr: float = 1e-3, **method_options):
        self.lr = lr
        ...

    def configure(self, setting: Setting):
        self.model = SimpleConvNet(setting.observation_space, setting.action_space)
        self.optimizer = Adam(...)
    
    def fit(self, train_env: DataLoader, valid_env: DataLoader):
        # Train on a task:
        for epoch in range(self.epochs_per_task):
            self.model.train()
            for train_batch in train_env:
                ...  # Training loop
            self.model.eval()
            for valid_batch in valid_env:
                ...  # Validation loop

    def get_actions(self, observations, action_space: gym.Space):
        x, task_labels = observations
        actions = y_pred = self.model(x, task_labels).argmax(-1)
        return actions

    def on_task_switch(self, task_id: Optional[int]):
        self.model.prepare_for_new_task(new_task=task_id)


class Setting(LightningDataModule):
    def __init__(self, dataset=MNIST, n_tasks: int = 5, **setting_options):
        ...
        self.observation_space = gym.spaces.Box(0, 1, shape=self.image_shape)
        self.action_space = gym.spaces.Discrete(n=self.n_classes)
        self.reward_space = gym.spaces.Discrete(n=self.n_classes)
        self.current_task = 0
        self.train_datasets: List[Dataset] = self.make_train_datasets()

    def prepare_data(self):
        self.download_datasets()

    def setup(self, stage: str):
        # Create the training / validation / testing datasets for each task:
        self.train_datasets: List[Dataset] = self.make_train_datasets()
        self.valid_datasets: List[Dataset] = ...
        self.test_datasets: List[Dataset] = ...

    def train_dataloader(self, batch_size: int, num_workers: int) -> DataLoader:
        return DataLoader(self.train_datasets[self.current_task], ...)

    def val_dataloader(self, batch_size: int, num_workers: int) -> DataLoader:
        return DataLoader(self.valid_datasets[self.current_task], ...)

    def test_dataloader(self, batch_size: int, num_workers: int) -> DataLoader:
        return DataLoader(self.test_datasets[self.current_task], ...)

    def apply(self, method: Method):
        method.configure(self)
        # Matrix with test performance on all tasks after learning each task
        transfer_matrix = np.zeros([self.n_tasks, self.n_tasks])

        for i in range(self.n_tasks):
            self.current_task = i
            method.fit(
                train_env = self.train_dataloader(...),
                valid_env = self.val_dataloader(...),
            )

            # Evaluate on all tasks:
            for j in range(self.n_tasks):
                self.current_task = i
                for test_batch in self.test_dataloader():
                    # Ask the Method for its predictions (actions).
                    x, y, task_labels = test_batch
                    if not self.task_labels_available_at_test_time:
                        task_labels = None  # e.g. class vs task-incremental Settings.
                    y_pred = method.get_actions((x, task_labels), self.action_space)
                    correct_predictions += (y_pred == y).sum()
                task_accuracy = correct_predictions / len(self.test_datasets[j])
                transfer_matrix[train_task][test_task] = task_accuracy

        plot_results(transfer_matrix)

        # Main objective: average performance on all tasks at the end of training.
        cl_objective = transfer_matrix[-1].mean()
        return cl_objective
            

def cl_experiment(dataset=MNIST, n_tasks: int = 5, lr: float = 1e-3, **etc):

    # Create the Model, optimizer, etc:
    model = SimpleConvNet(...)
    optimizer = ...


if __name__ == "__main__":
    cl_objective = cl_experiment(dataset=MNIST, n_tasks=5, ...)
    print("Objective: ", cl_objective)
