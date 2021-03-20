""" pseudocode_after.py """
# flake8: noqa


class SimpleConvNet(nn.Module):
    def __init__(self, input_shape: Tuple, n_classes: int = 10):
        ...
    def forward(self, observations: Setting.Observations) -> Tensor:
        ...

class Method(target_setting=Setting):
    def __init__(self, lr: float = 1e-3, **method_options):
        ...

    def configure(self, setting: Setting):
        self.model = SimpleConvNet(setting.observation_space, setting.action_space)
        self.optimizer = Adam(...)
    
    def fit(self, train_env: DataLoader, valid_env: DataLoader):
        # Train on a task:
        for epoch in range(self.epochs_per_task):
            for train_batch in train_env:
                ...  # Training loop
            for valid_batch in valid_env:
                ...  # Validation loop

    def get_actions(self, observations: Setting.Observations, action_space: gym.Space):
        actions = y_pred = self.model(observations).argmax(-1)
        return actions

    def on_task_switch(self, task_id: Optional[int]):
        self.model.prepare_for_new_task(new_task=task_id)

method = MyMethod()

for setting in [ClassIncrementalSetting("mnist"), TaskIncrementalSetting("mnist"),
                DomainIncrementalSetting("mnist"), MultiTaskSetting("mnist"),
                IIDSetting("mnist"),]:
    results = setting.apply(method)
    results.make_plots()


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


if __name__ == "__main__":
    setting = Setting(dataset_type=MNIST, data_dir="data", nb_tasks=5)
    method = Method(n_epochs_per_task=1, learning_rate=1e-3)

    results = setting.apply(method)
    print(f"Result: {results.objective}")
    results.show()
