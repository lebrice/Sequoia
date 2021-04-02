import gym
import torch
from avalanche.benchmarks.classic import PermutedMNIST
from avalanche.benchmarks.scenarios import Experience
from avalanche.models import SimpleMLP
from avalanche.training.strategies import Naive
from gym import spaces
from sequoia.methods import Method
from sequoia.settings.passive import PassiveEnvironment, TaskIncrementalSetting
from torch.nn import CrossEntropyLoss
from torch.optim import SGD


def environment_to_experience(env: PassiveEnvironment) -> Experience:
    """
    TODO: Somehow convert our 'Environments' / dataloaders into an Experience object?
    """
    return env


class AvalancheMethod(Method, target_setting=TaskIncrementalSetting):
    def __init__(self):
        # Config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def configure(self, setting: TaskIncrementalSetting):
        # model
        self.model = SimpleMLP(num_classes=10)
        # Prepare for training & testing
        self.optimizer = SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.criterion = CrossEntropyLoss()
        # Continual learning strategy
        self.cl_strategy = Naive(
            self.model,
            self.optimizer,
            self.criterion,
            train_mb_size=32,
            train_epochs=2,
            eval_mb_size=32,
            device=self.device,
        )

    def fit(self, train_env: PassiveEnvironment, valid_env: PassiveEnvironment):
        train_exp = environment_to_experience(train_env)
        valid_exp = environment_to_experience(valid_env)
        self.cl_strategy.train(train_exp, eval_streams=valid_exp, num_workers=4)
        # return super().fit(train_env, valid_env)

    def get_actions(
        self, observations: TaskIncrementalSetting.Observations, action_space: gym.Space
    ) -> TaskIncrementalSetting.Actions:
        # TODO: Perform inference with the model.
        y_pred = self.model(observations.x)
        return self.target_setting.Actions(y_pred=y_pred)


if __name__ == "__main__":
    setting = TaskIncrementalSetting(dataset="mnist", nb_tasks=5)
    method = AvalancheMethod()
    results = setting.apply(method)
