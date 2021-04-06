import gym
import torch
from avalanche.benchmarks.classic import PermutedMNIST
from avalanche.benchmarks.scenarios import Experience
from avalanche.models import SimpleMLP
from gym import spaces
from sequoia.methods import Method
from sequoia.settings.passive import PassiveEnvironment, TaskIncrementalSetting
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from sequoia.settings.passive import PassiveSetting

from .naive import Naive


class SequoiaExperience(Experience):
    def __init__(self, env: PassiveEnvironment, setting: PassiveSetting):
        super().__init__()
        self.env = env
        self.setting = setting
        from continuum import TaskSet
        from avalanche.benchmarks.utils.avalanche_dataset import AvalancheDataset
        task_set: TaskSet = env.dataset
        x, y, t = task_set._x, task_set._y, task_set._t
        from torch.utils.data import TensorDataset
        import torch
        x = torch.as_tensor(x)
        y = torch.as_tensor(y)
        dataset = TensorDataset(x, y)
        dataset = AvalancheDataset(dataset=dataset, task_labels=t, targets=y)
        self._dataset = dataset

    @property
    def dataset(self):
        return self._dataset

    @property
    def task_label(self):
        return self.setting.current_task_id

    @property
    def task_labels(self):
        return list(range(self.setting.nb_tasks))

    @property
    def current_experience(self):
        return self


def environment_to_experience(env: PassiveEnvironment, setting: PassiveSetting) -> Experience:
    """
    TODO: Somehow convert our 'Environments' / dataloaders into an Experience object?
    """
    return SequoiaExperience(env=env, setting=setting)


class AvalancheMethod(Method, target_setting=TaskIncrementalSetting):
    def __init__(self):
        # Config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def configure(self, setting: TaskIncrementalSetting):
        # model
        self.setting = setting
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
        train_exp = environment_to_experience(train_env, setting=self.setting)
        valid_exp = environment_to_experience(valid_env, setting=self.setting)
        self.cl_strategy.train(train_exp, eval_streams=[valid_exp], num_workers=4)
        # return super().fit(train_env, valid_env)

    def get_actions(
        self, observations: TaskIncrementalSetting.Observations, action_space: gym.Space
    ) -> TaskIncrementalSetting.Actions:
        # TODO: Perform inference with the model.
        y_pred = self.model(observations.x)
        return self.target_setting.Actions(y_pred=y_pred)




# from sequoia.settings.base import SettingABC
# class AvalancheSetting(SettingABC):
#     def __init__(self):
#         self.scenario = PermutedMNIST()

#     def apply(self, method, config=None):
#         return super().apply(method, config=config)




if __name__ == "__main__":
    setting = TaskIncrementalSetting(dataset="mnist", nb_tasks=5)
    method = AvalancheMethod()
    results = setting.apply(method)




