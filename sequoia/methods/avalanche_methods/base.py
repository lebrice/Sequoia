from typing import List, Optional

import gym
import numpy as np
import torch
import tqdm
from avalanche.benchmarks.classic import PermutedMNIST
from avalanche.benchmarks.scenarios import Experience
from avalanche.benchmarks.utils import as_avalanche_dataset
from avalanche.benchmarks.utils.avalanche_dataset import (AvalancheDataset,
                                                          _TaskSubsetDict)
from avalanche.models import SimpleMLP
from continuum import TaskSet
from gym import spaces
from gym.spaces.utils import flatdim
from sequoia.common.gym_wrappers.utils import IterableWrapper
from sequoia.common.spaces import Image
from sequoia.methods import Method
from sequoia.settings.passive import (ClassIncrementalSetting,
                                      PassiveEnvironment, PassiveSetting,
                                      TaskIncrementalSetting)
from sequoia.settings.passive.cl.objects import Actions, Observations, Rewards
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import TensorDataset

from .naive import Naive


class SequoiaExperience(IterableWrapper, Experience):
    def __init__(self, env: PassiveEnvironment, setting: ClassIncrementalSetting):
        super().__init__(env=env)
        self.setting = setting
        if env is setting.train_env:
            self.transforms = setting.train_transforms
        elif env is setting.val_env:
            self.transforms = setting.val_transforms
        else:
            assert env is setting.test_env
            self.transforms = setting.test_transforms

        self.task_id = setting.current_task_id

        all_observations: List[Observations] = []
        all_rewards: List[Rewards] = []

        for batch in tqdm.tqdm(self, desc="Converting environment into TensorDataset"):
            observations: Observations
            rewards: Optional[Rewards]
            if isinstance(batch, Observations):
                observations = batch
                rewards = None
            else:
                assert isinstance(batch, tuple) and len(batch) == 2
                observations, rewards = batch

            if rewards is None:
                # Need to send actions to the env before we can actually get the
                # associated Reward.
                # Here we sample a random action (no other choice really..) and so we
                # are going to get bad results in case the online performance is being
                # evaluated.
                action = self.env.action_space.sample()
                rewards = self.env.send(action)
                assert False, (observations.shapes, action.shapes, rewards.shapes)
            all_observations.append(observations)
            all_rewards.append(rewards)
        # TODO: This will be absolutely unfeasable for larger dataset like ImageNet.
        stacked_observations: Observations = Observations.concatenate(all_observations)

        assert all(
            y_i is not None for y in all_rewards for y_i in y
        ), "Need fully labeled train dataset for now."
        stacked_rewards: Rewards = Rewards.concatenate(all_rewards)

        dataset = TensorDataset(stacked_observations.x, stacked_rewards.y)
        dataset = AvalancheDataset(
            dataset=dataset,
            task_labels=stacked_observations.task_labels.tolist(),
            targets=stacked_rewards.y.tolist(),
        )
        self._dataset = dataset
        self.tasks_pattern_indices = dict({0: np.arange(len(self._dataset))})
        self.task_set = _TaskSubsetDict(self._dataset)
        # self._dataset = env
        # from avalanche.benchmarks import GenericScenarioStream
        # class FakeStream(GenericScenarioStream):
        #     pass
        # self.origin_stream = FakeStream("train", scenario="whatever")
        # self.origin_stream.name = "train"

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
        # Return the index of the
        return self.task_id

    @property
    def origin_stream(self) -> PassiveSetting:
        # NOTE: This 
        return self.setting

    def train(self):
        return self


def environment_to_experience(
    env: PassiveEnvironment, setting: PassiveSetting
) -> Experience:
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
        image_space: Image = setting.observation_space.x
        input_dims = flatdim(image_space)
        self.model = SimpleMLP(input_size=input_dims, num_classes=10)
        # Prepare for training & testing
        self.optimizer = SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.criterion = CrossEntropyLoss()
        # Continual learning strategy
        self.cl_strategy = Naive(
            self.model,
            self.optimizer,
            self.criterion,
            train_mb_size=1,
            train_epochs=2,
            eval_mb_size=1,
            device=self.device,
            eval_every=0,
        )

    def fit(self, train_env: PassiveEnvironment, valid_env: PassiveEnvironment):
        train_exp = environment_to_experience(train_env, setting=self.setting)
        valid_exp = environment_to_experience(valid_env, setting=self.setting)
        self.cl_strategy.train(train_exp, eval_streams=[valid_exp], num_workers=0)
        # return super().fit(train_env, valid_env)

    def get_actions(
        self, observations: TaskIncrementalSetting.Observations, action_space: gym.Space
    ) -> TaskIncrementalSetting.Actions:
        # TODO: Perform inference with the model.
        with torch.no_grad():
            logits = self.model(observations.x.to(self.device))
            y_pred = logits.argmax(-1)
            return self.target_setting.Actions(y_pred=y_pred)
