""" 'Wrapper' around a PassiveEnvironment from Sequoia, disguising it as an 'Experience'
from Avalanche.
"""
from typing import List, Optional

import numpy as np
import tqdm
from avalanche.benchmarks.scenarios import Experience
from avalanche.benchmarks.utils.avalanche_dataset import (AvalancheDataset,
                                                          _TaskSubsetDict)

from sequoia.common.gym_wrappers.utils import IterableWrapper
from sequoia.settings.passive import (ClassIncrementalSetting,
                                      PassiveEnvironment, PassiveSetting)
from sequoia.settings.passive.cl.objects import Observations, Rewards
from torch.utils.data import TensorDataset


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

        # all_observations: List[Observations] = []
        # all_rewards: List[Rewards] = []

        # for batch in tqdm.tqdm(self, desc="Converting environment into TensorDataset"):
        #     observations: Observations
        #     rewards: Optional[Rewards]
        #     if isinstance(batch, Observations):
        #         observations = batch
        #         rewards = None
        #     else:
        #         assert isinstance(batch, tuple) and len(batch) == 2
        #         observations, rewards = batch

        #     if rewards is None:
        #         # Need to send actions to the env before we can actually get the
        #         # associated Reward.
        #         # Here we sample a random action (no other choice really..) and so we
        #         # are going to get bad results in case the online performance is being
        #         # evaluated.
        #         action = self.env.action_space.sample()
        #         rewards = self.env.send(action)
        #         assert False, (observations.shapes, action.shapes, rewards.shapes)
        #     all_observations.append(observations)
        #     all_rewards.append(rewards)
        # # TODO: This will be absolutely unfeasable for larger dataset like ImageNet.
        # stacked_observations: Observations = Observations.concatenate(all_observations)

        # assert all(
        #     y_i is not None for y in all_rewards for y_i in y
        # ), "Need fully labeled train dataset for now."
        # stacked_rewards: Rewards = Rewards.concatenate(all_rewards)

        # dataset = TensorDataset(stacked_observations.x, stacked_rewards.y)
        # dataset = AvalancheDataset(
        #     dataset=dataset,
        #     task_labels=stacked_observations.task_labels.tolist(),
        #     targets=stacked_rewards.y.tolist(),
        # )
        class DummyDataset(AvalancheDataset):
            pass
            def train(self):
                return self
 
        self._dataset = ...
        self.tasks_pattern_indices = {} #dict({0: np.arange(len(self._dataset))})
        self.task_set = ... #_TaskSubsetDict(self._dataset)
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

