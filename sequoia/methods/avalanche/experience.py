""" 'Wrapper' around a PassiveEnvironment from Sequoia, disguising it as an 'Experience'
from Avalanche.
"""
from typing import List, Optional

import tqdm
from avalanche.benchmarks.scenarios import Experience
from avalanche.benchmarks.utils.avalanche_dataset import (
    AvalancheDataset,
    AvalancheDatasetType,
)
from torch.utils.data import TensorDataset

from sequoia.common.gym_wrappers.utils import IterableWrapper
from sequoia.settings.passive import (
    ClassIncrementalSetting,
    PassiveEnvironment,
    PassiveSetting,
)
from sequoia.settings.passive.cl.objects import Observations, Rewards


class SequoiaExperience(IterableWrapper, Experience):
    def __init__(self, env: PassiveEnvironment, setting: ClassIncrementalSetting):
        super().__init__(env=env)
        self.setting = setting
        self.type: str
        self.task_id = setting.current_task_id
        if env is setting.train_env:
            self.type = "Train"
            self.transforms = setting.train_transforms
        elif env is setting.val_env:
            self.type = "Valid"
            self.transforms = setting.val_transforms
        else:
            self.type = "Test"
            assert env is setting.test_env
            self.transforms = setting.test_transforms
        self.name = f"{self.type}_{self.task_id}"

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
                if observations.batch_size != action.shape[0]:
                    action = action[: observations.batch_size]

                rewards = self.env.send(action)
            all_observations.append(observations)
            all_rewards.append(rewards)
        # TODO: This will be absolutely unfeasable for larger dataset like ImageNet.
        stacked_observations: Observations = Observations.concatenate(all_observations)

        assert all(
            y_i is not None for y in all_rewards for y_i in y
        ), "Need fully labeled train dataset for now."
        stacked_rewards: Rewards = Rewards.concatenate(all_rewards)

        dataset = TensorDataset(stacked_observations.x, stacked_rewards.y)
        self._tensor_dataset = dataset
        self._dataset = AvalancheDataset(
            dataset=dataset,
            task_labels=stacked_observations.task_labels.tolist(),
            targets=stacked_rewards.y.tolist(),
            dataset_type=AvalancheDatasetType.CLASSIFICATION,
        )
        # self.task_pattern_indices = {}
        # self.task_set = ...

        # class DummyDataset(AvalancheDataset):
        #     pass
        #     def train(self):
        #         return self

        # self._dataset = self
        # self.tasks_pattern_indices = {} #dict({0: np.arange(len(self._dataset))})
        # self.task_set = ... #_TaskSubsetDict(self._dataset)
        # self._dataset = env
        # from avalanche.benchmarks import GenericScenarioStream
        # class FakeStream(GenericScenarioStream):
        #     pass
        # self.origin_stream = FakeStream("train", scenario="whatever")
        # self.origin_stream.name = "train"

    @property
    def dataset(self) -> AvalancheDataset:
        return self._dataset

    @dataset.setter
    def dataset(self, value: AvalancheDataset) -> None:
        self._dataset = value

    @property
    def task_label(self):
        """
        The task label. This value will never have value "None". However,
        for scenarios that don't produce task labels a placeholder value like 0
        is usually set. Beware that this field is meant as a shortcut to obtain
        a unique task label: it assumes that only patterns labeled with a
        single task label are present. If this experience contains patterns from
        multiple tasks, accessing this property will result in an exception.
        """
        if not self.setting.task_labels_at_test_time:
            return 0
        if self.type == "Test" and self.setting.task_labels_at_test_time:
            raise RuntimeError("More than one tasks present, can't use this property.")
        return self.task_id

    @property
    def task_labels(self):
        return self._tensor_dataset.tensors[-1]

    @property
    def current_experience(self):
        # Return the index of the
        return self.task_id

    @property
    def origin_stream(self) -> PassiveSetting:
        # NOTE: This
        class DummyStream(list):
            name = self.name

        # raise NotImplementedError
        return DummyStream()

    # def train(self):
    #     return self
