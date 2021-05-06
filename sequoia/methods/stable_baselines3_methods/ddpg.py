""" Method that uses the DDPG model from stable-baselines3 and targets the RL
settings in the tree.
"""
from dataclasses import dataclass
from typing import Callable, ClassVar, Optional, Type, Union

import gym
from gym import spaces
from simple_parsing import mutable_field
from stable_baselines3.ddpg import DDPG
from stable_baselines3.common.off_policy_algorithm import TrainFreq

from sequoia.common.hparams import log_uniform
from sequoia.methods import register_method
from sequoia.settings.active import ContinualRLSetting
from sequoia.utils.logging_utils import get_logger

from .off_policy_method import OffPolicyModel, OffPolicyMethod
logger = get_logger(__file__)


class DDPGModel(DDPG, OffPolicyModel):
    """ Customized version of the DDPG model from stable-baselines-3. """

    @dataclass
    class HParams(OffPolicyModel.HParams):
        """ Hyper-parameters of the DDPG Model. """
        # TODO: Add hparams specific to DDPG here.
        # The learning rate, it can be a function of the current progress (from
        # 1 to 0)
        learning_rate: Union[float, Callable] = log_uniform(1e-6, 1e-2, default=1e-3)

        # The verbosity level: 0 none, 1 training information, 2 debug
        verbose: int = 0

        train_freq: TrainFreq = (1, "episode")

        # Minibatch size for each gradient update
        batch_size: int = 100

        # How many gradient steps to do after each rollout (see ``train_freq``
        # and ``n_episodes_rollout``) Set to ``-1`` means to do as many gradient
        # steps as steps done in the environment during the rollout.
        gradient_steps: int = -1
        # gradient_steps: int = categorical(1, -1, default=-1)


@register_method
@dataclass
class DDPGMethod(OffPolicyMethod):
    """ Method that uses the DDPG model from stable-baselines3. """

    Model: ClassVar[Type[DDPGModel]] = DDPGModel

    # Hyper-parameters of the DDPG model.
    hparams: DDPGModel.HParams = mutable_field(DDPGModel.HParams)

    # Approximate limit on the size of the replay buffer, in megabytes.
    max_buffer_size_megabytes: float = 2_048.0

    def configure(self, setting: ContinualRLSetting):
        super().configure(setting)

    def create_model(self, train_env: gym.Env, valid_env: gym.Env) -> DDPGModel:
        return self.Model(env=train_env, **self.hparams.to_dict())

    def fit(self, train_env: gym.Env, valid_env: gym.Env):
        super().fit(train_env=train_env, valid_env=valid_env)

    def get_actions(
        self, observations: ContinualRLSetting.Observations, action_space: spaces.Space
    ) -> ContinualRLSetting.Actions:
        return super().get_actions(
            observations=observations, action_space=action_space,
        )

    def on_task_switch(self, task_id: Optional[int]) -> None:
        """ Called when switching tasks in a CL setting.

        If task labels are available, `task_id` will correspond to the index of
        the new task. Otherwise, if task labels aren't available, `task_id` will
        be `None`.

        todo: use this to customize how your method handles task transitions.
        """


if __name__ == "__main__":
    results = DDPGMethod.main()
    print(results)
