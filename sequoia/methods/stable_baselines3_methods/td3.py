""" TODO: Implement and test DDPG. """
from dataclasses import dataclass
from typing import Callable, ClassVar, Optional, Type, Union

import gym
from gym import spaces
from simple_parsing import mutable_field
from stable_baselines3.td3 import TD3
from stable_baselines3.common.off_policy_algorithm import TrainFreq

from sequoia.common.hparams import log_uniform
from sequoia.methods import register_method
from sequoia.settings.rl import ContinualRLSetting
from sequoia.utils.logging_utils import get_logger
from .off_policy_method import OffPolicyMethod, OffPolicyModel

logger = get_logger(__file__)


class TD3Model(TD3, OffPolicyModel):
    @dataclass
    class HParams(OffPolicyModel.HParams):
        """ Hyper-parameters of the TD3 model. """
        # TODO: Add HParams specific to TD3 here, if any, and also check that the
        # default values are correct.

        # The learning rate, it can be a function of the current progress (from
        # 1 to 0)
        learning_rate: Union[float, Callable] = log_uniform(1e-6, 1e-2, default=1e-3)

        # Minibatch size for each gradient update
        batch_size: int = 100
        # batch_size: int = categorical(1, 2, 4, 8, 16, 32, 128, default=32)

        train_freq: TrainFreq = (1, "episode")

        # How many gradient steps to do after each rollout (see ``train_freq``
        # and ``n_episodes_rollout``) Set to ``-1`` means to do as many gradient
        # steps as steps done in the environment during the rollout.
        gradient_steps: int = -1
        # gradient_steps: int = categorical(1, -1, default=1)


@register_method
@dataclass
class TD3Method(OffPolicyMethod):
    """ Method that uses the TD3 model from stable-baselines3. """

    Model: ClassVar[Type[TD3Model]] = TD3Model
    hparams: TD3Model.HParams = mutable_field(TD3Model.HParams)

    # Approximate limit on the size of the replay buffer, in megabytes.
    max_buffer_size_megabytes: float = 2_048.0

    def configure(self, setting: ContinualRLSetting):
        super().configure(setting)

    def create_model(self, train_env: gym.Env, valid_env: gym.Env) -> TD3Model:
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
    results = TD3Method.main()
    print(results)
