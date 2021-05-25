""" Method that uses the SAC model from stable-baselines3 and targets the RL
settings in the tree.
"""
from dataclasses import dataclass
from typing import ClassVar, Optional, Type, Union, Callable

import gym
from gym import spaces
from simple_parsing import mutable_field
from stable_baselines3.sac.sac import SAC

from sequoia.methods import register_method
from sequoia.common.hparams import log_uniform
from sequoia.settings.rl import ContinualRLSetting
from sequoia.utils.logging_utils import get_logger
from .off_policy_method import OffPolicyMethod, OffPolicyModel

logger = get_logger(__file__)


class SACModel(SAC, OffPolicyModel):
    """ Customized version of the SAC model from stable-baselines-3. """

    @dataclass
    class HParams(OffPolicyModel.HParams):
        """ Hyper-parameters of the SAC Model. """
        # The learning rate, it can be a function of the current progress (from
        # 1 to 0)
        learning_rate: Union[float, Callable] = log_uniform(1e-6, 1e-2, default=3e-4)
        buffer_size: int = 1_000_000
        learning_starts: int = 100
        batch_size: int = 256
        tau: float = 0.005
        gamma: float = 0.99
        train_freq = 1
        gradient_steps: int = 1
        # action_noise: Optional[ActionNoise] = None
        optimize_memory_usage: bool = False
        ent_coef: Union[str, float] = "auto"
        target_update_interval: int = 1
        target_entropy: Union[str, float] = "auto"
        use_sde: bool = False
        sde_sample_freq: int = -1


@register_method
@dataclass
class SACMethod(OffPolicyMethod):
    """ Method that uses the SAC model from stable-baselines3. """

    Model: ClassVar[Type[SACModel]] = SACModel

    # Hyper-parameters of the SAC model.
    hparams: SACModel.HParams = mutable_field(SACModel.HParams)

    # Approximate limit on the size of the replay buffer, in megabytes.
    max_buffer_size_megabytes: float = 2_048.0

    def configure(self, setting: ContinualRLSetting):
        super().configure(setting)

    def create_model(self, train_env: gym.Env, valid_env: gym.Env) -> SACModel:
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
        super().on_task_switch(task_id=task_id)


if __name__ == "__main__":
    results = SACMethod.main()
    print(results)
