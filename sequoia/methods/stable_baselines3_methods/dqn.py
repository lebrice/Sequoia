""" Method that uses the DQN model from stable-baselines3 and targets the RL
settings in the tree.
"""
from dataclasses import dataclass
from typing import Callable, ClassVar, Optional, Type, Union

import gym
from gym import spaces
from sequoia.common.hparams import categorical
from sequoia.common.transforms import ChannelsFirst
from sequoia.methods import register_method
from sequoia.settings.active import ContinualRLSetting
from sequoia.utils.logging_utils import get_logger
from simple_parsing import mutable_field
from simple_parsing.helpers.hparams import log_uniform, uniform
from stable_baselines3.dqn import DQN

from .off_policy_method import OffPolicyMethod, OffPolicyModel

logger = get_logger(__file__)


class DQNModel(DQN, OffPolicyModel):
    """ Customized version of the DQN model from stable-baselines-3. """

    @dataclass
    class HParams(OffPolicyModel.HParams):
        """ Hyper-parameters of the DQN model from `stable_baselines3`.

        The command-line arguments for these are created with simple-parsing.
        """

        # ------------------
        # overwritten hparams
        # The learning rate, it can be a function of the current progress (from
        # 1 to 0)
        learning_rate: Union[float, Callable] = log_uniform(1e-6, 1e-2, default=1e-4)
        # size of the replay buffer
        buffer_size: int = uniform(100_000, 10_000_000, default=1_000_000)
        # --------------------

        # How many steps of the model to collect transitions for before learning
        # starts.
        learning_starts: int = 50_000

        # Minibatch size for each gradient update
        batch_size: int = 32

        # Update the model every ``train_freq`` steps. Set to `-1` to disable.
        train_freq: int = 4
        # train_freq: int = categorical(1, 10, 100, 1_000, 10_000, default=4)

        # The soft update coefficient ("Polyak update", between 0 and 1) default
        # 1 for hard update
        tau: float = 1.0
        # tau: float = uniform(0., 1., default=1.0)
        # Update the target network every ``target_update_interval`` environment
        # steps.
        target_update_interval: int = categorical(
            1, 10, 100, 1_000, 10_000, default=10_000
        )
        # Fraction of entire training period over which the exploration rate is
        # reduced.
        exploration_fraction: float = 0.1
        # exploration_fraction: float = uniform(0.05, 0.3, default=0.1)
        # Initial value of random action probability.
        exploration_initial_eps: float = 1.0
        # exploration_initial_eps: float = uniform(0.5, 1.0, default=1.0)
        # final value of random action probability.
        exploration_final_eps: float = 0.05
        # exploration_final_eps: float = uniform(0, 0.1, default=0.05)
        # The maximum value for the gradient clipping.
        max_grad_norm: float = 10
        # max_grad_norm: float = uniform(1, 100, default=10)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        super().train(gradient_steps, batch_size=batch_size)


@register_method
@dataclass
class DQNMethod(OffPolicyMethod):
    """ Method that uses a DQN model from the stable-baselines3 package. """

    Model: ClassVar[Type[DQNModel]] = DQNModel

    # Hyper-parameters of the DQN model.
    hparams: DQNModel.HParams = mutable_field(DQNModel.HParams)

    # Approximate limit on the size of the replay buffer, in megabytes.
    max_buffer_size_megabytes: float = 1_024*10.

    def configure(self, setting: ContinualRLSetting):
        super().configure(setting)
        # NOTE: Need to change some attributes depending on the maximal number of steps
        # in the environment allowed in the given Setting.
        if setting.steps_per_phase:
            ten_percent_of_step_budget = setting.steps_per_phase // 10
            if self.hparams.target_update_interval > ten_percent_of_step_budget:
                # Same for the 'update target network' interval.
                self.hparams.target_update_interval = ten_percent_of_step_budget // 2
                logger.info(
                    f"Reducing the target network update interval to "
                    f"{self.hparams.target_update_interval}, because of the limit on "
                    f"training steps imposed by the Setting."
                )

    def create_model(self, train_env: gym.Env, valid_env: gym.Env) -> DQNModel:
        return self.Model(env=train_env, **self.hparams.to_dict())

    def fit(self, train_env: gym.Env, valid_env: gym.Env):
        super().fit(train_env=train_env, valid_env=valid_env)

    def get_actions(
        self, observations: ContinualRLSetting.Observations, action_space: spaces.Space
    ) -> ContinualRLSetting.Actions:
        obs = observations.x
        # Temp fix for monsterkong and DQN:
        if obs.shape == (64, 64, 3):
            obs = ChannelsFirst.apply(obs)
        predictions = self.model.predict(obs)
        action, _ = predictions
        assert action in action_space, (observations, action, action_space)
        return action

    def on_task_switch(self, task_id: Optional[int]) -> None:
        """ Called when switching tasks in a CL setting.

        If task labels are available, `task_id` will correspond to the index of
        the new task. Otherwise, if task labels aren't available, `task_id` will
        be `None`.

        todo: use this to customize how your method handles task transitions.
        """


if __name__ == "__main__":
    results = DQNMethod.main()
    print(results)
