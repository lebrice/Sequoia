""" Method that uses the DQN model from stable-baselines3 and targets the RL
settings in the tree.
"""
import warnings
from dataclasses import dataclass
from typing import Callable, ClassVar, Optional, Type, Union

import gym
from gym import spaces
from gym.spaces.utils import flatten_space
from simple_parsing import mutable_field
from stable_baselines3.dqn import DQN

from sequoia.common.hparams import uniform, log_uniform, categorical
from sequoia.methods import register_method
from sequoia.methods.stable_baselines3_methods.base import (
    SB3BaseHParams, StableBaselines3Method)
from sequoia.settings.active import ContinualRLSetting
from sequoia.utils.logging_utils import get_logger

logger = get_logger(__file__)


class DQNModel(DQN):
    """ Customized version of the DQN model from stable-baselines-3. """
    @dataclass
    class HParams(SB3BaseHParams):
        """ Hyper-parameters of the DQN model from `stable_baselines3`.

        The command-line arguments for these are created with simple-parsing.
        """
        # The learning rate, it can be a function of the current progress (from
        # 1 to 0)
        learning_rate: Union[float, Callable] = log_uniform(1e-6, 1e-2, default=1e-4)
        # size of the replay buffer
        buffer_size: int = uniform(100, 10_000_000, default=1_000_000)
        # How many steps of the model to collect transitions for before learning
        # starts.
        # learning_starts: int = uniform(1_000, 100_000, default=50_000)
        learning_starts: int = 50_000
        # Minibatch size for each gradient update
        # batch_size: Optional[int] = categorical(1, 2, 4, 8, 16, 32, 128, default=32)
        batch_size: int = 32
        # The soft update coefficient ("Polyak update", between 0 and 1) default
        # 1 for hard update
        # tau: float = uniform(0., 1., default=1.0)
        tau: float = 1.0
        # The discount factor
        # gamma: float = uniform(0.9, 0.9999, default=0.99)
        gamma: float = 0.99
        # Update the model every ``train_freq`` steps. Set to `-1` to disable.
        train_freq: int = categorical(1, 10, 100, 1_000, 10_000, default=10)
        # train_freq: int = 4
        # How many gradient steps to do after each rollout (see ``train_freq``
        # and ``n_episodes_rollout``) Set to ``-1`` means to do as many gradient
        # steps as steps done in the environment during the rollout.
        # gradient_steps: int = categorical(1, -1, default=1)
        gradient_steps: int = 1
        # Enable a memory efficient variant of the replay buffer at a cost of
        # more complexity.
        # See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        optimize_memory_usage: bool = False
        # Update the target network every ``target_update_interval`` environment
        # steps.
        target_update_interval: int = categorical(1, 10, 100, 1_000, 10_000, default=10_000)
        # Fraction of entire training period over which the exploration rate is
        # reduced.
        # exploration_fraction: float = uniform(0.05, 0.3, default=0.1)
        exploration_fraction: float = 0.1
        # Initial value of random action probability.
        # exploration_initial_eps: float = uniform(0.5, 1.0, default=1.0)
        exploration_initial_eps: float = 1.0
        # final value of random action probability.
        # exploration_final_eps: float = uniform(0, 0.1, default=0.05)
        exploration_final_eps: float = 0.05
        # The maximum value for the gradient clipping.
        # max_grad_norm: float = uniform(1, 100, default=10)
        max_grad_norm: float = 10
        # Whether to create a second environment that will be used for
        # evaluating the agent periodically. (Only available when passing string
        # for the environment)
        create_eval_env: bool = False
        # Whether or not to build the network at the creation
        # of the instance
        _init_setup_model: bool = True


@register_method
@dataclass
class DQNMethod(StableBaselines3Method):
    """ Method that uses a DQN model from the stable-baselines3 package. """
    Model: ClassVar[Type[DQNModel]] = DQNModel

    # Hyper-parameters of the DQN model.
    hparams: DQNModel.HParams = mutable_field(DQNModel.HParams)

    # Approximate limit on the size of the replay buffer, in megabytes.
    max_buffer_size_megabytes: float = 2_048.

    def configure(self, setting: ContinualRLSetting):
        super().configure(setting)

        # The default value for the buffer size in the DQN model is WAY too
        # large, so we re-size it depending on the size of the observations.

        flattened_observation_space = flatten_space(setting.observation_space)
        observation_size_bytes = flattened_observation_space.sample().nbytes

        # IF there are more than a few dimensions per observation, then we
        # should probably reduce the size of the replay buffer according to
        # the size of the observations.
        max_buffer_size_bytes = self.max_buffer_size_megabytes * 1024 * 1024
        max_buffer_length = max_buffer_size_bytes // observation_size_bytes

        if max_buffer_length == 0:
            raise RuntimeError(
                f"Couldn't even fit a single observation in the buffer, "
                f"given the  specified max_buffer_size_megabytes "
                f"({self.max_buffer_size_megabytes}) and the size of a "
                f"single observation ({observation_size_bytes} bytes)!"
            )

        if self.hparams.buffer_size > max_buffer_length:
            calculated_size_bytes = observation_size_bytes * self.hparams.buffer_size
            calculated_size_gb = calculated_size_bytes / 1024 ** 3
            warnings.warn(RuntimeWarning(
                f"The selected buffer size ({self.hparams.buffer_size} is "
                f"too large! (It would take roughly around "
                f"{calculated_size_gb:.3f}Gb to hold  many observations alone! "
                f"The buffer size will be capped at {max_buffer_length} "
                f"entries."
            ))

            self.hparams.buffer_size = int(max_buffer_length)

        # Don't use up too many of the observations from the task to fill up the buffer.
        # Truth is, we should probably get this to work first.
        
        # NOTE: Need to change some attributes depending on the maximal number of steps
        # in the environment allowed in the given Setting.
        if setting.max_steps:
            logger.info(
                f"Total training steps are limited to {setting.steps_per_task} steps "
                f"per task, {setting.max_steps} steps in total."
            )
            ten_percent_of_step_budget = setting.steps_per_task // 10
            
            if self.hparams.buffer_size > ten_percent_of_step_budget:
                warnings.warn(RuntimeWarning(
                    "Reducing max buffer size to ten percent of the step budget."
                ))
                self.hparams.buffer_size = ten_percent_of_step_budget

            if self.hparams.learning_starts > ten_percent_of_step_budget:
                logger.info(
                    f"The model was originally going to use the first "
                    f"{self.hparams.learning_starts} steps for pure random "
                    f"exploration, but the setting has a max number of steps set to "
                    f"{setting.max_steps}, therefore we will limit the number of "
                    f"exploration steps to 10% of that 'step budget' = "
                    f"{ten_percent_of_step_budget} steps."
                )
                self.hparams.learning_starts = ten_percent_of_step_budget

            if self.hparams.target_update_interval > ten_percent_of_step_budget:
                # Same for the 'update target network' interval.
                self.hparams.target_update_interval = ten_percent_of_step_budget // 2
                logger.info(
                    f"Reducing the target network update interval to "
                    f"{self.hparams.target_update_interval}, because of the limit on "
                    f"training steps imposed by the Setting."
                )

        logger.info(f"Will use a Replay buffer of size {self.hparams.buffer_size}.")

    def create_model(self, train_env: gym.Env, valid_env: gym.Env) -> DQNModel:
        return self.Model(env=train_env, **self.hparams.to_dict())

    def fit(self, train_env: gym.Env, valid_env: gym.Env):
        super().fit(train_env=train_env, valid_env=valid_env)

    def get_actions(self,
                    observations: ContinualRLSetting.Observations,
                    action_space: spaces.Space) -> ContinualRLSetting.Actions:
        return super().get_actions(
            observations=observations,
            action_space=action_space,
        )

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
