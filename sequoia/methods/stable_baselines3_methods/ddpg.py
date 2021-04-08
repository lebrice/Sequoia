""" Method that uses the DDPG model from stable-baselines3 and targets the RL
settings in the tree.
"""
import math
import warnings
from dataclasses import dataclass
from typing import Callable, ClassVar, Optional, Type, Union

import gym
from gym import spaces
from gym.spaces.utils import flatten_space
from simple_parsing import mutable_field
from stable_baselines3.ddpg import DDPG

from sequoia.common.hparams import categorical, log_uniform, uniform
from sequoia.methods import register_method
from sequoia.settings.active import ContinualRLSetting
from sequoia.utils.logging_utils import get_logger

from .base import SB3BaseHParams, StableBaselines3Method
logger = get_logger(__file__)


class DDPGModel(DDPG):
    """ Customized version of the DDPG model from stable-baselines-3. """

    @dataclass
    class HParams(SB3BaseHParams):
        """ Hyper-parameters of the DDPG Model. """
        # The learning rate, it can be a function of the current progress (from
        # 1 to 0)
        learning_rate: Union[float, Callable] = log_uniform(1e-6, 1e-2, default=1e-4)
        # size of the replay buffer
        buffer_size: int = uniform(100, 10_000_000, default=1_000_000)
        # How many steps of the model to collect transitions for before learning
        # starts.
        learning_starts: int = 50_000
        # learning_starts: int = uniform(1_000, 100_000, default=50_000)

        # Minibatch size for each gradient update
        batch_size: Optional[int] = 32
        # batch_size: Optional[int] = categorical(1, 2, 4, 8, 16, 32, 128, default=32)

        # The soft update coefficient ("Polyak update", between 0 and 1) default
        # 1 for hard update
        tau: float = 1.0
        # tau: float = uniform(0., 1., default=1.0)

        # The discount factor
        gamma: float = 0.99
        # gamma: float = uniform(0.9, 0.9999, default=0.99)

        # Update the model every ``train_freq`` steps. Set to `-1` to disable.
        train_freq: int = 10
        # train_freq: int = categorical(1, 10, 100, 1_000, 10_000, default=10)

        # How many gradient steps to do after each rollout (see ``train_freq``
        # and ``n_episodes_rollout``) Set to ``-1`` means to do as many gradient
        # steps as steps done in the environment during the rollout.
        gradient_steps: int = 1
        # gradient_steps: int = categorical(-1, 1, default=1)

        # Enable a memory efficient variant of the replay buffer at a cost of
        # more complexity.
        # See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        optimize_memory_usage: bool = False
        # Whether to create a second environment that will be used for
        # evaluating the agent periodically. (Only available when passing string
        # for the environment)
        create_eval_env: bool = False
        # Whether or not to build the network at the creation
        # of the instance
        _init_setup_model: bool = True


@register_method
@dataclass
class DDPGMethod(StableBaselines3Method):
    """ Method that uses the DDPG model from stable-baselines3. """

    Model: ClassVar[Type[DDPGModel]] = DDPGModel

    # Hyper-parameters of the DDPG model.
    hparams: DDPGModel.HParams = mutable_field(DDPGModel.HParams)

    # Approximate limit on the size of the replay buffer, in megabytes.
    max_buffer_size_megabytes: float = 2_048.0

    def configure(self, setting: ContinualRLSetting):
        super().configure(setting)
        # TODO: This is mostly just copied from DQN, there is probably need for a new
        # OffPolicyMethod class that would contain the code common to both?

        # The default value for the buffer size in the DQN model is WAY too
        # large, so we re-size it depending on the size of the observations.
        # NOTE: (issue #156) Only consider the images, not the task labels for these
        # buffer size calculations (since the task labels might be None and have the
        # np.object dtype).
        x_space = setting.observation_space.x
        flattened_observation_space = flatten_space(x_space)
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
            warnings.warn(
                RuntimeWarning(
                    f"The selected buffer size ({self.hparams.buffer_size} is "
                    f"too large! (It would take roughly around "
                    f"{calculated_size_gb:.3f}Gb to hold  many observations alone! "
                    f"The buffer size will be capped at {max_buffer_length} "
                    f"entries."
                )
            )

            self.hparams.buffer_size = int(max_buffer_length)

        # NOTE: Need to change some attributes depending on the maximal number of steps
        # in the environment allowed in the given Setting.
        if setting.max_steps:
            logger.info(
                f"Total training steps are limited to {setting.steps_per_task} steps "
                f"per task, {setting.max_steps} steps in total."
            )
            ten_percent_of_step_budget = setting.steps_per_phase // 10

            if self.hparams.buffer_size > ten_percent_of_step_budget:
                warnings.warn(
                    RuntimeWarning(
                        "Reducing max buffer size to ten percent of the step budget."
                    )
                )
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
                if self.hparams.train_freq != -1:
                    # Update the model at least 2 times during each task, and at most
                    # once per step.
                    self.hparams.train_freq = min(
                        self.hparams.train_freq, int(0.5 * ten_percent_of_step_budget),
                    )
                    self.hparams.train_freq = max(self.hparams.train_freq, 1)

                logger.info(f"Training frequency: {self.hparams.train_freq}")

        logger.info(f"Will use a Replay buffer of size {self.hparams.buffer_size}.")

        if setting.steps_per_phase:
            assert (
                isinstance(self.hparams.train_freq, int)
                or self.hparams.train_freq[1] == "steps"
            ), "Need the train freq units to be steps for now."

            # NOTE: We limit the number of training steps per task, such that we never
            # attempt to fill the buffer using more samples than the environment allows.
            if self.hparams.train_freq > setting.steps_per_phase:
                self.hparams.n_steps = math.ceil(0.1 * setting.steps_per_phase)
                logger.info(
                    f"Capping the n_steps to 10% of step budget length: "
                    f"{self.hparams.n_steps}"
                )

            self.train_steps_per_task = min(
                self.train_steps_per_task,
                setting.steps_per_phase - self.hparams.train_freq - 1,
            )
            logger.info(
                f"Limitting training steps per task to {self.train_steps_per_task}"
            )

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
