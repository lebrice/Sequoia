""" Base class used to not duplicate the tweaks made all the on-policy algos from SB3.
"""
import math
from abc import ABC
from dataclasses import dataclass
from typing import Callable, ClassVar, Dict, Mapping, Optional, Type, Union

import gym
import torch
from gym import spaces
from simple_parsing import mutable_field
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

from sequoia.common.hparams import uniform, log_uniform
from sequoia.settings.active import ContinualRLSetting
from sequoia.utils.logging_utils import get_logger

from .base import SB3BaseHParams, StableBaselines3Method

logger = get_logger(__file__)


class OnPolicyModel(OnPolicyAlgorithm, ABC):
    """ Tweaked version of the OnPolicyAlgorithm from SB3. """

    @dataclass
    class HParams(SB3BaseHParams):
        """ Hyper-parameters common to all on-policy algos from SB3. """

        # learning rate for the optimizer, it can be a function of the current
        # progress remaining (from 1 to 0)
        learning_rate: Union[float, Callable] = log_uniform(1e-7, 1e-2, default=1e-3)
        # The number of steps to run for each environment per update (i.e. batch size
        # is n_steps * n_env where n_env is number of environment copies running in
        # parallel)
        # NOTE: Default value here is much lower than in PPO, which might indicate
        # that this A2C is more "on-policy"? (i.e. that it requires data to be super
        # "fresh")?
        n_steps: int = uniform(3, 64, default=5, discrete=True)
        # Discount factor
        gamma: float = 0.99
        # gamma: float = uniform(0.9, 0.9999, default=0.99)

        # Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        # Equivalent to classic advantage when set to 1.
        gae_lambda: float = 1.0
        # gae_lambda: float = uniform(0.5, 1.0, default=1.0)

        # Entropy coefficient for the loss calculation
        ent_coef: float = 0.0
        # ent_coef: float = uniform(0.0, 1.0, default=0.0)

        # Value function coefficient for the loss calculation
        vf_coef: float = 0.5
        # vf_coef: float = uniform(0.01, 1.0, default=0.5)

        # The maximum value for the gradient clipping
        max_grad_norm: float = 0.5
        # max_grad_norm: float = uniform(0.1, 10, default=0.5)

        # Whether to use generalized State Dependent Exploration (gSDE) instead of
        # action noise exploration (default: False)
        use_sde: bool = False
        # use_sde: bool = categorical(True, False, default=False)

        # Sample a new noise matrix every n steps when using gSDE.
        # Default: -1 (only sample at the beginning of the rollout)
        sde_sample_freq: int = -1
        # sde_sample_freq: int = categorical(-1, 1, 5, 10, default=-1)

        # The log location for tensorboard (if None, no logging)
        tensorboard_log: Optional[str] = None

        # # Whether to create a second environment that will be used for evaluating the
        # # agent periodically. (Only available when passing string for the environment)
        # create_eval_env: bool = False

        # # Additional arguments to be passed to the policy on creation
        # policy_kwargs: Optional[Dict[str, Any]] = None

        # The verbosity level: 0 no output, 1 info, 2 debug
        verbose: int = 1

        # Seed for the pseudo random generators
        seed: Optional[int] = None

        # Device (cpu, cuda, ...) on which the code should be run.
        # Setting it to auto, the code will be run on the GPU if possible.
        device: Union[torch.device, str] = "auto"

        # :param _init_setup_model: Whether or not to build the network at the
        # creation of the instance
        # _init_setup_model: bool = True


@dataclass
class OnPolicyMethod(StableBaselines3Method, ABC):
    """ Method that uses the A2C model from stable-baselines3. """

    Model: ClassVar[Type[OnPolicyModel]] = OnPolicyModel

    # Hyper-parameters of the model/algorithm.
    hparams: OnPolicyModel.HParams = mutable_field(OnPolicyModel.HParams)

    def configure(self, setting: ContinualRLSetting):
        super().configure(setting=setting)
        if setting.steps_per_phase:
            if self.hparams.n_steps > setting.steps_per_phase:
                self.hparams.n_steps = math.ceil(0.1 * setting.steps_per_phase)
                logger.info(
                    f"Capping the n_steps to 10% of step budget length: "
                    f"{self.hparams.n_steps}"
                )
            # NOTE: We limit the number of trainign steps per task, such that we never
            # attempt to fill the buffer using more samples than the environment allows.
            self.train_steps_per_task = min(
                self.train_steps_per_task,
                setting.steps_per_phase - self.hparams.n_steps - 1,
            )
            logger.info(
                f"Limitting training steps per task to {self.train_steps_per_task}"
            )

    def create_model(self, train_env: gym.Env, valid_env: gym.Env) -> OnPolicyModel:
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

    def get_search_space(
        self, setting: ContinualRLSetting
    ) -> Mapping[str, Union[str, Dict]]:
        search_space = super().get_search_space(setting)
        if isinstance(setting.action_space, spaces.Discrete):
            # From stable_baselines3/common/base_class.py", line 170:
            # > Generalized State-Dependent Exploration (gSDE) can only be used with
            #   continuous actions
            # Therefore we remove related entries in the search space, so they keep
            # their default values.
            search_space.pop("use_sde", None)
            search_space.pop("sde_sample_freq", None)
        return search_space
