import warnings
from dataclasses import dataclass
from typing import ClassVar, Type, Union, Callable, Optional, Dict, Any

import gym
import numpy as np
import torch
from gym import Env, spaces
from simple_parsing import choice, mutable_field
from stable_baselines3.dqn import DQN
from stable_baselines3.dqn.dqn import DQNPolicy
from stable_baselines3.dqn.policies import CnnPolicy
from stable_baselines3.common.base_class import BaseAlgorithm, GymEnv
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

from settings import ContinualRLSetting
from utils import Serializable, Parseable
from methods import register_method

from .base import StableBaselines3Method, SB3BaseHParams
from utils.logging_utils import get_logger

logger = get_logger(__file__)

class DQNModel(DQN):
    @dataclass
    class HParams(SB3BaseHParams):
        """ Hyper-parameters of the DQN model from `stable_baselines3`.
        
        The command-line arguments for these are created with simple-parsing.
        """
        # The learning rate, it can be a function of the current progress (from
        # 1 to 0)
        learning_rate: Union[float, Callable] = 1e-4
        # size of the replay buffer
        buffer_size: int = 1000000
        # How many steps of the model to collect transitions for before learning
        # starts
        learning_starts: int = 50000
        # Minibatch size for each gradient update
        batch_size: Optional[int] = 32
        # The soft update coefficient ("Polyak update", between 0 and 1) default
        # 1 for hard update
        tau: float = 1.0
        # The discount factor
        gamma: float = 0.99
        # Update the model every ``train_freq`` steps. Set to `-1` to disable.
        train_freq: int = 4
        # How many gradient steps to do after each rollout (see ``train_freq``
        # and ``n_episodes_rollout``) Set to ``-1`` means to do as many gradient
        # steps as steps done in the environment during the rollout.
        gradient_steps: int = 1
        # Update the model every ``n_episodes_rollout`` episodes. Note that this
        # cannot be used at the same time as ``train_freq``. Set to `-1` to
        # disable.
        n_episodes_rollout: int = -1
        # Enable a memory efficient variant of the replay buffer at a cost of
        # more complexity.
        # See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        optimize_memory_usage: bool = False
        # Update the target network every ``target_update_interval`` environment
        # steps.
        target_update_interval: int = 10000
        # Fraction of entire training period over which the exploration rate is
        # reduced.
        exploration_fraction: float = 0.1
        # Initial value of random action probability.
        exploration_initial_eps: float = 1.0
        # final value of random action probability.
        exploration_final_eps: float = 0.05
        # The maximum value for the gradient clipping.
        max_grad_norm: float = 10
        # Whether to create a second environment that will be used for
        # evaluating the agent periodically. (Only available when passing string
        # for the environment)
        create_eval_env: bool = False 
        # Whether or not to build the network at the creation
        # of the instance
        _init_setup_model: bool = True

    def __init__(self,
                 policy,
                 env,
                 learning_rate=0.0001,
                 buffer_size=1000000,
                 learning_starts=50000,
                 batch_size=32,
                 tau=1.0,
                 gamma=0.99,
                 train_freq=4,
                 gradient_steps=1,
                 n_episodes_rollout=-1,
                 optimize_memory_usage=False,
                 target_update_interval=10000,
                 exploration_fraction=0.1,
                 exploration_initial_eps=1.0,
                 exploration_final_eps=0.05,
                 max_grad_norm=10,
                 tensorboard_log=None,
                 create_eval_env=False,
                 policy_kwargs=None,
                 verbose=0,
                 seed=None,
                 device='auto',
                 _init_setup_model=True,
                ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            n_episodes_rollout=n_episodes_rollout,
            optimize_memory_usage=optimize_memory_usage,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            max_grad_norm=max_grad_norm,
            tensorboard_log=tensorboard_log,
            create_eval_env=create_eval_env,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

@register_method
@dataclass
class DQNMethod(StableBaselines3Method):
    """ Method that uses a DQN model from the stable-baselines3 package. """
    Model: ClassVar[Type[DQNModel]] = DQNModel
    
    # Hyper-parameters of the DQN model.
    hparams: DQNModel.HParams = mutable_field(DQNModel.HParams)

    # Approximate limit on the size of the replay buffer, in megabytes.
    max_buffer_size_megabytes: float = 1024
    
    def configure(self, setting: ContinualRLSetting):
        super().configure(setting)
        from gym.spaces.utils import flatdim, flatten_space

        observation_dims = flatdim(setting.observation_space)
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
            
            self.hparams.buffer_size = max_buffer_length

        # Don't use up too many of the observations from the task to fill up the buffer.
        # Truth is, we should probably get this to work first.
        if not setting.known_task_boundaries_at_train_time:
            if self.hparams.buffer_size > setting.steps_per_task // 10:
                warnings.warn(RuntimeWarning(
                    "Operating on a ContinualRL setting, so we're limiting the "
                    "buffer size to 1/10 the number of steps in the training "
                    "loop."
                ))
                self.hparams.buffer_size = setting.steps_per_task // 10
        logger.info(f"Will use a Replay buffer of size {self.hparams.buffer_size}.")

if __name__ == "__main__":
    from settings import RLSetting
    # setting = RLSetting(dataset="CartPole-v1", observe_state_directly=True)
    setting, unused_args = RLSetting.from_known_args()
    # method = DQNMethod()
    method = DQNMethod.from_args(unused_args, strict=True)

    results = setting.apply(method)
    print(results.summary())

    exit()
    
    