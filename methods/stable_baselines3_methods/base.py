""" Example of creating an A2C agent using the simplebaselines3 package.

See https://stable-baselines3.readthedocs.io/en/master/guide/install.html
"""
import warnings
from abc import ABC
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Dict, Optional, Type, Union

import gym
import torch
from gym import Env, spaces
from simple_parsing import choice, mutable_field

from stable_baselines3.common.base_class import (BaseAlgorithm, BasePolicy,
                                                 DummyVecEnv, GymEnv, VecEnv,
                                                 VecTransposeImage, MaybeCallback,
                                                 is_image_space, is_wrapped)
from stable_baselines3.common.vec_env.obs_dict_wrapper import ObsDictWrapper

from common.gym_wrappers.batch_env.batched_vector_env import VectorEnv
from methods import register_method
from settings import Method, all_settings
from settings.active.rl import ContinualRLSetting, ClassIncrementalRLSetting
from settings.active.rl.continual_rl_setting import ContinualRLSetting
from settings.active.rl.wrappers import (NoTypedObjectsWrapper,
                                         RemoveTaskLabelsWrapper)
from utils import Parseable, Serializable
from utils.logging_utils import get_logger

logger = get_logger(__file__)

# "Patch" the _wrap_env function of the BaseAlgorithm class of
# stable_baselines, to make it recognize the VectorEnv from gym.vector as a
# vectorized environment.
@staticmethod
def _wrap_env(env: GymEnv, verbose: int = 0) -> VecEnv:
    # NOTE: We just want to change this single line here:
    # if not isinstance(env, VecEnv):
    if not (isinstance(env, (VecEnv, VectorEnv)) or isinstance(env.unwrapped, (VecEnv, VectorEnv))):
        if verbose >= 1:
            print("Wrapping the env in a DummyVecEnv.")
        env = DummyVecEnv([lambda: env])
    
    if is_image_space(env.observation_space) and not is_wrapped(env, VecTransposeImage):
        if verbose >= 1:
            print("Wrapping the env in a VecTransposeImage.")
        env = VecTransposeImage(env)

    # check if wrapper for dict support is needed when using HER
    if isinstance(env.observation_space, gym.spaces.dict.Dict):
        env = ObsDictWrapper(env)

    return env

BaseAlgorithm._wrap_env = _wrap_env

@dataclass
class SB3BaseHParams(Serializable, Parseable):
    """ Hyper-parameters of a model from the `stable_baselines3` package.
    
    The command-line arguments for these are created with simple-parsing.
    """
    # The policy model to use (MlpPolicy, CnnPolicy, ...)
    # TODO: Might need to overwrite/grow this 'choice dict' to support other
    # policies, if other methods use a different policy class. 
    policy: Optional[Union[str, Type[BasePolicy]]] = choice("MlpPolicy", "CnnPolicy", default=None)
    
    # # The environment to learn from.  If registered in Gym, can be str. Can be
    # # None for loading trained models)
    # env: Union[GymEnv, str, None]
    
    # # The base policy used by this method
    # policy_base: Type[BasePolicy]
    
    # learning rate for the optimizer, it can be a function of the current progress remaining (from 1 to 0)
    learning_rate: Union[float, Callable] = 1e-4
    # Additional arguments to be passed to the policy on creation
    policy_kwargs: Optional[Dict[str, Any]] = None
    # the log location for tensorboard (if None, no logging)
    tensorboard_log: Optional[str] = None
    # The verbosity level: 0 none, 1 training information, 2 debug
    verbose: int = 0
    # Device on which the code should run. By default, it will try to use a Cuda compatible device and fallback to cpu if it is not possible.
    device: Union[torch.device, str] = "auto"
    
    
    # # Whether the algorithm supports training with multiple environments (as in A2C)
    # support_multi_env: bool = False
    
    # Whether to create a second environment that will be used for evaluating the agent periodically. (Only available when passing string for the environment)
    create_eval_env: bool = False
    
    # # When creating an environment, whether to wrap it or not in a Monitor wrapper.
    # monitor_wrapper: bool = True
    
    # Seed for the pseudo random generators
    seed: Optional[int] = None
    # # Whether to use generalized State Dependent Exploration (gSDE) instead of action noise exploration (default: False)
    # use_sde: bool = False
    # # Sample a new noise matrix every n steps when using gSDE Default: -1 (only sample at the beginning of the rollout)
    # sde_sample_freq: int = -1
    

@dataclass
class StableBaselines3Method(Method, ABC, target_setting=ContinualRLSetting):
    """ Base class for the methods that use models from the stable_baselines3
    repo.
    """
    # Class variable that represents what kind of Model will be used.
    # (This is just here so we can easily create one Method class per model type
    # by just changing this class attribute.)
    Model: ClassVar[Type[BaseAlgorithm]]
    
    # HyperParameters of the Method.
    hparams: SB3BaseHParams = mutable_field(SB3BaseHParams)
    
    # The number of training steps to run per task.
    # NOTE: This shouldn't be set to more than 1 when applying this method on a
    # ContinualRLSetting, because we don't currently have a way of "resetting"
    # the nonstationarity in the environment, and there is only one task,
    # therefore if we trained for say 10 million steps, while the
    # non-stationarity only lasts for 10_000 steps, we'd have seen an almost
    # stationary distribution, since the environment would have stopped changing after 10_000 steps.
    # 
    train_steps_per_task: int = 10_000
        
    # Evaluate the agent every ``eval_freq`` timesteps (this may vary a little) 
    eval_freq: int = -1
    # callback(s) called at every step with state of the algorithm.
    callback: MaybeCallback = None
    # The number of timesteps before logging.
    log_interval: int = 100
    # the name of the run for TensorBoard logging
    tb_log_name: str = "run"
    # Evaluate the agent every ``eval_freq`` timesteps (this may vary a little)
    eval_freq: int = -1
    # Number of episode to evaluate the agent
    n_eval_episodes = 5
    # Path to a folder where the evaluations will be saved
    eval_log_path: Optional[str] = None
    
    
    def __post_init__(self):
        self.model: Optional[BaseAlgorithm] = None
        # Extra wrappers to add to the train_env and valid_env before passing
        # them to the `learn` method from stable-baselines3.
        self.extra_train_wrappers: List[Callable[[gym.Env], gym.Env]] = [
            RemoveTaskLabelsWrapper,
            NoTypedObjectsWrapper,
        ]
        self.extra_valid_wrappers: List[Callable[[gym.Env], gym.Env]] = [
            RemoveTaskLabelsWrapper,
            NoTypedObjectsWrapper,
        ]
        # Number of timesteps to train on for each task.
        self.total_timesteps_per_task: int = 0

    def configure(self, setting: ContinualRLSetting):
        # Delete the model, if present.
        self.model = None
        # For now, we don't batch the space because stablebaselines3 will add an
        # additional batch dimension if we do.
        # TODO: Still need to debug the batching stuff with stablebaselines
        setting.batch_size = None
        from common.transforms import ChannelsLastIfNeeded, Transforms
        # assert False, setting.train_transforms
        # BUG: Need to fix an issue when using the CnnPolicy and Atary envs, the
        # input shape isn't what they expect (only 2 channels instead of three
        # apparently.)
        setting.transforms = []
        setting.train_transforms = []
        setting.val_transforms = []
        setting.test_transforms = []

        if self.hparams.policy is None:     
            if setting.observe_state_directly:
                self.hparams.policy = "MlpPolicy"
            else:
                self.hparams.policy = "CnnPolicy"

        logger.debug(f"Will use {self.hparams.policy} as the policy.")

        # TODO: Need to figure out how many steps these methods need to be
        # trained for, as well as a way to "check" that training works.

        self.total_timesteps_per_task = setting.steps_per_task
        
        if not setting.known_task_boundaries_at_train_time:
            # We are in a ContinualRL setting, where `fit` will only be called
            # once and where the environment can only be traversed once.
            if self.train_steps_per_task > setting.max_steps:
                warnings.warn(RuntimeWarning(
                    f"Can't train for the requested {self.train_steps_per_task} "
                    f"steps, since we're (currently) only allowed one 'pass' "
                    f"through the environment when in a Continual-RL Setting."
                ))
            self.train_steps_per_task = setting.max_steps
        # Otherwise, we can train basically as long as we want on each task.

    def create_model(self, train_env: gym.Env, valid_env: gym.Env) -> BaseAlgorithm:
        return self.Model(env=train_env, **self.hparams.to_dict())
    
    def fit(self, train_env: gym.Env = None, valid_env: gym.Env = None):
        # Remove the extra information that the Setting gives us.
        for wrapper in self.extra_train_wrappers:
            train_env = wrapper(train_env)
        
        for wrapper in self.extra_valid_wrappers:
            valid_env = wrapper(valid_env)

        if self.model is None:
            self.model = self.create_model(train_env, valid_env)
        else:
            # TODO: "Adapt"/re-train the model on the new environment.
            self.model.set_env(train_env)

        # Decide how many steps to train on.
        total_timesteps = self.train_steps_per_task

        # TODO: Actually setup/customize the parametrers of the model and of this
        # "learn" method, and also make sure that this "works" and training converges.
        self.model = self.model.learn(
            # The total number of samples (env steps) to train on
            total_timesteps = total_timesteps,
            eval_env = valid_env,
            callback = self.callback,
            log_interval = self.log_interval,
            tb_log_name = self.tb_log_name,
            eval_freq = self.eval_freq,
            n_eval_episodes = self.n_eval_episodes,
            eval_log_path = self.eval_log_path,
            # whether or not to reset the current timestep number (used in logging)
            reset_num_timesteps = True,
        )

    def get_actions(self, observations: ContinualRLSetting.Observations, action_space: spaces.Space) -> ContinualRLSetting.Actions:
        obs = observations[0]
        predictions = self.model.predict(obs)
        action, _ = predictions
        if action not in action_space:
            assert len(action) == 1
            action = action.item()
        return action



