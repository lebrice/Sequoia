""" Method that uses the PPO model from stable-baselines3 and targets the RL
settings in the tree.
"""
from dataclasses import dataclass
from typing import ClassVar, Dict, Mapping, Optional, Type, Union

import gym
import torch
from gym import spaces
from simple_parsing import mutable_field
from stable_baselines3.ppo import PPO

from sequoia.common.hparams import categorical, log_uniform, uniform
from sequoia.methods import register_method
from sequoia.methods.stable_baselines3_methods.base import (
    SB3BaseHParams, StableBaselines3Method)
from sequoia.settings.active import ContinualRLSetting

class PPOModel(PPO):
    """ Proximal Policy Optimization algorithm (PPO) (clip version) - from SB3.

    Paper: https://arxiv.org/abs/1707.06347
    Code: The SB3 implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    and Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html
    """

    @dataclass
    class HParams(SB3BaseHParams):
        """ Hyper-parameters of the PPO Model. """

        # # The policy model to use (MlpPolicy, CnnPolicy, ...)
        # policy: Union[str, Type[ActorCriticPolicy]]

        # # The environment to learn from (if registered in Gym, can be str)
        # env: Union[GymEnv, str]

        # The learning rate, it can be a function of the current progress remaining
        # (from 1 to 0)
        learning_rate: float = log_uniform(1e-6, 1e-2, default=3e-4)

        # The number of steps to run for each environment per update (i.e. batch size
        # is n_steps * n_env where n_env is number of environment copies running in
        # parallel)
        # TODO: Limit this, as is done in A2C, based on the value of setting.max steps.
        n_steps: int = categorical(32, 128, 256, 1024, 2048, 4096, 8192, default=2048)

        # Minibatch size
        # batch_size: Optional[int] = categorical(16, 32, 64, 128, default=64)
        batch_size: int = 64

        # Number of epoch when optimizing the surrogate loss
        n_epochs: int = 10

        # Discount factor
        # gamma: float = uniform(0.9, 0.9999, default=0.99)
        gamma: float = 0.99

        # Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        # gae_lambda: float = uniform(0.8, 1.0, default=0.95)
        gae_lambda: float = 0.95

        # Clipping parameter, it can be a function of the current progress remaining
        # (from 1 to 0).
        # clip_range: float = uniform(0.05, 0.4, default=0.2)
        clip_range: float = 0.2

        # Clipping parameter for the value function, it can be a function of the current
        # progress remaining (from 1 to 0). This is a parameter specific to the OpenAI
        # implementation. If None is passed (default), no clipping will be done on the
        # value function. IMPORTANT: this clipping depends on the reward scaling.
        clip_range_vf: Optional[float] = None

        # Entropy coefficient for the loss calculation
        # ent_coef: float = uniform(0., 1., default=0.0)
        ent_coef: float = 0.0

        # Value function coefficient for the loss calculation
        # vf_coef: float = uniform(0.01, 1.0, default=0.5)
        vf_coef: float = 0.5

        # The maximum value for the gradient clipping
        # max_grad_norm: float = uniform(0.1, 10, default=0.5)
        max_grad_norm: float = 0.5

        # Whether to use generalized State Dependent Exploration (gSDE) instead of
        # action noise exploration (default: False)
        # use_sde: bool = categorical(True, False, default=False)
        use_sde: bool = False

        # Sample a new noise matrix every n steps when using gSDE Default: -1 (only
        # sample at the beginning of the rollout)
        # sde_sample_freq: int = categorical(-1, 1, 5, 10, default=-1)
        sde_sample_freq: int = -1

        # Limit the KL divergence between updates, because the clipping is not enough to
        # prevent large update see issue #213
        # (cf https://github.com/hill-a/stable-baselines/issues/213)
        # By default, there is no limit on the kl div.
        target_kl: Optional[float] = None

        # the log location for tensorboard (if None, no logging)
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

        # Device (cpu, cuda, ...) on which the code should be run. Setting it to auto,
        # the code will be run on the GPU if possible.
        device: Union[torch.device, str] = "auto"
        
        # Whether or not to build the network at the creation of the instance
        # _init_setup_model: bool = True



@register_method
@dataclass
class PPOMethod(StableBaselines3Method):
    """ Method that uses the PPO model from stable-baselines3. """
    Model: ClassVar[Type[PPOModel]] = PPOModel
    # Hyper-parameters of the PPO Model.
    hparams: PPOModel.HParams = mutable_field(PPOModel.HParams)

    def configure(self, setting: ContinualRLSetting):
        super().configure(setting=setting)

    def create_model(self, train_env: gym.Env, valid_env: gym.Env) -> PPOModel:
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


if __name__ == "__main__":
    results = PPOMethod.main()
    print(results)
