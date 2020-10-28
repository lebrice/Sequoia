""" Example of creating an A2C agent using the simplebaselines3 package.

See https://stable-baselines3.readthedocs.io/en/master/guide/install.html
"""
from abc import ABC
from typing import Optional, ClassVar, Type
import gym
from gym import spaces
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.base_class import BaseAlgorithm

from common.gym_wrappers.batch_env.batched_vector_env import VectorEnv
from settings import ContinualRLSetting, Method
from settings.active.rl.wrappers import RemoveTaskLabelsWrapper, NoTypedObjectsWrapper
from settings.active.rl.continual_rl_setting import ContinualRLSetting

from stable_baselines3.common.base_class import is_image_space, VecEnv, DummyVecEnv, VecTransposeImage 

class WrapEnvPatch:
    # Patch for the _wrap_env function of the BaseAlgorithm class of
    # stable_baselines, to make it recognize the VectorEnv from gym.vector as a
    # vectorized environment.
    def _wrap_env(self: BaseAlgorithm, env: gym.Env):
        # NOTE: We just want to change this single line here:
        # if not isinstance(env, VecEnv):
        if not (isinstance(env, (VecEnv, VectorEnv)) or isinstance(env.unwrapped, (VecEnv, VectorEnv))):
            if self.verbose >= 1:
                print("Wrapping the env in a DummyVecEnv.")
            env = DummyVecEnv([lambda: env])

        if is_image_space(env.observation_space) and not isinstance(env, VecTransposeImage):
            if self.verbose >= 1:
                print("Wrapping the env in a VecTransposeImage.")
            env = VecTransposeImage(env)
        return env


class A2CModel(WrapEnvPatch, A2C):
    pass

class PPOModel(WrapEnvPatch, PPO):
    pass


class StableBaselines3Method(Method, target_setting=ContinualRLSetting):
    Model: ClassVar[Type[BaseAlgorithm]] = A2CModel
    
    def __init__(self):
        self.model: Optional[BaseAlgorithm]

    def configure(self, setting: ContinualRLSetting):
        self.model = None
        # For now, we don't batch the space because stablebaselines3 will add an
        # additional batch dimension if we do.
        # TODO: Still need to debug the batching stuff with stablebaselines
        setting.train_batch_size = None
        setting.valid_batch_size = None
        setting.test_batch_size = None
        # Only one "epoch" of training for now. 
        self.total_timesteps = setting.max_steps

    def fit(self, train_env: gym.Env = None, valid_env: gym.Env = None):
        train_env = RemoveTaskLabelsWrapper(train_env)
        train_env = NoTypedObjectsWrapper(train_env)
        
        valid_env = RemoveTaskLabelsWrapper(valid_env)
        valid_env = NoTypedObjectsWrapper(valid_env)
        
        if self.model is None:
            self.model = self.Model('MlpPolicy', train_env, verbose=1)

        # TODO: Actually setup/customize the parametrers of the model.
        self.model.learn(total_timesteps=self.total_timesteps, eval_env=valid_env)

    def get_actions(self, observations: ContinualRLSetting.Observations, action_space: spaces.Space) -> ContinualRLSetting.Actions:
        obs = observations[0]
        predictions = self.model.predict(obs)
        action, _ = predictions
        return action


class A2CMethod(StableBaselines3Method):
    name: ClassVar[str] = "a2c"
    Model: ClassVar[Type[BaseAlgorithm]] = A2CModel


class PPOMethod(StableBaselines3Method):
    Model: ClassVar[Type[BaseAlgorithm]] = PPOModel


if __name__ == "__main__":
    setting = ContinualRLSetting.from_args()
    method = A2CMethod()
    
    # # TODO: Check out the wandb output.
    # import wandb
    # wandb.gym.monitor()
    results = setting.apply(method)
    print(results.summary())
    print(f"objective: {results.objective}")
    
        