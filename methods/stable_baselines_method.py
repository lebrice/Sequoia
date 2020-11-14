""" Example of creating an A2C agent using the simplebaselines3 package.

See https://stable-baselines3.readthedocs.io/en/master/guide/install.html
"""
from abc import ABC
from typing import Optional, ClassVar, Type
import gym
from gym import spaces


from common.gym_wrappers.batch_env.batched_vector_env import VectorEnv
from settings import ContinualRLSetting, Method
from settings.active.rl.wrappers import RemoveTaskLabelsWrapper, NoTypedObjectsWrapper
from settings.active.rl.continual_rl_setting import ContinualRLSetting
from utils.logging_utils import get_logger

logger = get_logger(__file__)

try:
    from stable_baselines3 import A2C, PPO, DDPG, DQN, SAC, TD3
    from stable_baselines3.common.base_class import BaseAlgorithm
    from stable_baselines3.common.base_class import is_image_space, VecEnv, DummyVecEnv, VecTransposeImage 
except ImportError as e:
    raise ImportError(f"The stable_baselines3 package needs to be in order to "
                      f"these Methods: {e} \n (you can install it with "
                      f"`pip install stable-baselines3[extra]`).")

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


class DQNModel(WrapEnvPatch, DQN):
    pass


class DDPGModel(WrapEnvPatch, DDPG):
    pass


class TD3Model(WrapEnvPatch, TD3):
    pass


class SACModel(WrapEnvPatch, SAC):
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
        self.total_timesteps = setting.steps_per_task

    def fit(self, train_env: gym.Env = None, valid_env: gym.Env = None):
        train_env = RemoveTaskLabelsWrapper(train_env)
        train_env = NoTypedObjectsWrapper(train_env)
        
        valid_env = RemoveTaskLabelsWrapper(valid_env)
        valid_env = NoTypedObjectsWrapper(valid_env)
        # TODO: Need to find a way to train the model on a new environment, besides re-creatign the model, obviously.
        if self.model is None:
            self.model = self.Model('MlpPolicy', train_env, verbose=1)
        else:
            self.model.set_env(train_env)

        # TODO: Actually setup/customize the parametrers of the model.
        self.model.learn(total_timesteps=self.total_timesteps, eval_env=valid_env)

    def get_actions(self, observations: ContinualRLSetting.Observations, action_space: spaces.Space) -> ContinualRLSetting.Actions:
        obs = observations[0]
        predictions = self.model.predict(obs)
        action, _ = predictions
        return action


class A2CMethod(StableBaselines3Method):
    # changing the 'name' in this case here, because the default name would be
    # 'a_2_c'.
    name: ClassVar[str] = "a2c" 
    Model: ClassVar[Type[BaseAlgorithm]] = A2CModel


class PPOMethod(StableBaselines3Method):
    Model: ClassVar[Type[BaseAlgorithm]] = PPOModel


class DQNMethod(StableBaselines3Method):
    Model: ClassVar[Type[BaseAlgorithm]] = DQNModel


class DDPGMethod(StableBaselines3Method):
    Model: ClassVar[Type[BaseAlgorithm]] = DDPGModel


class SACMethod(StableBaselines3Method):
    Model: ClassVar[Type[BaseAlgorithm]] = SACModel


class TD3Method(StableBaselines3Method):
    Model: ClassVar[Type[BaseAlgorithm]] = TD3Model


from settings import all_settings


if __name__ == "__main__":
    method = DQNMethod()
    # Evaluate on a single setting:
    setting = ContinualRLSetting.from_args()
    results = setting.apply(method)
    print(results.summary())
    print(f"objective: {results.objective}")
    exit()
    # Evaluate on all settings for the given datasets:
    
    from examples.quick_demo import evaluate_on_all_settings
    all_results = evaluate_on_all_settings(method, datasets=["CartPole-v0"])
    print(f"All results: {all_results}")
    # # TODO: Check out the wandb output.
    # import wandb
    # wandb.gym.monitor()

        
        