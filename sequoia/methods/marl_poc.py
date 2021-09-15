import gym
from sequoia.settings.base.objects import Observations
from sequoia.settings.base import Method
from sequoia.settings.base.setting import Setting as MARLSetting
from stable_baselines3.ppo.ppo import PPO
import gym
from gym.envs.registration import EnvSpec
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy

from pettingzoo.butterfly import pistonball_v4
from gym.vector.vector_env import VectorEnv
# from gym.envs.registration import register
import gym
from sequoia.settings.rl.continual.tasks import make_continuous_task
from typing import Dict
import random
import supersuit.vector.sb3_vector_wrapper

def make_pistonball_env(vector_env_type: str = "stable_baselines3") -> VectorEnv:
    env = pistonball_v4.parallel_env(
        n_pistons=20,
        local_ratio=0,
        time_penalty=-0.1,
        continuous=True,
        random_drop=True,
        random_rotate=True,
        ball_mass=0.75,
        ball_friction=0.3,
        ball_elasticity=1.5,
        max_cycles=125,
    )
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v0(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 3)
    vector_env: VectorEnv = ss.pettingzoo_env_to_vec_env_v0(env)
    env = ss.concat_vec_envs_v0(vector_env, 8, num_cpus=4, base_class=vector_env_type)
    env.spec = EnvSpec("Sequoia_PistonBall-v4", entry_point=make_pistonball_env)
    return env


gym.register("Sequoia_PistonBall-v4", entry_point=make_pistonball_env)


class DebugPPOMethod(Method, target_setting=MARLSetting):
    def __init__(self):
        super().__init__()
    
    def configure(self, setting: MARLSetting) -> None:
        pass

    def fit(self, train_env: gym.Env, valid_env: gym.Env):
        self.model = PPO("cnn", env=train_env, create_eval_env=False)
        self.model.learn(10000, eval_env=valid_env, n_eval_episodes=10)
    
    def get_actions(self, observations: MARLSetting.Observations, action_space: gym.Space) -> MARLSetting.Actions:
        return action_space.sample()


from sequoia.settings.rl import TaskIncrementalRLSetting
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage


def main():
    env = gym.make("Sequoia_pistonball-v0")
    val_env = gym.make("Sequoia_pistonball-v0")
    test_env = gym.make("Sequoia_pistonball-v0")
    
    setting = TaskIncrementalRLSetting(train_envs=[env], val_envs=[val_env], test_envs=[test_env])
    
    method = DebugPPOMethod()
    results = setting.apply(method)
    print(results)