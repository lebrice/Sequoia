import operator
from functools import partial

import gym
import numpy as np
import supersuit as ss
import supersuit.vector.sb3_vector_wrapper
from gym.envs.registration import EnvSpec
from gym.vector.vector_env import VectorEnv
from pettingzoo.butterfly import pistonball_v4
from sequoia.common.gym_wrappers.transform_wrappers import TransformObservation, TransformReward
from sequoia.methods.stable_baselines3_methods.base import RemoveInfoWrapper
from sequoia.settings.base import Method
from sequoia.settings.base.setting import Setting as MARLSetting
from stable_baselines3.sac.sac import SAC

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def make_pistonball_env(
    vector_env_type: Literal["gym", "stable_baselines3", "stable_baselines"] = "stable_baselines3"
) -> VectorEnv:
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
    if vector_env_type == "stable_baselines3":
        # Patch missing attributes
        env.reward_range = (-np.inf, np.inf)
        env.single_action_space = env.venv.action_space
    return env


gym.register("Sequoia_PistonBall-v4", entry_point=make_pistonball_env)


class DebugMARLMethod(Method, target_setting=MARLSetting):
    def __init__(self):
        super().__init__()

    def configure(self, setting: MARLSetting) -> None:
        pass

    def fit(self, train_env: gym.Env, valid_env: gym.Env):

        wrappers = [
            partial(TransformObservation, f=operator.itemgetter("x")),
            # partial(TransformAction, f=operator.itemgetter("y_pred"),
            partial(TransformReward, f=operator.itemgetter("y")),
            RemoveInfoWrapper,
        ]
        for wrapper in wrappers:
            train_env = wrapper(train_env)
        for wrapper in wrappers:
            valid_env = wrapper(valid_env)

        # self.model = PPO("CnnPolicy", env=train_env, create_eval_env=False)
        self.model = SAC("CnnPolicy", env=train_env, create_eval_env=False)
        self.model.learn(10000, eval_env=valid_env, n_eval_episodes=10)

    def get_actions(
        self, observations: MARLSetting.Observations, action_space: gym.Space
    ) -> MARLSetting.Actions:
        return action_space.sample()


from sequoia.settings.rl import TaskIncrementalRLSetting


def main():
    env = make_pistonball_env()  # gym.make("Sequoia_PistonBall-v4")
    val_env = make_pistonball_env()  # gym.make("Sequoia_PistonBall-v4")
    test_env = make_pistonball_env()  # gym.make("Sequoia_PistonBall-v4")

    setting = TaskIncrementalRLSetting(train_envs=[env], val_envs=[val_env], test_envs=[test_env])

    method = DebugMARLMethod()
    results = setting.apply(method)
    print(results)


if __name__ == "__main__":
    main()
