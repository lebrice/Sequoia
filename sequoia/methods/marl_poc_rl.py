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
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3 import PPO

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


# This creates a vectorized environment that has multiple pistonball environments concatenated together
# For testing, we set the num_vector_envs to 1 to create a single pistonball environment (vectorized across agents)
def make_pistonball_env(
        vector_env_type: Literal["gym", "stable_baselines3", "stable_baselines"] = "stable_baselines3",
        num_vector_envs: int = 8
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
    env = ss.concat_vec_envs_v0(vector_env, num_vector_envs, num_cpus=4, base_class=vector_env_type)
    env.spec = EnvSpec("Sequoia_PistonBall-v4", entry_point=make_pistonball_env)
    if vector_env_type == "stable_baselines3":
        # Patch missing attributes
        env.reward_range = (-np.inf, np.inf)
        env.single_action_space = env.venv.action_space
    return env

gym.register("Sequoia_PistonBall-v4", entry_point=make_pistonball_env)


class DebugMARLMethod(Method, target_setting=MARLSetting):
    def __init__(self, input_model=None):
        super().__init__()
        self.input_model = input_model

    def configure(self, setting: MARLSetting) -> None:
        pass

    def fit(self, train_env: gym.Env, valid_env: gym.Env):

        wrappers = [
            # partial(TransformObservation, f=operator.itemgetter("x")),
            # # partial(TransformAction, f=operator.itemgetter("y_pred"),
            partial(TransformReward, f=operator.itemgetter("y")),
            # RemoveInfoWrapper,
        ]
        for wrapper in wrappers:
            train_env = wrapper(train_env)
        for wrapper in wrappers:
            valid_env = wrapper(valid_env)

        # Allows input model; otherwise, initializes the model here
        if self.input_model is None:
            self.model = PPO("CnnPolicy", env=train_env, create_eval_env=False)
        else:
            self.model = self.input_model
        # self.model = SAC("CnnPolicy", env=train_env, create_eval_env=False)

        # self.model.learn(10000, eval_env=valid_env, n_eval_episodes=10)
        # self.model.learn(10000, n_eval_episodes=10)

        # self.model.learn(total_timesteps=200000)
        # Smaller timestep amount for debugging
        self.model.learn(total_timesteps=1000)


    def get_actions(
            self, observations: MARLSetting.Observations, action_space: gym.Space
    ) -> MARLSetting.Actions:
        return action_space.sample()


from sequoia.settings.rl import TaskIncrementalRLSetting


def main():
    pistonball_env = make_pistonball_env()  # gym.make("Sequoia_PistonBall-v4")
    pistonball_val_env = make_pistonball_env()  # gym.make("Sequoia_PistonBall-v4")
    pistonball_test_env = make_pistonball_env(num_vector_envs=1)  # gym.make("Sequoia_PistonBall-v4")

    model = PPO(
        CnnPolicy,
        pistonball_env,
        verbose=3,
        gamma=0.95,
        n_steps=256,
        ent_coef=0.0905168,
        learning_rate=0.00062211,
        vf_coef=0.042202,
        max_grad_norm=0.9,
        gae_lambda=0.99,
        n_epochs=5,
        clip_range=0.3,
        batch_size=256,
        # policy_kwargs={"monitor_wrapper": False}
    )

    # For sake of changing the environments, just initialize a dummy model for validation and test env
    model_val = PPO(
        CnnPolicy,
        pistonball_val_env,
        verbose=3,
        gamma=0.95,
        n_steps=256,
        ent_coef=0.0905168,
        learning_rate=0.00062211,
        vf_coef=0.042202,
        max_grad_norm=0.9,
        gae_lambda=0.99,
        n_epochs=5,
        clip_range=0.3,
        batch_size=256,
        # policy_kwargs={"monitor_wrapper": False}
    )

    model_test = PPO(
        CnnPolicy,
        pistonball_test_env,
        verbose=3,
        gamma=0.95,
        n_steps=256,
        ent_coef=0.0905168,
        learning_rate=0.00062211,
        vf_coef=0.042202,
        max_grad_norm=0.9,
        gae_lambda=0.99,
        n_epochs=5,
        clip_range=0.3,
        batch_size=256,
        # policy_kwargs={"monitor_wrapper": False}
    )

    pistonball_env = model.env
    pistonball_val_env = model_val.env
    pistonball_test_env = model_test.env

    # Hack: Set reward range because gym requries it
    # pistonball_env.reward_range = pistonball_env.venv.reward_range
    # pistonball_val_env.reward_range = pistonball_val_env.venv.reward_range
    # pistonball_test_env.reward_range = pistonball_test_env.venv.reward_range
    pistonball_env.reward_range = pistonball_env.venv.venv.reward_range
    pistonball_val_env.reward_range = pistonball_val_env.venv.venv.reward_range
    pistonball_test_env.reward_range = pistonball_test_env.venv.venv.reward_range
    # Hack: Setting single_action_space manually
    pistonball_env.single_action_space = pistonball_env.action_space
    pistonball_val_env.single_action_space = pistonball_val_env.action_space
    pistonball_test_env.single_action_space = pistonball_test_env.action_space

    # Apply transformation wrappers to envs
    wrappers = [
        # partial(TransformObservation, f=operator.itemgetter("x")),
        # # partial(TransformAction, f=operator.itemgetter("y_pred"),
        # partial(TransformReward, f=operator.itemgetter("y")),
        # RemoveInfoWrapper,
    ]
    for wrapper in wrappers:
        pistonball_env = wrapper(pistonball_env)
        pistonball_val_env = wrapper(pistonball_val_env)
        pistonball_test_env = wrapper(pistonball_test_env)

    # Not using TraditionalRLSetting b/c there is existing bug where it tries to create two tasks when there should be just one
    setting = TaskIncrementalRLSetting(train_envs=[pistonball_env], val_envs=[pistonball_val_env],
                                       test_envs=[pistonball_test_env])
    model.env = setting
    # model.eval_env = setting

    method = DebugMARLMethod(input_model=model)
    results = setting.apply(method)
    print(results)


if __name__ == "__main__":
    main()
