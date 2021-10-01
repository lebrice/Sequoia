import gym
import numpy as np
import supersuit as ss
from gym.envs.registration import EnvSpec
from gym.vector.vector_env import VectorEnv
from pettingzoo.butterfly import pistonball_v4
from sequoia.settings.base import Method
from sequoia.settings.base.setting import Setting as MARLSetting
from sequoia.settings.rl import IncrementalRLSetting
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


class MARLMethod(Method, target_setting=MARLSetting):
    def __init__(self, input_model=None):
        super().__init__()
        self.input_model = input_model

    def configure(self, setting: MARLSetting) -> None:
        pass

    def fit(self, train_env: gym.Env, valid_env: gym.Env):
        self.model = self.input_model

        self.model.learn(total_timesteps=200_000)
        # Smaller timestep amount for debugging
        # self.model.learn(total_timesteps=1000)


    def get_actions(
            self, observations: MARLSetting.Observations, action_space: gym.Space
    ) -> MARLSetting.Actions:
        obs = observations.x
        predictions = self.model.predict(obs)
        action, _ = predictions
        assert action in action_space, (observations, action, action_space)

        return action


def main():
    pistonball_env = make_pistonball_env()
    pistonball_val_env = make_pistonball_env()
    pistonball_test_env = make_pistonball_env(num_vector_envs=1)

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
    )

    pistonball_env = model.env

    # Set reward range because gym requries it
    pistonball_env.reward_range = pistonball_env.venv.venv.reward_range

    # Set single_action_space manually
    pistonball_env.single_action_space = pistonball_env.action_space

    setting = IncrementalRLSetting(
        train_envs=[pistonball_env],
        val_envs=[pistonball_val_env],
        test_envs=[pistonball_test_env],
        train_max_steps=250_000
    )
    model.env = setting

    method = MARLMethod(input_model=model)
    results = setting.apply(method)
    print(results)


if __name__ == "__main__":
    main()
