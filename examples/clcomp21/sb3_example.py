""" Example where we start from a Method from stable-baselines3 to solve the rl track.
"""
from dataclasses import dataclass
from typing import ClassVar, Dict, Mapping, Type, Union, Optional

import gym
from gym import spaces
from sequoia.methods.stable_baselines3_methods.ppo import PPOMethod, PPOModel
from sequoia.settings.rl import ContinualRLSetting
from simple_parsing import mutable_field

# from stable_baselines3.ppo.policies import ActorCriticCnnPolicy, ActorCriticPolicy


class CustomPPOModel(PPOModel):
    @dataclass
    class HParams(PPOModel.HParams):
        """ Hyper-parameters of the PPO Model. """


@dataclass
class CustomPPOMethod(PPOMethod):
    Model: ClassVar[Type[PPOModel]] = PPOModel
    # Hyper-parameters of the PPO Model.
    hparams: PPOModel.HParams = mutable_field(PPOModel.HParams)

    def configure(self, setting: ContinualRLSetting):
        super().configure(setting=setting)

    def create_model(self, train_env: gym.Env, valid_env: gym.Env) -> PPOModel:
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
        return super().get_search_space(setting)


if __name__ == "__main__":

    # Create the Setting.

    # CartPole-state for debugging:
    from sequoia.settings.rl import RLSetting

    setting = RLSetting(dataset="CartPole-v0")

    # OR: Incremental CartPole-state:
    from sequoia.settings.rl import IncrementalRLSetting

    setting = IncrementalRLSetting(
        dataset="CartPole-v0",
        monitor_training_performance=True,
        nb_tasks=1,
        train_steps_per_task=1_000,
        test_max_steps=2000,
    )

    # OR: Setting of the RL Track of the competition:
    # setting = IncrementalRLSetting.load_benchmark("rl_track")

    # Create the Method:
    method = CustomPPOMethod()

    # Apply the Method onto the Setting to get Results.
    results = setting.apply(method)
    print(results.summary())

    # BONUS: Running a hyper-parameter sweep:
    # method.hparam_sweep(setting)
