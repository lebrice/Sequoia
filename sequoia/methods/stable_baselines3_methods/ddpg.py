""" Method that uses the DDPG model from stable-baselines3 and targets the RL
settings in the tree.
"""
from dataclasses import dataclass
from typing import ClassVar, Type, Optional

import gym
from gym import spaces
from simple_parsing import mutable_field
from stable_baselines3.ddpg import DDPG

from sequoia.methods import register_method
from sequoia.settings.active import ContinualRLSetting

from .base import SB3BaseHParams, StableBaselines3Method


class DDPGModel(DDPG):
    """ Customized version of the DDPG model from stable-baselines-3. """
    @dataclass
    class HParams(SB3BaseHParams):
        """ Hyper-parameters of the DDPG Model. """
        # TODO: Create the fields from the DDPG constructor arguments.


@register_method
@dataclass
class DDPGMethod(StableBaselines3Method):
    """ Method that uses the DDPG model from stable-baselines3. """

    Model: ClassVar[Type[DDPGModel]] = DDPGModel

    # Hyper-parameters of the DDPG model.
    hparams: DDPGModel.HParams = mutable_field(DDPGModel.HParams)

    def configure(self, setting: ContinualRLSetting):
        super().configure(setting=setting)

    def create_model(self, train_env: gym.Env, valid_env: gym.Env) -> DDPGModel:
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


if __name__ == "__main__":
    results = DDPGMethod.main()
    print(results)
