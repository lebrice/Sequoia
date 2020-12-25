""" Method that uses the A2C model from stable-baselines3 and targets the RL
settings in the tree.
"""
from dataclasses import dataclass
from typing import ClassVar, Optional, Type

import gym
from gym import spaces
from simple_parsing import mutable_field
from stable_baselines3.a2c import A2C

from sequoia.methods import register_method
from sequoia.methods.stable_baselines3_methods.base import (
    SB3BaseHParams, StableBaselines3Method)
from sequoia.settings.active import ContinualRLSetting


class A2CModel(A2C):
    """ Customized version of the A2C Model from stable-baselines3. """
    @dataclass
    class HParams(SB3BaseHParams):
        """ Hyper-parameters of the A2C Model. """
        # TODO: Create the fields from the A2C constructor arguments.


@register_method
@dataclass
class A2CMethod(StableBaselines3Method):
    """ Method that uses the DDPG model from stable-baselines3. """
    # changing the 'name' in this case here, because the default name would be
    # 'a_2_c'.
    name: ClassVar[str] = "a2c"
    Model: ClassVar[Type[A2CModel]] = A2CModel

    # Hyper-parameters of the A2C model.
    hparams: A2CModel.HParams = mutable_field(A2CModel.HParams)

    def configure(self, setting: ContinualRLSetting):
        super().configure(setting=setting)

    def create_model(self, train_env: gym.Env, valid_env: gym.Env) -> A2CModel:
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
    results = A2CMethod.main()
    print(results)
