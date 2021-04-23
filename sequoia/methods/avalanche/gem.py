""" GEM Method from Avalanche. """
from dataclasses import dataclass
from typing import ClassVar, Optional, Type

import gym
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from avalanche.training.strategies import GEM, BaseStrategy

from sequoia.methods import register_method
from sequoia.settings.passive import (
    ClassIncrementalSetting,
    PassiveEnvironment,
    TaskIncrementalSetting,
)
from .base import AvalancheMethod


@register_method
@dataclass
class GEMMethod(AvalancheMethod, target_setting=ClassIncrementalSetting):
    """Gradient Episodic Memory (GEM) strategy from Avalanche.
    See GEM plugin for details.
    This strategy does not use task identities.
    """

    # number of patterns per experience in the memory
    patterns_per_exp: int = 100
    # Offset to add to the projection direction in order to favour backward transfer
    # (gamma in original paper).
    memory_strength: float = 0.5

    strategy_class: ClassVar[Type[BaseStrategy]] = GEM

    def configure(self, setting: ClassIncrementalSetting) -> None:
        super().configure(setting)

    def create_cl_strategy(self, setting: ClassIncrementalSetting) -> GEM:
        return super().create_cl_strategy(setting)

    def create_model(self, setting: ClassIncrementalSetting) -> Module:
        return super().create_model(setting)

    def make_optimizer(self, **kwargs) -> Optimizer:
        """ Creates the Optimizer object from the options. """
        return super().make_optimizer(**kwargs)

    def fit(self, train_env: PassiveEnvironment, valid_env: PassiveEnvironment):
        return super().fit(train_env=train_env, valid_env=valid_env)

    def get_actions(
        self, observations: TaskIncrementalSetting.Observations, action_space: gym.Space
    ) -> TaskIncrementalSetting.Actions:
        return super().get_actions(observations=observations, action_space=action_space)

    def on_task_switch(self, task_id: Optional[int]) -> None:
        # TODO: Figure out if it makes sense to use this at test time (no real need for)
        # this at train time, except maybe in multi-task setting? Even then, not totally
        # sure.
        pass


if __name__ == "__main__":
    from simple_parsing import ArgumentParser

    setting = TaskIncrementalSetting(
        dataset="mnist", nb_tasks=5, monitor_training_performance=True
    )
    # Create the Method, either manually or through the command-line:
    parser = ArgumentParser(__doc__)
    parser.add_arguments(GEMMethod, "method")
    args = parser.parse_args()
    method: GEMMethod = args.method

    results = setting.apply(method)
