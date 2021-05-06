""" SynapticIntelligence Method from Avalanche. """
from dataclasses import dataclass
from typing import ClassVar, Optional, Type

import gym
from avalanche.training.strategies import SynapticIntelligence, BaseStrategy
from sequoia.methods import register_method
from sequoia.settings.passive import (ClassIncrementalSetting,
                                      PassiveEnvironment,
                                      TaskIncrementalSetting)
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from .base import AvalancheMethod


@register_method
@dataclass
class SynapticIntelligenceMethod(AvalancheMethod, target_setting=ClassIncrementalSetting):
    """ The Synaptic Intelligence strategy from Avalanche.

    This is the Synaptic Intelligence PyTorch implementation of the
    algorithm described in the paper
    "Continuous Learning in Single-Incremental-Task Scenarios"
    (https://arxiv.org/abs/1806.08568)

    The original implementation has been proposed in the paper
    "Continual Learning Through Synaptic Intelligence"
    (https://arxiv.org/abs/1703.04200).

    The Synaptic Intelligence regularization can also be used in a different
    strategy by applying the :class:`SynapticIntelligencePlugin` plugin.
    """
    # Synaptic Intelligence lambda term. (TODO: Find the right default value for this)
    si_lambda: float = 0.5

    strategy_class: ClassVar[Type[BaseStrategy]] = SynapticIntelligence

    def configure(self, setting: ClassIncrementalSetting) -> None:
        super().configure(setting)

    def create_cl_strategy(self, setting: ClassIncrementalSetting) -> SynapticIntelligence:
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
    parser.add_arguments(SynapticIntelligenceMethod, "method")
    args = parser.parse_args()
    method: SynapticIntelligenceMethod = args.method

    results = setting.apply(method)
