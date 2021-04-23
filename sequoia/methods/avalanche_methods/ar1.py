""" AR1 Method from Avalanche. """
from dataclasses import dataclass
from typing import ClassVar, Optional, Type

import gym
from avalanche.training.strategies import AR1, BaseStrategy
from sequoia.methods import register_method
from sequoia.settings.passive import (ClassIncrementalSetting,
                                      PassiveEnvironment,
                                      TaskIncrementalSetting)
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from .base import AvalancheMethod


@register_method
@dataclass
class AR1Method(AvalancheMethod, target_setting=ClassIncrementalSetting):
    """ AR1 strategy from Avalanche.
    See AR1 plugin for details.
    This strategy does not use task identities.
    """
    # The learning rate (SGD optimizer).
    lr: float = 0.001
    # The momentum (SGD optimizer).
    momentum: float = 0.9
    # The L2 penalty used for weight decay.
    l2: float = 0.0005
    # The number of training epochs. Defaults to 4.
    train_epochs: int = 4
    # The initial update rate of BatchReNorm layers.
    init_update_rate: float = 0.01
    # The incremental update rate of BatchReNorm layers.
    inc_update_rate: float = 0.00005
    # The maximum r value of BatchReNorm layers.
    max_r_max: float = 1.25
    # The maximum d value of BatchReNorm layers.
    max_d_max: float = 0.5
    # The incremental step of r and d values of BatchReNorm layers.
    inc_step: float = 4.1e-05
    # The size of the replay buffer. The replay buffer is shared across classes.
    rm_sz: int = 1500
    # A string describing the name of the layer to use while freezing the lower
    # (nearest to the input) part of the model. The given layer is not frozen
    # (exclusive).
    freeze_below_layer: str = "lat_features.19.bn.beta"
    # The number of the layer to use as the Latent Replay Layer. Usually this is the
    # same of `freeze_below_layer`.
    latent_layer_num: int = 19
    # The Synaptic Intelligence lambda term. Defaults to 0, which means that the
    # Synaptic Intelligence regularization will not be applied.
    ewc_lambda: float = 0
    # The train minibatch size. Defaults to 1.
    train_mb_size: int = 128
    # The eval minibatch size. Defaults to 1.
    eval_mb_size: int = 128

    strategy_class: ClassVar[Type[BaseStrategy]] = AR1

    def configure(self, setting: ClassIncrementalSetting) -> None:
        super().configure(setting)

    def create_cl_strategy(self, setting: ClassIncrementalSetting) -> AR1:
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
    parser.add_arguments(AR1Method, "method")
    args = parser.parse_args()
    method: AR1Method = args.method

    results = setting.apply(method)
