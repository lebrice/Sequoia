""" LwF Method from Avalanche. """
from dataclasses import dataclass
from typing import ClassVar, Optional, Type, Union, Sequence

import gym
from avalanche.training.strategies import LwF, BaseStrategy
from sequoia.methods import register_method
from sequoia.settings.passive import (
    ClassIncrementalSetting,
    PassiveEnvironment,
    TaskIncrementalSetting,
)
from simple_parsing.helpers.hparams import uniform, categorical
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from .base import AvalancheMethod


@register_method
@dataclass
class LwFMethod(AvalancheMethod[LwF], target_setting=ClassIncrementalSetting):
    """ Learning without Forgetting strategy from Avalanche.
    See LwF plugin for details.
    This strategy does not use task identities.

    See the parent class `AvalancheMethod` for the other hyper-parameters and methods.
    """

    # changing the 'name' in this case here, because the default name would be
    # 'lw_f'.
    name: ClassVar[str] = "lwf"
    # distillation hyperparameter. It can be either a float number or a list containing
    # alpha for each experience.
    alpha: Union[float, Sequence[float]] = uniform(
        1e-2, 1, default=1
    )  # TODO: Check if the range makes sense.
    # softmax temperature for distillation
    temperature: float = uniform(
        1, 10, default=2
    )  # TODO: Check if the range makes sense.

    strategy_class: ClassVar[Type[LwF]] = LwF


if __name__ == "__main__":
    from simple_parsing import ArgumentParser

    setting = TaskIncrementalSetting(
        dataset="mnist", nb_tasks=5, monitor_training_performance=True
    )
    # Create the Method, either manually or through the command-line:
    parser = ArgumentParser(__doc__)
    parser.add_arguments(LwFMethod, "method")
    args = parser.parse_args()
    method: LwFMethod = args.method

    results = setting.apply(method)
