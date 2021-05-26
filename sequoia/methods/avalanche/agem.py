""" Method based on AGEM from [Avalanche](https://github.com/ContinualAI/avalanche).

See `avalanche.training.plugins.agem.AGEMPlugin` or
`avalanche.training.strategies.strategy_wrappers.AGEM` for more info.
"""
from dataclasses import dataclass
from typing import ClassVar, Type

from simple_parsing import ArgumentParser
from simple_parsing.helpers.hparams import uniform
from avalanche.training.strategies import AGEM, BaseStrategy

from sequoia.methods import register_method
from sequoia.settings.sl import ClassIncrementalSetting, TaskIncrementalSLSetting

from .base import AvalancheMethod


@register_method
@dataclass
class AGEMMethod(AvalancheMethod[AGEM]):
    """Average Gradient Episodic Memory (AGEM) strategy from Avalanche.
    See AGEM plugin for details.
    This strategy does not use task identities.

    See the parent class `AvalancheMethod` for the other hyper-parameters and methods.
    """

    # number of patterns per experience in the memory
    patterns_per_exp: int = uniform(10, 1000, default=100)
    # number of patterns in memory sample when computing reference gradient.
    sample_size: int = uniform(16, 256, default=64)

    strategy_class: ClassVar[Type[BaseStrategy]] = AGEM


if __name__ == "__main__":
    setting = TaskIncrementalSLSetting(
        dataset="mnist", nb_tasks=5, monitor_training_performance=True
    )
    # Create the Method, either manually or through the command-line:
    parser = ArgumentParser(__doc__)
    parser.add_arguments(AGEMMethod, "method")
    args = parser.parse_args()
    method: AGEMMethod = args.method

    results = setting.apply(method)
