""" Method based on GEM from [Avalanche](https://github.com/ContinualAI/avalanche).

See `avalanche.training.plugins.gem.GEMPlugin` or
`avalanche.training.strategies.strategy_wrappers.GEM` for more info.
"""
from dataclasses import dataclass
from typing import ClassVar, Type

from simple_parsing import ArgumentParser
from simple_parsing.helpers.hparams import uniform
from avalanche.training.strategies import GEM, BaseStrategy

from sequoia.methods import register_method
from sequoia.settings.sl import ClassIncrementalSetting, TaskIncrementalSLSetting

from .base import AvalancheMethod


@register_method
@dataclass
class GEMMethod(AvalancheMethod[GEM]):
    """Gradient Episodic Memory (GEM) strategy from Avalanche.
    See GEM plugin for details.
    This strategy does not use task identities.

    See the parent class `AvalancheMethod` for the other hyper-parameters and methods.
    """

    # number of patterns per experience in the memory
    patterns_per_exp: int = uniform(10, 1000, default=100)
    # Offset to add to the projection direction in order to favour backward transfer
    # (gamma in original paper).
    memory_strength: float = uniform(1e-2, 1.0, default=0.5)

    strategy_class: ClassVar[Type[BaseStrategy]] = GEM


if __name__ == "__main__":
    setting = TaskIncrementalSLSetting(
        dataset="mnist", nb_tasks=5, monitor_training_performance=True
    )
    # Create the Method, either manually or through the command-line:
    parser = ArgumentParser(__doc__)
    parser.add_arguments(GEMMethod, "method")
    args = parser.parse_args()
    method: GEMMethod = args.method

    results = setting.apply(method)
