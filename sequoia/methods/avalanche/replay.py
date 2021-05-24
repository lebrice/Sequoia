""" Method based on Replay from [Avalanche](https://github.com/ContinualAI/avalanche).

See `avalanche.training.plugins.replay.ReplayPlugin` or
`avalanche.training.strategies.strategy_wrappers.Replay` for more info.
"""
from dataclasses import dataclass
from typing import ClassVar, Type

from avalanche.training.strategies import Replay, BaseStrategy
from simple_parsing.helpers.hparams import uniform

from sequoia.methods import register_method
from sequoia.settings.sl import ClassIncrementalSetting, TaskIncrementalSLSetting

from .base import AvalancheMethod


@register_method
@dataclass
class ReplayMethod(AvalancheMethod[Replay], target_setting=ClassIncrementalSetting):
    """ Replay strategy from Avalanche.
    See Replay plugin for details.
    This strategy does not use task identities.

    See the parent class `AvalancheMethod` for the other hyper-parameters and methods.
    """

    # Replay buffer size.
    mem_size: int = uniform(100, 2_000, default=200)

    strategy_class: ClassVar[Type[BaseStrategy]] = Replay


if __name__ == "__main__":
    from simple_parsing import ArgumentParser

    setting = TaskIncrementalSLSetting(
        dataset="mnist", nb_tasks=5, monitor_training_performance=True
    )
    # Create the Method, either manually or through the command-line:
    parser = ArgumentParser(__doc__)
    parser.add_arguments(ReplayMethod, "method")
    args = parser.parse_args()
    method: ReplayMethod = args.method

    results = setting.apply(method)
