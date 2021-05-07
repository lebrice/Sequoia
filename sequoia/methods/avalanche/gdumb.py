""" EWC Method from Avalanche. """
from dataclasses import dataclass
from typing import ClassVar, Type

from avalanche.training.strategies import GDumb, BaseStrategy
from sequoia.methods import register_method
from sequoia.settings.passive import ClassIncrementalSetting, TaskIncrementalSetting

from simple_parsing.helpers.hparams import uniform

from .base import AvalancheMethod


@register_method
@dataclass
class GDumbMethod(AvalancheMethod[GDumb], target_setting=ClassIncrementalSetting):
    """GDumb strategy from Avalanche.
    See GDumbPlugin for more details.
    This strategy does not use task identities.

    See the parent class `AvalancheMethod` for the other hyper-parameters and methods.
    """
    name: ClassVar[str] = "gdumb"

    # replay buffer size.
    mem_size: int = uniform(100, 1_000, default=200)

    strategy_class: ClassVar[Type[BaseStrategy]] = GDumb


if __name__ == "__main__":
    from simple_parsing import ArgumentParser

    setting = TaskIncrementalSetting(
        dataset="mnist", nb_tasks=5, monitor_training_performance=True
    )
    # Create the Method, either manually or through the command-line:
    parser = ArgumentParser(__doc__)
    parser.add_arguments(GDumbMethod, "method")
    args = parser.parse_args()
    method: GDumbMethod = args.method

    results = setting.apply(method)
