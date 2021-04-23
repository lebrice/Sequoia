""" CWRStar Method from Avalanche. """
from dataclasses import dataclass
from typing import ClassVar, Optional, Type

from sequoia.methods import register_method
from sequoia.settings.passive import ClassIncrementalSetting, TaskIncrementalSetting

from avalanche.training.strategies import BaseStrategy, CWRStar

from .base import AvalancheMethod


@register_method
@dataclass
class CWRStarMethod(AvalancheMethod[CWRStar], target_setting=ClassIncrementalSetting):
    """ CWRStar strategy from Avalanche.
    See CWRStar plugin for details.
    This strategy does not use task identities.

    See the parent class `AvalancheMethod` for the other hyper-parameters and methods.
    """

    # Name of the CWR layer. Defaults to None, which means that the last fully connected
    # layer will be used.
    cwr_layer_name: Optional[str] = None

    strategy_class: ClassVar[Type[BaseStrategy]] = CWRStar


if __name__ == "__main__":
    from simple_parsing import ArgumentParser

    setting = TaskIncrementalSetting(
        dataset="mnist", nb_tasks=5, monitor_training_performance=True
    )
    # Create the Method, either manually or through the command-line:
    parser = ArgumentParser(__doc__)
    parser.add_arguments(CWRStarMethod, "method")
    args = parser.parse_args()
    method: CWRStarMethod = args.method

    results = setting.apply(method)
