""" 'Naive' method from [Avalanche](https://github.com/ContinualAI/avalanche).

See `avalanche.training.strategies.Naive` for more info.
"""
from typing import ClassVar, Type

from avalanche.training.strategies import BaseStrategy, Naive

from sequoia.settings.sl import TaskIncrementalSLSetting

from .base import AvalancheMethod


class NaiveMethod(AvalancheMethod[Naive]):
    """ 'Naive' Strategy from [Avalanche](https://github.com/ContinualAI/avalanche).

    The simplest (and least effective) Continual Learning strategy. Naive just
    incrementally fine tunes a single model without employing any method
    to contrast the catastrophic forgetting of previous knowledge.
    This strategy does not use task identities.

    Naive is easy to set up and its results are commonly used to show the worst
    performing baseline.

    See the parent class `AvalancheMethod` for the other hyper-parameters and methods.
    """

    strategy_class: ClassVar[Type[BaseStrategy]] = Naive


if __name__ == "__main__":
    setting = TaskIncrementalSLSetting(
        dataset="mnist", nb_tasks=5, monitor_training_performance=True
    )
    method = NaiveMethod()
    results = setting.apply(method)
