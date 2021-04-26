from .base import AvalancheMethod
from avalanche.training.strategies import Naive, BaseStrategy
from typing import ClassVar, Type
from sequoia.settings.passive import TaskIncrementalSetting


class NaiveMethod(AvalancheMethod[Naive]):
    """ 'Naive' Strategy from Avalanche.

    The simplest (and least effective) Continual Learning strategy. Naive just
    incrementally fine tunes a single model without employing any method
    to contrast the catastrophic forgetting of previous knowledge.
    This strategy does not use task identities.

    Naive is easy to set up and its results are commonly used to show the worst
    performing baseline.
    """

    strategy_class: ClassVar[Type[BaseStrategy]] = Naive


if __name__ == "__main__":
    setting = TaskIncrementalSetting(
        dataset="mnist", nb_tasks=5, monitor_training_performance=True
    )
    method = NaiveMethod()
    results = setting.apply(method)
