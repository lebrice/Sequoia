""" Method based on EWC from [Avalanche](https://github.com/ContinualAI/avalanche).

See `avalanche.training.plugins.ewc.EWCPlugin` or
`avalanche.training.strategies.strategy_wrappers.EWC` for more info.
"""
from dataclasses import dataclass
from typing import ClassVar, Optional, Type

from simple_parsing import ArgumentParser
from simple_parsing.helpers.hparams import categorical, uniform
from avalanche.training.strategies import EWC, BaseStrategy

from sequoia.methods import register_method
from sequoia.settings.sl import ClassIncrementalSetting, TaskIncrementalSLSetting

from .base import AvalancheMethod


@register_method
@dataclass
class EWCMethod(AvalancheMethod[EWC]):
    """
    Elastic Weight Consolidation (EWC) strategy from Avalanche.
    See EWC plugin for details.
    This strategy does not use task identities.

    See the parent class `AvalancheMethod` for the other hyper-parameters and methods.
    """

    strategy_class: ClassVar[Type[BaseStrategy]] = EWC

    # Hyperparameter to weigh the penalty inside the total loss. The larger the lambda,
    # the larger the regularization.
    ewc_lambda: float = uniform(
        1e-3, 1.0, default=0.1
    )  # todo: set the right value to use here.
    # `separate` to keep a separate penalty for each previous experience. `online` to
    # keep a single penalty summed with a decay factor over all previous tasks.
    mode: str = categorical("separate", "online", default="separate")
    # Used only if `mode` is 'online'. It specify the decay term of the
    # importance matrix.
    decay_factor: Optional[float] = uniform(0.0, 1.0, default=0.9)
    # if True, keep in memory both parameter values and importances for all previous
    # task, for all modes. If False, keep only last parameter values and importances. If
    # mode is `separate`, the value of `keep_importance_data` is set to be True.
    keep_importance_data: bool = categorical(True, False, default=False)


if __name__ == "__main__":

    setting = TaskIncrementalSLSetting(
        dataset="mnist", nb_tasks=5, monitor_training_performance=True
    )
    # Create the Method, either manually or through the command-line:
    parser = ArgumentParser(__doc__)
    parser.add_arguments(EWCMethod, "method")
    args = parser.parse_args()
    method: EWCMethod = args.method

    results = setting.apply(method)
