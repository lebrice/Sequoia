""" Method based on SynapticIntelligence from [Avalanche](https://github.com/ContinualAI/avalanche).

See `avalanche.training.plugins.synaptic_intelligence.SynapticIntelligencePlugin` or
`avalanche.training.strategies.strategy_wrappers.SynapticIntelligence` for more info.
"""
from dataclasses import dataclass
from typing import ClassVar, Type

from avalanche.training.strategies import BaseStrategy, SynapticIntelligence
from simple_parsing import ArgumentParser
from simple_parsing.helpers.hparams import uniform

from sequoia.methods import register_method
from sequoia.settings.sl import ClassIncrementalSetting, TaskIncrementalSetting

from .base import AvalancheMethod


@register_method
@dataclass
class SynapticIntelligenceMethod(
    AvalancheMethod[SynapticIntelligence], target_setting=ClassIncrementalSetting
):
    """ The Synaptic Intelligence strategy from Avalanche.

    This is the Synaptic Intelligence PyTorch implementation of the
    algorithm described in the paper
    "Continuous Learning in Single-Incremental-Task Scenarios"
    (https://arxiv.org/abs/1806.08568)

    The original implementation has been proposed in the paper
    "Continual Learning Through Synaptic Intelligence"
    (https://arxiv.org/abs/1703.04200).

    The Synaptic Intelligence regularization can also be used in a different
    strategy by applying the :class:`SynapticIntelligencePlugin` plugin.

    See the parent class `AvalancheMethod` for the other hyper-parameters and methods.
    """

    # Synaptic Intelligence lambda term.
    si_lambda: float = uniform(1e-2, 1.0, default=0.5)  # TODO: Check the range.

    strategy_class: ClassVar[Type[BaseStrategy]] = SynapticIntelligence


if __name__ == "__main__":

    setting = TaskIncrementalSetting(
        dataset="mnist", nb_tasks=5, monitor_training_performance=True
    )
    # Create the Method, either manually or through the command-line:
    parser = ArgumentParser(__doc__)
    parser.add_arguments(SynapticIntelligenceMethod, "method")
    args = parser.parse_args()
    method: SynapticIntelligenceMethod = args.method

    results = setting.apply(method)
