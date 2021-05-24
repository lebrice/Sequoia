""" Method based on AR1 from [Avalanche](https://github.com/ContinualAI/avalanche).

See `avalanche.training.strategies.ar1.AR1` for more info.
"""
from dataclasses import dataclass
from typing import ClassVar, Type

from avalanche.training.strategies import AR1, BaseStrategy
from simple_parsing.helpers.hparams import uniform, log_uniform

from sequoia.methods import register_method
from sequoia.settings.sl import ClassIncrementalSetting, TaskIncrementalSLSetting
from .base import AvalancheMethod


@register_method
@dataclass
class AR1Method(AvalancheMethod[AR1], target_setting=ClassIncrementalSetting):
    """ AR1 strategy from Avalanche.
    See AR1 plugin for details.
    This strategy does not use task identities.

    See the parent class `AvalancheMethod` for the other hyper-parameters and methods.
    """

    # The learning rate (SGD optimizer).
    lr: float = log_uniform(1e-6, 1e-2, default=0.001)
    # The momentum (SGD optimizer).
    momentum: float = uniform(0.9, 0.999, default=0.9)
    # The L2 penalty used for weight decay.
    l2: float = uniform(1e-6, 1e-3, default=0.0005)
    # The number of training epochs. Defaults to 4.
    train_epochs: int = uniform(1, 50, default=4)
    # The initial update rate of BatchReNorm layers.
    init_update_rate: float = 0.01
    # The incremental update rate of BatchReNorm layers.
    inc_update_rate: float = 0.00005
    # The maximum r value of BatchReNorm layers.
    max_r_max: float = 1.25
    # The maximum d value of BatchReNorm layers.
    max_d_max: float = 0.5
    # The incremental step of r and d values of BatchReNorm layers.
    inc_step: float = 4.1e-05
    # The size of the replay buffer. The replay buffer is shared across classes.
    rm_sz: int = uniform(500, 2000, default=1500)
    # A string describing the name of the layer to use while freezing the lower
    # (nearest to the input) part of the model. The given layer is not frozen
    # (exclusive).
    freeze_below_layer: str = "lat_features.19.bn.beta"
    # The number of the layer to use as the Latent Replay Layer. Usually this is the
    # same of `freeze_below_layer`.
    latent_layer_num: int = 19
    # The Synaptic Intelligence lambda term. Defaults to 0, which means that the
    # Synaptic Intelligence regularization will not be applied.
    ewc_lambda: float = uniform(0, 1, default=0)
    # The train minibatch size. Defaults to 128.
    train_mb_size: int = uniform(1, 512, default=128)
    # The eval minibatch size. Defaults to 128.
    eval_mb_size: int = uniform(1, 512, default=128)

    strategy_class: ClassVar[Type[BaseStrategy]] = AR1


if __name__ == "__main__":
    from simple_parsing import ArgumentParser

    setting = TaskIncrementalSLSetting(
        dataset="mnist", nb_tasks=5, monitor_training_performance=True
    )
    # Create the Method, either manually or through the command-line:
    parser = ArgumentParser(__doc__)
    parser.add_arguments(AR1Method, "method")
    args = parser.parse_args()
    method: AR1Method = args.method

    results = setting.apply(method)
