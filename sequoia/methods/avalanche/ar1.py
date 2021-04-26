""" AR1 Method from Avalanche. """
from dataclasses import dataclass
from typing import ClassVar, Type

from avalanche.training.strategies import AR1, BaseStrategy
from sequoia.methods import register_method
from sequoia.settings.passive import ClassIncrementalSetting, TaskIncrementalSetting

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
    lr: float = 0.001
    # The momentum (SGD optimizer).
    momentum: float = 0.9
    # The L2 penalty used for weight decay.
    l2: float = 0.0005
    # The number of training epochs. Defaults to 4.
    train_epochs: int = 4
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
    rm_sz: int = 1500
    # A string describing the name of the layer to use while freezing the lower
    # (nearest to the input) part of the model. The given layer is not frozen
    # (exclusive).
    freeze_below_layer: str = "lat_features.19.bn.beta"
    # The number of the layer to use as the Latent Replay Layer. Usually this is the
    # same of `freeze_below_layer`.
    latent_layer_num: int = 19
    # The Synaptic Intelligence lambda term. Defaults to 0, which means that the
    # Synaptic Intelligence regularization will not be applied.
    ewc_lambda: float = 0
    # The train minibatch size. Defaults to 1.
    train_mb_size: int = 128
    # The eval minibatch size. Defaults to 1.
    eval_mb_size: int = 128

    strategy_class: ClassVar[Type[BaseStrategy]] = AR1


if __name__ == "__main__":
    from simple_parsing import ArgumentParser

    setting = TaskIncrementalSetting(
        dataset="mnist", nb_tasks=5, monitor_training_performance=True
    )
    # Create the Method, either manually or through the command-line:
    parser = ArgumentParser(__doc__)
    parser.add_arguments(AR1Method, "method")
    args = parser.parse_args()
    method: AR1Method = args.method

    results = setting.apply(method)
