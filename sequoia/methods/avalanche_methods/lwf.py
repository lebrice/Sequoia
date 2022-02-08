""" Method based on LwF from [Avalanche](https://github.com/ContinualAI/avalanche).

See `avalanche.training.plugins.lwf.LwFPlugin` or
`avalanche.training.strategies.strategy_wrappers.LwF` for more info.
"""
from dataclasses import dataclass
from typing import ClassVar, Optional, Sequence, Type, Union

from avalanche.training.plugins.lwf import LwFPlugin as LwFPlugin_
from avalanche.training.strategies import LwF
from simple_parsing.helpers.hparams import uniform
from torch import Tensor

from sequoia.methods import register_method
from sequoia.settings.sl import SLSetting, TaskIncrementalSLSetting

from .base import AvalancheMethod


class LwFPlugin(LwFPlugin_):
    """Patching a little error that happens in the 'LwFPlugin' which happens when a
    Multi-Task model is used, and when we grow the output space after each task.
    """

    def _distillation_loss(self, out: Tensor, prev_out: Tensor) -> Tensor:
        """
        Compute distillation loss between output of the current model and
        and output of the previous (saved) model.
        """
        # Little "patch" to make sure this doesn't break if the shapes aren't exactly
        # the same:
        if out.shape != prev_out.shape:
            prev_outputs = prev_out.shape[-1]
            current_outputs = out.shape[-1]
            assert prev_outputs < current_outputs
            # Only consider the loss for the overlapping classes. We assume that the
            # first columns are for the same class, so this should be fine.
            out = out[..., :prev_outputs]

        return super()._distillation_loss(out=out, prev_out=prev_out)


@register_method
@dataclass
class LwFMethod(AvalancheMethod[LwF]):
    """Learning without Forgetting strategy from Avalanche.
    See LwF plugin for details.
    This strategy does not use task identities.

    See the parent class `AvalancheMethod` for the other hyper-parameters and methods.
    """

    # changing the 'name' in this case here, because the default name would be
    # 'lw_f'.
    name: ClassVar[str] = "lwf"
    # distillation hyperparameter. It can be either a float number or a list containing
    # alpha for each experience.
    alpha: Union[float, Sequence[float]] = uniform(
        1e-2, 1, default=1
    )  # TODO: Check if the range makes sense.
    # softmax temperature for distillation
    temperature: float = uniform(1, 10, default=2)  # TODO: Check if the range makes sense.

    strategy_class: ClassVar[Type[LwF]] = LwF

    def create_cl_strategy(self, setting: SLSetting) -> LwF:
        strategy = super().create_cl_strategy(setting)

        # Find and replace the 'LwFPlugin' with our "patched" version:
        plugin_index: Optional[int] = None
        for i, plugin in enumerate(strategy.plugins):
            if type(plugin) is LwFPlugin_:
                plugin_index = i
                break
        assert plugin_index is not None, "LwF strategy should have an LwF Plugin, no?"
        assert isinstance(plugin_index, int)

        old_plugin: LwFPlugin_ = strategy.plugins[plugin_index]
        new_plugin = LwFPlugin(alpha=old_plugin.alpha, temperature=old_plugin.temperature)
        new_plugin.prev_model = old_plugin.prev_model
        strategy.plugins[plugin_index] = new_plugin

        return strategy


if __name__ == "__main__":
    from simple_parsing import ArgumentParser

    setting = TaskIncrementalSLSetting(
        dataset="mnist", nb_tasks=5, monitor_training_performance=True
    )
    # Create the Method, either manually or through the command-line:
    parser = ArgumentParser(__doc__)
    parser.add_arguments(LwFMethod, "method")
    args = parser.parse_args()
    method: LwFMethod = args.method

    results = setting.apply(method)
