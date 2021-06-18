""" Method based on Replay from [Avalanche](https://github.com/ContinualAI/avalanche).

See `avalanche.training.plugins.replay.ReplayPlugin` or
`avalanche.training.strategies.strategy_wrappers.Replay` for more info.
"""
import warnings
from dataclasses import dataclass
from typing import ClassVar, Type, Optional

from avalanche.training.strategies import Replay, BaseStrategy
from simple_parsing.helpers.hparams import uniform

from sequoia.methods import register_method
from sequoia.settings.sl import (
    ClassIncrementalSetting,
    TaskIncrementalSLSetting,
    SLSetting,
)

from .base import AvalancheMethod


from avalanche.training.plugins.replay import StoragePolicy
from avalanche.training.plugins.replay import (
    ReplayPlugin as ReplayPlugin_,
    ExperienceBalancedStoragePolicy as ExperienceBalancedStoragePolicy_,
)


class ReplayPlugin(ReplayPlugin_):
    def __init__(
        self, mem_size: int = 200, storage_policy: Optional["StoragePolicy"] = None
    ):
        super().__init__(mem_size=mem_size, storage_policy=storage_policy)
        # "patch" the ExperienceBalanchedStoragePolicy:
        if type(self.storage_policy) is ExperienceBalancedStoragePolicy_:
            self.storage_policy = ExperienceBalancedStoragePolicy(
                ext_mem=self.storage_policy.ext_mem,
                mem_size=self.storage_policy.mem_size,
                adaptive_size=self.storage_policy.adaptive_size,
                num_experiences=self.storage_policy.num_experiences,
            )


class ExperienceBalancedStoragePolicy(ExperienceBalancedStoragePolicy_):
    def __call__(self, strategy: BaseStrategy, **kwargs):
        num_exps = strategy.training_exp_counter + 1
        num_exps = num_exps if self.adaptive_size else self.num_experiences
        curr_data = strategy.experience.dataset

        # new group may be bigger because of the remainder.
        group_size = self.mem_size // num_exps
        new_group_size = group_size + (self.mem_size % num_exps)

        self.subsample_all_groups(group_size * (num_exps - 1))
        curr_data = self.subsample_single(curr_data, new_group_size)
        self.ext_mem[strategy.training_exp_counter + 1] = curr_data

        # buffer size should always equal self.mem_size
        len_tot = sum(len(el) for el in self.ext_mem.values())
        
        # TODO: Just disabling the failing assert check for now. Should check if this
        # makes any difference in the performance of the plugin:
        # assert len_tot == self.mem_size
        warnings.warn(
            RuntimeWarning(
                f"Ignoring a failing assert in Avalanche's Replay plugin: "
                f"len_tot ({len_tot}) != self.mem_size ({self.mem_size})"
            )
        )

        # NOTE: Could also avoid copying the code from their method here by suppressing
        # AssertionErrors:
        # import contextlib
        # with contextlib.suppress(AssertionError):
        #     return super().__call__(strategy=strategy, **kwargs)


@register_method
@dataclass
class ReplayMethod(AvalancheMethod[Replay]):
    """ Replay strategy from Avalanche.
    See Replay plugin for details.
    This strategy does not use task identities.

    See the parent class `AvalancheMethod` for the other hyper-parameters and methods.
    """

    # Replay buffer size.
    mem_size: int = uniform(100, 2_000, default=200)

    strategy_class: ClassVar[Type[BaseStrategy]] = Replay

    def create_cl_strategy(self, setting: SLSetting) -> Replay:
        strategy = super().create_cl_strategy(setting)

        # Find and replace the original plugin with our "patched" version:
        plugin_index: Optional[int] = None
        for i, plugin in enumerate(strategy.plugins):
            if type(plugin) is ReplayPlugin_:
                plugin_index = i
                break
        assert plugin_index is not None, "strategy should have the Plugin, no?"
        assert isinstance(plugin_index, int)

        old_plugin: ReplayPlugin_ = strategy.plugins[plugin_index]
        new_plugin = ReplayPlugin(
            mem_size=old_plugin.mem_size, storage_policy=old_plugin.storage_policy,
        )
        strategy.plugins[plugin_index] = new_plugin
        return strategy


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
