""" Method based on GDumb from [Avalanche](https://github.com/ContinualAI/avalanche).

See `avalanche.training.plugins.gdumb.GDumbPlugin` or
`avalanche.training.strategies.strategy_wrappers.GDumb` for more info.

BUG: There appears to be a bug in the GDumb plugin, caused by a mismatch in the tensor
shapes when concatenating them into a TensorDataset, when batch size > 1.
"""
from dataclasses import dataclass
from typing import ClassVar, Type, Optional, Any, Dict, List, Tuple
from collections import defaultdict
import torch
import tqdm
from avalanche.benchmarks.utils import AvalancheDataset, AvalancheConcatDataset
from avalanche.training.strategies import GDumb, BaseStrategy
from avalanche.training.plugins.gdumb import GDumbPlugin as _GDumbPlugin
from simple_parsing import ArgumentParser
from simple_parsing.helpers.hparams import uniform
from torch import Tensor
from torch.utils.data import TensorDataset

from sequoia.methods import register_method
from sequoia.settings.passive import ClassIncrementalSetting, TaskIncrementalSetting
from sequoia.utils.logging_utils import get_logger

from .base import AvalancheMethod
logger = get_logger(__file__)


class GDumbPlugin(_GDumbPlugin):
    """ Patched version of the GDumbPlugin from Avalanche.

    The base implementation is quite inefficient: for each new item, it does an entire
    concatenation with the current dataset.
    This uses lists instead, and only concatenates once.

    It also uses the task labels from each sample in the dataset, rather than from the
    current experience, as there might be more than one task in the dataset.
    """

    def __init__(self, mem_size: int = 200):
        super().__init__(mem_size=mem_size)
        self.ext_mem: Dict[Any, Tuple[List[Tensor], List[Tensor]]] = {}
        # count occurrences for each class
        self.counter: Dict[Any, Dict[Any, int]] = {}

    def after_train_dataset_adaptation(self, strategy: BaseStrategy, **kwargs):
        """ Before training we make sure to organize the memory following
            GDumb approach and updating the dataset accordingly.
        """

        # for each pattern, add it to the memory or not
        dataset = strategy.experience.dataset

        pbar = tqdm.tqdm(dataset, desc="Exhausting dataset to create GDumb buffer")
        for pattern, target, task_id in pbar:
            target = torch.as_tensor(target)
            target_value = target.item()

            if len(pattern.size()) == 1:
                pattern = pattern.unsqueeze(0)

            current_counter = self.counter.setdefault(task_id, defaultdict(int))
            current_mem = self.ext_mem.setdefault(task_id, ([], []))

            if current_counter == {}:
                # any positive (>0) number is ok
                patterns_per_class = 1
            else:
                patterns_per_class = int(self.mem_size / len(current_counter.keys()))

            if (
                target_value not in current_counter
                or current_counter[target_value] < patterns_per_class
            ):
                # add new pattern into memory
                if sum(current_counter.values()) >= self.mem_size:
                    # full memory: replace item from most represented class
                    # with current pattern
                    to_remove = max(current_counter, key=current_counter.get)

                    # dataset_size = len(current_mem)
                    # for j in range(dataset_size):
                    #     if current_mem.tensors[1][j].item() == to_remove:
                    #         current_mem.tensors[0][j] = pattern
                    #         current_mem.tensors[1][j] = target
                    #         break

                    dataset_size = len(current_mem[0])
                    for j in range(dataset_size):
                        if current_mem[1][j].item() == to_remove:
                            current_mem[0][j] = pattern
                            current_mem[1][j] = target
                            break
                    current_counter[to_remove] -= 1
                else:
                    # memory not full: add new pattern
                    current_mem[0].append(pattern)
                    current_mem[1].append(target)

                # Indicate that we've changed the number of stored instances of this
                # class.
                current_counter[target_value] += 1

        task_datasets: Dict[Any, TensorDataset] = {}
        for task_id, task_mem_tuple in self.ext_mem.items():
            patterns, targets = task_mem_tuple
            task_dataset = TensorDataset(
                torch.stack(patterns, dim=0), torch.stack(targets, dim=0)
            )
            task_datasets[task_id] = task_dataset
            logger.debug(
                f"There are {len(task_dataset)} entries from task {task_id} in the new "
                f"dataset."
            )

        adapted_dataset = AvalancheConcatDataset(task_datasets.values())
        strategy.adapted_dataset = adapted_dataset


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

    # The number of training epochs.
    train_epochs: int = uniform(1, 100, default=20)

    strategy_class: ClassVar[Type[BaseStrategy]] = GDumb

    def create_cl_strategy(self, setting: ClassIncrementalSetting) -> GDumb:
        strategy = super().create_cl_strategy(setting)
        # TODO: Replace the GDumbPlugin with our own version, with the same parameters.
        old_gdumb_plugin_index: Optional[int] = None
        for i, plugin in enumerate(strategy.plugins):
            if isinstance(plugin, _GDumbPlugin):
                old_gdumb_plugin_index = i
                break

        if old_gdumb_plugin_index is None:
            raise RuntimeError("Couldn't find the Strategy's GDumb plugin!")

        old_gdumb_plugin: _GDumbPlugin = strategy.plugins.pop(old_gdumb_plugin_index)
        logger.info("Replacing the GDumbPlugin with our 'patched' version.")

        new_gdumb_plugin = GDumbPlugin(mem_size=old_gdumb_plugin.mem_size)
        # NOTE: Might not be necessarily, since those should be empty, but here we also
        # copy the state from the old plugin to the new one.
        new_gdumb_plugin.ext_mem = old_gdumb_plugin.ext_mem
        new_gdumb_plugin.counter = old_gdumb_plugin.counter

        strategy.plugins.insert(old_gdumb_plugin_index, new_gdumb_plugin)
        return strategy


if __name__ == "__main__":
    setting = TaskIncrementalSetting(
        dataset="mnist", nb_tasks=5, monitor_training_performance=True
    )
    # Create the Method, either manually or through the command-line:
    parser = ArgumentParser(__doc__)
    parser.add_arguments(GDumbMethod, "method")
    args = parser.parse_args()
    method: GDumbMethod = args.method

    results = setting.apply(method)
