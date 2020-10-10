""" Defines the IIDSetting, as a variant of the TaskIncremental setting with
only one task.
"""
from dataclasses import dataclass
from typing import (Callable, ClassVar, Dict, List, Optional, Tuple, Type,
                    TypeVar, Union)
import itertools
import tqdm
from torch import Tensor
from common.loss import Loss
from common.metrics import Metrics
from common.config import Config
from settings.base import Results
from utils.utils import constant, dict_union

from .. import TaskIncrementalSetting
from settings.passive.cl.batch_transforms import DropTaskLabels
from .iid_results import IIDResults

# TODO: Remove the task labels here.

@dataclass
class IIDSetting(TaskIncrementalSetting):
    """Your 'usual' learning Setting, where the samples are i.i.d.
    
    Implemented as a variant of Task-Incremental CL, but with only one task.
    
    """
    Results: ClassVar[Type[Results]] = IIDResults

    # Held constant, since this is an IID setting.
    nb_tasks: int = constant(1)
    increment: Union[int, List[int]] = constant(None)
    # A different task size applied only for the first task.
    # Desactivated if `increment` is a list.
    initial_increment: int = constant(None)
    # An optional custom class order, used for NC.
    class_order: Optional[List[int]] = constant(None)
    # Either number of classes per task, or a list specifying for
    # every task the amount of new classes (defaults to the value of
    # `increment`).
    test_increment: Optional[Union[List[int], int]] = constant(None)
    # A different task size applied only for the first test task.
    # Desactivated if `test_increment` is a list. Defaults to the
    # value of `initial_increment`.
    test_initial_increment: Optional[int] = constant(None)
    # An optional custom class order for testing, used for NC.
    # Defaults to the value of `class_order`.
    test_class_order: Optional[List[int]] = constant(None)

    def apply(self, method: "Method", config: Config):
        # TODO: Trying to figure out a way to also allow plain/simple
        # pytorch-lightning training in the case of IID settings/methods.
        # if method.target_setting in {TaskIncrementalSetting, ClassIncrementalSetting}:
        #     return super().evaluate(method)
        # else:
        #     # Plain old pytorch-lightning training.
        #     pass
        return super().apply(method, config)

    def test_loop(self, method: "Method") -> IIDResults:
        test_metrics = []
        env = self.test_dataloader()
        observations = env.reset()
        
        actions = method.get_actions(observations, env.action_space)
        with tqdm.tqdm(itertools.count(), total=len(env)) as pbar:
            for i in pbar:
                observations, rewards, done, info = env.step(actions)
                actions = method.get_actions(observations, env.action_space)

                batch_metrics = self.get_metrics(actions=actions, rewards=rewards)
                test_metrics.append(batch_metrics)
                pbar.set_postfix(batch_metrics.to_pbar_message())
                if done:
                    break

        return self.Results(test_metrics)


SettingType = TypeVar("SettingType", bound=IIDSetting)

if __name__ == "__main__":
    IIDSetting.main()
