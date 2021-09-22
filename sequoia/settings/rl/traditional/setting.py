""" 'Classical' RL setting.
"""
from dataclasses import dataclass
from typing import Callable, ClassVar, Dict, List, Any

import gym
from sequoia.utils.utils import constant
from simple_parsing.helpers import choice
from typing_extensions import Final

# NOTE: We can reuse those results for now, since they describe the same thing.
from ..discrete.results import DiscreteTaskAgnosticRLResults as TraditionalRLResults
from ..incremental import IncrementalRLSetting


@dataclass
class TraditionalRLSetting(IncrementalRLSetting):
    """ Your usual "Classical" Reinforcement Learning setting.

    Implemented as a MultiTaskRLSetting, but with a single task.
    """

    # Class variable that holds the dict of available environments.
    available_datasets: ClassVar[
        Dict[str, str]
    ] = IncrementalRLSetting.available_datasets.copy()
    # Which dataset/environment to use for training, validation and testing.
    dataset: str = choice(available_datasets, default="CartPole-v0")

    # IDEA: By default, only use one task, although there may actually be more than one.
    nb_tasks: int = 5

    stationary_context: Final[bool] = constant(True)
    known_task_boundaries_at_train_time: Final[bool] = constant(True)
    task_labels_at_train_time: Final[bool] = constant(True)
    task_labels_at_test_time: bool = False

    # Results: ClassVar[Type[Results]] = TaskSequenceResults

    def __post_init__(self):
        super().__post_init__()
        assert self.stationary_context

    def apply(self, method, config=None):
        results: IncrementalRLSetting.Results = super().apply(method, config=config)
        assert len(results.task_sequence_results) == 1
        return results.task_sequence_results[0]
        # result: TraditionalRLResults = TraditionalRLResults(task_results=results.task_sequence_results[0].task_results)
        result: TraditionalRLResults = results.task_sequence_results[0]
        # assert False, result._runtime
        return result

    @property
    def phases(self) -> int:
        """The number of training 'phases', i.e. how many times `method.fit` will be
        called.

        Defaults to the number of tasks, but may be different, for instance in so-called
        Multi-Task Settings, this is set to 1.
        """
        return 1

    # TODO: Double check whether actually need this method
    def getattr_recursive(self, name: str) -> Any:
        """Recursively check wrappers to find attribute.

        :param name: name of attribute to look for
        :return: attribute
        """
        all_attributes = self._get_all_attributes()
        if name in all_attributes:  # attribute is present in this wrapper
            attr = getattr(self, name)
        elif hasattr(self.venv, "getattr_recursive"):
            # Attribute not present, child is wrapper. Call getattr_recursive rather than getattr
            # to avoid a duplicate call to getattr_depth_check.
            attr = self.venv.getattr_recursive(name)
        else:  # attribute not present, child is an unwrapped VecEnv
            attr = getattr(self.venv, name)

        return attr

    # TODO: What is the correct way to do this?
    def reset(self):
        self.num_envs = self.train_env.num_envs
        reset_data = self.train_env.reset()
        return reset_data.x


