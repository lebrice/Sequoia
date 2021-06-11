""" 'Classical' RL setting.
"""
from dataclasses import dataclass
from typing import List, Callable
import gym
from ..incremental import IncrementalRLSetting
from sequoia.utils.utils import constant
from typing_extensions import Final
# NOTE: We can reuse those results for now, since they describe the same thing.
from ..discrete.results import DiscreteTaskAgnosticRLResults as TraditionalRLResults


@dataclass
class TraditionalRLSetting(IncrementalRLSetting):
    """ Your usual "Classical" Reinforcement Learning setting.

    Implemented as a MultiTaskRLSetting, but with a single task.
    """

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
