""" 'Classical' RL setting.
"""
from dataclasses import dataclass
from typing import List, Callable
import gym
from ..multi_task import MultiTaskRLSetting
from sequoia.utils.utils import constant


@dataclass
class TraditionalRLSetting(MultiTaskRLSetting):
    """ Your usual "Classical" Reinforcement Learning setting.

    Implemented as a MultiTaskRLSetting, but with a single task.
    """
    # IDEA: Only use the 'env' / dataset from the first task. 
    nb_tasks: int = 1

    def __post_init__(self):
        super().__post_init__()
        assert self.stationary_context

    @property
    def phases(self) -> int:
        """The number of training 'phases', i.e. how many times `method.fit` will be
        called.

        Defaults to the number of tasks, but may be different, for instance in so-called
        Multi-Task Settings, this is set to 1.
        """
        return 1
