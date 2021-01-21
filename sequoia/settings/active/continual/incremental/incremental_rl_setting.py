from dataclasses import dataclass
from functools import partial
from typing import Callable, ClassVar, Dict, Iterable, List, Tuple

import gym
from gym.wrappers import TimeLimit
from monsterkong_randomensemble.make_env import MetaMonsterKongEnv
from pytorch_lightning import LightningModule
from simple_parsing import choice

from sequoia.common.gym_wrappers import MultiTaskEnvironment
from sequoia.common.transforms import Transforms
from sequoia.utils import constant, dict_union
from sequoia.utils.logging_utils import get_logger

from ..continual_rl_setting import ContinualRLSetting, HideTaskLabelsWrapper
from ..gym_dataloader import GymDataLoader

logger = get_logger(__file__)


@dataclass
class IncrementalRLSetting(ContinualRLSetting):
    """ Continual RL setting the data is divided into 'tasks' with clear boundaries.

    By default, the task labels are given at train time, but not at test time.

    TODO: Decide how to implement the train procedure, if we give a single
    dataloader, we might need to call the agent's `on_task_switch` when we reach
    the task boundary.. Or, we could produce one dataloader per task, and then
    implement a custom `fit` procedure in the CLTrainer class, that loops over
    the tasks and calls the `on_task_switch` when needed.
    """

    # Number of tasks.
    nb_tasks: int = 10
    # Wether the task boundaries are smooth or sudden.
    smooth_task_boundaries: bool = constant(False)
    # Wether to give access to the task labels at train time.
    task_labels_at_train_time: bool = True
    # Wether to give access to the task labels at test time.
    task_labels_at_test_time: bool = False

    # Class variable that holds the dict of available environments.
    available_datasets: ClassVar[Dict[str, str]] = dict_union(
        ContinualRLSetting.available_datasets, {"monsterkong": "MetaMonsterKong-v0",},
    )
    dataset: str = choice(available_datasets, default="cartpole")

    # def __post_init__(self, *args, **kwargs):
    #     super().__post_init__(*args, **kwargs)

    def _make_wrappers(
        self,
        task_schedule: Dict[int, Dict],
        sharp_task_boundaries: bool,
        task_labels_available: bool,
        transforms: List[Transforms],
        starting_step: int,
        max_steps: int,
    ) -> List[Callable[[gym.Env], gym.Env]]:
        wrappers = super()._make_wrappers(
            task_schedule,
            sharp_task_boundaries,
            task_labels_available,
            transforms,
            starting_step,
            max_steps,
        )
        if self.dataset == "MetaMonsterKong-v0":
            # TODO: Limit the episode length?
            wrappers.insert(0, partial(TimeLimit, max_episode_steps=100))
        return wrappers

    def create_task_schedule(
        self, temp_env: MultiTaskEnvironment, change_steps: List[int]
    ) -> Dict[int, Dict]:
        task_schedule: Dict[int, Dict] = {}
        if isinstance(temp_env.unwrapped, MetaMonsterKongEnv):
            for i, task_step in enumerate(change_steps):
                task_schedule[task_step] = {"level": i}
            return task_schedule
        else:
            return super().create_task_schedule(
                temp_env=temp_env, change_steps=change_steps
            )
