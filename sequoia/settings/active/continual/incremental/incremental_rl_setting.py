from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from typing import ClassVar, Dict, List

from sequoia.common.gym_wrappers import MultiTaskEnvironment
from sequoia.utils import constant, dict_union
from sequoia.utils.logging_utils import get_logger
from simple_parsing import choice, field

from ..continual_rl_setting import ContinualRLSetting
from ..gym_dataloader import GymDataLoader

logger = get_logger(__file__)

try:
    with redirect_stdout(StringIO()):
        from meta_monsterkong.make_env import MetaMonsterKongEnv
except ImportError:
    monsterkong_installed = False
else:
    monsterkong_installed = True


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

    # The number of tasks. By default 0, which means that it will be set
    # depending on other fields in __post_init__, or eventually be just 1.
    nb_tasks: int = field(0, alias=["n_tasks", "num_tasks"])
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

    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)

        if self.dataset == "MetaMonsterKong-v0":
            # TODO: Limit the episode length in monsterkong?
            # TODO: Actually end episodes when reaching a task boundary, to force the
            # level to change?
            self.max_episode_steps = self.max_episode_steps or 500

    @property
    def phases(self) -> int:
        """The number of training 'phases', i.e. how many times `method.fit` will be
        called.
        
        In this Incremental-RL Setting, fit is called once per task.
        (Same as ClassIncrementalSetting in SL).
        """
        return self.nb_tasks

    def create_task_schedule(
        self, temp_env: MultiTaskEnvironment, change_steps: List[int]
    ) -> Dict[int, Dict]:
        task_schedule: Dict[int, Dict] = {}
        if monsterkong_installed:
            if isinstance(temp_env.unwrapped, MetaMonsterKongEnv):
                for i, task_step in enumerate(change_steps):
                    task_schedule[task_step] = {"level": i}
                return task_schedule
        return super().create_task_schedule(
            temp_env=temp_env, change_steps=change_steps
        )
