import operator
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from typing import Callable, ClassVar, Dict, List, Union

import gym
from sequoia.common.gym_wrappers import MultiTaskEnvironment
from sequoia.utils import constant, dict_union
from sequoia.utils.logging_utils import get_logger
from simple_parsing import field

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


mtenv_installed = False
try:
    from mtenv import MTEnv
    from mtenv.envs.registration import mtenv_registry

    mtenv_envs = [env_spec.id for env_spec in mtenv_registry.all()]
    mtenv_installed = True
except ImportError:
    mtenv_envs = []
    # Create a 'dummy' class so we can safely use MTEnv in the type hints below.
    # Additionally, isinstance(some_env, MTEnv) will always fail, which is good.

    class MTEnv(gym.Env):
        pass


mtenv_installed = False
mtenv_envs = []
try:
    from mtenv import MTEnv
    from mtenv.envs.registration import mtenv_registry

    mtenv_envs = [env_spec.id for env_spec in mtenv_registry.all()]
    mtenv_installed = True
except ImportError:
    # Create a 'dummy' class so we can safely use MTEnv in the type hints below.
    # Additionally, isinstance(some_env, MTEnv) will always fail when mtenv isn't
    # installed, which is good.
    class MTEnv(gym.Env):
        pass


metaworld_installed = False
metaworld_envs = []
try:
    import metaworld
    from metaworld import MetaWorldEnv
    from mtenv.envs.registration import mtenv_registry

    metaworld_envs = list(metaworld.ML10().train_classes.keys())
    metaworld_installed = True
except ImportError:
    # Create a 'dummy' class so we can safely use MetaWorldEnv in the type hints below.
    # Additionally, isinstance(some_env, MetaWorldEnv) will always fail when metaworld
    # isn't installed, which is good.
    class MetaWorldEnv(gym.Env):
        pass


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
        ContinualRLSetting.available_datasets, {"monsterkong": "MetaMonsterKong-v0"},
    )
    dataset: str = "CartPole-v0"

    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)

        if self.dataset == "MetaMonsterKong-v0":
            # TODO: Limit the episode length in monsterkong?
            # TODO: Actually end episodes when reaching a task boundary, to force the
            # level to change?
            self.max_episode_steps = self.max_episode_steps or 500
        # TODO: Really annoying little bugs with these three arguments!
        self.nb_tasks = self.max_steps // self.steps_per_task

        # TODO: If the dataset is an mtenv or metaworld env, don't try to use rendered
        # observations.

    @property
    def phases(self) -> int:
        """The number of training 'phases', i.e. how many times `method.fit` will be
        called.

        In this Incremental-RL Setting, fit is called once per task.
        (Same as ClassIncrementalSetting in SL).
        """
        return self.nb_tasks

    @staticmethod
    def _make_env(
        base_env: Union[str, gym.Env, Callable[[], gym.Env]],
        wrappers: List[Callable[[gym.Env], gym.Env]] = None,
        observe_state_directly: bool = False,
    ) -> gym.Env:
        """ Helper function to create a single (non-vectorized) environment.

        This is also used to create the env whenever `self.dataset` is a string that
        isn't registered in gym. This happens for example when using an environment from
        meta-world (or mtenv).
        """
        # Check if the env is registed in mtenv, and create the base env that way, since
        # it won't be possible to register the env that way.
        if isinstance(base_env, str):
            env_id = base_env
            if env_id in mtenv_envs:
                assert mtenv_installed, "mtenv is required to use this env."
                from mtenv import make

                base_env = make(env_id)
                # Add a wrapper that will remove the "tas"
                wrappers.insert(0, MTEnvAdapterWrapper)

            if env_id in metaworld_envs:
                assert metaworld_installed, "metaworld is required to use this env."
                # TODO: Should we use a particular benchmark here?
                # IDEA: Find the first benchmark that has an env with this name.
                for benchmark_class in [metaworld.ML10]:
                    benchmark = benchmark_class()
                    if env_id in benchmark.train_classes.keys():
                        base_env = benchmark.train_classes[env_id]
                        break
                else:
                    raise NotImplementedError(
                        f"Can't find a metaworld benchmark that uses env {env_id}"
                    )

        return ContinualRLSetting._make_env(
            base_env=base_env,
            wrappers=wrappers,
            observe_state_directly=observe_state_directly,
        )

    def create_task_schedule(
        self, temp_env: MultiTaskEnvironment, change_steps: List[int]
    ) -> Dict[int, Dict]:
        task_schedule: Dict[int, Dict] = {}

        if monsterkong_installed:
            if isinstance(temp_env.unwrapped, MetaMonsterKongEnv):
                for i, task_step in enumerate(change_steps):
                    task_schedule[task_step] = {"level": i}
                return task_schedule

        if isinstance(temp_env.unwrapped, MTEnv):
            for i, task_step in enumerate(change_steps):
                task_schedule[task_step] = operator.methodcaller("set_task_state", i)
            return task_schedule

        return super().create_task_schedule(
            temp_env=temp_env, change_steps=change_steps
        )

    def create_train_wrappers(self):
        return super().create_train_wrappers()


from sequoia.common.gym_wrappers import TransformObservation
import operator


class MTEnvAdapterWrapper(TransformObservation):
    # TODO: For now, we remove the task id portion of the space
    def __init__(self, env: "MTEnv", f: Callable = operator.itemgetter("env_obs")):
        super().__init__(env=env, f=f)
        # self.observation_space = self.env.observation_space["env_obs"]

    # def observation(self, observation):
    #     return observation["env_obs"]
