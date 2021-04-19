import operator
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from typing import Callable, ClassVar, Dict, List, Optional, Type, Union

import gym
from simple_parsing import field

from sequoia.common.gym_wrappers import MultiTaskEnvironment, TransformObservation
from sequoia.utils import constant, dict_union
from sequoia.utils.logging_utils import get_logger

from ..continual_rl_setting import ContinualRLSetting

logger = get_logger(__file__)

try:
    with redirect_stdout(StringIO()):
        from meta_monsterkong.make_env import MetaMonsterKongEnv
except ImportError:
    monsterkong_installed = False
else:
    monsterkong_installed = True


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
    from metaworld.envs.mujoco.mujoco_env import MujocoEnv

    metaworld_envs = list(metaworld.ML10().train_classes.keys())
    metaworld_installed = True
except ImportError:
    # Create a 'dummy' class so we can safely use MetaWorldEnv in the type hints below.
    # Additionally, isinstance(some_env, MetaWorldEnv) will always fail when metaworld
    # isn't installed, which is good.
    class MetaWorldEnv(gym.Env):
        pass

    class MujocoEnv(gym.Env):
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
        if not self.nb_tasks:
            # TODO: In case of the metaworld envs, we could device the 'default' nb of
            # tasks to use based on the number of available tasks
            pass

        super().__post_init__(*args, **kwargs)

        if self.dataset == "MetaMonsterKong-v0":
            # TODO: Limit the episode length in monsterkong?
            # TODO: Actually end episodes when reaching a task boundary, to force the
            # level to change?
            self.max_episode_steps = self.max_episode_steps or 500

        # FIXME: Really annoying little bugs with these three arguments!
        self.nb_tasks = self.max_steps // self.steps_per_task

    def _setup_fields_using_temp_env(self, temp_env: MultiTaskEnvironment):
        """ Setup some of the fields on the Setting using a temporary environment.

        This temporary environment only lives during the __post_init__() call.
        """
        super()._setup_fields_using_temp_env(temp_env)
        # TODO: If the dataset has a `max_path_length` attribute, then it's probably
        # a Mujoco / metaworld / etc env, and so we set a limit on the episode length to
        # avoid getting an error.
        max_path_length: Optional[int] = getattr(temp_env, "max_path_length", None)
        if self.max_episode_steps is None and max_path_length is not None:
            assert max_path_length > 0
            self.max_episode_steps = temp_env.max_path_length

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
        # Check if the env is registed in a known 'third party' gym-like package, and if
        # needed, create the base env in the way that package requires.
        if isinstance(base_env, str):
            env_id = base_env

            # Check if the id belongs to mtenv
            if mtenv_installed and env_id in mtenv_envs:
                from mtenv import make

                base_env = make(env_id)
                # Add a wrapper that will remove the task information, because we use
                # the same MultiTaskEnv wrapper for all the environments.
                wrappers.insert(0, MTEnvAdapterWrapper)

            if metaworld_installed and env_id in metaworld_envs:
                # TODO: Should we use a particular benchmark here?
                # For now, we find the first benchmark that has an env with this name.
                for benchmark_class in [metaworld.ML10]:
                    benchmark = benchmark_class()
                    if env_id in benchmark.train_classes.keys():
                        # TODO: We can either let the base_env be an env type, or
                        # actually instantiate it.
                        base_env: Type[MetaWorldEnv] = benchmark.train_classes[env_id]
                        # NOTE: (@lebrice) Here I believe it's better to just have the
                        # constructor, that way we re-create the env for each task.
                        # I think this might be better, as I don't know for sure that
                        # the `set_task` can be called more than once in metaworld.
                        # base_env = base_env_type()
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

        if isinstance(temp_env.unwrapped, (MetaWorldEnv, MujocoEnv)):
            # TODO: Which benchmark to choose?
            base_env = temp_env.unwrapped
            found = False
            # Find the benchmark that contains this type of env.
            for benchmark_class in [metaworld.ML10]:
                benchmark = benchmark_class()
                for env_name, env_class in benchmark.train_classes.items():
                    if isinstance(base_env, env_class):
                        # Found the right benchmark that contains this env class, now
                        # create the task schedule using
                        # the tasks.
                        found = True
                        break
                if found:
                    break
            if not found:
                raise NotImplementedError(
                    f"Can't find a benchmark with env class {type(base_env)}!"
                )

            # `benchmark` is here the right benchmark to use to create the tasks.
            training_tasks = [
                task for task in benchmark.train_tasks if task.env_name == env_name
            ]
            task_schedule = {
                step: operator.methodcaller("set_task", task)
                for step, task in zip(change_steps, training_tasks)
            }
            return task_schedule

        return super().create_task_schedule(
            temp_env=temp_env, change_steps=change_steps
        )

    def create_train_wrappers(self):
        return super().create_train_wrappers()


class MTEnvAdapterWrapper(TransformObservation):
    # TODO: For now, we remove the task id portion of the space
    def __init__(self, env: MTEnv, f: Callable = operator.itemgetter("env_obs")):
        super().__init__(env=env, f=f)
        # self.observation_space = self.env.observation_space["env_obs"]

    # def observation(self, observation):
    #     return observation["env_obs"]
