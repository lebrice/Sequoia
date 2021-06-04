import inspect
import json
import operator
import warnings
from contextlib import redirect_stdout
from dataclasses import dataclass
from functools import partial
from io import StringIO
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, List, Optional, Type, Union

import gym
from gym import spaces
from simple_parsing import field, list_field

from sequoia.common.gym_wrappers import MultiTaskEnvironment, TransformObservation
from sequoia.common.gym_wrappers.utils import is_monsterkong_env
from sequoia.common.spaces import Sparse
from sequoia.common.transforms import Transforms
from sequoia.utils import constant, dict_union
from sequoia.utils.logging_utils import get_logger

from ..discrete import DiscreteTaskAgnosticRLSetting
from sequoia.settings.rl.continual import ContinualRLSetting
from sequoia.settings.rl.envs import (
    ATARI_PY_INSTALLED,
    MetaMonsterKongEnv,
    MetaWorldEnv,
    MTEnv,
    MujocoEnv,
    metaworld_envs,
    METAWORLD_INSTALLED,
    MONSTERKONG_INSTALLED,
    mtenv_envs,
    MTENV_INSTALLED,
    MUJOCO_INSTALLED,
)

from .tasks import make_incremental_task, is_supported

logger = get_logger(__file__)


# TODO: When we add a wrapper to 'concat' two environments, then we can move this
# 'passing custom env for each task' feature up into DiscreteTaskAgnosticRL.


@dataclass
class IncrementalRLSetting(DiscreteTaskAgnosticRLSetting):
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
    available_datasets: ClassVar[
        Dict[str, str]
    ] = DiscreteTaskAgnosticRLSetting.available_datasets

    dataset: str = "CartPole-v0"

    train_envs: List[Union[str, Callable[[], gym.Env]]] = list_field()
    val_envs: List[Union[str, Callable[[], gym.Env]]] = list_field()
    test_envs: List[Union[str, Callable[[], gym.Env]]] = list_field()

    def __post_init__(self):
        if not self.nb_tasks:
            # TODO: In case of the metaworld envs, we could determine the 'default' nb
            # of tasks to use based on the number of available tasks
            pass

        if is_monsterkong_env(self.dataset):
            if self.force_pixel_observations:
                # Add this to the kwargs that will be passed to gym.make, to make sure that
                # we observe pixels, and not state.
                self.base_env_kwargs["observe_state"] = False
            elif self.force_state_observations:
                self.base_env_kwargs["observe_state"] = True

        self._using_custom_envs_foreach_task: bool = False
        if self.train_envs:
            self._using_custom_envs_foreach_task = True
            self.nb_tasks = len(self.train_envs)
            # TODO: Not sure what to do with the `self.dataset` field, because
            # ContinualRLSetting expects to have a single env, while we have more than
            # one, the __post_init__ tries to create the rest of the fields based on
            # `self.dataset`
            self.dataset = self.train_envs[0]

            if not self.val_envs:
                # TODO: Use a wrapper that sets a different random seed
                self.val_envs = self.train_envs.copy()
            if not self.test_envs:
                # TODO: Use a wrapper that sets a different random seed
                self.test_envs = self.train_envs.copy()
            if (
                self.train_task_schedule
                or self.valid_task_schedule
                or self.test_task_schedule
            ):
                raise RuntimeError(
                    "You can either pass `train/valid/test_envs`, or a task schedule, "
                    "but not both!"
                )
        else:
            if self.val_envs or self.test_envs:
                raise RuntimeError(
                    "Can't pass `val_envs` or `test_envs` without passing `train_envs`."
                )

        super().__post_init__()

        if self._using_custom_envs_foreach_task:
            # TODO: Use 'no-op' task schedules for now.
            # self.train_task_schedule.clear()
            # self.valid_task_schedule.clear()
            # self.test_task_schedule.clear()
            pass

            # TODO: Check that all the envs have the same observation spaces!
            # (If possible, find a way to check this without having to instantiate all
            # the envs.)

        if self.dataset == "MetaMonsterKong-v0":
            # TODO: Limit the episode length in monsterkong?
            # TODO: Actually end episodes when reaching a task boundary, to force the
            # level to change?
            self.max_episode_steps = self.max_episode_steps or 500

        # FIXME: Really annoying little bugs with these three arguments!
        self.nb_tasks = self.max_steps // self.steps_per_task

    @property
    def task_label_space(self) -> gym.Space:
        # TODO: Explore an alternative design for the task sampling, based more around
        # gym spaces rather than the generic function approach that's currently used?
        # IDEA: Might be cleaner to put this in the assumption class
        task_label_space = spaces.Discrete(self.nb_tasks)
        if not self.task_labels_at_train_time or not self.task_labels_at_test_time:
            sparsity = 1
            if (self.task_labels_at_train_time ^ self.task_labels_at_test_time):
                # We have task labels "50%" of the time, ish:
                sparsity = 0.5
            task_label_space = Sparse(task_label_space, sparsity=sparsity)
        return task_label_space
    
    
    def setup(self, stage: str = None) -> None:
        # Called before the start of each task during training, validation and
        # testing.
        super().setup(stage=stage)
        # What's done in ContinualRLSetting:
        # if stage in {"fit", None}:
        #     self.train_wrappers = self.create_train_wrappers()
        #     self.valid_wrappers = self.create_valid_wrappers()
        # elif stage in {"test", None}:
        #     self.test_wrappers = self.create_test_wrappers()
        if self._using_custom_envs_foreach_task:
            logger.debug(
                f"Using custom environments from `self.[train/val/test]_envs` for task "
                f"{self.current_task_id}."
            )
            # NOTE: Here is how this supports passing custom envs for each task: We just
            # switch out the value of this property, and let the
            # `train/val/test_dataloader` methods work as usual!
            self.dataset = self.train_envs[self.current_task_id]
            self.val_dataset = self.val_envs[self.current_task_id]
            self.test_dataset = self.test_envs[self.current_task_id]

            # TODO: Check that the observation/action spaces are all the same for all
            # the train/valid/test envs
            self._check_all_envs_have_same_spaces(
                envs_or_env_functions=self.train_envs, wrappers=self.train_wrappers,
            )
            # TODO: Inconsistent naming between `val_envs` and `valid_wrappers` etc.
            self._check_all_envs_have_same_spaces(
                envs_or_env_functions=self.val_envs, wrappers=self.valid_wrappers,
            )
            self._check_all_envs_have_same_spaces(
                envs_or_env_functions=self.test_envs, wrappers=self.test_wrappers,
            )
        else:
            # TODO: Should we populate the `self.train_envs`, `self.val_envs` and
            # `self.test_envs` fields here as well, just to be consistent?
            # base_env = self.dataset
            # def task_env(task_index: int) -> Callable[[], MultiTaskEnvironment]:
            #     return self._make_env(
            #         base_env=base_env,
            #         wrappers=[],
            #     )
            # self.train_envs = [partial(gym.make, self.dataset) for i in range(self.nb_tasks)]
            # self.val_envs = [partial(gym.make, self.dataset) for i in range(self.nb_tasks)]
            # self.test_envs = [partial(gym.make, self.dataset) for i in range(self.nb_tasks)]
            # assert False, self.train_task_schedule
            pass

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
        **base_env_kwargs: Dict,
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
            if MTENV_INSTALLED and env_id in mtenv_envs:
                from mtenv import make as mtenv_make

                # This is super weird. Don't undestand at all
                # why they are doing this. Makes no sense to me whatsoever.
                base_env = mtenv_make(env_id, **base_env_kwargs)

                # Add a wrapper that will remove the task information, because we use
                # the same MultiTaskEnv wrapper for all the environments.
                wrappers.insert(0, MTEnvAdapterWrapper)

            if METAWORLD_INSTALLED and env_id in metaworld_envs:
                # TODO: Should we use a particular benchmark here?
                # For now, we find the first benchmark that has an env with this name.
                import metaworld

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
            base_env=base_env, wrappers=wrappers, **base_env_kwargs,
        )

    def create_task_schedule(
        self, temp_env: gym.Env, change_steps: List[int]
    ) -> Dict[int, Dict]:
        task_schedule: Dict[int, Dict] = {}
        if self._using_custom_envs_foreach_task:
            # If custom envs were passed to be used for each task, then we don't create
            # a "task schedule", because the only reason we're using a task schedule is
            # when we want to change something about the 'base' env in order to get
            # multiple tasks.
            # Create a task schedule dict, just to fit in?
            for i, task_step in enumerate(change_steps):
                task_schedule[task_step] = {}
            return task_schedule

        # TODO: Make it possible to use something other than steps as keys in the task
        # schedule, something like a NamedTuple[int, DeltaType], e.g. Episodes(10) or
        # Steps(10), something like that!
        # IDEA: Even fancier, we could use a TimeDelta to say "do one hour of task 0"!!
        for step in change_steps:
            # TODO: Add a `stage` argument (an enum or something with 'train', 'valid'
            # 'test' as values, and pass it to this function. Tasks should be the same
            # in train/valid for now, given the same task Id.
            # TODO: When the Results become able to handle a different ordering of tasks
            # at train vs test time, allow the test task schedule to have different
            # ordering than train / valid.
            task = self.sample_task(env=temp_env, step=step, change_steps=change_steps)
            task_schedule[step] = task

        return task_schedule

    def sample_task(
        self, env: gym.Env, step: int, change_steps: List[int]
    ) -> Dict[str, Any]:
        """ Samples a task to be applied to the environment when it reaches the given
        step.
        """
        if self.task_sampling_function:
            return self.task_sampling_function(
                env,
                step=step,
                change_steps=change_steps,
                seed=self.config.seed if self.config else None,
            )

        # Check if there is a registered handler for this type of environment.
        # TODO: Maybe use the type of the env, rather then env.unwrapped, and have the
        # 'base' callable defer the call to the wrapped env?
        if make_task_for_env.dispatch(
            type(env.unwrapped)
        ) is make_task_for_env.dispatch(object):
            warnings.warn(
                RuntimeWarning(
                    f"Don't yet know how to create a task for env {env}, will use the environment as-is."
                )
            )

        # Use this singledispatch function to select how to sample a task depending on
        # the type of environment.
        return make_task_for_env(
            env.unwrapped,
            step=step,
            change_steps=change_steps,
            seed=self.config.seed if self.config else None,
        )

        return super().create_task_schedule(
            temp_env=temp_env, change_steps=change_steps
        )

    def create_train_wrappers(self):
        if self._using_custom_envs_foreach_task:
            # TODO: Maybe do something different here, since we don't actually want to
            # add a CL wrapper at all in this case?
            assert not any(self.train_task_schedule.values())
        return super().create_train_wrappers()

    def _check_all_envs_have_same_spaces(
        self,
        envs_or_env_functions: List[Union[str, gym.Env, Callable[[], gym.Env]]],
        wrappers: List[Callable[[gym.Env], gym.Wrapper]],
    ) -> None:
        """ Checks that all the environments in the list have the same
        observation/action spaces.
        """
        first_env = self._make_env(
            base_env=envs_or_env_functions[0], wrappers=wrappers, **self.base_env_kwargs
        )
        first_env.close()
        for task_id, task_env_id_or_function in zip(
            range(1, len(envs_or_env_functions)), envs_or_env_functions[1:]
        ):
            task_env = self._make_env(
                base_env=task_env_id_or_function,
                wrappers=wrappers,
                **self.base_env_kwargs,
            )
            task_env.close()
            if task_env.observation_space != first_env.observation_space:
                raise RuntimeError(
                    f"Env at task {task_id} doesn't have the same observation "
                    f"space ({task_env.observation_space}) as the environment of "
                    f"the first task: {first_env.observation_space}."
                )
            if task_env.action_space != first_env.action_space:
                raise RuntimeError(
                    f"Env at task {task_id} doesn't have the same action "
                    f"space ({task_env.action_space}) as the environment of "
                    f"the first task: {first_env.action_space}"
                )

    def _make_wrappers(
        self,
        base_env: Union[str, gym.Env, Callable[[], gym.Env]],
        task_schedule: Dict[int, Dict],
        # sharp_task_boundaries: bool,
        task_labels_available: bool,
        transforms: List[Transforms],
        starting_step: int,
        max_steps: int,
        new_random_task_on_reset: bool,
    ) -> List[Callable[[gym.Env], gym.Env]]:
        if self._using_custom_envs_foreach_task:
            if task_schedule:
                logger.warning(
                    RuntimeWarning(
                        f"Ignoring task schedule {task_schedule}, since custom envs were "
                        f"passed for each task!"
                    )
                )
            task_schedule = None

        wrappers = super()._make_wrappers(
            base_env=base_env,
            task_schedule=task_schedule,
            task_labels_available=task_labels_available,
            transforms=transforms,
            starting_step=starting_step,
            max_steps=max_steps,
            new_random_task_on_reset=new_random_task_on_reset,
        )

        if self._using_custom_envs_foreach_task:
            # If the user passed a specific env to use for each task, then there won't
            # be a MultiTaskEnv wrapper in `wrappers`, since the task schedule is
            # None/empty.
            # Instead, we will add a Wrapper that always gives the task ID of the
            # current task.

            # TODO: There are some 'unused' args above: `starting_step`, `max_steps`,
            # `new_random_task_on_reset` which are still passed to the super() call, but
            # just unused.
            if new_random_task_on_reset:
                raise NotImplementedError(
                    "TODO: Add a MultiTaskEnv wrapper of some sort that alternates "
                    " between the source envs."
                )

            assert not task_schedule
            task_label = self.current_task_id
            task_label_space = spaces.Discrete(self.nb_tasks)
            if not task_labels_available:
                task_label = None
                task_label_space = Sparse(task_label_space, sparsity=1.0)

            wrappers.append(
                partial(
                    AddTaskIDWrapper,
                    task_label=task_label,
                    task_label_space=task_label_space,
                )
            )

        if is_monsterkong_env(base_env):
            # TODO: Need to register a MetaMonsterKong-State-v0 or something like that!
            # TODO: Maybe add another field for 'force_state_observations' ?
            # if self.force_pixel_observations:
            pass
        return wrappers


class MTEnvAdapterWrapper(TransformObservation):
    # TODO: For now, we remove the task id portion of the space and of the observation
    # dicts.
    def __init__(self, env: MTEnv, f: Callable = operator.itemgetter("env_obs")):
        super().__init__(env=env, f=f)
        # self.observation_space = self.env.observation_space["env_obs"]

    # def observation(self, observation):
    #     return observation["env_obs"]


from typing import Any

from sequoia.common.gym_wrappers.multi_task_environment import add_task_labels
from sequoia.common.gym_wrappers.utils import IterableWrapper


class AddTaskIDWrapper(IterableWrapper):
    """ Wrapper that adds always the same given task id to the observations.

    Used when the list of envs for each task is passed, so that each env also has the
    task id as part of their observation space and in their observations.
    """

    def __init__(
        self, env: gym.Env, task_label: Optional[int], task_label_space: gym.Space
    ):
        super().__init__(env=env)
        self.task_label = task_label
        self.task_label_space = task_label_space
        self.observation_space = add_task_labels(
            self.env.observation_space, task_labels=task_label_space
        )

    def observation(self, observation: Union[IncrementalRLSetting.Observations, Any]):
        return add_task_labels(observation, self.task_label)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        return self.observation(obs), reward, done, info
