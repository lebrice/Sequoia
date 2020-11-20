import itertools
import warnings
import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import (Callable, ClassVar, Dict, List, Optional, Sequence, Type,
                    Union)

import gym
import numpy as np
from gym import spaces
from gym.envs.atari import AtariEnv
from gym.envs.classic_control import CartPoleEnv
from gym.wrappers import AtariPreprocessing

from common import Batch, Config
from common.gym_wrappers import (MultiTaskEnvironment, SmoothTransitions,
                                 TransformAction, TransformObservation,
                                 TransformReward)
from common.gym_wrappers.batch_env import BatchedVectorEnv, SyncVectorEnv, VectorEnv
from common.gym_wrappers.env_dataset import EnvDataset
from common.gym_wrappers.pixel_observation import PixelObservationWrapper
from common.gym_wrappers.sparse_space import Sparse
from common.gym_wrappers.step_callback_wrapper import StepCallbackWrapper
from common.gym_wrappers.utils import (IterableWrapper,
                                       classic_control_envs,
                                       classic_control_env_prefixes, 
                                       has_wrapper)
from common.metrics import RegressionMetrics
from common.transforms import Transforms
from settings.active import ActiveSetting
from settings.assumptions.incremental import (IncrementalSetting,
                                              TestEnvironment)
from settings.base import Method
from settings.base.results import Results
from simple_parsing import choice, list_field
from simple_parsing.helpers import dict_field
from torch import Tensor
from utils import dict_union, get_logger

from .. import ActiveEnvironment
from .gym_dataloader import GymDataLoader
from .make_env import make_batched_env
from .rl_results import RLResults
from .wrappers import (AddDoneToObservation, HideTaskLabelsWrapper,
                       NoTypedObjectsWrapper, RemoveTaskLabelsWrapper,
                       TypedObjectsWrapper, AddInfoToObservation)
from stable_baselines3.common.atari_wrappers import AtariWrapper

logger = get_logger(__file__)

# TODO: Implement a get_metrics (ish) in the Environment, not on the Setting!
# TODO: The validation environment will also call the on_task_switch when it
# reaches a task boundary, and there isn't currently a way to distinguish if
# that method is being called because of the training or because of the
# validation environment.


task_params: Dict[Union[Type[gym.Env], str], List[str]] = {
    "CartPole-v0": [
        "gravity", #: 9.8,
        "masscart", #: 1.0,
        "masspole", #: 0.1,
        "length", #: 0.5,
        "force_mag", #: 10.0,
        "tau", #: 0.02,
    ],
    # TODO: Add more of the classic control envs here.
    # TODO: Need to get the attributes to modify in each environment type and
    # add them here.
    AtariEnv: [
        # TODO: Maybe have something like the difficulty as the CL 'task' ?
        # difficulties = temp_env.ale.getAvailableDifficulties()
        # "game_difficulty",
    ],      
}


Environment = ActiveEnvironment["ContinualRLSetting.Observations",
                                "ContinualRLSetting.Observations",
                                "ContinualRLSetting.Rewards"]


@dataclass
class ContinualRLSetting(IncrementalSetting, ActiveSetting):
    """ Reinforcement Learning Setting where the environment changes over time.

    This is an Active setting which uses gym environments as sources of data.
    These environments' attributes could change over time following a task
    schedule. An example of this could be that the gravity increases over time
    in cartpole, making the task progressively harder as the agent interacts with
    the environment.
    """
    Results: ClassVar[Type[Results]] = RLResults
    
    @dataclass(frozen=True)
    class Observations(IncrementalSetting.Observations,
                       ActiveSetting.Observations):
        """ Observations in a continual RL Setting. """
        # Just as a reminder, these are the fields defined in the base classes:
        # x: Tensor
        # task_labels: Union[Optional[Tensor], Sequence[Optional[Tensor]]] = None
        
        # The 'done' part of the 'step' method. We add these two here in case a
        # method were to iterate on the environments in the dataloader-style so
        # they also have access to those.
        done: Optional[Sequence[bool]] = None
        # Same, for the 'info' portion of the result of 'step'.
        info: Optional[Sequence[Dict]] = None

    transforms: List[Transforms] = list_field(Transforms.to_tensor, Transforms.channels_first_if_needed)

    # Class variable that holds the dict of available environments.
    available_datasets: ClassVar[Dict[str, str]] = {
        "cartpole": "CartPole-v0",
        "pendulum": "Pendulum-v0",
        "breakout": "Breakout-v0",
        # "duckietown": "Duckietown-straight_road-v0"
    }
    # Which environment (a.k.a. "dataset") to learn on.
    # The dataset could be either a string (env id or a key from the
    # available_datasets dict), a gym.Env, or a callable that returns a single environment.
    # If self.dataset isn't one of those, an error will be raised.
    dataset: str = choice(available_datasets, default="breakout")

    # Max number of steps ("length" of the training and test "datasets").
    max_steps: int = 10_000
    # Number of steps per task. When left unset, takes the value of `max_steps`
    # divided by `nb_tasks`.
    steps_per_task: Optional[int] = None
    # Wether the task boundaries are smooth or sudden.
    smooth_task_boundaries: bool = True

    # Number of episodes to perform through the test environment in the test
    # loop. Depending on what an 'episode' might represent in your setting, this
    # could be as low as 1 (for example, supervised learning, episode == epoch).
    test_loop_episodes: int = 100

    # Wether we want to bypass the encoder entirely and use the observation
    # coming from the environment as our 'representation'. This can be useful to
    # debug environments like CartPole, for instance.
    observe_state_directly: bool = False

    train_task_schedule: Dict[int, Dict] = dict_field()
    valid_task_schedule: Dict[int, Dict] = dict_field()
    test_task_schedule: Dict[int, Dict] = dict_field()


    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)
        # Post processing of the 'dataset' field, so that it is a proper env id
        # if the name from the dict of available_dataset above was used, so for
        # instance: "breakout" -> "Breakout-v0".
        if self.dataset in self.available_datasets.keys():
            # the environment name was passed, rather than an id
            # (e.g. 'cartpole' -> 'CartPole-v0").
            self.dataset = self.available_datasets[self.dataset]
        
        if self.dataset in self.available_datasets.values():
            # dataset is already a dataset 'id' from the dict of available
            # dataset, all good, do nothing.
            pass
        
        elif isinstance(self.dataset, str):
            # The passed dataset is assumed to be an environment ID, but it
            # wasn't in the dict of available datasets! We issue a warning, but
            # proceed to let the user use whatever environment they want to.
            logger.warning(UserWarning(
                f"The chosen dataset/environment ({self.dataset}) isn't in the "
                f"available_datasets dict, so we can't garantee this will work!"
            ))
        elif isinstance(self.dataset, gym.Env):
            # A gym.Env object was passed as the 'dataset'.
            pass
        elif callable(self.dataset):
            # self.dataset is a callable (an env factory function).
            pass
        
        if not isinstance(self.dataset, (str, gym.Env)) and not callable(self.dataset):
            raise RuntimeError(f"`dataset` must be either a string, a gym.Env, "
                               f"or a callable. (got {self.dataset})")

        # Set the number of tasks depending on the increment, and vice-versa.
        # (as only one of the two should be used).
        assert self.max_steps
        if not self.nb_tasks:
            if self.steps_per_task:
                self.nb_tasks = self.max_steps // self.steps_per_task
            elif self.train_task_schedule:
                self.nb_tasks = len(self.train_task_schedule) - (1 if self.max_steps in self.train_task_schedule else 0)
            else:
                self.nb_tasks = 1
                self.steps_per_task = self.max_steps
        elif not self.steps_per_task:
            self.steps_per_task = int(self.max_steps / self.nb_tasks)
        
        if self.smooth_task_boundaries:
            # If we're operating in the 'Online/smooth task transitions' "regime",
            # then there is only one "task", and we don't have task labels.
            self.known_task_boundaries_at_train_time = False
            self.known_task_boundaries_at_test_time = False
            self.task_labels_at_train_time = False
            self.task_labels_at_test_time = False
            self.nb_tasks = 1
            self.steps_per_task = self.max_steps

        # Task schedules for training / validation and testing.

        # Create a temporary environment so we can extract the spaces.
        with self.make_env(self.dataset, self.temp_wrappers()) as temp_env:
            # Populate the task schedules created above.
            if not self.train_task_schedule:
                self.create_task_schedules(temp_env)
            # Set the spaces using the temp env.
            self.observation_space = temp_env.observation_space
            self.action_space = temp_env.action_space
            self.reward_range = temp_env.reward_range
            self.reward_space = getattr(temp_env, "reward_space",
                                        spaces.Box(low=self.reward_range[0],
                                                high=self.reward_range[1],
                                                shape=()))
        del temp_env
        # This attribute will hold a reference to the `on_task_switch` method of
        # the Method being currently applied on this setting.
        self.on_task_switch_callback: Callable[[Optional[int]], None] = None

    def create_task_schedules(self, temp_env: MultiTaskEnvironment) -> None:
        assert self.nb_tasks == 1
        self.train_task_schedule[0] = temp_env.default_task
        self.train_task_schedule[self.max_steps] = temp_env.random_task()
        
        # For now, set the validation and test tasks as the same sequence as the
        # train tasks.
        self.valid_task_schedule = self.train_task_schedule.copy() 
        self.test_task_schedule = self.train_task_schedule.copy()

    @staticmethod
    def make_env(base_env: Union[str, gym.Env, Callable[[], gym.Env]],
                 wrappers: List[Callable[[gym.Env], gym.Env]]=None) -> gym.Env:
        """ Make a single (not-batched) environment.
        
        Would be nice if we could just pass this as the env_fn to the batched
        environments, but this unfortunately can't work, because we have
        references to `self` here. 
        """
        env: gym.Env
        if isinstance(base_env, str):
            env = gym.make(base_env)
        elif isinstance(base_env, gym.Env):
            env = base_env
        elif callable(base_env):
            env = base_env()
        else:
            raise RuntimeError(
                f"base_env should either be a string, a callable, or a gym "
                f"env. (got {base_env})."
            )
        for wrapper in wrappers:
            env = wrapper(env)
        return env
    
    def temp_wrappers(self) -> List[Callable[[gym.Env], gym.Env]]:
        """ Gets the minimal wrappers needed to figure out the Spaces of the
        train_env.
        
        This is called in the 'constructor' (__post_init__) to set the Setting's
        observation/action/reward spaces, so this should depend on as little
        state from `self` as possible, since not all attributes have been
        defined at the point where this is called. 
        """
        wrappers: List[Callable[[gym.Env], gym.Env]] = []
        if not self.observe_state_directly:
            # If we are in a classic control env, and we dont want the state to
            # be fully-observable (i.e. we want pixel observations rather than
            # getting the pole angle, velocity, etc.), then add the
            # PixelObservation wrapper to the list of wrappers.
            
            # TODO: Change the BaselineMethod so that it uses an nn.Identity()
            # or the like in the case where setting.observe_state_directly is True.
            if ((isinstance(self.dataset, str) and
                 self.dataset.startswith(classic_control_env_prefixes))
                or isinstance(self.dataset, classic_control_envs)):
                wrappers.append(PixelObservationWrapper)
                            
        # TODO: Test & Debug this: Adding the Atari preprocessing wrapper.
        if self.dataset.startswith("Breakout") or isinstance(self.dataset, AtariEnv):
            wrappers.append(AtariWrapper)
            # wrappers.append(AtariPreprocessing)
            
        
        if not self.observe_state_directly:
            # Apply the image transforms to the env.
            wrappers.append(partial(TransformObservation, f=self.train_transforms))

        # Add a wrapper that creates the 'tasks' (non-stationarity in the env).
        # First, get the set of parameters that will be changed over time.
        cl_task_params = task_params.get(self.dataset, [])
        if self.smooth_task_boundaries:
            cl_wrapper = SmoothTransitions
        else:
            cl_wrapper = MultiTaskEnvironment
        # We want to have a 'task-label' space, but it will be filled with
        # None values when task boundaries are smooth.
        wrappers.append(partial(cl_wrapper, task_params=cl_task_params, add_task_id_to_obs=True))
        return wrappers

    def apply(self, method: Method, config: Config=None) -> "ContinualRLSetting.Results":
        """Apply the given method on this setting to producing some results."""
        self.config = config or Config.from_args(self._argv)
        method.config = self.config

        self.configure(method)
        method.configure(setting=self)
        
        logger.info(f"Train task schedule:" + json.dumps(self.train_task_schedule, indent="\t"))
        # Run the Training loop (which is defined in IncrementalSetting).
        self.train_loop(method)
        
        # NOTE: Since we want to be pass a reference to the method's
        # 'on_task_switch' callback, we have to store it here.
        if self.known_task_boundaries_at_test_time:
            self.on_task_switch_callback = getattr(method, "on_task_switch", None)
            if self.on_task_switch_callback is None:
                logger.warning(UserWarning(
                    f"Task boundaries are available at test time, but the "
                    f"method doesn't have an 'on_task_switch' callback."
                ))
        
        # Run the Test loop (which is defined in IncrementalSetting).
        results: RlResults = self.test_loop(method)
        logger.info("Results summary:")
        logger.info(results.summary())
        method.receive_results(self, results=results)
        return results

    @property
    def env_name(self) -> Optional[str]:
        """Returns the gym 'id' associated with the selected dataset/env, or an
        empty string if the env doesn't have a spec. 

        Returns
        -------
        str
            The env's spec.id, or an empty string if the env doesn't have one.
        """
        if self.dataset in self.available_datasets.values():
            return self.dataset
        if self.dataset in self.available_datasets.keys():
            return self.available_datasets[self.dataset]
        if isinstance(self.dataset, str):
            return self.dataset
        if self.dataset.spec:
            return self.dataset.spec.id
        logger.warning(RuntimeWarning(
            f"Can't tell what the selected env's name/id is, returning None. "
            f"(dataset is {self.dataset})"
        ))
        return None

    def setup(self, stage=None):
        # Not really doing anything here.
        return super().setup(stage=stage)

    def prepare_data(self, *args, **kwargs):
        # We don't really download anything atm.
        return super().prepare_data(*args, **kwargs)

    def train_dataloader(self, batch_size: int = None, num_workers: int = None) -> ActiveEnvironment:
        if not self.has_prepared_data:
            self.prepare_data()
        if not self.has_setup_fit:
            self.setup("fit")

        batch_size = batch_size or self.train_batch_size or self.batch_size
        num_workers = num_workers or self.num_workers
        logger.debug(f"batch_size: {batch_size}, num_workers: {num_workers}")

        if batch_size is None:
            env = self.make_env(self.dataset, wrappers=self.train_wrappers())
        elif num_workers is None:
            warnings.warn(UserWarning(
                f"Running {batch_size} environments in series (very slow!) "
                f"since the num_workers is None."
            ))
            env = SyncVectorEnv([
                partial(self.make_env, base_env=self.dataset, wrappers=self.train_wrappers())
                for _ in range(batch_size)
            ])
        else:
            env = BatchedVectorEnv(
                [partial(self.make_env, base_env=self.dataset, wrappers=self.train_wrappers())
                    for _ in range(batch_size) 
                ],
                n_workers=num_workers,
                # TODO: Still debugging shared memory + Sparse spaces.
                shared_memory=False,
            )

        if self.config and self.config.seed:
            if batch_size:
                # Seed each environment differently, but based on the base seed.
                env.seed([self.config.seed + i for i in range(dataloader.num_envs)])
            else:
                env.seed(self.config.seed)
        else:
            env.seed(None)

        # Apply the "post-batch" wrappers:
        # Add a wrapper that converts numpy arrays / etc to Observations/Rewards
        # and from Actions objects to numpy arrays.
        env = TypedObjectsWrapper(
            env,
            observations_type=self.Observations,
            rewards_type=self.Rewards,
            actions_type=self.Actions,
        )        
        # Create an IterableDataset from the env using the EnvDataset wrapper.
        dataset = EnvDataset(env, max_steps=self.steps_per_task)
        
        # Create a GymDataLoader for the EnvDataset.
        dataloader = GymDataLoader(dataset)
        
        self.train_env = dataloader
        return self.train_env

    def train_wrappers(self) -> List[Callable[[gym.Env], gym.Env]]:
        """Get the list of wrappers to add to each training environment.
        
        The result of this method must be pickleable when using
        multiprocessing.

        Returns
        -------
        List[Callable[[gym.Env], gym.Env]]
            [description]
        """
        wrappers: List[Callable[[gym.Env], gym.Env]] = []
        # TODO: Add some kind of Wrapper around the dataset to make it
        # semi-supervised?
        if not self.observe_state_directly:
            # If we are in a classic control env, and we dont want the state to
            # be fully-observable (i.e. we want pixel observations rather than
            # getting the pole angle, velocity, etc.), then add the
            # PixelObservation wrapper to the list of wrappers.
            # TODO: Change the BaselineMethod so that it uses an nn.Identity()
            # or the like in the case where setting.observe_state_directly is True.
            if ((isinstance(self.dataset, str) and
                 self.dataset.startswith(classic_control_env_prefixes))
                or isinstance(self.dataset, classic_control_envs)):
                wrappers.append(PixelObservationWrapper)
                
        # TODO: Test & Debug this: Adding the Atari preprocessing wrapper.
        if self.dataset.startswith("Breakout") or isinstance(self.dataset, AtariEnv):
            wrappers.append(AtariWrapper)
        
        if not self.observe_state_directly:
            # Wrapper to apply the image transforms to the env.
            wrappers.append(partial(TransformObservation, f=self.train_transforms))

        if self.smooth_task_boundaries:
            # Add a wrapper that creates smooth 'tasks' (changes in the env).
            # We allow iteration over the entire stream
            # (no start_step and max_step)
            # TODO: Should this maybe be a wrapper to be applied on top of a
            # MultiTaskEnvironment, rather than as a subclass/replacement to
            # MultiTaskEnvironment?
            assert not self.known_task_boundaries_at_train_time
            wrappers.append(partial(SmoothTransitions,
                task_schedule=self.train_task_schedule,
                # Add the 'task_id' space to the observation space, but it will
                # be Sparse, and all of its samples will always be 'None'.
                add_task_id_to_obs=True,
                # Add the 'task dict' to the 'info' dict.
                add_task_dict_to_info=True,
            ))

        elif self.known_task_boundaries_at_train_time:
            assert self.nb_tasks >= 1
            # Add a wrapper that creates sharp 'tasks'.
            # We add a restriction to prevent users from getting data from
            # previous or future tasks.
            starting_step = self.current_task_id * self.steps_per_task
            max_steps = (self.current_task_id + 1) * self.steps_per_task - 1

            # Add the task id to the observation.
            # Add the 'task dict' to the 'info' dict.
            wrappers.append(partial(MultiTaskEnvironment,
                task_schedule=self.train_task_schedule,
                add_task_id_to_obs=True,
                add_task_dict_to_info=True,
                starting_step=starting_step,
                max_steps=max_steps,
            ))
            # NOTE: Since we want a 'None' task label when not available,
            # instead of not adding the task labels above, we instead set
            # them to None with another wrapper here.
            # We could also add an argument to the MultiTaskEnvironment, but it 
            # already has enough responsability as it is imo.

        # Apply the "post-batch" wrappers:
        if not self.task_labels_at_train_time:
            # TODO: Hide or remove the task labels?
            # wrappers.append(RemoveTaskLabelsWrapper)
            wrappers.append(HideTaskLabelsWrapper)
        return wrappers
    
    def val_dataloader(self, batch_size: int = None, num_workers: int = None) -> Environment:
        if not self.has_prepared_data:
            self.prepare_data()
        if not self.has_setup_fit:
            self.setup("fit")
        
        batch_size = batch_size or self.valid_batch_size or self.batch_size
        num_workers = num_workers or self.num_workers
        logger.debug(f"batch_size: {batch_size}, num_workers: {num_workers}")
        
        if batch_size is None:
            env = self.make_env(self.dataset, wrappers=self.train_wrappers())
        elif num_workers is None:
            warnings.warn(UserWarning(
                f"Running {batch_size} environments in series (very slow!) "
                f"since the num_workers is None."
            ))
            env = SyncVectorEnv([
                partial(self.make_env, base_env=self.dataset, wrappers=self.train_wrappers())
                for _ in range(batch_size)
            ])
        else:
            env = BatchedVectorEnv(
                [partial(self.make_env, base_env=self.dataset, wrappers=self.train_wrappers())
                    for _ in range(batch_size) 
                ],
                n_workers=num_workers,
                # TODO: Still debugging shared memory + Sparse spaces.
                shared_memory=False,
            )
        
        if self.config.seed:
            if batch_size:
                # Seed each environment differently, but based on the base seed.
                env.seed([self.config.seed + i for i in range(dataloader.num_envs)])
            else:
                env.seed(self.config.seed)
        else:
            env.seed(None)

        # Apply the "post-batch" wrappers:
        env = TypedObjectsWrapper(
            env,
            observations_type=self.Observations,
            actions_type=self.Actions,
            rewards_type=self.Rewards,
        )
        dataset = EnvDataset(env, max_steps=self.steps_per_task)
        dataloader = GymDataLoader(dataset)
        self.val_env = dataloader
        return self.val_env
    
    def val_wrappers(self) -> List[Callable[[gym.Env], gym.Env]]:
        """Get the list of wrappers to add to each validation environment.
        
        The result of this method must be pickleable when using
        multiprocessing.

        Returns
        -------
        List[Callable[[gym.Env], gym.Env]]
            [description]
        """
        wrappers: List[Callable[[gym.Env], gym.Env]] = []
        if not self.observe_state_directly:
            if ((isinstance(self.dataset, str) and
                 self.dataset.startswith(classic_control_env_prefixes))
                or isinstance(self.dataset, classic_control_envs)):
                wrappers.append(PixelObservationWrapper)
                
        if self.dataset.startswith("Breakout") or isinstance(self.dataset, AtariEnv):
            wrappers.append(AtariWrapper)    
            # wrappers.append(AtariPreprocessing)    
        
        if not self.observe_state_directly:
            wrappers.append(partial(TransformObservation, f=self.val_transforms))
        
        # TODO: Should the validation environment have task labels if the train
        # env does but not the test env? 
        elif self.known_task_boundaries_at_train_time:
            starting_step = self.current_task_id * self.steps_per_task
            max_steps = (self.current_task_id + 1) * self.steps_per_task - 1
            wrappers.append(partial(MultiTaskEnvironment,
                task_schedule=self.valid_task_schedule,
                add_task_id_to_obs=True,
                add_task_dict_to_info=True,
                starting_step=starting_step,
                max_steps=max_steps,
            ))

        # TODO: See below, should the validation environment have task labels if
        # the train env does but not the test env? 
        if not self.task_labels_at_test_time:
            # TODO: The RemoveTaskLabelsWrapper makes the 'task label' space
            # Sparse with none_prob=1., but it still allows the user to see the
            # number of tasks.
            # TODO: Hide or remove the task labels?
            # wrappers.append(RemoveTaskLabelsWrapper)
            wrappers.append(HideTaskLabelsWrapper)
        return wrappers

    def test_dataloader(self, batch_size: int = None, num_workers: int = None) -> TestEnvironment:
        # NOTE: The test environment isn't just for the current task, it
        # contains the sequence of all tasks.
        if not self.has_prepared_data:
            self.prepare_data()
        if not self.has_setup_test:
            self.setup("test")

        batch_size = batch_size or self.test_batch_size or self.batch_size
        num_workers = num_workers or self.num_workers

        if batch_size is None:
            env = self.make_env(self.dataset, wrappers=self.test_wrappers())
        elif num_workers is None:
            warnings.warn(UserWarning(
                f"Running {batch_size} environments in series (very slow!) "
                f"since the num_workers is None."
            ))
            env = SyncVectorEnv([
                partial(self.make_env, base_env=self.dataset, wrappers=self.test_wrappers())
                for _ in range(batch_size)
            ])
        else:
            env = BatchedVectorEnv(
                [partial(self.make_env, base_env=self.dataset, wrappers=self.test_wrappers())
                    for _ in range(batch_size) 
                ],
                n_workers=num_workers,
                # TODO: Still debugging shared memory + Sparse spaces.
                shared_memory=False,
            )

        if self.config.seed:
            if batch_size:
                # Seed each environment differently, but based on the base seed.
                env.seed([self.config.seed + i for i in range(dataloader.num_envs)])
            else:
                env.seed(self.config.seed)
        else:
            env.seed(None)

        # If we know the task boundaries at test time, and the method has the
        # callback for it, then we add a callback wrapper that will invoke the
        # method's on_task_switch and pass it the task label if required when on
        # a task boundary.
        # This is different than the train or validation environments, since the
        # test environment might cover multiple tasks, while if the task labels
        # are available at train time, then each train/valid environment is for
        # a single task.

        env = TypedObjectsWrapper(
            env,
            observations_type=self.Observations,
            rewards_type=self.Rewards,
            actions_type=self.Actions,
        )
        dataset = EnvDataset(env, max_steps=self.steps_per_task)
        dataloader = GymDataLoader(dataset)

        # TODO: We should probably change the max_steps depending on the
        # batch size of the env.
        test_loop_max_steps = self.max_steps
        # TODO: Find where to configure this 'test directory' for the outputs of
        # the Monitor.
        test_dir = "results"
        self.test_env = ContinualRLTestEnvironment(
            dataloader,
            directory=test_dir,
            step_limit=self.max_steps,
            force=True,
        )
        return self.test_env

    def test_wrappers(self) -> List[Callable[[gym.Env], gym.Env]]:
        """Get the list of wrappers to add to a single test environment.
        
        The result of this method must be pickleable when using
        multiprocessing.

        Returns
        -------
        List[Callable[[gym.Env], gym.Env]]
            [description]
        """
        wrappers: List[Callable[[gym.Env], gym.Env]] = []
        if not self.observe_state_directly:
            if ((isinstance(self.dataset, str) and
                 self.dataset.startswith(classic_control_env_prefixes))
                or isinstance(self.dataset, classic_control_envs)):
                wrappers.append(PixelObservationWrapper)
                
        if self.dataset.startswith("Breakout") or isinstance(self.dataset, AtariEnv):
            wrappers.append(AtariWrapper)    
            # wrappers.append(AtariPreprocessing)    
        
        if not self.observe_state_directly:
            wrappers.append(partial(TransformObservation, f=self.test_transforms))

        if self.smooth_task_boundaries:
            wrappers.append(partial(SmoothTransitions,
                task_schedule=self.test_task_schedule,
                add_task_id_to_obs=True,
                # TODO: Figure this out, the info dicts have the env attributes atm.
                add_task_dict_to_info=self.task_labels_at_test_time,
            ))

        elif self.known_task_boundaries_at_test_time:
            # TODO: We maybe don't actually want to get an env for just one task, but instead get an environment spanning all tasks.
            # rather we want a test env that covers the whole 'stream'.
            starting_step = 0
            # TODO: Is the limit on the number of test steps the same as for training steps?
            max_steps = self.max_steps
            # starting_step = self.current_task_id * self.steps_per_task
            # max_steps = (self.current_task_id + 1) * self.steps_per_task - 1
            wrappers.append(partial(MultiTaskEnvironment,
                task_schedule=self.test_task_schedule,
                add_task_id_to_obs=True,
                add_task_dict_to_info=True,
                starting_step=starting_step,
                max_steps=max_steps,
            ))
        if not self.task_labels_at_test_time:
            # TODO: Hide or remove the task labels?
            # wrappers.append(RemoveTaskLabelsWrapper)
            wrappers.append(HideTaskLabelsWrapper)
        return wrappers


class ContinualRLTestEnvironment(TestEnvironment, IterableWrapper):
    def get_results(self) -> RLResults:
        # TODO: Create a RLMetrics object?
        rewards = self.get_episode_rewards()
        lengths = self.get_episode_lengths()
        total_steps = self.get_total_steps()
        
        assert has_wrapper(self.env, MultiTaskEnvironment), self.env
        task_steps = sorted(self.task_schedule.keys())
        
        assert 0 in task_steps
        import bisect
        nb_tasks = len(task_steps)
        assert nb_tasks >= 1
        episode_rewards: List[float] = [[] for _ in range(nb_tasks)]
        episode_lengths: List[int] = [[] for _ in range(nb_tasks)]
        episode_metrics: List[Metrics] = [[] for _ in range(nb_tasks)]

        for step, episode_reward, episode_length in zip(itertools.accumulate(lengths), rewards, lengths):
            # Given the step, find the task id.
            task_id = bisect.bisect_right(task_steps, step) - 1
            
            episode_rewards[task_id].append(episode_reward)
            episode_lengths[task_id].append(episode_length)
            episode_metric = RegressionMetrics(n_samples=episode_length, mse=episode_reward / episode_length)
            episode_metrics[task_id].append(episode_metric)

        return RLResults(
            episode_lengths=episode_lengths,
            episode_rewards=episode_rewards,
            test_metrics=episode_metrics,
        )

    def render(self, mode='human', **kwargs):
        from common.gym_wrappers.batch_env.tile_images import tile_images
        image_batch = super().render(mode=mode, **kwargs)
        if mode == "rgb_array":
            return tile_images(image_batch)
        return image_batch
        
    def _after_reset(self, observation):
        # Is this going to work fine when the observations are batched though?
        return super()._after_reset(observation)


if __name__ == "__main__":
    ContinualRLSetting.main()
