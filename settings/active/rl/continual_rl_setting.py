import itertools
from dataclasses import dataclass
from functools import partial
from typing import ClassVar, Dict, List, Type, Callable, Union, Optional
from pathlib import Path

import gym
import numpy as np
from gym import spaces
from gym.envs.classic_control import CartPoleEnv
from gym.envs.atari import AtariEnv
from simple_parsing import choice, list_field
from torch import Tensor

from common import Batch, Config
from common.gym_wrappers import (MultiTaskEnvironment, SmoothTransitions,
                                 TransformAction, TransformObservation,
                                 TransformReward)
from common.gym_wrappers.batch_env import BatchedVectorEnv
from common.gym_wrappers.env_dataset import EnvDataset
from common.gym_wrappers.pixel_observation import PixelObservationWrapper
from common.gym_wrappers.utils import classic_control_env_prefixes, IterableWrapper, has_wrapper
from common.gym_wrappers.step_callback_wrapper import StepCallbackWrapper
from common.gym_wrappers.sparse_space import Sparse
from common.metrics import RegressionMetrics
from common.transforms import Transforms
from settings.active import ActiveSetting
from settings.assumptions.incremental import IncrementalSetting, TestEnvironment
from settings.base.results import Results
from settings.base import Method
from utils import dict_union, get_logger

from .gym_dataloader import GymDataLoader
from .make_env import make_batched_env
from .rl_results import RLResults
from .wrappers import HideTaskLabelsWrapper, RemoveTaskLabelsWrapper, TypedObjectsWrapper, NoTypedObjectsWrapper
from .. import ActiveEnvironment

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

    transforms: List[Transforms] = list_field(Transforms.to_tensor, Transforms.channels_first_if_needed)

    # Class variable that holds the dict of available environments.
    available_datasets: ClassVar[Dict[str, str]] = {
        "cartpole": "CartPole-v0",
        "pendulum": "Pendulum-v0",
        "breakout": "Breakout-v0",
        "duckietown": "Duckietown-straight_road-v0"
    }
    # Which environment to learn on.
    dataset: str = choice(available_datasets, default="breakout")

    # Max number of steps ("length" of the training and test "datasets").
    max_steps: int = 10_000
    # Number of steps per task.
    steps_per_task: Optional[int] = None
    # Wether the task boundaries are smooth or sudden.
    smooth_task_boundaries: bool = True

    # Number of episodes to perform through the test environment in the test
    # loop. Depending on what an 'episode' might represent in your setting, this
    # could be as low as 1 (for example, supervised learning, episode == epoch).
    test_loop_episodes: int = 100
    
    
    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)
          
        # Set the number of tasks depending on the increment, and vice-versa.
        # (as only one of the two should be used).
        assert self.max_steps
        if not self.nb_tasks:
            if self.steps_per_task:
                self.nb_tasks = self.max_steps // self.steps_per_task
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
        self.train_task_schedule: Dict[int, Dict] = {}
        self.valid_task_schedule: Dict[int, Dict] = {}
        self.test_task_schedule: Dict[int, Dict] = {}

        # Create a temporary environment so we can extract the spaces.
        with self.make_temp_env() as temp_env:
            # Populate the task schedules created above.
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
    
    def make_temp_env(self) -> gym.Env:
        """ Creates a temporary environment.
        Will be called in the 'constructor' (__post_init__), so this should
        ideally depend on as little state as possible.
        """
        if self.env_name:
            temp_env = gym.make(self.env_name)
        else:
            assert callable(self.dataset), f"dataset should either be a string or a callable, got {self.dataset}"
            temp_env = self.dataset()
        # Apply the image transforms to the env.
        temp_env = TransformObservation(temp_env, f=self.train_transforms)
        # Add a wrapper that creates the 'tasks' (non-stationarity in the env).
        # First, get the set of parameters that will be changed over time.
        cl_task_params = task_params.get(self.env_name, task_params.get(type(temp_env.unwrapped), []))            
        
        if self.smooth_task_boundaries:
            cl_wrapper = SmoothTransitions
        else:
            cl_wrapper = MultiTaskEnvironment

        # We want to have a 'task-label' space, but it will be filled with
        # None values when task boundaries are smooth.
        temp_env = cl_wrapper(
            temp_env,
            task_params=cl_task_params,
            add_task_id_to_obs=True,
        )
        return temp_env
    
    
    
    def apply(self, method: Method, config: Config=None) -> "ContinualRLSetting.Results":
        """Apply the given method on this setting to producing some results."""
        self.config = config or Config.from_args(self._argv)
        method.config = self.config

        self.configure(method)
        method.configure(setting=self)
        
        # Run the Training loop (which is defined in IncrementalSetting).
        self.train_loop(method)
        
        
        # FIXME: Since we want to be pass a reference to the method's
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
    def env_name(self) -> str:
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
        elif self.dataset.spec:
            return self.dataset.spec.id
        return "" # No idea what the dataset name is, return None.
        # return self.dataset

    def setup(self, stage=None):
        return super().setup(stage=stage)

    def prepare_data(self, *args, **kwargs):
        return super().prepare_data(*args, **kwargs)

    def train_dataloader(self, batch_size: int = None) -> ActiveEnvironment:
        if not self.has_prepared_data:
            self.prepare_data()
        if not self.has_setup_fit:
            self.setup("fit")

        batch_size = batch_size or self.train_batch_size or self.batch_size

        # NOTE: Setting this to False when debugging can be very helpful,
        # especially when trying to solve problems related to the env wrappers
        # or the input preprocessing. Beware though, if the batch size isn't
        # small (< 32)  this will makes things incredibly slow.
        if batch_size is None:
            env = self.make_train_env()
        else:
            use_mp = batch_size is not None and batch_size > 1 # and not self.config.debug
            env = make_batched_env(
                self.env_name,
                wrappers=self.train_wrappers(),
                batch_size=batch_size,
                asynchronous=use_mp,
                # asynchronous=False,
                shared_memory=False,
            )
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

    def make_train_env(self) -> gym.Env:
        """ Make a single (not-batched) training environment. """
        if self.env_name:
            env = gym.make(self.env_name)
        else:
            assert callable(self.dataset), f"dataset should either be a string or a callable, got {self.dataset}"
            env = self.dataset()

        for wrapper in self.train_wrappers():
            env = wrapper(env)
        return env

    def train_wrappers(self) -> List[Callable[[gym.Env], gym.Env]]:
        """Get the list of wrappers to add to each training environment.
        
        The result of this method must be pickleable when using
        multiprocessing.

        Returns
        -------
        List[Callable[[gym.Env], gym.Env]]
            [description]

        Raises
        ------
        NotImplementedError
            [description]
        """
        wrappers: List[Callable[[gym.Env], gym.Env]] = []
        # TODO: Add some kind of Wrapper around the dataset to make it
        # semi-supervised?
        # When using something like CartPole or Pendulum, we'd need to add
        # a PixelObservations wrapper so we can get the 'observation'
        # When using something like CartPole, we'd need to add a
        # PixelObservations wrapper.
        if self.env_name.startswith(classic_control_env_prefixes):
            wrappers.append(PixelObservationWrapper)

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
    
    def val_dataloader(self, batch_size: int = None) -> Environment:
        if not self.has_prepared_data:
            self.prepare_data()
        if not self.has_setup_fit:
            self.setup("fit")
        
        batch_size = batch_size or self.valid_batch_size or self.batch_size

        if batch_size is None:
            env = self.make_val_env()
        else:
            use_mp = batch_size is not None and batch_size > 1 # and not self.config.debug
            env = make_batched_env(
                self.env_name,
                wrappers=self.val_wrappers(),
                batch_size=batch_size,
                asynchronous=use_mp,
                # asynchronous=False,
                shared_memory=False,
            )
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

    def make_val_env(self) -> gym.Env:
        """ Create a single (non-batched) validation environment. """
        if self.env_name:
            env = gym.make(self.env_name)
        else:
            assert callable(self.dataset), f"dataset should either be a string or a callable, got {self.dataset}"
            env = self.dataset()
        for wrapper in self.val_wrappers():
            env = wrapper(env)
        return env
    
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

        if self.env_name.startswith(classic_control_env_prefixes):
            wrappers.append(PixelObservationWrapper)

        wrappers.append(partial(TransformObservation, f=self.val_transforms))

        if self.smooth_task_boundaries:
            wrappers.append(partial(SmoothTransitions,
                task_schedule=self.valid_task_schedule,
                add_task_id_to_obs=True,
                add_task_dict_to_info=True,
            ))
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

    def test_dataloader(self, batch_size: int = None) -> TestEnvironment:
        if not self.has_prepared_data:
            self.prepare_data()
        if not self.has_setup_test:
            self.setup("test")

        batch_size = batch_size or self.test_batch_size or self.batch_size
        # TODO: The test environment shouldn't be just for the current task, it
        # should be the sequence of all tasks.
        use_mp = batch_size is not None and batch_size > 1 # and not self.config.debug
        # BUG: Turn shared_memory back to True once the bugs with shared memory
        # and Sparse spaces are fixed.
        if batch_size is None:
            env = self.make_test_env()
        else:
            env = make_batched_env(
                self.env_name,
                wrappers=self.test_wrappers(),
                batch_size=batch_size,
                asynchronous=use_mp,
                # asynchronous=False,
                shared_memory=False,
            )
        
        # If we know the task boundaries at test time, and the method has the
        # callback for it, then we add a callback wrapper that will invoke the
        # method's on_task_switch and pass it the task label if required when on
        # a task boundary.
        # This is different than the train or validation environments, since the
        # test environment might cover multiple tasks, while if the task labels
        # are available at train time, then each train/valid environment is for
        # a single task.
        if self.known_task_boundaries_at_test_time and self.on_task_switch_callback:
            def _on_task_switch(step: int, *arg) -> None:
                if step not in self.test_task_schedule:
                    return
                if self.task_labels_at_test_time:
                    task_steps = sorted(self.test_task_schedule.keys())
                    task_id = task_steps.index(step)
                    self.on_task_switch_callback(task_id)
                else:
                    self.on_task_switch_callback(None)
            env = StepCallbackWrapper(env, callbacks=[_on_task_switch])

        env = TypedObjectsWrapper(
            env,
            observations_type=self.Observations,
            rewards_type=self.Rewards,
            actions_type=self.Actions,
        )
        dataset = EnvDataset(env, max_steps=self.steps_per_task)
        dataloader = GymDataLoader(dataset)
        test_dir = "results"
        # TODO: We should probably change the max_steps depending on the
        # batch size of the env.
        test_loop_max_steps = self.max_steps
        self.test_env = ContinualRLTestEnvironment(
            dataloader,
            directory=test_dir,
            step_limit=self.max_steps,
            force=True,
        )
        return self.test_env

    def make_test_env(self) -> gym.Env:
        """ Make a single (not-batched) testing environment. """
        if self.env_name:
            env = gym.make(self.env_name)
        else:
            assert callable(self.dataset), f"dataset should either be a string or a callable, got {self.dataset}"
            env = self.dataset()

        for wrapper in self.test_wrappers():
            env = wrapper(env)
        return env

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

        if self.env_name.startswith(classic_control_env_prefixes):
            wrappers.append(PixelObservationWrapper)

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

    def get_metrics(self, actions, rewards):
        assert False, rewards
        return super().get_metrics(actions, rewards)


class ContinualRLTestEnvironment(TestEnvironment, IterableWrapper):
    def get_results(self) -> RLResults:
        rewards = self.get_episode_rewards()
        lengths = self.get_episode_lengths()
        total_steps = self.get_total_steps()
        
        assert has_wrapper(self.env, MultiTaskEnvironment), self.env
        task_steps = sorted(self.task_schedule.keys())
        
        assert 0 in task_steps, task_steps
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
            episode_metric = RegressionMetrics(mse=episode_reward / episode_length)
            episode_metrics[task_id].append(episode_metric)

        return RLResults(
            episode_lengths=episode_lengths,
            episode_rewards=episode_rewards,
            test_metrics=episode_metrics,
        )


if __name__ == "__main__":
    ContinualRLSetting.main()
