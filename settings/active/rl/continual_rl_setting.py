from dataclasses import dataclass
from functools import partial
from typing import ClassVar, Dict, List, Type, Callable, Union

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
from common.gym_wrappers.utils import classic_control_env_prefixes, IterableWrapper
from common.gym_wrappers.sparse_space import Sparse
from common.transforms import Transforms
from settings.active import ActiveSetting
from settings.assumptions.incremental import IncrementalSetting
from settings.base.results import Results
from settings.base import Method
from utils import dict_union, get_logger

from .gym_dataloader import GymDataLoader
from .make_env import make_batched_env
from .rl_results import RLResults
from .. import ActiveEnvironment

logger = get_logger(__file__)


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

    available_datasets: ClassVar[Dict[str, str]] = {
        "breakout": "Breakout-v0",
        "duckietown": "Duckietown-straight_road-v0"
    }
    # Which environment to learn on.
    dataset: str = choice(available_datasets, default="breakout")


    # Max number of steps ("length" of the training and test "datasets").
    max_steps: int = 10_000
    # Number of steps per task.
    steps_per_task: int = 5_000
    # Wether the task boundaries are smooth or sudden.
    smooth_task_boundaries: bool = True

    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)
          
        # Set the number of tasks depending on the increment, and vice-versa.
        # (as only one of the two should be used).
        if self.nb_tasks == 0:
            self.nb_tasks = self.max_steps // self.steps_per_task
        else:
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

        self.train_task_schedule: Dict[int, Dict] = {}
        self.valid_task_schedule: Dict[int, Dict] = {}
        self.test_task_schedule: Dict[int, Dict] = {}

        # Create a temporary environment so we can extract the spaces.
        with gym.make(self.env_name) as temp_env:
            # Apply the image transforms to the env.
            temp_env = TransformObservation(temp_env, f=self.train_transforms)
            # Add a wrapper that creates the 'tasks' (non-stationarity in the env).
            # First, get the set of parameters that will be changed over time. 
            cl_task_params = task_params.get(type(temp_env.unwrapped), [])

            if self.smooth_task_boundaries:
                temp_env = SmoothTransitions(
                    temp_env,
                    task_params=cl_task_params,
                    # We want to have a 'task-label' space, but it will be
                    # filled with None values. 
                    add_task_id_to_obs=True,
                )
            else:
                add_task_id_to_obs = (self.task_labels_at_train_time or self.task_labels_at_test_time)
                temp_env = MultiTaskEnvironment(
                    temp_env,
                    task_params=cl_task_params,
                    add_task_id_to_obs=add_task_id_to_obs,
                )

            # Start with the default task (step 0) and then add a new task
            # at intervals of `self.steps_per_task`
            for task_step in range(self.steps_per_task, self.max_steps, self.steps_per_task):
                self.train_task_schedule[task_step] = temp_env.random_task()
            assert len(self.train_task_schedule) == self.nb_tasks - 1
            
            # For now, set the validation and test tasks as the same sequence as the
            # train tasks.
            self.valid_task_schedule = self.train_task_schedule.copy() 
            self.test_task_schedule = self.train_task_schedule.copy()

            # Set the spaces using the temp env.
            self.observation_space = temp_env.observation_space
            self.action_space = temp_env.action_space
            self.reward_range = temp_env.reward_range
            self.reward_space = getattr(temp_env, "reward_space",
                                        spaces.Box(low=self.reward_range[0],
                                                high=self.reward_range[1],
                                                shape=()))
        del temp_env

    def apply(self, method: Method, config: Config=None) -> "ContinualRLSetting.Results":
        """Apply the given method on this setting to producing some results."""
        self.config = config or Config.from_args(self._argv)
        method.config = self.config

        self.configure(method)
        method.configure(setting=self)
        
        # Run the Training loop (which is defined in IncrementalSetting).
        self.train_loop(method)
        # Run the Test loop (which is defined in IncrementalSetting).
        results: RlResults = self.test_loop(method)
        
        logger.info(f"Resulting objective of Test Loop: {results.objective}")
        logger.info(results.summary())
        method.receive_results(self, results=results)
        return results

    @property
    def env_name(self) -> str:
        if self.dataset in self.available_datasets.values():
            return self.dataset
        if self.dataset in self.available_datasets.keys():
            return self.available_datasets[self.dataset]
        return self.dataset                        

    def setup(self, stage=None):
        return super().setup(stage=stage)

    def prepare_data(self, *args, **kwargs):
        return super().prepare_data(*args, **kwargs)

    def train_dataloader(self, *args, **kwargs):
        if not self.has_prepared_data:
            self.prepare_data()
        if not self.has_setup_fit:
            self.setup("fit")

        kwargs = dict_union(self.dataloader_kwargs, kwargs)
        batch_size: int = kwargs["batch_size"]

        # NOTE: Setting this to False when debugging can be very helpful,
        # especially when trying to solve problems related to the env wrappers
        # or the input preprocessing. Beware though, if the batch size isn't
        # small (< 32)  this will makes things incredibly slow.
        use_mp = batch_size is not None and batch_size > 1 # and not self.config.debug

        env = make_batched_env(
            self.env_name,
            wrappers=self.train_wrappers(),
            batch_size=batch_size,
            asynchronous=use_mp,
            shared_memory=False,
        )
        # Apply the "post-batch" wrappers:
        if not self.task_labels_at_train_time:
            env = RemoveTaskLabelsWrapper(env)
        # Add wrappers that converts numpy arrays / etc to Observations/Rewards
        # and from Actions objects to numpy arrays.
        env = TypedObjectsWrapper(
            env,
            observations_type=self.Observations,
            rewards_type=self.Rewards,
            actions_type=self.Actions,
        )

        # Create an IterableDataset from the env using the EnvDataset wrapper.
        dataset = EnvDataset(env)
        # Create a GymDataLoader for the EnvDataset.
        dataloader = GymDataLoader(dataset)

        self.train_env = dataloader
        return self.train_env

    def val_dataloader(self, batch_size: int = None) -> ActiveEnvironment:
        if not self.has_prepared_data:
            self.prepare_data()
        if not self.has_setup_fit:
            self.setup("fit")
        batch_size = batch_size or self.dataloader_kwargs["batch_size"]
        use_mp = batch_size is not None and batch_size > 1 # and not self.config.debug

        env = make_batched_env(
            self.env_name,
            wrappers=self.val_wrappers(),
            batch_size=batch_size,
            asynchronous=use_mp,
            shared_memory=False,
        )   
        # Apply the "post-batch" wrappers:
        # TODO: See below, should the validation environment have task labels if
        # the train env does but not the test env? 
        if not self.task_labels_at_test_time:
            # TODO: The RemoveTaskLabelsWrapper makes the 'task label' space
            # Sparse with none_prob=1., but it still allows the user to see the
            # number of tasks.
            env = RemoveTaskLabelsWrapper(env)
        env = TypedObjectsWrapper(
            env,
            observations_type=self.Observations,
            actions_type=self.Actions,
            rewards_type=self.Rewards,
        )
       
        dataset = EnvDataset(env)
        dataloader = GymDataLoader(dataset)

        self.val_env = dataloader
        return self.val_env

    def test_dataloader(self, *args, **kwargs):
        if not self.has_prepared_data:
            self.prepare_data()
        if not self.has_setup_test:
            self.setup("test")
        kwargs = dict_union(self.dataloader_kwargs, kwargs)
        batch_size = kwargs["batch_size"]
        
        use_mp = batch_size is not None and batch_size > 1 # and not self.config.debug
        # BUG: Turn shared_memory back to True once the bugs with shared memory
        # and Sparse spaces are fixed.
        env = make_batched_env(
            self.env_name,
            wrappers=self.test_wrappers(),
            batch_size=batch_size,
            asynchronous=use_mp,
            shared_memory=False,
        )
        # Apply the "post-batch" wrappers:
        if not self.task_labels_at_test_time:
            # TODO: The RemoveTaskLabelsWrapper makes the 'task label' space
            # Sparse with none_prob=1., but it still allows the user to see the
            # number of tasks.
            env = RemoveTaskLabelsWrapper(env)
            
        env = TypedObjectsWrapper(
            env,
            observations_type=self.Observations,
            rewards_type=self.Rewards,
            actions_type=self.Actions,
        )
        
        dataset = EnvDataset(env)
        dataloader = GymDataLoader(dataset)

        self.test_env = dataloader
        return self.test_env

    def make_train_env(self) -> gym.Env:
        """ Make a single (not-batched) training environment. """
        env = gym.make(self.env_name)
        for wrapper in self.train_wrappers():
            env = wrapper(env)
        if not self.task_labels_at_train_time:
            # TODO: Hide or remove the task labels?
            env = RemoveTaskLabelsWrapper(env)
            # env = HideTaskLabelsWrapper(env)
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
        return wrappers
    
    def make_val_env(self) -> gym.Env:
        """ Create a single (non-batched) validation environment. """
        env = gym.make(self.env_name)
        for wrapper in self.val_wrappers():
            env = wrapper(env)
        if not self.task_labels_at_test_time:
            env = RemoveTaskLabelsWrapper(env)
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
        return wrappers

    def make_test_env(self) -> gym.Env:
        """ Make a single (not-batched) testing environment. """
        env = gym.make(self.env_name)
        for wrapper in self.test_wrappers():
            env = wrapper(env)
        if not self.task_labels_at_test_time:
            env = RemoveTaskLabelsWrapper(env)
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
                add_task_dict_to_info=True,
            ))

        elif self.known_task_boundaries_at_test_time:
            starting_step = self.current_task_id * self.steps_per_task
            max_steps = (self.current_task_id + 1) * self.steps_per_task - 1
            wrappers.append(partial(MultiTaskEnvironment,
                task_schedule=self.test_task_schedule,
                add_task_id_to_obs=True,
                add_task_dict_to_info=True,
                starting_step=starting_step,
                max_steps=max_steps,
            ))
        return wrappers


class TypedObjectsWrapper(IterableWrapper):
    """ Wrapper that converts the observations and rewards coming from the env
    to the types of Observations and Rewards, respectively, and 
    """ 
    def __init__(self,
                 env: gym.Env,
                 observations_type: Type[ContinualRLSetting.Observations],
                 rewards_type: Type[ContinualRLSetting.Rewards],
                 actions_type: Type[ContinualRLSetting.Actions]):
        self.Observations = observations_type
        self.Rewards = rewards_type
        self.Actions = actions_type
        env = TransformObservation(env, f=self.Observations.from_inputs)
        env = TransformReward(env, f=self.Rewards.from_inputs)
        
        def convert_action_object_to_sample_from_action_space(action: ContinualRLSetting.Actions):
            if isinstance(action, Batch):
                assert len(action) == 1
                action = action[0]
            if isinstance(action, Tensor):
                action = action.cpu().numpy()
            return action

        convert_action_object_to_sample_from_action_space.space_change = lambda x: x
        
        env = TransformAction(env, f=convert_action_object_to_sample_from_action_space)
        super().__init__(env=env)

from common.gym_wrappers import TransformObservation
from typing import TypeVar, Tuple, Optional
T = TypeVar("T")


def remove_task_labels(observation: Tuple[T, int]) -> T:
    assert len(observation) == 2
    return observation[0]


class RemoveTaskLabelsWrapper(TransformObservation):
    def __init__(self, env: gym.Env, f=remove_task_labels):
        super().__init__(env, f=f)
        self.observation_space = self.space_change(self.env.observation_space)

    @classmethod
    def space_change(cls, input_space: gym.Space) -> gym.Space:
        assert isinstance(input_space, spaces.Tuple), input_space
        assert len(input_space) == 2
        return input_space[0]


def hide_task_labels(observation: Tuple[T, int]) -> Tuple[T, Optional[int]]:
    assert len(observation) == 2
    if isinstance(observation, Batch):
        return type(observation).from_inputs((observation[0], None))
    return observation[0], None


class HideTaskLabelsWrapper(TransformObservation):
    def __init__(self, env: gym.Env, f=hide_task_labels):
        super().__init__(env, f=f)
        self.observation_space = self.space_change(self.env.observation_space)
        
    
    @classmethod
    def space_change(cls, input_space: gym.Space) -> gym.Space:
        assert isinstance(input_space, spaces.Tuple)
        # TODO: If we create something like an OptionalSpace, we
        # would replace the second part of the tuple with it. We
        # leave it the same here for now.
        assert len(input_space) == 2
        
        task_label_space = input_space.spaces[1]
        if isinstance(task_label_space, Sparse):
            # Replace the existing 'Sparse' space with another one with the same
            # base but with none_prob = 1.0
            task_label_space = task_label_space.base
        assert not isinstance(task_label_space, Sparse)
        # Do we set the task label space as sparse? or do we just remote that
        # space?
        return spaces.Tuple([
            input_space[0],
            Sparse(task_label_space, none_prob=1.)
        ])



if __name__ == "__main__":
    ContinualRLSetting.main()
