from dataclasses import dataclass
from functools import partial
from typing import ClassVar, Dict, List, Type, Callable, Union

import gym
import numpy as np
from gym import spaces
from gym.envs.classic_control import CartPoleEnv
from gym.envs.atari import AtariEnv
from simple_parsing import choice
from torch import Tensor

from common import Batch
from common.gym_wrappers import (SmoothTransitions, TransformAction,
                                 TransformObservation, TransformReward)
from common.gym_wrappers.batch_env import BatchedVectorEnv
from utils.logging_utils import get_logger
from settings.active import ActiveSetting
from settings.assumptions.incremental import IncrementalSetting
from settings.base.results import Results
from utils.utils import dict_union

from .rl_results import RLResults
from .make_env import make_batched_env

       
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
class ContinualRLSetting(ActiveSetting, IncrementalSetting):
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
        
        self.train_task_schedule: Dict[int, Dict] = {}
        self.valid_task_schedule: Dict[int, Dict] = {}
        self.test_task_schedule: Dict[int, Dict] = {}

        temp_env = gym.make(self.env_name)
        # Apply the image transforms to the env.
        temp_env = TransformObservation(temp_env, f=self.train_transforms)
        # Add a wrapper that creates the 'tasks' (non-stationarity in the env).  
        cl_task_params = task_params.get(type(temp_env.unwrapped), [])
        temp_env = SmoothTransitions(temp_env, task_params=cl_task_params)

        # Start with the default task, and then add a new task boundary at
        # intervals of self.steps_per_task
        for task_step in range(self.steps_per_task, self.max_steps):
            self.train_task_schedule[task_step] = temp_env.random_task()
        # For now, set the validation and test tasks as the same sequence as the
        # train tasks.
        self.valid_task_schedule = self.train_task_schedule.copy() 
        self.test_task_schedule = self.train_task_schedule.copy()
        
        # Set the spaces using the temp env.
        # TODO: Should we add the task labels in the MultiTaskEnv wrapper?
        self.observation_space = spaces.Tuple([
            temp_env.observation_space, # images
            spaces.Discrete(self.nb_tasks), # task labels.
        ])
        self.action_space = temp_env.action_space
        self.reward_range = temp_env.reward_range
        self.reward_space = getattr(temp_env, "reward_space",
                                    spaces.Box(low=self.reward_range[0],
                                               high=self.reward_range[1],
                                               shape=()))
        temp_env.close()
        del temp_env
        
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
        batch_size = kwargs["batch_size"]

        env = self.make_env(batch_size, wrappers=self.train_wrappers())

        # TODO: Create a dataset from the env using EnvDataset (needs cleanup)
        from common.gym_wrappers.env_dataset import EnvDataset
        dataset = EnvDataset(env)
        
        # TODO: Create a GymDataLoader for the EnvDataset (needs cleanup)
        from .gym_dataloader import GymDataLoader
        dataloader = GymDataLoader(dataset)

        self.train_env = dataloader
        return self.train_env

    def train_wrappers(self) -> List[Callable]:
        # TODO: Add some kind of Wrapper around the dataset to make it
        # semi-supervised?
        wrappers = []
        # TODO: When using something like CartPole or Pendulum, we'd need to add
        # a PixelObservations wrapper.
        if self.env_name.startswith("CartPole"):
            from common.gym_wrappers.pixel_observation import PixelObservationWrapper
            wrappers.append(PixelObservationWrapper)
        wrappers.extend([
            # Apply the image transforms to the env.
            partial(TransformObservation, f=self.train_transforms),
            # Add a wrapper that creates the 'tasks' (non-stationarity in the env).  
            partial(SmoothTransitions, task_schedule=self.train_task_schedule),
        ])
        return wrappers

    def val_dataloader(self, *args, **kwargs):
        if not self.has_prepared_data:
            self.prepare_data()
        if not self.has_setup_fit:
            self.setup("fit")
        kwargs = dict_union(self.dataloader_kwargs, kwargs)
        batch_size = kwargs["batch_size"]
        env = self.make_env(batch_size, wrappers=self.val_wrappers())

        # Create a dataset from the env using EnvDataset (needs cleanup)
        from common.gym_wrappers.env_dataset import EnvDataset
        dataset = EnvDataset(env)
                
        # Create a GymDataLoader for the EnvDataset (needs cleanup)
        from .gym_dataloader import GymDataLoader
        dataloader = GymDataLoader(dataset)

        self.val_env = dataloader
        return self.val_env

    def val_wrappers(self) -> List[Callable]:
        wrappers = []
        # When using something like CartPole, we'd need to add a
        # PixelObservations wrapper.
        if self.env_name.startswith("CartPole"):
            from common.gym_wrappers.pixel_observation import PixelObservationWrapper
            wrappers.append(PixelObservationWrapper)
        wrappers.extend([
            # Apply the image transforms to the env.
            partial(TransformObservation, f=self.val_transforms),
            # Add a wrapper that creates the 'tasks' (non-stationarity in the env).  
            partial(SmoothTransitions, task_schedule=self.val_task_schedule),
        ])
        return wrappers

    def test_dataloader(self, *args, **kwargs):
        if not self.has_prepared_data:
            self.prepare_data()
        if not self.has_setup_test:
            self.setup("test")
        kwargs = dict_union(self.dataloader_kwargs, kwargs)
        batch_size = kwargs["batch_size"]
        env = self.make_env(batch_size, wrappers=self.test_wrappers())
        
        # Create a dataset from the env using EnvDataset (needs cleanup)
        from common.gym_wrappers.env_dataset import EnvDataset
        dataset = EnvDataset(env)
                
        # Create a GymDataLoader for the EnvDataset (needs cleanup)
        from .gym_dataloader import GymDataLoader
        dataloader = GymDataLoader(dataset)

        self.test_env = dataloader
        return self.test_env


    def test_wrappers(self) -> List[Callable]:
        # Apply the image transforms to the env.
        # Add a wrapper that creates the 'tasks' (non-stationarity in the env).  
        return [
            partial(TransformObservation, f=self.test_transforms),
            partial(SmoothTransitions, task_schedule=self.test_task_schedule),
        ]
    
    def make_env(self, batch_size: int, wrappers: List[Callable]=None) -> BatchedVectorEnv:
        env = make_batched_env(
            self.env_name,
            batch_size=batch_size,
            wrappers=wrappers,
        )
        # Add wrappers that convert to Observations/Rewards
        # and from Actions objects to npdarray/tensors.
        env = TransformObservation(env, f=self.Observations)
        env = TransformReward(env, f=self.Rewards)
        env = TransformAction(env, f=self.convert_action_to_ndarray)
        return env
    
    
    def convert_action_to_ndarray(self, action: Union["ContinualRLSetting.Actions", Tensor, np.ndarray]) -> np.ndarray:
        if isinstance(action, Batch):
            action = action[0]
        if isinstance(action, Tensor):
            action = action.cpu().numpy()
        if isinstance(self.action_space, spaces.Tuple):
            if isinstance(action, np.ndarray):
                action = action.tolist()
        return action

if __name__ == "__main__":
    ContinualRLSetting.main()
