import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Union, Sequence, Optional
from pathlib import Path

import gym
import torch
import tqdm
from gym import spaces
from simple_parsing import field
from torch import Tensor

from common import Metrics, ClassificationMetrics, RegressionMetrics
from utils import flag, constant, mean
from utils.logging_utils import get_logger

from settings.base import SettingABC, Method, Actions, Environment, Observations, Results, Rewards

logger = get_logger(__file__)


@dataclass
class IncrementalSetting(SettingABC):
    """ Mixin that defines methods that are common to all 'incremental' settings,
    where the data is separated into tasks, and where you may not always get the
    task labels.
    """
    @dataclass
    class Results(SettingABC.Results):
        test_metrics: List[List[Metrics]]
        @property
        def num_tasks(self) -> int:
            return len(self.test_metrics)
        @property
        def average_metrics_per_task(self) -> List[Metrics]:
            return list(map(mean, self.test_metrics))
        @property
        def average_metrics(self) -> Metrics:
            return mean(self.average_metrics_per_task)
        @property
        def objective(self) -> float:
            average_metrics = self.average_metrics
            if isinstance(average_metrics, ClassificationMetrics):
                return average_metrics.accuracy
            if isinstance(average_metrics, RegressionMetrics):
                return average_metrics.mse
            return average_metrics

    @dataclass(frozen=True)
    class Observations(SettingABC.Observations):
        """ Observations produced by an Incremental setting. 

        Adds the 'task labels' to the base Observation.
        """
        task_labels: Union[Optional[Tensor], Sequence[Optional[Tensor]]] = None

    # Wether task labels are available at train time.
    # NOTE: Forced to True at the moment.
    task_labels_at_train_time: bool = flag(default=True)
    # Wether task labels are available at test time.
    task_labels_at_test_time: bool = flag(default=False)
    # Wether we get informed when reaching the boundary between two tasks during
    # training. Only used when `smooth_task_boundaries` is False.

    # TODO: Setting constant for now, but we could add task boundary detection
    # later on!
    known_task_boundaries_at_train_time: bool = constant(True)
    # Wether we get informed when reaching the boundary between two tasks during
    # training. Only used when `smooth_task_boundaries` is False.
    known_task_boundaries_at_test_time: bool = constant(True)

    # TODO: Actually add the 'smooth' task boundary case.
    # Wether we have clear boundaries between tasks, or if the transition is
    # smooth.
    smooth_task_boundaries: bool = constant(False) # constant for now.
    # The number of tasks. By default 0, which means that it will be set
    # depending on other fields in __post_init__, or eventually be just 1. 
    nb_tasks: int = field(0, alias=["n_tasks", "num_tasks"])

    # Number of episodes to perform through the test environment in the test
    # loop. Depending on what an 'episode' might represent in your setting, this
    # could be as low as 1 (for example, supervised learning, episode == epoch).
    test_loop_episodes: int = 1
    
    # Attributes (not parsed through the command-line):
    _current_task_id: int = field(default=0, init=False)

    def __post_init__(self, *args, **kwargs):
        super().__post_init__(self, *args, **kwargs)
        # assert False, "This Shouldn't ever be called!"

    @property
    def current_task_id(self) -> Optional[int]:
        """ Get the current task id.
        
        TODO: Do we want to return None if the task labels aren't currently
        available? (at either Train or Test time?) Or if we 'detect' if
        this is being called from the method?
        """
        return self._current_task_id

    @current_task_id.setter
    def current_task_id(self, value: int) -> None:
        """ Sets the current task id. """
        self._current_task_id = value

    def train_loop(self, method: Method):
        """ (WIP): Runs an incremental training loop, wether in RL or CL."""
        for task_id in range(self.nb_tasks):
            logger.info(f"Starting training on task {task_id}")
            self.current_task_id = task_id

            if self.known_task_boundaries_at_train_time:
                # Inform the model of a task boundary. If the task labels are
                # available, then also give the id of the new task to the
                # method.
                # TODO: Should we also inform the method of wether or not the
                # task switch is occuring during training or testing?
                if not hasattr(method, "on_task_switch"):
                    logger.warning(UserWarning(
                        f"On a task boundary, but since your method doesn't "
                        f"have an `on_task_switch` method, it won't know about "
                        f"it! "
                    ))
                elif not self.task_labels_at_train_time:
                    method.on_task_switch(None)
                else:
                    method.on_task_switch(task_id)

            # Creating the dataloaders ourselves (rather than passing 'self' as
            # the datamodule):
            # success = trainer.fit(model, datamodule=self)
            # TODO: Pass the train_dataloader and val_dataloader methods, rather than the envs.
            task_train_loader = self.train_dataloader()
            task_valid_loader = self.val_dataloader()
            
            success = method.fit(
                train_env=task_train_loader,
                valid_env=task_valid_loader,
                # datamodule=self,
            )
            if success != 0:
                logger.debug(f"Finished Training on task {task_id}.")
            else:
                raise RuntimeError(
                    f"Something didn't work during training: "
                    f"method.fit() returned {success}"
                )

    def test_loop(self, method: Method) -> "IncrementalSetting.Results":
        """ (WIP): Runs an incremental test loop and returns the Results.

        The idea is that this loop should be exactly the same, regardless of if
        you're on the RL or the CL side of the tree.
        
        Args:
            method (Method): The Method to evaluate.

        Returns:
            `IncrementalSetting.Results`:  An object that holds the test metrics
            and that is used to define the `objective` - a float representing
            how 'good' this method was on this given setting).
            This object is also useful for creating the plots, serialization,
            and logging to wandb. See `Results` for more info.

        Important Notes:
        -   The PL way of doing this here would be something like:
            `test_results = method.test(datamodule=self)`, however, there are
            some issues with doing it this way (as I just recently learned):
            - This gives the method/model access to the labels at test time;
            - The Method/LightningModule gets the responsibility of creating the
              metrics we're interested in measuring in its `test_step` method.
            - It might be difficult to customize the test loop. For example,
              How would one go about adding some kind of 'test-time training'
              or OSAKA-like evaluation setup using the usual
              `[train/val/test]_step` methods?

            However, I'd rather not do that, and write out the test loop
            manually here, which also allows us more flexibility, but also has
            some downsides:
            - We don't currently support any of the Callbacks from
              pytorch-lightning during testing.

            For some subclasses (e.g `IIDSetting`), it might be totally fine to
            just use the usual Trainer.fit() and Trainer.test() methods, so feel
            free to overwrite this method with your own implementation if that
            makes your life easier.
        """ 
        # Create a list that will hold the test metrics encountered during each
        # task.
        # TODO: Instead of doing the loop manually here, we'd need to create a
        # single environment, which would have everything we need.
        from common.gym_wrappers.step_callback_wrapper import StepCallbackWrapper, StepCallback, Callback
        from settings.active.rl.wrappers import NoTypedObjectsWrapper, RemoveTaskLabelsWrapper, HideTaskLabelsWrapper
        
        test_env = self.test_dataloader()
        test_env: TestEnvironment
        
        if self.known_task_boundaries_at_test_time:
            # TODO: Add a 'callback' wrapper that calls the 'on_task_switch' of the method.
            def on_task_switch_callback(step: int, env: gym.Env, step_results: Tuple):
                if step in self.test_task_schedule.keys():
                    task_id = sorted(self.test_task_schedule.keys()).index(step)
                    if not hasattr(method, "on_task_switch"):
                        logger.warning(UserWarning(
                            f"On a task boundary, but since your method doesn't "
                            f"have an `on_task_switch` method, it won't know about "
                            f"it! "
                        ))
                    elif not self.task_labels_at_test_time:
                        method.on_task_switch(None)
                    else:
                        method.on_task_switch(task_id)

            test_env = StepCallbackWrapper(test_env, callbacks=[on_task_switch_callback])
            assert False, test_env
            raise NotImplementedError()
        
        try:
            # If the Method has `test` defined, use it. 
            method.test(test_env)
            test_env: TestEnvironment
            # Get the metrics from the test environment
            test_results: Results = test_env.get_results()
            return test_results
 
        except NotImplementedError:
            logger.info(f"Will query the method for actions at each step, "
                        f"since it doesn't implement a `test` method.")

        obs = test_env.reset()
        
        # TODO: Do we always have a maximum number of steps? or of episodes?
        # Will it work the same for Supervised and Reinforcement learning?
        max_steps: int = test_env.step_limit
        
        # Reset on the last step is causing trouble, since the env is closed.
        pbar = tqdm.tqdm(itertools.count(), total=max_steps, desc="Test")
        for step in pbar:
            action = method.get_actions(obs, test_env.action_space)
            obs, reward, done, info = test_env.step(action)
            
            if test_env.is_closed():
                break
            if done:
                obs = test_env.reset()

        test_results = test_env.get_results()
        return test_results
        # if not self.task_labels_at_test_time:
        #     # TODO: move this wrapper to common/wrappers.
        #     test_env = RemoveTaskLabelsWrapper(test_env)
    
    @abstractmethod
    def train_dataloader(self, *args, **kwargs) -> Environment["IncrementalSetting.Observations", Actions, Rewards]:
        """ Returns the DataLoader/Environment for the current train task. """  
        return super().train_dataloader(*args, **kwargs)
    
    @abstractmethod
    def val_dataloader(self, *args, **kwargs) -> Environment["IncrementalSetting.Observations", Actions, Rewards]:
        """ Returns the DataLoader/Environment used for validation on the
        current task.
        """  
        return super().val_dataloader(*args, **kwargs)
    
    @abstractmethod
    def test_dataloader(self, *args, **kwargs) -> Environment["IncrementalSetting.Observations", Actions, Rewards]:
        """ Returns the DataLoader/Environment for the current test task. """  
        return super().test_dataloader(*args, **kwargs)


class TestEnvironment(gym.wrappers.Monitor, ABC):
    """ Wrapper around a 'test' environment, which limits the number of steps
    and keeps tracks of the performance.
    """
    def __init__(self, env: gym.Env, directory: Path, step_limit: int = 1_000, no_rewards: bool = False, *args, **kwargs):
        super().__init__(env, directory, *args, **kwargs)
        self.step_limit = step_limit
        self.no_rewards = no_rewards
        self._closed = False
    
    def is_closed(self):
        return self._closed
    
    @abstractmethod
    def get_results(self) -> Results:
        """ Return how well the Method was applied on this environment.
        
        In RL, this would be based on the mean rewards, while in supervised
        learning it could be the average accuracy, for instance.

        Returns
        -------
        Results
            [description]
        """
        # TODO: In the case of the ClassIncremental Setting, we'd have to modify
        # this so we can set the 'Reward' to be stored (and averaged out, etc)
        # to be the accuracy? a Metrics object? idk.
        # TODO: Total reward over a number of steps? Over a number of episodes?
        # Average reward? What's the metric we care about in RL?
        rewards = self.get_episode_rewards()
        lengths = self.get_episode_lengths()
        total_steps = self.get_total_steps()
        return sum(rewards) / total_steps

    def step(self, action):
        # TODO: Its A bit uncomfortable that we have to 'unwrap' these here.. 
        from settings.active.rl.wrappers import unwrap_rewards, unwrap_actions, unwrap_observations
        action_for_stats = unwrap_actions(action)

        self._before_step(action_for_stats)
        
        observation, reward, done, info = self.env.step(action)
        observation_for_stats = unwrap_observations(observation)
        reward_for_stats = unwrap_rewards(reward)

        done = self._after_step(observation_for_stats, reward_for_stats, done, info)
    
        if self.get_total_steps() >= self.step_limit:
            done = True
            self.close()

        # Remove the rewards if they aren't allowed.
        if self.no_rewards:
            reward = None

        return observation, reward, done, info

    def close(self):
        self._closed = True
        return super().close()