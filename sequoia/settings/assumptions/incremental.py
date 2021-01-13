import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union, ClassVar, Type

import gym
import torch
import tqdm
import wandb
from gym import spaces
from gym.vector import VectorEnv
from simple_parsing import field
from torch import Tensor

from sequoia.settings.base import Setting
from sequoia.common import ClassificationMetrics, Metrics, RegressionMetrics
from sequoia.common.gym_wrappers.step_callback_wrapper import (
    Callback, StepCallbackWrapper)
from sequoia.common.gym_wrappers.utils import IterableWrapper
from sequoia.common.config import Config
from sequoia.settings.base import (Actions, Environment, Method, Results,
                                   Rewards, SettingABC)
from sequoia.utils import constant, flag, mean
from sequoia.utils.logging_utils import get_logger

logger = get_logger(__file__)
from .continual import ContinualSetting

@dataclass
class IncrementalSetting(ContinualSetting):
    """ Mixin that defines methods that are common to all 'incremental'
    settings, where the data is separated into tasks, and where you may not
    always get the task labels.
    
    Concretely, this holds the train and test loops that are common to the
    ClassIncrementalSetting (highest node on the Passive side) and ContinualRL
    (highest node on the Active side), therefore this setting, while abstract,
    is quite important. 
    
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
    class Observations(Setting.Observations):
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
    
    # Attributes (not parsed through the command-line):
    _current_task_id: int = field(default=0, init=False)

    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)
        
        self.train_env: Environment = None  # type: ignore
        self.val_env: Environment = None  # type: ignore
        self.test_env: TestEnvironment = None  # type: ignore

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
            # TODO: Pass the train_dataloader and val_dataloader methods, rather than the envs?
            task_train_loader = self.train_dataloader()
            task_valid_loader = self.val_dataloader()
            success = method.fit(
                train_env=task_train_loader,
                valid_env=task_valid_loader,
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
        
        NOTE: If `self.known_task_boundaries_at_test_time` is `True` and the
        method has the `on_task_switch` callback defined, then a callback
        wrapper is added that will invoke the method's `on_task_switch` and pass
        it the task id (or `None` if `not self.task_labels_available_at_test_time`) 
        when a task boundary is encountered.

        This `on_task_switch` 'callback' wrapper gets added the same way for
        Supervised or Reinforcement learning settings.
        
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

        test_env = self.test_dataloader()
        test_env: TestEnvironment
        
        if self.known_task_boundaries_at_test_time:
            def _on_task_switch(step: int, *arg) -> None:
                if step not in self.test_task_schedule:
                    return
                if not hasattr(method, "on_task_switch"):
                    logger.warning(UserWarning(
                        f"On a task boundary, but since your method doesn't "
                        f"have an `on_task_switch` method, it won't know about "
                        f"it! "
                    ))
                    return
                if self.task_labels_at_test_time:
                    task_steps = sorted(self.test_task_schedule.keys())
                    # TODO: If the ordering of tasks were different (shuffled
                    # tasks for example), then this wouldn't work, we'd need a
                    # list of the task ids or something like that.
                    task_id = task_steps.index(step)
                    logger.debug(f"Calling `method.on_task_switch({task_id})` "
                                 f"since task labels are available at test-time.")
                    method.on_task_switch(task_id)
                else:
                    logger.debug(f"Calling `method.on_task_switch(None)` "
                                 f"since task labels aren't available at "
                                 f"test-time, but task boundaries are known.")
                    method.on_task_switch(None)
            test_env = StepCallbackWrapper(test_env, callbacks=[_on_task_switch])

        try:
            # If the Method has `test` defined, use it. 
            method.test(test_env)
            test_env: TestEnvironment
            # Get the metrics from the test environment
            test_results: Results = test_env.get_results()
            print(f"Test results: {test_results}")
            return test_results
 
        except NotImplementedError:
            logger.info(f"Will query the method for actions at each step, "
                        f"since it doesn't implement a `test` method.")

        obs = test_env.reset()

        # TODO: Do we always have a maximum number of steps? or of episodes?
        # Will it work the same for Supervised and Reinforcement learning?
        max_steps: int = getattr(test_env, "step_limit", None)

        # Reset on the last step is causing trouble, since the env is closed.
        pbar = tqdm.tqdm(itertools.count(), total=max_steps, desc="Test")
        episode = 0
        for step in pbar:
            if test_env.is_closed():
                logger.debug(f"Env is closed")
                break
            # logger.debug(f"At step {step}")
            action = method.get_actions(obs, test_env.action_space)

            # logger.debug(f"action: {action}")
            # TODO: Remove this:
            if isinstance(action, Actions):
                action = action.y_pred
            if isinstance(action, Tensor):
                action = action.cpu().numpy()

            obs, reward, done, info = test_env.step(action)
            
            if done and not test_env.is_closed():
                # logger.debug(f"end of test episode {episode}")
                obs = test_env.reset()
                episode += 1
        
        test_env.close()
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
        """ Returns the Test Environment (for all the tasks). """  
        return super().test_dataloader(*args, **kwargs)


class TestEnvironment(gym.wrappers.Monitor,  IterableWrapper, ABC):
    """ Wrapper around a 'test' environment, which limits the number of steps
    and keeps tracks of the performance.
    """
    def __init__(self,
                 env: gym.Env,
                 directory: Path,
                 step_limit: int = 1_000,
                 no_rewards: bool = False,
                 config: Config = None,
                 *args, **kwargs):
        super().__init__(env, directory, *args, **kwargs)
        self.step_limit = step_limit
        self.no_rewards = no_rewards
        self._closed = False
        self._steps = 0
        self.config = config
        # if self.config.render:
        #     if wandb.run:
        #         wandb.gym.monitor()
            

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
        # logger.debug(f"Step {self._steps}")
        action_for_stats = action.y_pred if isinstance(action, Actions) else action

        self._before_step(action_for_stats)
        
        if isinstance(action, Tensor):
            action = action.cpu().numpy()
        observation, reward, done, info = self.env.step(action)
        observation_for_stats = observation.x
        reward_for_stats = reward.y

        # TODO: Always render when debugging? or only when the corresponding
        # flag is set in self.config? 
        try:
            if self.config and self.config.render and self.config.debug:
                self.render("human")
        except NotImplementedError:
            pass
        
        if isinstance(self.env.unwrapped, VectorEnv):
            done = all(done)
        else:
            done = bool(done)
        
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
