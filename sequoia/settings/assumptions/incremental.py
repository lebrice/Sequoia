import itertools
import json
import time
from abc import ABC, abstractmethod
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from itertools import accumulate, chain
from pathlib import Path
from typing import (ClassVar, Dict, List, Optional, Sequence, Tuple, Type,
                    TypeVar, Union)

import gym
import matplotlib.pyplot as plt
import torch
import tqdm
import wandb
from gym import spaces
from gym.vector import VectorEnv
from simple_parsing import field
from torch import Tensor

from sequoia.common import ClassificationMetrics, Metrics, RegressionMetrics
from sequoia.common.config import Config
from sequoia.common.gym_wrappers.step_callback_wrapper import (
    Callback, StepCallbackWrapper)
from sequoia.common.gym_wrappers.utils import IterableWrapper
from sequoia.settings.base import (Actions, Environment, Method, Results,
                                   Rewards, Setting, SettingABC)
from sequoia.utils import constant, flag, mean
from sequoia.utils.logging_utils import get_logger
from sequoia.utils.utils import add_prefix

from .continual import ContinualSetting

logger = get_logger(__file__)

from .incremental_results import (IncrementalResults, TaskResults,
                                  TaskSequenceResults)


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

    Results: ClassVar[Type[Results]] = IncrementalResults

    @dataclass(frozen=True)
    class Observations(Setting.Observations):
        """ Observations produced by an Incremental setting.

        Adds the 'task labels' to the base Observation.
        """
        task_labels: Union[Optional[Tensor], Sequence[Optional[Tensor]]] = None
    
    # TODO: Actually add the 'smooth' task boundary case.
    # Wether we have clear boundaries between tasks, or if the transition is
    # smooth.
    smooth_task_boundaries: bool = constant(False) # constant for now.
    
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

    # The number of tasks. By default 0, which means that it will be set
    # depending on other fields in __post_init__, or eventually be just 1.
    nb_tasks: int = field(0, alias=["n_tasks", "num_tasks"])

    # Attributes (not parsed through the command-line):
    _current_task_id: int = field(default=0, init=False)

    # TODO: Add wandb support for all methods, not just the baseline.
    
    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)

        self.train_env: Environment = None  # type: ignore
        self.val_env: Environment = None  # type: ignore
        self.test_env: TestEnvironment = None  # type: ignore
        
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._setting_logged_to_wandb: bool = False

    @property
    def current_task_id(self) -> Optional[int]:
        """ Get the current task id.
        
        TODO: Do we want to return None if the task labels aren't currently
        available? (at either Train or Test time?) Or if we 'detect' if
        this is being called from the method?
        
        TODO: This property doesn't really make sense in the Multi-Task SL or RL
        settings.
        """
        return self._current_task_id

    @current_task_id.setter
    def current_task_id(self, value: int) -> None:
        """ Sets the current task id. """
        self._current_task_id = value

    def task_boundary_reached(self, method: Method, task_id: int, training: bool):
        known_task_boundaries = (
            self.known_task_boundaries_at_train_time
            if training
            else self.known_task_boundaries_at_test_time
        )
        task_labels_available = (
            self.task_labels_at_train_time
            if training
            else self.task_labels_at_test_time
        )

        if known_task_boundaries:
            # Inform the model of a task boundary. If the task labels are
            # available, then also give the id of the new task to the
            # method.
            # TODO: Should we also inform the method of wether or not the
            # task switch is occuring during training or testing?
            if not hasattr(method, "on_task_switch"):
                logger.warning(
                    UserWarning(
                        f"On a task boundary, but since your method doesn't "
                        f"have an `on_task_switch` method, it won't know about "
                        f"it! "
                    )
                )
            elif not task_labels_available:
                method.on_task_switch(None)
            elif self.nb_tasks == 1:
                # NOTE: on_task_switch won't be called if there is only one "task",
                # (as-in one task in a 'sequence' of tasks).
                # TODO: in multi-task RL, i.e. RLSetting(dataset=..., nb_tasks=10),
                # for instance, then there are indeed 10 tasks, but `self.tasks`
                # is used here to describe the number of 'phases' in training and
                # testing.
                pass
            else:
                method.on_task_switch(task_id)

    def main_loop(self, method: Method) -> IncrementalResults:
        """ Runs an incremental training loop, wether in RL or CL. """
        # TODO: Add ways of restoring state to continue a given run?
        # For each training task, for each test task, a list of the Metrics obtained
        # during testing on that task.
        # NOTE: We could also just store a single metric for each test task, but then
        # we'd lose the ability to create a plots to show the performance within a test
        # task.
        # IDEA: We could use a list of IIDResults! (but that might cause some circular
        # import issues)
        results = self.Results()

        self._start_time = time.process_time()
        for task_id in range(self.nb_tasks):
            logger.info(
                f"Starting training"
                + (f" on task {task_id}." if self.nb_tasks > 1 else ".")
            )
            self.current_task_id = task_id
            self.task_boundary_reached(method, task_id=task_id, training=True)

            # Creating the dataloaders ourselves (rather than passing 'self' as
            # the datamodule):
            task_train_loader = self.train_dataloader()
            task_valid_loader = self.val_dataloader()
            success = method.fit(
                train_env=task_train_loader, valid_env=task_valid_loader,
            )
            task_train_loader.close()
            task_valid_loader.close()

            logger.info(f"Finished Training on task {task_id}.")

            test_metrics: TaskSequenceResults = self.test_loop(method)
            # Add a row to the transfer matrix.
            results.append(test_metrics)
            logger.info(f"Resulting objective of Test Loop: {test_metrics.objective}")

            if wandb.run:
                if not self._setting_logged_to_wandb:
                    logger.info(f"Detected that wandb is being used, will fill the `config` with values from the setting.")
                    wandb.config["method"] = method.get_name()
                    wandb.config["setting"] = self.get_name()
                    for k, v in self.to_dict().items():
                        if not k.startswith("_"):
                            wandb.config[f"setting/{k}"] = v
                    self._setting_logged_to_wandb = True

                d = add_prefix(test_metrics.to_log_dict(), prefix="Test", sep="/")
                # d = add_prefix(test_metrics.to_log_dict(), prefix="Test", sep="/")
                d["current_task"] = task_id
                wandb.log(d)

        self._end_time = time.process_time()
        results._runtime = self._start_time - self._end_time
        logger.info(f"Finished main loop in {self._end_time - self._start_time} seconds.")
        self.log_results(method, results)
        return results

    def log_results(self, method: Method, results: IncrementalResults)-> None:
        """
        TODO: Create the tabs we need to show up in wandb:
        1. Final
            - Average "Current/Online" performance (scalar)
            - Average "Final" performance (scalar)
            - Runtime
        2. Test
            - Task i (evolution over time (x axis is the task id, if possible))
        """
        method_name: str = self.get_name()
        setting_name: str = self.get_name()
        dataset = self.dataset
        logger.info(results.summary())
        final_dict = {
            "Final/Average Online Performance": results.average_online_performance.objective,
            "Final/Average Final Performance": results.average_final_performance.objective,
            "Final/Runtime (seconds)": self._end_time - self._start_time,
            "Final/CL Score": results.cl_score,
        }
        print(json.dumps(final_dict, indent="\t"))
        
        if wandb.run:
            wandb.summary["method"] = method_name
            wandb.summary["setting"] = setting_name
            if dataset and isinstance(dataset, str):
                wandb.summary["dataset"] = dataset

            # wandb.log(results.to_log_dict())
            
            # BUG: Sometimes logging a matplotlib figure causes a crash:
            # File "/home/fabrice/miniconda3/envs/sequoia/lib/python3.8/site-packages/plotly/matplotlylib/mplexporter/utils.py", line 246, in get_grid_style
            # if axis._gridOnMajor and len(gridlines) > 0:
            # AttributeError: 'XAxis' object has no attribute '_gridOnMajor'
            # wandb.log(results.make_plots())
            wandb.log(final_dict)
            wandb.log(results.make_plots())

            wandb.run.finish()

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
        """
        test_env = self.test_dataloader()
        
        test_env: TestEnvironment

        if self.known_task_boundaries_at_test_time and self.nb_tasks > 1:

            def _on_task_switch(step: int, *arg) -> None:
                # TODO: This attribute isn't on IncrementalSetting itself, it's defined
                # on ContinualRLSetting.
                if step not in test_env.boundary_steps:
                    return
                if not hasattr(method, "on_task_switch"):
                    logger.warning(
                        UserWarning(
                            f"On a task boundary, but since your method doesn't "
                            f"have an `on_task_switch` method, it won't know about "
                            f"it! "
                        )
                    )
                    return

                if self.task_labels_at_test_time:
                    # TODO: Should this 'test boundary' step depend on the batch size?
                    task_steps = sorted(test_env.boundary_steps)
                    # TODO: If the ordering of tasks were different (shuffled
                    # tasks for example), then this wouldn't work, we'd need a
                    # list of the task ids or something like that.
                    task_id = task_steps.index(step)
                    logger.debug(
                        f"Calling `method.on_task_switch({task_id})` "
                        f"since task labels are available at test-time."
                    )
                    method.on_task_switch(task_id)
                else:
                    logger.debug(
                        f"Calling `method.on_task_switch(None)` "
                        f"since task labels aren't available at "
                        f"test-time, but task boundaries are known."
                    )
                    method.on_task_switch(None)

            test_env = StepCallbackWrapper(test_env, callbacks=[_on_task_switch])

        try:
            # If the Method has `test` defined, use it.
            method.test(test_env)
            test_env.close()
            test_env: TestEnvironment
            # Get the metrics from the test environment
            test_results: Results = test_env.get_results()
            print(f"Test results: {test_results.to_log_dict()}")
            return test_results

        except NotImplementedError:
            logger.info(
                f"Will query the method for actions at each step, "
                f"since it doesn't implement a `test` method."
            )

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
        test_results: TaskSequenceResults = test_env.get_results()
        
        return test_results
        # return test_results
        # if not self.task_labels_at_test_time:
        #     # TODO: move this wrapper to common/wrappers.
        #     test_env = RemoveTaskLabelsWrapper(test_env)

    @abstractmethod
    def train_dataloader(
        self, *args, **kwargs
    ) -> Environment["IncrementalSetting.Observations", Actions, Rewards]:
        """ Returns the DataLoader/Environment for the current train task. """
        return super().train_dataloader(*args, **kwargs)

    @abstractmethod
    def val_dataloader(
        self, *args, **kwargs
    ) -> Environment["IncrementalSetting.Observations", Actions, Rewards]:
        """ Returns the DataLoader/Environment used for validation on the
        current task.
        """
        return super().val_dataloader(*args, **kwargs)

    @abstractmethod
    def test_dataloader(
        self, *args, **kwargs
    ) -> Environment["IncrementalSetting.Observations", Actions, Rewards]:
        """ Returns the Test Environment (for all the tasks). """
        return super().test_dataloader(*args, **kwargs)

class TestEnvironment(gym.wrappers.Monitor, IterableWrapper, ABC):
    """ Wrapper around a 'test' environment, which limits the number of steps
    and keeps tracks of the performance.
    """

    def __init__(
        self,
        env: gym.Env,
        directory: Path,
        step_limit: int = 1_000,
        no_rewards: bool = False,
        config: Config = None,
        *args,
        **kwargs,
    ):
        super().__init__(env, directory, *args, **kwargs)
        self.step_limit = step_limit
        self.no_rewards = no_rewards
        self._closed = False
        self._steps = 0
        self.config = config
        # if wandb.run:
        #     wandb.gym.monitor()

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


TestEnvironment.__test__ = False
