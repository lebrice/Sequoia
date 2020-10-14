import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Union, Sequence, Optional

import torch
import tqdm
from simple_parsing import field
from torch import Tensor

from common import Metrics, ClassificationMetrics, RegressionMetrics
from utils import flag, constant, mean
from utils.logging_utils import get_logger

from ..base import Actions, Environment, Observations, Results, Rewards
from ..setting_abc import Results, SettingABC
from ..method_abc import MethodABC

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
        assert False, "This Shouldn't ever be called!"
        # super().__post_init__(self, *args, **kwargs)

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

    def train_loop(self, method: MethodABC):
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

    
    def test_loop(self, method: MethodABC) -> "IncrementalSetting.Results":
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
        from methods import Method
        method: Method

        from settings.passive import PassiveEnvironment
        from settings.active import ActiveEnvironment
 
        # Create a list that will hold the test metrics encountered during each
        # task.
        test_metrics: List[List[Metrics]] = []
        for task_id in range(self.nb_tasks):
            logger.info(f"Starting testing on task {task_id}")
            self._current_task_id = task_id
            # assert not self.smooth_task_boundaries, "TODO: (#18) Make another 'Continual' setting that supports smooth task boundaries."
            
            # Inform the model of a task boundary. If the task labels are
            # available, then also give the id of the new task.
            # TODO: Should we also inform the method of wether or not the task
            # switch is occuring during training or testing?
            if self.known_task_boundaries_at_test_time:
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

            # Test loop:
            test_task_env = self.test_dataloader()
            task_metrics = []
            for episode in range(self.test_loop_episodes):
                episode_metrics = self.run_episode(
                    method=method,
                    env=test_task_env,
                )
                task_metrics.extend(episode_metrics)
            test_task_env.close()
            
            # Add the metrics for this task to the list of all metrics.
            test_metrics.append(task_metrics)

            average_task_metrics = mean(task_metrics)
            logger.info(f"Test Results on task {task_id}: {average_task_metrics}")

        results = self.Results(test_metrics=test_metrics)
        return results

    def run_episode(self, method: MethodABC, env: Environment) -> List[Metrics]:
        """ Apply the method on the env until it is done (one episode).
        Returns a list of the Metrics for each batch.
        
        In the CL/Supervised Learning context, one epoch might be one epoch.
        NOTE: This doesn't close the environment.
        """
        episode_metrics: List[Metrics] = []

        # Reset the environment, which gives the first batch of observations.
        observations: Observations = env.reset()

        # Create a nice progress bar.
        # NOTE: The env might now always have a length attribte, actually.
        total_batches = len(env)
        pbar = tqdm.tqdm(itertools.count(), total=total_batches-1)

        # Close the pbar before exiting.
        with pbar:
            for i in pbar:
                actions = method.get_actions(observations, env.action_space)
                observations, rewards, done, info = env.step(actions)
                
                batch_metrics = self.get_metrics(actions=actions, rewards=rewards)
                episode_metrics.append(batch_metrics)

                if isinstance(batch_metrics, Metrics):
                    # display metrics in the progress bar.
                    pbar.set_postfix(batch_metrics.to_pbar_message())
                else:
                    pbar.set_postfix({"batch metrics": batch_metrics})

                if done:
                    break
        return episode_metrics
    
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
