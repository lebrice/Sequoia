from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Union, Sequence, Optional
from torch import Tensor

import tqdm
from simple_parsing import field

from common import Metrics, ClassificationMetrics, RegressionMetrics
from utils import flag, constant, mean
from utils.logging_utils import get_logger

from ..base import Actions, Environment, Observations, Results, Rewards
from ..setting_abc import Results, SettingABC
from ..method_abc import MethodABC

logger = get_logger(__file__)


@dataclass
class IncrementalSetting(SettingABC):
    """ Mixin that defines methods that are common to all 'incrmenal' settings,
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
    # The number of tasks.
    nb_tasks: int = field(0, alias=["n_tasks", "num_tasks"])

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
                if not self.task_labels_at_train_time:
                    method.on_task_switch(None)
                else:
                    method.on_task_switch(task_id)

            # Creating the dataloaders ourselves (rather than passing 'self' as
            # the datamodule):
            # success = trainer.fit(model, datamodule=self)
            task_train_loader = self.train_dataloader()
            task_val_loader = self.val_dataloader()
            
            success = method.fit(
                train_dataloader=task_train_loader,
                valid_dataloader=task_val_loader,
                # datamodule=self,
            )
            if success:
                logger.debug(f"Finished Training on task {task_id}.")
            if not success:
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

        # Create a list that will hold the test metrics encountered during
        test_metrics: List[List[Metrics]] = [
            [] for _ in range(self.nb_tasks)
        ]

        for task_id in range(self.nb_tasks):
            logger.info(f"Starting testing on task {task_id}")
            self._current_task_id = task_id

            assert not self.smooth_task_boundaries, "TODO: (#18) Make another 'Continual' setting that supports smooth task boundaries."
            # Inform the model of a task boundary. If the task labels are
            # available, then also give the id of the new task.
            # TODO: Should we also inform the method of wether or not the task
            # switch is occuring during training or testing?
            if self.known_task_boundaries_at_test_time:
                if not self.task_labels_at_test_time:
                    method.on_task_switch(None)
                else:
                    method.on_task_switch(task_id)

            # Manual test loop:
            test_task_env = self.test_dataloader()
            # with tqdm.tqdm(test_task_env) as pbar:
            for observations, *rewards in test_task_env:
                # TODO: Remove this, just debugging atm.
                assert isinstance(observations, self.Observations), Observations
                rewards = rewards[0] if rewards else None
                if rewards is not None:
                    assert isinstance(rewards, self.Rewards), rewards
                if not self.task_labels_at_test_time:
                    assert observations.task_labels is None

                # Get the predicted label for this batch of inputs.
                actions = method.get_actions(observations)
                # if the reward is None, we need to get it from the env.
                rewards = test_task_env.send(actions)
                
                # Get the metrics for that batch.
                batch_metrics = self.get_metrics(actions=actions, rewards=rewards)
                
                # Save the metrics for this batch in the list above.
                test_metrics[task_id].append(batch_metrics)
                # pbar.set_postfix(batch_metrics.to_pbar_message())

            average_task_metrics = mean(test_metrics[task_id])
            logger.info(f"Test Results on task {task_id}: {average_task_metrics}")

        results = self.Results(test_metrics=test_metrics)
        return results

    def get_metrics(self,
                    actions: Actions,
                    rewards: Rewards) -> Union[float, Metrics]:
        """ Calculate the "metric" from the model predictions (actions) and the true labels (rewards).
        
        In this example, we return a 'Metrics' object:
        - `ClassificationMetrics` for classification problems,
        - `RegressionMetrics` for regression problems.
        
        We use these objects because they are awesome (they basically simplify
        making plots, wandb logging, and serialization), but you can also just
        return floats if you want, no problem.
        """
        assert False, (actions, rewards)
        from common.metrics import get_metrics
        # In this particular setting, we only use the y_pred from actions and
        # the y from the rewards.
        return get_metrics(y_pred=actions.y_pred, y=rewards.y)
    
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