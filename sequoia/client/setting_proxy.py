import itertools
import time
import warnings
from abc import ABC, abstractmethod
from functools import partial
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Callable

import gym
import numpy as np
import tqdm
from gym.vector.utils.spaces import batch_space
from torch import Tensor
from sequoia.common.config import Config
from sequoia.common.gym_wrappers import StepCallbackWrapper
from sequoia.methods import Method
from sequoia.settings import (
    Actions,
    ClassIncrementalSetting,
    IncrementalRLSetting,
    Observations,
    Results,
    Rewards,
    Setting,
)
from sequoia.settings.assumptions.incremental import (
    IncrementalSetting,
    TaskResults,
    TaskSequenceResults,
)
from sequoia.settings.base import SettingABC

from .env_proxy import EnvironmentProxy

logger = getLogger(__file__)

# IDEA: Dict that indicates for each setting, which attributes are *NOT* writeable.
_readonly_attributes: Dict[Type[Setting], List[str]] = {
    ClassIncrementalSetting: ["test_transforms"],
    IncrementalRLSetting: ["test_transforms"],
}
# IDEA: Dict that indicates for each setting, which attributes are *NOT* readable.
_hidden_attributes: Dict[Type[Setting], List[str]] = {
    ClassIncrementalSetting: ["test_class_order"],
    IncrementalRLSetting: ["test_task_schedule", "test_wrappers"],
}

SettingType = TypeVar("SettingType", bound=Setting)


class SettingProxy(SettingABC, Generic[SettingType]):
    """ Proxy for a Setting.

    TODO: Creating the Setting locally for now, but we'd spin-up or contact a gRPC
    service" that would have at least the following endpoints:

    - get_attribute(name: str) -> Any:
        returns the attribute from the setting, if that attribute can be read.

    - set_attribute(name: str, value: Any) -> bool:
        Sets the given attribute to the given value, if that is allowed.

    - train_dataloader()
    - val_dataloader()
    - test_dataloader()
    """

    # NOTE: Using __slots__ so we can detect errors if Method tries to set non-existent
    # attribute on the SettingProxy.
    # TODO: I don't think this has any effect, because we subclass SettingABC which
    # doesn't use __slots__.
    __slots__ = ["__setting", "_setting_type", "_train_env", "_val_env", "_test_env"]

    def __init__(
        self,
        setting_type: Type[SettingType],
        setting_config_path: Path = None,
        **setting_kwargs,
    ):
        self._setting_type = setting_type
        self.__setting: SettingType
        if setting_config_path:
            self.__setting = setting_type.load_benchmark(setting_config_path)
        else:
            self.__setting = setting_type(**setting_kwargs)
        self.__setting.monitor_training_performance = True
        super().__init__()

        self._train_env = None
        self._val_env = None
        self._test_env = None

    @property
    def observation_space(self) -> gym.Space:
        self.set_attribute("train_transforms", self.train_transforms)
        return self.get_attribute("observation_space")

    @property
    def action_space(self) -> gym.Space:
        return self.get_attribute("action_space")

    @property
    def reward_space(self) -> gym.Space:
        return self.get_attribute("reward_space")

    @property
    def val_env(self) -> EnvironmentProxy:
        return self._val_env

    @property
    def train_env(self) -> EnvironmentProxy:
        return self._train_env

    @property
    def test_env(self) -> EnvironmentProxy:
        return self._test_env

    @property
    def config(self) -> Config:
        return self.get_attribute("config")

    @config.setter
    def config(self, value: Config) -> None:
        self.set_attribute("config", value)

    def get_name(self):
        # TODO
        return self.__setting.get_name()

    def _is_readable(self, attribute: str) -> bool:
        if self._setting_type not in _hidden_attributes:
            return True
        return attribute not in _hidden_attributes[self._setting_type]

    def _is_writeable(self, attribute: str) -> bool:
        return attribute not in _readonly_attributes[self._setting_type]

    @property
    def batch_size(self) -> Optional[int]:
        return self.get_attribute("batch_size")

    @batch_size.setter
    def batch_size(self, value: Optional[int]) -> None:
        self.set_attribute("batch_size", value)

    @property
    def train_transforms(self) -> List[Callable]:
        return self.__setting.train_tansforms

    @train_transforms.setter
    def train_transforms(self, value: List[Callable]):
        self.__setting.train_transforms = value

    @property
    def val_transforms(self) -> List[Callable]:
        return self.__setting.val_tansforms

    @val_transforms.setter
    def val_transforms(self, value: List[Callable]):
        self.__setting.val_transforms = value

    @property
    def test_transforms(self) -> List[Callable]:
        return self.__setting.test_tansforms

    @test_transforms.setter
    def test_transforms(self, value: List[Callable]):
        self.__setting.test_transforms = value

    def apply(self, method: Method, config: Config = None) -> Results:
        # TODO: Figure out where the 'config' should be defined?
        method.configure(setting=self)
        # TODO: Not sure if the method is changing the train_transforms.
        # Run the Main loop.
        results: Results = self.main_loop(method)

        logger.info(f"Resulting objective of Test Loop: {results.objective}")
        logger.info(results.summary())
        method.receive_results(self, results=results)
        return results

    def get_attribute(self, name: str) -> Any:
        value = getattr(self.__setting, name)
        if value is None:
            return value
        if not isinstance(value, (int, str, bool, np.ndarray, gym.Space, list)):
            warnings.warn(
                RuntimeWarning(
                    f"TODO: Attribute {name} has a value of type {type(value)}, which "
                    f"wouldn't necessarily be easy to transfer with gRPC. This could "
                    f"mean that we need to implement this on the proxy itself. "
                )
            )
        return value

    def set_attribute(self, name: str, value: Any) -> None:
        return setattr(self.__setting, name, value)

    def train_dataloader(
        self, batch_size: int = None, num_workers: int = None
    ) -> EnvironmentProxy:
        # TODO: Faking this 'remote-ness' for now:

        batch_size = (
            batch_size if batch_size is not None else self.get_attribute("batch_size")
        )
        num_workers = (
            num_workers
            if num_workers is not None
            else self.get_attribute("num_workers")
        )

        self._train_env = EnvironmentProxy(
            env_fn=partial(
                self.__setting.train_dataloader,
                batch_size=batch_size,
                num_workers=num_workers,
            ),
            setting_type=self._setting_type,
        )
        return self._train_env

    def val_dataloader(
        self, batch_size: int = None, num_workers: int = None
    ) -> EnvironmentProxy:

        batch_size = (
            batch_size if batch_size is not None else self.get_attribute("batch_size")
        )
        num_workers = (
            num_workers
            if num_workers is not None
            else self.get_attribute("num_workers")
        )

        self._val_env = EnvironmentProxy(
            env_fn=partial(
                self.__setting.val_dataloader,
                batch_size=batch_size,
                num_workers=num_workers,
            ),
            setting_type=self._setting_type,
        )
        return self._val_env

    def test_dataloader(
        self, batch_size: int = None, num_workers: int = None
    ) -> EnvironmentProxy:

        batch_size = (
            batch_size if batch_size is not None else self.get_attribute("batch_size")
        )
        num_workers = (
            num_workers
            if num_workers is not None
            else self.get_attribute("num_workers")
        )

        self._test_env = EnvironmentProxy(
            env_fn=partial(
                self.__setting.test_dataloader,
                batch_size=batch_size,
                num_workers=num_workers,
            ),
            setting_type=self._setting_type,
        )
        return self._test_env

    def main_loop(self, method: Method) -> Results:
        # TODO: Implement the 'remote' equivalent of the main loop of the IncrementalSetting.

        test_results = self._setting_type.Results()
        test_results._online_training_performance = []

        # TODO: Fix this up, need to get the 'scaling factor' to use for the objective
        # here.
        dataset: str = self.get_attribute("dataset")
        test_results._objective_scaling_factor = (
            0.01 if dataset.startswith("MetaMonsterKong") else 1.0
        )

        method.set_training()

        nb_tasks = self.get_attribute("nb_tasks")
        known_task_boundaries_at_train_time = self.get_attribute(
            "known_task_boundaries_at_train_time"
        )
        task_labels_at_train_time = self.get_attribute("task_labels_at_train_time")
        start_time = time.process_time()

        # Send the train / val transforms to the 'remote' env.
        self.set_attribute("train_transforms", self.train_transforms)
        self.set_attribute("val_transforms", self.val_transforms)

        for task_id in range(nb_tasks):
            logger.info(
                f"Starting training" + (f" on task {task_id}." if nb_tasks > 1 else ".")
            )
            self.set_attribute("_current_task_id", task_id)

            if known_task_boundaries_at_train_time:
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
                elif not task_labels_at_train_time:
                    method.on_task_switch(None)
                else:
                    # NOTE: on_task_switch won't be called if there is only one "task",
                    # (as-in one task in a 'sequence' of tasks).
                    # TODO: in multi-task RL, i.e. RLSetting(dataset=..., nb_tasks=10),
                    # for instance, then there are indeed 10 tasks, but `self.tasks`
                    # is used here to describe the number of 'phases' in training and
                    # testing.
                    if nb_tasks > 1:
                        method.on_task_switch(task_id)

            task_train_loader = self.train_dataloader()
            task_valid_loader = self.val_dataloader()
            success = method.fit(
                train_env=task_train_loader, valid_env=task_valid_loader,
            )
            task_train_loader.close()
            task_valid_loader.close()

            test_results._online_training_performance.append(
                task_train_loader.get_online_performance()
            )

            test_loop_results = self.test_loop(method)
            test_results.append(test_loop_results)

            logger.info(f"Finished Training on task {task_id}.")

        runtime = time.process_time() - start_time
        test_results._runtime = runtime
        return test_results

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
        nb_tasks = self.get_attribute("nb_tasks")
        known_task_boundaries_at_test_time = self.get_attribute(
            "known_task_boundaries_at_test_time"
        )
        task_labels_at_test_time = self.get_attribute("task_labels_at_test_time")

        was_training = method.training
        method.set_testing()
        test_env = self.test_dataloader()

        if known_task_boundaries_at_test_time and nb_tasks > 1:
            # TODO: We need to have a way to inform the Method of task boundaries, if the
            # Setting allows it.
            # Not sure how to do this. It might be simpler to just do something like
            # `obs, rewards, done, info, task_switched = <endpoint>.step(actions)`?
            # # Add this wrapper that will call `on_task_switch` when the right step is
            # # reached.
            # test_env = StepCallbackWrapper(test_env, callbacks=[_on_task_switch])
            pass

        obs = test_env.reset()
        batch_size = test_env.batch_size
        max_steps: int = self.get_attribute("test_steps") // (batch_size or 1)

        # Reset on the last step is causing trouble, since the env is closed.
        pbar = tqdm.tqdm(itertools.count(), total=max_steps, desc="Test")
        episode = 0
        for step in pbar:
            if test_env.is_closed():
                logger.debug(f"Env is closed")
                break

            # BUG: This doesn't work if the env isn't batched.
            action_space = test_env.action_space
            env_is_batched = getattr(test_env, "num_envs", getattr(test_env, "batch_size", 0)) >= 1
            if env_is_batched:
                # NOTE: Need to pass an action space that actually reflects the batch
                # size, even for the last batch!
                obs_batch_size = obs.x.shape[0] if obs.x.shape else None
                action_space_batch_size = (
                    test_env.action_space.shape[0]
                    if test_env.action_space.shape
                    else None
                )
                if (
                    obs_batch_size is not None
                    and obs_batch_size != action_space_batch_size
                ):
                    action_space = batch_space(
                        test_env.single_action_space, obs_batch_size
                    )

            action = method.get_actions(obs, action_space)

            # logger.debug(f"action: {action}")
            obs, reward, done, info = test_env.step(action)

            # TODO: Add something to `info` that indicates when a task boundary is
            # reached, so that we can call the `on_task_switch` method on the Method
            # ourselves.

            if done and not test_env.is_closed():
                # logger.debug(f"end of test episode {episode}")
                obs = test_env.reset()
                episode += 1

        test_env.close()
        test_results = test_env.get_results()

        if was_training:
            method.set_training()

        return test_results

    # NOTE: Was experimenting with the idea of allowing the regular getattr and setattr
    # to forward calls to the remote. In the end I think it's better to explicitly
    # prevent any of these from happening.

    def __getattr__(self, name: str):
        # NOTE: This only ever gets called if the attribute was not found on the
        if self._is_readable(name):
            print(f"Accessing missing attribute {name} from the 'remote' setting.")
            return self.get_attribute(name)
        raise AttributeError(
            f"Attribute {name} is either not present on the setting, or not marked as "
            f"readable!"
        )

    # def __setattr__(self, name: str, value: Any) -> None:
    #     # Weird pytorch-lightning stuff:
    #     logger.debug(f"__setattr__ called for attribute {name}")
    #     if name in {"_setting_type", "__setting"}:
    #         assert name not in self.__dict__, f"Can't change attribute {name}"
    #         object.__setattr__(self, name, value)

    #     elif self._is_writeable(name):
    #         logger.info(f"Setting attribute {name} on the 'remote' setting.")
    #         self.set_attribute(name, value)
    #     else:
    #         raise AttributeError(f"Attribute {name} is marked as read-only!")
