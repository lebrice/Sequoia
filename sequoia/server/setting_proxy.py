import itertools
import warnings
from abc import ABC, abstractmethod
from functools import partial
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, Generic, List, Type, TypeVar

import gym
import numpy as np
import tqdm
from sequoia.common import Config
from sequoia.common.gym_wrappers import StepCallbackWrapper
from sequoia.methods import Method
from sequoia.settings import (
    Actions,
    ClassIncrementalSetting,
    Observations,
    Results,
    Rewards,
    Setting,
)
from sequoia.settings.assumptions import IncrementalSetting
from sequoia.settings.base import SettingABC
from torch import Tensor

from .env_proxy import EnvironmentProxy

logger = getLogger(__file__)

# IDEA: Dict that indicates for each setting, which attributes are *NOT* writeable.
_readonly_attributes: Dict[Type[Setting], List[str]] = {
    ClassIncrementalSetting: ["test_transforms"]
}
# IDEA: Dict that indicates for each setting, which attributes are *NOT* readable.
_hidden_attributes: Dict[Type[Setting], List[str]] = {
    ClassIncrementalSetting: ["test_class_order"]
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
    __slots__ = ["_setting", "_setting_type", "train_env", "valid_env", "test_env"]

    def __init__(
        self,
        setting_type: Type[SettingType],
        setting_config_path: Path = None,
        **setting_kwargs,
    ):
        self._setting_type = setting_type
        self._setting: SettingType
        if setting_config_path:
            self._setting = setting_type.load_benchmark(setting_config_path)
        else:
            self._setting = setting_type(**setting_kwargs)
        super().__init__()

    @property
    def observation_space(self) -> gym.Space:
        return self.get_attribute("observation_space")

    @property
    def action_space(self) -> gym.Space:
        return self.get_attribute("action_space")

    @property
    def reward_space(self) -> gym.Space:
        return self.get_attribute("reward_space")

    @property
    def config(self) -> Config:
        return self.get_attribute("config")
    
    @config.setter
    def config(self, value: Config) -> None:
        self.set_attribute("config", value)
    
    def get_name(self):
        # TODO
        return self._setting.get_name()
    
    def _is_readable(self, attribute: str) -> bool:
        return attribute not in _hidden_attributes[self._setting_type]

    def _is_writeable(self, attribute: str) -> bool:
        return attribute not in _readonly_attributes[self._setting_type]

    def apply(self, method: Method, config: Config = None) -> Results:
        # TODO: Figure out where the 'config' should be defined?
        method.configure(setting=self)

        # Run the Training loop.
        self.train_loop(method)
        # Run the Test loop.
        results: Results = self.test_loop(method)

        logger.info(f"Resulting objective of Test Loop: {results.objective}")
        logger.info(results.summary())
        method.receive_results(self, results=results)
        return results

    def get_attribute(self, name: str) -> Any:
        value = getattr(self._setting, name)
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
        return setattr(self._setting, name, value)

    def train_dataloader(
        self, batch_size: int = None, num_workers: int = None
    ) -> EnvironmentProxy:
        # TODO: Faking this 'remote-ness' for now:
        self.train_env = EnvironmentProxy(
            env_fn=partial(
                self._setting.train_dataloader,
                batch_size=batch_size,
                num_workers=num_workers,
            ),
            setting_type=self._setting_type,
        )
        return self.train_env

    def val_dataloader(
        self, batch_size: int = None, num_workers: int = None
    ) -> EnvironmentProxy:
        self.valid_env = EnvironmentProxy(
            env_fn=partial(
                self._setting.val_dataloader,
                batch_size=batch_size,
                num_workers=num_workers,
            ),
            setting_type=self._setting_type,
        )
        return self.valid_env

    def test_dataloader(
        self, batch_size: int = None, num_workers: int = None
    ) -> EnvironmentProxy:
        self.test_env = EnvironmentProxy(
            env_fn=partial(
                self._setting.test_dataloader,
                batch_size=batch_size,
                num_workers=num_workers,
            ),
            setting_type=self._setting_type,
        )
        return self.test_env

    def train_loop(self, method: Method):
        """ (WIP): Runs an incremental training loop, wether in RL or CL."""

        nb_tasks = self.get_attribute("nb_tasks")
        known_task_boundaries_at_train_time = self.get_attribute(
            "known_task_boundaries_at_train_time"
        )
        task_labels_at_train_time = self.get_attribute("task_labels_at_train_time")

        for task_id in range(nb_tasks):
            logger.info(
                f"Starting training" + (f" on task {task_id}." if nb_tasks > 1 else ".")
            )
            self.set_attribute("current_task_id", task_id)

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
            logger.info(f"Finished Training on task {task_id}.")

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
        max_steps: int = self.get_attribute("test_steps")

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
    #     if name in {"_setting_type", "_setting"}:
    #         assert name not in self.__dict__, f"Can't change attribute {name}"
    #         object.__setattr__(self, name, value)

    #     elif self._is_writeable(name):
    #         logger.info(f"Setting attribute {name} on the 'remote' setting.")
    #         self.set_attribute(name, value)
    #     else:
    #         raise AttributeError(f"Attribute {name} is marked as read-only!")
