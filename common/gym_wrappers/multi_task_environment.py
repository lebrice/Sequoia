import bisect
import random
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces
from gym.envs.classic_control import CartPoleEnv
from gym.envs.registration import register

from utils.logging_utils import get_logger

task_param_names: Dict[Union[Type[gym.Env], str], List[str]] = {
    CartPoleEnv: [
        "gravity",
        "masscart",
        "masspole",
        "length",
        "force_mag",
        "tau",
    ]
    # TODO: Add more of the classic control envs here.
}
logger = get_logger(__file__)


class MultiTaskEnvironment(gym.Wrapper):
    """ Creates 'tasks' by modifying attributes of the wrapped environment.

    This wrapper accepts a `task_schedule` dictionary, which maps from a given
    step to the attributes that are to be set at that task.

    For example, when wrapping the "CartPole-v0" environment, we could vary any
    of the "gravity", "masscart", "masspole", "length", "force_mag" or "tau"
    attributes:
    ```
    env = gym.make("CartPole-v0")
    env = MultiTaskEnvironment(env, task_schedule={
        # step -> attributes to set on the environment when step is reached.
        10: dict(length=2.0),
        20: dict(length=1.0, gravity=20.0),
        30: dict(length=0.5, gravity=5.0),
    })
    env.seed(123)
    env.reset()
    ```
    During steps 0-9, the environment is unchanged (length = 0.5).
    At step 10, the length of the pole will be set to 2.0
    At step 20, the length of the pole will be set to 1.0, and the gravity will
        be changed from its default value (9.8) to 20.
    etc.
    """
    def __init__(self,
                 env: gym.Env,
                 task_schedule: Dict[int, Dict[str, float]] = None,
                 task_params: List[str] = None,
                 noise_std: float = 0.2,
                 add_task_dict_to_info: bool = False,
                 add_task_id_to_obs: bool = False,
                 starting_step: int = 0,
                 max_steps: int = None):
        """ Wraps an environment, allowing it to be 'multi-task'.

        NOTE: Assumes that all the attributes in 'task_param_names' are floats
        for now.

        TODO: Do we want to add the task labels as a dictionary? or just an 'index'? 

        Args:
            env (gym.Env): The environment to wrap.
            task_param_names (List[str], optional): The attributes of the
                environment that will be allowed to change. Defaults to None.
            task_schedule (Dict[int, Dict[str, float]], optional): Schedule
                mapping from a given step number to the state that will be set
                at that time.
            noise_std (float, optional): The standard deviation of the noise
                used to create the different tasks.
        """
        super().__init__(env=env)
        self.env: gym.Env
        
        self.noise_std = noise_std
        if not task_params:
            unwrapped_type = type(env.unwrapped)
            if unwrapped_type in task_param_names:
                task_params = task_param_names[unwrapped_type]
            else:
                pass
                # logger.warning(UserWarning(
                #     f"You didn't pass any 'task params', and the task "
                #     f"parameters aren't known for this type of environment "
                #     f"({unwrapped_type}), so we can't make it multi-task with "
                #     f"this wrapper."
                # ))

        self._max_steps: Optional[int] = max_steps
        self._starting_step: int = starting_step
        self._steps: int = self._starting_step

        self._current_task: Dict = {}
        self._task_schedule: Dict[int, Dict[str, Any]] = OrderedDict()
        
        self.task_params: List[str] = task_params or []
        self.default_task: np.ndarray = self.current_task.copy()
        self.task_schedule = task_schedule or {}
        
        # Wether we will add a task id to the observation.
        self.add_task_id_to_obs = add_task_id_to_obs
        # Wether we will add the task dict (the values of the attributes) to the
        # 'info' dict.
        self.add_task_dict_to_info = add_task_dict_to_info
        
        if 0 not in self.task_schedule:
            self.task_schedule[0] = self.default_task
        
        n_tasks = len(self.task_schedule)
        
        if self.add_task_id_to_obs:
            self.observation_space = spaces.Tuple([
                self.env.observation_space,
                spaces.Discrete(n=n_tasks)
            ])
        
        self._closed = False
        
        self._on_task_switch_callback: Optional[Callable[[int], None]] = None

    @property
    def current_task_id(self) -> int:
        """ Returns the 'index' of the current task within the task schedule.
        """
        current_step = self._steps
        assert current_step >= 0
        task_steps: List[int] = sorted(self.task_schedule.keys())
        assert 0 in task_steps
        insertion_index = bisect.bisect_right(task_steps, current_step)
        # The current task id is the insertion index - 1
        current_task_index = insertion_index - 1
        return current_task_index

    def set_on_task_switch_callback(self, callback: Callable[[int], None]) -> None:
        self._on_task_switch_callback = callback
    
    def on_task_switch(self, task_id: int):
        logger.debug(f"Switching from {self.current_task_id} -> {task_id}.")
        # TODO: We could maybe use this to call the method's 'on_task_switch'
        # callback?
        if self._on_task_switch_callback:
            self._on_task_switch_callback(task_id)
    

    def step(self, *args, **kwargs):
        # If we reach a step in the task schedule, then we change the task to
        # that given step.
        if self._closed:
            raise gym.error.ClosedEnvironmentError("Can't step in closed env.")
        
        if self.steps in self.task_schedule:
            self.current_task = self.task_schedule[self.steps]
            # Adding this on_task_switch, since it could maybe be easier than
            # having to add a callback wrapper to use.
            task_id = sorted(self.task_schedule.keys()).index(self.steps)
            self.on_task_switch(task_id)
            
        observation, rewards, done, info = super().step(*args, **kwargs)
        if self.add_task_id_to_obs:
            observation = (observation, self.current_task_id)
        if self.add_task_dict_to_info:
            info.update(self.current_task)

        self.steps += 1       
        return observation, rewards, done, info

    def close(self, **kwargs) -> None:
        self.env.close(**kwargs)
        self._closed = True
    
    def reset(self, new_random_task: bool = False, **kwargs):
        """ Resets the wrapped environment.
        
        If `new_random_task` is True, this also sets a new random task as the
        current task.
        
        NOTE: This resets the wrapped env, but doesn't reset the number of steps
        taken, hence the 'task' progression according to the task_schedule
        doesn't change.
        """
        if self._closed:
            raise gym.error.ClosedEnvironmentError("Can't reset closed env.")
        if new_random_task:
            self.current_task = self.random_task()
        observation = self.env.reset(**kwargs)
        if self.add_task_id_to_obs:
            observation = (observation, self.current_task_id)
        return observation

    @property
    def steps(self) -> int:
        return self._steps

    @steps.setter
    def steps(self, value: int) -> None:
        if value < self._starting_step:
            value = self._starting_step 
        if self._max_steps is not None and value > self._max_steps:
            # Reached the maximum number of steps, stagnate.
            # TODO: What exactly should we do in this case? Should we close
            # the env? Or just stay at the same 'step' in the task schedule
            # forever?
            # TODO: Is this the "correct" way to limit the number of steps in
            # an environment?
            value = self._max_steps
        self._steps = value

    @property
    def current_task(self) -> Dict[str, Any]:
        # NOTE: This caching mechanism assumes that we are the only source
        # of potential change for these attributes.
        # At the moment, We're not really concerned with performance, so we
        # could turn it off it if misbehaves or causes bugs.
        if not self._current_task:
            # NOTE: We get the attributes from the unwrapped environment, which
            # effectively bypasses any wrappers. Don't know if this is good
            # practice, but oh well.
            self._current_task = OrderedDict(
                (name, getattr(self.env.unwrapped, name))
                for name in self.task_params
            )
        # Double-checking that the attributes didn't change somehow without us
        # knowing.
        # TODO: Maybe remove this when done debugging/testing this since it's a
        # little bit of a waste of compute.
        for attribute, value_in_dict in self._current_task.items():
            current_env_value = getattr(self.env.unwrapped, attribute)
            if value_in_dict != current_env_value:
                raise RuntimeError(
                    f"The value of the attribute '{attribute}' was changed from "
                    f"somewhere else! (value in _current_task: {value_in_dict}, "
                    f"value on env: {current_env_value})"
                )
        return self._current_task

    @current_task.setter
    def current_task(self, task: Union[Dict[str, float], Sequence[float]]):
        # logger.debug(f"(_step: {self.steps}): Setting the current task to {task}.")
        self._current_task.clear()
        self._current_task.update(self.default_task)

        if isinstance(task, dict):
            for k, value in task.items():
                assert isinstance(k, str), "The keys of the task dict should be strings."    
                self._current_task[k] = value
        else:
            assert len(task) == len(self.task_params), "lengths should match!"
            for k, value in zip(self.task_params, task):
                self._current_task[k] = value

        # Actually change the value of the task attributes in the environment.
        for name, param_value in self._current_task.items():
            assert hasattr(self.env.unwrapped, name), (
                f"the unwrapped environment doesn't have a {name} attribute!"
            )
            setattr(self.env.unwrapped, name, param_value)

    def random_task(self) -> Dict:
        """Samples a random 'task', i.e. a random set of attributes.
        How the random value for an attribute is sampled depends on the type of
        its default value in the envionment:

        - `int`, `float`, or `np.ndarray` attributes are sampled by multiplying
            the default value by a N(mean=1., std=`self.noise_std`). `int`
            attributes are then rounded to the nearest value.

        - `bool` attributes are sampled randomly from `True` and `False`.

        TODO: It might be cool to give an option for passing a prior that could
        be used for a given attribute, but it would add a bit too much
        complexity and isn't really needed atm.
 
        Raises:
            NotImplementedError: If the default value has an unsupported type.

        Returns:
            Dict: A dict of the attribute name, and the value that would be set
                for that attribute.
        """
        task: Dict = OrderedDict()
        for attribute, default_value in self.default_task.items():
            new_value = default_value
            if isinstance(default_value, (int, float, np.ndarray)):
                new_value *= random.normalvariate(1.0, self.noise_std)
                # Clip the value to be in the [0.1*default, 10*default] range.
                new_value = max(0.1 * default_value, new_value)
                new_value = min(10 * default_value, new_value)
                if isinstance(default_value, int):
                    new_value = round(new_value)
            elif isinstance(default_value, bool):
                new_value = random.choice([True, False])
            else:
                raise NotImplementedError(
                    f"TODO: Don't yet know how to sample a random value for "
                    f"attribute {attribute} with default value {default_value} of type "
                    f" {type(default_value)}."
                )
            task[attribute] = new_value
        return task

    def update_task(self, values: Dict = None, **kwargs):
        """Updates the current task with the params from values or kwargs.

        Important: Use this method to update properties of the current task,
        instead of trying modifying the `current_task` dictionary. For example,
        `env.current_task["length"] = 2.0` will NOT update the length of
        the pole in CartPole, whereas using `env.update_task(length=2.0)` will!

        NOTE: When passing a dictionary, any missing param is kept at its
        current value (not reset to the default value).
        """
        current_task = self.current_task.copy()
        if isinstance(values, dict):
            current_task.update(values)
        elif values is not None:
            raise RuntimeError(f"values can only be a dict or None (received {values}).")
        if kwargs:
            current_task.update(kwargs)
        self.current_task = current_task

    def seed(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            np.random.seed(seed)
        return self.env.seed(seed)

    def task_dict(self, task_array: np.ndarray) -> Dict[str, float]:
        assert len(task_array) == len(self.task_params), (
            "Lengths should match the number of task parameters."
        )
        return OrderedDict(zip(self.task_params, task_array))

    @property
    def task_schedule(self):
        return self._task_schedule

    @task_schedule.setter
    def task_schedule(self, value: Dict[str, Any]):
        self._task_schedule = OrderedDict()
        if 0 not in value:
            self._task_schedule[0] = self.default_task.copy()

        for step, task in sorted(value.items()):
            # Convert any numpy arrays or lists in the task schedule to dicts
            # mapping from attribute name to value to be set.
            if isinstance(task, (list, np.ndarray)):
                task = self.task_dict(task)
            if not isinstance(task, dict):
                raise RuntimeError(
                    f"Task schedule can only contain dicts, lists or numpy "
                    f"arrays, but got {task}!"
                )
            self._task_schedule[step] = task

        if self._steps in self._task_schedule:
            self.current_task = self._task_schedule[self._steps]

# def MultiTaskCartPole():
#     env = gym.make("CartPole-v0")
#     return MultiTaskEnvironment(env, noise_std=0.1)

# MULTI_TASK_CARTPOLE: str = 'MultiTaskCartPole-v1'

# try:
#     register(
#         id=MULTI_TASK_CARTPOLE,
#         entry_point='settings.active.continual.multi_task_environment:MultiTaskCartPole',
#     )
# except gym.error.Error:
#     pass
