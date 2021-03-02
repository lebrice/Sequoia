from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union, Generic

import gym
from gym.envs.classic_control import CartPoleEnv
from sequoia.utils import get_logger

logger = get_logger(__file__)
EnvType = TypeVar("Env", bound=gym.Env)
TaskType = TypeVar("TaskType", bound="Task")

task_param_names: Dict[Union[Type[gym.Env], str], List[str]] = {
    CartPoleEnv: ["gravity", "masscart", "masspole", "length", "force_mag", "tau",]
    # TODO: Add more of the classic control envs here.
}
task_param_names["CartPole-v0"] = task_param_names["CartPole-v1"] = task_param_names[
    CartPoleEnv
]

@dataclass
class Task(Generic[EnvType], ABC):
    """ ABC for a 'task', i.e. a 'thing' that affects the environment, restricting it
    to some subset of its state-space -ish.
    """

    @abstractmethod
    def __call__(self, env: EnvType) -> EnvType:
        """Applies this 'task' to the given environment, returning the modified env."""

    # TODO: IDEA: Use a contextmanager method to apply / un-apply the tasks!


@dataclass
class ApplyFunctionToEnv(Task):
    """Nonstationarity which applies a given function to the environment.
    """

    function: Callable
    args: Tuple[Any, ...] = ()
    kwargs: Dict[Any, Any] = field(default_factory=dict)

    def __call__(self, env: EnvType) -> EnvType:
        return self.function(env, *self.args, **self.kwargs)


@dataclass(init=False)
class ChangeEnvAttributes(Task[EnvType]):
    """Nonstationarity which changes the attributes of an environment.

    This modifies the environment in-place.
    """

    attribute_dict: Dict[str, Any] = field(default_factory={})

    def __init__(self, attribute_dict: Dict[str, Any] = None, **kwargs):
        super().__init__()
        self.attribute_dict = attribute_dict or dict(kwargs)

    def __call__(self, env: EnvType) -> EnvType:
        for key, value in self.attribute_dict.items():
            setattr(env.unwrapped, key, value)
        return env


# # TODO: Maybe use something like this for SL!
# from sequoia.settings.passive.passive_environment import PassiveEnvironment

# PassiveEnvironmentType = TypeVar("PassiveEnvironmentType", bound=PassiveEnvironment)


# @dataclass
# class UseDatasetSubset(Task[PassiveEnvironmentType]):
#     pass


def get_changeable_attributes(
    env: Union[str, gym.Env, Type[gym.Env]]
) -> Dict[str, Any]:
    """Returns the environment's attributes which could be changed to generate 'tasks'.

    Parameters
    ----------
    env : Union[str,gym.Env, Type[gym.Env]]
        Either an environment ID (str), a gym.Env instance, or a type of gym.Env.

    Returns
    -------
    Dict[str, Any]
        A dictionary mapping from attribute name to the default value for that
        attribute.

    Raises
    ------
    NotImplementedError
        If the env isn't supported.
    """
    param_names: Optional[List[str]] = None
    if isinstance(env, gym.Env):
        param_names = task_param_names.get(type(env.unwrapped))
    else:
        param_names = task_param_names.get(env)
    if not param_names:
        raise NotImplementedError(
            f"Don't yet know which attributes can be changed for env `env`."
        )

    param_values: Dict[str, Any] = {}
    for param_name in param_names:
        param_values = getattr(env, param_name)
    return param_values
