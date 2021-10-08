from collections.abc import Mapping
from dataclasses import is_dataclass, replace, fields
from functools import singledispatch
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, TypeVar, Union

import gym
import numpy as np
from gym import Space, spaces
from sequoia.common import Batch
from sequoia.common.gym_wrappers import IterableWrapper, TransformObservation
from sequoia.common.gym_wrappers.convert_tensors import supports_tensors
from sequoia.common.spaces import Sparse, TypedDictSpace
from sequoia.common.spaces.named_tuple import NamedTuple, NamedTupleSpace
from sequoia.settings.base.environment import Environment
from sequoia.settings.base.objects import (
    Actions,
    ActionType,
    Observations,
    ObservationType,
    Rewards,
    RewardType,
)
from torch import Tensor

T = TypeVar("T")


class TypedObjectsWrapper(IterableWrapper, Environment[ObservationType, ActionType, RewardType]):
    """Wrapper that converts the observations and rewards coming from the env
    to `Batch` objects.

    NOTE: Not super necessary atm, but this would perhaps be useful if methods
    are built and expect to have a given 'type' of observations to work with,
    then any new setting that inherits from their target setting should have
    observations that subclass/inherit from the observations of their parent, so
    as not to break compatibility.

    For example, if a Method targets the ClassIncrementalSetting, then it
    expects to receive "observations" of the type described by
    ClassIncrementalSetting.Observations, and if it were to be applied on a
    TaskIncrementalSLSetting (which inherits from ClassIncrementalSetting), then
    the observations from that setting should be isinstances (or subclasses of)
    the Observations class that this method was designed to receive!
    """

    def __init__(
        self,
        env: gym.Env,
        observations_type: ObservationType,
        rewards_type: RewardType,
        actions_type: ActionType,
        observation_space: TypedDictSpace = None,
        action_space: TypedDictSpace = None,
        reward_space: TypedDictSpace = None,
    ):
        self.Observations = observations_type
        self.Rewards = rewards_type
        self.Actions = actions_type
        super().__init__(env=env)

        observation_fields = fields(self.Observations)
        action_fields = fields(self.Actions)
        reward_fields = fields(self.Rewards)

        if not all([observation_fields, action_fields, reward_fields]):
            raise RuntimeError(
                f"The Observations/Actions/Rewards classes passed to the TypedObjectsWrapper all need to have at least one field!"
            )

        simple_spaces = (spaces.Box, spaces.Discrete, spaces.MultiDiscrete, spaces.MultiBinary)
        num_envs = getattr(self.env, "num_envs", None)

        # Set the observation space.
        if observation_space:
            self.observation_space = observation_space
        elif isinstance(self.env.observation_space, spaces.Dict):
            # Convert the spaces.Dict into a TypedDictSpace, or replace a TypedDictSpace's `dtype`.
            self.observation_space = TypedDictSpace(
                spaces=self.env.observation_space.spaces,
                dtype=self.Observations,
            )
        elif isinstance(self.env.observation_space, simple_spaces) and len(observation_fields) == 1:
            # we can get away with this since the class has only one field and the space is simple.
            field_name = observation_fields[0].name
            self.observation_space = TypedDictSpace(
                spaces={field_name: self.env.observation_space}, dtype=self.Observations
            )
        else:
            raise NotImplementedError(
                f"Need to pass the observation space to the TypedObjectsWrapper constructor when "
                f"the wrapped env's observation space isn't already a Dict or TypedDictSpace and "
                f"`Observations` has more than one field. (Observations: {self.Observations})"
            )

        # Set/construct the action space.
        if action_space:
            self.action_space = action_space
        elif isinstance(self.env.action_space, spaces.Dict):
            # Convert the spaces.Dict into a TypedDictSpace, or replace a TypedDictSpace's `dtype`.
            self.action_space = TypedDictSpace(
                spaces=self.env.action_space.spaces,
                dtype=self.Actions,
            )
        elif (isinstance(self.env.action_space, simple_spaces) and len(action_fields) == 1) or (
            isinstance(self.env.action_space, spaces.Tuple) and num_envs):
            field_name = action_fields[0].name
            self.action_space = TypedDictSpace(
                spaces={field_name: self.env.action_space}, dtype=self.Actions
            ) 
        else:
            raise NotImplementedError(
                "Need to pass the action space to the TypedObjectsWrapper constructor when "
                "the wrapped env's action space isn't already a Dict or TypedDictSpace and "
                "the Actions class doesn't have just one field."
                f"(wrapped action space: {self.env.action_space}, Actions: {self.Actions})"
            )

        # Set / construct the reward space.

        # Get the default reward space in case the wrapped env doesn't have a `reward_space` attr.
        default_reward_space = spaces.Box(
            low=self.env.reward_range[0],
            high=self.env.reward_range[1],
            shape=((num_envs,) if num_envs is not None else ()),
            dtype=np.float64,
        )

        if reward_space:
            self.reward_space = reward_space
        elif not hasattr(self.env, "reward_space"):
            if len(reward_fields) != 1:
                raise NotImplementedError(
                    "Need to pass the reward space to the TypedObjectsWrapper constructor when "
                    "the wrapped env doesn't have a `reward_space` attribute and the Rewards "
                    "class has more than one field."
                )
            field_name = reward_fields[0].name
            self.reward_space = TypedDictSpace(
                spaces={field_name: default_reward_space},
                dtype=self.Rewards,
            )
        elif isinstance(self.env.reward_space, spaces.Dict):
            # Convert the spaces.Dict into a TypedDictSpace, or replace a TypedDictSpace's `dtype`.
            self.reward_space = TypedDictSpace(
                spaces=self.env.reward_space.spaces,
                dtype=self.Rewards,
            )
        elif isinstance(self.env.reward_space, simple_spaces) and len(reward_fields) == 1:
            field_name = reward_fields[0].name
            self.reward_space = TypedDictSpace(
                spaces={field_name: self.env.reward_space},
                dtype=self.Rewards,
            )
        else:
            raise NotImplementedError(
                "Need to pass the reward space to the TypedObjectsWrapper constructor when "
                "the wrapped env's reward space isn't already a Dict or TypedDictSpace and "
                "the Rewards class doesn't have just one field."
            )

        # TODO: Using a TypedDictSpace for the action/reward spaces is a small change in code, but
        # will most likely have a large impact on all the methods and tests!
        # THis here can be used to 'turn off' the changes to those spaces done above:
        self.action_space = self.env.action_space
        self.reward_space = getattr(self.env, "reward_space", default_reward_space)

        # if isinstance(self.env.observation_space, NamedTupleSpace):
        #     self.observation_space = self.env.observation_space
        #     self.observation_space.dtype = self.Observations

    def step(
        self, action: ActionType
    ) -> Tuple[
        ObservationType, RewardType, Union[bool, Sequence[bool]], Union[Dict, Sequence[Dict]]
    ]:
        # "unwrap" the actions before passing it to the wrapped environment.
        action = self.action(action)
        observation, reward, done, info = self.env.step(action)
        # TODO: Make the observation space a Dict
        observation = self.observation(observation)
        reward = self.reward(reward)
        return observation, reward, done, info

    def observation(self, observation: Any) -> ObservationType:
        if isinstance(observation, self.Observations):
            return observation
        if isinstance(observation, tuple):
            # TODO: Dissallow this: shouldn't handle tuples since they can be quite ambiguous.
            # assert False, observation
            return self.Observations(*observation)
        if isinstance(observation, dict):
            try:
                return self.Observations(**observation)
            except TypeError:
                assert False, (self.Observations, observation)
        assert isinstance(observation, (Tensor, np.ndarray))
        return self.Observations(observation)

    def action(self, action: ActionType) -> Any:
        # TODO: Assert this eventually
        # assert isinstance(action, Actions), action
        if isinstance(action, Actions):
            action = action.y_pred
        if isinstance(action, Tensor) and not supports_tensors(self.env.action_space):
            action = action.detach().cpu().numpy()
        if action not in self.env.action_space:
            if isinstance(self.env.action_space, spaces.Tuple):
                action = tuple(action)
        return action

    def reward(self, reward: Any) -> RewardType:
        return self.Rewards(reward)

    def reset(self, **kwargs) -> ObservationType:
        observation = self.env.reset(**kwargs)
        return self.observation(observation)

    def __iter__(self):
        for batch in self.env:
            if isinstance(batch, tuple) and len(batch) == 2:
                yield self.observation(batch[0]), self.reward(batch[1])
            elif isinstance(batch, tuple) and len(batch) == 1:
                yield self.observation(batch[0])
            else:
                yield self.observation(batch)

    def send(self, action: ActionType) -> RewardType:
        action = self.action(action)
        reward = self.env.send(action)
        return self.reward(reward)


# TODO: turn unwrap into a single-dispatch callable.
# TODO: Atm 'unwrap' basically means "get rid of everything apart from the first
# item", which is a bit ugly.
# Unwrap should probably be a method on the corresponding `Batch` class, which could
# maybe accept a Space to fit into?
@singledispatch
def unwrap(obj: Any) -> Any:
    return obj
    # raise NotImplementedError(obj)


@unwrap.register(int)
@unwrap.register(float)
@unwrap.register(np.ndarray)
@unwrap.register(list)
def _unwrap_scalar(v):
    return v


@unwrap.register(Actions)
def _unwrap_actions(obj: Actions) -> Union[Tensor, np.ndarray]:
    return obj.y_pred


@unwrap.register(Rewards)
def _unwrap_rewards(obj: Rewards) -> Union[Tensor, np.ndarray]:
    return obj.y


@unwrap.register(Observations)
def _unwrap_observations(obj: Observations) -> Union[Tensor, np.ndarray]:
    # This gets rid of everything except just the image.
    # TODO: Keep the task labels? or no? For now, no.
    return obj.x


@unwrap.register(NamedTupleSpace)
def _unwrap_space(obj: NamedTupleSpace) -> Space:
    # This gets rid of everything except just the first item in the space.
    # TODO: Keep the task labels? or no? For now, no.
    return obj[0]


@unwrap.register(TypedDictSpace)
def _unwrap_space(obj: TypedDictSpace) -> spaces.Dict:
    # This gets rid of everything except just the first item in the space.
    # TODO: Keep the task labels? or no? For now, no.
    return spaces.Dict(obj.spaces)


class NoTypedObjectsWrapper(IterableWrapper):
    """Does the opposite of the 'TypedObjects' wrapper.

    Can be added on top of that wrapper to strip off the typed objects it
    returns and just returns tensors/np.ndarrays instead.

    This is used for example when applying a method from stable-baselines3, as
    they only want to get np.ndarrays as inputs.

    Parameters
    ----------
    IterableWrapper : [type]
        [description]
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = unwrap(self.env.observation_space)

    def step(self, action):
        if isinstance(action, Actions):
            action = unwrap(action)
        if hasattr(action, "detach"):
            action = action.detach()
        assert action in self.action_space, (action, type(action), self.action_space)
        observation, reward, done, info = self.env.step(action)
        observation = unwrap(observation)
        reward = unwrap(reward)
        return observation, reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return unwrap(observation)
