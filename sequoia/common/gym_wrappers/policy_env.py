"""TODO: Idea: create a wrapper that accepts a 'policy' which will decide an
action to take whenever the `action` argument to the `step` method is None.

This policy should then accept the 'state' or something like that.
"""
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Iterator,
    Optional,
    Tuple,
    TypeVar,
    Dict,
    Generic,
    Iterable,
)

import gym
from torch.utils.data import IterableDataset

from sequoia.common.batch import Batch
from sequoia.utils.logging_utils import get_logger

from .utils import StepResult

logger = get_logger(__file__)
# from sequoia.settings.base.environment import Environment
# from sequoia.settings.base.objects import (ActionType, ObservationType, RewardType)
ObservationType = TypeVar("ObservationType")
ActionType = TypeVar("ActionType")
RewardType = TypeVar("RewardType")

# Just for type hinting purposes.


class Environment(gym.Env, Generic[ObservationType, ActionType, RewardType]):
    def step(self, action: ActionType) -> Tuple[ObservationType, RewardType, bool, Dict]:
        raise NotImplementedError

    def reset(self) -> ObservationType:
        raise NotImplementedError


DatasetItem = TypeVar("DatasetItem")

# Type annotation for functions that will create the items of the
# IterableDataset below, given the current 'Context',
DatasetItemCreator = Callable[
    [
        ObservationType, # 'current' state
        ActionType, # actions applied on the 'current' state
        ObservationType, # resulting 'next' state
        RewardType, # rewards associated with the transition above
        bool, # Wether the 'next' state is final (i.e. the last in an episode)
        Dict, # the 'info' dict associated with the 'next' state (from Env.step)
    ],
    DatasetItem
]

@dataclass(frozen=True)
class StateTransition(Batch, Generic[ObservationType, ActionType]):
    observation: ObservationType
    action: ActionType
    next_observation: ObservationType

    # IDEA: Instead of creating extra properties like this, we could have fields
    # like 'field(aliases="bob")', and getattr and setattr would get/set the
    # corresponding attribute when an alias is used instead of the actual name.
    @property
    def state(self) -> ObservationType:
        return self.observation
    
    @property
    def next_state(self) -> ObservationType:
        return self.next_observation


# By default, the PolicyEnv will yield this kind of item:
DefaultDatasetItem = Tuple[StateTransition, RewardType]

def default_dataset_item_creator(observations: ObservationType,
                                 actions: ActionType,
                                 next_observations: ObservationType,
                                 rewards: RewardType,
                                 done: bool,
                                 info: Dict = None) -> DefaultDatasetItem:
    """Create an item of the IterableDataset below, given the current 'context'.

    Parameters
    ----------
    observations : Observations
        The 'starting' observations/state.
    actions : Actions
        The actions that were taken in the 'starting' state.
    next_observations : Observations
        The resulting observations in the 'end' state.
    rewards : Rewards
        The reward associated with that state transition and action.
    done : bool
        Wether the 'end' observations/state are the last of an episode.
    info : Dict, optional
        Info dict associated with the 'next' observation, by default None.

    Returns
    -------
    Tuple[StateTransition, Rewards]
        A Tuple of the form
        `Tuple[Tuple[Observations, Actions, Observations], Rewards]`.
    
    NOTE: `done` and `info` aren't used here, but you could use them in your own
    version of this function that you'd then pass to the PolicyEnv constructor
    or to the `set_policy` method.
    """
    state_transition = StateTransition(observations, actions, next_observations)
    return state_transition, rewards


class PolicyEnv(gym.Wrapper, IterableDataset, Iterable[DatasetItem]):
    """ Wrapper for an environment that adds the following capabilities:
    1. Makes it possible to call step(None), in which case the policy will be
       used to determine the action to take given the current observation and
       the action space.
    2. Creates an 'IterableDataset' from the env, where one iteration over the
       dataset is equivalent to one episode/trajectory in the environment.

       The types of items yielded by this iterator can be customized by passing
       a different callable to `make_dataset_item`.
       The default items are of type `Tuple[StateTransition, Rewards]`, where
       `StateTransition` is a tuple-like object of the form
       `Tuple<observations, actions, next_observations>`.
    """
    def __init__(self,
                 env: Environment[ObservationType, ActionType, RewardType],
                 policy: Optional[Callable[[Tuple], Any]] = None,
                 make_dataset_item: DatasetItemCreator = default_dataset_item_creator):
        super().__init__(env)
        self.make_dataset_item = make_dataset_item
        self.policy = policy
        self._step_result: Optional[StepResult] = None
        self._closed = False
        self._reset = False
        self._n_episodes: int = 0
        self._n_steps: int = 0
        self._n_steps_in_episode: int = 0
        self._observation: Optional[Observations] = None
        self._action: Optional[Actions] = None

    def set_policy(self, policy: Callable[[ObservationType, gym.Space], ActionType]) -> None:
        """ Sets a new policy to be used to generate missing actions. """
        self.policy = policy

    def step(self, action: Optional[Any] = None) -> StepResult:
        if action is None:
            if self.policy is None:
                raise RuntimeError("Need to have a policy set, since action is None.")
            if self._observation is None:
                raise RuntimeError("Reset should have been called before calling step")
            # Get the 'filler' action using the current policy.
            action = self.policy(self._observation, self.action_space)
            if action not in self.action_space:
                raise RuntimeError(f"The policy returned an action which isn't "
                                   f"in the action space: {action}")
        step_result = StepResult(*self.env.step(action))
        self._observation = step_result[0]
        self._n_steps += 1
        self._n_steps_in_episode += 1
        return step_result

    def close(self) -> None:
        self.env.close()
        self._reset = False
        self._closed = True
        self._observation = None

    def reset(self, *args, **kwargs) -> None:
        self._observation = self.env.reset(*args, **kwargs)
        self._reset = True
        self._n_steps_in_episode = 0
        return self._observation

    def __iter__(self) -> Iterator[DatasetItem]:
        """Iterator for an episode/trajectory in the env.
        
        This uses the policy to iteratively perform an episode in the env, and
        yields items at each step, which are the result of the
        `make_dataset_item` function. By default, these items are of the form
        `Tuple<Tuple<observations, actions, next_observation>, rewards>`.
        
        Returns
        -------
        Iterable[DatasetItem]
            Iterable for a 'trajectory' in the env.

        Yields
        -------
        DatasetItem
            The result of `make_dataset_item(current_context)`, by default a
            tuple of <StateTransition, RewardType>.

        Raises
        ------
        RuntimeError
            If no policy is set.
        """
        if not self.policy:
            raise RuntimeError("Need to have a policy set in order to iterate "
                               "on this env.")

        if not self._reset:
            # Reset the env, if needed.
            previous_observations = self.reset()
        else:
            # The env was just reset, so the observation was set to
            # self._observation.
            assert self._observation is not None
            previous_observations = self._observation

        logger.debug(f"Start of episode {self._n_episodes}")

        done = False
        while not done:
            logger.debug(f"steps (episode): {self._n_steps_in_episode}, total: {self._n_steps}")
            # Get the batch of actions using the policy.
            actions = self.policy(previous_observations, self.action_space)

            observations, rewards, done, info = self.step(actions)

            # TODO: Need to figure out what to yield here..
            yield self.make_dataset_item(
                observations=previous_observations,
                actions=actions,
                next_observations=observations,
                rewards=rewards,
                done=done,
                info=info,
            )
            # Update the 'previous' observation.
            previous_observations = observations
            
            if not isinstance(done, bool):
                if any(done):
                    raise RuntimeError(
                        "done should either be a bool or always false, since "
                        "we can't do partial resets."
                    )
                done = False

            self._n_episodes += 1

        logger.debug(f"Episode has ended.")
        self._reset = False
        
    