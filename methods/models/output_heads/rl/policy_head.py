import dataclasses
import itertools
from dataclasses import dataclass
from typing import Dict, Tuple, Union, List, Optional, TypeVar, Iterable, Any
from abc import abstractmethod, ABC
from collections import deque

import gym
import numpy as np
import torch
from gym import spaces
from torch import LongTensor, Tensor, nn
from torch.distributions import Categorical as Categorical_, Distribution

from common import Loss
from common.layers import Lambda
from common.gym_wrappers.batch_env.worker import FINAL_STATE_KEY
from methods.models.forward_pass import ForwardPass 
from utils.utils import prod
from utils.logging_utils import get_logger
from settings.base.objects import Actions, Observations, Rewards
from settings.active.rl.continual_rl_setting import ContinualRLSetting

from ..classification_head import ClassificationOutput, ClassificationHead
from ..output_head import OutputHead
logger = get_logger(__file__)
from collections import namedtuple

class Categorical(Categorical_):
    def __getitem__(self, index: int) -> "Categorical":
        return Categorical(logits=self.logits[index])
        # return Categorical(probs=self.probs[index])

    def __iter__(self) -> Iterable["Categorical"]:
        for index in range(self.logits.shape[0]):
            yield self[index]

@dataclass(frozen=True)
class PolicyHeadOutput(ClassificationOutput):
    """ WIP: Adds the action pdf to ClassificationOutput. """
    policy: Distribution

    @property
    def y_pred_log_prob(self) -> Tensor:
        """ returns the log probabilities for the chosen actions/predictions. """
        return self.policy.log_prob(self.y_pred)

    @property
    def y_pred_prob(self) -> Tensor:
        """ returns the log probabilities for the chosen actions/predictions. """
        return self.policy.probs(self.y_pred)

# BUG: Since its too complicated to try and get the final state
# from the info dict to look like the observation, I'm just
# going to discard it, and so whenever the get_episode_loss
# function is called, it won't receive the final observation
# in it.

## NOTE: Since the gym VectorEnvs actually auto-reset the individual
## environments (and also discard the final state, for some weird
## reason), I added a way to save it into the 'info' dict at the key
## 'final_state'. Assuming that the env this output head gets applied
## on adds the info dict to the observations (using the
## AddInfoToObservations wrapper, for instance), then the 'final'
## observation would be stored in the dict for this environment in
## the Observations object, while the 'observation' you get from step
## is the 'initial' observation of the new episode.             
 
               
class PolicyHead(ClassificationHead):
    
    @dataclass
    class HParams(ClassificationHead.HParams):
        # The discount factor for the Return term.
        gamma: float = 0.9
        # The maximum length of the buffer that will hold the most recent
        # states/actions/rewards of the current episode. When a batched
        # environment is used
        max_episode_window_length: int = 10
    
    def __init__(self,
                 input_size: int,
                 action_space: spaces.Discrete,
                 reward_space: spaces.Box,
                 hparams: "PolicyHead.HParams" = None,
                 name: str = "policy"):
        assert isinstance(action_space, spaces.Discrete), f"Only support discrete action space for now (got {action_space})."
        assert isinstance(reward_space, spaces.Box), f"Reward space should be a Box (scalar rewards) (got {reward_space})."
        super().__init__(
            input_size=input_size,
            action_space=action_space,
            reward_space=reward_space,
            hparams=hparams,
            name=name,
        )
        if not isinstance(self.hparams, self.HParams):
            current_hparams = self.hparams.to_dict()
            missing_args = [f.name for f in dataclasses.fields(self.HParams)
                            if f.name not in current_hparams]
            logger.warning(RuntimeWarning(
                f"Upgrading the hparams from type {type(self.hparams)} to "
                f"type {self.HParams}, which will use the default values for "
                f"the missing arguments {missing_args}"
            ))
            self.hparams = self.HParams.from_dict(current_hparams)
        self.hparams: PolicyHead.HParams
        self.action_space: spaces.Discrete
        self.reward_spaces: spaces.Box

        self.density: Categorical
        # List of buffers for each environment that will hold some items.
        self.episode_buffers: List[deque] = []
        # List that holds the 'initial' observations, when an episode boundary
        # is reached.
        self.initial_observations: List[Optional[Tuple]] = []

    def forward(self, observations: ContinualRLSetting.Observations, representations: Tensor) -> PolicyHeadOutput:
        """ Forward pass of a Policy head.
        
        NOTE: (@lebrice) This is identical to the forward pass of the
        ClassificationHead, except that the policy is stochastic: the actions
        are sampled from the probabilities described by the logits, rather than
        always selecting the action with the highest logit as in the
        ClassificationHead.
        
        TODO: This is actually more general than a classification head, no?
        Would it make sense to maybe re-order the "hierarchy" of the output
        heads, and make the ClassificationHead inherit from this one?
        """
        # Get the raw / unscaled logits for each action using the
        # ClassificationHead's forward method.

        # NOTE: Not sure if this is that useful.
        # Also, doesn't work on CUDA atm, for some reason.

        # Choose the actions according to their probabilities, rather than just
        # taking the action with highest probability, as is done in the
        # ClassificationHead.
        logits = self.dense(representations)
        # The policy is the distribution over actions given the current state.
        policy = Categorical(logits=logits)
        actions = policy.sample()
        output = PolicyHeadOutput(
            y_pred=actions,
            logits=logits,
            policy=policy,
        )
        return output
    
    def get_loss(self,
                 forward_pass: ForwardPass,
                 actions: PolicyHeadOutput,
                 rewards: ContinualRLSetting.Rewards) -> Loss:
        """ Given the forward pass, the actions produced by this output head and
        the corresponding rewards for the current step, get a Loss to use for
        training.
        
        NOTE: The training procedure is fundamentally on-policy atm, i.e. the
        observation is a single state, not a rollout, and the reward is the
        immediate reward at the current step.
        
        Therefore, what we do here is to first split things up and push the
        observations/actions/rewards into a per-environment buffer, of max
        length `self.hparams.max_episode_window_length`. These buffers get
        cleared when starting a new episode in their corresponding environment.
        
        The contents of this buffer are then rearranged and presented to the
        `get_episode_loss` method in order to get a loss for the given episode.
        The `get_episode_loss` method is also given the environment index, and
        is passed a boolean `episode_ended` that indicates wether the last
        items in the sequences it received mark the end of the episode.
        
        TODO: My hope is that this will allow us to implement RL methods that
        need a complete episode in order to give a loss to train with, as well
        as methods (like A2C, I think) which can give a Loss even when the
        episode isn't over yet.
        
        Also, standard supervised learning could possibly be recovered if you
        set the maximum length of the 'episode buffer' to 1.
        """
        observations: ContinualRLSetting.Observations = forward_pass.observations
        batch_size = observations.batch_size or 1
        # Extract the outputs of this head from the forwardpass object.
        # TODO: Its a bit dumb to have to retrieve our own predictions from the
        # forward pass dict this way. Might be more natural to get passed the
        # actions directly as an argument. That would also make it possible to
        # have more than one 'action' inside the forward pass object.
        actions: PolicyHeadOutput = forward_pass.actions
        rewards: ContinualRLSetting.Rewards
        assert isinstance(observations, ContinualRLSetting.Observations)
        # Note: we need to have a loss for each, here.
        done: Sequence[bool] = observations.done
        
        # Setup the buffers, which will hold the most recent observations,
        # actions and rewards within the current episode for each environment.
        if not self.episode_buffers:
            self.episode_buffers = [
                deque(maxlen=self.hparams.max_episode_window_length)
                for _ in range(batch_size)
            ]

        if not self.initial_observations:
            self.initial_observations = [
                None for _ in range(batch_size)
            ]

        # Split the batches into slices for each environment.
        per_env_observation = observations.as_list_of_tuples()
        per_env_action = actions.as_list_of_tuples()
        per_env_reward = rewards.as_list_of_tuples()

        per_env_items = zip(per_env_observation, per_env_action, per_env_reward)

        # Append the most recent elements into the buffer for that environment.
        for env_index, env_items in enumerate(per_env_items):
            env_observation, env_action, env_reward = env_items
            
            env_episode_buffer: deque = self.episode_buffers[env_index]
            if len(env_episode_buffer) == env_episode_buffer.maxlen:
                # TODO: When the 'initial' observation becomes older than the
                # length of the buffer, what should we do with it?
                # Possibly update the 'initial' observation?
                old_initial_obs = self.initial_observations[env_index]
                new_initial_obs = self.update_initial_observation(
                    env_index=env_index,
                    current_initial_observation=old_initial_obs,
                    env_episode_buffer=env_episode_buffer,
                )
            # We still add the obs, action, reward to the buffer, because we
            # want to keep the action and reward. We'll just discard the
            # observation when re-packing them below.
            self.episode_buffers[env_index].append((env_observation, env_action, env_reward))

        # Get a loss per environment:
        # NOTE: The buffers for each environment will most likely have different
        # lengths!

        total_loss = Loss(self.name)
        
        # Loop over each environment's buffer, retrieve and re-arrange the items
        # from the buffer before they get passed to the get_episode_loss method. 
        for env_index, episode_buffer in enumerate(self.episode_buffers):
            # NOTE: See above, when the done=True with vector envs, the
            # observation is the new initial observation.
            episode_ended = bool(done[env_index])
            
            # The buffer is a list of tuples, so we first 'split' those into a
            # tuple of lists.
            items: Tuple[List] = tuple_of_lists(episode_buffer)

            # We stored three items at each step, so we get three lists of tuples.
            env_observations_list: List[namedtuple] = items[0]
            env_actions_list: List[namedtuple] = items[1]
            env_rewards_list: List[namedtuple] = items[2]
            # env_observations_list, env_actions_list, env_rewards_list = items
            
            # Insert the current 'initial observation' at the start of the list.
            initial_observation = self.initial_observations[env_index]
            if initial_observation is not None:
                env_observations_list.insert(0, initial_observation)

            if episode_ended:
                # Safeguard the initial observation for that environment, and
                # also remove it from the list so it isn't passed to
                # get_episode_loss.
                # TODO: The 'initial' observation now has 'done'=True, and the
                # last one in the list doesn't!
                initial_observation = env_observations_list.pop(-1)
                initial_observation = initial_observation._replace(done=False)
                
                self.initial_observations[env_index] = initial_observation
            
            # Re-package the items into their original 'Batch' objects, but now
            # the 'batch' will be a sequence of items from the same environment.
            env_observations = tuple_of_lists(env_observations_list)
            env_actions = tuple_of_lists(env_actions_list)
            env_rewards = tuple_of_lists(env_rewards_list)
            
            # 'Stack' the items back into their original 'Batch' types.
            env_observations: ContinualRLSetting.Observations = type(observations).from_inputs(env_observations)             
            env_actions: ContinualRLSetting.Actions = type(actions).from_inputs(env_actions)             
            env_rewards: ContinualRLSetting.Rewards = type(rewards).from_inputs(env_rewards)
            
            
            loss = self.get_episode_loss(
                env_index=env_index,
                observations=env_observations,
                actions=env_actions,
                rewards=env_rewards,
                episode_ended=episode_ended
            )

            # If we are able to get a loss for this set of observations/actions/
            # rewards, then add it to the total loss.
            if loss is not None:
                total_loss += loss
            
            if episode_ended:
                # Clear the buffer if the episode ended. NOTE: the initial obs
                # are saved in self.initial_observations.
                episode_buffer.clear()
        
        if not isinstance(total_loss.loss, Tensor):
            assert total_loss.loss == 0.
            total_loss.loss = torch.zeros(1, requires_grad=True)
        return total_loss

    def update_initial_observation(self, 
                                   env_index: int,
                                   current_initial_observation: Any,
                                   env_episode_buffer: List[Tuple[Observations, Actions, Rewards]]):
        """ Update the 'initial observation', when the episode is longer than
        the max buffer size. We could possibly use this in the future maybe.
        
        Returns None for now.
        """
        return None
    
    def get_episode_loss(self,
                         env_index: int,
                         observations: ContinualRLSetting.Observations,
                         actions: ContinualRLSetting.Observations,
                         rewards: ContinualRLSetting.Rewards,
                         episode_ended: bool) -> Optional[Loss]:
        """Calculate a loss to train with, given the last (up to
        max_episode_window_length) observations/actions/rewards of the current
        episode in the environment at the given index in the batch.

        NOTE: While the Batch Observations/Actions/Rewards objects usually
        contain the "batches" of data coming from the N different environments,
        now they are actually a sequence of items coming from this single
        environment. For more info on how this is done, see the  
        """
        if not episode_ended:
            # This particular algorithm (REINFORCE) can't give a loss until the
            # end of the episode is reached.
            return None
        log_probabilities = torch.stack(
            [action.policy.log_prob(action.y_pred) for action in actions]
        )
        rewards = torch.stack([reward[0] for reward in rewards])
        loss = self.policy_gradient_optimized(rewards=rewards, log_probs=log_probabilities)
        
        # TODO: Add 'Metrics' for each episode?
        return Loss("policy_gradient", loss)

    def policy_gradient_optimized(self, rewards: List[float], log_probs: List[Tensor]):
        """Implementation of the REINFORCE algorithm.

        Adapted from https://medium.com/@thechrisyoon/deriving-policy-gradients-and-implementing-reinforce-f887949bd63

        Parameters
        ----------
        - episode_rewards : List[Tensor]

            The rewards at each step in an episode

        - episode_log_probs : List[Tensor]

            The log probabilities associated with the actions that were taken at
            each step.

        Returns
        -------
        Tensor
            The "policy gradient" resulting from that episode.
        """
        T = len(rewards)
        if not isinstance(log_probs, Tensor):
            log_probs = torch.stack(log_probs)
        rewards = torch.as_tensor(rewards).type_as(log_probs)
        # Construct a reward matrix, with previous rewards masked out (with each
        # row as a step along the trajectory).
        reward_matrix = rewards.expand([T, T]).triu()
        # Get the gamma matrix (upper triangular), see make_gamma_matrix for
        # more info.
        gamma_matrix = make_gamma_matrix(self.hparams.gamma, T, device=log_probs.device)

        discounted_rewards = (reward_matrix * gamma_matrix).sum(dim=-1)
        # normalize discounted rewards
        discounted_rewards = normalize(discounted_rewards)
        policy_gradient = - log_probs.dot(discounted_rewards)
        return policy_gradient

# @torch.jit.script
# @lru_cache()
def make_gamma_matrix(gamma: float, T: int, device=None) -> Tensor:
    """
    Create an upper-triangular matrix [T, T] with the gamma factors,
    starting at 1.0 on the diagonal, and decreasing exponentially towards
    the right.
    """
    gamma_matrix = torch.empty([T, T]).triu_()
    # Neat indexing trick to fill up the upper triangle of the matrix:
    rows, cols = torch.triu_indices(T, T)
    # Precompute all the powers of gamma in range [0, T]
    all_gammas = gamma ** torch.arange(T)
    # Put the right value at each entry in the upper triangular matrix.
    gamma_matrix[rows, cols] = all_gammas[cols - rows]
    return gamma_matrix.to(device) if device else gamma_matrix

def normalize(x: Tensor):
    return (x - x.mean()) / (x.std() + 1e-9)

T = TypeVar("T")

def tuple_of_lists(list_of_tuples: List[Tuple[T, ...]]) -> Tuple[List[T], ...]:
    return tuple(map(list, zip(*list_of_tuples)))

def list_of_tuples(tuple_of_lists: Tuple[List[T], ...]) -> List[Tuple[T, ...]]:
    return list(zip(*tuple_of_lists))