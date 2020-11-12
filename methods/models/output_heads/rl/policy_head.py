import dataclasses
import itertools
from dataclasses import dataclass
from typing import Dict, Tuple, Union, List, Optional, TypeVar, Iterable
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
from methods.models.forward_pass import ForwardPass 
from utils.utils import prod
from utils.logging_utils import get_logger
from settings.base.objects import Actions, Observations, Rewards
from settings.active.rl.continual_rl_setting import ContinualRLSetting

from ..classification_head import ClassificationOutput, ClassificationHead
from ..output_head import OutputHead
logger = get_logger(__file__)

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

        # Push stuff to the buffers for each environment.
        # Go from "tuples of lists" to "list of tuples":
        per_env_observation = observations.as_list_of_tuples()
        per_env_action = actions.as_list_of_tuples()
        per_env_reward = rewards.as_list_of_tuples()
        per_env_items = zip(per_env_observation, per_env_action, per_env_reward)

        for i, env_items in enumerate(per_env_items):
            # Append the most recent elements into the buffer for that environment.
            self.episode_buffers[i].append(env_items)

        # Get a loss per environment:
        # NOTE: The buffers for each environment will most likely have different
        # lengths!

        total_loss = Loss(self.name)
        for env_index, (episode_buffer, episode_ended) in enumerate(zip(self.episode_buffers, done)):
            ## Retrieve and re-arrange the items from the buffer before they get
            ## passed to the get_episode_loss method. 
            # The buffer is a list of tuples, so we first 'split' those into a
            n_steps = len(episode_buffer)
            # tuple of lists.
            items: Tuple[List] = tuple_of_lists(episode_buffer)
            
            # We stored three items at each step, so we get three lists of tuples
            env_observations_list, env_actions_list, env_rewards_list = items
            
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
                # Clear the buffer if the episode ended.
                episode_buffer.clear()
        
        if not isinstance(total_loss.loss, Tensor):
            assert total_loss.loss == 0.
            total_loss.loss = torch.zeros(1, requires_grad=True)
        return total_loss

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
        
        assert False, (actions, rewards)
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