""" Defines a (hopefully general enough) Output Head class to be used by the
BaselineMethod when applied on an RL setting.

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
is passed a boolean `done` that indicates wether the last
items in the sequences it received mark the end of the episode.

TODO: My hope is that this will allow us to implement RL methods that
need a complete episode in order to give a loss to train with, as well
as methods (like A2C, I think) which can give a Loss even when the
episode isn't over yet.

Also, standard supervised learning could be recovered by setting the
maximum length of the 'episode buffer' to 1, and consider all
observations as final, i.e., when episode length == 1
"""

import dataclasses
import itertools
from abc import ABC, abstractmethod
from collections import deque, namedtuple
from dataclasses import dataclass
from typing import (Any, Deque, Dict, Iterable, List, MutableSequence,
                    NamedTuple, Optional, Sequence, Tuple, TypeVar, Union)

import gym
import numpy as np
import torch
from gym import Space, spaces
from gym.spaces.utils import flatdim
from gym.vector.utils.numpy_utils import concatenate, create_empty_array
from simple_parsing import list_field
from torch import LongTensor, Tensor, nn
from torch.distributions import Distribution
from torch.optim.optimizer import Optimizer

from sequoia.common import Loss, Metrics
from sequoia.common.layers import Lambda
from sequoia.common.metrics.rl_metrics import (EpisodeMetrics,
                                               GradientUsageMetric, RLMetrics)
from sequoia.methods.models.forward_pass import ForwardPass
from sequoia.settings.active.continual import ContinualRLSetting
from sequoia.settings.base.objects import Actions, Observations, Rewards
from sequoia.utils.categorical import Categorical
from sequoia.utils.generic_functions import detach, get_slice, set_slice, stack
from sequoia.utils.logging_utils import get_logger
from sequoia.utils.utils import flag, prod
from ..classification_head import ClassificationHead, ClassificationOutput
from ..output_head import OutputHead

logger = get_logger(__file__)
T = TypeVar("T")



@dataclass(frozen=True)
class PolicyHeadOutput(ClassificationOutput):
    """ WIP: Adds the action pdf to ClassificationOutput. """
    
    # The distribution over the actions, either as a single
    # (batched) distribution or as a list of distributions, one for each
    # environment in the batch.
    action_dist: Categorical

    @property
    def y_pred_prob(self) -> Tensor:
        """ returns the probabilities for the chosen actions/predictions. """
        return self.action_dist.probs(self.y_pred)

    @property
    def y_pred_log_prob(self) -> Tensor:
        """ returns the log probabilities for the chosen actions/predictions. """
        return self.action_dist.log_prob(self.y_pred)

    @property
    def action_log_prob(self) -> Tensor:
        return self.y_pred_log_prob
    
    @property
    def action_prob(self) -> Tensor:
        return self.y_pred_log_prob



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
    """ [WIP] Output head for RL settings.
    
    Uses the REINFORCE algorithm to calculate its loss. 
    
    TODOs/issues:
    - Only currently works with batch_size == 1
    - The buffers are common to training/validation/testing atm..
    
    """

    @dataclass
    class HParams(ClassificationHead.HParams):
        hidden_layers: int = 0
        hidden_neurons: List[int] = list_field()
        # The discount factor for the Return term.
        gamma: float = 0.99
        
        # The maximum length of the buffer that will hold the most recent
        # states/actions/rewards of the current episode. When a batched
        # environment is used
        max_episode_window_length: int = 1000
        
        # Minumum number of epidodes that need to be completed in each env
        # before we update the parameters of the output head.
        min_episodes_before_update: int = 1

        # TODO: Add this mechanism, so that this method could work even when
        # episodes are very long.
        max_steps_between_updates: Optional[int] = None

        # NOTE: Here we have two options:
        # 1- `True`: sum up all the losses and do one larger backward pass,
        # and have `retrain_graph=False`, or
        # 2- `False`: Perform multiple little backward passes, one for each
        # end-of-episode in a single env, w/ `retain_graph=True`.
        # Option 1 is maybe more performant, as it might only require
        # unrolling the graph once, but would use more memory to store all the
        # intermediate graphs.
        accumulate_losses_before_backward: bool = flag(True)

    def __init__(self,
                 input_space: spaces.Space,
                 action_space: spaces.Discrete,
                 reward_space: spaces.Box,
                 hparams: "PolicyHead.HParams" = None,
                 name: str = "policy"):
        assert isinstance(input_space, spaces.Box), f"Only support Tensor (box) input space. (got {input_space})."
        assert isinstance(action_space, spaces.Discrete), f"Only support discrete action space (got {action_space})."
        assert isinstance(reward_space, spaces.Box), f"Reward space should be a Box (scalar rewards) (got {reward_space})."
        super().__init__(
            input_space=input_space,
            action_space=action_space,
            reward_space=reward_space,
            hparams=hparams,
            name=name,
        )
        logger.info("Output head hparams: " + self.hparams.dumps_json(indent='\t'))
        self.hparams: PolicyHead.HParams
        # Type hints for the spaces;    
        self.input_space: spaces.Box
        self.action_space: spaces.Discrete
        self.reward_space: spaces.Box

        # List of buffers for each environment that will hold some items.
        # TODO: Won't use the 'observations' anymore, will only use the
        # representations from the encoder, so renaming 'representations' to
        # 'observations' in this case.
        # (Should probably come up with another name so this isn't ambiguous).
        # TODO: Perhaps we should register these as buffers so they get
        # persisted correclty? But then we also need to make sure that the grad
        # stuff would work the same way..
        self.representations: List[Deque[Tensor]] = []
        # self.representations: List[deque] = []
        self.actions: List[Deque[PolicyHeadOutput]] = []
        self.rewards: List[Deque[ContinualRLSetting.Rewards]] = []

        # The actual "internal" loss we use for training.
        self.loss: Loss = Loss(self.name, metrics={self.name: RLMetrics()})
        self.batch_size: int = 0

        self.num_episodes_since_update: np.ndarray = np.zeros(1)
        self.num_steps_in_episode: np.ndarray = np.zeros(1)

        self._training: bool = True

    def create_buffers(self):
        """ Creates the buffers to hold the items from each env. """
        logger.debug(f"Creating buffers (batch size={self.batch_size})")
        logger.debug(f"Maximum buffer length: {self.hparams.max_episode_window_length}")
        
        self.representations = self._make_buffers()
        self.actions = self._make_buffers()
        self.rewards = self._make_buffers()
        
        self.num_steps_in_episode = np.zeros(self.batch_size, dtype=int)
        self.num_episodes_since_update = np.zeros(self.batch_size, dtype=int)

    def forward(self, observations: ContinualRLSetting.Observations, representations: Tensor) -> PolicyHeadOutput:
        """ Forward pass of a Policy head.

        TODO: Do we actually need the observations here? It is here so we have
        access to the 'done' from the env, but do we really need it here? or
        would there be another (cleaner) way to do this?
        """        
        if len(representations.shape) < 2:
            # Flatten the representations.
            representations = representations.reshape([-1, flatdim(self.input_space)])
        
        # Setup the buffers, which will hold the most recent observations,
        # actions and rewards within the current episode for each environment.
        if not self.batch_size:
            self.batch_size = representations.shape[0]
            self.create_buffers()
            
        representations = representations.float()
        
        logits = self.dense(representations)
        
        # The policy is the distribution over actions given the current state.
        action_dist = Categorical(logits=logits)
        sample = action_dist.sample()
        actions = PolicyHeadOutput(
            y_pred=sample,
            logits=logits,
            action_dist=action_dist,
        )
        return actions

    def get_loss(self,
                 forward_pass: ForwardPass,
                 actions: PolicyHeadOutput,
                 rewards: ContinualRLSetting.Rewards) -> Loss:
        """ Given the forward pass, the actions produced by this output head and
        the corresponding rewards for the current step, get a Loss to use for
        training.
        
        TODO: Replace the `forward_pass` argument with just `observations` and
        `representations` and provide the right (augmented) observations to the
        aux tasks. (Need to design that part later).
        """
        observations: ContinualRLSetting.Observations = forward_pass.observations
        representations: Tensor = forward_pass.representations
        if self.batch_size is None:
            assert len(representations.shape) == 2, (
                f"Need batched representations, with a shape [16, 128] or similar, but "
                f"representations have shape {representations.shape}."
            )
            self.batch_size = representations.shape[0]
            self.create_buffers()
        
        if not self.hparams.accumulate_losses_before_backward:
            # Reset the loss for the current step, if we're not accumulating it.
            self.loss = Loss(self.name)

        observations = forward_pass.observations
        representations = forward_pass.representations
        assert observations.done is not None, "need the end-of-episode signal"

        # Calculate the loss for each environment.
        for env_index, done in enumerate(observations.done):
            if done:
                # End of episode reached in that env!
                self.num_episodes_since_update[env_index] += 1
                self.num_steps_in_episode[env_index] = 0

            env_loss = self.get_episode_loss(env_index, done=done)
            
            if env_loss is not None:
                self.loss += env_loss

            if done:
                assert env_loss is not None
                self.clear_buffers(env_index)
            

        for env_index in range(self.batch_size):
            # Take a slice across the first dimension
            # env_observations = get_slice(observations, env_index)
            env_representations = representations[env_index]
            env_actions = actions.slice(env_index)
            # env_actions = actions[env_index, ...] # TODO: Is this nicer?
            env_rewards = rewards.slice(env_index)
            self.representations[env_index].append(env_representations)
            self.actions[env_index].append(env_actions)
            self.rewards[env_index].append(env_rewards)
        
        self.num_steps_in_episode += 1
        # TODO:
        # If we want to accumulate the losses before backward, then we just return self.loss
        # If we DONT want to accumulate the losses before backward, then we do the
        # 'small' backward pass, and return a detached loss.
        if self.hparams.accumulate_losses_before_backward:
            if all(self.num_episodes_since_update >= self.hparams.min_episodes_before_update):
                # Every environment has seen the required number of episodes.    
                # We return the accumulated loss, so that the model can do the backward
                # pass and update the weights.
                returned_loss = self.loss
                self.loss = Loss(self.name)
                self.detach_all_buffers()
                self.num_episodes_since_update[:] = 0            
                return returned_loss
            else:
                return Loss(self.name)
        else:
            # Perform the backward pass as soon as a loss is available (with
            # retain_graph=True).
            if all(self.num_episodes_since_update >= self.hparams.min_episodes_before_update):
                # Every environment has seen the required number of episodes.    
                # We return the loss for this step, with gradients, to indicate to the
                # Model that it can perform the backward pass and update the weights.
                returned_loss = self.loss
                self.loss = Loss(self.name)
                self.detach_all_buffers()
                self.num_episodes_since_update[:] = 0            
                return returned_loss

            elif self.loss.requires_grad:
                # Not all environments are done, but we have a Loss from one of them.
                self.loss.backward(retain_graph=True)
                # self.loss will be reset at each step in the `forward` method above.
                return self.loss.detach()
            
            else:
                # TODO: Why is self.loss non-zero here?
                if self.loss.loss != 0.:
                    # TODO: This is a weird edge-case, where at least one env produced
                    # a loss, but that loss doesn't require grad.
                    # This should only happen if the model isn't in training mode, for
                    # instance.
                    assert not self.training, self.loss
                    # return self.loss
                return self.loss
        assert False, f"huh? {self.loss}"
        return self.loss
                           
    def get_episode_loss(self,
                         env_index: int,
                         done: bool) -> Optional[Loss]:
        """Calculate a loss to train with, given the last (up to
        max_episode_window_length) observations/actions/rewards of the current
        episode in the environment at the given index in the batch.

        If `done` is True, then this is for the end of an episode. If `done` is
        False, the episode is still underway.

        NOTE: While the Batch Observations/Actions/Rewards objects usually
        contain the "batches" of data coming from the N different environments,
        now they are actually a sequence of items coming from this single
        environment. For more info on how this is done, see the  
        """
        inputs: Tensor
        actions: PolicyHeadOutput
        rewards: ContinualRLSetting.Rewards
        if not done:
            # This particular algorithm (REINFORCE) can't give a loss until the
            # end of the episode is reached.
            return None

        if len(self.actions[env_index]) == 0:
            logger.debug(f"Weird, asked to get episode loss, but there is "
                         f"nothing in the buffer?")
            return None

        inputs, actions, rewards = self.stack_buffers(env_index)

        episode_length = actions.batch_size
        assert len(inputs) == len(actions.y_pred) == len(rewards.y)

        if episode_length <= 1:
            # TODO: If the episode has len of 1, we can't really get a loss!
            logger.error("Episode is too short!")
            return None

        log_probabilities = actions.y_pred_log_prob
        rewards = rewards.y
        
        loss = self.policy_gradient(
            rewards=rewards,
            log_probs=log_probabilities,
            gamma=self.hparams.gamma,
        )
        
        gradient_usage = self.get_gradient_usage_metrics(env_index)
        # TODO: Add 'Metrics' for each episode?
        return Loss(self.name, loss, metrics={
            self.name: RLMetrics(episodes=[EpisodeMetrics(rewards=rewards)]),
            "gradient_usage": gradient_usage
        })

    def get_gradient_usage_metrics(self, env_index: int) -> GradientUsageMetric:
        """ Returns a Metrics object that describes how many of the actions
        from an episode that are used to calculate a loss still have their
        graphs, versus ones that don't have them (due to being created before
        the last model update, and therefore having been detached.)

        Does this by inspecting the contents of `self.actions[env_index]`. 
        """
        episode_actions = self.actions[env_index]
        n_stored_items = len(self.actions[env_index])
        n_items_with_grad = sum(v.logits.requires_grad for v in episode_actions)
        n_items_without_grad = n_stored_items - n_items_with_grad
        return GradientUsageMetric(
            used_gradients=n_items_with_grad,
            wasted_gradients=n_items_without_grad,
        )

    @staticmethod
    def get_returns(rewards: Union[Tensor, List[Tensor]], gamma: float) -> Tensor:
        """ Calculates the returns, as the sum of discounted future rewards at
        each step.
        """
        T = len(rewards)
        if not isinstance(rewards, Tensor):
            rewards = torch.as_tensor(rewards)
        # Construct a reward matrix, with previous rewards masked out (with each
        # row as a step along the trajectory).
        reward_matrix = rewards.expand([T, T]).triu()
        # Get the gamma matrix (upper triangular), see make_gamma_matrix for
        # more info.
        gamma_matrix = make_gamma_matrix(gamma, T, device=reward_matrix.device)
        # Multiplying by the gamma coefficients gives the discounted rewards.
        discounted_rewards = (reward_matrix * gamma_matrix)
        # Summing up over time gives the return at each step.
        return discounted_rewards.sum(-1)

    @staticmethod
    def policy_gradient(rewards: List[float], log_probs: Union[Tensor, List[Tensor]], gamma: float=0.95):
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
            The "vanilla policy gradient" / REINFORCE gradient resulting from
            that episode.
        """
        if isinstance(log_probs, Tensor):
            action_log_probs = log_probs
        else:
            action_log_probs = torch.stack(log_probs)
        reward_tensor = torch.as_tensor(rewards).type_as(action_log_probs)

        returns = PolicyHead.get_returns(reward_tensor, gamma=gamma)
        policy_gradient = - action_log_probs.dot(returns)
        return policy_gradient

    @property
    def training(self) -> bool:
        return self._training

    @training.setter
    def training(self, value: bool) -> None:
        # logger.debug(f"setting training to {value} on the Policy output head")
        if hasattr(self, "_training") and value != self._training:
            before = "train" if self._training else "test"
            after = "train" if value else "test"
            logger.debug(f"Clearing buffers, since we're transitioning between from {before}->{after}")
            self.clear_all_buffers()
            self.batch_size = None
            self.num_episodes_since_update[:] = 0
        self._training = value

    def clear_all_buffers(self) -> None:
        if self.batch_size is None:
            assert not self.rewards
            assert not self.representations
            assert not self.actions
            return
        for env_id in range(self.batch_size):
            self.clear_buffers(env_id)
        self.rewards.clear()
        self.representations.clear()
        self.actions.clear()
        
    def clear_buffers(self, env_index: int) -> None:
        """ Clear the buffers associated with the environment at env_index.
        """
        self.representations[env_index].clear()
        self.actions[env_index].clear()
        self.rewards[env_index].clear()

    def detach_all_buffers(self):
        for env_index in range(self.batch_size):
            self.detach_buffers(env_index)

    def detach_buffers(self, env_index: int) -> None:
        """ Detach all the tensors in the buffers for a given environment.
        
        We have to do this when we update the model while an episode in one of
        the enviroment isn't done.
        """
        # detached_representations = map(detach, )
        # detached_actions = map(detach, self.actions[env_index])
        # detached_rewards = map(detach, self.rewards[env_index])
        self.representations[env_index] = self._detach_buffer(self.representations[env_index])
        self.actions[env_index] = self._detach_buffer(self.actions[env_index])
        self.rewards[env_index] = self._detach_buffer(self.rewards[env_index])
        # assert False, (self.representations[0], self.representations[-1])

    def _detach_buffer(self, old_buffer: Sequence[Tensor]) -> deque:
            new_items = self._make_buffer()
            for item in old_buffer:
                detached = item.detach()
                new_items.append(detached)
            return new_items
    
    def _make_buffer(self, elements: Sequence[T] = None) -> Deque[T]:
        buffer: Deque[T] = deque(maxlen=self.hparams.max_episode_window_length)
        if elements:
            buffer.extend(elements)
        return buffer

    def _make_buffers(self) -> List[deque]:
        return [self._make_buffer() for _ in range(self.batch_size)]

    def stack_buffers(self, env_index: int):
        """ Stack the observations/actions/rewards for this env and return them.
        """
        # episode_observations = tuple(self.observations[env_index])
        episode_representations = tuple(self.representations[env_index])
        episode_actions = tuple(self.actions[env_index])
        episode_rewards = tuple(self.rewards[env_index])
        assert len(episode_representations)
        assert len(episode_actions)
        assert len(episode_rewards)
        stacked_inputs = stack(episode_representations)
        stacked_actions = stack(episode_actions)
        stacked_rewards = stack(episode_rewards)
        return stacked_inputs, stacked_actions, stacked_rewards

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
