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
from dataclasses import dataclass
from typing import Dict, Tuple, Union, List, Optional, TypeVar, Iterable, Any, Sequence, NamedTuple
from abc import abstractmethod, ABC
from collections import namedtuple, deque, OrderedDict

import gym
import numpy as np
import torch
from gym import spaces, Space
from gym.spaces.utils import flatdim
from gym.vector.utils.numpy_utils import concatenate, create_empty_array
from torch import LongTensor, Tensor, nn
from torch.distributions import Categorical as Categorical_, Distribution

from sequoia.common.metrics.rl_metrics import EpisodeMetrics, RLMetrics, GradientUsageMetric
from sequoia.common import Loss, Metrics
from sequoia.common.layers import Lambda
from sequoia.methods.models.forward_pass import ForwardPass 
from sequoia.utils.utils import prod
from sequoia.utils.logging_utils import get_logger
from sequoia.utils.generic_functions import stack, detach, get_slice, set_slice
from sequoia.settings.base.objects import Actions, Observations, Rewards
from sequoia.settings.active.rl.continual_rl_setting import ContinualRLSetting

from ..classification_head import ClassificationOutput, ClassificationHead
from ..output_head import OutputHead
logger = get_logger(__file__)


class Categorical(Categorical_):
    """ Simple little addition to the Categorical class, allowing it to be 'split'
    into a sequence of distributions (to help with the splitting in the output
    head below)
    """ 
    def __getitem__(self, index: int) -> "Categorical":
        return Categorical(logits=self.logits[index])
        # return Categorical(probs=self.probs[index])

    def __iter__(self) -> Iterable["Categorical"]:
        for index in range(self.logits.shape[0]):
            yield self[index]


@detach.register
def detach_categorical(v: Categorical) -> Categorical:
    return type(v)(logits=v.logits.detach())


@dataclass(frozen=True)
class PolicyHeadOutput(ClassificationOutput):
    """ WIP: Adds the action pdf to ClassificationOutput. """
    
    # The Policy, as a distribution over the actions, either as a single
    # (batched) distribution or as a list of distributions, one for each
    # environment in the batch. 
    policy: Union[Distribution, List[Distribution]]

    @property
    def y_pred_log_prob(self) -> Tensor:
        """ returns the log probabilities for the chosen actions/predictions. """
        if isinstance(self.policy, list):
            return torch.stack([
                policy.log_prob(y_pred)
                for policy, y_pred in zip(self.policy, self.y_pred)
            ])
        return self.policy.log_prob(self.y_pred)

    @property
    def y_pred_prob(self) -> Tensor:
        """ returns the log probabilities for the chosen actions/predictions. """
        if isinstance(self.policy, list):
            return torch.stack([
                policy.probs(y_pred)
                for policy, y_pred in zip(self.policy, self.y_pred)
            ])
        return self.policy.probs(self.y_pred)


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
        # The discount factor for the Return term.
        gamma: float = 0.9
        # The maximum length of the buffer that will hold the most recent
        # states/actions/rewards of the current episode. When a batched
        # environment is used
        max_episode_window_length: int = 1000
        
        # Minumum number of epidodes that need to be completed in each env
        # before we update the parameters of the output head.
        min_episodes_before_update: int = 1

    def __init__(self,
                 input_space: spaces.Space,
                 action_space: spaces.Discrete,
                 reward_space: spaces.Box,
                 hparams: "PolicyHead.HParams" = None,
                 name: str = "policy"):
        assert isinstance(action_space, spaces.Discrete), f"Only support discrete action space for now (got {action_space})."
        assert isinstance(reward_space, spaces.Box), f"Reward space should be a Box (scalar rewards) (got {reward_space})."
        super().__init__(
            input_space=input_space,
            action_space=action_space,
            reward_space=reward_space,
            hparams=hparams,
            name=name,
        )
        if not isinstance(self.hparams, self.HParams):
            # NOTE: This (getting the wrong hparams class) could happen for
            # instance when parsing a BaselineMethod from the command-line, the
            # default type of hparams on the method is BaselineModel.HParams,
            # which the `output_head` field doesn't have the right type exactly. 
            current_hparams = self.hparams.to_dict()
            missing_fields = [f.name for f in dataclasses.fields(self.HParams)
                            if f.name not in current_hparams]
            logger.warning(RuntimeWarning(
                f"Upgrading the hparams from type {type(self.hparams)} to "
                f"type {self.HParams}. This will try to fetch the values for "
                f"the missing fields {missing_fields} from the command-line. "
            ))
            # Get the missing values
            hparams = self.HParams.from_args(strict=False)
            for missing_field in missing_fields:
                current_hparams[missing_field] = getattr(hparams, missing_field) 
            self.hparams = self.HParams.from_dict(current_hparams)

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
        self.representations: List[deque] = []
        # self.representations: List[deque] = []
        self.actions: List[deque] = []
        self.rewards: List[deque] = []

        # The actual "internal" loss we use for training.
        self.loss: Loss = Loss(self.name, metrics={self.name: RLMetrics()})
        # The loss we expose and return in `get_loss` ( has all metrics etc.)
        self.detached_loss: Loss = self.loss.detach()

        self.batch_size: int = 0
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        
        self.has_reached_episode_end: Sequence[bool] = []
        self.num_episodes_since_update: Sequence[int] = []
        
        # See the two 'options' described below.
        self.option: int = 1
        
        
    def create_buffers(self):
        """ Creates the buffers to hold the items from each env. """
        logger.debug(f"Creating buffers (batch size={self.batch_size})")
        logger.debug(f"Maximum buffer length: {self.hparams.max_episode_window_length}")
        
        self.representations = self._make_buffers()
        self.actions = self._make_buffers()
        self.rewards = self._make_buffers()

        self.has_reached_episode_end = np.zeros(self.batch_size, dtype=bool)
        self.num_episodes_since_update = np.zeros(self.batch_size, dtype=int)
        self.num_detached_tensors = np.zeros(self.batch_size, dtype=int)
        self.num_grad_tensors = np.zeros(self.batch_size, dtype=int)
        self.metrics_per_env: List[RLMetrics] = []
        self.optimizer.zero_grad()
 
    def forward(self, observations: ContinualRLSetting.Observations, representations: Tensor) -> PolicyHeadOutput:
        """ Forward pass of a Policy head.

        TODO: Do we actually need the observations here? It is here so we have
        access to the 'done' from the env, but do we really need it here? or
        would there be another (cleaner) way to do this?

        NOTE: (@lebrice) This is identical to the forward pass of the
        ClassificationHead, except that the policy is stochastic: the actions
        are sampled from the probabilities described by the logits, rather than
        always selecting the action with the highest logit as in the
        ClassificationHead.
        
        TODO: This is actually more general than a classification head, no?
        Would it make sense to maybe re-order the "hierarchy" of the output
        heads, and make the ClassificationHead inherit from this one?
        """
        
        
        # TODO: (@lebrice) Need to calculate the loss HERE if the episode is
        # done, i.e. before doing the next forward pass!! (otherwise the actions
        # for the initial observation are based on the previous weights, which
        # won't be at the right version when we will backpropagate the loss for
        # the next episode, if it goes back until the initial observation!
        # Problem is, we would also need to backpropagate that loss BEFORE we
        # can do the forward pass on the new observations! Otherwise the first
        # actions won't have been based on the right weights!
        
        if len(representations.shape) < 2:
            representations = representations.reshape([-1, flatdim(self.input_space)])
        
        # Setup the buffers, which will hold the most recent observations,
        # actions and rewards within the current episode for each environment.
        if not self.batch_size:
            self.batch_size = representations.shape[0]
            self.create_buffers()

        self.detached_loss = Loss(self.name)

        # NOTE: This means we also need it at test-time though.
        assert observations.done is not None, "need the end-of-episode signal"
        if self.training:
            for env_index, done in enumerate(observations.done):
                if done:
                    # End of episode reached in that env!
                    self.has_reached_episode_end[env_index] = True
                    self.num_episodes_since_update[env_index] += 1

                env_loss = self.get_episode_loss(env_index, done=done)

                if done:
                    self.clear_buffers(env_index)

                if env_loss is None:
                    continue

                # TODO: Here we have two options: Either we:
                # (1): sum up all the losses and do one larger backward pass,
                # and have `retrain_graph=False`, or
                # (2): we do multiple little backward passes as soon as an
                # episode end is reached in a single env, w/ `retain_graph=True`.
                # Option 1 is maybe more performant, as it might only require
                # unrolling the graph once, but would use more memory.
                if self.option == 1:
                    # Option 1:
                    self.loss += env_loss
                    self.detached_loss += env_loss.detach()
                else:
                    # Option 2:
                    env_loss.backward(retain_graph=True)
                    # # Detach the loss so we can keep the metrics but not all the graphs.
                    self.detached_loss += env_loss.detach()

            if all(self.num_episodes_since_update >= self.hparams.min_episodes_before_update):
                # TODO: Maybe move this into a 'optimizer_step' method?
                # self.update_model()
                
                # logger.debug(f"Backpropagating loss: {self.loss}")
                if self.option == 1:    
                    # Option 1 from above:
                    # NOTE: Need to step as well, since the predictions below will
                    # depend on the model at the current time.
                    self.optimizer.zero_grad()
                    self.loss.backward()
                    self.optimizer.step()
                    # Reset the losses.
                    # self.detached_loss = self.loss.detach()
                    
                    self.loss = Loss(self.name)
                else:
                    # # Option 2 from above:
                    logger.debug(f"Updating model: ")
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                self.detach_all_buffers()
                
                self.has_reached_episode_end[:] = False
                self.num_episodes_since_update[:] = 0
                
                # self.detached_loss = self.loss.detach()
                # self.loss = Loss(self.name, metrics={self.name: RLMetrics()})

        
        # In either case, do we want to detach the representations? or not?
        representations = representations.detach().float()
        
        logits = self.dense(representations)
        # The policy is the distribution over actions given the current state.
        policy = Categorical(logits=logits)
        sample = policy.sample()
        actions = PolicyHeadOutput(
            y_pred=sample,
            logits=logits,
            policy=policy,
        )
        for env_index in range(self.batch_size):
            # Take a slice across the first dimension
            # env_observations = get_slice(observations, env_index)
            env_representations = representations[env_index]
            env_actions = get_slice(actions, env_index)
            self.representations[env_index].append(env_representations)
            self.actions[env_index].append(env_actions)
        
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

        for env_index in range(self.batch_size):
            # Take a slice across the first dimension
            # env_observations = get_slice(observations, env_index)
            env_rewards = get_slice(rewards, env_index)
            self.rewards[env_index].append(env_rewards)
        
        return self.detached_loss
                   
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
        
        episode_actions = self.actions[env_index]
        n_stored_items = len(self.actions[env_index])
        n_items_with_grad = sum(v.logits.requires_grad for v in episode_actions)
        n_items_without_grad = n_stored_items - n_items_with_grad
        self.num_grad_tensors[env_index] += n_items_with_grad
        self.num_detached_tensors[env_index] += n_items_without_grad
        # logger.debug(f"Env {env_index} produces a loss based on "
        #              f"{n_items_with_grad} tensors with grad and "
        #              f"{n_items_without_grad} without. ")

        gradient_usage = GradientUsageMetric(
            used_gradients=n_items_with_grad,
            wasted_gradients=n_items_without_grad,
        )
        
        # TODO: Add 'Metrics' for each episode?
        return Loss(self.name, loss, metrics={
            self.name: RLMetrics(episodes=[EpisodeMetrics(rewards=rewards)]),
            "gradient_usage": gradient_usage
        })


    def policy_gradient(self, rewards: List[float], log_probs: List[Tensor], gamma: float=0.95):
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
        gamma_matrix = make_gamma_matrix(gamma, T, device=log_probs.device)

        discounted_rewards = (reward_matrix * gamma_matrix).sum(dim=-1)
        # normalize discounted rewards
        discounted_rewards = normalize(discounted_rewards)
        policy_gradient = - log_probs.dot(discounted_rewards)
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
        def detach_buffer(old_buffer) -> deque:
            new_items = self._make_buffer()
            for item in old_buffer:
                detached = item.detach()
                new_items.append(detached)
            return new_items
        # detached_representations = map(detach, )
        # detached_actions = map(detach, self.actions[env_index])
        # detached_rewards = map(detach, self.rewards[env_index])
        self.representations[env_index] = detach_buffer(self.representations[env_index])
        self.actions[env_index] = detach_buffer(self.actions[env_index])
        self.rewards[env_index] = detach_buffer(self.rewards[env_index])
        # assert False, (self.representations[0], self.representations[-1])

    def _make_buffer(self, elements: Sequence[Any] = None) -> deque:
        buffer = deque(maxlen=self.hparams.max_episode_window_length)
        if elements:
            buffer.extend(elements)
        return buffer

    def _make_buffers(self) -> List[deque]:
        return [self._make_buffer() for _ in range(self.batch_size)]

    def stack_buffers(self, env_index: int):
        """Stack the observations/actions/rewards for this env and return them.
        """ 
        # episode_observations = tuple(self.observations[env_index])
        episode_representations = tuple(self.representations[env_index])
        episode_actions = tuple(self.actions[env_index])
        episode_rewards = tuple(self.rewards[env_index])
        # TODO: Could maybe use out=<some parameter on this module> to
        # prevent having to create new 'container' tensors at each step?

        # Make sure this all still works (should work even better) once we
        # change the obs spaces to dicts instead of Tuples.
        assert len(episode_representations)
        assert len(episode_actions)
        assert len(episode_rewards)
        stacked_inputs = stack(self.input_space, episode_representations)
        # stacked_actions = stack(self.action_space, episode_actions)
        # stacked_rewards = stack(self.reward_space, episode_rewards)

        # TODO: Update this to use 'stack' if we change the action/reward spaces
        y_preds = torch.stack([action.y_pred for action in episode_actions])
        logits = torch.stack([action.policy.logits for action in episode_actions])
        stacked_actions = PolicyHeadOutput(
            y_pred=y_preds,
            logits=logits,
            policy=Categorical(logits=logits),
        )
        rewards_type = type(episode_rewards[0])
        stacked_rewards = rewards_type(
            y=stack(self.reward_space, [reward.y for reward in episode_rewards])
        )
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
