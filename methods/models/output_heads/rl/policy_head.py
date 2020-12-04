""" Defines a (hopefully general enough) Output Head class to be used by the
BaselineMethod when applied on an RL setting.
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
from gym.vector.utils.numpy_utils import concatenate, create_empty_array
from torch import LongTensor, Tensor, nn
from torch.distributions import Categorical as Categorical_, Distribution

from common import Loss
from common.layers import Lambda
# from common.gym_wrappers.batch_env.worker import FINAL_STATE_KEY
from methods.models.forward_pass import ForwardPass 
from utils.utils import prod
from utils.logging_utils import get_logger
from utils.generic_functions import stack
from settings.base.objects import Actions, Observations, Rewards
from settings.active.rl.continual_rl_setting import ContinualRLSetting

from ..classification_head import ClassificationOutput, ClassificationHead
from ..output_head import OutputHead
logger = get_logger(__file__)

from utils.generic_functions import get_slice, set_slice

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
                 observation_space: spaces.Space,
                 representation_space: spaces.Space,
                 action_space: spaces.Discrete,
                 reward_space: spaces.Box,
                 hparams: "PolicyHead.HParams" = None,
                 name: str = "policy"):
        assert isinstance(action_space, spaces.Discrete), f"Only support discrete action space for now (got {action_space})."
        assert isinstance(reward_space, spaces.Box), f"Reward space should be a Box (scalar rewards) (got {reward_space})."
        super().__init__(
            observation_space=observation_space,
            representation_space=representation_space,
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
        # TODO: Won't use the 'observations' anymore, will only use the
        # representations from the encoder, so renaming 'representations' to
        # 'observations' in this case.
        # (Should probably come up with another name so this isn't ambiguous).
        self.observations: List[deque] = []
        # self.representations: List[deque] = []
        self.actions: List[deque] = []
        self.rewards: List[deque] = []
        
                        
        # List that holds the 'initial' observations, when an episode boundary
        # is reached.
        self.initial_observations: List[Optional[Tuple]] = []

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
        actions = policy.rsample()
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
        
        Also, standard supervised learning could be recovered by setting the
        maximum length of the 'episode buffer' to 1, and consider all
        observations as final, i.e., when episode length == 1
        """
        observations: ContinualRLSetting.Observations = forward_pass.observations
        representations: Tensor = forward_pass.representations
        assert isinstance(observations, ContinualRLSetting.Observations)

        batch_size = observations.batch_size or 1
        # Setup the buffers, which will hold the most recent observations,
        # actions and rewards within the current episode for each environment.
        if not self.observations:
            def make_buffers() -> List[deque]:
                return [
                    deque(maxlen=self.hparams.max_episode_window_length)
                    for _ in range(batch_size)
                ]
            self.observations = make_buffers()
            self.representations = make_buffers()
            self.actions = make_buffers() 
            self.rewards = make_buffers()

        total_loss = Loss(self.name)
        
        # Append the most recent elements into the buffer for that environment.
        # per_env_items = zip(env_observations, env_actions, env_rewards)
        for env_index in range(batch_size):
            # Slice the obs/actions/rewards, to recover the individual items
            # from each environment.
            # NOTE: Will reuse this 'env_observation' later below.
            env_observation = get_slice(observations, env_index)
            env_representations = get_slice(representations, env_index)
            env_action = get_slice(actions, env_index)
            env_reward = get_slice(rewards, env_index)

            done: bool = env_observation.done
            if isinstance(done, (Tensor, np.ndarray)):
                done = done.item()
            assert isinstance(done, bool), done

            # TODO: For now, we just overwrite (get rid of) the oldest items in
            # the buffers.
            self.representations[env_index].append(env_representations)
            self.actions[env_index].append(env_action)
            self.rewards[env_index].append(env_reward)

            if not done:
                # If this obs has done=True, assuming that the env is vectorized
                # and that the AddDoneToObs wrapper gets added *after* the
                # batching as recommended, then the rest of the obs is the
                # initial obs of the new episode.
                self.observations[env_index].append(env_observation)
            elif not self.observations[env_index]:
                raise RuntimeError(f"There are no observations in the buffer?")

            episode_obs = tuple(self.observations[env_index])
            episode_actions = tuple(self.actions[env_index])
            episode_rewards = tuple(self.rewards[env_index])
            
            # TODO: Maybe add a mechanism for disabling this 're-stacking' when
            # we always only compute a loss at the end of episodes?
                        
            # Make sure this all still works (should work even better) once we
            # change the obs to dicts instead of Batch objects.
            stacked_obs = stack(self.observation_space, episode_obs)
            # TODO: Could maybe use out=<some parameter on this module> to
            # prevent having to create new 'container' tensors all the time.
            # TODO: Update this if we change the action space
            y_preds = torch.stack([action.y_pred for action in episode_actions])
            logits = torch.stack([action.policy.logits for action in episode_actions])
            stacked_actions = type(actions)(
                y_pred=y_preds,
                logits=logits,
                policy=Categorical(logits=logits),
            )
            # stack(self.reward_space, episode_rewards)
            stacked_rewards = type(rewards)(
                y=stack(self.reward_space, [reward.y for reward in episode_rewards])
            )

            if done:
                # Clear the buffers.
                self.observations[env_index].clear()
                self.actions[env_index].clear()
                self.rewards[env_index].clear()

                # Add the new 'initial' obs in the buffer.
                # TODO: Move this somewhere else.
                from dataclasses import is_dataclass, replace
                if is_dataclass(env_observation):
                    env_observation = replace(env_observation, done=False)
                elif isinstance(env_observation, NamedTuple):
                    env_observation = env_observation._replace(done=False)
                elif isinstance(env_observation, dict):
                    assert "done" in env_observation
                    env_observation["done"] = False
                else:
                    raise NotImplementedError("TODO: Don't know how to set 'done=False' in env obs {env_observation}.")

                self.observations[env_index].append(env_observation)

            loss = self.get_episode_loss(
                env_index=env_index,
                observations=stacked_obs,
                actions=stacked_actions,
                rewards=stacked_rewards,
                done=done,
            )
            # If we are able to get a loss for this set of observations/actions/
            # rewards, then add it to the total loss.
            if loss is not None:
                total_loss += loss

        if not isinstance(total_loss.loss, Tensor):
            assert total_loss.loss == 0.
            total_loss.loss = torch.zeros(1, requires_grad=True)
        return total_loss
           
    def get_episode_loss(self,
                         env_index: int,
                         observations: ContinualRLSetting.Observations,
                         representations: Tensor,
                         actions: ContinualRLSetting.Observations,
                         rewards: ContinualRLSetting.Rewards,
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
        if not done:
            # This particular algorithm (REINFORCE) can't give a loss until the
            # end of the episode is reached.
            return None

        log_probabilities = actions.y_pred_log_prob
        rewards = rewards.y
        loss = self.policy_gradient(rewards=rewards, log_probs=log_probabilities, gamma=self.hparams.gamma)

        # TODO: Add 'Metrics' for each episode?
        return Loss(self.name, loss)


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