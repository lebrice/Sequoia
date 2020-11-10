import dataclasses
import itertools
from dataclasses import dataclass
from typing import Dict, Tuple, Union, List
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
    def __getitem__(self, index: int) -> Distribution:
        return Categorical(logits=self.logits[index])
        # return Categorical(probs=self.probs[index])

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
        max_episode_window_length: int = 50
    
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

        # NOTE: Since the action space is discrete(n), the index chosen also
        # corresponds to the chosen action.
        output = PolicyHeadOutput(
            policy=policy,
            logits=logits,
            y_pred=actions,
        )
        return output
    
    def get_loss(self, forward_pass: ForwardPass, actions: Actions, rewards: Rewards) -> Loss:
        """ Given the forward pass, the actions produced by this output head and
        the corresponding rewards, get a Loss to use for training.
        
        NOTE: The training procedure is fundamentally on-policy atm, i.e. the
        observation is a single state, not a rollout, and the reward is the
        immediate reward at the current step. Therefore, this should be taken
        into consideration when implementing an RL output head.
        
        TODO: Change the second argument to be `reward: Rewards` instead of `y`?
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
        
        if not self.episode_buffers:
            # Setup the buffers, which will hold the observations / actions /
            # rewards for an episode for each environment.
            self.episode_buffers = [
                deque(maxlen=self.hparams.max_episode_window_length)
                for _ in range(batch_size)
            ]

        # Go from "tuples of lists" to "list of tuples":
        per_env_observation = observations.as_list_of_tuples()
        per_env_action = actions.as_list_of_tuples()
        per_env_reward = rewards.as_list_of_tuples()

        per_env_items = zip(per_env_observation, per_env_action, per_env_reward)

        for i, env_items in enumerate(per_env_items):
            # Extract the actions/observations, etc for each environment.
            self.episode_buffers[i].append(env_items)

        total_loss = Loss(self.name)
        for env_index, episode_ended in enumerate(done):
            # If the episode in that environment is ended, then perform an
            # update with the REINFORCE algorithm.
            # TODO: Maybe unpack those and convert them into a 'batch' object?
            observations, actions, rewards = zip(*self.episode_buffers[env_index])
            loss = self.get_episode_loss(observations, actions, rewards, episode_ended)
            total_loss += loss

        return total_loss
        # rewards = y
        # m = actions.policy
        # loss =  - actions.y_pred_log_prob * rewards
        # return Loss(self.name, loss)

    def get_episode_loss(self, observations: ContinualRLSetting.Observations,
                               actions: ContinualRLSetting.Observations,
                               rewards: ContinualRLSetting.Rewards,
                               episode_ended: bool) -> Loss:
        """ Gets a loss, given  
        
        
        
        NOTE: While the Batch Observations/Actions/Rewards are from
        the different environments, now they are actually a sequence from a
        single environment. 
        """
        if episode_ended:
            assert False, "HEYHEY!"
        
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