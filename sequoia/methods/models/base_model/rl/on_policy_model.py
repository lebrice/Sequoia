from dataclasses import dataclass, replace
from functools import partial
from logging import getLogger as get_logger
from typing import (Any, Callable, Generic, List, NoReturn, Optional, Sequence,
                    SupportsFloat, Tuple, TypeVar, Union)

import torch
from gym import Wrapper, spaces
from gym.spaces.utils import flatdim, flatten, flatten_space
from gym.vector import VectorEnv
from gym.vector.utils import batch_space
from pytorch_lightning import LightningModule, Trainer
from sequoia.common.batch import Batch
from sequoia.common.config import Config
from sequoia.common.episode_collector import (OnPolicyEpisodeDataset,
                                              OnPolicyEpisodeLoader)
from sequoia.common.episode_collector.episode import Episode, Transition, StackedEpisode
from sequoia.common.episode_collector.policy import EpsilonGreedyPolicy
from sequoia.common.gym_wrappers.transform_wrappers import (
    TransformAction, TransformObservation)
from sequoia.common.spaces.typed_dict import TypedDictSpace
from sequoia.common.typed_gym import (_Action, _Env, _Observation_co, _Reward,
                                      _Space, _VectorEnv)
from sequoia.utils.generic_functions import move, stack
from simple_parsing.helpers.hparams.hyperparameters import HyperParameters
from torch import Tensor, nn
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader

logger = get_logger(__name__)


T_co = TypeVar("T_co", covariant=True)
F = TypeVar("F", bound=SupportsFloat)


@dataclass(frozen=True)
class Observation(Batch, Generic[T_co]):
    observation: T_co


@dataclass(frozen=True)
class Action(Batch, Generic[T_co]):
    action: T_co


@dataclass(frozen=True)
class Rewards(Batch, Generic[F]):
    reward: F


@dataclass(frozen=True)
class DiscreteAction(Action[int]):
    action: int
    logits: Sequence[float]

    @property
    def action_logits(self) -> Tensor:
        """The logits of the actions that were selected."""
        assert isinstance(self.action, int) or self.action.shape == ()
        return self.logits[self.action]


class DiscreteActionSpace(TypedDictSpace[DiscreteAction]):
    action: spaces.Discrete
    logits: spaces.Box


@dataclass(frozen=True)
class DiscreteActionBatch(Batch, Sequence[DiscreteAction]):
    action: Sequence[int]
    logits: Sequence[Sequence[float]]

    def __getitem__(self, index: int) -> DiscreteAction:
        if isinstance(index, str):
            return getattr(self, index)
        result_dtype = DiscreteAction if isinstance(index, int) else DiscreteActionBatch
        return result_dtype(action=self.action[index], logits=self.logits[index])

    @property
    def action_logits(self) -> Tensor:
        """The logits of the actions that were selected."""
        # action, log_probs = torch.broadcast_tensors(action, self.logits)
        # action = action[..., :1]
        # NOTE: This is hella ugly. But does what it's supposed to:
        return self.logits.gather(-1, self.action.unsqueeze(-1)).squeeze(-1)
        # return torch.stack([logit[action] for logit, action in zip(self.logits, self.action)])


class DiscreteActionBatchSpace(TypedDictSpace[DiscreteActionBatch]):
    action: spaces.MultiDiscrete
    logits: spaces.Box


@stack.register(DiscreteAction)
def _stack_action(v: DiscreteAction, *others: DiscreteAction) -> DiscreteActionBatch:
    return DiscreteActionBatch(
        action=stack(v.action, *[other.action for other in others]),
        logits=stack(v.logits, *[other.logits for other in others]),
    )


from typing import NoReturn, overload

# from sequoia.common.episode_collector.off_policy import make_env_loader
from sequoia.common.episode_collector.on_policy import make_env_loader
from sequoia.common.gym_wrappers.convert_tensors import ConvertToFromTensors


class OnPolicyModel(LightningModule):
    """On-Policy RL Model, written with PyTorch-Lightning."""

    @dataclass
    class HParams(HyperParameters):
        """Hyper-Parameters of the On Policy Model."""

        learning_rate: float = 1e-3
        batch_size: int = 32
        # TODO: Need to have a schedule for this:
        # probability of selectign a random action (to favour exploration).
        epsilon: float = 0.1
        gamma: float = 0.95

    def __init__(
        self,
        train_env: _Env[Any, Any, Any],
        hparams: HParams = None,
        config: Config = None,
        val_env: _Env[Observation, DiscreteActionBatch, Rewards] = None,
        episodes_per_train_epoch: int = 1_000,
        episodes_per_val_epoch: int = 10,
        steps_per_train_epoch: int = None,
        steps_per_val_epoch: int = None,
        recompute_forward_passes: bool = True,
    ):
        """NOTE: This assumes that the train/val envs have already been wrapped properly etc."""
        super().__init__()
        self.hp = hparams or self.HParams()
        self.config = config or Config()

        # TODO: Figure this out: Have the train env as a property?
        self._train_env: _Env[Observation, DiscreteActionBatch, Rewards]
        self.set_train_env(train_env)
        self._val_env: Optional[_Env[Observation, DiscreteActionBatch, Rewards]] = None
        if val_env is not None:
            self._val_env = self.wrap_env(val_env)

        self._train_dataloader: OnPolicyEpisodeLoader[Observation, DiscreteActionBatch, Rewards]
        self._val_dataloader: Optional[
            OnPolicyEpisodeLoader[Observation, DiscreteActionBatch, Rewards]
        ] = None

        self.net = self.create_network()
        
        self.episodes_per_train_epoch = episodes_per_train_epoch
        self.steps_per_train_epoch = steps_per_train_epoch
        self.episodes_per_val_epoch = episodes_per_val_epoch
        self.steps_per_val_epoch = steps_per_val_epoch
        self.recompute_forward_passes = recompute_forward_passes

        # Number of updates so far:
        self.n_policy_updates: int = 0
        self.recomputed_forward_passes = 0
        self.n_forward_passes = 0
        self.n_training_steps = 0
        self.n_validation_steps = 0
        self.wasted_forward_passes = 0

    def forward(
        self, observation: Observation[Tensor], action_space: _Space[DiscreteActionBatch]
    ) -> DiscreteActionBatch:
        random_action = action_space.sample()
        logits = self.net(observation.observation)
        action = logits.argmax(-1)
        self.n_forward_passes += 1
        return replace(random_action, logits=logits, action=action)

    def training_step(
        self,
        episode: StackedEpisode[_Observation_co, _Action, _Reward],
        batch_idx: int,
    ) -> Tensor:
        """Calculate a loss for a given episode.

        NOTE: The actions in the episode are *currently* from the same model that is being trained.
        accumulate_grad_batches controls the update frequency.
        """
        self.n_training_steps += 1
        
        episode_model_versions = set(episode.model_versions)
        if episode_model_versions == {self.n_policy_updates}:
            logger.debug(f"Batch {batch_idx}: all good, data is 100% on-policy.")
        else:
            logger.debug(f"Batch {batch_idx}: data isn't fully on-policy.")
            logger.debug(f"{episode.model_versions=}, {self.n_policy_updates=}")
            episode = self.handle_off_policy_data(episode)
            if episode is None:
                return None

        selected_action_logits = episode.actions.action_logits

        loss = vanilla_policy_gradient(
            rewards=episode.rewards.reward,
            log_probs=selected_action_logits,
            gamma=self.hp.gamma,
        )
        return {
            "loss": loss,
            "episode_length": len(episode),
            "return": sum(episode.rewards.reward),
        }
        return loss

    def validation_step(
        self,
        episode: Episode[_Observation_co, _Action, _Reward],
        batch_idx: int,
    ) -> Tensor:
        """ Calculate a loss for a given episode.
        """
        # TODO: In this case here, should we care if the episode is on-policy or not?
        selected_action_logits = episode.actions.action_logits
        loss = vanilla_policy_gradient(
            rewards=episode.rewards.reward,
            log_probs=selected_action_logits,
            gamma=self.hp.gamma,
        )
        self.trainer: Trainer
        if self.trainer and self.trainer.state.stage.value != "sanity_check":    
            self.n_validation_steps += 1
        return loss

    def handle_off_policy_data(
        self, episode: Episode[Observation, DiscreteAction, Rewards]
    ) -> Optional[Episode[Observation, DiscreteAction, Rewards]]:
        """Handle potentially partly-off-policy data. This is often necessary because of the
        dataloading of pytorch-lightning, which creates a 1-batch delay between the dataloader and the model updates.

        If you return None here, then that batch is ignored. If you return another Episode, it must
        be fully on-policy (i.e. have actions with grad-fns).
        """
        ## Option 1 (Simplest): Drop that episode (return None loss)
        ## Pros: Simple
        ## Cons: Wasteful, can't be used when update frequency == 1, otherwise no training.
        if not self.recompute_forward_passes:
            self.wasted_forward_passes += 1
            return None

        # Option 2: Recompute the forward pass for that episode:
        # assert False, (episode.model_versions, self.n_updates, episode.actions)
        # todo: fix for vectorenv, the `action_space` here might not actually work.
        # todo: Make Episodes immutable
        single_action_space = getattr(
            self.train_env, "single_action_space", self.train_env.action_space
        )
        action_space = batch_space(single_action_space, n=len(episode))
        action_space.dtype = DiscreteActionBatch

        old_actions = episode.actions
        new_actions = self(episode.observations, action_space=action_space)
        self.recomputed_forward_passes += 1

        # Replace the actions
        episode = replace(
            episode,
            actions=new_actions,
            model_versions=[self.n_policy_updates for _ in episode.model_versions],
        )
        return episode

    @property
    def train_env(self) -> _Env[Observation, DiscreteAction, Rewards]:
        return self._train_env

    @property
    def val_env(self) -> _Env[Observation, DiscreteAction, Rewards]:
        return self._val_env

    def set_train_env(self, env: _Env) -> None:
        """TODO: Could use this when reaching a second task for example."""
        # if not (isinstance(env.observation_space, TypedDictSpace)
        #     and issubclass(env.observation_space.dtype, Observation)):
        self._train_env = self.wrap_env(env)

    def wrap_env(self, env: _Env[Any, Any, Any]) -> _Env[Observation, Action, Rewards]:
        """TODO: Add wrappers around the given env, to make it compatible with this model."""
        if (
            isinstance(env.observation_space, TypedDictSpace)
            and issubclass(env.observation_space.dtype, Observation)
            and isinstance(env.action_space, TypedDictSpace)
            and issubclass(env.action_space.dtype, DiscreteAction)
            and hasattr(env, "reward_space")
            and isinstance(env.reward_space, TypedDictSpace)
            and issubclass(env.reward_space.dtype, Rewards)
        ):
            # All good. (env is probably already wrapped).
            return env

        if isinstance(env.observation_space, spaces.Box) and isinstance(
            env.action_space, spaces.Discrete
        ):
            # Use this on something like CartPole for example.
            env = ConvertToFromTensors(env, device=self.device)
            env = UseObjectsWrapper(env)
            return env

        raise NotImplementedError(env)

    def create_network(self) -> nn.Module:
        # Note: need to get the single observation space, in the case of a vectorenv.
        single_observation_space = getattr(
            self.train_env, "single_observation_space", self.train_env.observation_space
        )
        single_action_space = getattr(
            self.train_env, "single_action_space", self.train_env.action_space
        )
        input_dims = flatdim(single_observation_space.observation)
        logits_dims = flatdim(single_action_space.logits)
        return nn.Sequential(
            nn.Linear(input_dims, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, logits_dims),
        )

    def train_dataloader(self) -> DataLoader[Episode[Observation, DiscreteAction, Rewards]]:
        # note: might cause infinite recursion issues no?
        # TODO: Turn this on eventually. Just debugging atm.
        # train_policy = EpsilonGreedyPolicy(
        #     base_policy=self, epsilon=self.hp.epsilon, seed=self.config.seed
        # )
        train_policy = self
        self._train_dataloader = make_env_loader(
            self.train_env,
            policy=train_policy,
            max_episodes=self.episodes_per_train_epoch,
            max_steps=self.steps_per_train_epoch,
            seed=self.config.seed,
        )
        return self._train_dataloader

    def val_dataloader(
        self,
    ) -> Optional[DataLoader[Episode[Observation, DiscreteAction, Rewards]]]:
        if self.val_env is None:
            return None
        # dataset = EnvDataset(self.val_env, policy=self)
        # self._val_dataloader = EnvDataLoader(dataset)
        max_episodes = self.episodes_per_val_epoch
        max_steps = self.steps_per_val_epoch

        self.val_policy = torch.no_grad()(self)
        self._val_dataloader = make_env_loader(
            env=self.val_env,
            max_episodes=self.episodes_per_val_epoch,
            max_steps=self.steps_per_val_epoch,
            policy=self.val_policy,
        )
        return self._val_dataloader

    def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        if isinstance(batch, (Episode, Transition)):
            return move(batch, device=device)
        return super().transfer_batch_to_device(batch, device)

    def on_after_backward(self) -> None:
        super().on_after_backward()
        self.deploy_new_policies()

    def on_before_zero_grad(self, optimizer: Optimizer) -> None:
        super().on_before_zero_grad(optimizer)
        # print(f"Heyo.")
        # self.deploy_new_policies()

    def deploy_new_policies(self) -> None:
        """Updates the policies used by the train / val dataloaders"""
        logger.debug(
            f"Performing policy update #{self.n_policy_updates} at global step {self.global_step}"
        )
        debug_stuff = {
            "recomputed_forward_passes": self.recomputed_forward_passes,
            "wasted_forward_passes": self.wasted_forward_passes,
            "n_forward_passes": self.n_forward_passes,
            "n_updates": self.n_policy_updates,
        }
        logger.debug(f"Debug stuff: {debug_stuff}")
        self.log_dict(debug_stuff)
        # TODO: Update the epsilon using a scheduler.
        # train_policy = EpsilonGreedyPolicy(
        #     base_policy=self, epsilon=self.hp.epsilon, seed=self.config.seed
        # )
        self._train_dataloader.send(new_policy=self)
        if self._val_dataloader is not None:
            val_policy = torch.no_grad()(self)
            self._val_dataloader.send(new_policy=val_policy)
        self.n_policy_updates += 1

    def on_train_batch_end(
        self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        super().on_train_batch_end(outputs, batch, batch_idx, dataloader_idx)

    def configure_optimizers(self):
        return Adam(self.net.parameters(), lr=self.hp.learning_rate)


import numpy as np
from gym import Wrapper, spaces


# note: This is the same job as the `TypedObjectsWrapper`, basically.
class UseObjectsWrapper(Wrapper, _Env[Observation, DiscreteAction, Rewards]):
    def __init__(self, env: _Env[Observation, DiscreteAction, Rewards]) -> None:
        super().__init__(env)
        self.observation_space = TypedDictSpace(
            observation=env.observation_space, dtype=Observation
        )
        self.action_space = TypedDictSpace(
            action=env.action_space,
            logits=spaces.Box(
                -np.inf, np.inf, shape=(env.action_space.n,)
            ),  # todo: fix for vectorenv.
            dtype=DiscreteAction,
        )
        self.reward_space = TypedDictSpace(
            reward=spaces.Box(env.reward_range[0], env.reward_range[1], shape=()),
            dtype=Rewards,
        )

    def reset(self) -> Observation:
        return self.observation_space.dtype(observation=self.env.reset())

    def step(self, action: DiscreteAction) -> Tuple[Observation, Rewards, bool, dict]:
        action = action.action
        obs, reward, done, info = self.env.step(action)
        return (
            self.observation_space.dtype(observation=obs),
            self.reward_space.dtype(reward=reward),
            done,
            info,
        )


from gym.wrappers.transform_observation import TransformObservation


class FlattenedObs(TransformObservation):
    def __init__(self, env: _Env) -> None:
        super().__init__(env, f=lambda v: v)
        self.observation_space = flatten_space(self.env.observation_space)

    def observation(self, observation):
        return flatten(self.env.observation_space, observation)


def vanilla_policy_gradient(
    rewards: Sequence[float],
    log_probs: Union[Tensor, List[Tensor]],
    gamma: float = 0.95,
    normalize_returns: bool = True,
):
    """Implementation of the REINFORCE algorithm.

    Adapted from https://medium.com/@thechrisyoon/deriving-policy-gradients-and-implementing-reinforce-f887949bd63

    Parameters
    ----------
    - episode_rewards : Sequence[float]

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
    reward_tensor = torch.as_tensor(rewards, device=log_probs.device).type_as(action_log_probs)
    returns = discounted_sum_of_future_rewards(reward_tensor, gamma=gamma)
    if normalize_returns:
        returns = normalize(returns)
    # Need both tensors to be 1-dimensional for the dot-product below.
    action_log_probs = action_log_probs.reshape(returns.shape)
    policy_gradient = -action_log_probs.dot(returns)
    return policy_gradient


def discounted_sum_of_future_rewards(rewards: Union[Tensor, List[Tensor]], gamma: float) -> Tensor:
    """Calculates the returns, as the sum of discounted future rewards at
    each step.
    """
    if not isinstance(rewards, Tensor):
        rewards = torch.as_tensor(rewards)
    if rewards.ndim > 1:
        rewards = rewards.flatten()
    T = len(rewards)
    # Construct a reward matrix, with previous rewards masked out (with each
    # row as a step along the trajectory).
    reward_matrix = rewards.expand([T, T]).triu()
    # Get the gamma matrix (upper triangular), see make_gamma_matrix for
    # more info.
    gamma_matrix = make_gamma_matrix(gamma, T, device=reward_matrix.device)
    # Multiplying by the gamma coefficients gives the discounted rewards.
    discounted_rewards = reward_matrix * gamma_matrix
    # Summing up over time gives the return at each step.
    return discounted_rewards.sum(-1)


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


def normalize(x: Tensor, inplace: bool = False) -> Tensor:
    if inplace:
        return x.sub_(x.mean()).div_(x.std() + 1e-9)
    x = x - x.mean()
    x = x / (x.std() + 1e-9)
    return x


def main():
    pass


if __name__ == "__main__":
    main()
