from dataclasses import dataclass, replace
from functools import partial
from typing import (
    Any,
    Callable,
    Generic,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)
from gym.vector import VectorEnv
import torch
from gym import Wrapper
from gym.spaces.utils import flatdim, flatten, flatten_space
from pytorch_lightning import LightningModule
from sequoia.common.batch import Batch
from sequoia.common.config import Config
from sequoia.common.episode_collector import EnvDataLoader, EnvDataset
from sequoia.common.episode_collector.episode import Episode, Transition
from sequoia.common.episode_collector.policy import EpsilonGreedyPolicy
from sequoia.common.gym_wrappers.transform_wrappers import TransformAction
from sequoia.common.spaces.typed_dict import TypedDictSpace
from sequoia.common.typed_gym import (
    _Action,
    _Env,
    _Observation_co,
    _Reward,
    _Space,
    _VectorEnv,
)
from simple_parsing.helpers.hparams.hyperparameters import HyperParameters
from torch import Tensor, nn
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from gym.vector.utils import batch_space
from sequoia.utils.generic_functions import move


@dataclass(frozen=True)
class Observation(Batch):
    observation: Any


@dataclass(frozen=True)
class ObservationBatch(Batch):
    observation: Any


@dataclass(frozen=True)
class DiscreteAction(Batch):
    action: int
    logits: Sequence[float]

    @property
    def action_logits(self) -> Tensor:
        """The logits of the actions that were selected. """
        assert isinstance(self.action, int) or self.action.shape == ()
        return self.logits[self.action]


@dataclass(frozen=True)
class DiscreteActionBatch(Batch, Sequence[DiscreteAction]):
    action: Sequence[int]
    logits: Sequence[Sequence[float]]

    def __getitem__(self, index: int) -> DiscreteAction:
        if not isinstance(index, int):
            return super().__getitem__(index)
        result_dtype = DiscreteAction if isinstance(index, int) else DiscreteActionBatch
        return result_dtype(action=self.action[index], logits=self.logits[index])

    @property
    def action_logits(self) -> Tensor:
        """The logits of the actions that were selected. """
        # TODO: This really sucks. torch.index_select doesn't seem to work either.
        return torch.stack(
            [logit[action] for logit, action in zip(self.logits, self.action)]
        )


from sequoia.utils.generic_functions import stack


@stack.register(DiscreteAction)
def _stack_action(v: DiscreteAction, *others: DiscreteAction) -> DiscreteActionBatch:
    return DiscreteActionBatch(
        action=stack(v.action, *[other.action for other in others]),
        logits=stack(v.logits, *[other.logits for other in others]),
    )


@dataclass(frozen=True)
class Rewards(Batch):
    reward: float


@dataclass(frozen=True)
class RewardsBatch(Batch):
    reward: Sequence[float]


@stack.register(Rewards)
def _stack_rewards(v: Rewards, *others: Rewards) -> RewardsBatch:
    return RewardsBatch(reward=stack(v.reward, *[other.reward for other in others]),)


_Action = TypeVar(
    "_Action", bound=Union[DiscreteAction, DiscreteActionBatch], covariant=True
)

_Reward = TypeVar("_Reward", bound=Union[Rewards, RewardsBatch], covariant=True)
from sequoia.common.episode_collector.experience_replay import (
    ExperienceReplayLoader,
)


class BaseRLModel(LightningModule, Generic[_Observation_co, _Action, _Reward]):
    @dataclass
    class HParams(HyperParameters):
        learning_rate: float = 1e-3
        batch_size: int = 32

    def __init__(
        self,
        env: _Env[_Observation_co, _Action, _Reward],
        hparams: HParams = None,
        config: Config = None,
        val_env: _Env[_Observation_co, _Action, _Reward] = None,
    ):
        super().__init__()
        self.hp = hparams or self.HParams()
        self.config = config or Config()
        self._train_env = env
        self._val_env = val_env
        self.train_env = self.wrap_env(env)
        self.val_env = self.wrap_env(val_env) if val_env is not None else None
        self.net = self.create_network()

        self._train_dataloader: Optional[
            DataLoader[Episode[Observation, DiscreteActionBatch, RewardsBatch]]
        ] = None
        self._val_dataloader: Optional[
            DataLoader[Episode[Observation, DiscreteActionBatch, RewardsBatch]]
        ] = None
        # Number of updates so far:
        self.n_updates: int = 0
        self.recomputed_forward_passes = 0
        self.n_forward_passes = 0

    def forward(
        self, observation: Observation, action_space: _Space[DiscreteAction]
    ) -> DiscreteAction:
        random_action = action_space.sample()
        logits = self.net(observation)
        logits = self.net(observation.observation)
        action = logits.argmax(-1)
        return replace(random_action, logits=logits, action=action)

    def create_network(self) -> nn.Module:
        # Note: need to get the single observation space, in the case of a vectorenv.
        single_observation_space = (
            self.train_env.single_observation_space
            if isinstance(self.train_env, _VectorEnv)
            else self.train_env.observation_space
        )
        single_action_space = (
            self.train_env.single_action_space
            if isinstance(self.train_env, _VectorEnv)
            else self.train_env.action_space
        )
        input_dims = flatdim(single_observation_space)
        logits_dims = flatdim(single_action_space.logits)
        return nn.Sequential(
            nn.Linear(input_dims, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, logits_dims),
        )

    def train_dataloader(
        self,
    ) -> DataLoader[Transition[Observation, DiscreteAction, Rewards]]:
        # ) -> DataLoader[Episode[Observation, DiscreteAction, Rewards]]:
        # dataset = EnvDataset(self.train_env, policy=self)
        # self._train_dataloader = EnvDataLoader(dataset)
        self.epsilon = 0.1
        self.policy = EpsilonGreedyPolicy(base_policy=self, epsilon=self.epsilon, seed=self.config.seed)
        self._train_dataloader = ExperienceReplayLoader(
            env=self.train_env,
            batch_size=self.hp.batch_size,
            max_steps=10_000,
            max_episodes=100,
            policy=self.policy,
        )
        return self._train_dataloader

    def val_dataloader(
        self,
    ) -> Optional[DataLoader[Episode[Observation, DiscreteAction, Rewards]]]:
        if self.val_env is None:
            return None
        # dataset = EnvDataset(self.val_env, policy=self)
        # self._val_dataloader = EnvDataLoader(dataset)
        self._val_dataloader = ExperienceReplayLoader(
            env=self.val_env,
            batch_size=self.hp.batch_size,
            max_steps=10_000,
            max_episodes=100,
            policy=torch.no_grad(self.policy),
        )

        return self._val_dataloader

    def transfer_batch_to_device(
        self, batch: Any, device: torch.device, dataloader_idx: int
    ) -> Any:
        if isinstance(batch, (Episode, Transition)):
            return move(batch, device=device)
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def training_step(
        self, episode: Episode[_Observation_co, _Action, _Reward], batch_idx: int,
    ) -> Tensor:
        """ Calculate a loss for a given episode.

        NOTE: The actions in the episode are *currently* from the same model that is being trained.
        accumulate_grad_batches controls the update frequency.
        """
        print(episode.model_versions)
        if set(episode.model_versions) != {self.n_updates}:
            # Need to recompute the forward pass for at least some part of the actions.

            ## Option 1 (Simplest): Drop that episode (return None loss)
            ## Pros: Simple
            ## Cons: Wasteful, can't be used when update frequency == 1, otherwise no training.
            # return None

            # Option 2: Recompute the forward pass for that episode:

            # assert False, (episode.model_versions, self.n_updates, episode.actions)
            # todo: fix for vectorenv, the `action_space` here might not actually work.
            # todo: Make Episodes immutable

            single_action_space = getattr(self.train_env, "single_action_space", self.train_env.action_space)
            action_space = batch_space(single_action_space, n=len(episode))
            action_space.dtype = DiscreteActionBatch

            old_actions = episode.actions
            new_actions = self(episode.observations, action_space=action_space)
            self.recomputed_forward_passes += 1

            # Replace the actions

            episode = replace(
                episode,
                actions=new_actions,
                model_versions=[self.n_updates for _ in episode.model_versions],
            )
        else:
            print(f"all good")
        self.n_forward_passes += 1
        selected_action_logits = episode.actions.action_logits

        loss = vanilla_policy_gradient(
            rewards=episode.rewards.reward,
            log_probs=selected_action_logits,
            gamma=0.95,
        )
        return loss
        return super().training_step(*args, **kwargs)

    def optimizer_step(
        self,
        epoch: int = None,
        batch_idx: int = None,
        optimizer: Optimizer = None,
        optimizer_idx: int = None,
        optimizer_closure: Optional[Callable] = None,
        on_tpu: bool = None,
        using_native_amp: bool = None,
        using_lbfgs: bool = None,
    ) -> None:
        result = super().optimizer_step(
            epoch=epoch,
            batch_idx=batch_idx,
            optimizer=optimizer,
            optimizer_idx=optimizer_idx,
            optimizer_closure=optimizer_closure,
            on_tpu=on_tpu,
            using_native_amp=using_native_amp,
            using_lbfgs=using_lbfgs,
        )
        self._train_dataloader.send(new_policy=self)
        if self._val_dataloader:
            self._val_dataloader.send(new_policy=self)
        self.n_updates += 1
        print(f"Performing uptate #{self.n_updates} at step {self.global_step}")
        return result

    def configure_optimizers(self):
        return Adam(self.net.parameters(), lr=self.hp.learning_rate)

    def wrap_env(
        self, env: _Env[_Observation_co, _Action, _Reward]
    ) -> _Env[Observation, DiscreteAction, Rewards]:
        """ TODO: Add wrappers to conver the 'naked' actions/rewards into the typed objects we have. """
        if isinstance(env.action_space, spaces.Discrete):
            # Use this on something like CartPole for example.
            from sequoia.common.gym_wrappers.convert_tensors import ConvertToFromTensors
            env = ConvertToFromTensors(env, device=self.device)
            env = UseObjectsWrapper(env)
            # env = TransformObservation(
            #     env, f=partial(torch.as_tensor, device=self.device, dtype=self.dtype)
            # )
            # env = TransformAction(env, f=lambda act: act.detach().cpu().numpy())
            return env
        raise NotImplementedError(env)


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
    reward_tensor = torch.as_tensor(rewards, device=log_probs.device).type_as(
        action_log_probs
    )
    returns = discounted_sum_of_future_rewards(reward_tensor, gamma=gamma)
    if normalize_returns:
        returns = normalize(returns)
    # Need both tensors to be 1-dimensional for the dot-product below.
    action_log_probs = action_log_probs.reshape(returns.shape)
    policy_gradient = -action_log_probs.dot(returns)
    return policy_gradient


def discounted_sum_of_future_rewards(
    rewards: Union[Tensor, List[Tensor]], gamma: float
) -> Tensor:
    """ Calculates the returns, as the sum of discounted future rewards at
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
