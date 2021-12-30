from collections import defaultdict
from dataclasses import dataclass, replace
from functools import partial, singledispatch
from logging import getLogger as get_logger
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    Generic,
    List,
    NoReturn,
    Optional,
    Sequence,
    SupportsFloat,
    Tuple,
    TypeVar,
    Union,
)
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.trainer.states import RunningStage

import torch
from gym import Wrapper, spaces
from gym.spaces.utils import flatdim, flatten, flatten_space
from gym.vector import VectorEnv
from sequoia.common.episode_collector.utils import get_reward_space, get_single_reward_space
from sequoia.common.spaces.space import Space
from sequoia.common.spaces.utils import batch_space
from pytorch_lightning import LightningModule, Trainer
from torch import random
from sequoia.common.batch import Batch
from sequoia.common.config import Config
from sequoia.common.episode_collector import (
    OnPolicyEpisodeDataset,
    OnPolicyEpisodeLoader,
)
from sequoia.common.episode_collector.episode import Episode, Transition, StackedEpisode
from sequoia.common.episode_collector.policy import EpsilonGreedyPolicy
from sequoia.common.gym_wrappers.transform_wrappers import (
    TransformAction,
    TransformObservation,
)
from sequoia.common.spaces.typed_dict import TypedDictSpace
from sequoia.common.typed_gym import (
    _Action,
    _Env,
    _Observation_co,
    _Reward,
    _Space,
    _VectorEnv,
)
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
        # NOTE: This is hella ugly. But does what it's supposed to, and is likely quicker:
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
        episodes_per_update: int = 10

    def __init__(
        self,
        train_env: _Env[Any, Any, Any],
        hparams: HParams = None,
        config: Config = None,
        val_env: _Env = None,
        test_env: _Env = None,
        episodes_per_train_epoch: int = 1_000,
        episodes_per_val_epoch: int = 10,
        steps_per_train_epoch: int = None,
        steps_per_val_epoch: int = None,
        recompute_forward_passes: bool = True,
    ):
        """
        TODO: Not sure if this should assume that the provided envs are already wrapped/compatible
        with the model..
        - Upside: If we assume that they are already compatible, then we don't need to wrap them
          here. Makes the code simpler.
        - Downside: Might make this a little bit harder to use: Users need to wrap stuff before
          passing the models in.

        TODO: IF we were to set the input-output objects/interface on the model, How about using
        something like a 'data protocol' / TypedDicts?
        """
        super().__init__()
        self.hp = hparams or self.HParams()
        self.config = config or Config()

        # TODO: Figure this out: Have the train env as a property?
        self._train_env: _Env[
            Observation, DiscreteActionBatch, Rewards
        ] = self.wrap_env(train_env)
        self._val_env: Optional[
            _Env[Observation, DiscreteActionBatch, Rewards]
        ] = None
        self._test_env: Optional[
            _Env[Observation, DiscreteActionBatch, Rewards]
        ] = None

        if val_env is not None:
            self._val_env = self.wrap_env(val_env)
        if test_env is not None:
            self._test_env = self.wrap_env(test_env)

        self._train_dataloader: OnPolicyEpisodeLoader[
            Observation, DiscreteActionBatch, Rewards
        ]
        self._val_dataloader: Optional[
            OnPolicyEpisodeLoader[Observation, DiscreteActionBatch, Rewards]
        ] = None
        self._test_dataloader: Optional[
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
        self.n_recomputed_forward_passes = 0
        self.n_forward_passes = 0
        self.steps_per_trainer_stage: DefaultDict[RunningStage, int] = defaultdict(int)
        self.n_wasted_forward_passes = 0

    def forward(
        self,
        observation: Observation[Tensor],
        action_space: _Space[DiscreteActionBatch],
    ) -> DiscreteActionBatch:
        random_action = action_space.sample()
        # TODO: The action space passed here should already produce Tensors on the right device.
        # (It currently produces numpy arrays)
        assert observation.observation.device == self.device, (
            observation.observation.device,
            self.device,
        )
        logits = self.net(observation.observation)
        action = logits.argmax(-1)
        self.n_forward_passes += 1
        return replace(random_action, logits=logits, action=action)

    def to(self, *args: Any, **kwargs: Any):
        device_before = self.device
        result = super().to(*args, **kwargs)
        # TODO: Fix the mismatch between the train env's device and the model's device.
        # self.train_env.to_(device=self.device)
        return result

    def on_post_move_to_device(self) -> None:
        super().on_post_move_to_device()
        # Maybe do this here instead? Doesn't seem to be called though.

    def training_step(
        self,
        episode: StackedEpisode[_Observation_co, _Action, _Reward],
        batch_idx: int,
    ) -> Tensor:
        """Calculate a loss for a given episode.

        NOTE: The actions in the episode are *currently* from the same model that is being trained.
        accumulate_grad_batches controls the update frequency.
        """
        if batch_idx > 0 and batch_idx % self.hp.episodes_per_update == 0:
            # NOTE: Not explicitly updating the policies anymore. Relying instead on the fact that
            # the DataLoaders hold a pointer to `self`, and so the updates are 'seen' immediately.
            self.n_policy_updates += 1

            # This is the first batch after an update, so we invariably need to recompute the
            # forward passes (because the action logits were computed by the model from one step
            # ago, because of the 'profile iterable' thing in PL (with_is_last kind-of wrapper fn))
            logger.debug(
                f"Incrementing the number of policy updates to {self.n_policy_updates}."
            )
            episode = self.handle_off_policy_data(episode)
            if episode is None:
                # NOTE: Not counting the number of wasted updates in that case.
                return None

        return self.shared_step(episode, batch_idx=batch_idx)

    def validation_step(
        self,
        episode: StackedEpisode[_Observation_co, _Action, _Reward],
        batch_idx: int,
    ) -> Tensor:
        """ Calculate a loss for a given episode.
        """
        # TODO: In this case here, should we care if the episode is on-policy or not?
        return self.shared_step(episode=episode, batch_idx=batch_idx)

    def test_step(
        self,
        episode: StackedEpisode[_Observation_co, _Action, _Reward],
        batch_idx: int,
    ) -> Tensor:
        """ Calculate a test loss for a given episode.
        """
        # TODO: In this case here, should we care if the episode is on-policy or not?
        return self.shared_step(episode=episode, batch_idx=batch_idx)

    def shared_step(
        self,
        episode: StackedEpisode[_Observation_co, _Action, _Reward],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Tensor:
        """ Perform a single loss computation. """
        selected_action_logits = episode.actions.action_logits
        loss = vanilla_policy_gradient(
            rewards=episode.rewards.reward,
            log_probs=selected_action_logits,
            gamma=self.hp.gamma,
        )
        metrics = dict(
            mean_episode_length=len(episode),
            mean_episode_reward=sum(episode.rewards.reward),
        )
        if self.trainer:
            stage = self.trainer.state.stage
            self.log(f"episode_metrics/{stage.value}", metrics, prog_bar=True)
            self.steps_per_trainer_stage[stage] += 1
        step_output = {
            "loss": loss,
            "metrics": metrics,
        }
        return step_output

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
            self.n_wasted_forward_passes += 1
            return None

        # Option 2: Recompute the forward pass for that episode:
        # assert False, (episode.model_versions, self.n_updates, episode.actions)
        # todo: Test this out with a VectorEnv
        # NOTE: Get an action space that reflects the number of actions required:
        single_action_space = getattr(
            self.train_env, "single_action_space", self.train_env.action_space
        )
        action_space = batch_space(single_action_space, n=len(episode))
        action_space.dtype = DiscreteActionBatch

        # old_actions = episode.actions
        new_actions = self(episode.observations, action_space=action_space)
        self.n_recomputed_forward_passes += 1
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

    @property
    def test_env(self) -> _Env[Observation, DiscreteAction, Rewards]:
        return self._test_env

    def set_train_env(self, env: _Env) -> None:
        """TODO: Could use this when reaching a second task for example."""
        # if not (isinstance(env.observation_space, TypedDictSpace)
        #     and issubclass(env.observation_space.dtype, Observation)):
        self._train_env = self.wrap_env(env)

    def wrap_env(self, env: _Env[Any, Any, Any]) -> _Env[Observation, Action, Rewards]:
        """Adds wrappers around the given env, to make it compatible with this model.
        
        TODO: If we decide that the model should receive compatible envs, then make this a
        classmethod
        """
        # NOTE: This ugly looking check is essentially the "assumptions" of this model.
        # Any env that matches this, or that can be made to match this, is 'good'.
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
            env.action_space, (spaces.Discrete, spaces.MultiDiscrete)
        ):
            # TODO: For now there's only this simple case for envs like CartPole. But we could
            # recursively/iteratively call this function (or some kind of singledispatchmethod) that
            # adds the required wrappers for a given env, until the env matches what we want above.
            # Use this on something like CartPole for example.
            env = UseObjectsWrapper(env)
            env = ConvertToFromTensors(env, device=self.device)
            return env

        raise NotImplementedError(env, env.observation_space, env.action_space)

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

    def train_dataloader(
        self,
    ) -> DataLoader[Episode[Observation, DiscreteAction, Rewards]]:
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

        val_policy = torch.no_grad()(self)
        self._val_dataloader = make_env_loader(
            env=self.val_env,
            max_episodes=self.episodes_per_val_epoch,
            max_steps=self.steps_per_val_epoch,
            policy=val_policy,
        )
        return self._val_dataloader

    def test_dataloader(
        self,
    ) -> Optional[DataLoader[Episode[Observation, DiscreteAction, Rewards]]]:
        if self.test_env is None:
            return None
        # dataset = EnvDataset(self.val_env, policy=self)
        # self._val_dataloader = EnvDataLoader(dataset)
        max_episodes = self.episodes_per_val_epoch
        max_steps = self.steps_per_val_epoch

        test_policy = torch.no_grad()(self)
        self._test_dataloader = make_env_loader(
            env=self.test_env,
            max_episodes=self.episodes_per_val_epoch,
            max_steps=self.steps_per_val_epoch,
            policy=test_policy,
        )
        return self._test_dataloader

    # def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
    #     assert not isinstance(batch, Episode), batch
    #     if isinstance(batch, (StackedEpisode, Episode, Transition)):
    #         return move(batch, device=device)
    #     return super().transfer_batch_to_device(batch, device)

    def on_after_backward(self) -> None:
        super().on_after_backward()
        # self.deploy_new_policies()

    def on_before_zero_grad(self, optimizer: Optimizer) -> None:
        super().on_before_zero_grad(optimizer)
        # print(f"Heyo.")
        # self.deploy_new_policies()

    def configure_callbacks(self):
        return []

    def deploy_new_policies(self) -> None:
        """Updates the policies used by the train / val dataloaders"""
        logger.debug(
            f"Performing policy update #{self.n_policy_updates} at global step {self.global_step}"
        )
        # TODO: There isn't really a need to do this, since we are giving a "pointer" to `self`, the
        # policy gets updated anyway!
        debug_stuff = {
            "recomputed_forward_passes": self.n_recomputed_forward_passes,
            "wasted_forward_passes": self.n_wasted_forward_passes,
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

@singledispatch
def logits_space_for(action_space: spaces.Discrete) -> spaces.Box:
    return spaces.Box(-np.inf, np.inf, (action_space.n,))


@logits_space_for.register(spaces.MultiDiscrete)
def _batched_logits_space(action_space: spaces.MultiDiscrete) -> spaces.Box:
    assert len(set(action_space.nvec)) == 1
    return spaces.Box(-np.inf, np.inf, shape=(len(action_space.nvec), action_space.nvec[0]))


# note: This is the same job as the `TypedObjectsWrapper`, basically.
class UseObjectsWrapper(Wrapper, _Env[Observation, DiscreteAction, Rewards]):
    def __init__(self, env: _Env[Any, int, float]) -> None:
        super().__init__(env)
        assert isinstance(env.action_space, spaces.Discrete)
        reward_space = get_reward_space(env)
        assert isinstance(reward_space, spaces.Box)

        self.observation_space = TypedDictSpace(
            observation=env.observation_space, dtype=Observation
        )
        n_actions = env.action_space.n
        self.action_space = TypedDictSpace(
            action=env.action_space,
            logits=logits_space_for(env.action_space),
            dtype=DiscreteAction,
        )
        self.reward_space = TypedDictSpace(
            reward=reward_space,
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


class UseObjectsVectorEnvWrapper(Wrapper, _VectorEnv[Observation, DiscreteActionBatch, Rewards]):
    def __init__(self, env: _VectorEnv[Any, Any, Any]) -> None:
        super().__init__(env)
        assert isinstance(env.unwrapped, VectorEnv)
        assert isinstance(env.action_space, spaces.Discrete)
        assert isinstance(env.single_action_space, spaces.Discrete)
        reward_space = get_reward_space(env)
        single_reward_space = get_single_reward_space(env)
        assert isinstance(reward_space, spaces.Box)
        assert isinstance(single_reward_space, spaces.Box)
        assert len(set(env.action_space.nvec)) == 1
        n_actions = env.action_space.nvec[0]
        num_envs = env.num_envs
        
        self.single_observation_space = TypedDictSpace(
            observation=env.single_observation_space, dtype=Observation
        )
        self.observation_space = TypedDictSpace(
            observation=env.observation_space, dtype=Observation
        )
        # NOTE: using spaces.Box here, so the ConvertToTensors wrapper should be used *after* this
        # one.
        self.single_action_space = TypedDictSpace(
            action=env.single_action_space,
            logits=logits_space_for(env.action_space),
            dtype=DiscreteAction,
        )
        self.action_space = TypedDictSpace(
            action=env.action_space,
            logits=logits_space_for(env.action_space),
            dtype=DiscreteActionBatch,
        )
        self.reward_space = TypedDictSpace(
            reward=reward_space,
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
