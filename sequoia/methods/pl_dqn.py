# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Deep Reinforcement Learning: Deep Q-network (DQN)

The template illustrates using Lightning for Reinforcement Learning. The example builds a basic DQN using the
classic CartPole environment.

To run the template, just run:
`python reinforce_learn_Qnet.py`

After ~1500 steps, you will see the total_reward hitting the max score of 475+.
Open up TensorBoard to see the metrics:

`tensorboard --logdir default`

References
----------

[1] https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-
Second-Edition/blob/master/Chapter06/02_dqn_pong.py
"""

import argparse
from collections import defaultdict, deque, namedtuple, OrderedDict
import dataclasses
from typing import (
    Any,
    Callable,
    Iterator,
    List,
    Optional,
    Sequence,
    SupportsFloat,
    SupportsInt,
    Tuple,
    Type,
    Union,
)

import gym
from matplotlib import collections
import numpy as np
from simple_parsing import ArgumentParser, Serializable
import simple_parsing
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
import tqdm
import pytorch_lightning as pl
from typing import NamedTuple, Any
from torch import Tensor
from torch.nn import functional as F
from sequoia.common.spaces.typed_dict import TypedDictSpace
from gym.spaces import Discrete


class DQN(nn.Module):
    """Simple MLP network.

    >>> DQN(10, 5)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    DQN(
      (net): Sequential(...)
    )
    """

    def __init__(self, obs_size: int, n_actions: int, hidden_size: int = 128):
        """
        Args:
            obs_size: observation/state size of the environment
            n_actions: number of discrete actions available in the environment
            hidden_size: size of hidden layers
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(torch.as_tensor(x, dtype=torch.float32))


from typing import TypeVar, Generic
from dataclasses import dataclass


T = TypeVar("T", np.ndarray, Tensor)
T_ = TypeVar("T_", np.ndarray, Tensor)


@dataclass
class Experience(Generic[T]):
    """Experience for one step."""

    state: T
    action: SupportsInt
    reward: SupportsFloat
    done: bool
    new_state: T


# Named tuple for storing experience steps gathered in training
@dataclass()
class ExperienceBatch(Generic[T]):
    """Experience for more than one step.

    Note: neighbouring indices can be independant, i.e. this isn't a sequence of actions in an env.
    """

    states: T
    actions: T
    rewards: T
    dones: T
    new_states: T

    def __len__(self) -> int:
        return len(self.dones)

    def __getitem__(self, index: Union[int, slice]) -> Union[Experience[T], "ExperienceBatch[T]"]:
        if isinstance(index, int):
            return Experience(  # type: ignore
                state=self.states[index],
                action=self.actions[index],
                reward=self.rewards[index],
                done=bool(self.dones[index]),
                new_state=self.new_states[index],
            )
        return ExperienceBatch(
            states=self.states[index],
            actions=self.actions[index],
            rewards=self.rewards[index],
            dones=self.dones[index],
            new_states=self.new_states[index],
        )

    @classmethod
    def stack(cls, items: Sequence["Experience[T]"]) -> "ExperienceBatch[T]":
        field_names = set(f.name for item in items for f in dataclasses.fields(item))
        field_values = defaultdict(list)
        for item in items:
            for field_name in field_names:
                f_value = getattr(item, field_name)
                field_values[field_name].append(f_value)
        stack_fn = np.stack if isinstance(items[0].state, np.ndarray) else torch.stack
        return cls(  # type: ignore
            **{f_name + "s": stack_fn(f_values) for f_name, f_values in field_values.items()}
            # states=np.concatenate(states),
            # actions=np.concatenate(actions),
            # rewards=np.concatenate(rewards, dtype=np.float32),
            # dones=np.concatenate(dones, dtype=bool),
            # new_states=np.concatenate(next_states),
        )

    def _map(self, fn: Callable[[T], T_]) -> "ExperienceBatch[T_]":
        return type(self)(  # type: ignore
            **{f.name: fn(getattr(self, f.name)) for f in dataclasses.fields(self)}
        )

    def numpy(self) -> "ExperienceBatch[np.ndarray]":
        def _numpy(v) -> np.ndarray:
            return v.detach().cpu().numpy() if isinstance(v, Tensor) else np.array(v)

        return self._map(_numpy)

    def to_torch(self, *args, **kwargs) -> "ExperienceBatch[Tensor]":
        def _torch(v) -> Tensor:
            return torch.as_tensor(v, *args, **kwargs)

        return self._map(_torch)

    def to(self, device: Union[str, torch.device] = "cpu"):
        return self.to_torch(device=device)


class ReplayBuffer:
    """Replay Buffer for storing past experiences allowing the agent to learn from them.

    >>> ReplayBuffer(5)  # doctest: +ELLIPSIS
    <...reinforce_learn_Qnet.ReplayBuffer object at ...>
    """

    def __init__(self, capacity: int) -> None:
        """
        Args:
            capacity: size of the buffer
        """
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def append(self, experience: Experience[np.ndarray]) -> None:
        """Add experience to the buffer.

        Args:
            experience: tuple (state, action, reward, done, new_state)
        """
        self.buffer.append(experience)

    def sample(
        self,
        batch_size: int,
    ) -> ExperienceBatch[np.ndarray]:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        # states, actions, rewards, dones, next_states = zip(*(self.buffer[idx] for idx in indices))

        samples = [self.buffer[idx] for idx in indices]
        return ExperienceBatch.stack(samples)
        # return ExperienceBatch(
        #     states=np.array(states),
        #     actions=np.array(actions),
        #     rewards=np.array(rewards, dtype=np.float32),
        #     dones=np.array(dones, dtype=bool),
        #     new_states=np.array(next_states),
        # )
        # for sample in samples
        # return ExperienceBatch(
        #     states=np.stack([sample.state for sample in samples]),
        # )


class RLDataset(IterableDataset):
    """Iterable Dataset containing the ExperienceBuffer which will be updated with new experiences during training.

    >>> RLDataset(ReplayBuffer(5))  # doctest: +ELLIPSIS
    <...reinforce_learn_Qnet.RLDataset object at ...>
    """

    def __init__(self, buffer: ReplayBuffer, sample_size: int = 200) -> None:
        """
        Args:
            buffer: replay buffer
            sample_size: number of experiences to sample at a time
        """
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Iterator[Experience[np.ndarray]]:
        try:
            sampled_experience_batch = self.buffer.sample(self.sample_size)
            for i, sampled_experience in enumerate(sampled_experience_batch):
                yield sampled_experience
        except gym.error.ClosedEnvironmentError:
            raise StopIteration


class Agent:
    """Base Agent class handling the interaction with the environment.

    >>> env = gym.make("CartPole-v1")
    >>> buffer = ReplayBuffer(10)
    >>> Agent(env, buffer)  # doctest: +ELLIPSIS
    <...reinforce_learn_Qnet.Agent object at ...>
    """

    def __init__(self, env: gym.Env, replay_buffer: ReplayBuffer) -> None:
        """
        Args:
            env: training environment
            replay_buffer: replay buffer storing experiences
        """
        self.env = env
        self.replay_buffer = replay_buffer
        self.reset()
        self.state = self.env.reset()

    def reset(self) -> None:
        """Resets the environment and updates the state."""
        self.state = self.env.reset()

    def get_action(self, state: Tensor, net: nn.Module, epsilon: float) -> int:
        """Using the given network, decide what action to carry out using an epsilon-greedy policy.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            action
        """
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            q_values = net(state)
            _, action = torch.max(q_values, dim=-1)
            # TODO: Adapt this for batched actions.
            action = int(action.item())

        return action

    @torch.no_grad()
    def play_step(
        self, net: nn.Module, epsilon: float = 0.0, device: Union[str, torch.device] = "cpu"
    ) -> Tuple[float, bool]:
        """Carries out a single interaction step between the agent and the environment.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            reward, done
        """
        state = torch.as_tensor([self.state], device=torch.device(device))

        action = self.get_action(state=state, net=net, epsilon=epsilon)

        # do step in the environment
        new_state, reward, done, _ = self.env.step(action)

        exp = Experience(
            state=self.state,
            action=action,
            reward=reward,
            done=done,
            new_state=new_state,
        )

        self.replay_buffer.append(exp)

        self.state = new_state
        if done:
            self.state = self.env.reset()
        return reward, done


from dataclasses import dataclass


class DQNLightning(pl.LightningModule):
    """Basic DQN Model.

    >>> DQNLightning(env="CartPole-v1")  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    DQNLightning(
      (net): DQN(
        (net): Sequential(...)
      )
      (target_net): DQN(
        (net): Sequential(...)
      )
    )
    """

    @dataclass
    class HParams(Serializable):
        # Size of the batches.
        batch_size: int = 16

        # learning rate.
        lr: float = 1e-2

        # Discount factor.
        gamma: float = 0.99

        # Interval at which we update the target network.
        sync_rate: int = 10

        # Capacity of the replay buffer.
        replay_size: int = 1000

        # How many samples do we use to fill our buffer at the start of training.
        warm_start_steps: int = 1000

        # The frame at which epsilon should stop decaying.
        eps_last_frame: int = 1000

        # Starting value of epsilon.
        eps_start: float = 1.0

        # Final value of epsilon
        eps_end: float = 0.01

        # Max length of an episode.
        episode_length: int = 200

    def __init__(self, env: Union[str, gym.Env[np.ndarray, int]], hp: HParams = None) -> None:
        super().__init__()
        self.hp = hp or self.HParams()
        self.save_hyperparameters({"hp": self.hp.to_dict()})

        self.env = gym.make(env) if isinstance(env, str) else env
        from gym.spaces import Box, Discrete

        self.episode_length: Optional[int] = get_max_episode_length(self.env)

        if not isinstance(self.env.observation_space, Box):
            raise RuntimeError(
                f"Only works on envs with Box observation space, not {self.env.observation_space}."
            )
        if not isinstance(self.env.action_space, Discrete):
            raise RuntimeError(
                f"Only works on envs with Discrete action space, not {self.env.action_space}."
            )

        from gym.spaces.utils import flatdim

        # TODO: Adapt this to also work with image observations.
        obs_size = flatdim(self.env.observation_space)
        n_actions = self.env.action_space.n

        self.net = DQN(obs_size, n_actions)
        self.target_net = DQN(obs_size, n_actions)

        self.buffer = ReplayBuffer(self.hp.replay_size)
        self.agent = Agent(self.env, self.buffer)
        self.total_reward = 0
        self.episode_reward = 0
        self.populate(self.hp.warm_start_steps)

    def populate(self, steps: int = 1000) -> None:
        """Carries out several random steps through the environment to initially fill up the replay buffer with
        experiences.

        Args:
            steps: number of random steps to populate the buffer with
        """
        for i in range(steps):
            try:
                self.agent.play_step(self.net, epsilon=1.0)
            except gym.error.ClosedEnvironmentError:
                break

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes in a state `x` through the network and gets the `q_values` of each action as an output.

        Args:
            x: environment state

        Returns:
            q values
        """
        output = self.net(x)
        return output

    def dqn_mse_loss(self, batch: ExperienceBatch[Tensor]) -> torch.Tensor:
        """Calculates the mse loss using a mini batch from the replay buffer.

        Args:
            batch: current mini batch of replay data

        Returns:
            loss
        """
        states = batch.states
        actions = batch.actions
        rewards = batch.rewards.type(dtype=torch.float32)
        dones = batch.dones
        next_states = batch.new_states

        values: Tensor = self.net(states)
        state_action_values = values.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_values: Tensor = self.target_net(next_states).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.hp.gamma + rewards
        return F.mse_loss(state_action_values, expected_state_action_values)

    def training_step(self, batch: ExperienceBatch[Tensor], batch_idx: int) -> Tensor:
        """Carries out a single step through the environment to update the replay buffer. Then calculates loss
        based on the minibatch received.

        Args:
            batch: current mini batch of replay data
            batch_idx: batch index

        Returns:
            Training loss and log metrics
        """
        device = batch.states.device
        epsilon = max(
            self.hp.eps_end,
            self.hp.eps_start - (self.global_step + 1) / self.hp.eps_last_frame,
        )
        try:
            # step through environment with agent
            reward, done = self.agent.play_step(self.net, epsilon, device)
        except gym.error.ClosedEnvironmentError:
            print(f"Environment closed at batch {batch_idx}")
            self.trainer: pl.Trainer
            self.trainer.should_stop = True
            return

        self.episode_reward += reward

        # calculates training loss
        loss = self.dqn_mse_loss(batch)

        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0

        # Soft update of target network
        if self.global_step % self.hp.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        self.log_dict(
            {
                "total_reward": self.total_reward,
                "reward": reward,
                "steps": float(self.global_step),
            },
            prog_bar=True,
        )
        return loss

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = optim.Adam(self.net.parameters(), lr=self.hp.lr)
        return [optimizer]

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataset = RLDataset(self.buffer, sample_size=self.episode_length or 200)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hp.batch_size,
            sampler=None,
            collate_fn=ExperienceBatch.stack,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self.__dataloader()

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index if self.on_gpu else "cpu"

    @classmethod
    def add_model_specific_args(cls, parent_parser: ArgumentParser):  # pragma: no-cover
        parent_parser.add_arguments(cls.HParams, "hp")
        return parent_parser


def get_max_episode_length(env: Union[gym.Env, gym.Wrapper]) -> Optional[int]:
    """Inspects the env to get the max episode length, if it is wrapped with a
    `gym.wrappers.TimeLimit` wrapper.
    If the env isn't wrapped with a TimeLimit, then returns None.
    """
    while isinstance(env, gym.Wrapper):
        if isinstance(env, gym.wrappers.TimeLimit):
            return env._max_episode_steps
        env = env.env
    if env.spec is not None:
        return env.spec.max_episode_steps
    return None


from sequoia import Method
from sequoia.settings.rl import RLSetting, RLEnvironment
from sequoia.settings.rl.objects import Observations, Actions, Rewards


class PlDqnMethod(Method, target_setting=RLSetting):
    def __init__(self, hp: DQNLightning.HParams = None) -> None:
        super().__init__()
        self.hp = hp or DQNLightning.HParams()
        self.model: Optional[DQNLightning] = None

    def configure(self, setting: RLSetting) -> None:
        self.model = None
        self.train_max_steps = setting.train_max_steps

    def fit(self, train_env: gym.Env, valid_env: gym.Env):
        from sequoia.common.gym_wrappers import (
            TransformAction,
            TransformObservation,
            TransformReward,
        )

        # Our simple DQN model expects to get arrays / integer actions, so we adapt the env a bit
        # using some wrappers.
        train_env = TransformObservation(train_env, lambda obs: obs.x)
        train_env = TransformReward(train_env, lambda rew: rew.y)
        if isinstance(train_env.action_space, TypedDictSpace):
            actions_type: Type[Actions] = train_env.action_space.dtype
            # Make it possible to send just ints to the env, and wrap them up into an Actions object.
            train_env = TransformAction(train_env, lambda act: actions_type(y_pred=act))

        if self.model is None:
            self.model = DQNLightning(env=train_env, hp=self.hp)

        trainer = pl.Trainer(
            gpus=1, strategy="dp", val_check_interval=100, max_steps=self.train_max_steps
        )
        trainer.fit(self.model)

    def get_actions(self, observations: Observations, action_space: Discrete) -> Actions:
        assert self.model is not None
        with torch.no_grad():
            obs = torch.as_tensor(observations.x, device=self.model.device, dtype=self.model.dtype)
            v = self.model.forward(obs)
        selected_action = v.argmax(-1).cpu().numpy()
        return selected_action


def main() -> None:
    parser = ArgumentParser()
    parser = DQNLightning.add_model_specific_args(parser)
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    args = parser.parse_args()

    env = gym.make("CartPole-v1")

    from sequoia.settings.rl import TraditionalRLSetting

    setting = TraditionalRLSetting(dataset="cartpole", nb_tasks=1, train_max_steps=2_000)
    method = PlDqnMethod()
    from sequoia.common.config import Config

    results = setting.apply(method, config=Config(debug=True))
    print(results)
    return
    hp: DQNLightning.HParams = args.hp

    model = DQNLightning(env=env, hp=hp)
    pl.seed_everything(args.seed)

    trainer = pl.Trainer(gpus=1, strategy="dp", val_check_interval=100)

    trainer.fit(model)


if __name__ == "__main__":

    main()
