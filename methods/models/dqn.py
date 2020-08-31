"""TODO: Implement a DQN that uses the active dataloaders instead of
the gym environments (as is done in pl_bolts).
"""

from collections import OrderedDict
from typing import Tuple, Union

import torch
from pl_bolts.losses.rl import dqn_loss
from pl_bolts.models.rl.common.agents import ValueAgent
from pl_bolts.datamodules.experience_source import (
    ExperienceSourceDataset,
    DiscountedExperienceSource,
)
from pl_bolts.models.rl.dqn_model import DQN
from pytorch_lightning import Trainer
from torch import Tensor

from settings.active.rl.gym_dataloader import (GymDataLoader, GymDataset)
from torch.utils.data import DataLoader

class DQNAgent(DQN):
    def __init__(self,
                 env: Union[str, GymDataLoader],
                 gpus=0,
                 eps_start=1.0,
                 eps_end=0.02,
                 eps_last_frame=150000,
                 sync_rate=1000,
                 gamma=0.99,
                 learning_rate=0.0001,
                 batch_size=32,
                 replay_size=100000,
                 warm_start_size=10000,
                 num_samples=500, **kwargs):
        # Environment
        if isinstance(env, str):
            env = GymDataLoader(env, batch_size=batch_size, **kwargs)
            print(f"env: {env}")
        self.env = env
        batch_size = self.env.batch_size

        # all the rest should be about the same as in dqn_model.py
        self.obs_shape = self.env.observation_space.shape
        self.n_actions = self.env.action_space.n
        print(f"obs shape: {self.obs_shape}")
        print(f"n_actions: {self.n_actions}")

        # Model Attributes
        self.buffer = None
        self.source = None
        self.dataset = None

        self.net = None
        self.target_net = None
        self.build_networks()

        self.agent = ValueAgent(
            self.net,
            self.n_actions,
            eps_start=eps_start,
            eps_end=eps_end,
            eps_frames=eps_last_frame,
        )

        # Hyperparameters
        self.sync_rate = sync_rate
        self.gamma = gamma
        self.lr = learning_rate
        self.batch_size = batch_size
        self.replay_size = replay_size
        self.warm_start_size = warm_start_size
        self.sample_len = num_samples

        self.save_hyperparameters()

        # Metrics
        self.total_reward = 0
        self.episode_reward = 0
        self.episode_count = 0
        self.episode_steps = 0
        self.total_episode_steps = 0
        self.reward_list = []
        for _ in range(100):
            self.reward_list.append(-21)
        self.avg_reward = -21

    def prepare_data(self) -> None:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        self.datamodule.prepare_data()
        # super().prepare_data()
        # device = torch.device(self.trainer.root_gpu) if self.trainer.num_gpus >= 1 else self.device
        # self.source = ExperienceSource(self.env, self.agent, device)
        # self.buffer = ReplayBuffer(self.replay_size)
        # self.populate(self.warm_start_size)
        # self.dataset = RLDataset(self.buffer, self.sample_len)

    def train_dataloader(self) -> DataLoader:
        """Get train loader"""
        return DataLoader(dataset=self.dataset, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        """Get test loader"""
        return DataLoader(dataset=self.dataset, batch_size=self.batch_size)


    def training_step(self, batch: Tuple[Tensor, Tensor], _) -> OrderedDict:
        """
        Carries out a single step through the environment to update the replay buffer.
        Then calculates loss based on the minibatch recieved

        Args:
            batch: current mini batch of replay data
            _: batch number, not used

        Returns:
            Training loss and log metrics
        """
        print("Training step!")
        self.agent.update_epsilon(self.global_step)

        # step through environment with agent and add to buffer
        exp, reward, done = self.source.step()
        self.buffer.append(exp)

        self.episode_reward += reward
        self.episode_steps += 1

        # calculates training loss
        loss = dqn_loss(batch, self.net, self.target_net)

        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)

        if done:
            self.total_reward = self.episode_reward
            self.reward_list.append(self.total_reward)
            self.avg_reward = sum(self.reward_list[-100:]) / 100
            self.episode_count += 1
            self.episode_reward = 0
            self.total_episode_steps = self.episode_steps
            self.episode_steps = 0

        # Soft update of target network
        if self.global_step % self.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        log = {
            "total_reward": self.total_reward,
            "avg_reward": self.avg_reward,
            "train_loss": loss,
            "episode_steps": self.total_episode_steps,
        }
        status = {
            "steps": self.global_step,
            "avg_reward": self.avg_reward,
            "total_reward": self.total_reward,
            "episodes": self.episode_count,
            "episode_steps": self.episode_steps,
            "epsilon": self.agent.epsilon,
        }

        return OrderedDict(
            {
                "loss": loss,
                "avg_reward": self.avg_reward,
                "log": log,
                "progress_bar": status,
            }
        )


if __name__ == "__main__":
    datamodule = GymDataLoader("CartPole-v0", batch_size=2)
    model = DQNAgent(env=datamodule)
    trainer = Trainer()
    trainer.fit(model, datamodule=datamodule)

