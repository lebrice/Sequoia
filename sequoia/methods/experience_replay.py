"""A random baseline Method that gives random predictions for any input.

Should be applicable to any Setting.
"""
from collections import Iterable, OrderedDict
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Type
from argparse import ArgumentParser, Namespace

import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import tqdm
from torch import Tensor
from torchvision.models import ResNet, resnet18

from sequoia.common.metrics import ClassificationMetrics
from sequoia.methods import register_method
from sequoia.settings import ClassIncrementalSetting, Setting
from sequoia.settings.base import Actions, Environment, Method, Observations
from sequoia.utils import get_logger, singledispatchmethod


logger = get_logger(__file__)


@register_method
@dataclass
class ExperienceReplayMethod(Method, target_setting=ClassIncrementalSetting):
    """ Simple method that uses a replay buffer to reduce forgetting.
    """
    
    def __init__(self,
                 learning_rate: float = 0.1,
                 buffer_capacity: int = 200,
                 seed: int = None):
        self.learning_rate = learning_rate
        self.buffer_capacity = buffer_capacity

        self.net: ResNet
        self.buffer: Optional[Buffer] = None
        self.optim: torch.optim.Optimizer
        self.task: int = 0
        self.rng = np.random.RandomState(seed)
        if seed:
            torch.manual_seed(seed)
            torch.set_deterministic(True)
        
        self.epochs_per_task: int = 0
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

    def configure(self, setting: ClassIncrementalSetting):
        # create the model
        self.net = models.resnet18(pretrained=False)
        self.net.fc = nn.Linear(512, setting.num_classes)
        if torch.cuda.is_available():
            self.net = self.net.to(device=self.device)

        image_space: spaces.Box = setting.observation_space[0]
        # create the buffer
        
        if self.buffer_capacity:    
            self.buffer = Buffer(
                capacity=self.buffer_capacity,
                input_shape=image_space.shape,
                extra_buffers={"t": torch.LongTensor},
                rng=self.rng
            ).to(device=self.device)
        # optimizer
        self.optim = torch.optim.SGD(self.net.parameters(), lr=self.learning_rate)

    def fit(self, train_env: Environment, valid_env: Environment):
        self.net.train()
        # Simple example training loop, not using the validation loader.
        with tqdm.tqdm(train_env) as train_pbar:
            postfix = {}
            train_pbar.set_description(f"Training")
            for i, batch in enumerate(train_pbar):
                self.optim.zero_grad()

                obs, rew = batch
                obs = obs.to(device=self.device)
                rew = rew.to(device=self.device)
                x, y = obs.x, rew.y
                logits = self.net(x)
                loss = F.cross_entropy(logits, y)

                if self.task > 0 and self.buffer:
                    b_samples = self.buffer.sample(x.size(0))
                    b_logits = self.net(b_samples['x'])
                    loss += F.cross_entropy(logits, b_samples['y'])

                loss.backward()
                self.optim.step()

                # add to buffer
                if self.buffer:
                    self.buffer.add_reservoir({'x': x, 'y': y, 't': self.task})

    def get_actions(self, observations: Observations, action_space: gym.Space) -> Actions:
        observations = observations.to(device=self.device)
        logits = self.net(observations.x)
        pred = logits.max(1)[1]
        return pred

    def on_task_switch(self, task_id: Optional[int]):
        print(f"Switching from task {self.task} to task {task_id}")
        self.task = task_id

    @classmethod
    def add_argparse_args(cls, parser: ArgumentParser, dest: str = None) -> None:
        """Add the command-line arguments for this Method to the given parser.
        
        Parameters
        ----------
        parser : ArgumentParser
            The ArgumentParser. 
        dest : str, optional
            The 'base' destination where the arguments should be set on the
            namespace, by default None, in which case the arguments can be at
            the "root" level on the namespace.
        """
        prefix = f"{dest}." if dest else ""
        parser.add_argument(f"--{prefix}learning_rate", type=float, default=0.1)
        parser.add_argument(f"--{prefix}buffer_capacity", type=int, default=200)
        parser.add_argument(f"--{prefix}seed", type=int, default=None, help="Random seed")
    
    @classmethod
    def from_argparse_args(cls, args: Namespace, dest: str = None):
        """Extract the parsed command-line arguments from the namespace and
        return an instance of class `cls`.

        Parameters
        ----------
        args : Namespace
            The namespace containing all the parsed command-line arguments.
        dest : str, optional
            The , by default None

        Returns
        -------
        cls
            An instance of the class `cls`.
        """
        args = args if not dest else getattr(args, dest)
        return cls(
            learning_rate=args.learning_rate,
            buffer_capacity=args.buffer_capacity,
            seed=args.seed,
        )


class Buffer(nn.Module):
    def __init__(self,
                 capacity: int,
                 input_shape: Tuple[int, ...],
                 extra_buffers: Dict[str, Type[torch.Tensor]] = None,
                 rng: np.random.RandomState = None,
                 ):
        super().__init__()
        self.rng = rng or np.random.RandomState()

        bx = torch.zeros([capacity, *input_shape], dtype=torch.float)
        by = torch.zeros([capacity], dtype=torch.long)

        self.register_buffer('bx', bx)
        self.register_buffer('by', by)
        self.buffers = ['bx', 'by']

        extra_buffers = extra_buffers or {}
        for name, dtype in extra_buffers.items():
            tmp = dtype(capacity).fill_(0)
            self.register_buffer(f'b{name}', tmp)
            self.buffers += [f'b{name}']

        self.current_index = 0
        self.n_seen_so_far = 0
        self.is_full       = 0
        # (@lebrice) args isn't defined here:
        # self.to_one_hot  = lambda x : x.new(x.size(0), args.n_classes).fill_(0).scatter_(1, x.unsqueeze(1), 1)
        self.arange_like = lambda x : torch.arange(x.size(0)).to(x.device)
        self.shuffle     = lambda x : x[torch.randperm(x.size(0))]

    @property
    def x(self):
        return self.bx[:self.current_index]

    @property
    def y(self):
        raise NotImplementedError("Can't make y one-hot, dont have n_classes.")
        return self.to_one_hot(self.by[:self.current_index])

    def add_reservoir(self, batch: Dict[str, Tensor]) -> None:
        n_elem = batch['x'].size(0)

        # add whatever still fits in the buffer
        place_left = max(0, self.bx.size(0) - self.current_index)

        if place_left:
            offset = min(place_left, n_elem)

            for name, data in batch.items():
                buffer = getattr(self, f'b{name}')
                if isinstance(data, Iterable):
                    buffer[self.current_index: self.current_index + offset].data.copy_(data[:offset])
                else:
                    buffer[self.current_index: self.current_index + offset].fill_(data)

            self.current_index += offset
            self.n_seen_so_far += offset

            # everything was added
            if offset == batch['x'].size(0):
                return

        x = batch['x']
        self.place_left = False

        indices = torch.FloatTensor(x.size(0)-place_left).to(x.device).uniform_(0, self.n_seen_so_far).long()
        valid_indices: Tensor = (indices < self.bx.size(0)).long()

        idx_new_data = valid_indices.nonzero(as_tuple=False).squeeze(-1)
        idx_buffer   = indices[idx_new_data]

        self.n_seen_so_far += x.size(0)

        if idx_buffer.numel() == 0:
            return

        # perform overwrite op
        for name, data in batch.items():
            buffer = getattr(self, f'b{name}')
            if isinstance(data, Iterable):
                data = data[place_left:]
                buffer[idx_buffer] = data[idx_new_data]
            else:
                buffer[idx_buffer] = data

    def sample(self, n_samples: int, exclude_task: int = None) -> Dict[str, Tensor]:
        buffers = OrderedDict()
        if exclude_task is not None:
            assert hasattr(self, 'bt')
            valid_indices = (self.bt != exclude_task).nonzero().squeeze()
            for buffer_name in self.buffers:
                buffers[buffer_name] = getattr(self, buffer_name)[valid_indices]
        else:
            for buffer_name in self.buffers:
                buffers[buffer_name] = getattr(self, buffer_name)[:self.current_index]

        bx = buffers['bx']
        if bx.size(0) < n_samples:
            return buffers
        else:
            indices_np = self.rng.choice(bx.size(0), n_samples, replace=False)
            indices = torch.from_numpy(indices_np).to(self.bx.device)
            return OrderedDict({k[1:]: v[indices] for (k,v) in buffers.items()})


if __name__ == "__main__":
    ExperienceReplayMethod.main()
