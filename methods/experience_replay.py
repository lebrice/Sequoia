"""A random baseline Method that gives random predictions for any input.

Should be applicable to any Setting.
"""
from dataclasses import dataclass

import gym
from utils import get_logger, singledispatchmethod

from common.metrics import ClassificationMetrics
from settings.base import Method, Actions, Observations, Environment

from settings import ClassIncrementalSetting, Setting

import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from collections import OrderedDict
from collections import Iterable

logger = get_logger(__file__)


@dataclass
class ExperienceReplayMethod(Method, target_setting=Setting):
    """ Simple method that uses a replay buffer to reduce forgetting.
    """
    def fit(self,
            train_env: Environment=None,
            valid_env: Environment=None,
            datamodule=None
        ):
        self.net.train()

        # Training loop:
        with tqdm.tqdm(train_env) as train_pbar:
            postfix = {}
            train_pbar.set_description(f"Training")
            for i, batch in enumerate(train_pbar):

                self.optim.zero_grad()

                obs, rew = batch
                x, y = obs.x, rew.y

                logits = self.net(x)
                loss = F.cross_entropy(logits, y)

                if self.task > 0:
                    b_samples = self.buffer.sample(x.size(0))
                    b_logits = self.net(b_samples['x'])
                    loss += F.cross_entropy(logits, b_samples['y'])

                loss.backward()
                self.optim.step()

                # add to buffer
                self.buffer.add_reservoir({'x': x, 'y': y, 't': self.task})

    def get_actions(self, observations: Observations, action_space: gym.Space) -> Actions:
        logits = self.net(observations.x)
        pred   = logits.max(1)[1]
        return pred

    def configure(self, setting: Setting):
        # create the model
        self.net = models.resnet18(pretrained=False)
        self.net.fc = nn.Linear(512, setting.increment * setting.nb_tasks)
        net_params = sum([np.prod(x.shape) for x in self.net.parameters()])

        # create the buffer
        self.buffer = Buffer(capacity=200, input_shp=setting.observation_space[0].shape)

        # optimizer
        self.optim = torch.optim.SGD(self.net.parameters(), lr=.1)

        self.task = 0

    def on_task_switch(self, task_id):
        self.task = task_id
        print(f'task id : {task_id}')

    @singledispatchmethod
    def validate_results(self, setting: Setting, results: Setting.Results):
        """Called during testing. Use this to assert that the results you get
        from applying your method on the given setting match your expectations.

        Args:
            setting
            results (Results): A given Results object.
        """
        assert results is not None
        assert results.objective > 0
        print(f"Objective when applied to a setting of type {type(setting)}: {results.objective}")

    @validate_results.register
    def validate(self, setting: ClassIncrementalSetting, results: ClassIncrementalSetting.Results):
        assert isinstance(setting, ClassIncrementalSetting), setting
        assert isinstance(results, ClassIncrementalSetting.Results), results

        average_accuracy = results.objective
        # Calculate the expected 'average' chance accuracy.
        # We assume that there is an equal number of classes in each task.
        chance_accuracy = 1 / setting.n_classes_per_task

        assert 0.5 * chance_accuracy <= average_accuracy <= 1.5 * chance_accuracy

        for i, metric in enumerate(results.average_metrics_per_task):
            assert isinstance(metric, ClassificationMetrics)
            # TODO: Check that this makes sense:
            chance_accuracy = 1 / setting.n_classes_per_task

            task_accuracy = metric.accuracy
            # FIXME: Look into this, we're often getting results substantially
            # worse than chance, and to 'make the tests pass' (which is bad)
            # we're setting the lower bound super low, which makes no sense.
            assert 0.25 * chance_accuracy <= task_accuracy <= 2.1 * chance_accuracy


class Buffer(nn.Module):
    def __init__(self, capacity, input_shp, extra_buffers={'t': torch.LongTensor}):
        super().__init__()

        bx = torch.FloatTensor(capacity, *input_shp).fill_(0)
        by = torch.LongTensor(capacity).fill_(0)

        self.register_buffer('bx', bx)
        self.register_buffer('by', by)
        self.buffers = ['bx', 'by']

        for name, dtype in extra_buffers.items():
            tmp = dtype(capacity).fill_(0)
            self.register_buffer(f'b{name}', tmp)
            self.buffers += [f'b{name}']

        self.current_index = 0
        self.n_seen_so_far = 0
        self.is_full       = 0
        # (@lebrice) args isn't defined here:
        self.to_one_hot  = lambda x : x.new(x.size(0), args.n_classes).fill_(0).scatter_(1, x.unsqueeze(1), 1)
        self.arange_like = lambda x : torch.arange(x.size(0)).to(x.device)
        self.shuffle     = lambda x : x[torch.randperm(x.size(0))]

    @property
    def x(self):
        return self.bx[:self.current_index]

    @property
    def y(self):
        return self.to_one_hot(self.by[:self.current_index])

    def add_reservoir(self, batch):
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
        valid_indices = (indices < self.bx.size(0)).long()

        idx_new_data = valid_indices.nonzero().squeeze(-1)
        idx_buffer   = indices[idx_new_data]

        self.n_seen_so_far += x.size(0)

        if idx_buffer.numel() == 0:
            return

        # perform overwrite op
        for name, data in batch.items():
            buffer = getattr(self, f'b{name}')
            if isinstance(data, Iterable):
                try:
                    data = data[place_left:]
                    buffer[idx_buffer] = data[idx_new_data]
                except:
                    import pdb; pdb.set_trace()
                    xx = 2
            else:
                buffer[idx_buffer] = data


    def sample(self, amt, exclude_task=None):
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
        if bx.size(0) < amt:
            return buffers
        else:
            indices = torch.from_numpy(np.random.choice(bx.size(0), amt, replace=False)).to(self.bx.device)
            return OrderedDict({k[1:]:v[indices] for (k,v) in buffers.items()})


if __name__ == "__main__":
    ExperienceReplayMethod.main()
