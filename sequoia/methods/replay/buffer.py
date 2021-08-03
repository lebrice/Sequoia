from collections.abc import Iterable
from typing import Dict, Tuple, Type

import numpy as np
import torch
from torch import Tensor, nn


class Buffer(nn.Module):
    def __init__(
        self,
        capacity: int,
        input_shape: Tuple[int, ...],
        extra_buffers: Dict[str, Type[torch.Tensor]] = None,
        rng: np.random.RandomState = None,
    ):
        super().__init__()
        self.rng = rng or np.random.RandomState()
        assert capacity
        self.capacity = capacity

        bx = torch.zeros([capacity, *input_shape], dtype=torch.float)
        by = torch.zeros([capacity], dtype=torch.long)

        self.register_buffer("bx", bx)
        self.register_buffer("by", by)
        self.buffers = ["bx", "by"]

        extra_buffers = extra_buffers or {}
        for name, dtype in extra_buffers.items():
            tmp = dtype(capacity).fill_(0)
            self.register_buffer(f"b{name}", tmp)
            self.buffers += [f"b{name}"]

        self.current_index = 0
        self.n_seen_so_far = 0
        self.is_full = 0
        # (@lebrice) args isn't defined here:
        # self.to_one_hot  = lambda x : x.new(x.size(0), args.n_classes).fill_(0).scatter_(1, x.unsqueeze(1), 1)
        self.arange_like = lambda x: torch.arange(x.size(0)).to(x.device)
        self.shuffle = lambda x: x[torch.randperm(x.size(0))]

    def __len__(self) -> int:
        return self.current_index

    @property
    def x(self):
        return self.bx[: self.current_index]

    @property
    def y(self):
        raise NotImplementedError("Can't make y one-hot, dont have n_classes.")
        return self.to_one_hot(self.by[: self.current_index])

    def add_reservoir(self, batch: Dict[str, Tensor]) -> None:
        n_elem = batch["x"].size(0)

        # add whatever still fits in the buffer
        place_left = max(0, self.bx.size(0) - self.current_index)

        if place_left:
            offset = min(place_left, n_elem)

            for name, data in batch.items():
                buffer = getattr(self, f"b{name}")
                if isinstance(data, Iterable):
                    buffer[self.current_index : self.current_index + offset].data.copy_(
                        data[:offset]
                    )
                else:
                    buffer[self.current_index : self.current_index + offset].fill_(data)

            self.current_index += offset
            self.n_seen_so_far += offset

            # everything was added
            if offset == batch["x"].size(0):
                return

        x = batch["x"]
        self.place_left = False

        indices = (
            torch.FloatTensor(x.size(0) - place_left)
            .to(x.device)
            .uniform_(0, self.n_seen_so_far)
            .long()
        )
        valid_indices: Tensor = (indices < self.bx.size(0)).long()

        idx_new_data = valid_indices.nonzero(as_tuple=False).squeeze(-1)
        idx_buffer = indices[idx_new_data]

        self.n_seen_so_far += x.size(0)

        if idx_buffer.numel() == 0:
            return

        # perform overwrite op
        for name, data in batch.items():
            buffer = getattr(self, f"b{name}")
            if isinstance(data, Iterable):
                data = data[place_left:]
                buffer[idx_buffer] = data[idx_new_data]
            else:
                buffer[idx_buffer] = data

    def sample(self, n_samples: int, exclude_task: int = None) -> Dict[str, Tensor]:
        buffers = {}
        if exclude_task is not None:
            task_label_buffer_name = self.buffers[-1]
            assert task_label_buffer_name in ["bt", "btask_labels"], task_label_buffer_name
            assert hasattr(self, task_label_buffer_name)
            bt = getattr(self, task_label_buffer_name)
            valid_indices = (bt != exclude_task).nonzero().squeeze()
            for buffer_name in self.buffers:
                buffers[buffer_name] = getattr(self, buffer_name)[valid_indices]
        else:
            for buffer_name in self.buffers:
                buffers[buffer_name] = getattr(self, buffer_name)[: self.current_index]

        bx = buffers["bx"]
        if bx.size(0) < n_samples:
            return buffers
        indices_np = self.rng.choice(bx.size(0), n_samples, replace=False)
        indices = torch.from_numpy(indices_np).to(self.bx.device)
        return {k[1:]: v[indices] for k, v in buffers.items()}

