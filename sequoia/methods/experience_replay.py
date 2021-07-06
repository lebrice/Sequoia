""" Method that uses a replay buffer to prevent forgetting.

TODO: Refactor this to be based on the BaseMethod, possibly using an auxiliary task for
the Replay.
"""
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Type, Any, List
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
from torchvision.models import ResNet
from wandb.wandb_run import Run

from sequoia.methods import register_method
from sequoia.settings import ClassIncrementalSetting
from sequoia.settings.base import Actions, Environment, Method, Observations
from sequoia.utils import get_logger


logger = get_logger(__file__)


@register_method
@dataclass
class ExperienceReplayMethod(Method, target_setting=ClassIncrementalSetting):
    """ Simple method that uses a replay buffer to reduce forgetting.
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        buffer_capacity: int = 200,
        max_epochs_per_task: int = 10,
        weight_decay: float = 1e-6,
        seed: int = None,
    ):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.buffer_capacity = buffer_capacity

        self.net: ResNet
        self.buffer: Optional[Buffer] = None
        self.optim: torch.optim.Optimizer
        self.task: int = 0
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        if seed:
            torch.manual_seed(seed)
            torch.set_deterministic(True)

        self.epochs_per_task: int = max_epochs_per_task
        self.early_stop_patience: int = 2

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def configure(self, setting: ClassIncrementalSetting):
        # create the model
        self.net = models.resnet18(pretrained=False)
        self.net.fc = nn.Linear(512, setting.action_space.n)
        if torch.cuda.is_available():
            self.net = self.net.to(device=self.device)
        # Set drop_last to True, to avoid getting a batch of size 1, which makes
        # batchnorm raise an error.
        setting.drop_last = True
        image_space: spaces.Box = setting.observation_space["x"]
        # Create the buffer.
        if self.buffer_capacity:
            self.buffer = Buffer(
                capacity=self.buffer_capacity,
                input_shape=image_space.shape,
                extra_buffers={"t": torch.LongTensor},
                rng=self.rng,
            ).to(device=self.device)
        # Create the optimizer.
        self.optim = torch.optim.Adam(
            self.net.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def fit(self, train_env: Environment, valid_env: Environment):
        self.net.train()
        # Simple example training loop, not using the validation loader.
        best_val_loss = np.inf
        best_epoch = 0

        for epoch in range(self.epochs_per_task):
            train_pbar = tqdm.tqdm(train_env, desc=f"Training Epoch {epoch}")
            postfix = {}

            obs: ClassIncrementalSetting.Observations
            rew: ClassIncrementalSetting.Rewards
            for i, (obs, rew) in enumerate(train_pbar):
                self.optim.zero_grad()

                obs = obs.to(device=self.device)
                x = obs.x

                # FIXME: Batch norm will cause a crash if we pass x with batch_size==1!
                fake_batch = False
                if x.shape[0] == 1:
                    # Pretend like this has batch_size of 2 rather than just 1.
                    x = x.tile([2, *(1 for _ in x.shape[1:])])
                    x[1] += 1  # Just so the two samples aren't identical, otherwise
                    # maybe the batch norm std would be nan or something.
                    fake_batch = True
                logits = self.net(x)
                if fake_batch:
                    logits = logits[:1]  # Drop the 'fake' second item.

                if rew is None:
                    # If our online training performance is being measured, we might
                    # need to provide actions before we can get the corresponding
                    # rewards (image labels in this case).
                    y_pred = logits.argmax(1)
                    rew = train_env.send(y_pred)

                rew = rew.to(device=self.device)
                y = rew.y
                loss = F.cross_entropy(logits, y)

                postfix["loss"] = loss.detach().item()
                if self.task > 0 and self.buffer:
                    b_samples = self.buffer.sample(x.size(0))
                    b_logits = self.net(b_samples["x"])
                    loss_replay = F.cross_entropy(b_logits, b_samples["y"])
                    loss += loss_replay
                    postfix["replay loss"] = loss_replay.detach().item()

                loss.backward()
                self.optim.step()

                train_pbar.set_postfix(postfix)

                # Only add new samples to the buffer (only during first epoch).
                if self.buffer and epoch == 0:
                    self.buffer.add_reservoir({"x": x, "y": y, "t": self.task})

            # Validation loop:
            self.net.eval()
            torch.set_grad_enabled(False)
            val_pbar = tqdm.tqdm(valid_env)
            val_pbar.set_description(f"Validation Epoch {epoch}")
            epoch_val_loss = 0.0
            epoch_val_loss_list: List[float] = []

            for i, (obs, rew) in enumerate(val_pbar):
                obs = obs.to(device=self.device)
                x = obs.x
                logits = self.net(x)

                if rew is None:
                    y_pred = logits.argmax(-1)
                    rew = valid_env.send(y_pred)

                assert rew is not None
                rew = rew.to(device=self.device)
                y = rew.y
                val_loss = F.cross_entropy(logits, y).item()

                epoch_val_loss_list.append(val_loss)
                postfix["validation loss"] = val_loss
                val_pbar.set_postfix(postfix)
            torch.set_grad_enabled(True)
            epoch_val_loss_mean = np.mean(epoch_val_loss_list)

            if epoch_val_loss_mean < best_val_loss:
                best_val_loss = epoch_val_loss_mean
                best_epoch = epoch
            if epoch - best_epoch > self.early_stop_patience:
                print(f"Early stopping at epoch {epoch}.")
                # TODO: Reload the weights from the best epoch.
                break

    def get_actions(
        self, observations: Observations, action_space: gym.Space
    ) -> Actions:
        observations = observations.to(device=self.device)
        logits = self.net(observations.x)
        pred = logits.argmax(1)
        return pred  # Note: Here it's also fine to just return the predictions.

    def on_task_switch(self, task_id: Optional[int]):
        print(f"Switching from task {self.task} to task {task_id}")
        if self.training:
            self.task = task_id

    @classmethod
    def add_argparse_args(cls, parser: ArgumentParser, dest: str = "") -> None:
        """Add the command-line arguments for this Method to the given parser.

        Parameters
        ----------
        parser : ArgumentParser
            The ArgumentParser.
        dest : str, optional
            The 'base' destination where the arguments should be set on the
            namespace, by default empty, in which case the arguments can be at
            the "root" level on the namespace.
        """
        prefix = f"{dest}." if dest else ""
        parser.add_argument(f"--{prefix}learning_rate", type=float, default=1e-3)
        parser.add_argument(f"--{prefix}weight_decay", type=float, default=1e-6)
        parser.add_argument(f"--{prefix}buffer_capacity", type=int, default=200)
        parser.add_argument(f"--{prefix}max_epochs_per_task", type=int, default=10)
        parser.add_argument(
            f"--{prefix}seed", type=int, default=None, help="Random seed"
        )

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
            max_epochs_per_task=args.max_epochs_per_task,
            weight_decay=args.weight_decay,
            seed=args.seed,
        )

    def get_search_space(self, setting: ClassIncrementalSetting) -> Dict:
        return {
            "learning_rate": "loguniform(1e-4, 5e-1, default_value=1e-3)",
            "buffer_capacity": "uniform(1000, 100_000, default_value=10_000, discrete=True)",
            "weight_decay": "loguniform(1e-12, 1e-3, default_value=1e-6)",
            "early_stop_patience": "uniform(0, 2, default_value=1, discrete=True)",
        }

    def adapt_to_new_hparams(self, new_hparams: Dict[str, Any]) -> None:
        """Adapts the Method when it receives new Hyper-Parameters to try for a new run.

        It is required that this method be implemented if you want to perform HPO sweeps
        with Orion.

        NOTE: It is very strongly recommended that you always re-create your model and
        any modules / components that depend on these hyper-parameters inside the
        `configure` method! (Otherwise these new hyper-parameters will not be used in
        the next run)

        Parameters
        ----------
        new_hparams : Dict[str, Any]
            The new hyper-parameters being recommended by the HPO algorithm. These will
            have the same structure as the search space.
        """
        # Here we overwrite the corresponding attributes with the new suggested values
        # leaving other fields unchanged.
        # NOTE: These new hyper-paramers will be used in the next run in the sweep,
        # since each call to `configure` will create a new Model.
        self.learning_rate = new_hparams["learning_rate"]
        self.weight_decay = new_hparams["weight_decay"]
        self.buffer_capacity = new_hparams["buffer_capacity"]

    def setup_wandb(self, run: Run) -> None:
        """ Called by the Setting when using Weights & Biases, after `wandb.init`.

        This method is here to provide Methods with the opportunity to log some of their
        configuration options or hyper-parameters to wandb.

        NOTE: The Setting has already set the `"setting"` entry in the `wandb.config` by
        this point.

        Parameters
        ----------
        run : wandb.Run
            Current wandb Run.
        """
        run.config.update(
            dict(
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
                buffer_capacity=self.buffer_capacity,
                epochs_per_task=self.epochs_per_task,
                seed=self.seed,
            )
        )


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
            assert hasattr(self, "bt")
            valid_indices = (self.bt != exclude_task).nonzero().squeeze()
            for buffer_name in self.buffers:
                buffers[buffer_name] = getattr(self, buffer_name)[valid_indices]
        else:
            for buffer_name in self.buffers:
                buffers[buffer_name] = getattr(self, buffer_name)[: self.current_index]

        bx = buffers["bx"]
        if bx.size(0) < n_samples:
            return buffers
        else:
            indices_np = self.rng.choice(bx.size(0), n_samples, replace=False)
            indices = torch.from_numpy(indices_np).to(self.bx.device)
            return {k[1:]: v[indices] for (k, v) in buffers.items()}


if __name__ == "__main__":
    ExperienceReplayMethod.main()
