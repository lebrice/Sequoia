""" 'Patch' for the Trainer of Pytorch Lightning so it can use gym environment as
dataloaders (via the GymDataLoader class of Sequoia).
"""
import os
from dataclasses import dataclass
from functools import singledispatch
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Union

import gym
import pytorch_lightning.trainer.connectors.data_connector
import pytorch_lightning.utilities.apply_func
import torch
from pytorch_lightning import Callback
from pytorch_lightning import Trainer as _Trainer
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.trainer.connectors.data_connector import DataConnector
from pytorch_lightning.trainer.supporters import CombinedLoader
from pytorch_lightning.utilities.apply_func import apply_to_collection
from simple_parsing import choice
from torch.utils.data import DataLoader

from sequoia.common import Batch
from sequoia.common.config import Config
from sequoia.common.gym_wrappers.utils import IterableWrapper, has_wrapper
from sequoia.common.hparams import HyperParameters, uniform
from sequoia.settings.rl.continual.environment import GymDataLoader
from sequoia.settings.sl import PassiveEnvironment
from sequoia.utils.logging_utils import get_logger
from sequoia.utils.parseable import Parseable

logger = get_logger(__file__)


@dataclass
class TrainerConfig(HyperParameters, Parseable):
    """Configuration dataclass for a pytorch-lightning Trainer.

    See the docs for the Trainer from pytorch lightning for more info on the options.

    TODO: Pytorch Lightning already has a mechanism for adding argparse
    arguments for the Trainer.. It would be nice to find a way to use the 'native' way
    of adding arguments in PL in addition to using simple-parsing.
    """

    gpus: int = torch.cuda.device_count()
    overfit_batches: float = 0.0
    fast_dev_run: bool = False

    # Maximum number of epochs to train for.
    max_epochs: int = uniform(1, 100, default=10)

    # Number of nodes to use.
    num_nodes: int = 1
    accelerator: Optional[str] = None
    log_gpu_memory: bool = False

    val_check_interval: Union[int, float] = 1.0

    auto_scale_batch_size: Optional[str] = None
    auto_lr_find: bool = False
    # Floating point precision to use in the model. (See pl.Trainer)
    precision: int = choice(16, 32, default=32)

    default_root_dir: Path = Path(os.environ.get("RESULTS_DIR", os.getcwd() + "/results"))

    # How much of training dataset to check (floats = percent, int = num_batches)
    limit_train_batches: Union[int, float] = 1.0
    # How much of validation dataset to check (floats = percent, int = num_batches)
    limit_val_batches: Union[int, float] = 1.0
    # How much of test dataset to check (floats = percent, int = num_batches)
    limit_test_batches: Union[int, float] = 1.0

    # If ``True``, enable checkpointing.
    # It will configure a default ModelCheckpoint callback if there is no user-defined
    # ModelCheckpoint in the `callbacks`.
    checkpoint_callback: bool = True

    def make_trainer(
        self,
        config: Config,
        callbacks: Optional[List[Callback]] = None,
        loggers: Iterable[LightningLoggerBase] = None,
    ) -> "Trainer":
        """Create a Trainer object from the command-line args.
        Adds the given loggers and callbacks as well.
        """
        # FIXME: Trying to subclass the DataConnector to fix issues while iterating
        # over gym envs, that arise because of the _with_is_last() function from
        # lightning.
        import pytorch_lightning.trainer.trainer
        from pytorch_lightning.trainer.connectors.data_connector import DataConnector

        setattr(pytorch_lightning.trainer.trainer, "DataConnector", DataConnector)
        trainer = Trainer(
            logger=loggers,
            callbacks=callbacks,
            gpus=self.gpus,
            num_nodes=self.num_nodes,
            max_epochs=self.max_epochs,
            accelerator=self.accelerator,
            log_gpu_memory=self.log_gpu_memory,
            overfit_batches=self.overfit_batches,
            fast_dev_run=self.fast_dev_run,
            auto_scale_batch_size=self.auto_scale_batch_size,
            auto_lr_find=self.auto_lr_find,
            # TODO: Either move the log-dir-related stuff from Config to this
            # class, or figure out a way to pass the value from Config to this
            # function
            default_root_dir=self.default_root_dir,
            limit_train_batches=self.limit_train_batches,
            limit_val_batches=self.limit_val_batches,
            limit_test_batches=self.limit_train_batches,
            checkpoint_callback=self.checkpoint_callback,
            profiler=None,  # TODO: Seem to have an impact on the problem below.
        )
        return trainer


class Trainer(_Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, model, train_dataloader=None, val_dataloaders=None, datamodule=None):
        # TODO: Figure out what method to overwrite to fix the problem of accessing two
        # batches in a row in the environment. (with_is_last annoyance.)
        if isinstance(train_dataloader, gym.Env):
            if has_wrapper(train_dataloader, GymDataLoader):
                train_env = train_dataloader
                # raise NotImplementedError("TODO: Fix this.")
        return super().fit(
            model,
            train_dataloader=train_dataloader,
            val_dataloaders=val_dataloaders,
            datamodule=datamodule,
        )


# TODO: Debugging/fixing this buggy method from Pytorch-Lightning.


# def _apply_to_collection(
#     data: Any,
#     dtype: Union[type, tuple],
#     function: Callable,
#     *args,
#     wrong_dtype: Optional[Union[type, tuple]] = None,
#     **kwargs
# ) -> Any:


apply_to_collection = singledispatch(apply_to_collection)
setattr(pytorch_lightning.utilities.apply_func, "apply_to_collection", apply_to_collection)

# import pytorch_lightning.overrides.data_parallel
# setattr(pytorch_lightning.overrides.data_parallel, "apply_to_collection", apply_to_collection)


@apply_to_collection.register(Batch)
def _apply_to_batch(
    data: Batch,
    dtype: Union[type, tuple],
    function: Callable,
    *args,
    wrong_dtype: Optional[Union[type, tuple]] = None,
    **kwargs,
) -> Any:
    # assert False, f"YAY! {type(data)}"
    # logger.debug(f"{type(data)}, {dtype}, {function}, {args}, {wrong_dtype}, {kwargs}")
    return type(data)(
        **{
            k: apply_to_collection(v, dtype, function, *args, wrong_dtype=wrong_dtype, **kwargs)
            for k, v in data.items()
        }
    )


class ProfiledEnvironment(IterableWrapper, DataLoader):
    def __iter__(self):
        for i, obs in enumerate(super().__iter__()):
            # logger.debug(f"Step {i}, obs.done={obs.done}")
            done = obs.done
            if not isinstance(done, bool) or not done.shape:
                # TODO: When we have batch size of 1, or more generally in RL, do we
                # want one call to `trainer.fit` to last a given number of episodes ?
                # TODO: Look into the `max_steps` argument to Trainer.
                done = all(done)
            # done = done or self.is_closed()
            done = self.is_closed()
            yield i, (obs, done)


class PatchedDataConnector(DataConnector):
    def get_profiled_train_dataloader(self, train_dataloader: DataLoader):
        if isinstance(train_dataloader, CombinedLoader) and isinstance(
            train_dataloader.loaders, gym.Env
        ):
            env = train_dataloader.loaders
            # TODO: Replacing this 'CombinedLoader' on the Trainer with the env, since I
            # don't think we need it (not using multiple train dataloaders with PL atm.)
            self.trainer.train_dataloader = env
            if not isinstance(env.unwrapped, PassiveEnvironment):
                # Only really need to do this 'profile' thing for 'active' environments.
                return ProfiledEnvironment(env)
        else:
            # This gets called before each epoch, so we get here on the start of the
            # second training epoch.
            # TODO: Check that this isn't causing issues between tasks
            assert train_dataloader is self.trainer.train_dataloader

        profiled_dl = self.trainer.profiler.profile_iterable(
            enumerate(prefetch_iterator(train_dataloader)), "get_train_batch"
        )
        return profiled_dl


setattr(
    pytorch_lightning.trainer.connectors.data_connector,
    "DataConnector",
    PatchedDataConnector,
)
pytorch_lightning.trainer.connectors.data_connector.DataConnector = PatchedDataConnector
