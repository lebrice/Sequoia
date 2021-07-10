""" 'Patch' for the Trainer of Pytorch Lightning so it can use gym environment as
dataloaders (via the GymDataLoader class of Sequoia).
"""
import pytorch_lightning as pl
from pytorch_lightning import Trainer as _Trainer

""" Dataclass that holds the options (command-line args) for the Trainer
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Union

import torch
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LightningLoggerBase
from simple_parsing import choice, field

from sequoia.common.hparams import HyperParameters, uniform
from sequoia.common.config import Config
from sequoia.utils.parseable import Parseable
from sequoia.utils.logging_utils import get_logger


logger = get_logger(__file__)


@dataclass
class TrainerConfig(HyperParameters, Parseable):
    """ Configuration dataclass for a pytorch-lightning Trainer.

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
    ) -> Trainer:
        """ Create a Trainer object from the command-line args.
        Adds the given loggers and callbacks as well.
        """
        # FIXME: Trying to subclass the DataConnector to fix issues while iterating
        # over gym envs, that arise because of the _with_is_last() function from
        # lightning.
        from pytorch_lightning.trainer.connectors.data_connector import DataConnector
        import pytorch_lightning.trainer.trainer

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
import tqdm


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

    def run_train(self) -> None:
        self._pre_training_routine()

        if not self.is_global_zero and self.progress_bar_callback is not None:
            self.progress_bar_callback.disable()

        self.run_sanity_check(self.lightning_module)

        self.checkpoint_connector.has_trained = False

        # enable train mode
        self.model.train()
        torch.set_grad_enabled(True)

        from pytorch_lightning.trainer.trainer import Trainer, TrainerStatus, count, distributed_available
        # reload data when needed
        model = self.lightning_module
        self.train_loop.reset_train_val_dataloaders(model)

        # hook
        self.train_loop.on_train_start()

        try:
            if self.train_loop.should_skip_training():
                return
            # run all epochs
            epochs = range(self.current_epoch, self.max_epochs) if self.max_epochs else count(self.current_epoch)
            for epoch in epochs:

                # hook
                self.train_loop.on_train_epoch_start(epoch)

                # with self.profiler.profile("run_training_epoch"):
                    # run train epoch
                    # if isinstance(self.train_dataloader.loaders, gym.Env):
                    #     train_env = self.train_dataloader.loaders
                    #     if has_wrapper(train_env, GymDataLoader):
                    #         self.train_dataloader = train_env
                    
                self.train_loop.run_training_epoch()

                if self.max_steps and self.max_steps <= self.global_step:
                    self.train_loop.on_train_end()
                    return

                # early stopping
                met_min_epochs = (epoch >= self.min_epochs - 1) if self.min_epochs else True
                met_min_steps = self.global_step >= self.min_steps if self.min_steps else True

                if self.should_stop:
                    if met_min_epochs and met_min_steps:
                        self.train_loop.on_train_end()
                        return
                    else:
                        log.info(
                            'Trainer was signaled to stop but required minimum epochs'
                            f' ({self.min_epochs}) or minimum steps ({self.min_steps}) has'
                            ' not been met. Training will continue...'
                        )
                        self.should_stop = False

            # hook
            self.train_loop.on_train_end()

        except KeyboardInterrupt:
            rank_zero_warn('Detected KeyboardInterrupt, attempting graceful shutdown...')
            # user could press Ctrl+c many times... only shutdown once
            if not self.interrupted:
                self.state.status = TrainerStatus.INTERRUPTED
                self.on_keyboard_interrupt()
                # same treatment as below
                self.accelerator.on_train_end()
                self.state.stage = None
        except BaseException:
            self.state.status = TrainerStatus.INTERRUPTED
            if distributed_available() and self.world_size > 1:
                # try syncing remaing processes, kill otherwise
                self.training_type_plugin.reconciliate_processes(traceback.format_exc())
            # give accelerators a chance to finish
            self.accelerator.on_train_end()
            # reset bookkeeping
            self.state.stage = None
            raise


# TODO: Debugging/fixing this buggy method from Pytorch-Lightning.
# import numbers
# import warnings
# from typing import Any

# import torch
# from torch.nn import DataParallel
# from torch.nn.parallel import DistributedDataParallel

# from pytorch_lightning.core.lightning import LightningModule
# from pytorch_lightning.overrides.base import _LightningModuleWrapperBase
# from pytorch_lightning.overrides.distributed import LightningDistributedModule
from pytorch_lightning.utilities import rank_zero_warn
# from pytorch_lightning.utilities.apply_func import apply_to_collection
from functools import singledispatch

# def _apply_to_collection(
#     data: Any,
#     dtype: Union[type, tuple],
#     function: Callable,
#     *args,
#     wrong_dtype: Optional[Union[type, tuple]] = None,
#     **kwargs
# ) -> Any:

import pytorch_lightning.utilities.apply_func
from pytorch_lightning.utilities.apply_func import apply_to_collection
apply_to_collection = singledispatch(apply_to_collection)
setattr(pytorch_lightning.utilities.apply_func, "apply_to_collection", apply_to_collection)

import pytorch_lightning.overrides.data_parallel
setattr(pytorch_lightning.overrides.data_parallel, "apply_to_collection", apply_to_collection)

from sequoia.common import Batch
from sequoia.methods.models.forward_pass import ForwardPass
from typing import Callable, Any, Tuple

@apply_to_collection.register(Batch)
def _apply_to_batch(
    data: Batch,
    dtype: Union[type, tuple],
    function: Callable,
    *args,
    wrong_dtype: Optional[Union[type, tuple]] = None,
    **kwargs
) -> Any:
    # assert False, f"YAY! {type(data)}"
    # logger.debug(f"{type(data)}, {dtype}, {function}, {args}, {wrong_dtype}, {kwargs}")
    return type(data)(**{
        k: apply_to_collection(v, dtype, function, *args, wrong_dtype=wrong_dtype, **kwargs)
        for k, v in data.items()
    })


from pytorch_lightning.trainer.connectors.data_connector import DataConnector

import gym
from sequoia.common.gym_wrappers.utils import has_wrapper, IterableWrapper
from sequoia.settings.rl.continual.environment import GymDataLoader
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.utils.data import DataLoader


class ProfiledEnvironment(IterableWrapper, DataLoader):
    def __iter__(self):
        for i, obs in enumerate(super().__iter__()):
            logger.debug(f"Step {i}, obs.done={obs.done}")
            done = obs.done
            if not isinstance(done, bool) or not done.shape:
                done = all(done)
            done = done or self.is_closed()
            # BUG: The send() doesn't work correctly?
            yield i, (obs, done)

    def send(self, actions):
        assert False, "HEYO!"
        return super().send(actions)


class PatchedDataConnector(DataConnector):
    def get_profiled_train_dataloader(self, train_dataloader):
        # assert False, train_dataloader
        train_dataloader: CombinedLoader
        env = train_dataloader.loaders
        # TODO: Replacing this 'CombinedLoader', since I don't think we need it.
        self.trainer.train_dataloader = env
        
        if has_wrapper(env, GymDataLoader):
            return ProfiledEnvironment(env)
        profiled_dl = self.trainer.profiler.profile_iterable(
            enumerate(prefetch_iterator(train_dataloader)), "get_train_batch"
        )
        return profiled_dl

    def _with_is_last(self, iterable):
        """ Patch for this 'with_is_last' which messes up iterating over an RL env.
        """
        assert False, iterable
        if isinstance(iterable, gym.Env) and has_wrapper(iterable, GymDataLoader):
            env = iterable
            assert isinstance(env, IterableWrapper), env
            while not env.is_closed():
                for step, obs in enumerate(env):
                    if env.is_closed():
                        yield obs, True
                        break
                    else:
                        yield obs, False
        else:
            yield from super()._with_is_last(iterable)
            # it = iter(iterable)
            # last = next(it)
            # if hasattr(last, "done"):
            #     end_of_episode = last["done"]
            #     yield last, end_of_episode
            # for val in it:
            #     # yield last and has next
            #     if hasattr(last, "done"):
            #         end_of_episode = last["done"]
            #         yield last, end_of_episode
            #     else:
            #         yield last, False
            #     last = val
            # yield last, no longer has next
            # yield last, True

import pytorch_lightning.trainer.connectors.data_connector
setattr(pytorch_lightning.trainer.connectors.data_connector, "DataConnector", PatchedDataConnector)
pytorch_lightning.trainer.connectors.data_connector.DataConnector = PatchedDataConnector


import pytorch_lightning.trainer.supporters
from pytorch_lightning.trainer.supporters import prefetch_iterator
prefetch_iterator = singledispatch(prefetch_iterator)
setattr(pytorch_lightning.trainer.supporters, "prefetch_iterator", prefetch_iterator)
from typing import Generator, Iterable

@prefetch_iterator.register(gym.Env)
def prefetch_iterator_for_env(iterable: Iterable) -> Generator[Tuple[Any, bool], None, None]:
    assert False, iterable
    if isinstance(iterable, gym.Env) and has_wrapper(iterable, GymDataLoader):
        env = iterable
        assert isinstance(env, IterableWrapper), env
        while not env.is_closed():
            for step, obs in enumerate(env):
                if env.is_closed():
                    yield obs, True
                    break
                else:
                    yield obs, False
    else:
        yield from prefetch_iterator.dispatch(object)(iterable)

