import inspect
import json
import logging
from itertools import tee
from collections import OrderedDict, defaultdict
from dataclasses import InitVar, asdict, dataclass, is_dataclass
from pathlib import Path
from typing import (Any, ClassVar, Dict, Generator, Iterable, List, Optional,
                    Tuple, Type, Union, Iterator, Callable)

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
import tqdm
import wandb
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Sampler, TensorDataset
from torchvision.datasets import VisionDataset

from common.losses import LossInfo, TrainValidLosses, get_supervised_metrics, get_supervised_accuracy
from common.metrics import (ClassificationMetrics, Metrics, RegressionMetrics,
                            get_metrics)
from config import Config as ConfigBase

from datasets import Datasets, DatasetConfig

from datasets.data_utils import train_valid_split
from datasets.subset import ClassSubset, Subset
from models.classifier import Classifier
from simple_parsing import choice, field, mutable_field, subparsers
from simple_parsing.helpers import FlattenedAccess
from utils.json_utils import Serializable
from tasks import AuxiliaryTask, Tasks
from utils import utils
from utils.early_stopping import EarlyStoppingOptions, early_stopping
from utils.logging_utils import cleanup, get_logger, pbar
from utils.save_job import SaverWorker, SaveTuple, save
from utils.utils import add_prefix, common_fields, is_nonempty_dir

logger = get_logger(__file__)

@dataclass
class ExperimentBase(Serializable):
    """Base-class for an Experiment.

    Important attributes:
    - `config` (ExperimentBase.Config): Contains all the settings related to an 
        experiment which don't have any influence on model performance (e.g.
        log_dir, log_interval, wandb stuff, etc.).
        NOTE: Can also be used subclassed to configure the evaluation setup.
        For instance, in a task-incremental setup, can be used to configure the
        number of classes seen per task, or the size of a replay buffer, etc. 
    
    - `hparams` (Classifier.HParams): The Hyper-parameters of the experiment.
        It's attributes should have an impact on the performance of the model on
        the evaluated task.

    - `state` (ExperimentBase.State): Holds the state of an experiment, used for
        checkpointing and resuming runs.

    NOTE: We also use some experiment addons, which act as mixins to extend this
    class and add some optional, use-case specific behaviour.
    For instance, one addon adds a "callback" to periodically save some fake
    images when using a generative auxiliary task.
    Another adds settings related to Replay, etc.

    TODO: Get some feedback about the inheritance setup used for Addons.
    """
    @dataclass
    class Config(ConfigBase):
        """ Configuration of an Experiment.

        These settings shouldn't need to be tuned.
        These attributes will be parsed from the command-line using simple-parsing.
        """
        # Which dataset to use.
        dataset: DatasetConfig = choice({
            d.name: d.value for d in Datasets
        }, default=Datasets.cifar100.name)

        # Path to restore the state from at the start of training.
        # NOTE: Currently, should point to a json file, with the same format as the one created by the `save()` method.
        restore_from_path: Optional[Path] = None

        restart_from_cp: bool = True #wether to reload from checkpoint

        #valid_fraction
        valid_fraction: float = 0.1

    @dataclass
    class State(Serializable):
        """ Dataclass used to store the state of an experiment.
        
        This object should contain everything we want to be able to save/restore.
        NOTE: We aren't going to parse these from the command-line.
        """
        global_step: int = 0
        model_weights_path: Optional[Path] = None
        # Container for train/valid losses that are logged periodically.
        all_losses: TrainValidLosses = mutable_field(TrainValidLosses, repr=False)

        def __post_init__(self):
            self.global_step = self.all_losses.latest_step()

    # Experiment Config: non-tunable parameters specific to an experiment.
    config: Config = mutable_field(Config)
    # Model Hyper-parameters (tunable) settings.
    hparams: Classifier.HParams = mutable_field(Classifier.HParams)
    # State of the experiment (not parsed form command-line).
    state: State = mutable_field(State, init=False, metadata=dict(to_dict=False))

    def __post_init__(self):
        """ Called after __init__, used to initialize all missing fields.
        
        You can use this method to initialize the fields that aren't parsed from
        the command-line, such as `model`, etc.
        
        NOTE: The fields created here are not included in the result of
        `self.to_dict()`, therefore they will also not be logged to wandb 
        when `wandb.init` is called.
        """
        if isinstance(self.config.device, tuple):
            if len(self.config.device)==1:
                self.config.device = self.config.device[-1]

        self.train_dataset: Dataset = NotImplemented
        self.valid_dataset: Dataset = NotImplemented
        self.test_dataset: Dataset = NotImplemented

        self.train_loader: DataLoader = NotImplemented
        self.valid_loader: DataLoader = NotImplemented
        self.test_loader: DataLoader = NotImplemented
        if self.config.debug:
            logger.setLevel(logging.DEBUG)
        
        # Background queue and worker for saving stuff to disk asynchronously.

        ctx = mp.get_context("spawn") 
        self.background_queue: ctx.Queue = ctx.Queue()
        self.saver_worker: Optional[SaverWorker] = None

    def __del__(self):
        print("Destroying the 'Experiment' object.")
        self.cleanup()

    def run(self):
        raise NotImplementedError("Implement your own run method in a derived class!")

    def launch(self):
        """ Launches the experiment.
        
        TODO: Clean this up. It isn't clear exactly where the separation is
        between the Experiment.run() method and this one.
        """
        if self.config.verbose:
            logger.info(f"Experiment: {self.dumps_yaml(indent=4)}")
            print("=" * 40)
        
        if self.config.use_wandb:
            config_dict = self.to_dict()
            # Take out the `state` key from the config dict:
            config_dict.pop("state", None)
            run = self.config.wandb_init(config_dict=config_dict)
        

        logger.info(f"Launching experiment at log dir {self.config.log_dir}")

        if self.done:
            logger.info(f"Experiment is already done! Exiting.")
            exit()

        if self.started:
            logger.info(f"Experiment is incomplete at directory {self.config.log_dir}")
            # TODO: Resume an interrupted experiment
        try:
            logger.info("-" * 10 + f"Starting experiment '{type(self).__name__}' " + "-" * 10)

            self.run()

            logger.info("-" * 10 + f"Experiment '{type(self).__name__}' is done." + "-" * 10)
            self.cleanup()
        
        except Exception as e:
            print(f"Experiment crashed: {e}")
            raise e

    def setup(self):
        """Prepare everything before training begins: Saves/restores state,
        loads the datasets, model weights, etc.
        """
        # Create the model
        self.model = self.init_model()

        # Save the Config, Hparams and State(?) to a file.
        # TODO: should we also save the state here?
        # TODO: This could potentially overwrite an existing file!
        self.config.log_dir.mkdir(exist_ok=True, parents=True)
        self.checkpoints_dir.mkdir(exist_ok=True, parents=True)

        self.save(self.config.log_dir / "experiment.yaml")

        # If the experiment was already started, or a 'restore_from' argument
        # was passed:
        if self.started or self.config.restore_from_path is not None and self.config.restart_from_cp:
            logger.info(f"Experiment was already started in the past.")
            self.load_state(self.config.restore_from_path)
        elif self.done:
            logger.info(f"Experiment is already done.")
        else:
            # We are starting fresh, so we save the fresh new state, but not the
            # model weights.
            self.save_state(save_model_weights=False)

    def cleanup(self):
        print("Cleaning up after the experiment is done.")
        if hasattr(self, "saver_worker") and self.saver_worker:
            self.background_queue.put(None)
            self.saver_worker.join(timeout=120)
        print("Successfully closed everything")

    def load_datasets(self, valid_fraction: float=0.1, train_transform = None, valid_transform = None, test_transform = None) -> Tuple[Dataset, Optional[Dataset], Dataset]:
        """ Loads the training, validation and test datasets. """
        train_dataset, valid_dataset, test_dataset = self.config.dataset.load(data_dir=self.config.data_dir, train_transform = train_transform, valid_transform=valid_transform, test_transform=test_transform)
        #valid_dataset: Optional[Dataset] = None
        valid_fraction = self.config.valid_fraction
        if valid_fraction > 0:
            train_dataset, valid_dataset = train_valid_split(train_dataset, valid_dataset, valid_fraction)
        return train_dataset, valid_dataset, test_dataset

    def load_state(self, state_json_path: Path=None) -> None:
        """ save/restore the state from a previous run. """
        state_json_path = state_json_path or self.checkpoints_dir / "state.json"
        logger.info(f"Restoring state from {state_json_path}")
        # Load the 'State' object from the json file
        self.state = self.State.load_json(state_json_path)
        if self.config.debug and self.config.verbose:
            logger.debug(f"state: {self.state}")
        
        if self.state is None:
            raise RuntimeError(
                f"State shouldn't be None!\n"
                f"(Tried to load from {state_json_path})"
            )

        if self.state.model_weights_path:
            logger.info(f"Restoring model weights from {self.state.model_weights_path}")
            state_dict = torch.load(
                self.state.model_weights_path,
                map_location=self.config.device,
            )
            self.model.load_state_dict(state_dict)
        # TODO: Make sure that restoring at some arbitrary global_step works.
        logger.info(f"Starting at global step = {self.state.global_step}.")

    def init_model(self) -> Classifier:
        print("init model")
        from models import get_model_class_for_dataset
        model_class = get_model_class_for_dataset(self.config.dataset)
        model = model_class(hparams=self.hparams, config=self.config)
        
        return model

    def train(self,
              train_dataloader: Union[Dataset, DataLoader, Iterator],                
              valid_dataloader: Union[Dataset, DataLoader, Iterator],    
              test_dataloader:Union[Dataset, DataLoader, Iterator],
              epochs: int,                
              description: str=None,
              early_stopping_options: EarlyStoppingOptions=None,
              use_accuracy_as_metric: bool=None,                
              temp_save_dir: Path=None,
              steps_per_epoch: int = None) -> TrainValidLosses:
        """Trains on the `train_dataloader` and evaluates on `valid_dataloader`.

        Periodically evaluates on validation batches during each epoch, as well
        as doing a full pass through the validation dataset after each epoch.
        The weights at the point at which the model had the best validation
        performance are always re-loaded at the end.
        
        NOTE: If `early_stopping_options` is None, then the value from
        `self.config.early_stopping` is used. Same goes for
        `use_accuracy_as_metric`.
        
        NOTE: The losses are no logged to wandb, so you should log them yourself
        after this method completes.

        TODO: Add a way to resume training if it was previously interrupted.
        For instance, it might be useful to keep track of the number of epochs
        performed in the current task (for TaskIncremental)

        TODO: save/load the `all_losses` object to temp_save_dir at a given
        interval during training, using a saver thread.


        Args:
            - train_dataloader (Union[Dataset, DataLoader]): Training dataset or
                dataloader.
            - valid_dataloader (Union[Dataset, DataLoader]): [description]
            - epochs (int): Number of epochs to train for. 
            - description (str, optional): A description to use in the
                progressbar. Defaults to None.
            - early_stopping_options (EarlyStoppingOptions, optional): Options
                for configuring the early stopping hook. Defaults to None.
            - use_accuracy_as_metric (bool, optional): If `True`, accuracy will
                be used as a measure of performance. Otherwise, the total
                validation loss is used. Defaults to False.
            - temp_save_file (Path, optional): Path where the intermediate state
                should be saved/restored from. Defaults to None.

        Returns:
            TrainValidLosses: An object containing the training and validation
            losses during training (every `log_interval` steps) to be logged.
        """                   
        
        if isinstance(train_dataloader, Dataset):
            train_dataloader = self.get_dataloader(train_dataloader)
        if isinstance(valid_dataloader, Dataset):
            valid_dataloader = self.get_dataloader(valid_dataloader)
        
        early_stopping_options = early_stopping_options or self.config.early_stopping

        if use_accuracy_as_metric is None:
            use_accuracy_as_metric = self.config.use_accuracy_as_metric

        # The --debug_steps argument can be used to shorten the dataloaders.
        steps_per_epoch = len(train_dataloader) if steps_per_epoch is None else steps_per_epoch
        if self.config.debug_steps:
            from itertools import islice
            steps_per_epoch = self.config.debug_steps
            train_dataloader = islice(train_dataloader, 0, steps_per_epoch)  # type: ignore
        logger.debug(f"Steps per epoch: {steps_per_epoch}")
        
        # LossInfo objects at each step of validation        
        validation_losses: List[LossInfo] = []
        # Container for the train and valid losses every `log_interval` steps.
        all_losses = TrainValidLosses()
        
        if temp_save_dir:
            temp_save_dir.mkdir(exist_ok=True, parents=True)
            
            all_losses_path = temp_save_dir / "all_losses.json"
            if all_losses_path.exists():
                logger.info(f"Loading all_losses from {all_losses_path}")
                all_losses = TrainValidLosses.load_json(all_losses_path)

            from itertools import count
            for i in count(start=1):
                loss_path = temp_save_dir / f"val_loss_{i}.json"
                if not loss_path.exists():
                    break
                else:
                    assert len(validation_losses) == (i-1)
                    validation_loss = LossInfo.load_json(loss_path)
                    validation_losses.append(validation_loss)

            logger.info(f"Reloaded {len(validation_losses)} existing validation losses")
            logger.info(f"Latest step: {all_losses.latest_step()}.")
        
        # Get the latest step
        # NOTE: At the moment, will always be zero, but if we reload
        # `all_losses` from a file, would give you the step to start from.
        starting_step = all_losses.latest_step() or self.state.global_step
        starting_epoch = len(validation_losses) + 1

        if early_stopping_options:
            logger.info(f"Using early stopping with options {early_stopping_options}")
        
        # Hook to keep track of the best model.
        best_model_watcher = self.keep_best_model(
            use_acc=use_accuracy_as_metric,
            save_path=self.checkpoints_dir / f"best_model{description.replace(' ', '_').lower()}.pth",
        )        
        best_step = starting_step
        best_epoch = starting_epoch
        next(best_model_watcher) # Prime the generator
        
        # Hook to test for convergence.
        convergence_checker = early_stopping(
            options=early_stopping_options,
            use_acc=use_accuracy_as_metric,
        )
        next(convergence_checker) # Prime the generator
        # Hook for periodically evaluating the performance on batches from the
        # validation dataset during training.
        if valid_dataloader is not None:
            valid_loss_gen = self.valid_performance_generator(valid_dataloader)
        else: 
            valid_loss_gen = None
        
        # List to hold the length of each epoch (should all be the same length)
        epoch_lengths: List[int] = []
        for epoch in range(starting_epoch, epochs + 1):

            #prevent iterator exhaustion
            if isinstance(train_dataloader, Iterator):
                train_dataloader, train_dataloader_ = tee(train_dataloader)
            else: 
                train_dataloader_ = train_dataloader

            epoch_start_step = self.state.global_step     
            pbar = tqdm.tqdm(train_dataloader_, total=steps_per_epoch)
            desc = description or "" 
            desc += " " if desc and not desc.endswith(" ") else ""
            desc += f"Epoch {epoch}"
            pbar.set_description(desc + " Train")
            self.train_epoch(epoch, pbar, valid_loss_gen, all_losses=all_losses)
            
            epoch_length = self.global_step - epoch_start_step
            epoch_lengths.append(epoch_length)

            if epoch % self.config.log_interval_test_epochs == 0:
                if test_dataloader is not None and self.config.log_interval_test_epochs>0:
                    # perform a test epoch every n iterations.
                    test_desc = desc + " Test"
                    test_loss_info = self.test(test_dataloader, description=test_desc, name="Test_full")
                    self.log({'Test_full':test_loss_info.to_dict()})

            # perform a validation epoch.
            if valid_dataloader is not None:
                val_desc = desc + " Valid"
                val_loss_info = self.test(valid_dataloader, description=val_desc, name="Valid_full")
                validation_losses.append(val_loss_info)

            

                if temp_save_dir:
                    # Save these files in the background using the saver process.
                    self.save_job(val_loss_info, temp_save_dir / f"val_loss_{i}.json")
                    self.save_job(all_losses, temp_save_dir / f"all_losses.json")
                
                # Inform the best model watcher of the latest performance of the model.
                best_step = best_model_watcher.send(val_loss_info)
                logger.debug(f"Best step so far: {best_step}")

                best_epoch = best_step // int(np.mean(epoch_lengths))
                logger.debug(f"Best epoch so far: {best_epoch}")

                converged = convergence_checker.send(val_loss_info)
            else:
                converged = False

            if converged:
                logger.info(f"Training Converged at epoch {epoch}. Best valid performance was at epoch {best_epoch}")
                break
        try:
            # Re-load the best weights
            best_model_watcher.send(None)
        except StopIteration:
            pass
        
        try:
            convergence_checker.close()
            best_model_watcher.close()
            valid_loss_gen.close()
        except:
            pass
        
        logger.info(f"Best step: {best_step}, best_epoch: {best_epoch}, ")
        all_losses.keep_up_to_step(best_step)

        # TODO: Should we also return the array of validation losses at each epoch (`validation_losses`)?
        return all_losses
    
    def train_epoch(self, epoch, pbar: Iterable, valid_loss_gen: Generator, all_losses:TrainValidLosses) -> Tuple[LossInfo, LossInfo]:
        # Message for the progressbar
        message: Dict[str, Any] = OrderedDict()
        for batch_idx, train_loss in enumerate(self.train_iter(pbar)):
            train_loss.drop_tensors()
            if batch_idx % self.config.log_interval == 0:
                # get loss on a batch of validation data:if 
                if valid_loss_gen is not None:
                    valid_loss = next(valid_loss_gen)
                    valid_loss.drop_tensors()
                    message.update(valid_loss.to_pbar_message())
                else:
                    valid_loss = LossInfo(total_loss=0)

                all_losses[self.state.global_step] = (train_loss, valid_loss)

                message.update(train_loss.to_pbar_message())
                
                pbar.set_postfix(message)
                self.log({
                    "Train": train_loss,
                    "Valid": valid_loss,
                })
    
    def load_weights(self, load_path):
        try:
            state_dict = torch.load(load_path, map_location=self.config.device)
            self.model.load_state_dict(state_dict)
            return True
        except FileNotFoundError:
            logger.info(f"Tried to load the self-sup. pretrained model from {load_path}, but it doesnt exist yet.")
            return False

            
    def save_weights(self, save_path:Path):
            state_dict = self.model.state_dict()
            torch.save(state_dict, str(save_path))

    def keep_best_model(self, use_acc: bool=False, save_path: Path=None) -> Generator[int, Optional[LossInfo], None]:
        # Path where the best model weights will be saved.
        save_path = save_path or self.checkpoints_dir / "model_best.pth"
        # Temporary file, used to make sure we don't corrupt the best weights
        # file if the script is killed while copying stuff.
        save_path_tmp = save_path.with_suffix(".tmp")

        def save_weights():
            state_dict = self.model.state_dict()
            torch.save(state_dict, save_path_tmp)
            save_path_tmp.replace(save_path)
            self.state.model_weights_path = save_path
            logger.info(f"Saved best model weights to path {save_path}.")
        
        best_perf: Optional[float] = None
        
        step = self.state.global_step
        best_step: int = step

        loss_info: Optional[LossInfo] = (yield step)

        while loss_info is not None:
            step = self.state.global_step

            val_loss = loss_info.total_loss.item()
            
            if use_acc:
                supervised_metrics = get_supervised_metrics(loss_info)
                assert supervised_metrics, "Can't use accuracy since there are no supervised metrics in given loss.."
                val_acc = supervised_metrics.accuracy

            if use_acc and (best_perf is None or val_acc > best_perf):
                best_step = step
                best_perf = val_acc
                logging.info(f"New best model at step {step}, Val Acc: {val_acc}")
                save_weights()
            elif not use_acc and  (best_perf is None or val_loss < best_perf):
                best_step = step
                best_perf = val_loss
                logging.info(f"New best model at step {step}, Val Loss: {val_loss}")
                save_weights()
            else:
                # Model at current step is not the best model.
                pass

            loss_info = yield best_step

        # Reload the weights of the best model.
        logger.info(f"Reloading weights of the best model (global step: {best_step})")
        self.load_weights(save_path)

    def valid_performance_generator(self, valid_dataloader: Union[Dataset, DataLoader]) -> Generator[LossInfo, None, None]:
        if isinstance(valid_dataloader, Dataset):
            valid_dataloader = self.get_dataloader(valid_dataloader)
        while True:
            for batch in valid_dataloader:
                data, target = self.preprocess(batch)
                yield self.test_batch(data, target, name="Valid")
        logger.info("Somehow exited the infinite while loop!")

    def train_iter(self, dataloader: DataLoader) -> Iterable[LossInfo]:
        self.model.train()
        for batch in dataloader:
            data, target = self.preprocess(batch)
            yield self.train_batch(data, target)

    def preprocess(self, batch: Union[Tuple[Tensor], Tuple[Tensor, Tensor], Tuple[Tuple[Tensor, Tensor], Tensor]]) -> Tuple[Tensor, Optional[Tensor]]:
        data = batch[0].to(self.model.in_device)
        target = batch[1].to(self.model.out_device) if len(batch) == 2 else None  # type: ignore
        return data, target

    def train_batch(self, data: Tensor, target: Optional[Tensor], name: str="Train") -> LossInfo:
        self.model.train()
        self.model.optimizer.zero_grad()

        batch_loss_info = self.model.get_loss(data, target, name=name)
        total_loss = batch_loss_info.total_loss
        total_loss.backward()

        self.step(global_step=self.global_step)

        self.global_step += data.shape[0]
        return batch_loss_info


    def test(self, dataloader: Union[Dataset, DataLoader], description: str=None, name: str="Test") -> LossInfo:
        if isinstance(dataloader, Dataset):
            dataloader = self.get_dataloader(dataloader)

        pbar = tqdm.tqdm(dataloader)
        desc = (description or "Test Epoch")    
        
        pbar.set_description(desc)
        total_loss = LossInfo(name)
        message: Dict[str, Any] = OrderedDict()

        for batch_idx, loss in enumerate(self.test_iter(pbar)):
            total_loss += loss

            if batch_idx % self.config.log_interval == 0:
                message.update(total_loss.to_pbar_message())
                pbar.set_postfix(message)
        
        total_loss.drop_tensors()
        return total_loss.detach()

    def test_iter(self, dataloader: DataLoader) -> Iterable[LossInfo]:
        #self.model.eval()
        for batch in dataloader:
            data, target = self.preprocess(batch)
            yield self.test_batch(data, target)

    def test_batch(self, data: Tensor, target: Tensor=None, name: str="Test") -> LossInfo:
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            loss = self.model.get_loss(data, target, name=name)
        if was_training:
            self.model.train()
        return loss
    
    def step(self, global_step:int, **kwargs):
        return self.model.optimizer_step(global_step=self.global_step, **kwargs)


    def get_dataloader(self, dataset: Dataset, sampler: Sampler=None, shuffle: bool=True, batch_size:int = None) -> DataLoader:
        if sampler is None:
            return DataLoader(
                dataset,
                batch_size = batch_size if batch_size is not None else self.hparams.batch_size,
                shuffle=shuffle,
                num_workers=self.config.num_workers,
                pin_memory=self.config.use_cuda,
            )
        else:
            return DataLoader(
                dataset,
                batch_size = batch_size if batch_size is not None else self.hparams.batch_size,
                sampler=sampler,
                num_workers=self.config.num_workers,
                pin_memory=self.config.use_cuda,
            )

    def save_experiment(self, save_path: Path, blocking: bool=False):
        save_dir = save_path if save_path.is_dir() else save_path
        save_dir.mkdir(exist_ok=True, parents=True)

        saved_weights_path = save_dir / "model_weights.pth"
        state_dict = {
            k: t.detach().cpu() for k, t in self.model.state_dict().items()
        }
        self.save_job(obj=state_dict, path=saved_weights_path, blocking=blocking)
        # Update the `state` attribute to point to the new checkpoints file. 
        self.state.model_weights_path = saved_weights_path

    def save_state(self, save_dir: Path=None, save_model_weights: bool=True, blocking: bool=True) -> None:
        """Saves the state of the experiment to the directory.

        Args:
            save_dir (Path, optional): Dicretory to save results in. Defaults to
               None, in which case `self.checkpoints_dir` is used.
            save_model_weights (bool, optional): [description]. Defaults to True.
            blocking (bool, optional): [description]. Defaults to True.
        """
        # Use checkpoints dir if save_dir is not given.
        save_dir = save_dir or self.checkpoints_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        if save_model_weights:
            saved_weights_path = save_dir / "model_weights.pth"
            state_dict = {
                k: t.detach().cpu() for k, t in self.model.state_dict().items()
            }
            self.save_job(obj=state_dict, path=saved_weights_path, blocking=blocking)
            # Update the `state` attribute to point to the new checkpoints file. 
            self.state.model_weights_path = saved_weights_path
        self.save_job(obj=self.state, path=save_dir / "state.json", blocking=blocking)

    def save_job(self, obj: Any, path: Path, blocking: bool=True) -> None:
        """Save the object `obj` to path `path`.

        If `blocking` is False, uses a background process. Otherwise, blocks
        until saving is complete.

        Args:
            path (Path): Path to save to.
            obj (Any): object to save. (if JsonSerializable, will be saved to json)
            blocking (bool, optional): Wether to wait for the operation to
                finish, or to delegate to a background process. Defaults to False.
        """
        assert isinstance(path, Path), f"positional argument 'path' should be a Path! (got {path})"
        if blocking:
           save(obj, save_path=path)
        else:
            if self.saver_worker is None:
                self.saver_worker = SaverWorker(self.config, self.background_queue)
            if not self.saver_worker.is_alive():
                self.saver_worker.start()
            self.background_queue.put(SaveTuple(save_path=path, obj=obj))


    def log(self, message: Dict[str, Any], step: int=None, once: bool=False, prefix: str=""):
        for k, v in message.items():
            if isinstance(v, (LossInfo, Metrics)):
                message[k] = v.to_log_dict()

        message = cleanup(message, sep="/")

        if prefix:
            message = utils.add_prefix(message, prefix)
        
        # if we want to long once (like a final result, step should be None)
        # else, if not given, we use the global step.
        step = None if once else (step or self.global_step)
        
        if self.config.use_wandb:
            wandb.log(message, step=step)

        if self.config.debug and self.config.verbose:
            print(message)

    def _folder(self, folder: Union[str, Path], create: bool=True) -> Path:
        path = self.config.log_dir / folder
        if create and not path.is_dir():
            path.mkdir(parents=True)
        return path

    @property
    def plots_dir(self) -> Path:
        return self._folder("plots")

    @property
    def samples_dir(self) -> Path:
        return self._folder("samples")

    @property
    def checkpoints_dir(self) -> Path:
        return self.config.log_dir / "checkpoints"

    @property
    def log_dir(self) -> Path:
        # Accessing this property doesn't create the folder.
        return self.config.log_dir

    @property
    def results_dir(self) -> Path:
        return self._folder("results", create=False)

    @property
    def config_path(self) -> Path:
        return self.log_dir / "config.pt"

    @property
    def started(self) -> bool:
        return is_nonempty_dir(self.checkpoints_dir)

    @property
    def global_step(self) -> int:
        """ Proxy for `self.state.global_step`. """
        return self.state.global_step
    
    @global_step.setter
    def global_step(self, value: int) -> None:
        self.state.global_step = value

    @property
    def done(self) -> bool:
        """Returns wether or not the experiment is complete.
        
        Returns:
            bool: Wether the experiment is complete or not (wether the
            results_dir exists and contains files).
        """
        import os
        scratch_dir = os.environ.get("SCRATCH")
        if scratch_dir:
            log_dir = self.config.log_dir.relative_to(self.config.log_dir_root)
            results_dir = Path(scratch_dir) / "SSCL" / log_dir / "results"
            if results_dir.exists() and is_nonempty_dir(results_dir):
                # Results already exists in $SCRATCH, therefore experiment is done.
                logger.info(f"Experiment is already done (non-empty folder at {results_dir}) Exiting.")
                return True
        return self.started and is_nonempty_dir(self.results_dir)
