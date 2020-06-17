import inspect
import json
import logging
import os
import time
import hashlib
from abc import ABC, abstractmethod
from collections import MutableMapping, OrderedDict, defaultdict
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import (Any, ClassVar, Dict, Generator, Iterable, List, Optional,
                    Tuple, Type, Union)

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
import tqdm
import wandb
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Sampler, TensorDataset
from torchvision.datasets import VisionDataset

from common.losses import LossInfo, TrainValidLosses
from common.metrics import (ClassificationMetrics, Metrics, RegressionMetrics,
                            get_metrics)
from config import Config
from datasets import DatasetConfig
from datasets.cifar import Cifar10, Cifar100
from datasets.fashion_mnist import FashionMnist
from datasets.mnist import Mnist
from datasets.subset import ClassSubset, Subset
from models.classifier import Classifier
from save_job import SaveTuple
from simple_parsing import choice, field, mutable_field, subparsers
from simple_parsing.helpers import FlattenedAccess
from tasks import AuxiliaryTask, Tasks
from utils import utils
from utils.early_stopping import EarlyStoppingOptions, early_stopping
from utils.json_utils import JsonSerializable, take_out_unsuported_values
from utils.logging_utils import pbar
from utils.utils import add_prefix, common_fields, is_nonempty_dir

logger = Config.get_logger(__file__)

@dataclass  # type: ignore
class ExperimentBase(JsonSerializable):
    """Base-class for an Experiment.
    """
    # Model Hyper-parameters
    hparams: Classifier.HParams = mutable_field(Classifier.HParams)
    
    dataset: DatasetConfig = choice({
        "mnist": Mnist(),
        "fashion_mnist": FashionMnist(),
        "cifar10": Cifar10(),
        "cifar100": Cifar100(),
    }, default="mnist")

    config: Config = mutable_field(Config)
    # Notes about this particular experiment. (will be logged to wandb if used.)
    notes: Optional[str] = None
    
    model: Classifier = field(default=None, init=False)

    no_wandb_cleanup: bool = False
        
    @dataclass
    class State(JsonSerializable):
        """ Dataclass used to store the state of the experiment.
        
        This object should contain everything we want to be able to save/restore.
        NOTE: We aren't going to parse these from the command-line.
        """
        global_step: int = 0
        model_weights_path: Optional[Path] = None
        # Container for train/valid losses that are logged periodically.
        all_losses: TrainValidLosses = mutable_field(TrainValidLosses, repr=False)

    def __post_init__(self):
        """ Called after __init__, used to initialize all missing fields.
        
        You can use this method to initialize the fields that aren't parsed from
        the command-line, such as `model`, etc.
        Additionally, the fields created here are not added in the wandb logs.       
        """
        AuxiliaryTask.input_shape   = self.dataset.x_shape

        # Set these shared attributes so that all the Auxiliary tasks can be created.
        if isinstance(self.dataset, (Mnist, FashionMnist)):
            AuxiliaryTask.input_shape = self.dataset.x_shape
            AuxiliaryTask.hidden_size = self.hparams.hidden_size

        self.train_dataset: Dataset = NotImplemented
        self.valid_dataset: Dataset = NotImplemented
        self.test_dataset: Dataset = NotImplemented
        self.train_loader: DataLoader = NotImplemented
        self.valid_loader: DataLoader = NotImplemented
        self.test_loader: DataLoader = NotImplemented

        self.global_step: int = 0
        self.logger = self.config.get_logger(inspect.getfile(type(self)))
        if self.config.debug:
            logger.setLevel(logging.DEBUG)
        
        self._samples_dir: Optional[Path] = None
        
        if self.notes:
            with open(self.log_dir / "notes.txt", "w") as f:
                f.write(self.notes)
        
        from save_job import SaverWorker
        mp.set_start_method("spawn")
        self.background_queue = mp.Queue()
        self.saver_worker: Optional[SaverWorker] = None

        self.state = self.State()
        self.md5 = hashlib.md5(str(self.hparams).encode('utf-8') + str(self).encode('utf-8')).hexdigest()

    def __del__(self):
        print("Destroying the 'Experiment' object.")
        self.cleanup()

    @abstractmethod
    def run(self):
        pass

    def cleanup(self):
        print("Cleaning up after the experiment is done.")
        if self.saver_worker:
            self.background_queue.put(None)
            self.saver_worker.join(timeout=120)
        print("Successfully closed everything")

    def load_datasets(self) -> Tuple[Dataset, Dataset]:
        """ Loads the training and test datasets. """
        train_dataset, test_dataset = self.dataset.load(data_dir=self.config.data_dir)
        return train_dataset, test_dataset


    def train_valid_split(self, train_dataset: VisionDataset, valid_fraction: float=0.2) -> Tuple[VisionDataset, VisionDataset]:
        n = len(train_dataset)
        valid_len: int = int((n * valid_fraction))
        train_len: int = n - valid_len
        
        indices = np.arange(n, dtype=int)
        np.random.shuffle(indices)
        
        valid_indices = indices[:valid_len]
        train_indices = indices[valid_len:]

        train = Subset(train_dataset, train_indices)
        valid = Subset(train_dataset, valid_indices)
        logger.info(f"Training samples: {len(train)}, Valid samples: {len(valid)}")
        return train, valid


    def init_model(self) -> Classifier:
        print("init model")
        model = self.get_model_for_dataset(self.dataset)
        model.to(self.config.device)
        return model

    def get_model_for_dataset(self, dataset: DatasetConfig) -> Classifier:
        from models.mnist import MnistClassifier
        from models.cifar import Cifar10Classifier, Cifar100Classifier

        if isinstance(dataset, (Mnist, FashionMnist)):
            return MnistClassifier(hparams=self.hparams, config=self.config)
        elif isinstance(dataset, Cifar10):
            return Cifar10Classifier(hparams=self.hparams, config=self.config)
        elif isinstance(dataset, Cifar100):
            return Cifar100Classifier(hparams=self.hparams, config=self.config)
        else:
            raise NotImplementedError(f"TODO: add a model for dataset {dataset}.")

    def train(self,
              train_dataloader: Union[Dataset, DataLoader],                
              valid_dataloader: Union[Dataset, DataLoader],                
              epochs: int,                
              description: str=None,
              early_stopping_options: EarlyStoppingOptions=None,
              use_accuracy_as_metric: bool=None,                
              temp_save_dir: Path=None) -> TrainValidLosses:
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
        steps_per_epoch = len(train_dataloader)
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
        starting_step = all_losses.latest_step() or self.global_step
        starting_epoch = len(validation_losses) + 1

        if early_stopping_options:
            logger.info(f"Using early stopping with options {early_stopping_options}")
        
        # Hook to keep track of the best model.
        best_model_watcher = self.keep_best_model(
            use_acc=use_accuracy_as_metric,
            save_path=self.checkpoints_dir / "best_model.pth",
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
        valid_loss_gen = self.valid_performance_generator(valid_dataloader)
        
        # Message for the progressbar
        message: Dict[str, Any] = OrderedDict()
        # List to hold the length of each epoch (should all be the same length)
        epoch_lengths: List[int] = []

        for epoch in range(starting_epoch, epochs + 1):
            pbar = tqdm.tqdm(train_dataloader, total=steps_per_epoch)
            desc = description or "" 
            desc += " " if desc and not desc.endswith(" ") else ""
            desc += f"Epoch {epoch}"
            pbar.set_description(desc + " Train")

            epoch_start_step = self.global_step
            for batch_idx, train_loss in enumerate(self.train_iter(pbar)):
                train_loss.drop_tensors()
                
                if batch_idx % self.config.log_interval == 0:
                    # get loss on a batch of validation data:
                    valid_loss = next(valid_loss_gen)
                    valid_loss.drop_tensors()

                    all_losses[self.global_step] = (train_loss, valid_loss)

                    message.update(train_loss.to_pbar_message())
                    message.update(valid_loss.to_pbar_message())
                    pbar.set_postfix(message)

                    self.log({
                        "Train": train_loss,
                        "Valid": valid_loss,
                    })

            epoch_length = self.global_step - epoch_start_step
            epoch_lengths.append(epoch_length)

            # perform a validation epoch.
            val_desc = desc + " Valid"
            val_loss_info = self.test(valid_dataloader, description=val_desc, name="Valid")
            validation_losses.append(val_loss_info)

            if temp_save_dir:
                # Save these files in the background using the saver process.
                self.save(temp_save_dir / f"val_loss_{i}.json", val_loss_info)
                self.save(temp_save_dir / f"all_losses.json", all_losses)
            
            # Inform the best model watcher of the latest performance of the model.
            best_step = best_model_watcher.send(val_loss_info)
            logger.debug(f"Best step so far: {best_step}")

            best_epoch = best_step // int(np.mean(epoch_lengths))
            logger.debug(f"Best epoch so far: {best_epoch}")

            converged = convergence_checker.send(val_loss_info)
            if converged:
                logger.info(f"Training Converged at epoch {epoch}. Best valid performance was at epoch {best_epoch}")
                break

        try:
            # Re-load the best weights
            best_model_watcher.send(None)
        except StopIteration:
            pass
        
        convergence_checker.close()
        best_model_watcher.close()
        valid_loss_gen.close()

        logger.info(f"Best step: {best_step}, best_epoch: {best_epoch}, ")
        all_losses.keep_up_to_step(best_step)

        # TODO: Should we also return the array of validation losses at each epoch (`validation_losses`)?
        return all_losses

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
        
        def load_weights():
            state_dict = torch.load(save_path, map_location=self.config.device)
            self.model.load_state_dict(state_dict)
        
        best_perf: Optional[float] = None
        
        step = self.global_step
        best_step: int = step

        loss_info: Optional[LossInfo] = (yield step)

        while loss_info is not None:
            step = self.global_step

            val_loss = loss_info.total_loss.item()
            from task_incremental import get_supervised_metrics
            supervised_metrics = get_supervised_metrics(loss_info)
            
            if use_acc:
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
        load_weights()


    def valid_performance_generator(self, valid_dataloader: Union[Dataset, DataLoader]) -> Generator[LossInfo, None, None]:
        if isinstance(valid_dataloader, Dataset):
            valid_dataloader = self.get_dataloader(valid_dataloader)
        while True:
            for batch in valid_dataloader:
                data = batch[0].to(self.model.device)
                target = batch[1].to(self.model.device) if len(batch) == 2 else None
                yield self.test_batch(data, target, name="Valid")
        logger.info("Somehow exited the infinite while loop!")

    def train_iter(self, dataloader: DataLoader) -> Iterable[LossInfo]:
        self.model.train()
        for batch in dataloader:
            data, target = self.preprocess(batch)
            yield self.train_batch(data, target)

    def preprocess(self, batch: Union[Tuple[Tensor], Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Optional[Tensor]]:
        data = batch[0].to(self.model.device)
        target = batch[1].to(self.model.device) if len(batch) == 2 else None  # type: ignore
        return data, target

    def train_batch(self, data: Tensor, target: Optional[Tensor], name: str="Train") -> LossInfo:
        self.model.train()
        self.model.optimizer.zero_grad()

        batch_loss_info = self.model.get_loss(data, target, name=name)
        total_loss = batch_loss_info.total_loss
        total_loss.backward()

        self.model.optimizer_step(global_step=self.global_step)

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
        self.model.eval()
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

    def get_dataloader(self, dataset: Dataset, sampler: Sampler=None) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.config.use_cuda,
        )
    
    def save_state(self, save_dir: Path=None, save_model_weights: bool=True) -> None:
        save_dir = save_dir or self.checkpoints_dir
        
        # Store the most up to date global step in the state object.
        self.state.global_step = self.global_step or self.state.global_step
        
        model_state_dict: Dict[str, Tensor] = OrderedDict()
        if save_model_weights:
            for k, tensor in self.model.state_dict().items():
                model_state_dict[k] = tensor.detach().cpu()

        logger.debug(f"Saving state (in background) to save_dir {save_dir}")
        self.save(save_dir / "state.json", self.state)
        if model_state_dict:
            self.save(save_dir / "model_weights.pth", model_state_dict)

    def save(self, path: Path, obj: Any, blocking: bool=True) -> None:
        """Save the object `obj` to path `path`.

        If `blocking` is False, uses a background process. Otherwise, blocks
        until saving is complete. 

        Args:
            path (Path): Path to save to.
            obj (Any): object to save. (if JsonSerializable, will be saved to json)
            blocking (bool, optional): Wether to wait for the operation to
                finish, or to delegate to a background process. Defaults to False.
        """
        from save_job import SaverWorker, save
        assert isinstance(path, Path), f"positional argument 'path' should be a Path! (got {path})"
        if blocking:
           save(obj, save_path=path)
        else:
            if self.saver_worker is None:
                self.saver_worker = SaverWorker(self.config, self.background_queue)
            if not self.saver_worker.is_alive():
                self.saver_worker.start()
            self.background_queue.put(SaveTuple(save_path=path, obj=obj))

    def log(self, message: Union[str, Dict, LossInfo], value: Any=None, step: int=None, once: bool=False, prefix: str="", always_print: bool=False):
        if always_print or (self.config.debug and self.config.verbose):
            print(message, value if value is not None else "")

        # with open(self.log_dir / "log.txt", "a") as f:
        #     print(message, value, file=f)

        if self.config.use_wandb:
            # if we want to long once (like a final result, step should be None)
            # else, if not given, we use the global step.
            step = None if once else (step or self.global_step)
            if message is None:
                return
            message_dict: Dict
            if isinstance(message, dict):
                message_dict = OrderedDict()
                for k, v in message.items():
                    if isinstance(v, (LossInfo, Metrics, TrainValidLosses)):
                        v = v.to_log_dict()
                    message_dict[k] = v
            elif isinstance(message, (LossInfo, Metrics)):
                message_dict = message.to_log_dict()
            elif isinstance(message, str) and value is not None:
                message_dict = {message: value}
            elif isinstance(message, str):
                return
            else:
                message_dict = message  # type: ignore
            
            if prefix:
                message_dict = utils.add_prefix(message_dict, prefix)

            avv_knn = []
            def wandb_cleanup(d, parent_key='', sep='/', exclude_type=list):
                items = []
                for k, v in d.items():
                    new_key = parent_key + sep + k if parent_key else k
                    #if 'knn_losses' in k and 'Verbose' not in k:
                    #    task_measuree, task_measured = [int(s) for s in k if s.isdigit()]
                    #    mode = k.split('/')[-1]
                    #    if mode=='valid':
                    #        avv_knn.append(message_dict[f'knn_losses[{task_measuree}][{task_measured}]/{mode}']['metrics']['KNN']['accuracy'])
                    #    items.append((f'KNN_per_task/knn_{mode}_task_{task_measured}',message_dict[f'knn_losses[{task_measuree}][{task_measured}]/{mode}'][
                    #                            'metrics']['KNN']['accuracy']))

                    if 'cumul_losses' in k:
                        new_key = 'Cumulative'

                    elif 'task_losses' in k: 
                        task_measuree, task_measured = [int(s) for s in k if s.isdigit()]
                        new_key = 'Task_losses'+sep + f'Task{task_measured}'
                    elif '[' in new_key and 'Verbose' not in new_key:
                        new_key = 'Verbose/'+new_key

                    if isinstance(v, MutableMapping):
                        items.extend(wandb_cleanup(v, new_key, sep=sep).items())
                    else:
                        if not type(v)==exclude_type:
                            items.append((new_key, v))
                return dict(items)

            if not self.no_wandb_cleanup:
                message_dict = wandb_cleanup(message_dict)
                if len(avv_knn) > 0:
                    message_dict['KNN_per_task/avv_knn'] = np.mean(avv_knn)
                #message_dict['task/currently_learned_task'] = self.state.i
                message_dict = wandb_cleanup(message_dict)

            wandb.log(message_dict, step=step)

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
        if self._samples_dir:
            return self._samples_dir
        self._samples_dir = self._folder("samples")
        return self._samples_dir
    
    @samples_dir.setter
    def samples_dir(self, value: Path) -> None:
        self._samples_dir = value

    @property
    def checkpoints_dir(self) -> Path:
        return self.config.log_dir / "checkpoints" / self.md5

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
    def done(self) -> bool:
        """Returns wether or not the experiment is complete.
        
        Returns:
            bool: Wether the experiment is complete or not (wether the
            results_dir exists and contains files).
        """
        scratch_dir = os.environ.get("SCRATCH")
        if scratch_dir:
            log_dir = self.config.log_dir.relative_to(self.config.log_dir_root)
            results_dir = Path(scratch_dir) / "SSCL" / log_dir / "results"
            if results_dir.exists() and is_nonempty_dir(results_dir):
                # Results already exists in $SCRATCH, therefore experiment is done.
                logger.info(f"Experiment is already done (non-empty folder at {results_dir}) Exiting.")
                return True
        return self.started and is_nonempty_dir(self.results_dir)
    
    def save_to_results_dir(self, results: Dict[Union[str, Path], Any]):
        self.results_dir.mkdir(parents=True, exist_ok=True)
        for path, result in results.items():
            path = Path(path) if isinstance(path, str) else path
            if path.suffix in {".csv", ""}:
                array = result.detach().numpy() if isinstance(result, Tensor) else result
                np.savetxt(self.results_dir / path.with_suffix(".csv"), array, delimiter=",")
            elif path.suffix == ".json":
                with open(self.results_dir / path, "w") as f:
                    json.dump(result, f, indent="\t")

    def to_config_dict(self) -> Dict:
        d = asdict(self)
        d['md5'] = self.md5
        d = take_out_unsuported_values(d)
        return d

    def to_dict(self) -> Dict:
        return self.to_config_dict()
        

# Load up the addons, each of which adds independent, useful functionality to the Experiment base-class.
# TODO: This might not be the cleanest/most elegant way to do it, but it's better than having files with 1000 lines in my opinion.
from addons import (ExperimentWithKNN, ExperimentWithVAE,
                    LabeledPlotRegionsAddon, TestTimeTrainingAddon)


@dataclass  # type: ignore
class Experiment(ExperimentWithKNN, ExperimentWithVAE,
                 TestTimeTrainingAddon, LabeledPlotRegionsAddon):
    """ Describes the parameters of an experimental setting.
    
    (ex: Mnist_iid, Mnist_continual, Cifar10, etc. etc.)
    
    To create a new experiment, subclass this class, and add/change what you
    need to customize.

    TODO: Maybe add some code for saving/restoring experiments here?
    """
    pass
