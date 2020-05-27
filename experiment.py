import inspect
import json
import logging
import os
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
from simple_parsing import choice, field, mutable_field, subparsers
from simple_parsing.helpers import FlattenedAccess
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Sampler, TensorDataset

from common.losses import LossInfo, TrainValidLosses
from common.metrics import (ClassificationMetrics, RegressionMetrics,
                            get_metrics, Metrics)
from config import Config
from datasets import DatasetConfig
from datasets.cifar import Cifar10, Cifar100
from datasets.fashion_mnist import FashionMnist
from datasets.mnist import Mnist
from models.classifier import Classifier
from tasks import AuxiliaryTask, Tasks
from utils import utils
from utils.json_utils import JsonSerializable, take_out_unsuported_values
from utils.logging import pbar
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
        self.train_loader: DataLoader = NotImplemented
        self.valid_loader: DataLoader = NotImplemented

        self.global_step: int = 0
        self.logger = self.config.get_logger(inspect.getfile(type(self)))
        if self.config.debug:
            logger.setLevel(logging.DEBUG)
        
        self._samples_dir: Optional[Path] = None
        
        if self.notes:
            with open(self.log_dir / "notes.txt", "w") as f:
                f.write(self.notes)
        
        from save_job import SaverWorker
        self.background_queue = mp.Queue()
        self.saver_worker: Optional[SaverWorker] = None

        self.state = self.State()

    def __del__(self):
        print("Destroying the 'Experiment' object.")

    @abstractmethod
    def run(self):
        pass

    def cleanup(self):
        print("Cleaning up after the experiment is done.")
        self.background_queue.put(None)
        if self.saver_worker and self.saver_worker.is_alive():
            self.saver_worker.join()
        print("Successfully closed everything")

    def load_datasets(self) -> Tuple[Dataset, Dataset]:
        """ Setup the dataloaders and other settings before training. """
        self.train_dataset, self.valid_dataset = self.dataset.load(data_dir=self.config.data_dir)
        self.train_loader = self.get_dataloader(self.train_dataset)
        self.valid_loader = self.get_dataloader(self.valid_dataset)
        return self.train_dataset, self.valid_dataset

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

    def train_until_convergence(self, train_dataset: Dataset,
                                      valid_dataset: Dataset,
                                      max_epochs: int,
                                      description: str=None,
                                      patience: int=None,
                                      temp_save_file: Path=None) -> TrainValidLosses:
        # TODO: Add a way to resume training if it was previously interrupted.
        # For instance, it might be useful to keep track of the number of epochs
        # performed in the current task (for TaskIncremental)

        # TODO: save/load the `all_losses` object to temp_save_file at a given
        # interval during training, using a saver thread.

        train_dataloader = self.get_dataloader(train_dataset)
        valid_dataloader = self.get_dataloader(valid_dataset)
        n_steps = len(train_dataloader)
        
        if self.config.debug_steps:
            from itertools import islice
            n_steps = self.config.debug_steps
            train_dataloader = islice(train_dataloader, 0, n_steps)  # type: ignore

        all_losses = TrainValidLosses()
        # Get the latest step
        # NOTE: At the moment, will always be zero, but if we reload
        # `all_losses` from a file, would give you the step to start from.
        starting_step = all_losses.latest_step()

        valid_loss_gen = self.valid_performance_generator(valid_dataset)
        
        best_valid_acc: Optional[float] = None
        counter = 0
        
        # Early stopping: number of validation epochs with increasing loss after
        # which we exit training.
        patience = patience or self.config.patience

        message: Dict[str, Any] = OrderedDict()
        for epoch in range(max_epochs):
            pbar = tqdm.tqdm(train_dataloader, total=n_steps)
            desc = description or "" 
            desc += " " if desc and not desc.endswith(" ") else ""
            desc += f"Epoch {epoch}"
            pbar.set_description(desc + " Train")
            
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

                    train_log_dict = train_loss.to_log_dict()
                    valid_log_dict = valid_loss.to_log_dict()
                    self.log({"Train": train_log_dict, "Valid": valid_log_dict})
            
            # perform a validation epoch.
            val_desc = desc + " Valid"
            val_loss_info = self.test(valid_dataset, description=val_desc)
            val_acc = val_loss_info.metrics[Tasks.SUPERVISED].accuracy
            
            if best_valid_acc is None or val_acc.item() > best_valid_acc:
                counter = 0
                best_valid_acc = val_acc.item()
            else:
                counter += 1
                print(f"Validation Acc hasn't increased over the last {counter} epochs.")
                if counter == patience:
                    print(f"Exiting at step {self.global_step}, as validation acc hasn't increased over the last {patience} epochs.")
                    break
        return all_losses

    def valid_performance_generator(self, valid_dataset: Dataset) -> Generator[LossInfo, None, None]:
        periodic_valid_dataloader = self.get_dataloader(valid_dataset)
        while True:
            for batch in periodic_valid_dataloader:
                data = batch[0].to(self.model.device)
                target = batch[1].to(self.model.device) if len(batch) == 2 else None
                yield self.test_batch(data, target)

    def train_iter(self, dataloader: DataLoader) -> Iterable[LossInfo]:
        self.model.train()
        for batch in dataloader:
            data, target = self.preprocess(batch)
            yield self.train_batch(data, target)

    def preprocess(self, batch: Union[Tuple[Tensor], Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Optional[Tensor]]:
        data = batch[0].to(self.model.device)
        target = batch[1].to(self.model.device) if len(batch) == 2 else None  # type: ignore
        return data, target

    def train_batch(self, data: Tensor, target: Optional[Tensor]) -> LossInfo:
        self.model.optimizer.zero_grad()

        batch_loss_info = self.model.get_loss(data, target)
        total_loss = batch_loss_info.total_loss
        total_loss.backward()

        self.model.optimizer_step(global_step=self.global_step)

        self.global_step += data.shape[0]
        return batch_loss_info

    def test(self, dataset: Dataset, description: str=None, name: str="Test") -> LossInfo:
        dataloader = self.get_dataloader(dataset)
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
        return total_loss

    def test_iter(self, dataloader: DataLoader) -> Iterable[LossInfo]:
        self.model.eval()
        for batch in dataloader:
            data, target = self.preprocess(batch)
            yield self.test_batch(data, target)

    def test_batch(self, data: Tensor, target: Tensor=None) -> LossInfo:
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            loss = self.model.get_loss(data, target)
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
    
    def save(self, save_dir: Path=None, save_model_weights: bool=True) -> None:
        if self.saver_worker is None:
            from save_job import SaverWorker
            self.saver_worker = SaverWorker(self.config, self.background_queue)
            self.saver_worker.start()

        # If there are common attributes between the Experiment and the State
        # objects, then also copy them over into the State to be saved.
        # NOTE: (FN) This is a bit extra, I don't think its needed.
        for name, (v1, v2) in common_fields(self, self.state):
            logger.debug(f"Copying the '{name}' attribute into the 'State' object to be saved.")
            setattr(self.state, name, v1)
        
        self.state.global_step = self.global_step
        save_dir = save_dir or self.checkpoints_dir
        
        model_state_dict: Optional[Dict[str, Tensor]] = None
        if save_model_weights:
            model_state_dict = OrderedDict()
            tensor: Tensor
            for k, tensor in self.model.state_dict().items():
                model_state_dict[k] = tensor.detach().cpu()

        logger.debug(f"Saving state (in background) to save_dir {save_dir}")
        self.background_queue.put({
            "save_dir": save_dir,
            "state": self.state,
            "model_state_dict": model_state_dict,
        })
        


        

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
            elif isinstance(message, LossInfo):
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
                    if 'knn_losses' in k and 'Verbose' not in k:
                        task_measuree, task_measured = [int(s) for s in k if s.isdigit()]
                        mode = k.split('/')[-1]
                        if mode=='valid':
                            avv_knn.append(message_dict[f'knn_losses[{task_measuree}][{task_measured}]/{mode}']['metrics']['KNN']['accuracy'])
                        items.append((f'KNN_per_task/knn_{mode}_task_{task_measured}',message_dict[f'knn_losses[{task_measuree}][{task_measured}]/{mode}'][
                                                'metrics']['KNN']['accuracy']))

                    elif 'cumul_losses' in k:
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
                message_dict['task/currently_learned_task'] = self.state.i
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
                self.log(f"Experiment is already done (non-empty folder at {results_dir}) Exiting.")
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
        d = take_out_unsuported_values(d)
        return d

    def to_dict(self) -> Dict:
        return self.to_config_dict()
        

# Load up the addons, each of which adds independent, useful functionality to the Experiment base-class.
# TODO: This might not be the cleanest/most elegant way to do it, but it's better than having files with 1000 lines in my opinion.
from addons import (ExperimentWithEWC, ExperimentWithKNN, ExperimentWithVAE,
                    LabeledPlotRegionsAddon, TestTimeTrainingAddon)


@dataclass  # type: ignore
class Experiment(ExperimentWithEWC, ExperimentWithKNN, ExperimentWithVAE,
                 TestTimeTrainingAddon, LabeledPlotRegionsAddon, ):
    """ Describes the parameters of an experimental setting.
    
    (ex: Mnist_iid, Mnist_continual, Cifar10, etc. etc.)
    
    To create a new experiment, subclass this class, and add/change what you
    need to customize.

    TODO: Maybe add some code for saving/restoring experiments here?
    """
    pass
