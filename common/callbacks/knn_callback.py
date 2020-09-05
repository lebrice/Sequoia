""" Callback that evaluates representations with a KNN after each epoch.

TODO: The code here is split into too many functions and its a bit confusing.
    Will Need to rework that at some point.

"""

import itertools
import math
from dataclasses import InitVar, asdict, dataclass
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from pytorch_lightning import (Callback, LightningDataModule, LightningModule,
                               Trainer)
from sklearn.metrics import log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, Dataset

from common.loss import Loss
from methods.models.model import Model
from settings import Setting
from settings.passive.cl.setting import ClassIncrementalSetting
from simple_parsing import field, mutable_field
from utils.logging_utils import get_logger, pbar
from utils.utils import roundrobin, take

logger = get_logger(__file__)

@dataclass
class KnnClassifierOptions:
    """ Set of options for configuring the KnnClassifier. """
    n_neighbors: int = field(default=5, alias="n_neighbours") # Number of neighbours.
    metric: str = "cosine"
    algorithm: str = "auto" # See the sklearn docs
    leaf_size: int = 30  # See the sklearn docs
    p: int = 2  # see the sklean docs
    n_jobs: Optional[int] = -1  # see the sklearn docs.

@dataclass
class KnnCallback(Callback): 
    """ Addon that adds the option of evaluating representations with a KNN.
    
    TODO: Perform the KNN evaluations in different processes using multiprocessing.
    TODO: We could even evaluate the representations of a DIFFERENT dataset with
    the KNN, if the shapes were compatible with the model! For example, we could
    train the model on some CL/RL/etc task, like Omniglot or something, and at
    the same time, evaluate how good the model's representations are at
    disentangling the classes from MNIST or Fashion-MNIST or something else
    entirely! This could be nice when trying to argue about better generalization
    in the model's representations.
    """
    # Options for the KNN classifier
    knn_options: KnnClassifierOptions = mutable_field(KnnClassifierOptions)
    # Maximum number of examples to take from the dataloaders. When None, uses
    # the entire training/validaton/test datasets.
    knn_samples: int = 0

    def __post_init__(self):
        self.max_num_batches: int = 0

        self.model: Model
        self.trainer: Trainer

    def on_train_start(self, trainer, pl_module):
        """Called when the train begins."""
        self.trainer = trainer
        self.model = pl_module
        self.setting: ClassIncrementalSetting

    def setup(self, trainer, pl_module, stage: str):
        """Called when fit or test begins"""
        super().setup(trainer, pl_module, stage)

    def on_epoch_end(self, trainer: Trainer, pl_module: Model):
        self.trainer = trainer
        self.model = pl_module
        self.setting = self.model.setting
        config = self.model.config

        if self.knn_samples > 0:
            batch_size = pl_module.batch_size
            # We round this up so we always take at least one batch_size of
            # samples from each dataloader.
            self.max_num_batches = math.ceil(self.knn_samples / batch_size)
            logger.debug(f"Taking a maximum of {self.max_num_batches} batches from each dataloader.")
            
            if config.debug:
                self.knn_samples = min(self.knn_samples, 100)


            valid_knn_loss, test_knn_loss = self.evaluate_knn(pl_module)

            # assert False, trainer.callback_metrics.keys()
            loss: Optional[Loss] = trainer.callback_metrics.get("loss_object")
            if loss:
                assert "knn/valid" not in loss.losses
                assert "knn/test" not in loss.losses
                loss.losses["knn/valid"] = valid_knn_loss
                loss.losses["knn/test"] = test_knn_loss
    
    def log(self, loss_object: Loss):
        if self.trainer.logger:
            self.trainer.logger.log_metrics(loss_object.to_log_dict())

    def get_dataloaders(self, model: LightningModule, mode: str) -> List[DataLoader]:
        """ Retrieve the train/val/test dataloaders for all 'tasks'. """
        setting = model.datamodule
        assert setting, "The LightningModule must have its 'datamodule' attribute set for now."
        # if the setting defines a dataloaders() method, those are for each of the tasks, which is what we want!
        fn = getattr(setting, f"{mode}_dataloaders",
               getattr(setting, f"{mode}_dataloader")
            )
        loaders = fn()
        if isinstance(loaders, DataLoader):
            return [loaders]
        assert isinstance(loaders, list)
        return loaders

    def evaluate_knn(self, model: LightningModule) -> Tuple[Loss, Loss]:
        """ Evaluate the representations with a KNN in the context of CL.

        We shorten the train dataloaders to take only the first
        `knn_samples` samples in order to save some compute.
        TODO: Figure out a way to cleanly add the metrics from the callback to
        the ``log dict'' which is returned by the model. Right now they are
        only printed / logged to wandb directly from here. 
        """
        setting = model.datamodule
        assert isinstance(setting, Setting)
        # TODO: Remove this if we want to use this for something else than a
        # Continual setting in the future.
        assert isinstance(setting, ClassIncrementalSetting)
        num_classes = setting.num_classes

        # Check wether the method has access to the task labels at train/test time.
        task_labels_at_test_time: bool = False        
        from settings import TaskIncrementalSetting
        if isinstance(setting, TaskIncrementalSetting):
            if setting.task_labels_at_test_time:
                task_labels_at_test_time = True
        # TODO: Figure out a way to make sure that we get at least one example
        # of each class to fit the KNN.
        self.knn_samples = max(self.knn_samples, num_classes ** 2)
        self.max_num_batches = math.ceil(self.knn_samples / model.batch_size)
        logger.info(f"number of classes: {num_classes}")
        logger.info(f"Number of KNN samples: {self.knn_samples}")
        logger.debug(f"Taking a maximum of {self.max_num_batches} batches from each dataloader.")


        train_loaders: List[DataLoader] = self.get_dataloaders(model, mode="train")
        valid_loaders: List[DataLoader] = self.get_dataloaders(model, mode="val")
        test_loaders:  List[DataLoader] = self.get_dataloaders(model, mode="test")

        # Only take the first `knn_samples` samples from each dataloader.
        def shorten(dataloader: DataLoader):
            return take(dataloader, n=self.max_num_batches)
        
        if self.max_num_batches:
            train_loaders = list(map(shorten, train_loaders))
            valid_loaders = list(map(shorten, valid_loaders))
            test_loaders = list(map(shorten, test_loaders))

        # Create an iterator that alternates between each of the train dataloaders.
        # NOTE: we shortened each of the dataloaders just to be sure that we get at least
        train_loader = roundrobin(*train_loaders)

        h_x, y = get_hidden_codes_array(
            model=model,
            dataloader=train_loader,
            description="KNN (Train)"
        )
        train_loss, scaler, knn_classifier = fit_knn(
            x=h_x,
            y=y,
            options=self.knn_options,
            num_classes=num_classes,
            loss_name="knn/train"
        )
        logger.info(f"KNN Train Acc: {train_loss.accuracy:.2%}")
        self.log(train_loss)
        total_valid_loss = Loss("knn/valid")
        
        # Save the current task ID so we can reset it after testing.
        starting_task_id = model.setting.current_task_id

        for i, dataloader in enumerate(valid_loaders):
            if task_labels_at_test_time:
                model.on_task_switch(i, training=False)
            loss_i = evaluate(
                model=model,
                dataloader=dataloader,
                loss_name=f"[{i}]",
                scaler=scaler,
                knn_classifier=knn_classifier,
                num_classes=setting.num_classes_in_task(i)
            )
            # We use `.absorb(loss_i)` here so that the metrics get merged.
            # That way, if we access `total_valid_loss.accuracy`, this gives the
            # accuracy over all the validation tasks.
            # If we instead used `+= loss_i`, then loss_i would become a subloss
            # of `total_valid_loss`, since they have different names.
            # TODO: Explain this in more detail somewhere else.
            total_valid_loss.absorb(loss_i)
            logger.info(f"KNN Valid[{i}] Acc: {loss_i.accuracy:.2%}")
            self.log(loss_i)


        logger.info(f"KNN Average Valid Acc: {total_valid_loss.accuracy:.2%}")
        self.log(total_valid_loss)

        total_test_loss = Loss("knn/test")
        for i, dataloader in enumerate(test_loaders):
            if task_labels_at_test_time:
                model.on_task_switch(i, training=False)
            
            # TODO Should we set the number of classes to be the number of
            # classes in the current task?

            loss_i = evaluate(
                model=model,
                dataloader=dataloader,
                loss_name=f"[{i}]",
                scaler=scaler,
                knn_classifier=knn_classifier,
                num_classes=num_classes,
            )
            total_test_loss.absorb(loss_i)
            logger.info(f"KNN Test[{i}] Acc: {loss_i.accuracy:.2%}")
            self.log(loss_i)

        if task_labels_at_test_time:
            model.on_task_switch(starting_task_id, training=False)

        logger.info(f"KNN Average Test Acc: {total_test_loss.accuracy:.2%}")
        self.log(total_test_loss)
        return total_valid_loss, total_test_loss 


def evaluate(model: Model,
             dataloader: DataLoader,
             loss_name: str,
             scaler: StandardScaler,
             knn_classifier: KNeighborsClassifier,
             num_classes: int) -> Loss:
    """Evaluates the 'quality of representations' using a KNN.

    Assumes that the knn classifier was fitted on the same classes as
    the ones present in the dataloader.

    Args:
        model (Classifier): a Classifier model to use to encode samples.
        dataloader (DataLoader): a dataloader.
        loss_name (str): name to give to the resulting loss.
        scaler (StandardScaler): the scaler used during fitting.
        knn_classifier (KNeighborsClassifier): The KNN classifier.

    Returns:
        Loss: The loss object containing metrics and a 'total loss'
        which isn't a tensor in this case (since passing through the KNN
        isn't a differentiable operation).
    """
    h_x_test, y_test = get_hidden_codes_array(
        model,
        dataloader,
        description=f"KNN ({loss_name})",
    )
    train_classes = set(knn_classifier.classes_)
    test_classes = set(y_test)
    # Check that the same classes were used.
    assert test_classes.issubset(train_classes), (
        f"y and y_test should contain the same classes: "
        f"(train classes: {train_classes}, "
        f"test classes: {test_classes})."
    )
    test_loss = get_knn_performance(
        x_t=h_x_test, y_t=y_test,
        loss_name=loss_name,
        scaler=scaler,
        knn_classifier=knn_classifier,
        num_classes=num_classes,
    )
    test_loss.loss = torch.as_tensor(test_loss.loss)
    logger.info(f"{loss_name} Acc: {test_loss.accuracy:.2%}")
    return test_loss


def get_hidden_codes_array(model: Model, dataloader: DataLoader, description: str="KNN") -> Tuple[np.ndarray, np.ndarray]:
    """ Gets the hidden vectors and corresponding labels. """
    h_x_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []

    for batch in pbar(dataloader, description, leave=False):
        x, y = model.preprocess_batch(batch)
        assert isinstance(x, Tensor), type(x)

        # We only do KNN with examples that have a label.
        assert y is not None, f"Should have a 'y' for now! {x}, {y}"
        if y is not None:
            # TODO: There will probably be some issues with trying to use
            # the model's encoder to encode stuff when using DataParallel or
            # DistributedDataParallel, as PL might be interfering somehow.
            h_x = model.encode(x.to(model.device))
            h_x_list.append(h_x.detach().cpu().numpy())
            y_list.append(y.detach().cpu().numpy())
    codes = np.concatenate(h_x_list)
    labels = np.concatenate(y_list)
    return codes.reshape(codes.shape[0], -1), labels


def fit_knn(x: np.ndarray,
            y: np.ndarray,
            num_classes: int,
            options: KnnClassifierOptions=None,
            loss_name: str="knn") -> Tuple[Loss, StandardScaler, KNeighborsClassifier]:
    # print(x.shape, y.shape, x_t.shape, y_t.shape)
    options = options or KnnClassifierOptions()
   
    scaler = StandardScaler()
    x_s = scaler.fit_transform(x)
    # Create and train the Knn Classifier using the options as the kwargs
    knn_classifier = KNeighborsClassifier(**asdict(options)).fit(x_s, y)
    train_loss = get_knn_performance(
        x_t=x,
        y_t=y,
        scaler=scaler,
        knn_classifier=knn_classifier,
        num_classes=num_classes,
    )
    return train_loss, scaler, knn_classifier


def get_knn_performance(x_t: np.ndarray,
                        y_t: np.ndarray,
                        scaler: StandardScaler,
                        knn_classifier: KNeighborsClassifier,
                        num_classes: int,
                        loss_name: str="KNN",) -> Loss:
    # Flatten the inputs to two dimensions only.
    x_t = x_t.reshape(x_t.shape[0], -1)
    assert len(x_t.shape) == 2
    x_t = scaler.transform(x_t)
    y_t_prob = knn_classifier.predict_proba(x_t)

    classes = knn_classifier.classes_
    # make sure the classes are sorted:
    assert np.array_equal(sorted(classes), classes)

    if y_t_prob.shape[-1] == num_classes:
        y_t_logits = y_t_prob
    else:
        # Not all classes were encountered, so we need to 'expand' the predicted
        # logits to the right shape.
        logger.info(f"{y_t_prob.shape} {num_classes}")
        num_classes = max(num_classes, y_t_prob.shape[-1])

        y_t_logits = np.zeros([y_t_prob.shape[0], num_classes], dtype=y_t_prob.dtype)
        
        for i, logits in enumerate(y_t_prob):
            for label, logit in zip(classes, logits):
                y_t_logits[i][label-1] = logit
    
    ## We were constructing this to reorder the classes in case the ordering was
    ## not the same between the KNN's internal `classes_` attribute and the task
    ## classes, However I'm not sure if this is necessary anymore.

    # y_t_logits = np.zeros((y_t.size, y_t.max() + 1))
    # for i, label in enumerate(classes):
    #     y_t_logits[:, label] = y_t_prob[:, i]
    
    # We get the Negative Cross Entropy using the scikit-learn function, but we
    # could instead get it using pytorch's function (maybe even inside the
    # Loss object!
    nce_t = log_loss(y_true=y_t, y_pred=y_t_prob, labels=classes)
    # BUG: There is sometimes a case where some classes aren't present in
    # `classes_`, and as such the ClassificationMetrics object created in the
    # Loss constructor has an error. 
    test_loss = Loss(loss_name, loss=nce_t, y_pred=y_t_logits, y=y_t)
    return test_loss
