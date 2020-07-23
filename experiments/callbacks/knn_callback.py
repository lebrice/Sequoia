import itertools
from dataclasses import InitVar, asdict, dataclass
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from pytorch_lightning import Callback, Trainer
from sklearn.metrics import log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, Dataset

from common.losses import LossInfo
from models.cl_classifier import ContinualClassifier
from models.classifier import Classifier
from setups.cl.class_incremental_setting import ClassIncrementalSetting
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

    def on_epoch_end(self, trainer: Trainer, pl_module: Classifier):
        self.trainer = trainer
        self.model = pl_module

        if isinstance(pl_module, ContinualClassifier):
            self.evaluate_knn_class_incremental(pl_module)
        else:
            raise NotImplementedError("TODO: Re-implement the KNN evaluation for IID training (should be pretty simple).")
    
    def log(self, loss_info: LossInfo):
        if self.trainer.logger:
            self.trainer.logger.log_metrics(loss_info.to_log_dict())


    def evaluate_knn_class_incremental(self, pl_module: ContinualClassifier,
                                             max_num_samples: int = 10_000):
        """ Evaluate the representations with a KNN in the context of CL.

        We shorten the train dataloaders to take only the first
        `max_num_samples` samples in order to save some compute.
        # TODO: Figure out a way to cleanly add the metrics from the callback to
        # the ``log dict'' which is returned by the model. Right now they are
        # only printed / logged to wandb from here. 
        
        """
        train_loaders: List[DataLoader] = pl_module.train_dataloaders()
        valid_loaders: List[DataLoader] = pl_module.val_dataloaders()
        test_loaders:  List[DataLoader] = pl_module.test_dataloaders()

        # Create an iterator that alternates between each of the train dataloaders.
        entire_train_dataset_iterator = roundrobin(*train_loaders)
        # Only take the first `max_num_samples` samples from that iterator.
        # We round this up so we always take at least batch_size samples.
        max_num_batches = int(max_num_samples / pl_module.hp.batch_size)
        train_loader = take(entire_train_dataset_iterator, n=max_num_batches)        

        

        h_x, y = get_hidden_codes_array(
            model=pl_module,
            dataloader=train_loader,
            description="KNN (Train)"
        )
        train_classes = np.unique(y)
        train_loss, scaler, knn_classifier = fit_knn(
            x=h_x,
            y=y,
            options=self.knn_options,
            loss_name="train"
        )
        logger.info(f"KNN Train Acc: {train_loss.accuracy:.2%}")
        self.log(train_loss)

        total_valid_loss = LossInfo("knn/valid")
        for i, dataloader in enumerate(valid_loaders):
            dataloader = take(dataloader, max_num_batches)
            loss_i = evaluate(
                model=pl_module,
                dataloader=dataloader,
                loss_name=f"[{i}]",
                scaler=scaler,
                knn_classifier=knn_classifier,
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

        total_test_loss = LossInfo("knn/test")
        for i, dataloader in enumerate(test_loaders):
            dataloader = take(dataloader, max_num_batches)
            loss_i = evaluate(
                model=pl_module,
                dataloader=dataloader,
                loss_name=f"[{i}]",
                scaler=scaler,
                knn_classifier=knn_classifier,
            )
            total_test_loss.absorb(loss_i)
            logger.info(f"KNN Test[{i}] Acc: {loss_i.accuracy:.2%}")
            self.log(loss_i)

        logger.info(f"KNN Average Test Acc: {total_test_loss.accuracy:.2%}")
        self.log(total_test_loss)
        




def evaluate(model: Classifier,
             dataloader: DataLoader,
             loss_name: str,
             scaler: StandardScaler,
             knn_classifier: KNeighborsClassifier) -> LossInfo:
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
        LossInfo: The loss object containing metrics and a 'total loss'
        which isn't a tensor in this case (since passing through the KNN
        isn't a differentiable operation).
    """
    h_x_test, y_test = get_hidden_codes_array(model, dataloader,  description=f"KNN ({loss_name})")
    train_classes = set(knn_classifier.classes_)
    test_classes = set(y_test)
    # Check that the same classes were used.
    assert test_classes.issubset(train_classes), (
        f"y and y_test should contain the same classes: "
        f"(train classes: {train_classes}, "
        f"test classes: {test_classes})."
    )
    test_loss = evaluate_knn(
        x_t=h_x_test, y_t=y_test,
        loss_name=loss_name,
        scaler=scaler,
        knn_classifier=knn_classifier,
    )
    # logger.info(f"{loss_name} Acc: {test_loss.accuracy:.2%}")
    return test_loss

def get_hidden_codes_array(model: Classifier, dataloader: DataLoader, description: str="KNN") -> Tuple[np.ndarray, np.ndarray]:
    """ Gets the hidden vectors and corresponding labels. """
    h_x_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    for batch in pbar(dataloader, description, leave=False):
        x, y = model.preprocess_batch(batch)
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


def fit_knn(x: np.ndarray, y: np.ndarray, options: KnnClassifierOptions=None, loss_name: str="knn") -> Tuple[LossInfo, StandardScaler, KNeighborsClassifier]:
    # print(x.shape, y.shape, x_t.shape, y_t.shape)
    options = options or KnnClassifierOptions()
   
    scaler = StandardScaler()
    x_s = scaler.fit_transform(x)
    # Create and train the Knn Classifier using the options as the kwargs
    knn_classifier = KNeighborsClassifier(**asdict(options)).fit(x_s, y)
    train_loss = evaluate_knn(x_t=x, y_t=y, scaler=scaler, knn_classifier=knn_classifier)
    return train_loss, scaler, knn_classifier


def evaluate_knn(x_t: np.ndarray, y_t: np.ndarray, scaler: StandardScaler, knn_classifier: KNeighborsClassifier, loss_name: str="KNN") -> LossInfo:
    # Flatten the inputs to two dimensions only.
    x_t = x_t.reshape(x_t.shape[0], -1)
    assert len(x_t.shape) == 2
    x_t = scaler.transform(x_t)
    y_t_prob = knn_classifier.predict_proba(x_t)
    
    y_t_logits = y_t_prob
    classes = knn_classifier.classes_
    
    assert np.array_equal(sorted(classes), classes)
    ## We were constructing this to reorder the classes in case the ordering was
    ## not the same between the KNN's internal `classes_` attribute and the task
    ## classes, However I'm not sure if this is necessary anymore.

    # y_t_logits = np.zeros((y_t.size, y_t.max() + 1))
    # for i, label in enumerate(classes):
    #     y_t_logits[:, label] = y_t_prob[:, i]

    # We get the Negative Cross Entropy using the scikit-learn function, but we
    # could instead get it using pytorch's function (maybe even inside the
    # LossInfo object!
    nce_t = log_loss(y_true=y_t, y_pred=y_t_prob, labels=classes)
    test_loss = LossInfo(loss_name, total_loss=nce_t, y_pred=y_t_logits, y=y_t)
    return test_loss
