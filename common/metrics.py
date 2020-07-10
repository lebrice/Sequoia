from dataclasses import dataclass, InitVar, asdict
from typing import Dict, Optional, Union, List
import numpy as np
import torch
import torch.nn as nn
#from models import Classifier
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
import torch.nn.functional as functional
from utils.json_utils import encode
from sklearn.metrics import roc_auc_score
from utils.json_utils import Serializable
from utils.logging_utils import get_logger
from simple_parsing import mutable_field, field

logger = get_logger(__file__)


@dataclass 
class Metrics(Serializable):
    n_samples: int = 0

    x:      InitVar[Optional[Tensor]] = None
    h_x:    InitVar[Optional[Tensor]] = None
    y_pred: InitVar[Optional[Tensor]] = None
    y:      InitVar[Optional[Tensor]] = None
    
    @torch.no_grad()
    def __post_init__(self,
                      x: Tensor=None,
                      h_x: Tensor=None,
                      y_pred: Tensor=None,
                      y: Tensor=None):
        """Creates metrics given `y_pred` and `y`.
        
        NOTE: Doesn't use `x` and `h_x` for now.
 
        Args:
            x (Tensor, optional): The input Tensor. Defaults to None.
            h_x (Tensor, optional): The hidden representation for x. Defaults to None.
            y_pred (Tensor, optional): The predicted label. Defaults to None.
            y (Tensor, optional): The true label. Defaults to None.
        """
        # get the batch size:
        for tensor in [x, h_x, y_pred, y]:
            if tensor is not None:
                self.n_samples = tensor.shape[0]
                break
    
    def __add__(self, other):
        # Metrics instances shouldn't be added together, as the base classes
        # should implement the method. We just return the other.
        return other

    def to_log_dict(self, verbose: bool=False) -> Dict:
        return OrderedDict({"n_samples": self.n_samples})

    def to_dict(self) -> Dict:
        return self.to_log_dict()

    def to_pbar_message(self) -> Dict[str, Union[str, float]]:
        return OrderedDict()


@dataclass
class AUCMetric(Metrics, Serializable):
    acum_acc: float = 0.  # type: ignore
    updates_counter: int = 1

    def to_log_dict(self, verbose: bool=True) -> Dict:
        d = super().to_log_dict()
        d["auc"] = float(self.auc)
        return d

    def __add__(self, other):
        if isinstance(other, AUCMetric):
            acum_acc = self.acum_acc + other.acum_acc
            updates_counter = self.updates_counter + other.updates_counter
        elif isinstance(other,ClassificationMetrics):
            acum_acc = self.acum_acc + other.accuracy
            updates_counter = self.updates_counter + 1
        else:
            return other
        return AUCMetric(acum_acc=acum_acc, updates_counter=updates_counter)

    @property
    def auc(self):
        return self.acum_acc / self.updates_counter


@dataclass
class RegressionMetrics(Metrics, Serializable):
    mse: Tensor = 0.  # type: ignore

    def __post_init__(self,
                      x: Tensor=None,
                      h_x: Tensor=None,
                      y_pred: Tensor=None,
                      y: Tensor=None):
        super().__post_init__(x=x, h_x=h_x, y_pred=y_pred, y=y)
        if y_pred is not None and y is not None:
            if y.shape != y_pred.shape:
                print("Shapes aren't the same!", y_pred.shape, y.shape)
            else:
                self.mse = functional.mse_loss(y_pred, y)

    def __add__(self, other: "RegressionMetrics") -> "RegressionMetrics":
        mse = torch.zeros_like(
            self.mse if self.mse is not None else
            other.mse if other.mse is not None else
            torch.zeros(1)
        )
        if self.mse is not None:
            mse = mse + self.mse
        if other.mse is not None:
            mse = mse + other.mse
        return RegressionMetrics(
            n_samples=self.n_samples + other.n_samples,
            mse=mse,
        )
    
    def to_log_dict(self, verbose: bool=True) -> Dict:
        d = super().to_log_dict()
        d["mse"] = float(self.mse)
        return d

    def to_pbar_message(self) -> Dict[str, Union[str, float]]:
        message = super().to_pbar_message()
        message["mse"] = float(self.mse.item())
        return message


@dataclass
class ClassificationMetrics(Metrics):
    confusion_matrix: Optional[Tensor] = field(default=None, repr=False)
    # fields we generate from the confusion matrix (if provided)
    accuracy: float = 0.
    class_accuracy: Tensor = field(default=None, repr=False)  # type: ignore
    
    def __post_init__(self,
                      x: Tensor=None,
                      h_x: Tensor=None,
                      y_pred: Tensor=None,
                      y: Tensor=None):

        # get the batch size:
        for tensor in [x, h_x, y_pred, y]:
            if tensor is not None:
                self.n_samples = tensor.shape[0]
                break
        
        if self.confusion_matrix is None and y_pred is not None and y is not None:
            self.confusion_matrix = get_confusion_matrix(y_pred=y_pred, y=y)

        #TODO: add other useful metrics (potentially ones using x or h_x?)
        if self.confusion_matrix is not None:
            if not isinstance(self.confusion_matrix, Tensor):
                self.confusion_matrix = torch.as_tensor(self.confusion_matrix)
            self.accuracy = get_accuracy(self.confusion_matrix)
            self.class_accuracy = get_class_accuracy(self.confusion_matrix)

    def __add__(self, other: "ClassificationMetrics") -> "ClassificationMetrics":

        # TODO: Might be a good idea to add a `task` attribute to Metrics or
        # LossInfo objects, in order to check that we aren't adding the class
        # accuracies or confusion matrices from different tasks by accident.
        # We could also maybe add them but fuse them properly, for instance by
        # merging the class accuracies and confusion matrices?
        # 
        # For example, if a first metric has class accuracy [0.1, 0.5] 
        # (n_samples=100) and from a task with classes [0, 1] is added to a
        # second Metrics with class accuracy [0.9, 0.8] (n_samples=100) for task
        # with classes [0,3], the resulting Metrics object would have a 
        # class_accuracy of [0.5 (from (0.1+0.9)/2 = 0.5), 0.5, 0 (no data), 0.8]
        # n_samples would then also have to be split on a per-class basis.
        # n_samples could maybe be just the sum of the confusion matrix entries?
        # 
        # As for the confusion matrices, they could be first expanded to fit the
        # range of both by adding empty columns/rows to each and then be added
        # together.
        confusion_matrix: Optional[Tensor] = None
        if self.n_samples == 0:
            return other
        if not isinstance(other, ClassificationMetrics):
            return NotImplemented
        if self.confusion_matrix is None:
            if other.confusion_matrix is None:
                confusion_matrix = None
            else:
                confusion_matrix = other.confusion_matrix.clone()
        elif other.confusion_matrix is None:
            confusion_matrix = self.confusion_matrix.clone()
        else:
            confusion_matrix = self.confusion_matrix + other.confusion_matrix

        result = ClassificationMetrics(
            n_samples=self.n_samples + other.n_samples,
            confusion_matrix=confusion_matrix,
        )
        return result
    
    def to_log_dict(self, verbose: bool=True) -> Dict:
        d = super().to_log_dict()
        d["accuracy"] = float(self.accuracy)
        d["class_accuracy"] = self.class_accuracy.tolist()
        if self.confusion_matrix is not None:
            d["confusion_matrix"] = self.confusion_matrix.tolist()
        return d

    def __str__(self):
        s = super().__str__()
        s = s.replace(f"accuracy={self.accuracy}", f"accuracy={self.accuracy:.3%}")
        return s
    
    def to_pbar_message(self) -> Dict[str, Union[str, float]]:
        message = super().to_pbar_message()
        message["acc"] = f"{self.accuracy:.2%}"
        return message    

@torch.no_grad()
def get_metrics(y_pred: Union[Tensor, np.ndarray],
                y: Union[Tensor, np.ndarray],
                x: Union[Tensor, np.ndarray]=None,
                h_x: Union[Tensor, np.ndarray]=None) -> Union[ClassificationMetrics, RegressionMetrics]:
    y = torch.as_tensor(y)
    y_pred = torch.as_tensor(y_pred)
    if x is not None:
        x = torch.as_tensor(x)
    if h_x is not None:
        h_x = torch.as_tensor(h_x)
    if y.is_floating_point():
        return RegressionMetrics(x=x, h_x=h_x, y_pred=y_pred, y=y)
    else:
        return ClassificationMetrics(x=x, h_x=h_x, y_pred=y_pred, y=y)


@dataclass
class AROC(Metrics):
    aroc: float = 0.
    def __post_init__(self, model: nn.Module, dataset: Dataset, dataset_ood: Dataset=None):
        if dataset_ood is None:
            self.aroc = get_auroc_classification(dataset, model)
        else:
            self.aroc = get_auroc_ood(dataset, dataset_ood, model)


    def to_log_dict(self, verbose: bool=True) -> Dict:
        d = super().to_log_dict()
        d["aroc"] = float(self.aroc)
        return d

    def to_pbar_message(self) -> Dict[str, Union[str, float]]:
        message = super().to_pbar_message()
        message["aroc"] = float(self.aroc.item())
        return message



    
@torch.no_grad()
def get_confusion_matrix(y_pred: Tensor, y: Tensor) -> Tensor:
    """ Taken from https://discuss.pytorch.org/t/how-to-find-individual-class-accuracy/6348
    """
    n_classes = y_pred.shape[-1]
    confusion_matrix = torch.zeros(n_classes, n_classes)
    _, y_preds = torch.max(y_pred, 1)
    for y_t, y_p in zip(y.view(-1), y_preds.view(-1)):
        confusion_matrix[y_t.long(), y_p.long()] += 1
    return confusion_matrix

@torch.no_grad()
def accuracy(y_pred: Tensor, y: Tensor) -> float:
    confusion_mat = get_confusion_matrix(y_pred, y)
    batch_size = y_pred.shape[0]
    _, predicted = y_pred.max(dim=1)
    acc = (predicted == y).sum(dtype=torch.float) / batch_size
    return acc.item()


def get_accuracy(confusion_matrix: Tensor) -> float:
    return (confusion_matrix.diag().sum() / confusion_matrix.sum()).item()


def class_accuracy(y_pred: Tensor, y: Tensor) -> Tensor:
    confusion_mat = get_confusion_matrix(y_pred, y)
    return confusion_mat.diag()/confusion_mat.sum(1).clamp_(1e-10, 1e10)


def get_class_accuracy(confusion_matrix: Tensor) -> Tensor:    
    return confusion_matrix.diag()/confusion_matrix.sum(1).clamp_(1e-10, 1e10)


def prepare_ood_datasets(true_dataset, ood_dataset):
    ood_dataset.transform = true_dataset.transform

    datasets = [true_dataset, ood_dataset]

    anomaly_targets = torch.cat(
        (torch.zeros(len(true_dataset)), torch.ones(len(ood_dataset)))
    )

    concat_datasets = torch.utils.data.ConcatDataset(datasets)

    dataloader = torch.utils.data.DataLoader(
        concat_datasets, batch_size=500, shuffle=False, num_workers=6, pin_memory=False
    )

    return dataloader, anomaly_targets


def get_auroc_ood(true_dataset, ood_dataset, model):
    dataloader, anomaly_targets = prepare_ood_datasets(true_dataset, ood_dataset)

    scores, accuracies = loop_over_dataloader(model, dataloader)

    accuracy = np.mean(accuracies[: len(true_dataset)])
    roc_auc = roc_auc_score(anomaly_targets, scores)

    return accuracy, roc_auc


def get_auroc_classification(dataset, model):
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=500, shuffle=False, num_workers=6, pin_memory=False
    )

    scores, accuracies = loop_over_dataloader(model, dataloader)

    accuracy = np.mean(accuracies)
    roc_auc = roc_auc_score(1 - accuracies, scores)

    return accuracy, roc_auc


def loop_over_dataloader(model, dataloader):
    was_training = model.encoder.training
    model.eval()
    with torch.no_grad():
        scores = []
        accuracies = []
        for data, target in dataloader:
            data = data.cuda()
            target = target.cuda()

            output = model.logits(model.encode(data))[1]
            kernel_distance, pred = output.max(1)

            accuracy = pred.eq(target)
            accuracies.append(accuracy.cpu().numpy())

            scores.append(-kernel_distance.cpu().numpy())

    scores = np.concatenate(scores)
    accuracies = np.concatenate(accuracies)
    if was_training:
        model.train()
    return scores, accuracies