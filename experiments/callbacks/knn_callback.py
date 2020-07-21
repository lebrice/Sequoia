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
from models.classifier import Classifier
from simple_parsing import field, mutable_field
from utils.logging_utils import pbar, get_logger

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
    """

    # Options for the KNN classifier 
    knn_options: KnnClassifierOptions = mutable_field(KnnClassifierOptions)

    def on_epoch_end(self, trainer: Trainer, pl_module: Classifier):
        self.trainer = trainer
        self.model = pl_module

        train = pl_module.train_dataloader()
        test  = pl_module.test_dataloader()
        val  = pl_module.val_dataloader()

        if isinstance(test, DataLoader):
            test = [test]
        if isinstance(val, DataLoader):
            val = [val]

        h_x, y = self.get_hidden_codes_array(train, description="KNN (Train)")
        train_classes = np.unique(y)
        train_loss, scaler, knn_classifier = fit_knn(
            x=h_x,
            y=y,
            options=self.knn_options
        )
        logger.info(f"Train KNN Acc: {train_loss.accuracy:.2%}")
        total_test_knn_loss = LossInfo("KNN/Test")
        
        for i, (val_dataloader, test_dataloader) in enumerate(zip(test, val)):
            h_x_test, y_test = self.get_hidden_codes_array(test_dataloader,  description=f"KNN (test[{i}])")
            test_classes = np.unique(y_test)
            assert np.array_equal(train_classes, test_classes), (
                f"y and y_test should contain the same classes: "
                f"(train classes: {train_classes}, "
                f"test classes: {test_classes})."
            )
            test_loss = evaluate_knn(
                x_t=h_x_test, y_t=y_test, scaler=scaler, knn_classifier=knn_classifier,
                loss_name=f"KNN/Test[{i}]"
            )
            logger.info(f"Test[{i}]  KNN Acc: {test_loss.accuracy:.2%}")
            total_test_knn_loss.absorb(test_loss)

        logger.info(f"Average Test KNN Acc: {total_test_knn_loss.accuracy:.2%}")

        if self.trainer.logger:
            train_loss.name = "KNN/Train"
            test_loss.name = "KNN/Test"
            self.trainer.logger.log_metrics(train_loss.to_log_dict())
            self.trainer.logger.log_metrics(total_test_knn_loss.to_log_dict())

    def get_hidden_codes_array(self, dataloader: DataLoader, description: str="KNN") -> Tuple[np.ndarray, np.ndarray]:
        """ Gets the hidden vectors and corresponding labels. """
        h_x_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []
        for batch in pbar(dataloader, description):
            x, y = self.model.preprocess_batch(batch)
            # We only do KNN with examples that have a label.
            assert y is not None, f"Should have a 'y' for now! {x}, {y}"
            if y is not None:
                h_x = self.model.encode(x.to(self.model.device))
                h_x_list.append(h_x.detach().cpu().numpy())
                y_list.append(y.detach().cpu().numpy())
        codes = np.concatenate(h_x_list)
        labels = np.concatenate(y_list)
        return codes.reshape(codes.shape[0], -1), labels


def fit_knn(x: np.ndarray, y: np.ndarray, options: KnnClassifierOptions=None) -> Tuple[LossInfo, StandardScaler, KNeighborsClassifier]:
    # print(x.shape, y.shape, x_t.shape, y_t.shape)
    options = options or KnnClassifierOptions()
   
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    # Create and train the Knn Classifier using the options as the kwargs
    knn_classifier = KNeighborsClassifier(**asdict(options)).fit(x, y)

    classes = knn_classifier.classes_
    # print("classes: ", classes)

    # y_pred = knn_classifier.predict(x)
    y_prob = knn_classifier.predict_proba(x)
    # print(y_pred.shape, y_prob.shape, train_score)

    y_logits = np.zeros((y.size, y.max() + 1))
    for i, label in enumerate(classes):
        y_logits[:, label] = y_prob[:, i]
    
    # print(y_prob.shape)
    nce = log_loss(y_true=y, y_pred=y_prob, labels=classes)
    train_loss = LossInfo("KNN", total_loss=nce, y_pred=y_logits, y=y)

    return train_loss, scaler, knn_classifier


def evaluate_knn(x_t: np.ndarray, y_t: np.ndarray, scaler: StandardScaler, knn_classifier: KNeighborsClassifier, loss_name: str="KNN") -> LossInfo:
    # Flatten the inputs to two dimensions only.
    x_t = x_t.reshape(x_t.shape[0], -1)
    assert len(x_t.shape) == 2
    x_t = scaler.transform(x_t)
    y_t_prob = knn_classifier.predict_proba(x_t)
    y_t_logits = np.zeros((y_t.size, y_t.max() + 1))
    
    classes = knn_classifier.classes_
    for i, label in enumerate(classes):
        y_t_logits[:, label] = y_t_prob[:, i]
    nce_t = log_loss(y_true=y_t, y_pred=y_t_prob, labels=classes)
    test_loss = LossInfo(loss_name, total_loss=nce_t, y_pred=y_t_logits, y=y_t)
    return test_loss
