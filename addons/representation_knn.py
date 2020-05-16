from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from simple_parsing import mutable_field
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from torch import Tensor
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, Dataset

from common.losses import LossInfo
from experiment import ExperimentBase
from utils.logging import pbar


@dataclass
class KnnClassifierOptions:
    """ Set of options for configuring the KnnClassifier. """
    n_neighbors: int = 5 # Number of neighbours.
    metric: str = "cosine"
    algorithm: str = "auto" # See the sklearn docs
    leaf_size: int = 30  # See the sklearn docs
    p: int = 2  # see the sklean docs
    n_jobs: Optional[int] = -1  # see the sklearn docs.


@dataclass  # type: ignore
class ExperimentWithKNN(ExperimentBase): 
    """ Addon that adds the option of evaluating representations with a KNN. """
    knn_options: KnnClassifierOptions = mutable_field(KnnClassifierOptions)

    @torch.no_grad()
    def test_knn(self, train_dataset: Dataset, test_dataset: Dataset, description: str="") -> Tuple[LossInfo, LossInfo]:
        """TODO: Test the representations using a KNN classifier. """
        
        def get_hidden_codes_array(dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
            """ Gets the hidden vectors and corresponding labels. """
            h_x_list: List[np.ndarray] = []
            y_list: List[np.ndarray] = []
            for batch in pbar(dataloader, description):
                x, y = self.preprocess(batch)
                # We only do KNN with examples that have a label.
                if y is not None:
                    h_x = self.model.encode(x)
                    h_x_list.append(h_x.detach().cpu().numpy())
                    y_list.append(y.detach().cpu().numpy())
            return np.concatenate(h_x_list), np.concatenate(y_list)
        
        train_dataloader = self.get_dataloader(train_dataset)
        h_x, y = get_hidden_codes_array(train_dataloader)
        
        test_dataloader = self.get_dataloader(test_dataset)
        h_x_test, y_test = get_hidden_codes_array(test_dataloader)
        
        train_loss, test_loss = evaluate_knn(
            x=h_x, y=y,
            x_t=h_x_test, y_t=y_test,
        )
        # train_loss = LossInfo("KNN-Train", y_pred=y_pred_train, y=y)
        # test_loss = LossInfo("KNN-Test", y_pred=y_pred_train, y=y)
        return train_loss, test_loss


def evaluate_knn(x: np.ndarray, y: np.ndarray, x_t: np.ndarray, y_t: np.ndarray, options: KnnClassifierOptions=None) -> Tuple[LossInfo, LossInfo]:
    # print(x.shape, y.shape, x_t.shape, y_t.shape)
    options = options or KnnClassifierOptions()
    
    # Flatten the inputs to two dimensions only.
    x = x.reshape(x.shape[0], -1)
    x_t = x_t.reshape(x_t.shape[0], -1)

    scaler = StandardScaler()

    assert len(x.shape) == 2
    assert len(x_t.shape) == 2
    train_classes = np.unique(y)
    test_classes = np.unique(y_t)
    assert np.array_equal(train_classes, test_classes), f"y and y_test should contain the same classes: (y: {train_classes}, y_t: {test_classes})."
    
    x = scaler.fit_transform(x)
    x_t = scaler.transform(x_t)

    clf = KNeighborsClassifier(**asdict(options)).fit(x, y)
    classes = clf.classes_
    # print("classes: ", classes)

    y_pred = clf.predict(x)
    y_prob = clf.predict_proba(x)
    train_score = clf.score(x, y)
    # print(y_pred.shape, y_prob.shape, train_score)

    y_logits = np.zeros((y_pred.size, y_pred.max()+1))
    for i, label in enumerate(classes):
        y_logits[:, label] = y_prob[:, i]
    
    # print(y_prob.shape)
    nce = log_loss(y_true=y, y_pred=y_prob, labels=classes)
    train_loss = LossInfo("KNN", total_loss=nce, y_pred=y_logits, y=y)

    y_t_pred = clf.predict(x_t)
    y_t_prob = clf.predict_proba(x_t)
    test_score = clf.score(x_t, y_t)
    # print(y_t_pred.shape, y_t_prob.shape, test_score)

    y_t_logits = np.zeros((y_t_pred.size, y_t_pred.max()+1))
    for i, label in enumerate(classes):
        y_t_logits[:, label] = y_t_prob[:, i]
    
    nce_t = log_loss(y_true=y_t, y_pred=y_t_prob, labels=classes)
    test_loss = LossInfo("KNN", total_loss=nce_t, y_pred=y_t_logits, y=y_t)

    # train_acc = np.mean(y_pred == y)
    # test_acc = np.mean(y_t_pred == y_t)
    # print(f"train_acc: {train_acc:.2%}")
    # print(f"test_acc: {test_acc:.2%}")
    return train_loss, test_loss

