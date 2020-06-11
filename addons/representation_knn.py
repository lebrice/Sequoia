from dataclasses import asdict, dataclass, InitVar
from typing import List, Optional, Tuple, Union

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
from utils.logging_utils import pbar
from simple_parsing import field

@dataclass
class KnnClassifierOptions:
    """ Set of options for configuring the KnnClassifier. """
    n_neighbors: int = field(default=5, alias="n_neighbours") # Number of neighbours.
    metric: str = "cosine"
    algorithm: str = "auto" # See the sklearn docs
    leaf_size: int = 30  # See the sklearn docs
    p: int = 2  # see the sklean docs
    n_jobs: Optional[int] = -1  # see the sklearn docs.


@dataclass  # type: ignore
class ExperimentWithKNN(ExperimentBase): 
    """ Addon that adds the option of evaluating representations with a KNN.
    
    TODO: Perform the KNN evaluations in different processes using multiprocessing.
    """

    @dataclass
    class Config(ExperimentBase.Config):
        # Options for the KNN classifier 
        knn_options: KnnClassifierOptions = mutable_field(KnnClassifierOptions)

    config: InitVar[Config]

    @torch.no_grad()
    def test_knn(self,
                 train: Union[Dataset,DataLoader],
                 test: Union[Dataset, DataLoader],
                 description: str="") -> Tuple[LossInfo, LossInfo]:
        """TODO: Test the representations using a KNN classifier. """
        if not isinstance(train, DataLoader):
            train = self.get_dataloader(train)
        if not isinstance(test, DataLoader):
            test = self.get_dataloader(test)
        
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

        h_x, y = get_hidden_codes_array(train)
        h_x_test, y_test = get_hidden_codes_array(test)
        
        train_loss, test_loss = evaluate_knn(
            x=h_x, y=y,
            x_t=h_x_test, y_t=y_test, options=self.config.knn_options
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

    x_t = scaler.transform(x_t)
    # y_t_pred = knn_classifier.predict(x_t)
    y_t_prob = knn_classifier.predict_proba(x_t)
    # print(y_t_pred.shape, y_t_prob.shape, test_score)

    y_t_logits = np.zeros((y_t.size, y_t.max() + 1))
    for i, label in enumerate(classes):
        y_t_logits[:, label] = y_t_prob[:, i]
    
    nce_t = log_loss(y_true=y_t, y_pred=y_t_prob, labels=classes)
    test_loss = LossInfo("KNN", total_loss=nce_t, y_pred=y_t_logits, y=y_t)

    # train_acc = np.mean(y_pred == y)
    # test_acc = np.mean(y_t_pred == y_t)
    # print(f"train_acc: {train_acc:.2%}")
    # print(f"test_acc: {test_acc:.2%}")
    return train_loss, test_loss

