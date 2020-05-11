from dataclasses import asdict, dataclass
from typing import Optional, List, Tuple

import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from torch.nn.functional import one_hot

from common.losses import LossInfo


@dataclass
class KnnClassifierOptions:
    """ Set of options for configuring the KnnClassifier. """
    n_neighbors: int = 5 # Number of neighbours.
    metric: str = "cosine"
    algorithm: str = "auto" # See the sklearn docs
    leaf_size: int = 30  # See the sklearn docs
    p: int = 2  # see the sklean docs
    n_jobs: Optional[int] = -1  # see the sklearn docs.


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
    train_loss = LossInfo("KNN", total_loss=train_score, y_pred=y_logits, y=y)

    y_t_pred = clf.predict(x_t)
    y_t_prob = clf.predict_proba(x_t)
    test_score = clf.score(x_t, y_t)
    # print(y_t_pred.shape, y_t_prob.shape, test_score)

    y_t_logits = np.zeros((y_t_pred.size, y_t_pred.max()+1))
    for i, label in enumerate(classes):
        y_t_logits[:, label] = y_t_prob[:, i]
    
    test_loss = LossInfo("KNN", total_loss=test_score, y_pred=y_t_logits, y=y_t)

    # train_acc = np.mean(y_pred == y)
    # test_acc = np.mean(y_t_pred == y_t)
    # print(f"train_acc: {train_acc:.2%}")
    # print(f"test_acc: {test_acc:.2%}")
    return train_loss, test_loss
