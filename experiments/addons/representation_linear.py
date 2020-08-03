from dataclasses import asdict, dataclass, InitVar
from typing import List, Optional, Tuple, Union, Callable

import numpy as np
import torch
from simple_parsing import mutable_field
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from torch import Tensor
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, Dataset

from common.losses import LossInfo
from .addon import ExperimentAddon
from utils.logging_utils import pbar
from utils.eval_utils import get_MLP_losses
from simple_parsing import field

@dataclass
class LinearClassifierOptions:
    """ Set of options for configuring the KnnClassifier. """
    random_state: int = 0
    solver: str = 'lbfgs'
    multi_class: str = 'multinomial'
    max_iter: int = 1000
    verbose: bool = 0


@dataclass  # type: ignore
class LinearClassifierAddon(ExperimentAddon): 
    """ Addon that adds the option of evaluating representations with a KNN.
    
    TODO: Perform the KNN evaluations in different processes using multiprocessing.
    """

    @dataclass
    class Config(ExperimentAddon.Config):
        # Options for the Linear classifier 
        logistic_options: LinearClassifierOptions = mutable_field(LinearClassifierOptions)

    config: InitVar[Config]


    
    def evaluate_logistic(self, 
                 train: Union[Dataset,DataLoader],
                 test: Union[Dataset, DataLoader],
                 preprocess:Callable,
                 description: str="") -> Tuple[LossInfo, LossInfo]:
        
        if isinstance(train, Dataset):
            train = self.get_dataloader(train)
        if isinstance(test, Dataset):
            test = self.get_dataloader(test)

        X, y = preprocess(train)
        Xt, yt = preprocess(test)

        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        Xt = scaler.transform(Xt)

        clf = LogisticRegression(
            random_state=self.config.logistic_options.random_state, solver=self.config.logistic_options.solver, multi_class=self.config.logistic_options.multi_class, max_iter=self.config.logistic_options.max_iter, verbose=0,
        ).fit(X, y)
        classes = clf.classes_
        y_prob = clf.predict_proba(X)
        y_t_prob = clf.predict_proba(Xt)

        y_logits = np.zeros((y.size, y.max() + 1))
        for i, label in enumerate(classes):
            y_logits[:, label] = y_prob[:, i]

        y_t_logits = np.zeros((yt.size, yt.max() + 1))
        for i, label in enumerate(classes):
            y_t_logits[:, label] = y_t_prob[:, i]
        

        nce = log_loss(y_true=y, y_pred=y_prob, labels=classes)
        nce_t = log_loss(y_true=yt, y_pred=y_t_prob, labels=classes)
        

        train_loss = LossInfo("Linear_train", total_loss=nce, y_pred=y_logits, y=y)
        test_loss = LossInfo("Linear_test", total_loss=nce_t, y_pred=y_t_logits, y=yt)
        del X, Xt, yt, y

        return train_loss, test_loss

    def evaluate_MLP(self, 
                 train: Union[Dataset,DataLoader],
                 test: Union[Dataset, DataLoader],
                 preprocess:Callable,
                 description: str="") -> Tuple[LossInfo, LossInfo]:
        
        if isinstance(train, Dataset):
            train = self.get_dataloader(train)
        if isinstance(test, Dataset):
            test = self.get_dataloader(test)

        X, y = preprocess(train)
        Xt, yt = preprocess(test)
        return get_MLP_losses(X, y, Xt, yt, self.config.random_seed, description)