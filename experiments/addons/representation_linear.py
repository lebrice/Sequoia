from dataclasses import asdict, dataclass, InitVar
from typing import List, Optional, Tuple, Union

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
                 description: str="") -> Tuple[LossInfo, LossInfo]:
        
        if not isinstance(train, DataLoader):
            train = self.get_dataloader(train)
        if not isinstance(test, DataLoader):
            test = self.get_dataloader(test)

        def get_hidden_codes_array(dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
            """ Gets the hidden vectors and corresponding labels. """
            was_training = self.model.training
            self.model.eval()
            h_x_list: List[np.ndarray] = []
            y_list: List[np.ndarray] = []
            for batch in pbar(dataloader, description):
                x, y = self.preprocess(batch)
                # We only do KNN with examples that have a label.
                if y is not None:
                    h_x = self.model.encode(x)
                    h_x_list.append(h_x.detach().cpu().numpy())
                    y_list.append(y.detach().cpu().numpy())
            if was_training:
                self.model.train()
            return np.concatenate(h_x_list), np.concatenate(y_list)

        X, y = get_hidden_codes_array(train)
        Xt, yt = get_hidden_codes_array(test)

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
                 description: str="") -> Tuple[LossInfo, LossInfo]:
        
        if not isinstance(train, DataLoader):
            train = self.get_dataloader(train)
        if not isinstance(test, DataLoader):
            test = self.get_dataloader(test)

        def get_hidden_codes_array(dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
            """ Gets the hidden vectors and corresponding labels. """
            was_training = self.model.training
            self.model.eval()
            h_x_list: List[np.ndarray] = []
            y_list: List[np.ndarray] = []
            for batch in pbar(dataloader, description):
                x, y = self.preprocess(batch)
                y = self.model.rescale_target(y)
                # We only do KNN with examples that have a label.
                if y is not None:                    
                    x, y, _ = self.model.preprocess_inputs(x,y, None)
                    h_x = self.model.encode(x)
                    h_x_list.append(h_x.detach().cpu().numpy())
                    y_list.append(y.detach().cpu().numpy())
            if was_training:
                self.model.train()
            return np.concatenate(h_x_list), np.concatenate(y_list)

        X, y = get_hidden_codes_array(train)
        Xt, yt = get_hidden_codes_array(test)

        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        Xt = scaler.transform(Xt)
        clf = MLPClassifier(random_state=self.config.random_seed, max_iter=100).fit(X, y)
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

        from tasks.tasks import Tasks
        train_loss = LossInfo("Linear_train")
        train_loss= train_loss + LossInfo(Tasks.SUPERVISED, total_loss=nce, y_pred=y_logits, y=y)

        test_loss = LossInfo("Linear_test")
        test_loss = test_loss + LossInfo(Tasks.SUPERVISED, total_loss=nce_t, y_pred=y_t_logits, y=yt)
        del X, Xt, yt, y

        return train_loss, test_loss

    