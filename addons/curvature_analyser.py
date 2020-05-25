

from torch.utils.data import DataLoader, Dataset, TensorDataset
from typing import (Any, Dict, List, NamedTuple, Optional, Tuple, Type,
                    TypeVar, Union)
from torch import nn as nn
from utils.nngeometry.nngeometry.layercollection import LayerCollection
from models.classifier import Classifier
from .ewc import GaussianPrior
from utils.nngeometry.nngeometry.metrics import FIM
from utils.nngeometry.nngeometry.object.pspace import (PSpaceBlockDiag,
                                                       PSpaceDiag, PSpaceKFAC)



class Analyser(object):
    def __init__(self, model: Classifier):
        '''
        Wrapper constructor.
        @param model: Classifier to wrap
        '''
        self.model = model

    def __getattr__(self, attr):
        # see if this object has attr
        # NOTE do not use hasattr, it goes into
        # infinite recurrsion
        if attr in self.__dict__:
            # this object has it
            return getattr(self, attr)
        # proxy to the wrapped object
        return getattr(self.model, attr)

    def __call__(self, *args, **kwargs):
        return self.model.__call__(*args, **kwargs)

    def analyse_curvature(self, i, multihead:bool, loader: DataLoader, n_way: int):
            self.model.eval()
            print('analysing curvature ...')

            layer_collection = LayerCollection()
            model = nn.Sequential(self.model.encoder, self.model.task_classifiers[str(i)])
            for l, mod in model[0].named_modules():
                mod_class = mod.__class__.__name__
                if mod_class in ['Linear', 'Conv2d']:
                    layer_collection.add_layer_from_model(model, mod)
            if multihead:
                F_linear_kfac = FIM(layer_collection=layer_collection,
                                         model=model,
                                         loader=loader,
                                         representation=PSpaceKFAC,
                                         n_output=n_way,
                                         device=self.device)
            else:
                #single_head
                F_linear_kfac = FIM(layer_collection=layer_collection,
                                    model=model,
                                    loader=loader,
                                    representation=PSpaceKFAC,
                                    n_output=n_way,
                                    device=self.device)
            sum, eig_val, sum_max_ev = F_linear_kfac.sum_of_ev()
            fisher_norm = F_linear_kfac.frobenius_norm()
            return fisher_norm, sum
