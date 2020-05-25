
from copy import copy, deepcopy
from dataclasses import dataclass
from typing import (Any, Dict, List, NamedTuple, Optional, Tuple, Type,
                    TypeVar, Union)

import torch
import torch.nn as nn
from torch import Tensor
from torch import nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from common.losses import LossInfo
from common.task import Task
from experiment import ExperimentBase
from models.classifier import Classifier
from utils.nngeometry.nngeometry.layercollection import LayerCollection
from utils.nngeometry.nngeometry.metrics import FIM
from utils.nngeometry.nngeometry.object.pspace import (PSpaceBlockDiag,
                                                       PSpaceDiag, PSpaceKFAC)
from utils.nngeometry.nngeometry.object.vector import PVector


@dataclass  # type: ignore
class ExperimentWithEWC(ExperimentBase):
    """ Evaluates the model in the same setting as the OML paper's Figure 3.
    """
    use_ewc: bool = False
    # Coefficient of the EWC regularizer. Higher lamda -> more penalty for
    # changing the parameters between tasks.
    ewc_lamda: float = 0.

    def __post_init__(self):
        super().__post_init__()
        if self.ewc_lamda > 0:
            self.use_ewc = True

    def init_model(self) -> Classifier:
        self.logger.debug("init model")
        model = super().init_model()
        if self.use_ewc:
            self.logger.info(f"Using EWC with a lambda of {self.ewc_lamda}")
            #TODO: n_ways should be self.n_classes_per_task, but model outputs 10 way classifier instead of self.n_classes_per_task - way
            model = EWC_wrapper(model, lamda=self.ewc_lamda, n_ways=100, device=self.config.device)
        return model


class GaussianPrior(object):
    def __init__(self, model: torch.nn.Module, n_output:int, loader: DataLoader,
                 reg_matrix: str ="kfac", variant: str ='classif_logits', device: str='cuda'):

        assert reg_matrix=="kfac", 'Only kfac EWC is implement'
        assert variant == "classif_logits", 'Only classif_logits is iplemented as a metric variant'

        self.reg_matrix = reg_matrix
        self.model = model
        layer_collection_bn = LayerCollection()
        layer_collection = LayerCollection()
        for l, mod in model[0].named_modules():
            mod_class = mod.__class__.__name__
            if mod_class in ['Linear', 'Conv2d']:
                layer_collection.add_layer_from_model(model, mod)
            elif mod_class in ['BatchNorm1d', 'BatchNorm2d']:
                layer_collection_bn.add_layer_from_model(model, mod)

        self.F_linear_kfac = FIM(layer_collection=layer_collection,
                            model=model,
                            loader=loader,
                            representation=PSpaceKFAC,
                            n_output=n_output,
                            variant=variant,
                            device=device)

        self.F_bn_blockdiag = FIM(layer_collection=layer_collection_bn,
                             model=model,
                             loader=loader,
                             representation=PSpaceBlockDiag,
                             n_output=n_output,
                             variant=variant,
                             device=device)
        self.prev_params = PVector.from_model(model).clone().detach()

        n_parameters = layer_collection_bn.numel() + layer_collection.numel()
        print(f'\n{str(n_parameters)} parameters')
        print("Done calculating curvature matrix")

    def consolidate(self, new_prior, task):
        self.prev_params = PVector.from_model(new_prior.model).clone().detach()
        if isinstance(self.F_linear_kfac.data, dict):
            for (n, p), (n_, p_) in zip(self.F_linear_kfac.data.items(),new_prior.F_linear_kfac.data.items()):
                for item, item_ in zip(p, p_):
                    item.data = ((item.data*(task))+deepcopy(item_.data))/(task+1) #+ self.F_.data[n]
        else:
            self.F_linear_kfac.data = ((deepcopy(new_prior.F_bn_blockdiag.data)) + self.F_bn_blockdiag.data * (task)) / (task + 1)

        if isinstance(self.F_bn_blockdiag.data, dict):
            for (n, p), (n_, p_) in zip(self.F_bn_blockdiag.data.items(), new_prior.F_bn_blockdiag.data.items()):
                for item, item_ in zip(p, p_):
                    item.data = ((item.data * (task)) + deepcopy(item_.data)) / (task + 1)  # + self.F_.data[n]
        else:
            self.F_bn_blockdiag.data = ((deepcopy(new_prior.F_bn_blockdiag.data)) + self.F_bn_blockdiag.data * (task)) / (task + 1)

    def regularizer(self, model):
        params0_vec = PVector.from_model(model)
        v = params0_vec - self.prev_params
        reg_1 = self.F_linear_kfac.vTMv(v)
        reg_2 = self.F_bn_blockdiag.vTMv(v)
        # print(reg)
        return reg_1 + reg_2


class EWC_wrapper(object):
    def __init__(self, model: Classifier, lamda: float, n_ways: int, device='cuda'):
        '''
        Wrapper constructor.
        @param model: Classifier to wrap
        '''
        self.model = model
        self.device = device
        self.lamda = lamda
        self.current_task_loader: DataLoader = None
        self.prior: Optional[GaussianPrior] = None
        self.n_ways: int = n_ways
        self.tasks_seen: List[int] = []

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

    def regularizer_ewc(self):
        if self.prior is None:
            return Variable(torch.zeros(1)).to(self.device)
        else:
            return self.prior.regularizer(nn.Sequential(self.model.encoder))#, self.model.classifier))

    def get_loss(self, x: Tensor, y: Tensor = None) -> LossInfo:
        loss = self.model.get_loss(x, y)
        ewc_loss = LossInfo(
            name='EWC',
            total_loss=self.lamda * self.regularizer_ewc()
        )
        loss += ewc_loss
        return loss

    def on_task_switch(self, task: Task) -> None:
        self.calculate_ewc_prior(task)

    def calculate_ewc_prior(self, task: Task):
        task_number: int = task.index
        assert isinstance(task_number, int), f"Task number should be an int, got {task_number}"

        if task_number not in self.tasks_seen:
            self.current_task = task
            self.model.eval()
            assert self.current_task_loader is not None, (
                'Task loader should be set to the loader of the current task before switching the tasks'
            )
            print(f"Calculating Fisher on task {self.current_task}")
            #single_head OR multi_head
            prior = GaussianPrior(
                nn.Sequential(self.model.encoder, self.model.classifier),
                self.n_ways,
                self.current_task_loader,
                device=self.device
            )
            if self.prior is not None:
                self.prior.consolidate(prior, task_number)
            else:
                self.prior = prior
            self.tasks_seen.append(task_number)
            del prior
        else:
            print(f'Task {task_number} was learned before, fisher is not updated')
