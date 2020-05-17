
import torch
import torch.nn as nn
from copy import copy, deepcopy
from common.losses import LossInfo
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from models.classifier import Classifier
from utils.nngeometry.nngeometry.layercollection import LayerCollection
from utils.nngeometry.nngeometry.object.pspace import PSpaceKFAC, PSpaceBlockDiag,PSpaceDiag
from utils.nngeometry.nngeometry.object.vector import PVector
from utils.nngeometry.nngeometry.metrics import FIM
from typing import (Any, Dict, List, NamedTuple, Optional, Tuple, Type, TypeVar, Union)

class GaussianPrior(object):
    def __init__(self, model: torch.nn.Module, n_output:int, loader: DataLoader,
                 reg_matrix: str ="kfac", variant: str ='classif_logits', device: str='cuda'):

        assert reg_matrix=="kfac", print('Only kfac EWC is implement')
        assert variant == "classif_logits", print('Only classif_logits is iplemented as a metric variant')

        self.reg_matrix = reg_matrix
        print("Calculating Fisher " + reg_matrix)
        self.model = model
        layer_collection_bn = LayerCollection()
        layer_collection = LayerCollection()
        for l, mod in model.named_modules():
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
    def __init__(self, model:Classifier, lamda: float, n_ways: int, device='cuda'):
        '''
        Wrapper constructor.
        @param model: Classifier to wrap
        '''
        self.model = model
        self.device = device
        self.lamda = lamda
        self.current_task_loader = None
        self.prior = None
        self.n_ways = n_ways
        self.tasks_seen = []
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
            return self.prior.regularizer(nn.Sequential(*[self.model.encoder, self.model.classifier]))

    def get_loss(self, x: Tensor, y: Tensor = None) -> LossInfo:
        reg = LossInfo('Train')
        reg.total_loss = self.lamda * self.regularizer_ewc()
        loss = self.model.get_loss(x,y) + reg
        return loss

    @property
    def current_task_id(self) -> Optional[str]:
        #getter
        return self.model._current_task_id

    @current_task_id.setter
    def current_task_id(self, value: Optional[Union[int, str]]):
        #setter and on task switch
        assert type(value) == int, print("When switching tasks with ewc, use the task number as task_id")
        self.on_task_switch(value)
        self.model.current_task_id = value

    def on_task_switch(self, task_number: int):
        if task_number>0:
            if task_number not in self.tasks_seen:
                self.current_task = task_number
                self.model.eval()
                assert self.current_task_loader!=None, print('Task loader should be set to the loader of the current task before switching the tasks')
                prior = GaussianPrior(nn.Sequential(*[self.model.encoder, self.model.classifier]), self.n_ways, self.current_task_loader, device=self.device)
                if self.prior is not None:
                    self.prior.consolidate(prior, task_number)
                else:
                    self.prior = prior
                self.tasks_seen.append(task_number)
                del prior
            else:
                print(f'Task {task_number} was learned before, fisher is not updated')
        else:
           print('Learning task 0, no EWC')