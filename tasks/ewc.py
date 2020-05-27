from copy import copy, deepcopy
from typing import (Any, Dict, List, NamedTuple, Optional, Tuple, Type,
                    TypeVar, Union)
from torch import nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from common.losses import LossInfo
from common.task import Task
from utils.nngeometry.nngeometry.layercollection import LayerCollection
from utils.nngeometry.nngeometry.metrics import FIM
from utils.nngeometry.nngeometry.object.pspace import (PSpaceBlockDiag,
                                                       PSpaceDiag, PSpaceKFAC)
from utils.nngeometry.nngeometry.object.vector import PVector
from dataclasses import dataclass, field

import torch
from torch import Tensor

from common.losses import LossInfo
from tasks.auxiliary_task import AuxiliaryTask


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


class EWC(AuxiliaryTask):
    @dataclass
    class Options(AuxiliaryTask.Options):
        pass
    def __init__(self,
                 name: str="ewc",
                 options: "EWC.Options"=None):
        super().__init__(name=name, options=options)
        self.options: EWC.Options
        self.current_task_loader = None
        self.n_ways = None
        self.prior = None
        self.tasks_seen=[]

    def on_task_switch(self, task: Task, **kwargs)-> None:
        """ Executed when the task switches (to either a new or known task). """
        self.calculate_ewc_prior(task)
        #set n_ways of the next task
        self.n_ways = len(task.classes)
        #set data loader of the next task
        self.current_task_loader = kwargs.get('train_loader', None)

    def regularizer_ewc(self):
        if self.prior is None:
            return Variable(torch.zeros(1)).to(self.device)
        else:
            return self.prior.regularizer(nn.Sequential(self.AuxiliaryTask.encoder))#, self.model.classifier))

    def get_loss(self, x: Tensor, h_x: Tensor, y_pred: Tensor, y: Tensor = None) -> LossInfo:
        ewc_loss = LossInfo(
            name='EWC',
            total_loss=self.regularizer_ewc()
        )
        return ewc_loss

    def calculate_ewc_prior(self, task: Task):
        task_number: int = task.index
        assert isinstance(task_number, int), f"Task number should be an int, got {task_number}"
        if task_number>0:
            if task_number not in self.tasks_seen:
                AuxiliaryTask.encoder.eval()
                AuxiliaryTask.classifier.eval()
                assert self.current_task_loader is not None, (
                    'Task loader should be set to the loader of the current task before switching the tasks'
                )
                print(f"Calculating Fisher on task {self.current_task}")
                #single_head OR multi_head
                prior = GaussianPrior(
                    nn.Sequential(AuxiliaryTask.encoder, AuxiliaryTask.classifier),
                    self.n_ways,
                    self.current_task_loader,
                    device= 'cuda' if self.device==torch.device("cuda") else 'cpu'
                )
                if self.prior is not None:
                    self.prior.consolidate(prior, task_number)
                else:
                    self.prior = prior
                self.tasks_seen.append(task_number)
                del prior
            else:
                print(f'Task {task_number} was learned before, fisher is not updated')
        else:
            print('No EWC on task 0')

