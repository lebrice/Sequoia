from dataclasses import dataclass, field
from enum import Enum
import copy
from typing import List
from functools import partial
from typing import Tuple, Union, Optional
import torch
from torch import nn
from torch import Tensor
from common.task import Task
import torch.nn.functional as F
from simple_parsing import choice, field, list_field
from typing import (Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Type, TypeVar, Union)
from torchvision.transforms import Compose, Lambda, ToPILImage, ToTensor
from torchvision.transforms.functional import to_tensor

from common.losses import LossInfo, cosine_similarity_distil_loss, PKT, CRDLoss
from common.metrics import RegressionMetrics, get_metrics
from simple_parsing import mutable_field
from simple_parsing.helpers import Serializable
from tasks.auxiliary_task import AuxiliaryTask

from utils import cuda_available
from pl_bolts.models.self_supervised.simclr import SimCLREvalDataTransform, SimCLRTrainDataTransform
#from tasks.simclr.simclr_task_ptl import SimCLREvalDataTransform_, SimCLRTrainDataTransform_

try:
    from .simclr.falr_nt.falr.source.models import BYOL, LinearHead
    from .simclr.falr_nt.falr.source.config import HParams, ExperimentType
except ImportError as e:
    print(f"Couldn't import the modules from the falr submodule: {e}")
    print("Make sure to run `git submodule init; git submodule update`")
    exit()

class FalrHParams(HParams, Serializable):
    pass


class BYOL_loss(nn.Module):
    """Probabilistic Knowledge Transfer for deep representation learning
    Code from author: https://github.com/passalis/probabilistic_kt"""
    def __init__(self, *args, **kwargs):
        super(BYOL_loss, self).__init__()

    def forward(self, f_s, f_t):
        return self.u_loss_fn(f_s, f_t)
    
    @staticmethod
    def u_loss_fn(x, y):
        return -2 * F.cosine_similarity(x,y).mean() #(x * y).sum(dim=-1)


class BYOL_Task(AuxiliaryTask, BYOL):

    @dataclass 
    class Options(AuxiliaryTask.Options):
        """ Options for the SimCLR aux task. """
        # Hyperparameters from the falr submodule.
        #coefficient: int = 1 #this one is set in 'aux_task', can b e hardcoded here for debuging

        byol_options: FalrHParams = mutable_field(FalrHParams)

        # the moving average decay factor for the target encoder
        byol_momentum: float = 0.99 

        # the projection size
        proj_dim: int = 128 # the projection size

        #weight given to the long term loss (it 0 its not used)
        lt_lambda: float = 0.
        
        #which distilation loss to use between the tasks
        lt_loss: str = choice(['crd', 'pkt','cosine'], default='pkt')

        #where to apply distilation loss
        l_distil: str = choice(['repr', 'proj', 'pred'], default='repr')
        
    def __init__(self, name: str="BYOL", options: "BYOL_Task.Options"=None):
        #instead of calling AuxiliaryTask.._init_
        self.name: str = name or type(self).__qualname__
        self.options = options or type(self).Options()
        self.device: torch.device = torch.device("cuda" if cuda_available else "cpu")
        self.options: BYOL_Task.Options
        self.previous_task: Task = Task(index=0)
        self.current_task: Task = Task(index=0)  
        BYOL.__init__(self, self.options.byol_options)

        self.options.image_size = AuxiliaryTask.input_shape[-1]
        #self.hparams.double_augmentation = True
        self.options.repr_dim = AuxiliaryTask.hidden_size

        ###init BYOLD###
        self.m = self.options.byol_momentum
        self.u_loss_fn = BYOL_loss()

        # Online network
        self.projector = LinearHead(self.options.repr_dim, self.options.repr_dim*2, self.options.proj_dim)
        self.predictor = LinearHead(self.options.proj_dim, self.options.repr_dim*2, self.options.proj_dim)

        # Target network
        self.encoder_t = copy.deepcopy(self.encoder)
        
        self.projector_t = LinearHead(self.options.repr_dim,  self.options.repr_dim*2, self.options.proj_dim)

        #long-term targets: these target networks are updated after each task (see on def on_task_switch)
        if self.options.lt_lambda>0:
            self.encoder_lt = copy.deepcopy(self.encoder)
            self.projector_lt = LinearHead(self.options.repr_dim,  self.options.repr_dim*2, self.options.proj_dim)
            self.predictor_lt = LinearHead(self.options.proj_dim, self.options.repr_dim*2, self.options.proj_dim)

        if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                self.encoder_t = nn.DataParallel(self.encoder_t)
                self.projector_t = nn.DataParallel(self.projector_t)
                self.projector = nn.DataParallel(self.projector)
                self.predictor = nn.DataParallel(self.predictor)
                self.encoder_lt = nn.DataParallel(self.encoder_lt)
                self.projector_lt = nn.DataParallel(self.projector_lt)
                self.predictor_lt = nn.DataParallel(self.predictor_lt)

        for param_t in self.encoder_t.parameters():
            param_t.requires_grad = False  # not update by gradient
        for param_t in self.projector_t.parameters():
            param_t.requires_grad = False  # not update by gradient

        self.transform_train =  Compose([ToPILImage(), SimCLRTrainDataTransform(input_height=self.options.image_size)]) # img1, img2
        self.transform_eval = Compose([ToPILImage(), SimCLREvalDataTransform(input_height=self.options.image_size)])

        if self.options.lt_loss == 'pkt':
            self.distil_loss_lt = PKT()
        elif self.options.lt_loss == 'crd':
            raise NotImplementedError
            self.distil_loss_lt = self.options.lt_loss(nce_k=16384, nce_t = 0.07, nce_m = 0.5, s_dim = self.options.repr_dim, t_dim = self.options.repr_dim, feat_dim = self.options.proj_dim, n_data = self.current_task.n_data_points)
        else:
            self.distil_loss_lt = BYOL_loss()

    @property
    def encoder(self) -> nn.Module:
        return AuxiliaryTask.encoder
    
    def reinit(self):
        # Online network
        def weight_reset(m):      
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()

        self.projector.apply(weight_reset)
        self.predictor.apply(weight_reset)
        # Target network
        self.encoder_t = copy.deepcopy(self.encoder).to(self.device)
        self.projector_t.apply(weight_reset)
        #these targets are updated after each task
        if self.options.lt_lambda>0:
            self.encoder_lt = copy.deepcopy(self.encoder).to(self.device)
            self.projector_lt.apply(weight_reset)
            self.predictor_lt.apply(weight_reset)

    def get_loss(self, x: Tensor, h_x: Tensor, y_pred: Tensor, y: Tensor=None) -> Tuple[LossInfo, Tensor, Tensor]:
        rg = x.requires_grad

        if len(x.shape)==4:
            x = self.preprocess(x)
        elif len(x.shape)==5:
            pass
        else:
            raise Exception(f'wrong X shape {x.shape}')
        
        im_q, im_k = x.transpose(0,1)
        im_q.requires_grad_(rg)
        im_k.requires_grad_(rg)
        loss, q_e = self.forward(im_q, im_k, None)
        loss_info = LossInfo(name=self.name, total_loss=loss)

        #we return concateneted augmentations of images, loss, and concateneted hidden representations
        return torch.cat([im_q,im_k], dim=0), loss_info, torch.cat(q_e, dim=0)

    def preprocess(self, data:Tensor) -> Tensor:
        x_device = data.device
        data = data.cpu()
        if self.encoder.training:
            data = torch.stack([torch.stack(self.transform_train(x_i)) for x_i in data], dim=0)  # [B, 2, C, H, W]    
        else:            
            data = torch.stack([torch.stack(self.transform_eval(x_i)) for x_i in data], dim=0)  # [B, 2, C, H, W]        
        return data.to(x_device)
        
    def forward(self, im_o, im_t, s_labels) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # compute online features
        #view 1
        q_e_1 = self.encoder(im_o).view(-1,self.options.repr_dim)
        #q_e_1 = F.normalize(q_e_1, p=2, dim=1)
        q_1_proj = self.projector(q_e_1)
        #q_1_proj = F.normalize(q_1_proj, p=2, dim=1)
        q_1 = self.predictor(q_1_proj)
        q_1 = F.normalize(q_1, p=2, dim=1)
        
        #view 2
        q_e_2 = self.encoder(im_t).view(-1,self.options.repr_dim)
        #q_e_2 = F.normalize(q_e_2, p=2, dim=1)
        q_2_proj = self.projector(q_e_2)
        #q_2_proj = F.normalize(q_2_proj, p=2, dim=1)
        q_2 = self.predictor(q_2_proj)
        q_2 = F.normalize(q_2, p=2, dim=1)

        # compute target features
        with torch.no_grad():
            self._update_target_network()
            #view 1
            k_e_1 = self.encoder_t(im_o).view(-1,self.options.repr_dim)
            #k_e_1 = F.normalize(k_e_1, p=2, dim=1)
            k_1 = self.projector_t(k_e_1)
            k_1 = F.normalize(k_1, p=2, dim=1)

            #view 2
            k_e_2 = self.encoder_t(im_t).view(-1,self.options.repr_dim)
            #k_e_2 = F.normalize(k_e_2, p=2, dim=1)
            k_2 = self.projector_t(k_e_2)
            k_2 = F.normalize(k_2, p=2, dim=1)

        #long term targets
        loss_lt = 0
        if self.options.lt_lambda>0 and self.current_task.index>0:
            with torch.no_grad():
                #view 1
                k_e_1_lt = self.encoder_lt(im_o).view(-1,self.options.repr_dim)
                k_1_lt_proj = self.projector_lt(k_e_1_lt)
                #k_1_lt = self.predictor_lt(k_1_lt_proj)
                #k_1_lt = F.normalize(k_1_lt, p=2, dim=1)

                #view 2
                k_e_2_lt = self.encoder_lt(im_t).view(-1,self.options.repr_dim)
                k_2_lt_proj = self.projector_lt(k_e_2_lt)
                #k_2_lt = self.predictor_lt(k_2_lt_proj)
                #k_2_lt = F.normalize(k_2_lt, p=2, dim=1)
            
            #lt losses
            if self.options.l_distil == 'repr':
                cl_k1 = k_e_1_lt
                cl_k2 = k_e_2_lt
                cl_q1 = q_e_1
                cl_q2 = q_e_2

            elif self.options.l_distil == 'proj':
                cl_k1 = k_1_lt_proj
                cl_k2 = k_2_lt_proj
                cl_q1 = q_1_proj
                cl_q2 = q_2_proj
            
            if not isinstance(self.distil_loss_lt, CRDLoss) and not isinstance(self.distil_loss_lt, PKT):
                # CRD and PKT normalize internaly
                cl_k1 = F.normalize(cl_k1, p=2, dim=1)
                cl_k2 = F.normalize(cl_k2, p=2, dim=1)

                cl_q1 = F.normalize(cl_q1, p=2, dim=1)
                cl_q2 = F.normalize(cl_q2, p=2, dim=1)

            loss_lt_1 = self.distil_loss_lt(cl_q2, cl_k1.detach())
            loss_lt_2 = self.distil_loss_lt(cl_q1, cl_k2.detach())

            loss_lt = (loss_lt_1 + loss_lt_2)/2 #.mean()

        #s_logits = self.classifier(q_e)
        loss_1 = self.u_loss_fn(q_1, k_2.detach())
        loss_2 = self.u_loss_fn(q_2, k_1.detach())
        loss = (loss_1 + loss_2)/2
        loss = loss + self.options.lt_lambda * loss_lt
        #return loss and tuple of representations
        return loss, (q_e_1, q_e_2)
    
    def on_task_switch(self, new_task: Task, **kwargs) -> None:
        if new_task.index > self.current_task.index:
            self.current_task = new_task
            if self.enabled:
                self.encoder_lt = copy.deepcopy(self.encoder)
                
                self.projector_lt = copy.deepcopy(self.projector)#LinearHead(self.options.repr_dim,  self.options.repr_dim*2, self.options.proj_dim)
                self.predictor_lt = copy.deepcopy(self.predictor)
                if torch.cuda.device_count() > 1:
                    print("Let's use", torch.cuda.device_count(), "GPUs!")
                    self.projector_lt = nn.DataParallel(self.projector_lt)
                    self.predictor_lt = nn.DataParallel(self.predictor_lt)