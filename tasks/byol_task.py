from dataclasses import dataclass, field
from enum import Enum
import copy
from typing import List
from typing import Tuple, Union, Optional
import torch
from torch import Tensor
import torch.nn.functional as F
from torchvision.transforms import Compose, Lambda, ToPILImage, ToTensor
from torchvision.transforms.functional import to_tensor

from common.losses import LossInfo
from common.metrics import RegressionMetrics, get_metrics
from simple_parsing import mutable_field
from simple_parsing.helpers import Serializable
from tasks.auxiliary_task import AuxiliaryTask

from utils import cuda_available
from pl_bolts.models.self_supervised.simclr import SimCLREvalDataTransform, SimCLRTrainDataTransform

try:
    from .simclr.falr_nt.falr.source.models import BYOL, LinearHead
    from .simclr.falr_nt.falr.source.config import HParams, ExperimentType


except ImportError as e:
    print(f"Couldn't import the modules from the falr submodule: {e}")
    print("Make sure to run `git submodule init; git submodule update`")
    exit()

class FalrHParams(HParams, Serializable):
    pass



class BYOL_Task(AuxiliaryTask, BYOL):

    @dataclass
    class Options(AuxiliaryTask.Options):
        """ Options for the SimCLR aux task. """
        # Hyperparameters from the falr submodule.
        byol_options: FalrHParams = mutable_field(FalrHParams)

        byol_momentum: float = 0.99 # the moving average decay factor for the target encoder

    def __init__(self, name: str="BYOL", options: "BYOL_Task.Options"=None):
        #instead of calling AuxiliaryTask.._init_
        self.name: str = name or type(self).__qualname__
        self.options = options or type(self).Options()
        self.device: torch.device = torch.device("cuda" if cuda_available else "cpu")
        self.options: BYOL_Task.Options
        BYOL.__init__(self, self.options.byol_options)

        self.options.image_size = AuxiliaryTask.input_shape[-1]
        #self.hparams.double_augmentation = True
        self.options.repr_dim = AuxiliaryTask.hidden_size

        ###init BYOLD###
        self.u_loss_fn = F.mse_loss
        self.m = self.options.byol_momentum

        # Online network
        self.projector = LinearHead(self.options.repr_dim, self.options.repr_dim*2, self.options.byol_options.proj_dim)
        self.predictor = LinearHead(self.options.byol_options.proj_dim, self.options.repr_dim*2, self.options.byol_options.proj_dim)

        # Target network
        self.encoder_t = copy.deepcopy(self.encoder)
        self.projector_t = LinearHead(self.options.repr_dim,  self.options.repr_dim*2, self.options.byol_options.proj_dim)

        for param_t in self.encoder_t.parameters():
            param_t.requires_grad = False  # not update by gradient
        for param_t in self.projector_t.parameters():
            param_t.requires_grad = False  # not update by gradient

        self.transform_train =  Compose([
            ToPILImage(),SimCLRTrainDataTransform(self.options.image_size)]) # img1, img2
        self.transform_valid = Compose([ToPILImage(), SimCLREvalDataTransform(self.options.image_size)])
    
    @property
    def encoder(self):
        return AuxiliaryTask.encoder

    def get_loss(self, x: Tensor, h_x: Tensor, y_pred: Tensor, y: Tensor=None) -> Tuple[LossInfo, Tensor, Tensor]:
        x = self.preprocess(x)
        im_q, im_k = x.transpose(0,1)
        u_logits, u_target, q_e = self.forward(im_q, im_k, None)
        loss = self.u_loss_fn(u_logits, u_target)
        loss_info = LossInfo(name=self.name, total_loss=loss)
        return loss_info, q_e

    def preprocess(self, data:Tensor) -> Tensor:
        x_device = data.device
        data = data.cpu()
        data = torch.stack([torch.stack(self.transform_train(x_i)) for x_i in data], dim=0)  # [2*B, C, H, W]        
        return data.to(x_device)
    
    def forward(self, im_o, im_t, s_labels) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # compute online features
        q_e = self.encoder(im_o).view(-1,self.options.repr_dim)
        q = self.projector(q_e)
        q = self.predictor(q)
        q = F.normalize(q, p=2, dim=1)

        # compute target features
        with torch.no_grad():
            self._update_target_network()
            k = self.projector_t(self.encoder_t(im_t).view(-1,self.options.repr_dim))
            k = F.normalize(k, p=2, dim=1)
        
        #s_logits = self.classifier(q_e)

        return q, k, q_e