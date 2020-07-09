from dataclasses import dataclass, field
from enum import Enum
from typing import List
from typing import Tuple, Union, Optional
import torch
import copy
from torch import Tensor
from torch import nn
from torchvision.transforms import Compose, Lambda, ToPILImage, ToTensor
from torchvision.transforms.functional import to_tensor

from common.losses import LossInfo


from utils import cuda_available
import torch.nn.functional as F
from common.metrics import RegressionMetrics, get_metrics
from simple_parsing import mutable_field
from simple_parsing.helpers import Serializable
from tasks.auxiliary_task import AuxiliaryTask

try:
    from .simclr.falr_nt.falr.source.models import MoCo, LinearHead
    from .simclr.falr_nt.falr.source.config import HParams, ExperimentType


except ImportError as e:
    print(f"Couldn't import the modules from the falr submodule: {e}")
    print("Make sure to run `git submodule init; git submodule update`")
    exit()


from pl_bolts.datamodules import CIFAR10DataModule, STL10DataModule, ImagenetDataModule
from pl_bolts.models.self_supervised.moco.transforms import (
    Moco2TrainCIFAR10Transforms,
    Moco2EvalCIFAR10Transforms,
    Moco2TrainSTL10Transforms,
    Moco2EvalSTL10Transforms,
    Moco2TrainImagenetTransforms,
    Moco2EvalImagenetTransforms
)


class FalrHParams(HParams, Serializable):
    pass


class MoCo_Task(AuxiliaryTask, MoCo):
    @dataclass
    class Options(AuxiliaryTask.Options):
        """ Options for the SimCLR aux task. """
        # Hyperparameters from the falr submodule.
        coefficient:int =1

        falr_options: FalrHParams = mutable_field(FalrHParams)

        moco_momentum:float = 0.999 #the moving average decay for moco encoder

        moco_size: int = 1280*7 #size of moco replay buffer
        xent_temp: float = 0.5 #tempreture for xent loss

    def __init__(self, name: str="MoCo", options: AuxiliaryTask.Options=None):
        #instead of calling AuxiliaryTask._init_
        self.name: str = name or type(self).__qualname__
        self.options = options or type(self).Options()
        self.device: torch.device = torch.device("cuda" if cuda_available else "cpu")
        self.options: MoCo_Task.Options

        self.options.image_size = AuxiliaryTask.input_shape[-1]
        self.options.proj_dim = self.options.falr_options.proj_dim
        self.options.cifar = 1 #ignore
        self.options.repr_dim = AuxiliaryTask.hidden_size
        self.options._encoder_k = copy.deepcopy(self.encoder) 
        self.options.torchvision_model = self.options.falr_options.torchvision_model
        MoCo.__init__(self, self.options)
        self.batch_size = None

        self.transform_train =  Compose([
            ToPILImage(),Moco2TrainImagenetTransforms(self.options.image_size)]) # img1, img2
        self.transform_valid = Compose([ToPILImage(), Moco2EvalImagenetTransforms(self.options.image_size)])

    @property
    def encoder(self):
        return AuxiliaryTask.encoder
    
    @encoder.setter
    def encoder(self, encoder):
        pass

    @property
    def encoder_k(self):
        return self.options._encoder_k

    @encoder_k.setter
    def encoder_k(self, encoder_k):
        pass

    
    
    def forward(self, im_q, im_k) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q_e = self.encoder(im_q)
        q = self.projector(q_e)  # queries: NxC
        q = F.normalize(q, p=2, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            k = self.projector_k(self.encoder_k(im_k))  # keys: NxC
            k = F.normalize(k, p=2, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        # dequeue and enqueue
        if self.K % k.shape[0] == 0:
            #k = k[:len(k)-(self.K % k.shape[0])]
            self._dequeue_and_enqueue(k)

        return logits, labels, q_e


    def get_loss(self, x: Tensor, h_x: Tensor, y_pred: Tensor, y: Tensor=None) -> Tuple[LossInfo, Tensor, Tensor]:
        if self.batch_size is None:
            self.batch_size = len(x)
        x = self.preprocess_moco(x)
        img_q, img_k = x.transpose(0,1)  
        u_logits, u_labels, h = self.forward(img_q, img_k)
        loss = F.cross_entropy(u_logits.float(), u_labels.long())        
        loss_info = LossInfo(name=self.name, total_loss=loss)
        return loss_info, h

    
    def preprocess_moco(self, data:Tensor) -> Tensor:
        x_device = data.device
        data = data.cpu()
        data = torch.stack([torch.stack(self.transform_train(x_i)) for x_i in data], dim=0)  # [2*B, C, H, W]        
        return data.to(x_device)