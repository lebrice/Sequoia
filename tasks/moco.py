from dataclasses import dataclass, field
from enum import Enum
from typing import List
from typing import Tuple, Union, Optional
import torch
from torch import Tensor
from torch import nn
from torchvision.transforms import Compose, Lambda, ToPILImage, ToTensor
from torchvision.transforms.functional import to_tensor

from common.losses import LossInfo

from torch.nn import functional as F
from common.metrics import RegressionMetrics, get_metrics
from simple_parsing import mutable_field
from simple_parsing.helpers import Serializable
from tasks.auxiliary_task import AuxiliaryTask

from pl_bolts.models.self_supervised import MocoV2

from pl_bolts.datamodules import CIFAR10DataModule, STL10DataModule, ImagenetDataModule
from pl_bolts.models.self_supervised.moco.transforms import (
    Moco2TrainCIFAR10Transforms,
    Moco2EvalCIFAR10Transforms,
    Moco2TrainSTL10Transforms,
    Moco2EvalSTL10Transforms,
    Moco2TrainImagenetTransforms,
    Moco2EvalImagenetTransforms
)


class MoCo(AuxiliaryTask, MocoV2):

    @dataclass
    class Options(AuxiliaryTask.Options):
        """ Options for the SimCLR aux task. """
        # Hyperparameters from the falr submodule.
        coefficient:int =1

    def __init__(self, name: str="MoCo", options: AuxiliaryTask.Options=None):
        AuxiliaryTask.__init__(self, name=name, options=options)
        #MocoV2.__init__(self, name=name, options=options)
        self.options: MoCo.Options
        # Set the same values for equivalent hyperparameters
        self.options.image_size = AuxiliaryTask.input_shape[-1]
        self.options.repr_dim = AuxiliaryTask.hidden_size

        self.transform_train =  Compose([
            ToPILImage(),Moco2TrainImagenetTransforms(self.options.image_size)]) # img1, img2
        self.transform_valid = Compose([ToPILImage(), Moco2EvalImagenetTransforms(self.options.image_size)])

        
    
    def init_encoders(self, base_encoder):
        return AuxiliaryTask.encoder, AuxiliaryTask.encoder

    def forward(self, img_q, img_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(img_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            if self.use_ddp or self.use_ddp2:
                img_k, idx_unshuffle = self._batch_shuffle_ddp(img_k)

            k = self.encoder_k(img_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            if self.use_ddp or self.use_ddp2:
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.hparams.softmax_temperature

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        labels = labels.type_as(logits)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels


    def get_loss(self, x: Tensor, h_x: Tensor, y_pred: Tensor, y: Tensor=None) -> Tuple[LossInfo, Tensor, Tensor]:
        x = self.preprocess_moco(x)
        batch = (x.transpose(0,1),y)
        result, h = self.training_step(batch, batch_idx=0)
        loss = result['loss']
        log = result['log']
        loss_info = LossInfo(name=self.name, total_loss=loss)
        loss_info.metrics = log
        return loss_info, h

    
    def preprocess_moco(self, data:Tensor) -> Tensor:
        x_device = data.device
        data = data.cpu()
        data = torch.stack([torch.stack(self.transform_train(x_i)) for x_i in data], dim=0)  # [2*B, C, H, W]        
        return data.to(x_device)