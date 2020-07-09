from dataclasses import dataclass, field
from enum import Enum
from typing import List
from typing import Tuple, Union, Optional
import torch
from torch import Tensor
from torchvision.transforms import Compose, Lambda, ToPILImage, ToTensor
from torchvision.transforms.functional import to_tensor

from common.losses import LossInfo

from torch.nn import functional as F
from common.metrics import RegressionMetrics, get_metrics
from simple_parsing import mutable_field
from simple_parsing.helpers import Serializable
from tasks.auxiliary_task import AuxiliaryTask

from pl_bolts.models.self_supervised import SimCLR
from pl_bolts.models.self_supervised.simclr.simclr_module import Projection
from pl_bolts.models.self_supervised.simclr import SimCLREvalDataTransform, SimCLRTrainDataTransform

from pl_bolts.datamodules import CIFAR10DataModule, STL10DataModule, ImagenetDataModule


class SimCLRTask(AuxiliaryTask, SimCLR):

    @dataclass
    class Options(AuxiliaryTask.Options):
        """ Options for the SimCLR aux task. """
        # Hyperparameters from the falr submodule.

    def __init__(self, name: str="SimCLR", options: "SimCLRTask.Options"=None):
        AuxiliaryTask.__init__(self, name=name, options=options)
        #SimCLR.__init__(self, name=name, options=options)
        self.options: SimCLRTask.Options
        # Set the same values for equivalent hyperparameters
        self.options.image_size = AuxiliaryTask.input_shape[-1]
        self.options.repr_dim = AuxiliaryTask.hidden_size

        self.transform_train =  Compose([
            ToPILImage(),SimCLRTrainDataTransform(self.options.image_size)]) # img1, img2
        self.transform_valid = Compose([ToPILImage(), SimCLREvalDataTransform(self.options.image_size)])

        
    
    def init_encoder(self):
        return AuxiliaryTask.encoder
    
    def init_projection(self):
        return Projection(input_dim=AuxiliaryTask.hidden_size, output_dim = 128)
    
    def forward(self, x):
        h = self.encoder(x)
        z = self.projection(h.squeeze())
        return h, z

    def training_step(self, batch, batch_idx):
        if isinstance(self.datamodule, STL10DataModule):
            labeled_batch = batch[1]
            unlabeled_batch = batch[0]
            batch = unlabeled_batch

        (img_1, img_2), y = batch
        h1, z1 = self.forward(img_1)
        h2, z2 = self.forward(img_2)

        # return h1, z1, h2, z2
        loss = self.loss_func(z1, z2, self.hparams.loss_temperature)
        log = {'train_ntx_loss': loss}

        # don't use the training signal, just finetune the MLP to see how we're doing downstream
        if self.online_evaluator:
            if isinstance(self.datamodule, STL10DataModule):
                (img_1, img_2), y = labeled_batch

            with torch.no_grad():
                h1, z1 = self.forward(img_1)

            # just in case... no grads into unsupervised part!
            z_in = z1.detach()

            z_in = z_in.reshape(z_in.size(0), -1)
            mlp_preds = self.non_linear_evaluator(z_in)
            mlp_loss = F.cross_entropy(mlp_preds, y)
            loss = loss + mlp_loss
            log['train_mlp_loss'] = mlp_loss

        result = {
            'loss': loss,
            'log': log
        }

        return result, (h1, h2)


    def get_loss(self, x: Tensor, h_x: Tensor, y_pred: Tensor, y: Tensor=None) -> Tuple[LossInfo, Tensor, Tensor]:
        x = self.preprocess_simclr(x)
        batch = (x.transpose(0,1),y)
        result, h = self.training_step(batch, batch_idx=0)
        loss = result['loss']
        log = result['log']
        loss_info = LossInfo(name=self.name, total_loss=loss)
        loss_info.metrics = log
        return loss_info, h

    
    def preprocess_simclr(self, data:Tensor) -> Tensor:
        x_device = data.device
        data = data.cpu()
        data = torch.stack([torch.stack(self.transform_train(x_i)) for x_i in data], dim=0)  # [2*B, C, H, W]        
        return data.to(x_device)