from dataclasses import dataclass, field
from enum import Enum
import copy
from typing import List
from typing import Tuple, Union, Optional
import torch
from torch import Tensor
from common.task import Task
from torchvision.transforms import Compose, Lambda, ToPILImage, ToTensor
from torchvision.transforms.functional import to_tensor

from utils import cuda_available
from common.losses import LossInfo

from common.losses import LossInfo, cosine_similarity_distil_loss, PKT, CRDLoss
from torch.nn import functional as F
from common.metrics import RegressionMetrics, get_metrics
from simple_parsing import mutable_field, choice
from simple_parsing.helpers import Serializable
from tasks.auxiliary_task import AuxiliaryTask
from tasks.byol_task import BYOL_loss

from pl_bolts.models.self_supervised import SimCLR
from pl_bolts.models.self_supervised.simclr.simclr_module import Projection
from pl_bolts.models.self_supervised.simclr import SimCLREvalDataTransform, SimCLRTrainDataTransform

from pl_bolts.datamodules import CIFAR10DataModule, STL10DataModule, ImagenetDataModule

try:
    from .falr_nt.falr.source.models import LinearHead
except ImportError as e:
    print(f"Couldn't import the modules from the falr submodule: {e}")
    print("Make sure to run `git submodule init; git submodule update`")
    exit()

class SimCLRTrainDataTransform_(SimCLRTrainDataTransform):
    def __init__(self, dobble = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dobble = dobble

    def __call__(self, sample):
        #returns three rando transformations of the same sample
        transform = self.train_transform
        xi = transform(sample)
        if self.dobble:
            xj = transform(sample)
        else:
            xj = ToTensor()(sample)
        return xj, xi

class SimCLREvalDataTransform_(SimCLREvalDataTransform):
    def __init__(self, dobble = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dobble = dobble

    def __call__(self, sample):
        #returns three rando transformations of the same sample
        transform = self.test_transform
        xi = transform(sample)
        if self.dobble:
            xj = transform(sample)
        else:
            xj = ToTensor()(sample)
        return xj, xi

class SimCLRTask(AuxiliaryTask, SimCLR):

    @dataclass
    class Options(AuxiliaryTask.Options):
        """ Options for the SimCLR aux task. """
        # Hyperparameters from the falr submodule.
        #coefficient: int = 1   

        #weight given to the long term loss (it 0 its not used)
        lt_lambda: float = 1.

        #temp of contrastive loss
        loss_temperature: float = 0.5

        # the projection size
        proj_dim: int = 128 # the projection size

        #whether to use predictor for cl
        #simclr_use_predictor_cl: bool = 0

        #which distilation loss to use
        lt_loss: str = choice(['crd', 'pkt','cosine'], default='pkt')

        #where to apply distilation loss
        l_distil: str = choice(['repr', 'proj', 'pred'], default='repr')

    def __init__(self, name: str="SimCLR", options: "SimCLRTask.Options"=None):
        self.name: str = name or type(self).__qualname__
        self.options = options or type(self).Options()
        self.device: torch.device = torch.device("cuda" if cuda_available else "cpu")
        self.options: SimCLRTask.Options
        # Set the same values for equivalent hyperparameters
        self.options.image_size = AuxiliaryTask.input_shape[-1]
        self.options.repr_dim = AuxiliaryTask.hidden_size

        self.transform_train =  Compose([
            ToPILImage(),SimCLRTrainDataTransform(self.options.image_size)]) # img1, img2
        self.transform_eval = Compose([ToPILImage(), SimCLREvalDataTransform(self.options.image_size)])
        SimCLR.__init__(self)

        #these targets are updated after each task
        self.predictor_lt = LinearHead(self.options.proj_dim, self.options.repr_dim*2, self.options.proj_dim)
        self.encoder_lt = None
        self.projector_lt = None

        self.current_task: Task = Task(index=0)

        if self.options.lt_loss == 'pkt':
            self.distil_loss_lt = PKT()
        elif self.options.lt_loss == 'crd':
            raise NotImplementedError
            self.distil_loss_lt = CRDLoss(nce_k=16384, nce_t = 0.07, nce_m = 0.5, s_dim = self.options.repr_dim, t_dim = self.options.repr_dim, feat_dim = self.options.proj_dim, n_data = 1000)
        else:
            self.distil_loss_lt = BYOL_loss()

        #factor for distilation loss: used to at the beginning of a task to bring simnclr loss and distillation loss on the same scale
        self.factor_loss_lt = None
    
    
    
    def init_encoder(self):
        return AuxiliaryTask.encoder
    
    def init_projection(self):
        return LinearHead(self.options.repr_dim, self.options.repr_dim*2, self.options.proj_dim)
    
    def forward(self, x):
        h = self.encoder(x)
        z = self.projection(h.view(-1,self.options.repr_dim))
        return h, z

    def training_step(self, img_1, img_2, y, batch_idx, name):
        #if isinstance(self.datamodule, STL10DataModule):
        #    labeled_batch = batch[1]
        #    unlabeled_batch = batch[0]
        #    batch = unlabeled_batch

        #(img_1, img_2), y = batch

        h1, z1 = self.forward(img_1)
        z1_norm = F.normalize(z1, p=2, dim=1)
        h2, z2 = self.forward(img_2)
        z2_norm = F.normalize(z2, p=2, dim=1)

        loss_lt = None
        #long term loss: distilation of representation between the tasks
        if self.current_task.index>0 and self.options.lt_lambda>0:
            with torch.no_grad():
                h3_1 = self.encoder_lt(img_1)
                z3_1 = self.projector_lt(h3_1)
                z3_1_norm = F.normalize(z3_1, p=2, dim=1)

                h3_2 = self.encoder_lt(img_2)   
                z3_2 = self.projector_lt(h3_2)  
                z3_2_norm = F.normalize(z3_2, p=2, dim=1)

            #lt losses
            if self.options.l_distil == 'repr':
                lt_target_1 = h3_1
                lt_target_2 = h3_2

                lt_q_1 = h1
                lt_q_2 = h2

            elif self.options.distill == 'proj':
                lt_target_1 = z3_1
                lt_target_2 = z3_2

                lt_q_1 = z1
                lt_q_2 = z2
            
            if not isinstance(self.distil_loss_lt, CRDLoss) and not isinstance(self.distil_loss_lt, PKT):
                # CRD and PKT normalize internaly
                lt_target_1 = F.normalize(lt_target_1, p=2, dim=1)
                lt_target_2 = F.normalize(lt_target_2, p=2, dim=1)

                lt_q_1 = F.normalize(lt_q_1, p=2, dim=1)
                lt_q_2 = F.normalize(lt_q_2, p=2, dim=1)

            loss_lt_1 = self.distil_loss_lt(lt_q_1, lt_target_2.detach())
            loss_lt_2 = self.distil_loss_lt(lt_q_2, lt_target_1.detach())
            loss_lt = ((loss_lt_1 + loss_lt_2)/2)

        
        loss_ntx = self.loss_func(z1_norm, z2_norm, self.options.loss_temperature)

        if loss_lt is not None:
            if self.factor_loss_lt is None:
                #set once at the beginning of a task
                self.factor_loss_lt = (loss_ntx / loss_lt).detach()

            loss_lt = 0.5 * (self.factor_loss_lt * loss_lt)
            loss_ntx = 0.5 * loss_ntx
            loss_into_lt = LossInfo(name=f'{name}_{self.options.lt_loss}', total_loss=loss_lt)
            loss_info_ntx = LossInfo(name='{name}_ntx_loss', total_loss=loss_ntx)
            loss = loss_info_ntx + loss_into_lt
        else:
            loss_info_ntx = LossInfo(name='{name}_ntx_loss', total_loss=loss_ntx)
            loss = loss_info_ntx 

        return loss, (h1, h2)


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

        if self.encoder.training:
            loss, h = self.training_step(im_q, im_k, y, batch_idx=0, name='train')
        else:
            loss, h = self.training_step(im_q, im_k, im_lt, y, batch_idx=0, name='valid')
        #we return concateneted augmentations of images, loss, and concateneted hidden representations
        return torch.cat([im_q,im_k], dim=0), loss, torch.cat(h, dim=0)

    def preprocess_simclr(self, data:Tensor) -> Tensor:
        x_device = data.device
        data = data.cpu()
        if self.encoder.training:
            data = torch.stack([torch.stack(self.transform_train(x_i)) for x_i in data], dim=0)  # [2*B, C, H, W]  
        else:
            data = torch.stack([torch.stack(self.transform_eval(x_i)) for x_i in data], dim=0)  # [2*B, C, H, W]                    
        return data.to(x_device)

    def on_task_switch(self, new_task: Task, **kwargs) -> None:
        if new_task.index > self.current_task.index:
            self.current_task = new_task
            self.factor_loss_lt = None
            if self.enabled:
                self.encoder_lt = copy.deepcopy(self.encoder).to(self.device)
                self.projector_lt = copy.deepcopy(self.projection).to(self.device)#
                if torch.cuda.device_count() > 1:
                    print("Let's use", torch.cuda.device_count(), "GPUs!")
                    self.projector_lt = nn.DataParallel(self.projector_lt)