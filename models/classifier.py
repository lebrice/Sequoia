import copy
import os
import hashlib
from timeit import default_timer as timer
from functools import partial
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import (Any, Callable, Dict, List, NamedTuple, Optional, Tuple,
                    Type, TypeVar, Union)

import torch
from torch import Tensor, nn, optim

from torch.optim.lr_scheduler import StepLR
from pl_bolts.optimizers.layer_adaptive_scaling import LARS
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import models
from models.model_parallel import PipelineParallelResNet50
from models.resnet_small import ResNet18
from models.CNN13 import CNN13
from torchvision.utils import save_image

from common.layers import ConvBlock, Flatten
from common.losses import LossInfo
from common.metrics import accuracy, get_metrics
from common.task import Task
from config import Config
from models.output_head import OutputHead, LinearOutputHead, OutputHead_DUQ
from simple_parsing import MutableField as mutable_field
from simple_parsing import choice, field, list_field
from tasks import AuxiliaryTask, AuxiliaryTaskOptions, Tasks
from utils.json_utils import Serializable
from utils.logging_utils import get_logger
from utils.utils import fix_channels
from tasks.mixup import sup_mixup

logger = get_logger(__file__)

encoder_models: Dict[str, Type[nn.Module]] = {
    "simple_cnn": None,
    "vgg16": models.vgg16, #  {'encoder': models.vgg16, 'z': 512, 'identity': 'classifier'},
    "resnet18": models.resnet18, # {'encoder': models.resnet18, 'z': 512, 'identity': 'fc'},
    "resnet34": models.resnet34,
    "resnet50": models.resnet50, 
    "resnet50_parallel": PipelineParallelResNet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
    "alexnet": models.alexnet,
    # "squeezenet": models.squeezenet1_0,  # Not supported yet (weird output shape)
    "densenet": models.densenet161,
    "cnn13": CNN13,
    # "inception": models.inception_v3,  # Not supported yet (creating model takes forever?)
    # "googlenet": models.googlenet,  # Not supported yet (creating model takes forever?)
    # "shufflenet": models.shufflenet_v2_x1_0,
    # "mobilenet": models.mobilenet_v2,
    # "resnext50_32x4d": models.resnext50_32x4d,
    # "wide_resnet50_2": models.wide_resnet50_2,
    # "mnasnet": models.mnasnet1_0,
}

output_heads: Dict[str, Type[nn.Module]] = {
    "linear": LinearOutputHead,
    "rbf": OutputHead_DUQ #https://arxiv.org/pdf/2003.02037.pdf
}

optimizers: Dict[str, Type[torch.optim.Optimizer]] = {
        "adam":torch.optim.Adam,
        "sgd": partial(torch.optim.SGD, momentum=0.9, weight_decay=1e-6)
        }

def key_for_dict(dict_in: Dict) -> Callable:
    def key_for_model(encoder_model_fn: Union[Callable[[Any], nn.Module], Type[nn.Module]]) -> str:
        """Returns the key of the given encoding model in the `dict_in` dict.

        Args:
            encoder_model_fn (Union[Callable[[Any], nn.Module], Type[nn.Module]]): 
                A model class or model function.

        Raises:
            RuntimeError: If there is no value of `encoder_model_fn` in the dict.

        Returns:
            str: The key in `dict_in` which has `encoder_model_fn` as value.
        """
        if encoder_model_fn is None:
            return None
        for k, v in dict_in.items():
            if v is encoder_model_fn:
                return k
        raise RuntimeError(f"Can't find the key for encoder fn {encoder_model_fn} in the dict {dict_in}")
    return key_for_model


class Classifier(nn.Module):
    @dataclass
    class HParams(Serializable):
        """ Set of hyperparameters for the classifier.

        We use [simple_parsing](www.github.com/lebrice/simpleparsing) to
        generate command-line arguments for each attribute of this class.
        """
        #for mixup of labeled data: if 0 no mixup is used, otherwise the alpha parameter for the beta distribution from where the mixing lambda is drawn (mainly implemented for ICT)
        mixup_sup_alpha: float = 0.

        batch_size: int = 128   # Input batch size for training.
        learning_rate: float = field(default=1e-3, alias="-lr")  # learning rate.
        weight_decay: float = 1e-6 #weight decay
        momentum: float = 0.9 #momentum
        lars_eta: float = 0.001 #lars eta


        lr_sched_step: float = 0 #lr scheduler step

        lr_sched_gamma: float = 0.5

        # Dimensions of the hidden state (feature extractor/encoder output).
        hidden_size: int = 512 #100

        #compute supervised loss on top of semi-features (no extra forward pass for supervised loss part)
        entangle_sup: bool = True 

        # Prevent gradients of the classifier from backpropagating into the encoder.
        detach_classifier: bool = False

        # Use an encoder architecture from the torchvision.models package.
        encoder_model: Type[nn.Module] = choice(
            encoder_models,
            default="simple_cnn",
            encoding_fn=key_for_dict(encoder_models),
            decoding_fn=encoder_models.get,
        )

        # Use the pretrained weights of the ImageNet model from torchvision.
        pretrained_model: bool = False

        # Freeze the weights of the pretrained encoder (except the last layer,
        # which projects from their hidden size to ours).
        freeze_pretrained_model: bool = False

        # Wether to create one output head per task.
        # TODO: It makes no sense to have multihead=True when the model doesn't
        # have access to task labels. Need to figure out how to manage this between TaskIncremental and Classifier.
        multihead: bool = True

        aux_tasks: AuxiliaryTaskOptions = field(default_factory=AuxiliaryTaskOptions)

        # Use either adam or sgd optimizer
        optimizer: str = choice(['sgd', 'adam', 'lars'], default='sgd')

        #output head type 
        type_output_head: Type[nn.Module] = choice(
            output_heads,
            default="linear",
            encoding_fn=key_for_dict(output_heads),
            decoding_fn=output_heads.get,
        )

        #Weight for gradient penalty (default: 0)"
        l_gradient_penalty: float = 0.

    def __init__(self,
                 input_shape: Tuple[int, ...],
                 num_classes: int,
                 encoder: nn.Module,
                 hparams: HParams,
                 config: Config):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        print(self.num_classes)
        # Feature extractor
        self.encoder = encoder 
        # Classifier output layer 
        self.hparams: Classifier.HParams = hparams
        self.config = config
        self.hidden_size = hparams.hidden_size

        self.in_device = self.out_device = self.device = self.config.device
        if isinstance(self.config.device, Tuple):
            self.in_device = self.config.device[0]
            self.out_device = self.config.device[-1]


        # Classes of the current "task".
        # By default, contains all the classes in `range(0, self.num_classes)`.
        # When using a multihead approach (e.g. EWC), set `current_task` to the
        # classes found within the current task when training or evaluating.
        # NOTE: Order matters: task (0, 1) is not the same as (1, 0) (for now)
        # TODO: Replace the multiple classifier heads with something like CN-DPM so we can actually do task-free CL.
        self._default_task = Task(classes=list(range(self.num_classes)))
        self._current_task = self._default_task
        
        
        # Hyperparameters of the "output head" module.
        self.output_head: OutputHead.HParams = self.hparams.type_output_head.HParams()
        # Classifier for the default task.
        self.default_output_head = self.hparams.type_output_head(
            input_size=self.hidden_size,
            output_size=self.num_classes,
            hparams=self.output_head
        )

        if isinstance(self.default_output_head, LinearOutputHead):
            self.classification_loss = nn.CrossEntropyLoss()
        else:
            #we are in DUQ mode
            def bce_loss_fn(y_pred, y):
                y = F.one_hot(y, self.num_classes).float() 
                bce = F.binary_cross_entropy(y_pred, y, reduction="sum").div(
                    self.num_classes * y_pred.shape[0]
                )
                return bce
            self.classification_loss = bce_loss_fn



        # Dictionary that maps from task classes to output head to be used.
        # By default, contains a single output head that serves all classes.
        self.output_heads: Dict[str, OutputHead] = nn.ModuleDict()  # type: ignore 
        logger.info(f"output heads: {self.output_heads}")

        # Share the relevant parameters with all the auxiliary tasks.
        # We do this by setting class attributes.
        AuxiliaryTask.hidden_size   = self.hidden_size
        AuxiliaryTask.input_shape   = self.input_shape
        AuxiliaryTask.encoder       = self.encoder
        AuxiliaryTask.classifier    = self.default_output_head # TODO: Also update this class attribute when switching tasks. 
        AuxiliaryTask.preprocessing = self.preprocess_inputs
        AuxiliaryTask.device        = self.device
        
        # Dictionary of auxiliary tasks.
        self.tasks: Dict[str, AuxiliaryTask] = self.hparams.aux_tasks.create_tasks(
            input_shape=input_shape,
            hidden_size=self.hidden_size
        )

        # Dictionary of enabled auxiliary tasks.
        self.active_tasks: Dict[str, AuxiliaryTask] = {name: task for name, task in self.tasks.items() if task.enabled}

        if self.config.debug and self.config.verbose:
            logger.debug(self)
            logger.debug("Auxiliary tasks:")
            for task_name, task in self.tasks.items():
                logger.debug(f"{task.name}: {task.coefficient}")


        self.optimizer, self.lr_scheduler = self.configure_optimizers() #self.hparams.optimizer(lr=self.hparams.learning_rate, params=self.parameters())
        # if torch.cuda.device_count() > 1:
        #         print("Let's use", torch.cuda.device_count(), "GPUs!")
        #         self.encoder = nn.DataParallel(self.encoder)
        #         self.default_output_head = nn.DataParallel(self.default_output_head)
        self.to(self.device)

    def to(self, device: Optional[Union[int, torch.device, Tuple[torch.device]]], *args, **kwargs):
        if isinstance(device, tuple):    
            self.encoder = self.encoder.to(self.in_device)    
            self.default_output_head = self.default_output_head.to(self.out_device)
        else:
            # if torch.cuda.device_count() > 1:
            #     print("Let's use", torch.cuda.device_count(), "GPUs!")
            #     #from utils.utils import convert_model
            #     #self.encoder = convert_model(self.encoder)       
            #     self.encoder = nn.DataParallel(self.encoder)
            #     self.default_output_head = nn.DataParallel(self.default_output_head)
                #return
            
            for name, task in self.active_tasks.items():
                if torch.cuda.device_count() > 1:
                    print("Let's use", torch.cuda.device_count(), "GPUs!")
                    #from utils.utils import convert_model
                    #self.encoder = convert_model(self.encoder)       
                    taskr = nn.DataParallel(task)
                    #self.default_output_head = nn.DataParallel(self.default_output_head)

            super().to(device, *args, **kwargs)

    def configure_optimizers(self):
        if self.hparams.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(), self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == 'lars':
            optimizer = LARS(
                self.parameters(), lr=self.hparams.learning_rate, momentum=self.hparams.momentum,
                weight_decay=self.hparams.weight_decay, eta=self.hparams.lars_eta)
        
        elif self.hparams.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.hparams.learning_rate, momentum=self.hparams.momentum,
                weight_decay=self.hparams.weight_decay)      
        else:
            raise ValueError(f'Invalid optimizer: {self.optimizer}')
        #step_size=self.hparams.lr_sched_step, gamma=self.hparams.lr_sched_gamma)
        return optimizer, None


    def supervised_loss(self, x: Tensor,
                              y: Tensor,
                              h_x: Tensor=None, 
                              y_pred: Tensor=None) -> LossInfo:
                              
        
        if y_pred is None:
            y = self.rescale_target(y)
            x, y, _ = self.preprocess_inputs(x, y)

        #start = timer()  
        h_x = self.encode(x) if h_x is None else h_x
        y_pred = self.logits(h_x) if y_pred is None else y_pred
        y = y.view(-1)
        #end = timer()
        #print("metrics ", end - start)

        loss_f = self.classification_loss
        #input mixup on labeled samples
        if self.hparams.mixup_sup_alpha:
            x_mixed, loss_f = sup_mixup(x,y, self.hparams.mixup_sup_alpha)
        loss = loss_f(y_pred, y)
        #assert x.shape[0]==h_x.shape[0]==y_pred.shape[0]==y.shape[0]
        metrics = get_metrics(x=x, h_x=h_x, y_pred=y_pred, y=y)
        
        loss_info = LossInfo(
            name=Tasks.SUPERVISED,
            total_loss=loss,
            tensors=(dict(x=x, h_x=h_x, y_pred=y_pred, y=y)),
        )
        loss_info.metrics[Tasks.SUPERVISED] = metrics
        return loss_info

        
    def get_loss(self, x: Union[Tensor, Dict[str, Tensor]], y: Tensor=None, name: str=""):
        if isinstance(x, dict):
            # TODO: Select which 'augmented' input to use.
            pass
        if y is not None and y.shape[0] != x.shape[0]:
            raise RuntimeError("Whole batch can either be fully labeled or "
                               "fully unlabeled, but not a mix of both (for now)")
        y = self.rescale_target(y)
        if self.hparams.l_gradient_penalty > 0:
            x.requires_grad_(True)

        y_pred = None
        y_aug = None
        h_x = None
        x_aug = x
        if not self.hparams.entangle_sup:
            #we do an extra forward pass to calculate supervised loss (othervise it is calculated on the augmented representations)
            h_x = self.encode(x)
            y_pred = self.logits(h_x)
            if isinstance(y_pred, tuple):
                y_pred = y_pred[1]
        
        total_loss = LossInfo(name)
        total_loss.total_loss = torch.zeros(1, device=self.out_device)       

        for task_name, aux_task in self.active_tasks.items():
            #if aux_task.enabled:
            x, aux_task_loss, h_x = aux_task.get_scaled_loss(x, h_x=h_x, y_pred=None, y=y) #), device=self.config.device)
            total_loss += aux_task_loss
        if self.training:
            #h_x [2, bs, 515], x [bs, 2, c, h, w]
            x, y, h_x = self.preprocess_inputs(x, y, h_x)
        else:
            x, _, h_x = self.preprocess_inputs(x, None, h_x)

        # TODO: [improvement] Support a mix of labeled / unlabeled data at the example-level.
        if y is not None:
            if h_x is None and y_pred is None:
                h_x = self.encode(x)
            
            #select indicies of labeled samples
            indx_y = ((y>=0).nonzero()).view(-1)
            #selet labeled samples and their representaitons
            y = y[indx_y]
            x = x[indx_y]
            y_target = y
            h_x = h_x[indx_y]

            if y_pred is None:
                #supervised loss computed on top of semi-representations
                y_pred = self.logits(h_x)
                if isinstance(y_pred, tuple):
                    y_pred = y_pred[1]
            else:   
                y_pred=y_pred[indx_y]
                
            supervised_loss = self.supervised_loss(x=x, y=y_target, h_x=h_x, y_pred=y_pred)
            total_loss.tensors["x"] = x.detach().cpu()
            total_loss.tensors["h_x"] = h_x.detach().cpu()
            total_loss.tensors["y_pred"] = y_pred.detach().cpu()
            total_loss += supervised_loss

        
        if self.hparams.l_gradient_penalty>0 and self.encoder.training and not self.hparams.detach_classifier:
            grad_penalty = LossInfo(name= 'grad_penalty')
            if isinstance(x_aug, tuple):
                x_aug = x_aug[0]
            #x_aug = x_aug[indx_y]
            grad_penalty.total_loss = self.hparams.l_gradient_penalty * self.calc_gradient_penalty(x_aug, y_pred)
            total_loss += grad_penalty
        
        x.requires_grad_(False)

        if self.config.debug and self.config.verbose:
            for name, loss in total_loss.losses.items():
                logger.debug(name, loss.total_loss, loss.metrics)
        
        #update the prototypes fi we are using DUQ
        if isinstance(self.classifier,  OutputHead_DUQ):
            if isinstance(x_aug, tuple):
                x_aug_1 = x_aug[0][indx_y]
                x_aug_2 = x_aug[1][indx_y]
                x_aug = torch.cat([x_aug_1,x_aug_2], dim=0)
                if x_aug.shape[0]!=y_target.shape[0]:
                    y_target = torch.cat([y_target,y_target], dim=0)
            else: 
                x_aug = x_aug[indx_y]

                
            assert x_aug.shape[0]==y_target.shape[0]
            self.classifier.prepare_embedings_update(x_aug, y_target)
        return total_loss
    
    def calc_gradient_penalty(self, x, y_pred):

        def calc_gradients_input(x, y_pred):
            gradients = torch.autograd.grad(
                outputs=y_pred,
                inputs=x,
                grad_outputs=torch.ones_like(y_pred),
                create_graph=True,
            )[0]

            gradients = gradients.flatten(start_dim=1)

            return gradients
        gradients = calc_gradients_input(x, y_pred)

        # L2 norm
        grad_norm = gradients.norm(2, dim=1)

        # Two sided penalty
        gradient_penalty = ((grad_norm - 1) ** 2).mean()

        return gradient_penalty

    def encode(self, x: Tensor):
        x, _, _ = self.preprocess_inputs(x, None)
        return self.encoder(x)
    
    def rescale_target(self,y:Tensor):
        if y is not None and self.hparams.multihead:
            # y_unique are the (sorted) unique values found within the batch.
            # idx[i] holds the index of the value at y[i] in y_unique, s.t. for
            # all i in range(0, len(y)) --> y[i] == y_unique[idx[i]]
            y_unique, idx = y.unique(sorted=True, return_inverse=True)
            #take out negtive labels as those correspond to unlabeled samples
            y_unique = y_unique[(y_unique>=0).nonzero().view(-1)]
            # TODO: Could maybe decide which output head to use depending on the labels
            # (perhaps like the "labels trick" from https://arxiv.org/abs/1803.10123)
            if not (set(y_unique.tolist()) <= set(self.current_task.classes)):
                raise RuntimeError(
                    f"There are labels in the batch that aren't part of the "
                    f"current task! \n(Current task: {self.current_task}, "
                    f"batch labels: {y_unique})"
                )

            # NOTE: No need to do this when in the default task (all classes).
            if self.current_task != self._default_task:
                # if we are in the default task, no need to do this.
                # Re-label the given batch so the losses/metrics work correctly.
                new_y = copy.copy(y) # torch.empty_like(y)
                for i, label in enumerate(self.current_task.classes):
                    new_y[y == label] = i
                y = new_y
        return y

        
    def preprocess_inputs(self, x: Tensor, y: Tensor=None, h_x: Tensor=None) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """Preprocess the input tensor x before it is passed to the encoder.
        
        By default this does nothing. When subclassing the Classifier or 
        switching datasets, you might want to change this behaviour.

        Parameters
        ----------
        - x : Tensor
        
            a batch of inputs.
        
        Returns
        -------
        Tensor
            The preprocessed inputs.
        """
        #Process 'x'
        sp = x.shape
        if len(sp) ==5:
            bs, ncrops, c, h, w = x.size()
            x1, x2 = x.transpose(0,1)
            x = torch.cat([x1, x2], dim=0)
        elif len(sp) ==4:
            bs, c, h, w = x.size()
        else:
            raise Exception(f'Data shape {x.shape} not suported')
        
        if x.shape[1:] != self.input_shape:
            x = fix_channels(x)


        if y is not None:
            # if len(y)==2:   
            #     y = torch.stack(y)
            if x.shape[0] > y.shape[0]:
                #y = torch.stack([y,y]).unsqueeze(dim=2).transpose(0,1)
                #y = y.reshape(-1)
                y = torch.cat([y,y], dim=0)
            assert x.shape[0] == y.shape[0]

        if h_x is not None:
            h_shape = h_x.shape
            # if len(h_shape)==2:   
            #     pass
            if len(h_shape)==3: #[2, bs, 512]
                h1, h2 = h_x
                h_x = torch.cat([h1, h2], dim=0)
            

        return x, y, h_x

    def on_task_switch(self, task: Task, **kwargs) -> None:
        """Indicates to the model that it is working on a new task.

        Args:
            task_classes (Tuple[int, ...]): Tuple of integers, indicates the classes that are currently trained on.
        """
        # Setting the current_task attribute also creates the output head if needed.
        self.current_task = task
        # also inform the auxiliary tasks that the task switched.
        for name, aux_task in self.active_tasks.items():
            aux_task.on_task_switch(task, **kwargs)
        self.current_task = task

    def get_output_head(self, task: Task) -> OutputHead:
        """ Returns the output head for a given task.
        """
        if self.hparams.multihead:
            return self.output_heads[task.dumps()]
        else:
            return self.default_output_head

    
    def model_pretrained_unique_attributes(self) -> List:
        #returnsa list of unique atributes that identify a model if using pretraining (apply e.g. md5 on these attributes)
        def get_active_self_sup_tasks()-> Dict[str, float]:
            res = {}
            for task_name, aux_task in self.tasks.items():
                if aux_task.coefficient > 0 and task_name in ['vae', 'simclr','ae','rotation', 'jigsaw', 'brightness']:
                    res[task_name]=aux_task.coefficient
            return res
        active_self_sup_tasks = get_active_self_sup_tasks()
        return [active_self_sup_tasks, self.encoder]

    @property
    def classifier(self) -> OutputHead:
        if self.hparams.multihead:
            return self.get_output_head(self.current_task)
        # Return the default output head.
        return self.default_output_head

    @property
    def current_task(self) -> Task:
        return self._current_task

    def copy_tasks(self):
        import copy 
        copy_tasks = {}
        for name, task in self.active_tasks.items():
            copy_tasks.setdefault(name, copy.deepcopy(task))
        return copy_tasks

    @current_task.setter
    def current_task(self, task: Task):
        """ Sets the current task.
        
        Used to create output heads when using a multihead model.
        """
        assert isinstance(task, Task), f"Please set the current_task by passing a `Task` object."
        self._current_task = task
        
        if not self.hparams.multihead:
            # not a multihead model, so we just return.
            logger.debug(f"just returning, since we're not a multihead model.")
            return

        task_str = task.dumps()
        if task_str not in self.output_heads:
            # If there isn't an output head for this task
            logger.debug(f"Creating a new output head for task {task}.")
            new_output_head = LinearOutputHead(
                input_size=self.hidden_size,
                output_size=len(task.classes), 
                hparams=self.output_head,
            ).to(self.out_device)
            if torch.cuda.device_count() > 1:
                new_output_head = nn.DataParallel(new_output_head)

            # Store this new head in the module dict and add params to optimizer.
            self.output_heads[task_str] = new_output_head
            self.optimizer.add_param_group({"params": new_output_head.parameters()})

            task_head = new_output_head
        else:
            task_head = self.get_output_head(task)

        # Update the classifier used by auxiliary tasks:
        AuxiliaryTask.classifier = task_head

    def logits(self, h_x: Tensor, task:Task = None) -> Tensor:
        if self.hparams.detach_classifier:
            h_x = h_x.detach()
        if task==None:
            return self.classifier(h_x)#.to(h_x.device)(h_x)
        else:
            return self.get_output_head(task).to(h_x.device)(h_x)
    
    def load_state_dict(self, state_dict: Dict[str, Tensor], strict: bool=False) -> Tuple[List[str], List[str]]:
        starting_task = self.current_task
        # Set the task ID attribute to create all the needed output heads. 
        for key in state_dict:
            if key.startswith("output_heads"):
                task_json_str = key.split(".")[1]
                task = Task.loads(task_json_str)
                # Set the task ID attribute to create all the needed output heads.
                #self.current_task = task
                self.on_task_switch(task)

        # Reset the task_id to the starting value.
        self.current_task = starting_task
        missing, unexpected = super().load_state_dict(state_dict, strict)
        # TODO: Make sure the mean-encoder and mean-output-head modules are loaded property when using Mixup.
        return missing, unexpected
    
    def optimizer_step(self, global_step: int, **kwargs) -> None:
        """Updates the model by calling `self.optimizer.step()`.
        Additionally, also informs the auxiliary tasks that the model got
        updated.
        """
        self.optimizer.step()
        #try:
        #    self.lr_scheduler.step()
        #except:
        #    pass
        for name, task in self.tasks.items():
            if task.enabled:
                task.on_model_changed(global_step=global_step, **kwargs)
        if isinstance(self.classifier,  OutputHead_DUQ):
             with torch.no_grad():
                was_training = self.training
                self.eval()
                self.classifier.update_embeddings(self.encoder)
                if was_training:
                    self.train()
    
    def reset_encoder_parameters(self):
        logger.info('Reinitializing the encoder')
        def weight_reset(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()
        self.encoder.apply(weight_reset)
        for name, task in self.tasks.items():
            if task.enabled:
                task.reinit()


        
