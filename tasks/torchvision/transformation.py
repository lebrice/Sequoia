import torch
import torchvision
from torch import nn, Tensor
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from typing import Dict, Set, Callable, Any, List, Tuple, Optional
from tasks.bases import AuxiliaryTask
from functools import wraps

from common.losses import LossInfo
from common.layers import Flatten


def wrap_pil_transform(function: Callable):
    def _transform(img_x, arg):
        x = TF.to_pil_image(img_x)
        x = function(x, arg)
        return TF.to_tensor(x).view(img_x.shape)

    @wraps(function)
    def _pil_transform(x: Tensor, arg: Any):
       return torch.cat([_transform(x_i, arg) for x_i in x]).view(x.shape)
    return _pil_transform


class ClassifyTransformationTask(AuxiliaryTask):
    """
    Generates an AuxiliaryTask for an arbitrary transformation function.

    Tries to classify which argument was passed to the function.
    """
    def __init__(self,
                 function: Callable[[Tensor, Any], Tensor],
                 function_args: List[Any],
                 compare_with_original: bool=True,
                 options: AuxiliaryTask.Options=None):
        super().__init__(options=options)
        self.function = function
        self.function_args: List[Any] = function_args
        self.compare_with_original = compare_with_original

        self.loss = nn.CrossEntropyLoss()

        self.nargs = len(self.function_args)
        self.choose_transformation = nn.Sequential(
            Flatten(),
            nn.Linear(self.hidden_size * (2 if compare_with_original else 1), self.nargs),
            nn.Sigmoid(),
        )

    def get_loss(self, x: Tensor, h_x: Tensor, y_pred: Tensor, y: Tensor=None) -> LossInfo:
        loss_info = LossInfo()
        batch_size: int = x.shape[0]
        ones = torch.ones(batch_size, dtype=torch.long)
        for i, fn_arg in enumerate(self.function_args):
            # vector of 0's for arg 0, vector of 1's for arg 1, etc.
            true_label = i * ones
            x_t = self.function(x, fn_arg)  # type: ignore
            h_x_t = self.encode(x_t)

            arg_pred_input = h_x_t
            if self.compare_with_original:
                arg_pred_input = torch.cat([h_x, h_x_t], dim=-1)

            arg_pred = self.choose_transformation(arg_pred_input)
            
            loss_info.tensors["x_t"] = x_t
            loss_info.tensors["h_x_t"] = h_x_t
            loss_info.tensors["arg_pred"] = arg_pred
            
            t_loss = self.loss(arg_pred, true_label)
            loss_info.losses[f"{self.name}_{fn_arg}"] = t_loss
            loss_info.total_loss += t_loss
        return loss_info

    @property
    def name(self):
        return self.function.__name__

class RegressTransformationTask(AuxiliaryTask):
    """
    Generates an AuxiliaryTask for an arbitrary transformation function.

    Tries to classify which argument was passed to the function.
    """
    def __init__(self,
                 function: Callable[[Tensor, Any], Tensor],
                 function_args: List[Any]=None,
                 compare_with_original=True,
                 function_arg_range: Tuple[float, float]=None,
                 n_calls: int = 2,
                 options: AuxiliaryTask.Options=None):
        super().__init__(options=options)
        self.function = function
        self.function_args = function_args
        self.compare_with_original = compare_with_original
        self.function_arg_range = function_arg_range
        self.n_calls = n_calls
        
        if self.function_arg_range is not None:
            self.min_arg = self.function_arg_range[0]
            self.max_arg = self.function_arg_range[1]
            self.arg_mean = (self.min_arg + self.max_arg) / 2
            self.arg_range = self.max_arg - self.min_arg

        self.loss = torch.dist
        self.regress_transformation = nn.Sequential(
            nn.Linear(self.hidden_size * (2 if compare_with_original else 1), 1),
            nn.Sigmoid(),
        )

    def get_loss(self, x: Tensor, h_x: Tensor, y_pred: Tensor, y: Tensor=None) -> Tensor:
        loss_info = LossInfo()
        x = self.preprocess(x)
        batch_size: int = x.shape[0]
        ones = torch.ones(batch_size, dtype=torch.long)
        

        if self.function_arg_range is not None:
            # sample a random argument in the range [self.min_arg, self.max_arg]
            arg_values = torch.rand([self.n_calls, batch_size]) * (self.max_arg - self.min_arg)
            arg_values += self.min_arg        
        else:
            assert self.function_args is not None
            arg_values = self.function_args

        for i, arg_value in enumerate(arg_values):
            # transform x using that argument value
            x_t = self.function(x, arg_value)  # type: ignore
            h_x_t = self.encode(x_t)
            
            arg_pred_input = h_x_t
            if self.compare_with_original:
                arg_pred_input = torch.cat([h_x, h_x_t], dim=-1)

            # predict the argument that was used in the transformation function.
            arg_pred = self.regress_transformation(arg_pred_input)

            loss_info.tensors[f"{self.name}.x_t"] = x_t
            loss_info.tensors[f"{self.name}.h_x_t"] = h_x_t
            loss_info.tensors[f"{self.name}.arg_pred"] = arg_pred
            
            t_loss = self.loss(arg_pred, arg_value)
            loss_info.losses[f"{self.name}_{arg_value}"] = t_loss
            loss_info.total_loss += t_loss

        #     print(f"Loss for {self.function.__name__}(x, {fn_arg}):", t_loss.item())
        # print(f"Total loss for {self.name}: {total_loss.item()}")
        return loss_info

