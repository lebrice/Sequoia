from typing import Any, Callable, List, Tuple
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from common.layers import Flatten
from common.losses import LossInfo
from common.metrics import get_metrics, ClassificationMetrics

from tasks.auxiliary_task import AuxiliaryTask
from functools import wraps
from abc import abstractmethod
from torchvision.transforms import functional as TF


def wrap_pil_transform(function: Callable):
    def _transform(img_x, arg):
        x = TF.to_pil_image(img_x.cpu())
        x = function(x, arg)
        return TF.to_tensor(x).view(img_x.shape).to(img_x)

    @wraps(function)
    def _pil_transform(x: Tensor, arg: Any):
       return torch.cat([_transform(x_i, arg) for x_i in x]).view(x.shape)
    return _pil_transform


class TransformationBasedTask(AuxiliaryTask):
    """
    Generates an AuxiliaryTask for an arbitrary transformation function.

    Tries to classify or regress which argument was passed to the function,
    given only the transformed code, if `compare_with_original` is False, else
    given the original and transformed codes. 
    """

    @dataclass
    class Options(AuxiliaryTask.Options):
        """Command-line options for the Transformation-based auxiliary task."""
        # Wether or not both the original and transformed codes should be passed
        # to the auxiliary layer in order to detect the transformation. 
        compare_with_original: bool = True

    def __init__(self,
                 function: Callable[[Tensor, Any], Tensor],
                 function_args: List[Any],
                 loss: Callable,
                 auxiliary_layer: nn.Module=None,
                 options: Options=None):
        super().__init__(options=options)
        self.function = function
        self.name = self.function.__name__
        self.function_args = function_args
        self.alphas: Tensor = torch.Tensor(self.function_args)
        self.options = options or self.Options()
        self.nargs = len(self.function_args)
        # which loss to use. CrossEntropy when classifying, or MSE when regressing.
        self.loss = loss

        if auxiliary_layer is not None:
            self.auxiliary_layer = auxiliary_layer
        else:
            input_dims = AuxiliaryTask.hidden_size
            if self.options.compare_with_original:
                input_dims *= 2
            self.auxiliary_layer = nn.Sequential(
                Flatten(),
                nn.Linear(input_dims, self.nargs),
            )

    # def get_loss(self, x: Tensor, h_x: Tensor, y_pred: Tensor, y: Tensor=None) -> LossInfo:
    #     loss_info = LossInfo()
    #     batch_size = x.shape[0]
    #     for fn_arg, alpha in zip(self.function_args, self.alphas):
    #         loss_i = self.get_loss_for_arg(x=x, h_x=h_x, fn_arg=fn_arg, alpha=alpha)
    #         loss_info.losses[f"{self.name}_{fn_arg}"] = loss_i.total_loss
    #         loss_info += loss_i
    #         # print(f"{self.name}_{fn_arg}", loss_i.metrics) 
    #     return loss_info

    def get_loss_for_arg(self, x: Tensor, h_x: Tensor, fn_arg: Any, alpha: Tensor) -> LossInfo:
        loss_info = LossInfo()
        alpha = alpha.to(x.device)

        # Transform X using the function:
        x_t = self.function(x, fn_arg)
        # Get the code for the transformed x:
        h_x_t = self.encode(x_t)

        aux_layer_input = h_x_t
        if self.options.compare_with_original:
            aux_layer_input = torch.cat([h_x, h_x_t], dim=-1)

        alpha_t = self.auxiliary_layer(aux_layer_input)
    
        # Save some tensors for debugging purposes:
        loss_info.tensors["x_t"] = x_t
        loss_info.tensors["h_x_t"] = h_x_t
        loss_info.tensors["alpha_t"] = alpha_t

        # get the metrics for this particular argument (accuracy, mse, etc.)
        loss_info.metrics = get_metrics(x=x_t, h_x=h_x_t, y_pred=alpha_t, y=alpha)

        loss_info.total_loss = self.loss(alpha_t, alpha)
        # print(self.name, loss_info.metrics)
        return loss_info


class ClassifyTransformationTask(TransformationBasedTask):
    """
    Generates an AuxiliaryTask for an arbitrary transformation function.

    Tries to classify which argument was passed to the function.
    """
    def __init__(self,
                 function: Callable[[Tensor, Any], Tensor],
                 function_args: List[Any],
                 options: TransformationBasedTask.Options=None):
        super().__init__(function=function,
                         function_args=function_args,
                         loss=nn.CrossEntropyLoss(),
                         options=options)
    
    def get_loss(self, x: Tensor, h_x: Tensor, y_pred: Tensor=None, y: Tensor=None) -> LossInfo:
        # Alpha is the classification target.
        # It indicates which transformation argument was used. 
        # I.e. a vector of 0's for function_args[0], 1's for function_args[1], etc.
        batch_size = x.shape[0]
        ones = torch.ones(batch_size, dtype=torch.long)
        self.alphas = [ones * i for i in range(self.nargs)]
        loss_info = LossInfo()
        batch_size = x.shape[0]
        for fn_arg, alpha in zip(self.function_args, self.alphas):
            loss_i = self.get_loss_for_arg(x=x, h_x=h_x, fn_arg=fn_arg, alpha=alpha)
            loss_info.losses[f"{self.name}_{fn_arg}"] = loss_i.total_loss
            loss_info += loss_i
            # print(f"{self.name}_{fn_arg}", loss_i.metrics) 
        return loss_info


class RegressTransformationTask(TransformationBasedTask):
    """
    Generates an AuxiliaryTask for an arbitrary transformation function.

    Tries to Regress which argument value was passed to the function.

    Can either use a list of function arguments, or a range from which to sample
    the argument values uniformly. 
    """
    def __init__(self,
                 function: Callable[[Tensor, Any], Tensor],
                 function_args: List[Any]=None,
                 function_arg_range: Tuple[float, float]=None,
                 n_calls: int = 2,
                 options: TransformationBasedTask.Options=None):
        super().__init__(
            function=function,
            function_args=function_args or [],
            loss=torch.dist,
            options=options,
        )
        assert function_args or function_arg_range, "One of function_args or function_arg_range must be set."
        self.function_arg_range = function_arg_range
        self.n_calls = n_calls
        
        if self.function_arg_range:
            self.min_arg = self.function_arg_range[0]
            self.max_arg = self.function_arg_range[1]
            self.arg_mean = (self.min_arg + self.max_arg) / 2
            self.arg_range = self.max_arg - self.min_arg

    def get_alphas(self, batch_size: int) -> Tensor:
        alphas: Tensor
        if self.function_args is not None and len(self.function_args) != 0:
            if isinstance(self.function_args, Tensor):
                return self.function_args
            return torch.Tensor(self.function_args)
        else:
            # sample a random argument in the range [self.min_arg, self.max_arg]
            alphas = torch.rand([self.n_calls]) * (self.max_arg - self.min_arg)
            alphas += self.min_arg
            return alphas
            
    def get_loss(self, x: Tensor, h_x: Tensor, y_pred: Tensor=None, y: Tensor=None) -> LossInfo:
        alphas = self.get_alphas(x.shape[0])
        loss_info = LossInfo()
        batch_size = x.shape[0]
        for alpha in alphas:
            fn_arg = alpha
            loss_i = self.get_loss_for_arg(x=x, h_x=h_x, fn_arg=fn_arg, alpha=alpha)
            loss_info.losses[f"{self.name}_{fn_arg}"] = loss_i.total_loss
            loss_info += loss_i
            # print(f"{self.name}_{fn_arg}", loss_i.metrics) 
        return loss_info
