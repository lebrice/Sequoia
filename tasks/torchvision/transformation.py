import torch
import torchvision
from torch import nn, Tensor
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from typing import Dict, Set, Callable, Any, List, Tuple
from tasks.bases import AuxiliaryTask



class ClassifyTransformationTask(AuxiliaryTask):
    """
    Generates an AuxiliaryTask for an arbitrary transformation function.

    Tries to classify which argument was passed to the function.
    """
    def __init__(self,
                 function: Callable[[Tensor, Any], Tensor],
                 function_args: List[Any],
                 options: AuxiliaryTask.Options):
        super().__init__(options=options)
        self.function = function
        self.function_args: List[Dict[str, Any]] = function_args

        self.loss = nn.CrossEntropyLoss()

        self.nargs = len(self.function_args)
        self.choose_transformation = nn.Linear(self.hidden_size, self.nargs)


    def pil_transform(self, function: Callable):
        def _transform(x, arg):
            x = TF.to_pil_image(x)
            x = function(x, arg)
            return TF.to_tensor(x)
        
        def _pil_transform(x: Tensor, arg: Any):
            return torch.cat([_transform(x_i, arg) for x_i in x])
        return _pil_transform



    def get_loss(self, x: Tensor, h_x: Tensor, y_pred: Tensor, y: Tensor=None) -> Tensor:
        total_loss = torch.zeros(1)
        batch_size: int = x.shape[0]
        ones = torch.ones(batch_size, dtype=torch.long)

        for i, fn_arg in enumerate(self.function_args):
            # vector of 0's for arg 0, vector of 1's for arg 1, etc.
            true_label = i * ones
            x_transformed = self.function(x, fn_arg)  # type: ignore
            h_x = self.encode(x_transformed)
            arg_pred = self.choose_transformation(h_x)
            
            t_loss = self.loss(arg_pred, true_label)
            total_loss += t_loss
        #     print(f"Loss for {self.function.__name__}(x, {fn_arg}):", t_loss.item())
        # print(f"Total loss for {self.name}: {total_loss.item()}")
        return total_loss

