import math
import itertools  
import copy 
from collections import defaultdict
from utils.logging_utils import get_logger
from collections import OrderedDict
from dataclasses import InitVar, asdict, dataclass
from pathlib import Path
from typing import (Any, Dict, Iterable, List, Optional, Set, Tuple, TypeVar,
                    Union)

import torch
from torch import nn
from torch import Tensor

from simple_parsing import field, mutable_field
from utils.json_utils import Serializable
from utils.logging_utils import cleanup, get_logger
from utils.utils import add_dicts, add_prefix

from .metrics import (ClassificationMetrics, Metrics, RegressionMetrics,
                      get_metrics, AUCMetric)

logger = get_logger(__file__)

def cosine_similarity_distil_loss(self, output_net, target_net, eps=0.0000001):
        # Normalize each vector by its norm
        output_net_norm = torch.sqrt(torch.sum(output_net ** 2, dim=1, keepdim=True))
        output_net = output_net / (output_net_norm + eps)
        output_net[output_net != output_net] = 0

        target_net_norm = torch.sqrt(torch.sum(target_net ** 2, dim=1, keepdim=True))
        target_net = target_net / (target_net_norm + eps)
        target_net[target_net != target_net] = 0

        # Calculate the cosine similarity
        model_similarity = torch.mm(output_net, output_net.transpose(0, 1))
        target_similarity = torch.mm(target_net, target_net.transpose(0, 1))

        # Scale cosine similarity to 0..1
        model_similarity = (model_similarity + 1.0) / 2.0
        target_similarity = (target_similarity + 1.0) / 2.0

        # Transform them into probabilities
        model_similarity = model_similarity / torch.sum(model_similarity, dim=1, keepdim=True)
        target_similarity = target_similarity / torch.sum(target_similarity, dim=1, keepdim=True)

        # Calculate the KL-divergence
        loss = torch.mean(target_similarity * torch.log((target_similarity + eps) / (model_similarity + eps)))

        return loss


@dataclass
class LossInfo(Serializable):
    """ Simple object to store the losses and metrics for a given task. 
    
    Used to simplify the return type of the various `get_loss` functions.    
    """
    name: str = ""
    coefficient: Union[float, Tensor] = 1.0
    total_loss: Tensor = 0.  # type: ignore
    losses:  Dict[str, "LossInfo"] = mutable_field(OrderedDict)
    tensors: Dict[str, Tensor]     = mutable_field(OrderedDict, repr=False, to_dict=False)
    metrics: Dict[str, Metrics]    = mutable_field(OrderedDict)

    x:      InitVar[Optional[Tensor]] = None
    h_x:    InitVar[Optional[Tensor]] = None
    y_pred: InitVar[Optional[Tensor]] = None
    y:      InitVar[Optional[Tensor]] = None

    def __post_init__(self, x: Tensor=None, h_x: Tensor=None, y_pred: Tensor=None, y: Tensor=None):
        if self.name and self.name not in self.metrics:
            if y_pred is not None and y is not None:
                self.metrics[self.name] = get_metrics(y_pred=y_pred, y=y)
        for name, tensor in self.tensors.items():
            if not isinstance(tensor, Tensor):
                tensor = torch.as_tensor(tensor)
            if tensor.requires_grad:
                self.tensors[name] = tensor.detach()
        if isinstance(self.total_loss, list):
            self.total_loss = torch.as_tensor(self.total_loss)
        
        for name, loss in self.losses.items():
            if isinstance(loss, dict):
                self.losses[name] = LossInfo.from_dict(loss)

    @property
    def metric(self) -> Optional[Metrics]:
        """Shortcut for `self.metrics[self.name]`.

        Returns:
            Optional[Metrics]: The metrics associated with this LossInfo.
        """
        return self.metrics.get(self.name)

    @property
    def accuracy(self) -> float:
        assert isinstance(self.metric, ClassificationMetrics)
        return self.metric.accuracy
    
    @property
    def mse(self) -> Tensor:
        assert isinstance(self.metric, RegressionMetrics)
        return self.metric.mse

    def __add__(self, other: "LossInfo") -> "LossInfo":
        """Adds two LossInfo instances together.
        
        Adds the losses, total loss and metrics. Overwrites the tensors.
        Keeps the name of the first one. This is useful when doing something
        like:
        
        ```
        total_loss = LossInfo("Test")
        for x, y in dataloader:
            total_loss += model.get_loss(x=x, y=y)
        ```      
        
        Returns
        -------
        LossInfo
            The merged/summed up LossInfo.
        """
        name = self.name
        total_loss = self.total_loss + other.total_loss
        
        if self.name == other.name:
            losses  = add_dicts(self.losses, other.losses)
            metrics = add_dicts(self.metrics, other.metrics)
        else:
            # IDEA: when the names don't match, store the entire LossInfo
            # object into the 'losses' dict, rather than a single loss tensor.
            losses = add_dicts(self.losses, {other.name: other})
            # TODO: setting in the 'metrics' dict, we are duplicating the
            # metrics, since they now reside in the `self.metrics[other.name]`
            # and `self.losses[other.name].metrics` attributes.
            metrics = self.metrics
            # metrics = add_dicts(self.metrics, {other.name: other.metrics})
        
        tensors = add_dicts(self.tensors, other.tensors, add_values=False)
        return LossInfo(
            name=name,
            coefficient=self.coefficient,
            total_loss=total_loss,
            losses=losses,
            tensors=tensors,
            metrics=metrics,
        )
    
    def __iadd__(self, other: "LossInfo") -> "LossInfo":
        """Adds LossInfo to `self` in-place.
        
        Adds the losses, total loss and metrics. Overwrites the tensors.
        Keeps the name of the first one. This is useful when doing something
        like:
        
        ```
        total_loss = LossInfo("Test")
        for x, y in dataloader:
            total_loss += model.get_loss(x=x, y=y)
        ```      
        
        Returns
        -------
        LossInfo
            `self`: The merged/summed up LossInfo.
        """
        self.total_loss = self.total_loss + other.total_loss
        if self.name == other.name:
            self.losses  = add_dicts(self.losses, other.losses)
            self.metrics = add_dicts(self.metrics, other.metrics)
        else:
            # IDEA: when the names don't match, store the entire LossInfo
            # object into the 'losses' dict, rather than a single loss tensor.
            self.losses = add_dicts(self.losses, {other.name: other})
        self.tensors = add_dicts(self.tensors, other.tensors, add_values=False)
        return self

    def __mul__(self, coefficient: Union[float,Tensor]) -> "LossInfo":
        """ Scale each loss tensor by `coefficient`.

        Returns
        -------
        LossInfo
            returns a scaled LossInfo instance.
        """
        return LossInfo(
            name=self.name,
            coefficient=self.coefficient * coefficient,
            total_loss=self.total_loss * coefficient,
            losses=OrderedDict([
                (k, value * coefficient) for k, value in self.losses.items()
            ]),
            metrics=self.metrics,
            tensors=self.tensors,
        )

    @property
    def unscaled_losses(self):
        return OrderedDict([
            (k, value / self.coefficient) for k, value in self.losses.items()
        ])

    def to_log_dict(self, verbose: bool=False) -> Dict[str, Union[str, float, Dict]]:
        # TODO: Could also produce some wandb plots and stuff here
        return self.to_dict()

    def to_pbar_message(self):
        """ Smaller, less-detailed version of `self.to_log_dict()` (doesn't recurse into sublosses)
        meant to be used in progressbars.
        """
        message: Dict[str, Union[str, float]] = OrderedDict()
        message["Loss"] = float(self.total_loss.item())

        if self.metric:
            message[self.name] = self.metric.to_pbar_message()

        for name, loss_info in self.losses.items():
            message[name] = loss_info.to_pbar_message()

        prefix = (self.name + " ") if self.name else ""
        message = add_prefix(message, prefix)

        return cleanup(message, sep=" ")
    
    def to_dict(self):
        self.detach()
        self.drop_tensors()
        return super().to_dict()
    
    def drop_tensors(self) -> None:
        self.tensors.clear()
        for n, loss in self.losses.items():
            loss.drop_tensors()

    def absorb(self, other: "LossInfo") -> None:
        """Absorbs `other` into `self`, merging the losses and metrics.

        Args:
            other (LossInfo): Another loss to 'merge' into this one.
        """
        new_name = self.name
        old_name = other.name
        new_other = LossInfo(name=new_name)
        new_other.total_loss = other.total_loss
        # acumulate the metrics:
        new_other.metrics = OrderedDict([
            (k.replace(old_name, new_name), v) for k, v in other.metrics.items() 
        ])
        new_other.losses = OrderedDict([
            (k.replace(old_name, new_name), v) for k, v in other.losses.items() 
        ])
        self += new_other
    
    def all_metrics(self) -> Dict[str, Metrics]:
        result: Dict[str, Metrics] = {}
        result.update(self.metrics)
        for name, loss in self.losses.items():
            result.update(loss.all_metrics())
        if self.name:
            prefix = self.name
            if not prefix.endswith(" "):
                prefix += " "
            return add_prefix(result, prefix)
        return result


def get_supervised_metrics(loss: LossInfo, mode: str="Test") -> Union[ClassificationMetrics, RegressionMetrics]:
    from tasks.tasks import Tasks
    if Tasks.SUPERVISED not in loss.losses:
        loss = loss.losses[mode]
    metric = loss.losses[Tasks.SUPERVISED].metrics[Tasks.SUPERVISED]
    return metric


def get_supervised_accuracy(loss: LossInfo, mode: str="Test") -> float:
    # TODO: this is ugly. There is probably a cleaner way, but I can't think of it right now. 
    try:
        supervised_metric = get_supervised_metrics(loss, mode=mode)
        return supervised_metric.accuracy
    except KeyError as e:
        print(f"Couldn't find the supervised accuracy in the `LossInfo` object: Key error: {e}")
        print(loss.dumps(indent="\t", sort_keys=False))
        raise e


@dataclass
class TrainValidLosses(Serializable):
    """ Helper class to store the train and valid losses during training. """
    train_losses: Dict[int, LossInfo] = field(default_factory=OrderedDict)
    valid_losses: Dict[int, LossInfo] = field(default_factory=OrderedDict)

    def __iadd__(self, other: Union["TrainValidLosses", Tuple[Dict[int, LossInfo], Dict[int, LossInfo]]]) -> "TrainValidLosses":
        if isinstance(other, TrainValidLosses):
            self.train_losses.update(other.train_losses)
            self.valid_losses.update(other.valid_losses)
        elif isinstance(other, tuple):
            self.train_losses.update(other[0])
            self.valid_losses.update(other[1])
        else:
            return NotImplemented
        self.drop_tensors()
        return self
    
    def __setitem__(self, index: int, value: Tuple[LossInfo, LossInfo]) -> None:
        self.train_losses[index] = value[0].detach()
        self.valid_losses[index] = value[1].detach()

    def __getitem__(self, index: int) -> Tuple[LossInfo, LossInfo]:
        return (
            self.train_losses[index],
            self.valid_losses[index]
        )

    def items(self) -> Iterable[Tuple[int, Tuple[Optional[LossInfo], Optional[LossInfo]]]]:
        train_keys = set(self.train_losses).union(set(self.valid_losses))
        for k in sorted(train_keys):
            yield k, (self.train_losses.get(k), self.valid_losses.get(k))

    def all_loss_names(self) -> Set[str]:
        all_loss_names: Set[str] = set()
        for loss_info in itertools.chain(self.train_losses.values(), 
                                         self.valid_losses.values()):
            all_loss_names.update(loss_info.losses)
        return all_loss_names
    
    def latest_step(self) -> int:
        """Returns the latest global_step in the dicts."""
        return max(itertools.chain(self.train_losses, self.valid_losses), default=0)

    def keep_up_to_step(self, step: int) -> None:
        """Keeps only the losses up to step `step`.

        Args:
            step (int): the maximum step (inclusive) to keep.
        """
        for k in filter(lambda k: k > step, list(self.train_losses.keys())):
            self.train_losses.pop(k)
        for k in filter(lambda k: k > step, list(self.valid_losses.keys())):
            self.valid_losses.pop(k)

    def add_step(self, offset: int):
        """Adds the value of `offset` to all the keys in the dictionary.
        Args:
            offset (int): A value to add to all the keys.
        """
        new_train_losses: Dict[int, LossInfo] = OrderedDict()
        new_valid_losses: Dict[int, LossInfo] = OrderedDict()
        for k in list(self.train_losses.keys()):
            new_train_losses[k + offset] = self.train_losses.pop(k)
        for k in list(self.valid_losses.keys()):
            new_valid_losses[k + offset] = self.valid_losses.pop(k)
        self.train_losses = new_train_losses
        self.valid_losses = new_valid_losses

    def drop_tensors(self) -> None:
        for l in self.train_losses.values():
            l.drop_tensors()
        for l in self.valid_losses.values():
            l.drop_tensors()

# @encode.register
# def encode_losses(obj: TrainValidLosses) -> Dict:
#     train_losses_dict = OrderedDict((k, encode(v)) for k, v in obj.train_losses.items())
#     valid_losses_dict = OrderedDict((k, encode(v)) for k, v in obj.valid_losses.items())
#     return {
#         "train_losses": train_losses_dict,
#         "valid_losses": valid_losses_dict,
#     }

@dataclass
class AUC_Meter(Serializable):
    """
    Class to keep runing statistics about performance, such as AUC.
    """ 
    online_metric_dict: Dict[str, Metrics] = field(default_factory=dict)  
              
    def update(self, new_element: LossInfo) -> LossInfo:
        all_metrics = new_element.all_metrics()
        self.online_metric_dict = add_dicts(self.online_metric_dict, all_metrics)
        res = {}
        for k, v in all_metrics.items():
            if isinstance(v, ClassificationMetrics):
                res[f'AUC/{k}'] = self.online_metric_dict[k].accuracy

        new_element.metrics.update(res)
        return new_element

eps = 1e-7

class PKT(nn.Module):
    """Probabilistic Knowledge Transfer for deep representation learning
    Code from author: https://github.com/passalis/probabilistic_kt"""
    def __init__(self, *args, **kwargs):
        super(PKT, self).__init__()

    def forward(self, f_s, f_t):
        return self.cosine_similarity_loss(f_s, f_t)

    @staticmethod
    def cosine_similarity_loss(output_net, target_net, eps=0.0000001):
        # Normalize each vector by its norm
        # output_net_norm = torch.sqrt(torch.sum(output_net ** 2, dim=1, keepdim=True))
        # output_net = output_net / (output_net_norm + eps)
        # output_net[output_net != output_net] = 0

        # target_net_norm = torch.sqrt(torch.sum(target_net ** 2, dim=1, keepdim=True))
        # target_net = target_net / (target_net_norm + eps)
        # target_net[target_net != target_net] = 0

        # Calculate the cosine similarity
        model_similarity = torch.mm(output_net, output_net.transpose(0, 1))
        target_similarity = torch.mm(target_net, target_net.transpose(0, 1))

        # Scale cosine similarity to 0..1
        model_similarity = (model_similarity + 1.0) / 2.0
        target_similarity = (target_similarity + 1.0) / 2.0

        # Transform them into probabilities
        model_similarity = model_similarity / torch.sum(model_similarity, dim=1, keepdim=True)
        target_similarity = target_similarity / torch.sum(target_similarity, dim=1, keepdim=True)

        # Calculate the KL-divergence
        loss = torch.mean(target_similarity * torch.log((target_similarity + eps) / (model_similarity + eps)))

        return loss


class CRDLoss(nn.Module):
    """CRD Loss function
    includes two symmetric parts:
    (a) using teacher as anchor, choose positive and negatives over the student side
    (b) using student as anchor, choose positive and negatives over the teacher side
    Args:
        opt.s_dim: the dimension of student's feature
        opt.t_dim: the dimension of teacher's feature
        opt.feat_dim: the dimension of the projection space
        opt.nce_k: number of negatives paired with each positive
        opt.nce_t: the temperature
        opt.nce_m: the momentum for updating the memory buffer
        opt.n_data: the number of samples in the training set, therefor the memory buffer is: opt.n_data x opt.feat_dim
    """
    def __init__(self, s_dim, t_dim, feat_dim, nce_k, nce_t, nce_m, n_data):
        super(CRDLoss, self).__init__()
        self.embed_s = Embed(s_dim, feat_dim)
        self.embed_t = Embed(t_dim, feat_dim)
        self.contrast = ContrastMemory(feat_dim, n_data, nce_k, nce_t, nce_m)
        self.criterion_t = ContrastLoss(n_data)
        self.criterion_s = ContrastLoss(n_data)

    def forward(self, f_s, f_t, idx, contrast_idx=None):
        """
        Args:
            f_s: the feature of student network, size [batch_size, s_dim]
            f_t: the feature of teacher network, size [batch_size, t_dim]
            idx: the indices of these positive samples in the dataset, size [batch_size]
            contrast_idx: the indices of negative samples, size [batch_size, nce_k]
        Returns:
            The contrastive loss
        """
        f_s = self.embed_s(f_s)
        f_t = self.embed_t(f_t)
        out_s, out_t = self.contrast(f_s, f_t, idx, contrast_idx)
        s_loss = self.criterion_s(out_s)
        t_loss = self.criterion_t(out_t)
        loss = s_loss + t_loss
        return loss


class ContrastLoss(nn.Module):
    """
    contrastive loss, corresponding to Eq (18)
    """
    def __init__(self, n_data):
        super(ContrastLoss, self).__init__()
        self.n_data = n_data

    def forward(self, x):
        bsz = x.shape[0]
        m = x.size(1) - 1

        # noise distribution
        Pn = 1 / float(self.n_data)

        # loss for positive pair
        P_pos = x.select(1, 0)
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()

        # loss for K negative pair
        P_neg = x.narrow(1, 1, m)
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_()

        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz

        return loss


class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x


class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class ContrastMemory(nn.Module):
    """
    memory buffer that supplies large amount of negative samples.
    """
    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5):
        super(ContrastMemory, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.K = K

        self.register_buffer('params', torch.tensor([K, T, -1, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory_v1', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_v2', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, v1, v2, y, idx=None):
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z_v1 = self.params[2].item()
        Z_v2 = self.params[3].item()

        momentum = self.params[4].item()
        batchSize = v1.size(0)
        outputSize = self.memory_v1.size(0)
        inputSize = self.memory_v1.size(1)

        # original score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)
        # sample
        weight_v1 = torch.index_select(self.memory_v1, 0, idx.view(-1)).detach()
        weight_v1 = weight_v1.view(batchSize, K + 1, inputSize)
        out_v2 = torch.bmm(weight_v1, v2.view(batchSize, inputSize, 1))
        out_v2 = torch.exp(torch.div(out_v2, T))
        # sample
        weight_v2 = torch.index_select(self.memory_v2, 0, idx.view(-1)).detach()
        weight_v2 = weight_v2.view(batchSize, K + 1, inputSize)
        out_v1 = torch.bmm(weight_v2, v1.view(batchSize, inputSize, 1))
        out_v1 = torch.exp(torch.div(out_v1, T))

        # set Z if haven't been set yet
        if Z_v1 < 0:
            self.params[2] = out_v1.mean() * outputSize
            Z_v1 = self.params[2].clone().detach().item()
            print("normalization constant Z_v1 is set to {:.1f}".format(Z_v1))
        if Z_v2 < 0:
            self.params[3] = out_v2.mean() * outputSize
            Z_v2 = self.params[3].clone().detach().item()
            print("normalization constant Z_v2 is set to {:.1f}".format(Z_v2))

        # compute out_v1, out_v2
        out_v1 = torch.div(out_v1, Z_v1).contiguous()
        out_v2 = torch.div(out_v2, Z_v2).contiguous()

        # update memory
        with torch.no_grad():
            l_pos = torch.index_select(self.memory_v1, 0, y.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(v1, 1 - momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v1 = l_pos.div(l_norm)
            self.memory_v1.index_copy_(0, y, updated_v1)

            ab_pos = torch.index_select(self.memory_v2, 0, y.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(v2, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v2 = ab_pos.div(ab_norm)
            self.memory_v2.index_copy_(0, y, updated_v2)

        return out_v1, out_v2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class AliasMethod(object):
    """
    From: https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    """
    def __init__(self, probs):

        if probs.sum() > 1:
            probs.div_(probs.sum())
        K = len(probs)
        self.prob = torch.zeros(K)
        self.alias = torch.LongTensor([0]*K)

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.prob[kk] = K*prob
            if self.prob[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self.alias[small] = large
            self.prob[large] = (self.prob[large] - 1.0) + self.prob[small]

            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller+larger:
            self.prob[last_one] = 1

    def cuda(self):
        self.prob = self.prob.to(device)
        self.alias = self.alias.to(device)

    def draw(self, N):
        """ Draw N samples from multinomial """
        K = self.alias.size(0)

        kk = torch.zeros(N, dtype=torch.long, device=self.prob.device).random_(0, K)
        prob = self.prob.index_select(0, kk)
        alias = self.alias.index_select(0, kk)
        # b is whether a random number is greater than q
        b = torch.bernoulli(prob)
        oq = kk.mul(b.long())
        oj = alias.mul((1-b).long())

        return oq + oj