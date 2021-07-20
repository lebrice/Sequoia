import torch
import numpy as np
from torch import nn
from typing import Tuple
from torch.nn.parameter import Parameter
from torchvision import transforms

import operator


def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1


class Buffer(nn.Module):
    """
    The memory buffer of rehearsal method.
    """

    def __init__(self, buffer_size, loss_priority=False, balanced=True, balance_task_labels=False ):
        super().__init__()
        self.buffer_size = buffer_size
        self.register_buffer("_num_seen_examples", torch.tensor(0).int())
        self.attributes = ["examples", "labels",
                           "logits", "task_labels", "loss_scores"]
        self.dict = {}
        self.loss_priority = loss_priority
        self.balanced = balanced
        self.balance_task_labels=balance_task_labels
        self._device = {}

        self.W = torch.exp(torch.log(torch.rand(()))/buffer_size)
        self.remainder = 0

    def add_attributes(self, new_attributes):
        """add attributes to be saved in the buffer

        Args:
            new_attributes (list): list of attribute names to be stored in the buffer
        """
        self.attributes = self.attributes[:-1]
        self.attributes += new_attributes
        self.attributes.append("loss_scores")

    @property
    def num_seen_examples(self):
        return self._num_seen_examples.item()

    def init_tensors(self, attr_dict) -> None:
        """
        Initializes just the required tensors.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        """
        for attr_str in self.attributes:
            attr = attr_dict[attr_str]
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if attr_str.endswith(
                    "els") else torch.float32
                b = torch.zeros(self.buffer_size, *attr.shape[1:], dtype=typ)
                self._device[attr_str] = attr.device
                self.register_buffer(attr_str, b)

    # def merge_scores(self):
    #     if self.loss_priority:
    #         scaling_factor = self.loss_scores.abs().mean() * self.balance_scores.abs().mean()
    #         norm_importance = self.loss_scores / scaling_factor
    #         presoftscores = 0.5 * norm_importance + 0.5 * self.balance_scores

    #         if presoftscores.max() - presoftscores.min() != 0:
    #             presoftscores = (presoftscores - presoftscores.min()) / (presoftscores.max() - presoftscores.min())
    #         self.scores = presoftscores / presoftscores.sum()
    #     else:
    #         self.scores = self.balance_scores/ self.balance_scores.sum()

    def _get_index(self):
        indices = torch.tensor([[i] for i in range(len(self.labels))])
        proba = None
        if self.balanced:
            label_max = max(self.dict.items(), key=operator.itemgetter(1))[0]
            if self.balance_task_labels:
                indices = (self.task_labels == label_max).nonzero()
            else:
                indices = (self.labels == label_max).nonzero()
        if self.loss_priority:
            scores = self.loss_scores[indices]
            proba = (scores.squeeze() / scores.sum()).numpy()

        index = np.random.choice(range(len(indices)), p=proba, size=1)
        return indices[index]

    def BalancedPriorityReservoir(self, N, m):
        if N < m:
            return N

        rn = np.random.randint(0, N)
        if rn < m:
            return self._get_index()
        else:
            return -1

    def _update_dict(self, label):
        if label.item() in self.dict:
            self.dict[label.item()] += 1
        else:
            self.dict[label.item()] = 1

    def update_data(self, dict_data):
        for param_name, param_value in dict_data.items():
            if hasattr(self, param_name):
                param_attr = getattr(self, param_name)                
                param_attr[param_value[0]] = param_value[1].cpu()


    def add_data(self, dict_data):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        dict_data: dictionary including the following defaults key, value
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :return:
        """
        if not hasattr(self, "examples"):
            self.init_tensors(dict_data)
        # sets the local variables
        assert "labels" in dict_data
        labels = dict_data["labels"].cpu()
        if self.balance_task_labels: 
            "task_labels" in dict_data
            task_labels = dict_data["task_labels"].cpu()

        n_examples = len(labels)

        i = 0
        last_used_example = -1
        if self.num_seen_examples < self.buffer_size:
            i += np.min([n_examples, self.buffer_size-self.num_seen_examples])
            for param_name, param_value in dict_data.items():
                if hasattr(self, param_name):
                    param_attr = getattr(self, param_name)
                    param_attr[self.num_seen_examples:self.num_seen_examples +
                               i] = param_value[:i].cpu()
            if self.balance_task_labels:
                for j in range(i):
                    self._update_dict(task_labels[j])
            else:    
                for j in range(i):
                    self._update_dict(labels[j])
            self._num_seen_examples += i
            last_used_example += i

        while i < n_examples:
            if self.remainder > 0:
                i += self.remainder
            else:
                i += torch.floor(torch.log(torch.rand(())) /
                                 torch.log(1.-self.W)).int().item() + 1

            if i < n_examples:
                index = self._get_index()
                if self.balance_task_labels:
                    self.dict[self.task_labels[index].item()] -= 1    
                else:
                    self.dict[self.labels[index].item()] -= 1
                for param_name in dict_data.keys():
                    if hasattr(self, param_name):
                        param_attr = getattr(self, param_name)
                        param_attr[index] = dict_data[param_name][i].cpu()
                self._num_seen_examples += i
                last_used_example = i
                if self.balance_task_labels:
                    self._update_dict(task_labels[i])
                else:    
                    self._update_dict(labels[i])
                self.remainder = 0
                self.W = self.W * \
                    torch.exp(torch.log(torch.rand(()))/self.buffer_size)

            else:
                self.remainder = i - (n_examples - last_used_example)

    def get_data(
        self, size: int, transform: transforms = None, return_indexes=False
    ) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return: a dictionary of attributes
        """
        if size > self.num_seen_examples:
            size = self.num_seen_examples

        range = np.min([self.buffer_size, self.num_seen_examples])

        choice = np.random.choice(range, size=size, replace=False)
        if transform is None:
            ret_dict = {"examples": self.examples[choice].to(
                self._device["examples"])}
        else:
            ret_dict = {
                "examples": torch.stack(
                    [
                        transform(ee.to(self._device["examples"]))
                        for ee in self.examples[choice]
                    ]
                ),
            }
        for attr_str in self.attributes[1:-1]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_dict[attr_str] = attr[choice].to(self._device[attr_str])
        if not return_indexes:
            return ret_dict
        else:
            ret_dict.update({"indices":torch.as_tensor(choice)})
            return  ret_dict

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def get_all_data(self, transform: transforms = None) -> Tuple:
        """
        Return all the items in the memory buffer.
        :param transform: the transformation to be applied (data augmentation)
        :return: a dict with all the items in the memory buffer
        """
        range = np.min([self.buffer_size, self.num_seen_examples])

        if transform is None:
            ret_dict = {"examples": self.examples.to(
            self._device["examples"])}
        else:
            ret_dict = {
                "examples": torch.stack(
                    [
                        transform(ee.to(self._device["examples"]))
                        for ee in self.examples
                    ]
                ),
            }
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_dict.update({attr_str: attr[:range]})
        return ret_dict

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                setattr(self, attr_str, None)
        self._num_seen_examples = torch.tensor(0).int()

    def reset(self) -> None:
        self._num_seen_examples = torch.tensor(0).int()
