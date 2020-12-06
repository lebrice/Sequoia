"""Base class for the Model to be used as part of a Method.

This is meant

TODO: There is a bunch of work to be done here.
"""
import gym
import dataclasses
import itertools
import functools
from abc import ABC
from dataclasses import dataclass
from collections import abc as collections_abc
from typing import (Any, ClassVar, Dict, Generic, List, NamedTuple, Optional,
                    Sequence, Tuple, Type, TypeVar, Union)

import numpy as np
import torch
from gym import spaces
from gym.spaces.utils import flatdim
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.core.decorators import auto_move_data
from pytorch_lightning.core.lightning import ModelSummary, log
from simple_parsing import list_field
from torch import nn, Tensor
from torch.optim.optimizer import Optimizer

from common.config import Config
from common.loss import Loss
from common.transforms import SplitBatch, Transforms
from common.batch import Batch
from settings.base.setting import Setting, SettingType, Observations, Actions, Rewards
from settings.assumptions.incremental import IncrementalSetting
from settings import ContinualRLSetting
from settings import Environment, Observations, Actions, Rewards
from utils.logging_utils import get_logger

from .base_hparams import BaseHParams
from ..output_heads import OutputHead, RegressionHead, ClassificationHead, PolicyHead
from ..forward_pass import ForwardPass


logger = get_logger(__file__)

class BaseModel(LightningModule, Generic[SettingType]):
    """ Base model LightningModule (nn.Module extended by pytorch-lightning)
    
    WIP: (@lebrice): Trying to tidy up the hierarchy of the different kinds of
    models a little bit. 
    
    This model splits the learning task into a representation-learning problem
    and a downstream task (output head) applied on top of it.   

    The most important method to understand is the `get_loss` method, which
    is used by the [train/val/test]_step methods which are called by
    pytorch-lightning.
    """
    @dataclass
    class HParams(BaseHParams):
        """ HParams of the Model. """

    def __init__(self, setting: SettingType, hparams: HParams, config: Config):
        super().__init__()
        self.setting: SettingType = setting
        self.hp = hparams
        
        self.Observations: Type[Observations] = setting.Observations
        self.Actions: Type[Actions] = setting.Actions
        self.Rewards: Type[Rewards] = setting.Rewards
        
        self.observation_space: gym.Space = setting.observation_space
        self.action_space: gym.Space = setting.action_space
        self.reward_space: gym.Space = setting.reward_space
        
        self.input_shape  = self.observation_space[0].shape
        self.reward_shape = self.reward_space.shape
        
        self.split_batch_transform = SplitBatch(observation_type=self.Observations,
                                                reward_type=self.Rewards)
        self.config: Config = config
        # TODO: Decided to Not set this property, so the trainer doesn't
        # fallback to using it instead of the passed datamodules/dataloaders.
        # self.datamodule: LightningDataModule = setting

        # TODO: Debug this, seemed to be causing issues because it was trying to
        # save the 'setting' argument, which contained some things that weren't
        # pickleable.
        all_params_dict = {
            "hparams": hparams.to_dict(),
            "config": config.to_dict(),
        }
        self.save_hyperparameters(all_params_dict)

        # (Testing) Setting this attribute is supposed to help with ddp/etc
        # training in pytorch-lightning. Not 100% sure.
        # self.example_input_array = torch.rand(self.batch_size, *self.input_shape)
        
        # Create the encoder and the output head.
        # Space of our encoder representations.
        self.representation_space: gym.Space
        if isinstance(setting, ContinualRLSetting) and setting.observe_state_directly:
            self.encoder = nn.Sequential()
            # self.representation_space = self.observation_space
            self.representation_space = self.observation_space[0]
            self.hidden_size = flatdim(self.observation_space[0])
        else:
            self.encoder, self.hidden_size = self.hp.make_encoder()
            # TODO: Check that the outputs of the encoders are actually
            # flattened. I'm not sure they all are, which case the samples
            # wouldn't match with this space. 
            self.representation_space = spaces.Box(-np.inf, np.inf, (self.hidden_size,), np.float32)

        from common.gym_wrappers.convert_tensors import wrap_space
        self.representation_space = wrap_space(self.representation_space)

        self.output_head = self.create_output_head()

    @auto_move_data
    def forward(self, observations:  IncrementalSetting.Observations) -> Dict[str, Tensor]:
        """ Forward pass of the Model.

        Returns a ForwardPass object (or a dict)
        """        
        # Encode the observation to get representations.
        representations = self.encode(observations)
        # Pass the observations and representations to the output head to get
        # the 'action' (prediction).
        actions = self.get_actions(observations, representations)

        forward_pass = ForwardPass(
            observations=observations,
            representations=representations,
            actions=actions,
        )
        return forward_pass

    def encode(self, observations: Observations) -> Tensor:
        """Encodes a batch of samples `x` into a hidden vector.

        Args:
            observations (Union[Tensor, Observation]): Tensor of Observation
            containing a batch of samples (before preprocess_observations).

        Returns:
            Tensor: The hidden vector / embedding for that sample, with size
                [<batch_size>, `self.hp.hidden_size`].
        """
        assert isinstance(observations, self.Observations)
        # If there's any additional 'input preprocessing' to do, do it here.
        # NOTE (@lebrice): This is currently done this way so that we don't have
        # to pass transforms to the settings from the method side.
        preprocessed_observations = self.preprocess_observations(observations)
        # Here in this base model the encoder only takes the 'x' from the observations.
        h_x = self.encoder(preprocessed_observations.x)
        if isinstance(h_x, list) and len(h_x) == 1:
            # Some pretrained encoders sometimes give back a list with one tensor. (?)
            h_x = h_x[0]
        return h_x

    
    def get_actions(self, observations: Observations, representations: Tensor) -> Actions:
        """ Pass the required inputs to the output head and get predictions.
        
        NOTE: This method is basically just here so we can customize what we
        pass to the output head, or what we take from it, similar to the
        `encode` method.
        """
        if self.hp.detach_output_head:
            representations = representations.detach()

        actions = self.output_head(
            observations=observations,
            representations=representations
        )
        assert isinstance(actions, self.Actions)
        return actions

    def create_output_head(self) -> OutputHead:
        """ Create an output head for the current action space. """
        
        output_head: OutputHead
        # TODO: Are there any bounds on the representation space?

        if isinstance(self.action_space, spaces.Discrete):
            if isinstance(self.reward_space, spaces.Discrete):
                action_space = self.action_space
                reward_space = self.reward_space
                if self.hp.multihead:
                    # The action space for each head is smaller (each head does
                    # n_t-way classification, rather than N-way, with n_t being
                    # the number of classes per task.)
                    from common.gym_wrappers.convert_tensors import wrap_space
                    from settings.passive import ClassIncrementalSetting
                    assert isinstance(self.setting, ClassIncrementalSetting)
                    n_classes_in_current_task = self.setting.num_classes_in_current_task()
                    # TODO: Is it imperative that the output should be a single
                    # logit if we're doing binary (2-way) classification? Or is
                    # it alright to use softmax over 2 logits? 
                    action_space = spaces.Discrete(n_classes_in_current_task)
                    reward_space = spaces.Discrete(n_classes_in_current_task)
                    wrap_space(action_space)
                    wrap_space(reward_space)

                # Classification problem:
                # self.output_shape = (self.action_space.n,)
                output_head = ClassificationHead(
                    input_space=self.representation_space,
                    action_space=action_space,
                    reward_space=reward_space,
                    hparams=self.hp.output_head,
                )
            else:
                # RL problem, reward is a scalar.
                # self.output_shape = self.reward_space.shape
                output_head = PolicyHead(
                    input_space=self.representation_space,
                    action_space=self.action_space,
                    reward_space=self.reward_space,
                    hparams=self.hp.output_head,
                )
        elif isinstance(self.action_space, spaces.Box):
            # Regression problem
            self.output_shape = self.action_space.shape
            output_head = RegressionHead(
                input_space=self.representation_space,
                action_space=self.action_space,
                reward_space=self.reward_space,
                hparams=self.hp.output_head,
            )
        else:
            raise NotImplementedError(f"Unsupported action space: {self.action_space}")
        
        # Add the new parameters to the Optimizer, if it already exists.
        if self.trainer:
            optimizer: Optimizer = self.optimizers()
            assert isinstance(optimizer, Optimizer)
            optimizer.add_param_group({"params": output_head.parameters()})

        output_head = output_head.to(self.device)
        return output_head

    def training_step(self,
                      batch: Tuple[Observations, Optional[Rewards]],
                      *args,
                      **kwargs):
        return self.shared_step(
            batch,
            *args,
            environment=self.setting.train_env,
            loss_name="train",
            **kwargs
        )

    def validation_step(self,
                      batch: Tuple[Observations, Optional[Rewards]],
                      *args,
                      **kwargs):
        return self.shared_step(
            batch,
            *args,
            environment=self.setting.val_env,
            loss_name="val",
            **kwargs
        )

    def test_step(self,
                      batch: Tuple[Observations, Optional[Rewards]],
                      *args,
                      **kwargs):
        return self.shared_step(
            batch,
            *args,
            environment=self.setting.test_env,
            loss_name="test",
            **kwargs
        )

    def shared_step(self,
                    batch: Tuple[Observations, Rewards],
                    batch_idx: int,
                    environment: Environment,
                    loss_name: str,
                    dataloader_idx: int = None,
                    optimizer_idx: int = None) -> Dict:
        """
        This is the shared step for this 'example' LightningModule. 
        Feel free to customize/change it if you want!
        """
        if dataloader_idx is not None:
            assert isinstance(dataloader_idx, int)
            loss_name += f"/{dataloader_idx}"

        # Split the batch into observations and rewards.
        # NOTE: Only in the case of the Supervised settings do we ever get the
        # Rewards at the same time as the Observations.
        # TODO: It would be nice if we could actually do the same things for
        # both sides of the tree here..
        observations, rewards = self.split_batch(batch)
        # Get the forward pass results, containing:
        # - "observation": the augmented/transformed/processed observation.
        # - "representations": the representations for the observations.
        # - "actions": The actions (predictions)
        forward_pass: ForwardPass = self(observations)
        
        # get the actions from the forward pass:
        actions = forward_pass.actions
        
        if rewards is None:
            # Get the reward from the environment (the dataloader).
            if self.config.debug:
                environment.render()
                # import matplotlib.pyplot as plt
                # plt.waitforbuttonpress(10)
            
            rewards = environment.send(actions)
            assert rewards is not None

        loss: Loss = self.get_loss(forward_pass, rewards, loss_name=loss_name)
        
        return {
            "loss": loss.loss,
            "log": loss.to_log_dict(),
            "progress_bar": loss.to_pbar_message(),
            "loss_object": loss,
        }
        # The above can also easily be retrieved like this:
        result = loss.to_pl_dict()
        return result
    
    @auto_move_data
    def split_batch(self, batch: Any) -> Tuple[Observations, Rewards]:
        """ Splits the batch into the observations and the rewards. 
        
        Uses the types defined on the setting that this model is being applied
        on (which were copied to `self.Observations` and `self.Actions`) to
        figure out how many fields each type requires.

        TODO: This is slightly confusing, should probably get rid of this.
        """
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            observations, rewards = batch
            if (isinstance(observations, self.Observations) and
                isinstance(rewards, self.Rewards)):
                return observations, rewards
        return self.split_batch_transform(batch)

    def get_loss(self,
                 forward_pass: Dict[str, Tensor],
                 rewards: Rewards = None,
                 loss_name: str = "") -> Loss:
        """Gets a Loss given the results of the forward pass and the reward.

        Args:
            forward_pass (Dict[str, Tensor]): Results of the forward pass.
            reward (Tensor, optional): The reward that resulted from the action
                chosen in the forward pass. Defaults to None.
            loss_name (str, optional): The name for the resulting Loss.
                Defaults to "".

        Returns:
            Loss: a Loss object containing the loss tensor, associated metrics
            and sublosses.
        
        This could look a bit like this, for example:
        ```
        action = forward_pass["action"]
        predicted_reward = forward_pass["predicted_reward"]
        nce = self.loss_fn(predicted_reward, reward)
        loss = Loss(loss_name, loss=nce)
        return loss
        ```
        """
        assert loss_name
        # Create an 'empty' Loss object with the given name, so that we always
        # return a Loss object, even when `y` is None and we can't the loss from
        # the output_head.
        total_loss = Loss(name=loss_name)
        if rewards:
            assert rewards.y is not None
            # TODO: If we decide to re-organize the forward pass object to also
            # contain the predictions of the self-supervised tasks, (atm they
            # perform their 'forward pass' in their get_loss functions)
            # then we could change 'actions' to be a dict, and index the
            # dict with the 'name' of each output head, like so:
            # actions_of_head = forward_pass.actions[self.output_head.name]
            # rewards_of_head = forward_pass.rewards[self.output_head.name]

            # For now though, we only have one "prediction" in the actions:
            actions = forward_pass.actions

            # So far we only use 'y' from the rewards in the output head.
            supervised_loss = self.output_head.get_loss(forward_pass, actions=actions, rewards=rewards)
            total_loss += supervised_loss
            
            
            
        return total_loss

    def preprocess_observations(self, observations: Observations) -> Observations:
        assert isinstance(observations, Observations)
        return observations

    def preprocess_rewards(self, reward: Rewards) -> Rewards:
        return reward

    def configure_optimizers(self):
        return self.hp.make_optimizer(self.parameters())

    @property
    def batch_size(self) -> int:
        return self.hp.batch_size

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        self.hp.batch_size = value 
    
    @property
    def learning_rate(self) -> float:
        return self.hp.learning_rate

    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        self.hp.learning_rate = value

    def on_task_switch(self, task_id: int, training: bool = False) -> None:
        """Called when switching between tasks.
        
        Args:
            task_id (int): the Id of the task.
            training (bool): Wether we are currently training or valid/testing.
        """

    def summarize(self, mode: str = ModelSummary.MODE_DEFAULT) -> ModelSummary:
        model_summary = ModelSummary(self, mode=mode)
        log.debug('\n' + str(model_summary))
        return model_summary
