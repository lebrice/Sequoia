"""Base class for the Model to be used as part of a Method.

This is meant

TODO: There is a bunch of work to be done here.
"""
from dataclasses import dataclass
from typing import Any, Dict, Generic, Optional, Tuple, Type, TypeVar

import gym
import numpy as np
import torch
from gym import Space, spaces
from gym.spaces.utils import flatdim
from pytorch_lightning import LightningModule
from pytorch_lightning.core.decorators import auto_move_data
from pytorch_lightning.core.lightning import ModelSummary, log
from simple_parsing import choice
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer

from sequoia.common.config import Config
from sequoia.common.gym_wrappers.convert_tensors import add_tensor_support
from sequoia.common.loss import Loss
from sequoia.common.transforms import SplitBatch
from sequoia.common.spaces import Image
from sequoia.settings.active import ActiveSetting, ContinualRLSetting
from sequoia.settings.passive import PassiveSetting
from sequoia.settings.base import Environment
from sequoia.settings.assumptions.incremental import IncrementalSetting
from sequoia.settings.base.setting import Actions, Observations, Rewards
from sequoia.utils.logging_utils import get_logger
from sequoia.utils.module_dict import ModuleDict

from ..fcnet import FCNet
from ..forward_pass import ForwardPass
from ..output_heads import (
    ActorCriticHead,
    ClassificationHead,
    OutputHead,
    PolicyHead,
    RegressionHead,
)
from ..output_heads.rl.episodic_a2c import EpisodicA2C
from .base_hparams import BaseHParams

logger = get_logger(__file__)
SettingType = TypeVar("SettingType", bound=IncrementalSetting)


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

        # Which algorithm to use for the output head when in an RL setting.
        # TODO: Run the PolicyHead in the following conditions:
        # - Compare the big backward pass vs many small ones
        # - Try to have it learn from pixel input, if possible
        # - Try to have it learn on a multi-task RL setting,
        # TODO: Finish the ActorCritic and EpisodicA2C heads.
        rl_output_head_algo: Type[OutputHead] = choice(
            {
                "reinforce": PolicyHead,
                "a2c_online": ActorCriticHead,
                "a2c_episodic": EpisodicA2C,
            },
            default=EpisodicA2C,
        )

    def __init__(self, setting: SettingType, hparams: HParams, config: Config):
        super().__init__()
        self.setting: SettingType = setting
        self.hp: BaseModel.HParams = hparams

        self.Observations: Type[Observations] = setting.Observations
        self.Actions: Type[Actions] = setting.Actions
        self.Rewards: Type[Rewards] = setting.Rewards

        # Choose what type of output head to use depending on the kind of
        # Setting.
        self.OutputHead: Type[OutputHead] = self.output_head_type(setting)

        self.observation_space: gym.Space = setting.observation_space
        self.action_space: gym.Space = setting.action_space
        self.reward_space: gym.Space = setting.reward_space

        self.input_shape = self.observation_space[0].shape
        self.reward_shape = self.reward_space.shape

        # TODO: Remove this:
        self.split_batch_transform = SplitBatch(
            observation_type=self.Observations, reward_type=self.Rewards
        )
        self.config: Config = config
        # TODO: Decided to Not set this property, so the trainer doesn't
        # fallback to using it instead of the passed datamodules/dataloaders.
        # self.datamodule: LightningDataModule = setting

        # (Testing) Setting this attribute is supposed to help with ddp/etc
        # training in pytorch-lightning. Not 100% sure.
        # self.example_input_array = torch.rand(self.batch_size, *self.input_shape)

        # Create the encoder and the output head.
        # Space of our encoder representations.
        self.representation_space: gym.Space
        if isinstance(setting, ContinualRLSetting) and setting.observe_state_directly:
            # ISSUE # 62: Need to add a dense network instead of no encoder, and
            # change the PolicyHead to have only one layer.
            # Only pass the image, not the task labels to the encoder (for now).
            input_dims = flatdim(self.observation_space[0])
            output_dims = self.hp.new_hidden_size or 128

            self.encoder = FCNet(
                in_features=input_dims,
                out_features=output_dims,
                hidden_layers=3,
                hidden_neurons=[256, 128, output_dims],
                activation=nn.ReLU,
            )
            self.representation_space = add_tensor_support(
                spaces.Box(low=-np.inf, high=np.inf, shape=[output_dims])
            )
            self.hidden_size = output_dims
        else:
            # TODO: Refactor this 'make_encoder' being on the hparams, its a bit
            # weird.
            self.encoder, self.hidden_size = self.hp.make_encoder()
            # TODO: Check that the outputs of the encoders are actually
            # flattened. I'm not sure they all are, which case the samples
            # wouldn't match with this space.
            self.representation_space = spaces.Box(
                -np.inf, np.inf, (self.hidden_size,), np.float32
            )

        logger.info(f"Moving encoder to device {self.config.device}")
        self.encoder = self.encoder.to(self.config.device)

        self.representation_space = add_tensor_support(self.representation_space)

        # Upgrade the type of hparams for the output head based on the setting, if
        # needed.
        if not isinstance(self.hp.output_head, self.OutputHead.HParams):
            self.hp.output_head = self.hp.output_head.upgrade(
                target_type=self.OutputHead.HParams
            )
        # Then, create the 'default' output head.
        self.output_head: OutputHead = self.create_output_head(task_id=0)

    @auto_move_data
    def forward(self, observations: IncrementalSetting.Observations) -> ForwardPass:
        """ Forward pass of the Model.

        Returns a ForwardPass object (acts like a dict of Tensors.)
        """
        # If there's any additional 'input preprocessing' to do, do it here.
        # NOTE (@lebrice): This is currently done this way so that we don't have
        # to pass transforms to the settings from the method side.
        observations = self.preprocess_observations(observations)
        # Encode the observation to get representations.
        assert observations.x.device == self.device

        representations = self.encode(observations)
        # Pass the observations and representations to the output head to get
        # the 'action' (prediction).

        if self.hp.detach_output_head:
            representations = representations.detach()

        actions = self.output_head(
            observations=observations, representations=representations
        )
        forward_pass = ForwardPass(
            observations=observations, representations=representations, actions=actions,
        )
        return forward_pass

    def encode(self, observations: Observations) -> Tensor:
        """Encodes a batch of samples `x` into a hidden vector.

        Args:
            observations (Union[Tensor, Observation]): Tensor of Observation
            containing a batch of samples (before preprocess_observations).

        Returns:
            Tensor: The hidden vector / embedding for that sample, with size
                [B, `self.hidden_size`].
        """
        # Here in this base model the encoder only takes the 'x' from the
        # observations.
        x = torch.as_tensor(observations.x, device=self.device, dtype=self.dtype)
        assert x.device == self.device
        encoder_parameters = list(self.encoder.parameters())
        encoder_device = (
            encoder_parameters[0].device if encoder_parameters else self.device
        )
        # BUG: WHen using the EWCTask, there seems to be some issues related to which
        # device the model is stored on.

        if encoder_device != self.device:
            x = x.to(encoder_device)
            # self.encoder = self.encoder.to(self.device)

        h_x = self.encoder(x)

        if encoder_device != self.device:
            h_x = h_x.to(self.device)

        if isinstance(h_x, list) and len(h_x) == 1:
            # Some pretrained encoders sometimes give back a list with one tensor. (?)
            h_x = h_x[0]
        if not isinstance(h_x, Tensor):
            h_x = torch.as_tensor(h_x, device=self.device, dtype=self.dtype)
        return h_x

    def create_output_head(self, task_id: Optional[int]) -> OutputHead:
        """Create an output head for the current action and reward spaces.

        NOTE: This assumes that the input, action and reward spaces don't change
        between tasks.

        Parameters
        ----------
        task_id : Optional[int]
            ID of the task associated with this new output head. Can be `None`, which is
            interpreted as saying that either that task labels aren't available, or that
            this output head will be used for all tasks.

        Returns
        -------
        OutputHead
            The new output head for the given task.
        """
        # NOTE: This assumes that the input, action and reward spaces don't change
        # between tasks.
        # TODO: Maybe add something like `setting.get_action_space(task_id)`
        input_space: Space = self.representation_space
        action_space: Space = self.action_space
        reward_space: Space = self.reward_space
        hparams: OutputHead.HParams = self.hp.output_head
        # NOTE: self.OutputHead is the type of output head used for the current setting.
        # NOTE: Could also use a name for the output head using the task id, for example
        output_head_name = None  # Use the name defined on the output head.
        output_head = self.OutputHead(
            input_space=input_space,
            action_space=action_space,
            reward_space=reward_space,
            hparams=hparams,
            name=output_head_name,
        ).to(self.device)

        # Do not add the output head's parameters to the optimizer of the whole model,
        # if it already has an `optimizer` attribute of its own. (NOTE: this isn't the
        # case in practice so far)
        add_to_optimizer = not getattr(output_head, "optimizer", None)
        if add_to_optimizer:
            # Add the new parameters to the Optimizer, if it already exists.
            # If we don't yet have a Trainer, the Optimizer hasn't been created
            # yet. Once it is created though, it will get the parameters of this output
            # head from `self.parameters()` is passed to its constructor, since the
            # output head will be stored in `self.output_heads`.
            if self.trainer:
                optimizer: Optimizer = self.optimizers()
                assert isinstance(optimizer, Optimizer)
                optimizer.add_param_group({"params": output_head.parameters()})

        return output_head

    def output_head_type(self, setting: SettingType) -> Type[OutputHead]:
        """ Return the type of output head we should use in a given setting.
        """
        if isinstance(setting, ActiveSetting):
            if not isinstance(setting.action_space, spaces.Discrete):
                raise NotImplementedError("Only support discrete actions for now.")
            assert issubclass(self.hp.rl_output_head_algo, OutputHead)
            return self.hp.rl_output_head_algo

        assert isinstance(setting, PassiveSetting)

        if isinstance(setting.action_space, spaces.Discrete):
            # Discrete actions: i.e. classification problem.
            if isinstance(setting.reward_space, spaces.Discrete):
                # Classification problem: Discrete action, Discrete rewards (labels).
                return ClassificationHead
            # Reinforcement learning problem: Discrete action, float rewards.
            # TODO: There might be some RL environments with discrete
            # rewards, right? For instance CartPole is, on-paper, a discrete
            # reward setting, since its always 1.
        if isinstance(setting.action_space, spaces.Box):
            # Regression problem: For now there is only RL that has such a
            # space.
            return RegressionHead

        raise NotImplementedError(f"Unsupported action space: {setting.action_space}")

    def training_step(
        self, batch: Tuple[Observations, Optional[Rewards]], batch_idx: int, **kwargs
    ):
        return self.shared_step(
            batch,
            batch_idx,
            environment=self.setting.train_env,
            loss_name="train",
            **kwargs,
        )

    def validation_step(
        self, batch: Tuple[Observations, Optional[Rewards]], batch_idx: int, **kwargs
    ):
        return self.shared_step(
            batch,
            batch_idx=batch_idx,
            environment=self.setting.val_env,
            loss_name="val",
            **kwargs,
        )

    def test_step(
        self, batch: Tuple[Observations, Optional[Rewards]], batch_idx: int, **kwargs
    ):
        return self.shared_step(
            batch,
            batch_idx=batch_idx,
            environment=self.setting.test_env,
            loss_name="test",
            **kwargs,
        )

    def shared_step(
        self,
        batch: Tuple[Observations, Rewards],
        batch_idx: int,
        environment: Environment,
        loss_name: str,
        dataloader_idx: int = None,
        optimizer_idx: int = None,
    ) -> Dict:
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

        # FIXME: Remove this, debugging:
        assert isinstance(observations, Observations), observations
        assert isinstance(observations.x, Tensor), observations.shapes
        # Get the forward pass results, containing:
        # - "observation": the augmented/transformed/processed observation.
        # - "representations": the representations for the observations.
        # - "actions": The actions (predictions)
        forward_pass: ForwardPass = self(observations)

        # get the actions from the forward pass:
        actions = forward_pass.actions

        if rewards is None:
            # Get the reward from the environment (the dataloader).
            if self.config.debug and self.config.render:
                environment.render("human")
                # import matplotlib.pyplot as plt
                # plt.waitforbuttonpress(10)

            rewards = environment.send(actions)
            assert rewards is not None

        loss: Loss = self.get_loss(forward_pass, rewards, loss_name=loss_name)
        return {
            "loss": loss.loss,
            "loss_object": loss,
        }

    def split_batch(self, batch: Any) -> Tuple[Observations, Rewards]:
        """ Splits the batch into the observations and the rewards.

        Uses the types defined on the setting that this model is being applied
        on (which were copied to `self.Observations` and `self.Actions`) to
        figure out how many fields each type requires.

        TODO: This is slightly confusing, should probably get rid of this.
        """
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            observations, rewards = batch
            assert isinstance(observations, self.Observations)
            assert rewards is None or isinstance(rewards, self.Rewards)
            # raise NotImplementedError(
            #     f"Expected to receive observations of type {self.Observations}, but "
            #     f"got {observations} of type {type(observations)} instead."
            # )
        else:
            assert isinstance(batch, self.Observations)
            observations = batch
            rewards = None

        observations = observations.torch(device=self.device)
        if rewards is not None:
            rewards = rewards.torch(device=self.device)
        return observations, rewards

    def get_loss(
        self, forward_pass: ForwardPass, rewards: Rewards = None, loss_name: str = ""
    ) -> Loss:
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
            supervised_loss = self.output_head_loss(
                forward_pass, actions=actions, rewards=rewards
            )
            total_loss += supervised_loss

        return total_loss

    def output_head_loss(
        self, forward_pass: ForwardPass, actions: Actions, rewards: Rewards
    ) -> Loss:
        """ Gets the Loss of the output head. """
        # TODO: The rewards can still contain just numpy arrays, keeping it so for now.
        assert actions.device == self.device  # == rewards.device (would be None)
        return self.output_head.get_loss(
            forward_pass, actions=actions, rewards=rewards,
        )

    def preprocess_observations(self, observations: Observations) -> Observations:
        assert isinstance(observations, self.Observations)
        # TODO: Make sure this also works in the supervised setting.
        # Convert all numpy arrays to tensors if possible.
        # TODO: Make sure this still works in settings without task labels (
        # None in numpy arrays)
        observations = observations.torch(device=self.device)
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

    def on_task_switch(self, task_id: Optional[int]) -> None:
        """Called when switching between tasks.

        Args:
            task_id (Optional[int]): the Id of the task.
        """

    def shared_modules(self) -> Dict[str, nn.Module]:
        """Returns any trainable modules in `self` that are shared across tasks.

        By giving this information, these weights can then be used in
        regularization-based auxiliary tasks like EWC, for example.

        Returns
        -------
        Dict[str, nn.Module]:
            Dictionary mapping from name to the shared modules, if any.
        """
        shared_modules: Dict[str, nn.Module] = ModuleDict()

        if self.encoder:
            shared_modules["encoder"] = self.encoder
        if self.output_head:
            shared_modules["output_head"] = self.output_head
        return shared_modules

    def summarize(self, mode: str = ModelSummary.MODE_DEFAULT) -> ModelSummary:
        model_summary = ModelSummary(self, mode=mode)
        log.debug("\n" + str(model_summary))
        return model_summary

    def _are_batched(self, observations: IncrementalSetting.Observations) -> bool:
        """ Returns wether these observations are batched. """
        assert isinstance(self.observation_space, spaces.Tuple)

        if observations.task_labels is not None:
            if isinstance(observations.task_labels, int):
                return True
            assert isinstance(observations.task_labels, (np.ndarray, Tensor))
            return observations.task_labels.shape and observations.task_labels.shape[0]

        x_space: spaces.Box = self.observation_space[0]

        if isinstance(x_space, Image):
            return observations.x.ndim == 4

        if len(x_space.shape) == 4:
            # This would occur when the observation space isn't an Image space, for some
            # reason.
            return observations.x.ndim == 4

        if not isinstance(x_space, spaces.Box):
            raise NotImplementedError(
                f"Don't know how to tell if obs space {x_space} is batched, only "
                f"support Box spaces for the observation's 'x' for now."
            )

        # self.observation_space *should* usually reflect the shapes of individual
        # (non-batched) observations.
        if len(x_space.shape) == 1:
            return observations.x.ndim == 2

        if len(x_space.shape) == 3:
            return observations.x.ndim == 4

        if len(x_space.shape) == 2:
            # TODO: This assumes that any observation space with ndim == 2 is already
            # batched. This currently only makes sense for cartpole.
            if self.setting.dataset == "CartPole-v0":
                return observations.x.ndim == 2
            # assert False, self.setting.train_env.observation_space

        return not len(observations.x.shape) == len(x_space.shape)


from simple_parsing.helpers.serialization import register_decoding_fn

register_decoding_fn(Type[OutputHead], lambda v: v)
