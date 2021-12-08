"""Base for the model used by the `BaseMethod`.

This model is basically just an encoder and an output head. Both of these can be
switched out/customized as needed.
"""
from dataclasses import dataclass
from typing import (
    Any,
    ClassVar,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)
import dataclasses

import gym
import numpy as np
import torch
import torchvision.models as tv_models
from gym import Space, spaces
from gym.spaces.utils import flatdim
from pytorch_lightning import LightningModule
from pytorch_lightning.core.lightning import ModelSummary, log
from sequoia.common.config import Config
from sequoia.common.gym_wrappers.convert_tensors import add_tensor_support
from sequoia.common.hparams import HyperParameters, categorical, log_uniform, uniform
from sequoia.common.loss import Loss
from sequoia.common.spaces import Image
from sequoia.common.transforms import SplitBatch
from sequoia.methods.models.output_heads import OutputHead
from sequoia.settings.assumptions.incremental import IncrementalAssumption
from sequoia.settings.base import Environment
from sequoia.settings.base.setting import Actions, Observations, Rewards
from sequoia.settings.rl import ContinualRLSetting, RLSetting
from sequoia.settings.sl import SLSetting
from sequoia.utils import Parseable, Serializable
from sequoia.utils.logging_utils import get_logger
from sequoia.utils.pretrained_utils import get_pretrained_encoder
from simple_parsing import choice, mutable_field
from simple_parsing.helpers.hparams import HyperParameters
from simple_parsing.helpers.serialization import register_decoding_fn
from torch import Tensor, nn, optim
from torch.optim.optimizer import Optimizer  # type: ignore

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
from ..simple_convnet import SimpleConvNet


logger = get_logger(__file__)
SettingType = TypeVar("SettingType", bound=IncrementalAssumption)

available_optimizers: Dict[str, Type[Optimizer]] = {
    "sgd": optim.SGD,
    "adam": optim.Adam,
    "rmsprop": optim.RMSprop,
}
available_encoders: Dict[str, Type[nn.Module]] = {
    "vgg16": tv_models.vgg16,
    "resnet18": tv_models.resnet18,
    "resnet34": tv_models.resnet34,
    "resnet50": tv_models.resnet50,
    "resnet101": tv_models.resnet101,
    "resnet152": tv_models.resnet152,
    "alexnet": tv_models.alexnet,
    "densenet": tv_models.densenet161,
    # TODO: Add the self-supervised pl modules here!
    "simple_convnet": SimpleConvNet,
}


class Model(LightningModule, Generic[SettingType]):
    """ Basic Model to be used by a Method.
    
    Based on the `LightningModule` (nn.Module extended by pytorch-lightning).
    This Model can be trained on either Supervised or Reinforcement Learning environments.

    This model splits the learning task into a representation-learning problem
    and a downstream task (output head) applied on top of it.

    The most important method to understand is the `get_loss` method, which
    is used by the [train/val/test]_step methods which are called by
    pytorch-lightning.
    """

    @dataclass
    class HParams(HyperParameters):
        """ HParams of the Model. """

        # Class variable versions of the above dicts, for easier subclassing.
        # NOTE: These don't get parsed from the command-line.
        available_optimizers: ClassVar[
            Dict[str, Type[Optimizer]]
        ] = available_optimizers.copy()
        available_encoders: ClassVar[
            Dict[str, Type[nn.Module]]
        ] = available_encoders.copy()

        # Learning rate of the optimizer.
        learning_rate: float = log_uniform(1e-6, 1e-2, default=1e-3)
        # L2 regularization term for the model weights.
        weight_decay: float = log_uniform(1e-12, 1e-3, default=1e-6)
        # Which optimizer to use.
        optimizer: Type[Optimizer] = categorical(
            available_optimizers, default=optim.Adam
        )
        # Use an encoder architecture from the torchvision.models package.
        encoder: Type[nn.Module] = categorical(
            available_encoders,
            default=tv_models.resnet18,
            # TODO: Only using these two by default when performing a sweep.
            probabilities={"resnet18": 0.5, "simple_convnet": 0.5},
        )

        # Batch size to use during training and evaluation.
        batch_size: Optional[int] = None

        # Number of hidden units (before the output head).
        # When left to None (default), the hidden size from the pretrained
        # encoder model will be used. When set to an integer value, an
        # additional Linear layer will be placed between the outputs of the
        # encoder in order to map from the pretrained encoder's output size H_e
        # to this new hidden size `new_hidden_size`.
        new_hidden_size: Optional[int] = None
        # Retrain the encoder from scratch.
        train_from_scratch: bool = False
        # Wether we should keep the weights of the pretrained encoder frozen.
        freeze_pretrained_encoder_weights: bool = False

        # Settings for the output head.
        # TODO: This could be overwritten in a subclass to do classification or
        # regression or RL, etc.
        output_head: OutputHead.HParams = mutable_field(OutputHead.HParams)

        # Wether the output head should be detached from the representations.
        # In other words, if the gradients from the downstream task should be
        # allowed to affect the representations.
        detach_output_head: bool = False

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
        self.hp: Model.HParams = hparams

        self.Observations: Type[Observations] = setting.Observations
        self.Actions: Type[Actions] = setting.Actions
        self.Rewards: Type[Rewards] = setting.Rewards

        # Choose what type of output head to use depending on the kind of
        # Setting.
        self.OutputHead: Type[OutputHead] = self.output_head_type(setting)

        self.observation_space: gym.Space = setting.observation_space
        self.action_space: gym.Space = setting.action_space
        self.reward_space: gym.Space = setting.reward_space

        self.input_shape = self.observation_space.x.shape
        self.reward_shape = self.reward_space.shape

        self.config: Config = config
        # NOTE: do NOT set the `datamodule` property, otherwise the trainer will ignore
        # the passed train/val/test dataloader from the Setting.
        # self.datamodule: LightningDataModule = setting

        # (Testing) Setting this attribute is supposed to help with ddp/etc
        # training in pytorch-lightning. Not 100% sure.
        # self.example_input_array = torch.rand(self.batch_size, *self.input_shape)

        # Create the encoder and the output head.
        # Space of our encoder representations.
        self.representation_space: gym.Space
        observing_state = not isinstance(setting.observation_space.x, Image)
        if isinstance(setting, ContinualRLSetting) and observing_state:
            # ISSUE # 62: Need to add a dense network instead of no encoder, and
            # change the PolicyHead to have only one layer.
            # Only pass the image, not the task labels to the encoder (for now).
            input_dims = flatdim(self.observation_space["x"])
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
            self.encoder, self.hidden_size = self.make_encoder()
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

    def make_encoder(self) -> Tuple[nn.Module, int]:
        """Creates an Encoder model and returns the number of output dimensions.

        Returns:
            Tuple[nn.Module, int]: the encoder and the hidden size.
            
        TODO: Could instead return its output space, in case we didn't necessarily want
        to flatten the representations (e.g. for image segmentation tasks).
        """
        # Get the chosen type of encoder
        encoder_type: Type[nn.Module] = self.hp.encoder
        # This does a few things:
        # 1. Instantiate the model (with pretrained weights if desired)
        # 2. Infer the output size of the model
        # 3. Remove the output fully-connected layer, if present.
        encoder, hidden_size = get_pretrained_encoder(
            encoder_model=encoder_type,
            pretrained=not self.hp.train_from_scratch,
            freeze_pretrained_weights=self.hp.freeze_pretrained_encoder_weights,
            new_hidden_size=self.hp.new_hidden_size,
        )
        return encoder, hidden_size

    def forward(self, observations: IncrementalAssumption.Observations) -> ForwardPass:
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
        # NOTE: Need to put a `rewards` field in this forward_pass, so we can pass it
        # to the training_step_end method, which will calculate and aggregate the loss
        forward_pass = ForwardPass(
            observations=observations,
            representations=representations,
            actions=actions,
            rewards=None,
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
        if isinstance(setting, RLSetting):
            if not isinstance(setting.action_space, spaces.Discrete):
                raise NotImplementedError("Only support discrete actions for now.")
            assert issubclass(self.hp.rl_output_head_algo, OutputHead)
            return self.hp.rl_output_head_algo

        assert isinstance(setting, SLSetting)

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
        self,
        batch: Tuple[Observations, Optional[Rewards]],
        batch_idx: int,
        environment: Environment = None,
        dataloader_idx: int = None,
        optimizer_idx: int = None,
    ) -> ForwardPass:
        return self.shared_step(
            batch,
            batch_idx=batch_idx,
            environment=environment or self.setting.train_env,
            phase="train",
            dataloader_idx=dataloader_idx,
            optimizer_idx=optimizer_idx,
        )

    def validation_step(
        self,
        batch: Tuple[Observations, Optional[Rewards]],
        batch_idx: int,
        environment: Environment = None,
        dataloader_idx: int = None,
    ) -> ForwardPass:
        return self.shared_step(
            batch,
            batch_idx=batch_idx,
            environment=environment or self.setting.val_env,
            phase="val",
            dataloader_idx=dataloader_idx,
        )

    def test_step(
        self,
        batch: Tuple[Observations, Optional[Rewards]],
        batch_idx: int,
        environment: Environment = None,
        dataloader_idx: int = None,
    ) -> ForwardPass:
        return self.shared_step(
            batch,
            batch_idx=batch_idx,
            environment=environment or self.setting.test_env,
            phase="test",
            dataloader_idx=dataloader_idx,
        )

    def shared_step(
        self,
        batch: Tuple[Observations, Optional[Rewards]],
        batch_idx: int,
        environment: Environment,
        phase: str,
        dataloader_idx: int = None,
        optimizer_idx: int = None,
    ) -> ForwardPass:
        """ Main logic of the "forward pass".

        This is used as part of `training_step`, `validation_step` and `test_step`.
        See the PL docs for `training_step` for more info. 

        NOTE: The prediction / environment interaction / loss calculation has been
        moved into the `shared_step_end` method for DP to also work.
        """

        # Split the batch into observations and (maybe) rewards.
        observations: Observations
        rewards: Optional[Rewards]
        if isinstance(batch, tuple) and len(batch) == 2:
            observations, rewards = batch
        else:
            # TODO: Rework this: Will be an Episode!
            assert isinstance(batch, self.Observations), batch
            observations, rewards = batch, None

        # Get the forward pass results, containing:
        # - "observation": the augmented/transformed/processed observation.
        # - "representations": the representations for the observations.
        # - "actions": The actions (predictions)
        forward_pass: ForwardPass = self(observations)
        if rewards is not None:
            forward_pass = dataclasses.replace(forward_pass, rewards=rewards)
        return forward_pass

    def training_step_end(self, step_outputs: Union[Loss, List[Loss]]) -> Loss:
        loss_object: Loss = self.shared_step_end(
            step_outputs=step_outputs, phase="train", environment=self.setting.train_env
        )
        loss = loss_object.loss
        if not isinstance(loss, Tensor) or not loss.requires_grad:
            # NOTE: There might be no loss at some steps, because for instance
            # we haven't reached the end of an episode in an RL setting.
            return None

        # NOTE In RL, we can only update the model's weights on steps where the output
        # head has as loss, because the output head has buffers of tensors whose grads
        # would become invalidated if we performed the optimizer step.
        if loss.requires_grad and not self.automatic_optimization:
            output_head_loss = loss_object.losses.get(self.output_head.name)
            update_model = (
                output_head_loss is not None and output_head_loss.requires_grad
            )
            optimizer = self.optimizers()

            self.manual_backward(loss, optimizer, retain_graph=not update_model)
            if update_model:
                optimizer.step()
                optimizer.zero_grad()
        # BUG: Need to return this dict, otherwise the optimizer closure in the DP
        # accelerator fails (it only expects to get `dict` or `Tensor` values for
        # `training_step_output` in `_process_training_step_output`)
        # return loss
        # NOTE: the 'hidden' key isn't currently used, but it could be in the future if
        # we added support for BBPT, i.e. recurrent policies or output heads, etc.
        return {"loss": loss, "hidden": loss_object.tensors.get("hidden")}

    def validation_step_end(
        self, step_outputs: Union[ForwardPass, List[ForwardPass]]
    ) -> Loss:
        return self.shared_step_end(
            step_outputs=step_outputs, phase="val", environment=self.setting.val_env
        )

    def test_step_end(
        self, step_outputs: Union[ForwardPass, List[ForwardPass]]
    ) -> Loss:
        return self.shared_step_end(
            step_outputs=step_outputs, phase="test", environment=self.setting.test_env
        )

    def shared_step_end(
        self,
        step_outputs: Union[ForwardPass, List[ForwardPass]],
        phase: str,
        environment: Environment,
    ) -> Loss:
        """ Called with the outputs of each replica's `[train/validation/test]_step`:

        - Sends the Actions from each worker to the environment to obtain rewards, if
          necessary;
        - Calculates the loss, given the merged forward pass and the rewards/labels; 
        - Aggregates the losses/metrics from each replica, logs the relevant values, and
          returns the aggregated losses and metrics (a single Loss object).
        """
        forward_pass: ForwardPass
        if isinstance(step_outputs, list):
            forward_pass = ForwardPass.concatenate(step_outputs)
        else:
            forward_pass = step_outputs

        # get the actions from the forward pass:
        actions = forward_pass.actions
        rewards: Optional[Rewards] = forward_pass.rewards

        if rewards is None:
            # Get the reward from the environment (the dataloader).
            if self.config.debug and self.config.render:
                environment.render("human")
                # import matplotlib.pyplot as plt
                # plt.waitforbuttonpress(10)
            assert isinstance(actions, Actions), actions
            rewards = environment.send(actions)
            assert rewards is not None

        # BUG: Rewards is array of [None]s in TraditionalSL and MultiTask SL!
        assert isinstance(rewards, Rewards), rewards
        # Now that we have the rewards, we calculate the loss.

        loss: Loss = self.get_loss(forward_pass, rewards, loss_name=phase)
        loss_tensor: Tensor = loss.loss
        if loss_tensor == 0.:
            return loss
        loss_pbar_dict = loss.to_pbar_message()
        for key, value in loss_pbar_dict.items():
            assert not isinstance(value, dict), "shouldn't be nested at this point!"
            self.log(key, value, prog_bar=self.config.debug, logger=False)
            logger.debug(f"{key}: {value}")

        loss_log_dict = loss.to_log_dict(verbose=self.config.verbose)
        for key, value in loss_log_dict.items():
            assert not isinstance(value, dict), "shouldn't be nested at this point!"
            self.log(key, value, prog_bar=False, logger=True)
        return loss

    def split_batch(self, batch: Any) -> Tuple[Observations, Optional[Rewards]]:
        """ Splits the batch into the observations and the rewards.

        Uses the types defined on the setting that this model is being applied
        on (which were copied to `self.Observations` and `self.Actions`) to
        figure out how many fields each type requires.

        TODO: This is slightly confusing, should probably get rid of this.
        """
        observations: Observations
        rewards: Optional[Rewards]
        if isinstance(batch, self.Observations):
            observations, rewards = batch, None
        else:
            assert isinstance(batch, (tuple, list)) and len(batch) == 2
            observations, rewards = batch

        assert isinstance(observations, self.Observations), (
            observations,
            type(observations),
            self.Observations,
        )
        # Move the observations to the right device, and convert numpy arrays to
        # tensors.
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
        optimizer_class: Type[Optimzier] = self.hp.optimizer
        options = {
            "lr": self.hp.learning_rate,
            "weight_decay": self.hp.weight_decay,
        }
        return optimizer_class(
            self.parameters(),
            lr=self.hp.learning_rate,
            weight_decay=self.hp.weight_decay,
        )

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
        shared_modules: Dict[str, nn.Module] = nn.ModuleDict()

        if self.encoder:
            shared_modules["encoder"] = self.encoder
        if self.output_head:
            shared_modules["output_head"] = self.output_head
        return shared_modules

    # def summarize(self, mode: str = ModelSummary.MODE_DEFAULT) -> ModelSummary:
    #     model_summary = ModelSummary(self, mode=mode)
    #     log.debug("\n" + str(model_summary))
    #     return model_summary

    def _are_batched(self, observations: IncrementalAssumption.Observations) -> bool:
        """ Returns wether these observations are batched. """
        assert isinstance(self.observation_space, spaces.Dict)

        # if observations.task_labels is not None:
        #     if isinstance(observations.task_labels, int):
        #         return True
        #     assert isinstance(observations.task_labels, (np.ndarray, Tensor))
        #     assert False, observations.shapes
        #     return observations.task_labels.shape and observations.task_labels.shape[0]

        x_space: spaces.Box = self.observation_space["x"]

        if isinstance(x_space, Image) or len(x_space.shape) == 4:
            return observations.x.ndim == 4

        if not isinstance(x_space, spaces.Box):
            raise NotImplementedError(
                f"Don't know how to tell if obs space {x_space} is batched, only "
                f"support Box spaces for the observation's 'x' for now."
            )

        # self.observation_space *should* usually reflect the shapes of individual
        # (non-batched) observations.
        return observations.x.ndim == len(x_space.shape) + 1


# Registering this handler for decoding the type of output head to use (a field in the
# hparams) from a dictionary.
register_decoding_fn(Type[OutputHead], lambda v: v)
