import torch
from torch import nn
from pytorch_lightning.callbacks import Callback
from sequoia.methods.base_method import BaseMethod
from sequoia.settings.sl import TaskIncrementalSLSetting
from pytorch_lightning import Trainer
import torch

from sequoia.methods.base_method import BaseModel
from dataclasses import dataclass
from simple_parsing.helpers import mutable_field
from sequoia.methods.base_method import BaseModel
from dataclasses import dataclass
from pytorch_lightning import LightningModule, Trainer, Callback
from simple_parsing.helpers.hparams import HyperParameters, uniform
from typing import Union, List, Tuple, Optional, Mapping, Dict, Any

from sequoia.settings.assumptions import IncrementalAssumption as IncrementalSetting


#
# TODO: Make this compatible with multiple output heads (broken as of now)
#

class PackNet(Callback, nn.Module):
    @dataclass
    class HParams(HyperParameters):
        """ Hyper-parameters of the Packnet callback. """

        # Ratio of training epochs to finetuning epochs.
        # TODO: Need to adapt this a bit to better work with the BaseMethod, which has
        # a `max_epochs` parameter, and also uses early stopping. Maybe it could be
        # something like: Do a minimum of 3 epochs of training, and the rest is just
        # fine-tuning?

        prune_instructions: Union[float, List[float]] = uniform(.1, .9, default=.5)

        train_epochs: int = uniform(1, 5, default=3)
        fine_tune_epochs: int = uniform(0, 5, default=1)

    def __init__(self, n_tasks: int, hparams: "PackNet.HParams"):
        super().__init__()
        self.n_tasks = n_tasks
        self.prune_instructions = hparams.prune_instructions
        # Set up an array of quantiles for pruning procedure
        if n_tasks:
            self.config_instructions()

        self.PATH = None
        self.epoch_split = (hparams.train_epochs, hparams.fine_tune_epochs)
        self.current_task = 0
        self.masks = []  # 3-dimensions: task, layer, parameter mask
        self.mode = None
        self._enabled = True

    def enable(self) -> None:
        """ Enable the PackNet behaviour. """
        self._enabled = True

    def disable(self) -> None:
        """ Disable the PackNet behaviour. """
        self._enabled = False

    def prune(self, model: nn.Module, prune_quantile: float):
        """
        Create task-specific mask and prune least relevant weights
        :param prune_quantile: The percentage of weights to prune as a decimal
        """
        # Calculate Quantile
        all_prunable = torch.tensor([])
        mask_idx = 0
        for mod_name, mod in model.named_modules():
            if (isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear)) and 'output_head' not in mod_name:
                for name, param_layer in mod.named_parameters():
                    if 'bias' not in name:
                        # get fixed weights for this layer
                        prev_mask = torch.zeros(param_layer.size(), dtype=torch.bool, requires_grad=False)

                        for task in self.masks:
                            prev_mask |= task[mask_idx]

                        p = param_layer.masked_select(~prev_mask)

                        if p is not None:
                            all_prunable = torch.cat((all_prunable.view(-1), p), -1)

                        mask_idx += 1

        cutoff = torch.quantile(torch.abs(all_prunable), q=prune_quantile)

        mask_idx = 0
        mask = []  # create mask for this task
        with torch.no_grad():
            for mod_name, mod in model.named_modules():
                if (isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear)) and 'output_head' not in mod_name:
                    for name, param_layer in mod.named_parameters():
                        if 'bias' not in name:
                            # get weight mask for this layer
                            prev_mask = torch.zeros(param_layer.size(), dtype=torch.bool, requires_grad=False)  # p
                            for task in self.masks:
                                prev_mask |= task[mask_idx]

                            curr_mask = torch.abs(param_layer).ge(cutoff)  # q
                            curr_mask = torch.logical_and(curr_mask, ~prev_mask)  # (q & ~p)

                            # Zero non masked weights
                            param_layer *= (curr_mask | prev_mask)

                            mask.append(curr_mask)
                            mask_idx += 1

        self.masks.append(mask)

    def fine_tune_mask(self, model: nn.Module):
        """
        Zero the gradient of pruned weights this task as well as previously fixed weights
        Apply this mask before each optimizer step during fine-tuning
        """
        assert len(self.masks) > self.current_task

        mask_idx = 0
        for mod_name, mod in model.named_modules():
            if (isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear)) and 'output_head' not in mod_name:
                for name, param_layer in mod.named_parameters():
                    if 'bias' not in name:
                        param_layer.grad *= self.masks[self.current_task][mask_idx]
                        mask_idx += 1

    def training_mask(self, model: nn.Module):
        """
        Zero the gradient of only fixed weights for previous tasks
        Apply this mask after .backward() and before
        optimizer.step() at every batch of training a new task
        """
        if len(self.masks) == 0:
            return

        mask_idx = 0
        for mod_name, mod in model.named_modules():
            if (isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear)) and 'output_head' not in mod_name:
                for name, param_layer in mod.named_parameters():
                    if 'bias' not in name:
                        # get mask of weights from previous tasks
                        prev_mask = torch.zeros(param_layer.size(), dtype=torch.bool, requires_grad=False)
                        for task_masks in self.masks:
                            prev_mask |= task_masks[mask_idx]

                        # zero grad of previous fixed weights
                        param_layer.grad *= ~prev_mask

                        mask_idx += 1

    def fix_biases(self, model: nn.Module):
        """
        Fix the gradient of bias parameters
        """
        for mod in model.modules():
            if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                for name, param_layer in mod.named_parameters():
                    if 'bias' in name:
                        param_layer.requires_grad = False

    def fix_batch_norm(self, model: nn.Module):
        """
        Fix batch norm gain, bias, running mean and variance
        """
        for mod in model.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.affine = False
                for param_layer in mod.parameters():
                    param_layer.requires_grad = False

    def fix_output_heads(self, model: nn.Module):
        for name, mod in model.named_modules():
            if 'output_head' in name:
                for param_layer in mod.parameters():
                    param_layer.requires_grad = False

    def apply_eval_mask(self, model, task_idx: int):
        """
        Revert to network state for a specific task
        :param model: the model for this packnet
        :param task_idx: the task id to be evaluated (0 - > n_tasks)
        """
        # TODO: The weights are changed in-place, therefore this is irreversible, right?
        # (@lebrice): It would be nicer if you could either:
        # - mask out the weights temporarily using some sort of context manager OR
        # - Undo the masking before applying the mask for a different task ID than the
        #   current one.
        # 
        # Is this why you were saving / restoring the state of the
        # model below?
        # 
        # The context manager idea could be implemented with something like this:
        # import contextlib
        # @contextlib.context_manager
        # def temporary_version():
        #     # 1. Store the weights that will be masked out (as efficiently as possible, but
        #     # worst case is just copying all the current weights)
        #     # 2. Change weights in-place (mask them out)
        #     yield
        #     # 3. Restore the masked out weights
        assert len(self.masks) > task_idx

        mask_idx = 0
        with torch.no_grad():
            for mod_name, mod in model.named_modules():
                if (isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear)) and 'output_head' not in mod_name:
                    for name, param_layer in mod.named_parameters():
                        if 'bias' not in name:

                            # get indices of all weights from previous masks
                            prev_mask = torch.zeros_like(param_layer, dtype=torch.bool, requires_grad=False)
                            for i in range(0, task_idx + 1):
                                prev_mask |= self.masks[i][mask_idx]

                            # zero out all weights that are not in the mask for this task
                            param_layer *= prev_mask

                            mask_idx += 1

    def mask_remaining_params(self, model: nn.Module):
        """
        Create mask for remaining parameters
        """
        mask_idx = 0
        mask = []
        for mod_name, mod in model.named_modules():
            if (isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear)) and 'output_head' not in mod_name:
                for name, param_layer in mod.named_parameters():
                    if 'bias' not in name:
                        prev_mask = torch.zeros(param_layer.size(), dtype=torch.bool, requires_grad=False)
                        for task in self.masks:
                            prev_mask |= task[mask_idx]

                        # Create mask of remaining parameters
                        layer_mask = ~prev_mask
                        mask.append(layer_mask)

                        mask_idx += 1
        self.masks.append(mask)

    def total_epochs(self) -> int:
        return self.epoch_split[0] + self.epoch_split[1]

    def config_instructions(self):
        """
        Create pruning instructions for this task split
        :return: None
        """
        assert self.n_tasks is not None

        if not isinstance(self.prune_instructions, list):  # if a float is passed in
            assert 0 < self.prune_instructions < 1
            self.prune_instructions = [self.prune_instructions] * (self.n_tasks - 1)
        assert len(self.prune_instructions) == self.n_tasks - 1, "Must give prune instructions for each task"

    def save_final_state(self, model, PATH='model_weights.pth'):
        """
        Save the final weights of the model after training
        :param model: pl_module
        :param PATH: The path to weights file
        """
        self.PATH = PATH
        torch.save(model.state_dict(), PATH)

    def load_final_state(self, model):
        """

         Load the final state of the model
         """
        model.load_state_dict(torch.load(self.PATH))

    def on_init_end(self, trainer):
        self.mode = 'train'

    def on_after_backward(self, trainer: Trainer, pl_module: LightningModule):
        if self.mode == 'train':
            self.training_mask(pl_module)

        elif self.mode == 'fine_tune':
            self.fine_tune_mask(pl_module)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule, *args, **kwargs):

        if pl_module.current_epoch == self.epoch_split[0] - 1:  # Train epochs completed
            self.mode = 'fine_tune'
            if self.current_task == self.n_tasks - 1:
                self.mask_remaining_params(pl_module)
            else:
                self.prune(
                    model=pl_module,
                    prune_quantile=self.prune_instructions[self.current_task])

        elif pl_module.current_epoch == self.total_epochs() - 1:  # Train and fine tune epochs completed
            self.fix_biases(pl_module)  # Fix biases after first task
            self.fix_batch_norm(pl_module)  # Fix batch norm mean, var, and params
            # self.fix_output_heads(pl_module)
            self.save_final_state(pl_module)  # Not sure I like this here though:
            self.mode = 'train'


from sequoia.methods.trainer import TrainerConfig
from sequoia.common.config import Config
from sequoia.settings import Setting


@dataclass
class PackNetMethod(BaseMethod, target_setting=IncrementalSetting):
    # NOTE: these two fields are also used to create the command-line arguments.
    # HyperParameters of the method.
    hparams: BaseModel.HParams = mutable_field(BaseModel.HParams)
    # Configuration options.
    config: Config = mutable_field(Config)
    # Options for the Trainer object.
    trainer_options: TrainerConfig = mutable_field(TrainerConfig)
    # Hyper-Parameters of the PackNet callback
    packnet_hparams: PackNet.HParams = mutable_field(PackNet.HParams)

    def __init__(
            self,
            hparams: BaseModel.HParams = None,
            config: Config = None,
            trainer_options: TrainerConfig = None,
            packnet_hparams: PackNet.HParams = None,
            **kwargs,
    ):
        super().__init__(hparams=hparams, config=config, trainer_options=trainer_options)
        self.packnet_hparams = packnet_hparams or PackNet.HParams()
        self.p_net: PackNet  # This gets set in configure

    def configure(self, setting: Setting):
        # NOTE: super().configure creates the Trainer and calls `configure_callbacks()`,
        # so we have to create `self.p_net` before calling `super().configure`.
        self.p_net = PackNet(
            n_tasks=setting.nb_tasks,
            hparams=self.packnet_hparams,
        )

        self.p_net.current_task = -1
        self.p_net.config_instructions()
        super().configure(setting)

    def fit(self, train_env, valid_env):
        super().fit(train_env=train_env, valid_env=valid_env)

    def on_task_switch(self, task_id: Optional[int]) -> None:
        """Called when switching between tasks.
        
        Args:
            task_id (int, optional): the id of the new task. When None, we are
            basically being informed that there is a task boundary, but without
            knowing what task we're switching to.
        """
        super().on_task_switch(task_id=task_id)
        if task_id is not None and len(self.p_net.masks) > task_id:
            self.p_net.load_final_state(model=self.model)
            self.p_net.apply_eval_mask(task_idx=task_id, model=self.model)
        self.p_net.current_task = task_id

    def configure_callbacks(self, setting: TaskIncrementalSLSetting = None) -> List[Callback]:
        """Create the PyTorch-Lightning Callbacks for this Setting.

        These callbacks will get added to the Trainer in `create_trainer`.

        Parameters
        ----------
        setting : SettingType
            The `Setting` on which this Method is going to be applied.

        Returns
        -------
        List[Callback]
            A List of `Callback` objects to use during training.
        """
        callbacks = super().configure_callbacks(setting=setting)
        assert self.p_net not in callbacks
        if not setting.stationary_context:
            callbacks.append(self.p_net)
        return callbacks

    def create_trainer(self, setting) -> Trainer:
        """Creates a Trainer object from pytorch-lightning for the given setting.

        Args:

        Returns:
            Trainer: the Trainer object.
        """
        self.trainer_options.max_epochs = self.packnet_hparams.train_epochs + self.packnet_hparams.fine_tune_epochs
        return super().create_trainer(setting)

    def adapt_to_new_hparams(self, new_hparams: Dict[str, Any]) -> None:
        """Adapts the Method when it receives new Hyper-Parameters to try for a new run.

            It is required that this method be implemented if you want to perform HPO sweeps
            with Orion.

            Parameters
            ----------
            new_hparams : Dict[str, Any]
                The new hyper-parameters being recommended by the HPO algorithm. These will
                have the same structure as the search space.
        """
        self.hparams = self.hparams.replace(**new_hparams)
        self.packnet_hparams = self.packnet_hparams.replace(**new_hparams["packnet_hparams"])

    def get_search_space(self, setting: Setting) -> Mapping[str, Union[str, Dict]]:
        """Returns the search space to use for HPO in the given Setting.

        Parameters
        ----------
        setting : Setting
            The Setting on which the run of HPO will take place.

        Returns
        -------
        Mapping[str, Union[str, Dict]]
            An orion-formatted search space dictionary, mapping from hyper-parameter
            names (str) to their priors (str), or to nested dicts of the same form.
        """
        hparam_priors: Dict = super().get_search_space(setting=setting)
        hparam_priors["packnet_hparams"] = self.packnet_hparams.get_orion_space_dict()
        return hparam_priors
