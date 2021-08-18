from torch import nn
from sequoia.methods.base_method import BaseMethod
from sequoia.settings.sl import TaskIncrementalSLSetting
from pytorch_lightning.callbacks import EarlyStopping
import torch
from torch import Tensor
from simple_parsing.helpers import mutable_field
from sequoia.methods.base_method import BaseModel
from dataclasses import dataclass
from pytorch_lightning import LightningModule, Trainer, Callback
from simple_parsing.helpers.hparams import HyperParameters, uniform
from typing import (
    Union,
    List,
    Optional,
    Mapping,
    Dict,
    Any,
    Type,
    Sequence,
    Iterable,
    Tuple,
)

from sequoia.settings.assumptions import IncrementalAssumption as IncrementalSetting


class PackNet(Callback, nn.Module):
    """PyTorch-Lightning Callback that implements the PackNet algorithm for CL.

    TODO: Add a citation for the PackNet paper.
    """

    @dataclass
    class HParams(HyperParameters):
        """Hyper-parameters of the Packnet callback."""

        prune_instructions: Union[float, List[float]] = uniform(0.1, 0.9, default=0.5)

        train_epochs: int = uniform(1, 5, default=1)
        fine_tune_epochs: int = uniform(0, 5, default=1)

    def __init__(
            self,
            n_tasks: int,
            hparams: Optional["PackNet.HParams"]=None,
            prunable_types: Sequence[Type[nn.Module]] = (nn.Conv2d, nn.Linear),
            ignore_modules: Sequence[str] = None,
            ignore_parameters: Sequence[str] = ("bias",),
    ):
        """Create the PackNet callback.

        Parameters
        ----------
        n_tasks : int
            Number of tasks.
        hparams : PackNet.HParams
            Configuration options (hyper-parameters) of the PackNet algorithm.
        prunable_types : Sequence[Type[nn.Module]], optional
            The types of nn.Modules to consider for pruning. By default, only consideres
            layers of types `nn.Conv2d` and `nn.Linear`.
        ignore_modules : Sequence[str], optional
            List of flags for module names that should be ignored by PackNet.
            When one of these values is found within the name of a module, it is
            ignored. Doesn't ignore any modules by default.
        parameters_to_ignore : List[str], optional
            List of flags for parameter names that should be ignored by PackNet.
            When one of these values is found within the name of a parameter, it is
            ignored. Defaults to ["bias"].
        """
        super().__init__()
        hparams = hparams or self.HParams()
        self.n_tasks = n_tasks
        self.prune_instructions = hparams.prune_instructions
        self.prunable_types = prunable_types or [nn.Conv2d, nn.Linear]
        self.ignore_modules = list(ignore_modules or [])
        self.ignore_parameters = list(ignore_parameters or [])
        # Set up an array of quantiles for pruning procedure
        if n_tasks:
            self.config_instructions()

        self.PATH = None
        self.epoch_split = (hparams.train_epochs, hparams.fine_tune_epochs)
        self.current_task = 0
        # 3-dimensions: task, layer, parameter mask
        self.masks: List[Dict[str, Tensor]] = []
        self.mode: str = None
        self.params_dict: dict = None

    def filtered_parameter_iterator(
            self, module: nn.Module
    ) -> Iterable[Tuple[str, nn.Parameter]]:
        """Iterator that, given a module, yields tuples with the full name of the
        parameters that will be modified by the PackNet callback, as well as the
        parameters themselves.

        This is used to remove a bit of boilerplate code in the for loops below.

        Parameters
        ----------
        module : nn.Module
            The module to iterate over.

        Returns
        -------
        Iterable[Tuple[str, nn.Parameter]]
            An Iterator of tuples containing parameter names ('{mod_name}.{param_name}')
            and parameters.
        """
        for mod_name, mod in module.named_modules():
            if not isinstance(mod, self.prunable_types):
                continue
            if any(ignored in mod_name for ignored in self.ignore_modules):
                continue
            for param_name, param in mod.named_parameters():
                if any(ignored in param_name for ignored in self.ignore_parameters):
                    continue

                param_full_name = f"{mod_name}.{param_name}"
                yield param_full_name, param

    @torch.no_grad()
    def prune(self, model: nn.Module, prune_quantile: float) -> Dict[str, Tensor]:
        """Create task-specific mask and prune least relevant weights

        [extended_summary]

        Parameters
        ----------
        model : nn.Module
            The model to be pruned.
        prune_quantile : float
            The percentage of weights to prune as a decimal.

        Returns
        -------
        Dict[str, Tensor]
            The masks to use to prune the layers of the given model.
        """
        # Calculate Quantile
        all_prunable_tensors: List[Tensor] = []

        for param_full_name, param_layer in self.filtered_parameter_iterator(model):
            # get fixed weights for this layer (on the same device)
            prev_mask = torch.zeros_like(param_layer, dtype=torch.bool)

            for task_masks in self.masks:
                if param_full_name in task_masks:
                    prev_mask |= task_masks[param_full_name]

            p = param_layer.masked_select(~prev_mask)

            if p is not None:
                all_prunable_tensors.append(p)

        all_parameters_tensor = torch.cat(all_prunable_tensors, -1)
        cutoff = torch.quantile(torch.abs(all_parameters_tensor), q=prune_quantile)

        masks = {}  # create mask for this task
        for param_full_name, param_layer in self.filtered_parameter_iterator(model):
            # get weight mask for this layer
            # p
            prev_mask = torch.zeros_like(param_layer, dtype=torch.bool)

            for task_masks in self.masks:
                # TODO: check for bug here
                # if param_full_name in task_masks:
                prev_mask |= task_masks[param_full_name]

            curr_mask = torch.abs(param_layer).ge(cutoff)  # q
            curr_mask &= ~prev_mask  # (q & ~p)

            # Zero non masked weights
            param_layer *= curr_mask | prev_mask

            masks[param_full_name] = curr_mask

        return masks

    def fine_tune_mask(self, model: nn.Module):
        """
        Zero the gradient of pruned weights this task as well as previously fixed weights
        Apply this mask before each optimizer step during fine-tuning
        """
        assert len(self.masks) > self.current_task
        for param_full_name, param in self.filtered_parameter_iterator(model):
            param.grad *= self.masks[self.current_task][param_full_name]

    def training_mask(self, model: nn.Module):
        """
        Zero the gradient of only fixed weights for previous tasks
        Apply this mask after .backward() and before
        optimizer.step() at every batch of training a new task
        """
        if len(self.masks) == 0:
            return

        for param_full_name, param in self.filtered_parameter_iterator(model):
            # get mask of weights from previous tasks
            prev_mask = torch.zeros_like(param, dtype=torch.bool)

            for task_masks in self.masks:
                # FIXME: Get the mask if it exists, otherwise set one and move on.
                # if param_full_name not in task_masks:
                #     task_masks[param_full_name] = torch.zeros_like(param, dtype=torch.bool)
                prev_mask |= task_masks[param_full_name]

            # zero grad of previous fixed weights
            # param.grad[prev_mask] = 0. # (NOTE: Equivalent)
            param.grad *= ~prev_mask

    def fix_biases(self, model: nn.Module):
        """
        Fix the gradient of prunable bias parameters
        """
        for mod_name, mod in model.named_modules():
            if not isinstance(mod, self.prunable_types):
                continue
            if any(ignore in mod_name for ignore in self.ignore_modules):
                continue
            for name, param_layer in mod.named_parameters():
                if "bias" in name:
                    param_layer.requires_grad = False

    def fix_batch_norm(self, model: nn.Module):
        """
        Fix batch norm gain, bias, running mean and variance
        """
        for mod_name, mod in model.named_modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.affine = False
                for param_layer in mod.parameters():
                    param_layer.requires_grad = False

    def set_params_dict(self, model: nn.Module):
        """
        Set a dictionary containing all prunable parameters
        useful for fixing all layers, but may be wasted memory
        """
        # TODO: This dict actually doesn't copy the parameters, it saves references.
        self.params_dict = dict()
        for param_full_name, param in self.filtered_parameter_iterator(model):
            self.params_dict[param_full_name] = param

    def fix_all_layers(self, model: nn.Module):
        """
        Fix grad of all parameters outside of params_dict
        """
        self.set_params_dict(model)  # Not necessary for fixed model

        # Fix grad of all non-prunable layers in this
        for mod_name, mod in model.named_modules():
            for param_name, param_layer in mod.named_parameters():
                key = f"{mod_name}.{param_name}"
                if key not in self.params_dict:
                    param_layer.requires_grad = False

    @torch.no_grad()
    def apply_eval_mask(self, model: nn.Module, task_idx: int):
        """
        Revert to final trained network state and apply mask for given task
        :param model: the model to apply the eval mask to
        :param task_idx: the task id to be evaluated (0 - > n_tasks)
        """

        assert len(self.masks) > task_idx
        for param_full_name, param in self.filtered_parameter_iterator(model):
            # get indices of all weights from previous masks
            prev_mask = torch.zeros_like(param, dtype=torch.bool)
            for task_id in range(0, task_idx + 1):
                prev_mask |= self.masks[task_id][param_full_name]

            # zero out all weights that are not in the mask for this task
            # param[prev_mask] = 0. (NOTE: Equivalent)
            param *= prev_mask

    def mask_remaining_params(self, model: nn.Module) -> Dict[str, Tensor]:
        """
        Create mask for remaining parameters
        """
        masks = {}
        for param_full_name, param in self.filtered_parameter_iterator(model):
            # Get mask of all weights assigned to previous tasks
            prev_mask = torch.zeros_like(param, dtype=torch.bool)
            for task_masks in self.masks:
                prev_mask |= task_masks[param_full_name]
            # Create mask of remaining parameters
            layer_mask = ~prev_mask
            masks[param_full_name] = layer_mask
        return masks
        # self.masks.append(mask)

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
        assert (
                len(self.prune_instructions) == self.n_tasks - 1
        ), "Must give prune instructions for every task"

    def save_final_state(self, model, PATH="model_weights.pth"):
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

    def on_init_end(self, trainer: Trainer):
        self.mode = "train"

    def on_after_backward(self, trainer: Trainer, pl_module: LightningModule):
        if self.mode == "train":
            self.training_mask(pl_module)

        elif self.mode == "fine_tune":
            self.fine_tune_mask(pl_module)

    def on_train_epoch_end(
            self, trainer: Trainer, pl_module: LightningModule, *args, **kwargs
    ):
        super().on_train_epoch_end(trainer, pl_module)
        if pl_module.current_epoch == self.epoch_split[0] - 1:  # Train epochs completed
            self.mode = "fine_tune"
            new_masks: Dict[str, Tensor]
            if self.current_task == self.n_tasks - 1:
                new_masks = self.mask_remaining_params(pl_module)
            else:
                new_masks = self.prune(
                    model=pl_module,
                    prune_quantile=self.prune_instructions[self.current_task],
                )
            self.masks.append(new_masks)

        elif (
                pl_module.current_epoch == self.total_epochs() - 1
        ):  # Train and fine tune epochs completed
            self.fix_biases(pl_module)  # Fix biases after first task
            self.fix_batch_norm(pl_module)  # Fix batch norm mean, var, and params

            # TODO: This may cause issues with output heads
            # self.fix_all_layers(pl_module)  # Fix all other layers -> may not be necessary?

<<<<<<< HEAD
            self.save_final_state(self.model)
=======
            self.save_final_state(pl_module)
>>>>>>> Fix min_epochs issue with pl_example_packnet
            self.mode = "train"


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
        super().__init__(
            hparams=hparams, config=config, trainer_options=trainer_options
        )
        self.packnet_hparams = packnet_hparams or PackNet.HParams()
        self.p_net: PackNet  # This gets set in configure

    def configure(self, setting: Setting):
        # NOTE: super().configure creates the Trainer and calls `configure_callbacks()`,
        # so we have to create `self.p_net` before calling `super().configure`.

        # Ignore all the modules that are task-specific when the setting gives task ids:
        # NOTE: Always ignore the `output_heads` dict, as it contains output heads for
        # each task.
        # NOTE: `model.output_heads[<current_task>]` is the same as `model.output_head`.
        ignored_modules: List[str] = ["output_heads"]
        if setting.task_labels_at_test_time:
            # Also ignore the main output_head.
            ignored_modules.append("output_head")

        self.p_net = PackNet(
            n_tasks=setting.nb_tasks,
            hparams=self.packnet_hparams,
            ignore_modules=ignored_modules,
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

    def configure_callbacks(
            self, setting: TaskIncrementalSLSetting = None
    ) -> List[Callback]:
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

        for i in range(len(callbacks)):
            if isinstance(callbacks[i], EarlyStopping):
                callbacks.pop(i)
        print(callbacks)
        if not setting.stationary_context:
            callbacks.append(self.p_net)
        return callbacks

    def create_trainer(self, setting) -> Trainer:
        """Creates a Trainer object from pytorch-lightning for the given setting.
        Returns:
            Trainer: the Trainer object.
        """
        self.trainer_options.max_epochs = (
                self.packnet_hparams.train_epochs + self.packnet_hparams.fine_tune_epochs
        )

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
        self.packnet_hparams = self.packnet_hparams.replace(
            **new_hparams["packnet_hparams"]
        )

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
