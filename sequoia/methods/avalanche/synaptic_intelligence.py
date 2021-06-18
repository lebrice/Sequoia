""" Method based on SynapticIntelligence from [Avalanche](https://github.com/ContinualAI/avalanche).

See `avalanche.training.plugins.synaptic_intelligence.SynapticIntelligencePlugin` or
`avalanche.training.strategies.strategy_wrappers.SynapticIntelligence` for more info.
"""
from dataclasses import dataclass
from typing import ClassVar, Optional, Set, Type

import torch
from avalanche.training.plugins.synaptic_intelligence import (
    ParamDict,
    EwcDataType,
    SynDataType,
)
from avalanche.training.plugins.synaptic_intelligence import (
    SynapticIntelligencePlugin as SynapticIntelligencePlugin_,
)
from avalanche.training.strategies import BaseStrategy, SynapticIntelligence
from simple_parsing import ArgumentParser
from simple_parsing.helpers.hparams import uniform
from torch.nn import Module
from torch import Tensor
import numpy as np

from sequoia.methods import register_method
from sequoia.settings.sl import (
    ClassIncrementalSetting,
    SLSetting,
    TaskIncrementalSLSetting,
)
from .base import AvalancheMethod


class SynapticIntelligencePlugin(SynapticIntelligencePlugin_):
    # TODO: Why do they have everything as a static method rather than as a classmethod?
    # Makes it almost impossible to extend this SynapticIntelligencePlugin!
    @staticmethod
    @torch.no_grad()
    def extract_weights(
        model: Module, target: ParamDict, excluded_parameters: Set[str]
    ):
        params = SynapticIntelligencePlugin_.allowed_parameters(
            model, excluded_parameters
        )
        # Getting this error:
        # RuntimeError: The expanded size of the tensor (128) must match the existing
        # size (256) at non-singleton dimension 0.  Target sizes: [128].
        # Tensor sizes: [256]
        # TODO: @lebrice For now I'll just replace the entries in that 'target' dict if
        # the shapes don't match, and hope it still works.
        for name, param in params:
            # target[name][...] = param.detach().cpu().flatten()
            if param.shape == target[name].shape:
                target[name][...] = param.detach().cpu().flatten()
            else:
                # Replace the entries with a different shape, rather than replacing their data
                # as done above?
                target[name].data = param.detach().cpu().flatten()

    @staticmethod
    @torch.no_grad()
    def extract_grad(model, target: ParamDict, excluded_parameters: Set[str]):
        params = SynapticIntelligencePlugin_.allowed_parameters(
            model, excluded_parameters
        )

        # Store the gradients into target
        for name, param in params:
            # BUG: Getting AttributeError: 'NoneType' object has no attribute 'detach'
            if param.grad is not None:
                target[name][...] = param.grad.detach().cpu().flatten()

    @staticmethod
    def compute_ewc_loss(
        model, ewc_data: EwcDataType, excluded_parameters: Set[str], device, lambd=0.0
    ):
        params = SynapticIntelligencePlugin_.allowed_parameters(
            model, excluded_parameters
        )

        loss = None
        for name, param in params:
            weights = param.to(device).flatten()  # Flat, not detached
            param_ewc_data_0 = ewc_data[0][name].to(device)  # Flat, detached
            param_ewc_data_1 = ewc_data[1][name].to(device)  # Flat, detached

            # BUG: Getting RuntimeError: inconsistent tensor size, expected tensor [128]
            # and src [256] to have the same number of elements, but got 128 and 256
            # elements respectively
            if param_ewc_data_1.shape == param_ewc_data_0.shape == weights.shape:
                syn_loss: Tensor = torch.dot(
                    param_ewc_data_1, (weights - param_ewc_data_0) ** 2
                ) * (lambd / 2)
            else:
                # FIXME: For now, I'll just consider the 'common' elements?
                param_0_cols = param_ewc_data_0.shape[-1]
                param_1_cols = param_ewc_data_1.shape[-1]
                # Weird: why does param_0 have *more* columns than param_1?
                assert param_0_cols > param_1_cols
                # Assuming that the first indices are the common weights between tasks:
                param_ewc_data_0 = param_ewc_data_0[..., :param_1_cols]
                weights = weights[..., :param_1_cols]

                syn_loss: Tensor = torch.dot(
                    param_ewc_data_1, (weights - param_ewc_data_0) ** 2
                ) * (lambd / 2)

            if loss is None:
                loss = syn_loss
            else:
                loss += syn_loss

        return loss

    @staticmethod
    @torch.no_grad()
    def post_update(model, syn_data: SynDataType, excluded_parameters: Set[str]):
        SynapticIntelligencePlugin_.extract_weights(
            model, syn_data["new_theta"], excluded_parameters
        )
        SynapticIntelligencePlugin_.extract_grad(
            model, syn_data["grad"], excluded_parameters
        )

        for param_name in syn_data["trajectory"]:
            # BUG: Getting RuntimeError: The size of tensor a (128) must match the size
            # of tensor b (256) at non-singleton dimension 0
            # syn_data['trajectory'][param_name] += \
            #     syn_data['grad'][param_name] * (
            #             syn_data['new_theta'][param_name] -
            #             syn_data['old_theta'][param_name])
            destination: Tensor = syn_data["trajectory"][param_name]
            grad: Tensor = syn_data["grad"][param_name]
            new_theta: Tensor = syn_data["new_theta"][param_name]
            old_theta: Tensor = syn_data["old_theta"][param_name]
            if not (
                destination.shape == grad.shape == new_theta.shape == old_theta.shape
            ):
                destination_cols = destination.shape[-1]
                grad_cols = grad.shape[-1]
                new_theta_cols = new_theta.shape[-1]
                old_theta_cols = old_theta.shape[-1]
                assert grad_cols < new_theta_cols and new_theta_cols == old_theta_cols
                # FIXME: @lebrice Chop the last two? or extend the grad? Extending the
                # grad with zeros for now (no idea what that implies though!)
                grad_extension = grad.new_zeros(
                    size=[*grad.shape[:-1], new_theta_cols - grad_cols]
                )
                grad = torch.cat([grad, grad_extension], -1)

                destination_extension = destination.new_zeros(
                    size=[*destination.shape[:-1], new_theta_cols - destination_cols]
                )
                destination = torch.cat([destination, destination_extension], -1)

            assert destination.shape == grad.shape == new_theta.shape == old_theta.shape
            destination += grad * (new_theta - old_theta)
            # Replace the entry (in case we replaced the `destination` variable above).
            syn_data["trajectory"][param_name] = destination

    @staticmethod
    @torch.no_grad()
    def update_ewc_data(
        net,
        ewc_data: EwcDataType,
        syn_data: SynDataType,
        clip_to: float,
        excluded_parameters: Set[str],
        c=0.0015,
    ):
        SynapticIntelligencePlugin.extract_weights(
            net, syn_data["new_theta"], excluded_parameters
        )
        eps = 0.0000001  # 0.001 in few task - 0.1 used in a more complex setup

        for param_name in syn_data["cum_trajectory"]:
            # BUG: Getting RuntimeError: The size of tensor a (128) must match the size
            # of tensor b (256) at non-singleton dimension 0
            # syn_data['cum_trajectory'][param_name] += \
            #     c * syn_data['trajectory'][param_name] / (
            #             np.square(syn_data['new_theta'][param_name] -
            #                       ewc_data[0][param_name]) + eps)
            cum_trajectory = syn_data["cum_trajectory"][param_name]
            trajectory = syn_data["trajectory"][param_name]
            new_theta = syn_data["new_theta"][param_name]
            ewc_data_0 = ewc_data[0][param_name]

            if not (
                cum_trajectory.shape
                == trajectory.shape
                == new_theta.shape
                == ewc_data_0.shape
            ):
                cum_trajectory_cols = cum_trajectory.shape[-1]
                trajectory_cols = trajectory.shape[-1]
                new_theta_cols = new_theta.shape[-1]
                ewc_data_0_cols = ewc_data_0.shape[-1]
                assert (
                    cum_trajectory_cols
                    < trajectory_cols
                    == new_theta_cols
                    == ewc_data_0_cols
                )

                # FIXME: @lebrice Extending the cum_trajectory with zeros for now (no
                # idea what that implies though!)
                cum_trajectory_extension = cum_trajectory.new_zeros(
                    size=[
                        *cum_trajectory.shape[:-1],
                        trajectory_cols - cum_trajectory_cols,
                    ]
                )
                cum_trajectory = torch.cat(
                    [cum_trajectory, cum_trajectory_extension], -1
                )

            cum_trajectory += c * trajectory / (np.square(new_theta - ewc_data_0) + eps)
            # Reset the cum_trajectory variable in the dict, just in case we replaced
            # the variable above.
            syn_data["cum_trajectory"][param_name] = cum_trajectory

        for param_name in syn_data["cum_trajectory"]:
            ewc_data[1][param_name] = torch.empty_like(
                syn_data["cum_trajectory"][param_name]
            ).copy_(-syn_data["cum_trajectory"][param_name])

        # change sign here because the Ewc regularization
        # in Caffe (theta - thetaold) is inverted w.r.t. syn equation [4]
        # (thetaold - theta)
        for param_name in ewc_data[1]:
            ewc_data[1][param_name] = torch.clamp(ewc_data[1][param_name], max=clip_to)
            ewc_data[0][param_name] = syn_data["new_theta"][param_name].clone()


# TODO: Why do they have everything as a static method rather than as a classmethod?
# Makes it almost impossible to extend this SynapticIntelligencePlugin!
SynapticIntelligencePlugin_.extract_weights = SynapticIntelligencePlugin.extract_weights
SynapticIntelligencePlugin_.extract_grad = SynapticIntelligencePlugin.extract_grad
SynapticIntelligencePlugin_.compute_ewc_loss = (
    SynapticIntelligencePlugin.compute_ewc_loss
)
SynapticIntelligencePlugin_.post_update = SynapticIntelligencePlugin.post_update
SynapticIntelligencePlugin_.update_ewc_data = SynapticIntelligencePlugin.update_ewc_data


@register_method
@dataclass
class SynapticIntelligenceMethod(AvalancheMethod[SynapticIntelligence]):
    """ The Synaptic Intelligence strategy from Avalanche.

    This is the Synaptic Intelligence PyTorch implementation of the
    algorithm described in the paper
    "Continuous Learning in Single-Incremental-Task Scenarios"
    (https://arxiv.org/abs/1806.08568)

    The original implementation has been proposed in the paper
    "Continual Learning Through Synaptic Intelligence"
    (https://arxiv.org/abs/1703.04200).

    The Synaptic Intelligence regularization can also be used in a different
    strategy by applying the :class:`SynapticIntelligencePlugin` plugin.

    See the parent class `AvalancheMethod` for the other hyper-parameters and methods.
    """

    # Synaptic Intelligence lambda term.
    si_lambda: float = uniform(1e-2, 1.0, default=0.5)  # TODO: Check the range.

    strategy_class: ClassVar[Type[BaseStrategy]] = SynapticIntelligence

    def create_cl_strategy(self, setting: SLSetting) -> SynapticIntelligence:
        strategy = super().create_cl_strategy(setting)

        # Find and replace the original plugin with our "patched" version:
        plugin_index: Optional[int] = None
        for i, plugin in enumerate(strategy.plugins):
            if type(plugin) is SynapticIntelligencePlugin_:
                plugin_index = i
                break
        assert plugin_index is not None, "strategy should have the Plugin, no?"
        assert isinstance(plugin_index, int)

        old_plugin: SynapticIntelligencePlugin_ = strategy.plugins[plugin_index]
        new_plugin = SynapticIntelligencePlugin(
            si_lambda=old_plugin.si_lambda,
            excluded_parameters=old_plugin.excluded_parameters,
            # device=old_plugin.device,
        )
        new_plugin.ewc_data = old_plugin.ewc_data
        new_plugin.syn_data = old_plugin.syn_data
        new_plugin._device = old_plugin._device

        strategy.plugins[plugin_index] = new_plugin
        return strategy


if __name__ == "__main__":

    setting = TaskIncrementalSLSetting(
        dataset="mnist", nb_tasks=5, monitor_training_performance=True
    )
    # Create the Method, either manually or through the command-line:
    parser = ArgumentParser(__doc__)
    parser.add_arguments(SynapticIntelligenceMethod, "method")
    args = parser.parse_args()
    method: SynapticIntelligenceMethod = args.method

    results = setting.apply(method)
