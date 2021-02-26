"""Elastic Weight Consolidation as an Auxiliary Task.

This is a simplified version of EWC, that only currently uses the L2 norm, rather
than the Fisher Information Matrix.

TODO: If it's worth it, we could re-add the 'real' EWC using the nngeometry
package, (which I don't think we need to have as a submodule).
"""

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from typing import Type, Optional, Deque, List

from gym.spaces.utils import flatdim
from nngeometry.metrics import FIM
from nngeometry.object.pspace import PMatAbstract, PMatDiag, PMatKFAC, PVector
from simple_parsing import choice
from torch import Tensor, nn
from torch.utils.data import DataLoader

from sequoia.common.loss import Loss
from sequoia.common.hparams import uniform
from sequoia.methods.aux_tasks.auxiliary_task import AuxiliaryTask
from sequoia.methods.models.forward_pass import ForwardPass
from sequoia.methods.models.output_heads import ClassificationHead, RegressionHead
from sequoia.settings.base.objects import Observations
from sequoia.utils.logging_utils import get_logger
from sequoia.utils.utils import dict_intersection

logger = get_logger(__file__)


class EWCTask(AuxiliaryTask):
    """ Elastic Weight Consolidation, implemented as a 'self-supervision-style'
    Auxiliary Task.

    ```bibtex
    @article{kirkpatrick2017overcoming,
        title={Overcoming catastrophic forgetting in neural networks},
        author={Kirkpatrick, James and Pascanu, Razvan and Rabinowitz, Neil and Veness,
        Joel and Desjardins, Guillaume and Rusu, Andrei A and Milan, Kieran and Quan,
        John and Ramalho, Tiago and Grabska-Barwinska, Agnieszka and others},
        journal={Proceedings of the national academy of sciences},
        volume={114},
        number={13},
        pages={3521--3526},
        year={2017},
        publisher={National Acad Sciences}
    }
    ```
    """

    name: str = "ewc"

    @dataclass
    class Options(AuxiliaryTask.Options):
        """ Options of the EWC auxiliary task. """
        # Coefficient of the EWC auxilary task.
        # NOTE: It seems to be the case that, at least just for EWC, the coefficient
        # can be often be much greater than 1, hence why we overwrite the prior over
        # that hyper-parameter here.
        coefficient: float = uniform(0., 100., default=1.)
        # Batchsize to be used when computing FIM (unused atm)
        batch_size_fim: int = 64
        # Number of observations to use for FIM calculation
        sample_size_fim: int = 400
        # Fisher information representation type  (diagonal or block diagobnal).
        fim_representation: Type[PMatAbstract] = choice(
            {"diagonal": PMatDiag, "block_diagonal": PMatKFAC,}, default=PMatDiag,
        )

    def __init__(
        self, *args, name: str = None, options: "EWC.Options" = None, **kwargs
    ):
        super().__init__(*args, options=options, name=name, **kwargs)
        self.options: EWCTask.Options
        self.previous_task: Optional[int] = None
        self._i: int = 0
        self.n_switches: int = 0
        self.previous_model_weights: Optional[PVector] = None
        self.observation_collector: Deque[Observations] = deque(
            maxlen=self.options.sample_size_fim
        )
        self.fisher_information_matrices: List[PMatAbstract] = []

    def consolidate(self, new_fims: List[PMatAbstract], task: Optional[int]) -> None:
        """ Consolidates the new and current fisher information matrices.

        Parameters
        ----------
        new_fims : List[PMatAbstract]
            The list of new fisher information matrices.
        task : Optional[int]
            The id of the previous task, when task labels are available, or the number
            of task switches encountered so far when task labels are not available.
        """
        if not self.fisher_information_matrices:
            self.fisher_information_matrices = new_fims
            return

        if task is None:
            # Count the number of task switches, and use that as the task.
            task = self.n_switches

        for i, (fim_previous, fim_new) in enumerate(
            zip(self.fisher_information_matrices, new_fims)
        ):
            # consolidate the FIMs
            if fim_previous is None:
                self.fisher_information_matrices[i] = fim_new
            else:
                # consolidate the fim_new into fim_previous in place
                if isinstance(fim_new, PMatDiag):
                    # TODO: This is some kind of weird online-EWC related magic:
                    fim_previous.data = (
                        deepcopy(fim_new.data) + fim_previous.data * (task)
                    ) / (task + 1)

                elif isinstance(fim_new.data, dict):
                    # TODO: This is some kind of weird online-EWC related magic:
                    for _, (prev_param, new_param) in dict_intersection(
                        fim_previous.data, fim_new.data
                    ):
                        for prev_item, new_item in zip(prev_param, new_param):
                            prev_item.data = (
                                prev_item.data * task + deepcopy(new_item.data)
                            ) / (task + 1)

                self.fisher_information_matrices[i] = fim_previous

    def on_task_switch(self, task_id: Optional[int]):
        """ Executed when the task switches (to either a known or unknown task).
        """
        if not self.enabled:
            return

        logger.info(f"On task switch called: task_id={task_id}")

        if self._shared_net is None:
            logger.info(
                f"On task switch called: task_id={task_id}, EWC cannot be "
                f"applied as there are no shared weights."
            )

        elif self.previous_task is None and self.n_switches == 0 and not task_id:
            self.previous_task = task_id
            logger.info("Starting the first task, no EWC update.")
            self.n_switches += 1

        elif self._model.training:
            calculate_FIM = False
            if task_id is None and self.previous_task is None:
                #setting without task IDs, still calculate FIM
                calculate_FIM = True
            elif task_id > self.previous_task:
                calculate_FIM = True
            else:
                raise NotImplementedError

            if calculate_FIM:
                # we dont want to go here at test time.
                # NOTE: We also switch between unknown tasks.
                logger.info(
                    f"Switching tasks: {self.previous_task} -> {task_id}: "
                    f"Updating the EWC 'anchor' weights."
                )
                self.previous_task = task_id
                device = self._model.config.device
                self.previous_model_weights = (
                    PVector.from_model(self._shared_net.to(device)).clone().detach()
                )

                # Create a Dataloader from the stored observations.
                obs_type: Type[Observations] = type(self.observation_collector[0])
                dataset = [obs.as_namedtuple() for obs in self.observation_collector]
                # Or, alternatively (see the note below on why we don't use this):
                # stacked_observations: Observations = obs_type.stack(self.observation_collector)
                # dataset = TensorDataset(*stacked_observations.as_namedtuple())

                # NOTE: This is equivalent to just using the same batch size as during
                # training, as each Observations in the list is already a batch.
                # NOTE: We keep the same batch size here as during training because for
                # instance in RL, it would be weird to suddenly give some new batch size,
                # since the buffers would get cleared and re-created just for these forward
                # passes
                dataloader = DataLoader(dataset, batch_size=None, collate_fn=None)

                # Create the parameters to be passed to the FIM function. These may vary a
                # bit, depending on if we're being applied in a classification setting or in
                # a regression setting (not done yet)
                variant: str
                if isinstance(self._model.output_head, ClassificationHead):
                    variant = "classif_logits"
                    n_output = self._model.action_space.n

                    def fim_function(*inputs) -> Tensor:
                        observations = obs_type(*inputs).to(self._model.device)
                        forward_pass: ForwardPass = self._model(observations)
                        actions = forward_pass.actions
                        return actions.logits

                elif isinstance(self._model.output_head, RegressionHead):
                    # NOTE: This hasn't been tested yet.
                    variant = "regression"
                    n_output = flatdim(self._model.action_space)

                    def fim_function(*inputs) -> Tensor:
                        observations = obs_type(*inputs).to(self._model.device)
                        forward_pass: ForwardPass = self._model(observations)
                        actions = forward_pass.actions
                        return actions.y_pred

                else:
                    raise NotImplementedError("TODO")

                new_fim = FIM(
                    model=self._shared_net,
                    loader=dataloader,
                    representation=self.options.fim_representation,
                    n_output=n_output,
                    variant=variant,
                    function=fim_function,
                    device=self._model.device,
                )

                # TODO: There was maybe an idea to use another fisher information matrix for
                # the critic in A2C, but not doing that atm.
                new_fims = [new_fim]
                self.consolidate(new_fims, task=self.previous_task)
                self.n_switches += 1
                self.observation_collector.clear()

    @property
    def _shared_net(self) -> Optional[nn.Module]:
        """
        Returns 'None' if there is not shared network part, othervise returns the shared net
        """
        if self._model.encoder is None:
            return None
        elif isinstance(self._model.encoder, nn.Sequential):
            if len(self._model.encoder) == 0:
                return None
        return self._model.encoder

    def get_loss(self, forward_pass: ForwardPass, y: Tensor = None) -> Loss:
        """ Gets the EWC loss.
        """
        if self._model.training:      
            self.observation_collector.append(forward_pass.observations)

        if self.previous_task is None or not self.enabled or self._shared_net is None:
            # We're in the first task: do nothing.
            return Loss(name=self.name)

        loss = 0.0
        v_current = PVector.from_model(self._shared_net)

        for fim in self.fisher_information_matrices:
            diff = v_current - self.previous_model_weights
            loss += fim.vTMv(diff)
        self._i += 1
        ewc_loss = Loss(name=self.name, loss=loss)
        return ewc_loss
