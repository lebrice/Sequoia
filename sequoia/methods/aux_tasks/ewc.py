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
from contextlib import contextmanager

from gym.spaces.utils import flatdim
from nngeometry.metrics import FIM
from nngeometry.object.pspace import PMatAbstract, PMatDiag, PMatKFAC, PVector
from simple_parsing import choice
from torch import Tensor
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
        coefficient: float = uniform(0.0, 100.0, default=1.0)
        # Batchsize to be used when computing FIM (unused atm)
        batch_size_fim: int = 64
        # Number of observations to use for FIM calculation
        sample_size_fim: int = 400
        # Fisher information representation type  (diagonal or block diagobnal).
        fim_representation: Type[PMatAbstract] = choice(
            {"diagonal": PMatDiag, "block_diagonal": PMatKFAC}, default=PMatDiag,
        )

    def __init__(
        self, *args, name: str = None, options: "EWCTask.Options" = None, **kwargs
    ):
        super().__init__(*args, options=options, name=name, **kwargs)
        self.options: EWCTask.Options

        # The id of the current/most recent task the model has been trained on.
        self.current_training_task: Optional[int] = None
        # The id of the previous task the model was trained on.
        self.previous_training_task: Optional[int] = None
        # The ids of all the tasks trained on so far, not including the current task.
        self.previous_training_tasks: List[Optional[int]] = []

        self.previous_model_weights: Optional[PVector] = None
        self.observation_collector: Deque[Observations] = deque(
            maxlen=self.options.sample_size_fim
        )
        self.fisher_information_matrices: List[PMatAbstract] = []
        # When True, ignore task boundaries (no EWC update).
        # This is used mainly because of the need for executing forward passes when
        # calculating the new FIMs, and the MultiheadModel class might then call
        # `on_task_switch`, so we don't want to recurse.
        self._ignore_task_boundaries: bool = False

        if not self.model.shared_modules():
            # TODO: This might cause a bug, if  some auxiliary task were to replace the
            # encoder and also be 'activated' after this task. This is a really obscure
            # edge case though.
            logger.warning(
                RuntimeWarning(
                    "Disabling the EWC auxiliary task, since there appears to be no "
                    "shared weights between tasks!"
                )
            )
            self.disable()

    def get_loss(self, forward_pass: ForwardPass, y: Tensor = None) -> Loss:
        """ Gets the EWC loss.
        """
        if self.training:
            self.observation_collector.append(forward_pass.observations)

        if not self.enabled or self.previous_model_weights is None:
            # We're in the first task: do nothing.
            return Loss(name=self.name)

        loss = 0.0
        v_current = self.get_current_model_weights()

        for fim in self.fisher_information_matrices:
            diff = v_current - self.previous_model_weights
            loss += fim.vTMv(diff)

        ewc_loss = Loss(name=self.name, loss=loss)
        return ewc_loss

    def on_task_switch(self, task_id: Optional[int]):
        """ Executed when the task switches (to either a known or unknown task).
        """
        if not self.enabled:
            return
        logger.debug(f"On task switch called: task_id={task_id}")

        if self._ignore_task_boundaries:
            logger.info("Ignoring task boundary (probably from recursive call)")
            return

        if not self.training:
            logger.debug("Task boundary at test time, no EWC update.")
            return
        # Two cases:
        # - Setting without task IDs --> still calculate the FIMs at each task boundary.
        # - Setting with IDs --> calculate the FIMs before training on new tasks.

        # Setting without task labels. Task ids: None -> None -> None  (always None)
        if task_id is None:
            # Here we use the number of task boundaries as a 'fake' task id, meaning we
            # treat each task as if it has never been encountered before.
            if self.current_training_task is None:
                # Start of first task, no EWC update.
                self.current_training_task = 0
            else:
                self.previous_training_task = self.current_training_task
                self.current_training_task += 1
                self.update_anchor_weights(new_task_id=self.current_training_task)

        # Setting with task labels. Task ids: 0 -> 1 -> 2 -> 1 -> 3 -> 5 -> 11 -> 5 etc.
        else:
            if self.current_training_task is None:
                logger.info("Starting the first task, no EWC update.")
                self.current_training_task = task_id
            elif task_id == self.current_training_task:
                logger.info("Switching to same task, no EWC update.")
            elif task_id in self.previous_training_tasks:
                logger.info(f"Switching to known task {task_id}, no EWC update.")
            else:
                logger.info(f"Switching to new task {task_id}, updating EWC params.")
                self.previous_training_task = self.current_training_task
                self.previous_training_tasks.append(self.current_training_task)
                self.current_training_task = task_id
                self.update_anchor_weights(new_task_id=self.current_training_task)

    def update_anchor_weights(self, new_task_id: int) -> None:
        """Update the FIMs and other EWC params before starting training on a new task.

        Parameters
        ----------
        new_task_id : int
            The ID of the new task.
        """
        # we dont want to go here at test time.
        # NOTE: We also switch between unknown tasks.
        logger.info(
            f"Updating the EWC 'anchor' weights before starting training on "
            f"task {new_task_id}"
        )
        self.previous_model_weights = self.get_current_model_weights().clone().detach()

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
        # TODO: Would be nice to have a progress bar here.

        # Create the parameters to be passed to the FIM function. These may vary a
        # bit, depending on if we're being applied in a classification setting or in
        # a regression setting (not done yet)
        variant: str
        # TODO: Change this conditional to be based on the type of action space, rather
        # than of output head.
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

        with self._ignoring_task_boundaries():
            # Prevent recursive calls to `on_task_switch` from affecting us (can be
            # called from MultiheadModel). (TODO: MultiheadModel will be fixed soon.)
            # layer_collection = LayerCollection.from_model(self.model.shared_modules())
            # nngeometry BUG: this doesn't work when passing the layer
            # collection instead of the model
            new_fim = FIM(
                model=self.model.shared_modules(),
                loader=dataloader,
                representation=self.options.fim_representation,
                n_output=n_output,
                variant=variant,
                function=fim_function,
                device=self._model.device,
                layer_collection=None,
            )

        # TODO: There was maybe an idea to use another fisher information matrix for
        # the critic in A2C, but not doing that atm.
        new_fims = [new_fim]
        self.consolidate(new_fims, task=new_task_id)
        self.observation_collector.clear()

    @contextmanager
    def _ignoring_task_boundaries(self):
        """ Contextmanager used to temporarily ignore task boundaries (no EWC update).
        """
        self._ignore_task_boundaries = True
        yield
        self._ignore_task_boundaries = False

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

        assert task is not None, "Should have been given an int task id (even if fake)."

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

    def get_current_model_weights(self) -> PVector:
        return PVector.from_model(self.model.shared_modules())
