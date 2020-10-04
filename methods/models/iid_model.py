from dataclasses import dataclass, replace

from settings import IIDSetting
from utils import constant, get_logger

from .task_incremental_model import TaskIncrementalModel

logger = get_logger(__file__)


class IIDModel(TaskIncrementalModel[IIDSetting]):
    """ Model for an IID setting. 
    
    This is implemented quite simply a TaskIncrementalClassifier, but with only
    one train/val/test task.
    """
    @dataclass
    class HParams(TaskIncrementalModel.HParams):
        multihead: bool = constant(False)

    def preprocess_observations(self, observation) -> TaskIncrementalModel.Observation:
        return replace(observation, task_labels=None)
    
    # TODO: These can actually be used here, since we don't really care about
    # the metrics over time during training, we just want the aggregated metrics!
    
    # def validation_epoch_end(
    #         self,
    #         outputs: Union[List[Dict[str, Tensor]], List[List[Dict[str, Tensor]]]]
    #     ) -> Dict[str, Dict[str, Tensor]]:
    #     return self._shared_epoch_end(outputs, loss_name="val")

    # def test_epoch_end(
    #         self,
    #         outputs: Union[List[Dict[str, Tensor]], List[List[Dict[str, Tensor]]]]
    #     ) -> Dict[str, Dict[str, Tensor]]:
    #     # assert False, outputs
    #     return self._shared_epoch_end(outputs, loss_name="test")

    # def _shared_epoch_end(
    #     self,
    #     outputs: Union[List[Dict[str, Tensor]], List[List[Dict[str, Tensor]]]],
    #     loss_name: str="",
    # ) -> Dict[str, Dict[str, Tensor]]:
        
    #     # Sum of the metrics acquired during the epoch.
    #     # NOTE: This is the 'online' metrics in the case of a training/val epoch
    #     # and the 'average' & 'online' in the case of a test epoch (as they are
    #     # the same in that case).

    #     epoch_loss: Loss = Loss(name=loss_name)

    #     if not isinstance(outputs[0], list):
    #         # We used only a single dataloader.
    #         for output in outputs:
    #             if isinstance(output, list):
    #                 # we had multiple test/val dataloaders (i.e. multiple tasks)
    #                 # We get the loss for each task at each step. The outputs are for each of the dataloaders.
    #                 for i, task_output in enumerate(output):
    #                     task_loss = task_output["loss_object"] 
    #                     epoch_loss += task_loss
    #             elif isinstance(output, dict):
    #                 # There was a single dataloader: `output` is the dict returned
    #                 # by (val/test)_step.
    #                 loss_info = output["loss_object"]
    #                 epoch_loss += loss_info
    #             else:
    #                 raise RuntimeError(f"Unexpected output: {output}")
    #     else:
    #         for i, dataloader_output in enumerate(outputs):
    #             loss_i: Loss = Loss(name=f"{i}")
    #             for output in dataloader_output:
    #                 if isinstance(output, dict) and "loss_object" in output:
    #                     loss_info = output["loss_object"]
    #                     loss_i += loss_info
    #                 else:
    #                     raise RuntimeError(f"Unexpected output: {output}")
    #             epoch_loss += loss_i
    #     # TODO: Log stuff here?
    #     for name, value in epoch_loss.to_log_dict().items():
    #         logger.info(f"{name}: {value}")
    #         if self.logger:
    #             self.logger.log(name, value)
    #     return epoch_loss.to_pl_dict()