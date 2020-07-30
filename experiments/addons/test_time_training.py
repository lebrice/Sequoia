from dataclasses import dataclass

from torch import Tensor
import copy
from common.losses import LossInfo
from .addon import ExperimentAddon
from simple_parsing import mutable_field
from typing import Union
from torch.utils.data import Dataset, DataLoader
from datasets.data_utils import unlabeled



@dataclass  # type: ignore
class TestTimeTrainingAddon(ExperimentAddon):
    """ Experiment where we also perform self-supervised training at test-time.
    """
    @dataclass
    class Config(ExperimentAddon.Config):
        #we can either asume local stationarity on the batch-level or on the task level

        #number of training iterations per batch (assume batch level stationarity)
        test_time_ioterations_batch: int = 0

        #number of test time training epochs (assume task level stationarity)
        test_time_training_epochs: int = 0
    
    config: Config = mutable_field(Config)


    def test(self, dataloader: Union[Dataset, DataLoader], description: str = None, name: str = "Test") -> LossInfo:
        model_st_dict_before = None
        optimizer_st_dict_before = None
        tasks_copy = None
        if self.config.test_time_training_epochs>0 and 'Task' in name and self.state.j != self.state.i:
            #we copy the model and do some training epochs on test data
            model_st_dict_before = copy.deepcopy(self.model.state_dict())
            optimizer_st_dict_before = copy.deepcopy(self.model.optimizer.state_dict())

            #copy task specific modules (like projectors etc.)
            tasks_copy = self.model.copy_tasks()

            if isinstance(dataloader, Dataset):
                dataloader = self.get_dataloader(dataloader)
            
            #set remembering learing rate
            for g in self.model.optimizer.param_groups:
                g['lr'] = 0.001
            _ = self.train(
                        train_dataloader=unlabeled(dataloader),
                        valid_dataloader=None,
                        test_dataloader=dataloader, #just to keep track of performance at test time training time
                        epochs=self.config.active_remembering_epochs,
                        description=f"{name} Remembering (Unsupervised)",
                        temp_save_dir=self.checkpoints_dir / f"{name}_remembering",
                    )
        loss_info = super().test(dataloader, description=description, name=name)

        #reload the model as before the remembering phase
        if model_st_dict_before is not None:
            self.model.load_state_dict(model_st_dict_before)
        if optimizer_st_dict_before is not None:
            self.model.optimizer.load_state_dict(optimizer_st_dict_before)
        for g in self.model.optimizer.param_groups:
            g['lr'] = self.model.hparams.learning_rate
        
        #reload tasks as before the remembering phase
        if tasks_copy is not None:
            for name, task in self.model.tasks.items():
                if name in  tasks_copy.keys() and task.enabled:
                        task = copy.deepcopy(tasks_copy.get(name))
            del tasks_copy
        return loss_info 
    

    def test_batch(self, data: Tensor, target: Tensor=None, **kwargs) -> LossInfo:  # type: ignore
        #TODO: rethink how to backup the model here
        if self.config.test_time_ioterations_batch > 0:
            for _ in range(self.config.test_time_ioterations_batch):
                #do several iterations of training on this batch (optimaly using different augmentations each time)
                super().train_batch(data, None, **kwargs)  # type: ignore
        return super().test_batch(data, target, **kwargs)  # type: ignore
