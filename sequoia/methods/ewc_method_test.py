""" TODO: Tests for the EWC Method. """

from .ewc_method import EwcMethod, EwcModel
from sequoia.settings import ClassIncrementalSetting
from sequoia.common import Loss
import numpy as np
from torch import Tensor


def test_has_no_ewc_loss_first_task(monkeypatch):
    setting = ClassIncrementalSetting(dataset="mnist")
    total_ewc_losses_per_task = np.zeros(setting.nb_tasks)

    _training_step = EwcModel.training_step

    def fake_training_step(self: EwcModel, batch, batch_idx: int, *args, **kwargs):
        step_results = _training_step(self, batch, batch_idx, *args, **kwargs)
        loss_object: Loss = step_results["loss_object"]
        if "ewc" in loss_object.losses:
            ewc_loss_obj = loss_object.losses["ewc"]
            ewc_loss = ewc_loss_obj.total_loss
            if isinstance(ewc_loss, Tensor):
                ewc_loss = ewc_loss.detach().cpu().numpy()
            total_ewc_losses_per_task[self.current_task] += ewc_loss
        return step_results

    monkeypatch.setattr(EwcModel, "training_step", fake_training_step)

    # _on_epoch_end = EwcModel.on_epoch_end

    # def fake_on_epoch_end(self, *args, **kwargs):
    #     assert False, f"heyo: {total_ewc_losses_per_task}"
    #     return _on_epoch_end(self, *args, **kwargs)

    # monkeypatch.setattr(EwcModel, "on_epoch_end", fake_on_epoch_end)
    method = EwcMethod(debug=True, max_epochs=1)
    method.configure(setting)

    # for task in range(setting.nb_tasks):
    #     setting.current_task_id = task
        
    #     method.fit(
    #         setting.train_dataloader(),
    #         setting.val_dataloader(),
    #     )
    #     if task == 1:
    #         assert False, total_ewc_losses_per_task

    results = setting.apply(method)
    assert False, total_ewc_losses_per_task
    assert False, results.objective

