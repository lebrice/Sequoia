""" TODO: Tests for the EWC Method. """

from .ewc_method import EwcMethod, EwcModel
from sequoia.settings import ClassIncrementalSetting
from sequoia.common import Loss
import numpy as np
from torch import Tensor
import pytest


@pytest.mark.timeout(300)
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

    _fit = EwcMethod.fit

    at_all_points_in_time = []

    def fake_fit(self, train_env, valid_env):
        print(f"starting task {self.model.current_task}: {total_ewc_losses_per_task}")
        total_ewc_losses_per_task[:] = 0
        _fit(self, train_env, valid_env)
        at_all_points_in_time.append(total_ewc_losses_per_task.copy())

    monkeypatch.setattr(EwcMethod, "fit", fake_fit)

    # _on_epoch_end = EwcModel.on_epoch_end

    # def fake_on_epoch_end(self, *args, **kwargs):
    #     assert False, f"heyo: {total_ewc_losses_per_task}"
    #     return _on_epoch_end(self, *args, **kwargs)

    # # monkeypatch.setattr(EwcModel, "on_epoch_end", fake_on_epoch_end)
    method = EwcMethod(debug=True, max_epochs=1)
    results = setting.apply(method)
    assert (at_all_points_in_time[0] == 0).all()
    assert at_all_points_in_time[1][1] != 0
    assert at_all_points_in_time[2][2] != 0
    assert at_all_points_in_time[3][3] != 0
    assert at_all_points_in_time[4][4] != 0

