""" TODO: Tests for the EWC Method. """

from functools import partial

import numpy as np
import pytest
from torch import Tensor
from sequoia.common import Loss
from sequoia.settings.rl import (
    IncrementalRLSetting,
    TraditionalRLSetting,
    TaskIncrementalRLSetting,
)
from sequoia.settings.sl import (
    ClassIncrementalSetting,
    TraditionalSLSetting,
    MultiTaskSLSetting,
    TaskIncrementalSLSetting,
)

from .ewc_method import EwcMethod, EwcModel


@pytest.mark.timeout(300)
def test_task_incremental_mnist(monkeypatch):
    setting = TaskIncrementalSLSetting(dataset="mnist", monitor_training_performance=True)
    total_ewc_losses_per_task = np.zeros(setting.nb_tasks)

    _training_step = EwcModel.training_step

    def wrapped_training_step(self: EwcModel, batch, batch_idx: int, *args, **kwargs):
        step_results = _training_step(self, batch, batch_idx, *args, **kwargs)
        loss_object: Loss = step_results["loss_object"]
        if "ewc" in loss_object.losses:
            ewc_loss_obj = loss_object.losses["ewc"]
            ewc_loss = ewc_loss_obj.total_loss
            if isinstance(ewc_loss, Tensor):
                ewc_loss = ewc_loss.detach().cpu().numpy()
            total_ewc_losses_per_task[self.current_task] += ewc_loss
        return step_results

    monkeypatch.setattr(EwcModel, "training_step", wrapped_training_step)

    _fit = EwcMethod.fit

    at_all_points_in_time = []

    def wrapped_fit(self, train_env, valid_env):
        print(f"starting task {self.model.current_task}: {total_ewc_losses_per_task}")
        total_ewc_losses_per_task[:] = 0
        _fit(self, train_env, valid_env)
        at_all_points_in_time.append(total_ewc_losses_per_task.copy())

    monkeypatch.setattr(EwcMethod, "fit", wrapped_fit)

    # _on_epoch_end = EwcModel.on_epoch_end

    # def fake_on_epoch_end(self, *args, **kwargs):
    #     assert False, f"heyo: {total_ewc_losses_per_task}"
    #     return _on_epoch_end(self, *args, **kwargs)

    # # monkeypatch.setattr(EwcModel, "on_epoch_end", fake_on_epoch_end)
    method = EwcMethod(max_epochs=1)
    results = setting.apply(method)
    assert (at_all_points_in_time[0] == 0).all()
    assert at_all_points_in_time[1][1] != 0
    assert at_all_points_in_time[2][2] != 0
    assert at_all_points_in_time[3][3] != 0
    assert at_all_points_in_time[4][4] != 0

    assert 0.95 <= results.average_online_performance.objective
    assert 0.15 <= results.average_final_performance.objective


@pytest.mark.parametrize(
    "non_cl_setting_fn",
    [
        partial(ClassIncrementalSetting, nb_tasks=1),
        MultiTaskSLSetting,
        TraditionalSLSetting,
        TraditionalRLSetting,
        partial(IncrementalRLSetting, nb_tasks=1),
        partial(TaskIncrementalRLSetting, nb_tasks=1),
    ],
)
def test_raises_warning_when_applied_to_non_cl_setting(non_cl_setting_fn):
    """ When applied onto a non-CL setting like IID or Multi-Task SL (or RL), the
    EWCMethod should raise a warning, and disable the auxiliary task.
    """
    method = EwcMethod()
    setting = non_cl_setting_fn()

    with pytest.warns(RuntimeWarning):
        method.configure(setting)
