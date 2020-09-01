from .task_incremental_method import TaskIncrementalMethod
from .models.task_incremental_model import TaskIncrementalModel
import pytest
from simple_parsing import ParsingError


def test_parsing_hparams_multihead():
    """Test that parsing the multihead field works as expected. """
    hp = TaskIncrementalModel.HParams.from_args("")
    assert hp.multihead

    with pytest.raises(ParsingError):
        hp = TaskIncrementalModel.HParams.from_args("--multihead")
        assert hp.multihead

    hp = TaskIncrementalModel.HParams.from_args("--multihead=False")
    assert not hp.multihead

    hp = TaskIncrementalModel.HParams.from_args("--multihead True")
    assert hp.multihead

    hp = TaskIncrementalModel.HParams.from_args("--multihead False")
    assert not hp.multihead


def test_fast_dev_run_multihead(tmp_path: Path):
    setting = TaskIncrementalSetting(
        dataset="mnist",
        increment=2,
    )
    method: TaskIncrementalMethod = TaskIncrementalMethod.from_args(f"""
        --debug
        --fast_dev_run
        --default_root_dir {tmp_path}
        --log_dir_root {tmp_path}
        --multihead True
        --batch_size 100
    """)
    results: TaskIncrementalResults = method.apply_to(setting)
    metrics = results.task_metrics
    assert metrics
    for metric in metrics:
        if isinstance(metric, ClassificationMetrics):
            assert metric.confusion_matrix.shape == (2, 2)
    validate_results(results, setting)
