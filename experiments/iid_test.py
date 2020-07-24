from .iid import main, IID, Results
from simple_parsing import ArgumentParser
import pytest
import shlex

@pytest.mark.skip("No need for this test if the next one is performed.")
@pytest.mark.execution_timeout(20) # Shouldn't take longer than 20 secs to run.
def test_iid_fast_dev_run():
    parser = ArgumentParser()
    parser.add_arguments(IID, dest="experiment")
    args = parser.parse_args(shlex.split(
        """
        --debug --seed 123 --fast_dev_run --dataset mnist --max_knn_samples 10
        --log_dir_root results/testing
        """
    ))
    experiment: IID = args.experiment
    results: IIDResults = experiment.launch()
    test_loss = results.test_loss

    assert test_loss.accuracy == 0.25


# Shouldn't take longer than 60 secs to run.
@pytest.mark.execution_timeout(60) 
def test_iid_single_epoch():
    parser = ArgumentParser()
    parser.add_arguments(IID, dest="experiment")
    args = parser.parse_args(shlex.split(
        """
        --debug --seed 123 --max_epochs 1 --dataset mnist --max_knn_samples 10
        --log_dir_root results/testing
        """
    ))
    experiment: IID = args.experiment
    results: IIDResults = experiment.launch()
    test_loss = results.test_loss
    assert test_loss.accuracy >= 0.25

