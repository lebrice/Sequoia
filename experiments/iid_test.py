import shlex

import pytest

from simple_parsing import ArgumentParser
from tasks.tasks import Tasks

from .iid import IID, Results, main

# Shouldn't take longer than 30 secs to run.
@pytest.mark.execution_timeout(30) 
def test_iid_fast_dev_run_self_supervised():
    parser = ArgumentParser()
    parser.add_arguments(IID, dest="experiment")
    args = parser.parse_args(shlex.split(
        """
        --debug
        --seed 123
        --fast_dev_run
        --dataset mnist
        --max_knn_samples 10
        --log_dir_root results/testing
        --simclr.coef 1
        --vae.coef 1
        --ae.coef 1
        --rotation.coef 1
        """
    ))
    experiment: IID = args.experiment
    results: IIDResults = experiment.launch()
    test_loss = results.test_loss
    print(test_loss.losses.keys())
    assert Tasks.SIMCLR in test_loss.losses
    assert Tasks.VAE in test_loss.losses
    assert Tasks.AE in test_loss.losses
    assert Tasks.ROTATION in test_loss.losses
        

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
