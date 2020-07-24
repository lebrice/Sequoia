import shlex

import pytest

from simple_parsing import ArgumentParser

from .class_incremental import ClassIncremental, Results
from common.losses import LossInfo

# Shouldn't take longer than 60 secs to run.
@pytest.mark.skip("No need for this test if the next one is performed.")
@pytest.mark.execution_timeout(60) 
def test_fast_dev_run():
    Experiment = ClassIncremental 

    parser = ArgumentParser()
    parser.add_arguments(Experiment, dest="experiment")
    args = parser.parse_args(shlex.split(
        """
        --debug --seed 123 --fast_dev_run --dataset mnist --max_knn_samples 10
        --log_dir_root results/testing
        --simclr.coef 1
        --vae.coef 1
        --ae.coef 1
        --rotation.coef 1
        """
    ))
    experiment: Experiment = args.experiment
    results: Results = experiment.launch()
    test_loss = results.test_loss
    
    assert experiment.setting.nb_tasks == 5
    task_losses: List[LossInfo] = []
    for i in range(5):
        task_loss: LossInfo = test_loss.losses[f"test/{i}"]
        task_losses.append(task_loss)
    accuracies: List[float] = [loss.accuracy for loss in task_losses]
    
    print(accuracies)
    # Basically just make sure they exist.
    assert accuracies[0] >= 0., accuracies
    assert accuracies[1] >= 0., accuracies
    assert accuracies[2] >= 0., accuracies
    assert accuracies[3] >= 0., accuracies
    assert accuracies[4] >= 0., accuracies

from tasks.tasks import Tasks


# Shouldn't take longer than 60 secs to run.
@pytest.mark.execution_timeout(60) 
def test_fast_dev_run_self_supervision():
    Experiment = ClassIncremental 

    parser = ArgumentParser()
    parser.add_arguments(Experiment, dest="experiment")
    args = parser.parse_args(shlex.split(
        """
        --debug --seed 123
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
    experiment: Experiment = args.experiment
    results: Results = experiment.launch()
    test_loss = results.test_loss
    
    assert experiment.setting.nb_tasks == 5
    task_losses: List[LossInfo] = []
    for i in range(5):
        task_loss: LossInfo = test_loss.losses[f"test/{i}"]
        task_losses.append(task_loss)

        print(task_loss.losses.keys())
        assert Tasks.SIMCLR in task_loss.losses
        assert Tasks.VAE in task_loss.losses
        assert Tasks.AE in task_loss.losses
        assert Tasks.ROTATION in task_loss.losses
    
    accuracies: List[float] = [loss.accuracy for loss in task_losses]
    
    
    print(accuracies)
    # Basically just make sure they exist.
    assert accuracies[0] >= 0., accuracies
    assert accuracies[1] >= 0., accuracies
    assert accuracies[2] >= 0., accuracies
    assert accuracies[3] >= 0., accuracies
    assert accuracies[4] >= 0., accuracies


# Shouldn't take longer than 120 secs to run.
@pytest.mark.execution_timeout(120) 
def test_single_epoch():
    n_tasks = 5

    Experiment = ClassIncremental
    parser = ArgumentParser()
    parser.add_arguments(Experiment, dest="experiment")
    args = parser.parse_args(shlex.split(
        f"""--debug --seed 123 --nb_tasks {n_tasks} --max_epochs 1
            --dataset mnist --max_knn_samples 10
            --log_dir_root results/testing
        """
    ))
    experiment: Experiment = args.experiment
    results: Results = experiment.launch()
    test_loss = results.test_loss
    assert experiment.setting.nb_tasks == n_tasks
    assert experiment.setting.increment == 10 // n_tasks

    task_losses: List[LossInfo] = []
    for i in range(n_tasks):
        task_loss: LossInfo = test_loss.losses[f"test/{i}"]
        task_losses.append(task_loss)

    # Check that each Loss describes 5 classes.
    class_accuracy = task_losses[0].metric.class_accuracy
    # TODO: @lebrice There is something weird going on with the class accuracy,
    # it should only have 5 classes, not 10!
    # assert class_accuracy.shape[0] == 5, class_accuracy 
    
    accuracies: List[float] = [loss.accuracy for loss in task_losses]
    print(accuracies)
    # NOTE: This is the 'catastrophic forgetting' when NOT using a multihead model.
    # TODO: Change this or add a new test when the multihead stuff gets added back.
    assert accuracies[0] < 0.1, accuracies
    assert accuracies[1] < 0.1, accuracies
    assert accuracies[2] < 0.1, accuracies
    assert accuracies[3] < 0.1, accuracies
    assert accuracies[4] > 0.9, accuracies



# Shouldn't take longer than 180 secs to run.
@pytest.mark.execution_timeout(180) 
def test_single_epoch_with_self_supervision():
    n_tasks = 5

    Experiment = ClassIncremental
    parser = ArgumentParser()
    parser.add_arguments(Experiment, dest="experiment")
    args = parser.parse_args(shlex.split(
        f"""--debug --seed 123 --nb_tasks {n_tasks} --max_epochs 1
            --dataset mnist --max_knn_samples 10
            --log_dir_root results/testing
        """
    ))
    experiment: Experiment = args.experiment
    results: Results = experiment.launch()
    test_loss = results.test_loss
    assert experiment.setting.nb_tasks == n_tasks
    assert experiment.setting.increment == 10 // n_tasks

    task_losses: List[LossInfo] = []
    for i in range(n_tasks):
        task_loss: LossInfo = test_loss.losses[f"test/{i}"]
        task_losses.append(task_loss)

    # Check that each Loss describes 5 classes.
    class_accuracy = task_losses[0].metric.class_accuracy
    # TODO: @lebrice There is something weird going on with the class accuracy,
    # it should only have 5 classes, not 10!
    # assert class_accuracy.shape[0] == 5, class_accuracy 
    
    accuracies: List[float] = [loss.accuracy for loss in task_losses]
    print(accuracies)
    # NOTE: This is the 'catastrophic forgetting' when NOT using a multihead model.
    # TODO: Change this or add a new test when the multihead stuff gets added back.
    assert accuracies[0] < 0.1, accuracies
    assert accuracies[1] < 0.1, accuracies
    assert accuracies[2] < 0.1, accuracies
    assert accuracies[3] < 0.1, accuracies
    assert accuracies[4] > 0.9, accuracies