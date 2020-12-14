"""
TODO: Write some tests that also help illustrate how the Loss class works.
"""
from .loss import Loss


def test_demo():
    """ Simple test to demonstrate addition of Loss objects. """
    loss = Loss("total")
    loss += Loss("task_a", loss=1.23, metrics={"accuracy": 0.95})
    loss += Loss("task_b", loss=2.10)
    loss += Loss("task_c", loss=3.00)
    # Get a dict to be logged, for example with wandb.
    loss_dict = loss.to_log_dict()
    assert loss_dict == {
        'loss': 6.33,
        'losses/task_a/loss': 1.23,
        'losses/task_a/accuracy': 0.95,
        'losses/task_b/loss': 2.1,
        'losses/task_c/loss': 3.0
    }


def test_all_metrics():
    """ Using `all_metrics()` gives a dict of all the metrics in the Loss.
    """
    loss = Loss("total")
    loss += Loss("task_a", loss=1.23, metrics={"accuracy": 0.95})
    loss += Loss("task_b", loss=2.10)
    loss += Loss("task_c", loss=3.00)
    assert loss.all_metrics() == {
        'total/task_a/accuracy': 0.95,
    }
    

def test_to_log_dict_order():
    """ Simple test to demonstrate addition of Loss objects. """
    task_a_loss = Loss("task_a", loss=1.23, metrics={"accuracy": 0.95})
    task_b_loss = Loss("task_b", loss=2.10)
    task_c_loss = Loss("task_c", loss=3.00)
    total_loss = Loss("total") + task_a_loss + task_b_loss + task_c_loss
    loss_dict = total_loss.to_log_dict()
    assert loss_dict == {
        'loss': 6.33,
        'losses/task_a/loss': 1.23,
        'losses/task_a/accuracy': 0.95,
        'losses/task_b/loss': 2.1,
        'losses/task_c/loss': 3.0
    }