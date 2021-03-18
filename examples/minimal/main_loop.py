""" WIP: Example that shows the body of the 'main loop' in Sequoia. """
from typing import List

from sequoia.common.metrics import ClassificationMetrics, Metrics
from sequoia.methods import Method
from sequoia.settings import ClassIncrementalSetting
from sequoia.settings.assumptions.incremental import IncrementalSetting

from .basic.quick_demo import DemoMethod


def main_loop(setting: IncrementalSetting, method: Method):
    # Let the Method configure itself for this Setting.
    method.configure(setting)

    # Results is the transfer matrix.
    results: List[List[Metrics]] = []

    for task_id in range(setting.nb_tasks):
        method.set_training()
        setting.current_task_id = task_id

        # Inform the Method of a task boundary, when applicable in this Setting.
        if setting.known_task_boundaries_at_train_time:
            # Give the ID of the new task, when applicable in this Setting.
            method.on_task_switch(
                task_id=task_id if setting.task_labels_at_train_time else None
            )

        method.fit(setting.train_dataloader(), setting.val_dataloader())

        test_results = test_loop(setting, method)
        results.append(test_results)

    return results


def test_loop(setting: IncrementalSetting, method: Method) -> List[Metrics]:
    """Perform the "test loop" of this setting.

    Parameters
    ----------
    setting : IncrementalSetting
        [description]
    method : Method
        [description]

    Returns
    -------
    List[Metrics]
        Metrics for each task in the Setting.
    """
    test_results: List[Metrics] = []
    method.set_testing()
    for test_task_id in range(setting.nb_tasks):
        test_task_results = Metrics()

        setting.current_task_id = test_task_id
        test_env = setting.test_dataloader()

        # Interact with the test environment until it is exhausted:

        while not test_env.is_closed():
            obs = test_env.reset()
            done = False
            while not done:
                action = method.get_actions(obs, test_env.action_space)
                obs, reward, done, info = test_env.step(action)
                # In this case it's simple because we can always get a metrics at
                # each step in SL. in RL we'd have to check for each env if there
                # was an end of episode.
                test_task_results += ClassificationMetrics(
                    y_pred=action.y_pred, y=reward.y
                )

                # When the environment is vectorized, done is an array of booleans.
                if not isinstance(done, bool):
                    done = all(done)

        # Add the results for this task:
        test_results.append(test_task_results)
    return test_results


if __name__ == "__main__":
    setting = ClassIncrementalSetting(
        dataset="synbols",
        nb_tasks=12,
        monitor_training_performance=True,
        known_task_boundaries_at_test_time=False,
        batch_size=32,
        num_workers=4,
    )
    method = DemoMethod()
    results_ish = main_loop(setting, method)
