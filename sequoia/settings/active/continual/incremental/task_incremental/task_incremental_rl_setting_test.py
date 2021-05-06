from typing import List

from sequoia.common.gym_wrappers import MultiTaskEnvironment

from .task_incremental_rl_setting import TaskIncrementalRLSetting


def test_task_schedule_is_used():
    """ Test that the tasks are switching over time. """
    setting = TaskIncrementalRLSetting(
        dataset="CartPole-v0", max_steps=100, steps_per_task=50, nb_tasks=2,
    )

    default_length = 0.5

    for task_id in range(2):
        setting.current_task_id = task_id

        env = setting.train_dataloader(batch_size=None)
        env: MultiTaskEnvironment
        assert len(setting.train_task_schedule) == 2
        assert len(setting.valid_task_schedule) == 2
        assert len(setting.test_task_schedule) == 2

        starting_length = env.length

        _ = env.reset()
        lengths: List[float] = []
        for i in range(setting.steps_per_phase):
            obs, reward, done, info = env.step(env.action_space.sample())
            # NOTE: If we're done on the last step, we can't reset, since that would go
            # over the step budget.
            if done and i != setting.steps_per_phase - 1:
                env.reset()
            # Get the length of the pole from the environment.    
            length = env.length
            lengths.append(length)

        if task_id == 0:
            assert starting_length == default_length
            assert all(length == default_length for length in lengths)

        else:
            # The length of the pole is different than the default length
            assert starting_length != default_length
            # The length shouldn't be changing over time.
            assert all(length == starting_length for length in lengths)

    # assert False, (lengths[:2], lengths[-2:])
