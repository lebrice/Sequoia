from .task_incremental_rl_setting import TaskIncrementalRLSetting
from sequoia.common.gym_wrappers import MultiTaskEnvironment


def test_task_schedule_is_used():
    # TODO: Figure out a way to test that the tasks are switching over time.
    setting = TaskIncrementalRLSetting(
        dataset="CartPole-v0",
        max_steps = 100,
        steps_per_task=50,
        nb_tasks=2,
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
        
        observations = env.reset()
        lengths: List[float] = []
        for i in range(100):
            obs, reward, done, info = env.step(env.action_space.sample())
            if done:
                env.reset()
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
