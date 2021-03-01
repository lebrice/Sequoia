from .task_schedule_env import TaskScheduleEnv
from .add_task_labels import NamedTuple
from gym.envs.classic_control import CartPoleEnv


def test_simple_episode_schedule():
    nb_tasks = 5
    from sequoia.common.gym_wrappers import EnvDataset

    lengths = {
        # Add offset since length = 0 causes nans in obs.
        i: 0.2 + 0.1 * i
        for i in range(nb_tasks)
    }
    envs = [EnvDataset(CartPoleEnv()) for _ in range(nb_tasks)]
    for i, env in enumerate(envs):
        env.unwrapped.length = lengths[i]

    n_episodes_per_task = 2
    env = TaskScheduleEnv(envs, episode_schedule={
        i * n_episodes_per_task: i
        for i in range(nb_tasks)
    })

    ## gym-stype interaction
    
    task_ids = []
    for episode in range(n_episodes_per_task * nb_tasks):
        obs = env.reset()
        task_id = obs[1]
        assert env.length == lengths[task_id]
        task_ids.append(task_id)
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            assert obs[1] == task_id

    assert task_ids == sum([[i for _ in range(n_episodes_per_task)] for i in range(nb_tasks)], [])
    
    
    ## dataloader-stype iteration
    
    env = TaskScheduleEnv(envs, episode_schedule={
        i * n_episodes_per_task: i
        for i in range(nb_tasks)
    })
    task_ids = []
    for episode in range(nb_tasks * n_episodes_per_task):
        for i, obs in enumerate(env):
            assert isinstance(obs, NamedTuple)
            if i == 0:
                task_index = obs[1]
                task_ids.append(task_index)
            else:
                assert obs[1] == task_index

            action = env.action_space.sample()
            reward = env.send(action)

    assert task_ids == sum([[i for _ in range(n_episodes_per_task)] for i in range(nb_tasks)], [])


